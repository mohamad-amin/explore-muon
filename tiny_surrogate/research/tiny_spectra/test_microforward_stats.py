"""Synthetic contracts for the explicitly selected per-forward statistic clock."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from torch.torch_version import TorchVersion

from research.adamw_spectra.model import StatLinear
from .model import GPT, ModelConfig, StepStatLinear
from .optim import output_second_moments
from .train import run


STAT_FIELDS = ("input_mean", "input_sq", "input_weight", "input_cov", "input_cov_mean", "input_cov_weight")


class MicroforwardContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.threads)

    def test_fp32_matches_reference_including_unequal_last_microbatch(self):
        torch.manual_seed(41)
        observed = StepStatLinear(4, 3, decay=.8, cov_decay=.6, cov_stride=3, stats_clock="microforward")
        reference = StatLinear(4, 3, bias=False, decay=.8, cov=True, cov_decay=.6, cov_stride=3)
        reference.load_state_dict(observed.state_dict())
        x = torch.randn(5, 9, 4)
        x[:, 0] = 1e6  # The excluded position must not enter either statistic.
        observed.begin_step()
        for chunk in (x[:3], x[3:]):
            a, b = observed(chunk), reference(chunk)
            torch.testing.assert_close(a, b, atol=0, rtol=0)
            a.sum().backward()
            b.sum().backward()
        before_finish = {name: getattr(observed, name).clone() for name in STAT_FIELDS}
        observed.finish_step()
        for name in STAT_FIELDS:
            torch.testing.assert_close(getattr(observed, name), getattr(reference, name), atol=1e-7, rtol=1e-6)
            torch.testing.assert_close(getattr(observed, name), before_finish[name], atol=0, rtol=0)
        torch.testing.assert_close(observed.weight.grad, reference.weight.grad, atol=0, rtol=0)
        self.assertAlmostEqual(observed.input_cov_weight.item(), 1-.6**2, places=6)
        self.assertAlmostEqual(observed.input_weight.item(), 1-.8**2, places=6)
        self.assertEqual(observed._step_forwards.item(), 2)

    def test_gram_precision_is_fp32_inside_bfloat16_autocast(self):
        torch.manual_seed(76)
        layer = StepStatLinear(4, 3, cov_decay=.6, cov_stride=2, stats_clock="microforward")
        x = torch.randn(3, 9, 4)
        rows = x[:, 1::2].reshape(-1, 4)
        expected = .4 * (rows.T @ rows / len(rows))
        layer.begin_step()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = layer(x)
        layer.finish_step()
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(layer.input_cov.dtype, torch.float32)
        torch.testing.assert_close(layer.input_cov, expected, atol=1e-7, rtol=1e-6)

    def test_reference_midpoint_clock_and_explicit_step_end(self):
        layer = StepStatLinear(2, 2, decay=.99, cov_decay=.998, stats_clock="microforward")
        x = torch.randn(4, 3, 2)
        layer.begin_step()
        for _ in range(128):
            layer(x)
        with self.assertRaisesRegex(ValueError, "cannot be overridden"):
            layer.finish_step(.998)
        layer.finish_step()
        self.assertAlmostEqual(layer.input_cov_weight.item(), 1-.998**128, places=6)
        self.assertAlmostEqual(layer.input_weight.item(), 1-.99**128, places=6)
        self.assertEqual(layer._total_forwards.item(), 128)
        before = layer.input_cov.clone()
        layer(x * 1e5)  # Outside the collection window.
        torch.testing.assert_close(layer.input_cov, before, atol=0, rtol=0)
        layer.begin_step()
        layer(x[:1])
        layer.finish_step()
        self.assertEqual(layer._step_forwards.item(), 1)
        self.assertEqual(layer._total_forwards.item(), 129)
        self.assertAlmostEqual(layer.input_cov_weight.item(), 1-.998**129, places=6)

    def test_gn_eval_and_no_grad_do_not_change_input_statistics(self):
        torch.manual_seed(17)
        model = GPT(ModelConfig(vocab_size=9, n_layer=1, n_embd=8, n_head=2, seq_len=8,
                                track_input_stats=True, track_input_cov=True,
                                stats_clock="microforward", cov_decay=.998))
        x, y = torch.randint(9, (2, 8)), torch.randint(9, (2, 8))
        model.begin_step_stats()
        model(x, y).backward()
        before = {name: value.clone() for name, value in model.named_buffers()}
        grads = {name: value.grad.clone() for name, value in model.named_parameters()}
        moments = output_second_moments(model, x, y, precision="fp32",
                                       generator=torch.Generator().manual_seed(88))
        self.assertEqual(len(moments), 6)
        model.eval()
        model(x)
        model.train()
        with torch.no_grad():
            model(x)
        for name, value in model.named_buffers():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
        for name, value in model.named_parameters():
            torch.testing.assert_close(value.grad, grads[name], atol=0, rtol=0)
        model.finish_step_stats()

    def test_standalone_training_wires_each_forward_without_double_commit(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "synthetic.txt"
            data.write_text("abcd" * 256)
            cfg = dict(seed=17, method="ts", lr=.008, aux_lr=.002, momentum=.9,
                       alpha=.25, out_beta=.25, decay=.01, soap_beta2=.9,
                       root_refresh=1, cov_ema=.998, stats_ema=.99, stats_clock="microforward",
                       cov_stride=2, out_sequences=2, out_ema=.8,
                       n_layer=1, n_embd=8, n_head=2, seq_len=8,
                       batch_tokens=32, total_tokens=64, microbatch_sequences=2,
                       warmup_fraction=.034, cooldown_fraction=.1, grad_clip=1.,
                       validation_tokens=16, eval_every=1, precision="fp32", device="cpu", cpu_threads=1,
                       data_path=str(data), data_sha256=hashlib.sha256(data.read_bytes()).hexdigest())
            with contextlib.redirect_stdout(io.StringIO()):
                summary = run(cfg, root / "run")
            self.assertEqual(summary["status"], "complete")
            metadata = json.loads((root / "run/metadata.json").read_text())
            self.assertEqual(metadata["covariance_clock"], "EMA per collected training microforward")
            self.assertEqual(metadata["model_config"]["stats_clock"], "microforward")
            with torch.serialization.safe_globals([TorchVersion]):
                saved = torch.load(root / "run/final.pt", map_location="cpu", weights_only=True)
            for state in saved["optimizer"]["model_statistics"].values():
                self.assertEqual(state["_total_forwards"].item(), 4)
                self.assertAlmostEqual(state["input_cov_weight"].item(), 1-.998**4, places=7)
                self.assertAlmostEqual(state["input_weight"].item(), 1-.99**4, places=7)


if __name__ == "__main__":
    unittest.main()

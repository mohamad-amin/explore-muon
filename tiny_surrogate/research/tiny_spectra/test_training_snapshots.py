"""CPU trajectory invariance and replay contracts for optional model snapshots."""

import contextlib
from dataclasses import asdict
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch

from research.tiny_spectra.data import load_corpus
from research.tiny_spectra.model import GPT, ModelConfig
from research.tiny_spectra.train import evaluate, model_hash, run, save_model_snapshot


def scientific_metrics(path):
    return [{key: value for key, value in json.loads(line).items()
             if key not in ("elapsed_seconds", "step_seconds")}
            for line in (path / "metrics.jsonl").read_text().splitlines()]


class TrainingSnapshotContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, data_path):
        return dict(data_path=str(data_path), seed=173, method="spd", lr=.008,
                    aux_lr=.002, momentum=.8, alpha=.25, out_beta=.25,
                    decay=.01, soap_beta2=.9, root_refresh=2, cov_ema=.9,
                    cov_stride=1, out_sequences=2, out_ema=.8,
                    n_layer=1, n_embd=16, n_head=2, seq_len=8,
                    batch_tokens=24, total_tokens=112, microbatch_sequences=2,
                    warmup_fraction=.2, cooldown_fraction=.2, grad_clip=1.,
                    validation_tokens=32, eval_every=2, precision="fp32",
                    device="cpu", cpu_threads=1)

    def test_enabled_trajectory_identical_and_intermediate_replays_dev_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "input.txt"
            data.write_text("To be, or not to be!\n" * 50)
            cfg = self.config(data)
            with contextlib.redirect_stdout(io.StringIO()):
                plain = run(cfg, root / "plain")
                saved = run(dict(cfg, keep_model_every=2), root / "saved")
            self.assertFalse((root / "plain/models").exists())
            names = sorted(p.name for p in (root / "saved/models").iterdir())
            self.assertEqual(names, [f"step{s:06d}.pt" for s in (0, 1, 2, 4, 5)])
            self.assertEqual(scientific_metrics(root / "plain"), scientific_metrics(root / "saved"))
            self.assertEqual(plain["full_validation_nll"], saved["full_validation_nll"])
            # Byte identity is checked per final tensor, independent of pickle metadata.
            # Legacy final.pt contains TorchVersion metadata; these files were
            # generated immediately above inside this private temporary folder.
            first = torch.load(root / "plain/final.pt", weights_only=False)
            second = torch.load(root / "saved/final.pt", weights_only=False)
            for name in first["model"]:
                self.assertEqual(first["model"][name].numpy().tobytes(),
                                 second["model"][name].numpy().tobytes())
            corpus = load_corpus(data)
            starts = torch.load(root / "saved/windows.pt", weights_only=True)["validation"]
            metrics = {x["step"]: x for x in scientific_metrics(root / "saved")}
            for step in (0, 1, 2, 4, 5):
                checkpoint = torch.load(root / f"saved/models/step{step:06d}.pt", weights_only=True)
                self.assertEqual(checkpoint["format"], "tiny_spectra_model_only_v1")
                self.assertFalse(checkpoint["resumable"])
                self.assertNotIn("optimizer", checkpoint)
                self.assertEqual(checkpoint["step"], step)
                self.assertEqual(checkpoint["tokens"], min(step * 24, 112))
                model = GPT(ModelConfig(**checkpoint["model_config"]))
                model.load_state_dict(checkpoint["model"], strict=True)
                self.assertEqual(model_hash(model), checkpoint["parameter_sha256"])
                value, _ = evaluate(model, corpus, "val", starts, cfg, torch.device("cpu"))
                self.assertEqual(value, metrics[step]["validation_nll"])

    def test_saving_preserves_rng_parameters_gradients_and_all_statistics(self):
        cfg = ModelConfig(vocab_size=13, n_layer=1, n_embd=16, n_head=2,
                          seq_len=8, track_input_stats=True, track_input_cov=True)
        model = GPT(cfg)
        model.begin_step_stats()
        model(torch.randint(13, (2, 8)), torch.randint(13, (2, 8))).backward()
        model.finish_step_stats(.9)
        before_tensors = {name: tensor.clone() for name, tensor in
                          list(model.named_parameters()) + list(model.named_buffers())}
        before_grads = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
        before_modes = {name: (module.training, getattr(module, "collect_stats", None))
                        for name, module in model.named_modules()}
        before_torch_rng, before_python_rng = torch.get_rng_state().clone(), random.getstate()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "models/step000001.pt"
            expected = model_hash(model)
            self.assertEqual(save_model_snapshot(model, cfg, target, step=1, tokens=16), expected)
            self.assertTrue(torch.equal(before_torch_rng, torch.get_rng_state()))
            self.assertEqual(before_python_rng, random.getstate())
            for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
                torch.testing.assert_close(tensor, before_tensors[name], rtol=0, atol=0)
            for name, parameter in model.named_parameters():
                torch.testing.assert_close(parameter.grad, before_grads[name], rtol=0, atol=0)
            self.assertEqual(before_modes, {name: (module.training, getattr(module, "collect_stats", None))
                                           for name, module in model.named_modules()})
            payload = torch.load(target, weights_only=True)
            self.assertEqual(payload["model_config"], asdict(cfg))
            self.assertEqual(set(payload["model"]), set(model.state_dict()))
            self.assertFalse(any("input_cov" in name or "_step_" in name for name in payload["model"]))
            self.assertTrue(all(t.device.type == "cpu" and not t.requires_grad
                                and not isinstance(t, torch.nn.Parameter) for t in payload["model"].values()))
            self.assertFalse(target.with_suffix(".pt.tmp").exists())
            with self.assertRaises(FileExistsError):
                save_model_snapshot(model, cfg, target, step=1, tokens=16)

    def test_failed_save_publishes_no_partial_checkpoint(self):
        cfg = ModelConfig(vocab_size=13, n_layer=1, n_embd=16, n_head=2, seq_len=8)
        model = GPT(cfg)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "models/step000001.pt"
            def fail(payload, stream):
                stream.write(b"partial serialization")
                raise OSError("simulated write failure")
            with patch("research.tiny_spectra.train.torch.save", side_effect=fail):
                with self.assertRaises(OSError):
                    save_model_snapshot(model, cfg, target, step=1, tokens=16)
            self.assertFalse(target.exists())
            self.assertEqual(target.with_suffix(".pt.tmp").read_bytes(), b"partial serialization")

    def test_invalid_cadence_fails_before_creating_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "out"
            for cadence in (-1, 1.5, True):
                with self.assertRaisesRegex(ValueError, "keep_model_every"):
                    run(dict(keep_model_every=cadence), target)
                self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()

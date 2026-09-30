"""CPU checks for pairing, causal prediction and the optimizer-statistics contract."""

from dataclasses import replace
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from research.adamw_spectra.model import GPT as ProductionGPT
from research.adamw_spectra.model import ModelConfig as ProductionConfig
from research.adamw_spectra.model import StatLinear
from research.tiny_spectra.data import corpus_from_text, load_corpus, sha256_bytes
from research.tiny_spectra.model import GPT, ModelConfig, StepStatLinear


class ModelContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def config(self, **kwargs):
        return ModelConfig(vocab_size=13, n_layer=2, n_embd=16, n_head=4, seq_len=8, **kwargs)

    def test_shapes_loss_and_parameter_roles(self):
        model = GPT(self.config())
        x, targets = torch.randint(13, (2, 8)), torch.randint(13, (2, 8))
        logits, loss = model(x, targets, return_logits=True)
        self.assertEqual(logits.shape, (2, 8, 13))
        self.assertTrue(torch.equal(loss, F.cross_entropy(logits.flatten(0, 1), targets.flatten())))
        loss.backward()
        self.assertTrue(all(p.grad is not None and p.grad.isfinite().all() for p in model.parameters()))
        self.assertEqual(len(model.body_parameters()), 12)
        self.assertEqual(set(model.body_parameters()),
                         {n for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2})
        self.assertIsNot(model.embed.weight, model.head.weight)
        self.assertEqual(model.num_parameters(), sum(p.numel() for p in model.parameters()))
        self.assertFalse(any(name.endswith("bias") for name, _ in model.named_parameters()))

    def test_future_tokens_cannot_change_prefix_logits(self):
        model = GPT(self.config()).eval()
        x = torch.randint(13, (2, 8))
        changed = x.clone()
        changed[:, 4:] = (changed[:, 4:] + 1) % 13
        with torch.no_grad():
            a, b = model(x), model(changed)
        torch.testing.assert_close(a[:, :4], b[:, :4], atol=0, rtol=0)
        self.assertGreater((a[:, 4:] - b[:, 4:]).abs().max().item(), 0)

    def test_instrumented_model_preserves_initialization_and_rng(self):
        config = self.config(track_input_stats=True, track_input_cov=True)
        torch.manual_seed(19)
        observed = GPT(config)
        after_instrumented = torch.rand(5)
        torch.manual_seed(19)
        plain = GPT(replace(config, track_input_stats=False, track_input_cov=False))
        after_plain = torch.rand(5)
        for name, parameter in observed.named_parameters():
            torch.testing.assert_close(parameter, dict(plain.named_parameters())[name], atol=0, rtol=0)
        torch.testing.assert_close(after_instrumented, after_plain, atol=0, rtol=0)
        production_fields = ProductionConfig.__dataclass_fields__
        torch.manual_seed(19)
        production = ProductionGPT(ProductionConfig(**{k: getattr(config, k) for k in production_fields}))
        for name, parameter in observed.named_parameters():
            torch.testing.assert_close(parameter, dict(production.named_parameters())[name], atol=0, rtol=0)
        self.assertTrue(all(isinstance(m, StatLinear) and m.cov for m in observed.body_modules().values()))

    def test_step_stats_do_not_change_forward_or_gradient(self):
        torch.manual_seed(5)
        plain = GPT(self.config())
        observed = GPT(self.config(track_input_stats=True, track_input_cov=True))
        observed.load_state_dict(plain.state_dict())
        x, y = torch.randint(13, (3, 8)), torch.randint(13, (3, 8))
        observed.begin_step_stats()
        a, b = plain(x, y), observed(x, y)
        a.backward()
        b.backward()
        observed.finish_step_stats(ema=.9)
        torch.testing.assert_close(a, b, atol=0, rtol=0)
        for name, p in observed.named_parameters():
            torch.testing.assert_close(p.grad, dict(plain.named_parameters())[name].grad, atol=0, rtol=0)
        for module in observed.body_modules().values():
            self.assertGreater(module.input_cov_weight.item(), 0)
            self.assertTrue(module.input_cov.isfinite().all())
        self.assertEqual(set(observed.state_dict()), set(plain.state_dict()))

    def test_stats_pool_rows_once_and_ignore_position_zero(self):
        module = StepStatLinear(3, 2, cov_stride=2)
        samples = torch.arange(5 * 6 * 3).reshape(5, 6, 3).float() / 10
        samples[:, 0] = 1e5
        module.begin_step()
        module(samples[:2])
        module(samples[2:])
        self.assertEqual(module.input_cov_weight.item(), 0)
        module.finish_step(ema=.75)
        rows = samples[:, 1::2].reshape(-1, 3)
        torch.testing.assert_close(module.input_cov / module.input_cov_weight, rows.T @ rows / len(rows))
        torch.testing.assert_close(module.input_mean / module.input_weight, samples[:, 1:].mean((0, 1)))
        torch.testing.assert_close(module.input_sq / module.input_weight, samples[:, 1:].square().sum(-1).mean())
        self.assertAlmostEqual(module.input_cov_weight.item(), .25)

    def test_stats_are_microbatch_partition_invariant_and_no_validation_leak(self):
        a, b = StepStatLinear(3, 2), StepStatLinear(3, 2)
        samples = torch.randn(7, 5, 3)
        a.begin_step()
        b.begin_step()
        a(samples)
        b(samples[3:])
        b(samples[:3])
        # Evaluation under an active step and no-grad training forwards must be ignored.
        b.eval()
        b(samples * 1e6)
        b.train()
        with torch.no_grad():
            b(samples * 1e6)
        a.finish_step(.9)
        b.finish_step(.9)
        for name in ("input_mean", "input_sq", "input_weight", "input_cov", "input_cov_mean", "input_cov_weight"):
            torch.testing.assert_close(getattr(a, name), getattr(b, name))
        before = a.input_cov.clone()
        a(samples * 1e6)  # No active collection window.
        torch.testing.assert_close(a.input_cov, before, atol=0, rtol=0)

    def test_covariance_stays_fp32_under_model_autocast(self):
        module = StepStatLinear(3, 2, cov_stride=1)
        x = torch.randn(3, 8, 3)
        module.begin_step()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            output = module(x)
        module.finish_step(ema=0)
        rows = x[:, 1:].reshape(-1, 3)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(module.input_cov.dtype, torch.float32)
        torch.testing.assert_close(module.input_cov, rows.T @ rows / len(rows), atol=0, rtol=0)


class DataContracts(unittest.TestCase):
    def corpus(self):
        return corpus_from_text("".join(chr(1000 + i) for i in range(200)),
                                train_fraction=.8, test_fraction=.1)

    def test_contiguous_disjoint_splits_and_shifted_targets(self):
        corpus = self.corpus()
        self.assertEqual(corpus.boundaries, {"train": (0, 160), "val": (160, 180), "test": (180, 200)})
        sets = [set(corpus.tokens(split).tolist()) for split in ("train", "val", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        x, y = corpus.batch("train", 1, 8, positions=[151])
        torch.testing.assert_close(x[:, 1:], y[:, :-1])
        self.assertEqual(y[0, -1].item(), corpus.tokens("train")[-1].item())
        with self.assertRaises(ValueError):
            corpus.batch("train", 1, 8, positions=[152])

    def test_paired_random_windows_and_fixed_positions(self):
        corpus = self.corpus()
        a, b = torch.Generator().manual_seed(123), torch.Generator().manual_seed(123)
        x, y, positions = corpus.batch("train", 12, 8, generator=a, return_positions=True)
        x2, y2 = corpus.batch("train", 12, 8, generator=b)
        self.assertTrue(torch.equal(x, x2) and torch.equal(y, y2))
        x3, y3 = corpus.batch("train", 12, 8, positions=positions)
        self.assertTrue(torch.equal(x, x3) and torch.equal(y, y3))
        with self.assertRaises(ValueError):
            corpus.batch("train", 12, 8)

    def test_global_window_stream_can_be_repartitioned_by_batch(self):
        corpus = self.corpus()
        small_rng, big_rng = torch.Generator().manual_seed(71), torch.Generator().manual_seed(71)
        small = [corpus.batch("train", 3, 8, generator=small_rng)[0] for _ in range(4)]
        big = corpus.batch("train", 12, 8, generator=big_rng)[0]
        torch.testing.assert_close(torch.cat(small), big, atol=0, rtol=0)

    def test_vocabulary_checksums_and_local_load_have_no_network(self):
        text = "Hello, world!\nαβγ\n" * 30
        a, b = corpus_from_text(text), corpus_from_text(text)
        self.assertEqual(a.manifest, b.manifest)
        self.assertEqual(a.vocabulary, tuple(sorted(set(text))))
        self.assertEqual(a.decode(a.encode(text)), text)
        self.assertEqual(a.manifest["raw_utf8_sha256"], sha256_bytes(text.encode()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.txt"
            path.write_bytes(text.encode())
            with patch("urllib.request.urlopen", side_effect=AssertionError("Unexpected network")):
                loaded = load_corpus(path, expected_sha256=sha256_bytes(text.encode()))
            torch.testing.assert_close(loaded.token_ids, a.token_ids)
            with self.assertRaises(ValueError):
                load_corpus(path, expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()

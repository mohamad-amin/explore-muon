"""Numerical and end-to-end qualification, using stdlib unittest."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from .data import TokenStream, check_disjoint, token_views
from .measure import QUANTILES, adamw_direction, spectrum, measure_updates
from .model import GPT, ModelConfig
from .sweep import select_candidate
from .train import learning_rate, load_config, run, token_budget, make_optimizer


def write_shard(path, values):
    header = np.zeros(256, dtype="<i4")
    header[:3] = (20240520, 1, len(values))
    with Path(path).open("wb") as handle:
        handle.write(header.tobytes())
        handle.write(np.asarray(values, dtype="<u2").tobytes())


def tiny_config(root):
    write_shard(root / "train.bin", np.arange(2048) % 31)
    write_shard(root / "val.bin", (np.arange(2048) * 7 + 3) % 31)
    return load_config(overrides={
        "model": {"vocab_size": 32, "n_layer": 2, "n_embd": 16, "n_head": 2, "seq_len": 8},
        "train_pattern": str(root / "train.bin"), "validation_pattern": str(root / "val.bin"),
        "batch_tokens": 32, "microbatch_sequences": 2, "total_tokens": 176,
        "validation_tokens": 32, "validation_every": 2, "warmup_steps": 1,
        "checkpoint_every": 2, "cpu_threads": 1, "precision": "fp32", "spectra_every": 1})


class MeasurementTests(unittest.TestCase):
    def test_update_matches_actual_delta_over_multiple_steps(self):
        torch.manual_seed(12)
        parameter = torch.nn.Parameter(torch.randn(7, 4, dtype=torch.float64))
        optimizer = torch.optim.AdamW([parameter], lr=.013, betas=(.81, .94),
                                     eps=.003, weight_decay=.17, foreach=False)
        for step in range(1, 6):
            before = parameter.detach().clone()
            parameter.grad = torch.randn_like(parameter) * step
            optimizer.param_groups[0]["lr"] = lr = .013 / step
            optimizer.step()
            buffers = {k: v.clone() for k, v in optimizer.state[parameter].items()}
            update, _ = adamw_direction(optimizer, parameter)
            adaptive_delta = parameter.detach() - before + lr * .17 * before
            torch.testing.assert_close(adaptive_delta, -lr * update, rtol=1e-11, atol=1e-15)
            for key, value in buffers.items():
                self.assertTrue(torch.equal(value, optimizer.state[parameter][key]))

    def test_rectangular_known_spectrum_and_descending_quantiles(self):
        values = torch.arange(10, 0, -1, dtype=torch.float32)
        matrix = torch.cat((torch.diag(values), torch.zeros(10, 3)), dim=1)
        original = matrix.clone()
        stats, result = spectrum(matrix)
        expected = (values / values.norm()).numpy()
        np.testing.assert_allclose(result, expected, rtol=2e-6)
        for q, index in zip(QUANTILES, (0, 2, 4, 7, 8)):
            self.assertAlmostEqual(stats["quantiles"][str(q)], float(expected[index]), places=6)
        self.assertAlmostEqual(stats["normalized_energy_sum"], 1.0, places=6)
        self.assertAlmostEqual(stats["stable_rank"], float(values.square().sum() / 100), places=5)
        self.assertTrue(torch.equal(matrix, original))
        np.testing.assert_allclose(spectrum(matrix.T)[1], expected, rtol=2e-6)

    def test_zero_and_nonfinite_are_explicit(self):
        stats, values = spectrum(torch.zeros(4, 9))
        self.assertTrue(stats["zero_matrix"])
        self.assertIsNone(stats["quantiles"]["0.5"])
        self.assertTrue((values == 0).all())
        with self.assertRaises(ValueError):
            spectrum(torch.full((4, 4), float("nan")))

    def test_nearly_uniform_full_size_update_normalization(self):
        # First-step Adam directions approach sign(gradient), unlike Gaussian
        # matrices; this exposes inaccurate long FP32 norm reductions.
        torch.set_num_threads(1)
        matrix = torch.full((2048, 512), .999, dtype=torch.float32)
        matrix[::2] *= -1
        expected_norm = abs(float(matrix[0, 0])) * 1024
        stats, values = spectrum(matrix)
        self.assertAlmostEqual(stats["frobenius_norm"], expected_norm, places=7)
        self.assertAlmostEqual(stats["normalized_energy_sum"], 1., places=5)
        self.assertAlmostEqual(float(values[0]), 1., places=5)

    def test_fast_optimizer_implementations_and_batched_measurement(self):
        torch.manual_seed(19)
        initial = torch.randn(8, 5)
        gradients = [torch.randn_like(initial) for _ in range(5)]
        final = {}
        for implementation in ("single", "foreach", "fused"):
            model = torch.nn.Linear(5, 8, bias=False)
            model.weight.data.copy_(initial)
            config = load_config(overrides={"optimizer_impl": implementation})
            optimizer, resolved = make_optimizer(model, config, "cpu")
            self.assertEqual(resolved, implementation)
            for grad in gradients:
                model.weight.grad = grad.clone()
                before = model.weight.detach().clone()
                optimizer.step()
                update, _ = adamw_direction(optimizer, model.weight)
                expected = before * (1 - config["learning_rate"] * config["weight_decay"])
                expected -= config["learning_rate"] * update
                torch.testing.assert_close(model.weight, expected, atol=3e-7, rtol=2e-6)
            buffers = {k: v.clone() for k, v in optimizer.state[model.weight].items()}
            rows, arrays = measure_updates(optimizer, {"weight": model.weight})
            expected_stats, expected_values = spectrum(update)
            np.testing.assert_array_equal(arrays["weight"], expected_values)
            self.assertEqual(rows["weight"]["quantiles"], expected_stats["quantiles"])
            for key, value in buffers.items():
                self.assertTrue(torch.equal(value, optimizer.state[model.weight][key]))
            final[implementation] = model.weight.detach().clone()
        for value in final.values():
            torch.testing.assert_close(value, final["single"], atol=3e-7, rtol=2e-6)


class DataAndScheduleTests(unittest.TestCase):
    def test_shard_boundary_targets_no_wrap_and_overlap(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            write_shard(root / "a.bin", np.arange(10))
            write_shard(root / "b.bin", np.arange(10, 20))
            stream = TokenStream(str(root / "*.bin"))
            x, y = stream.batch(8, 2, 4, "cpu")
            self.assertEqual(x.flatten().tolist(), list(range(8, 16)))
            self.assertEqual(y.flatten().tolist(), list(range(9, 17)))
            packed = stream.device_tokens(4, 13, "cpu")
            for offset, count in ((0, 8), (8, 4)):
                cached_x, cached_y = token_views(packed, offset, count, 4)
                eager_x, eager_y = stream.batch(4 + offset, count // 4, 4, "cpu")
                self.assertTrue(torch.equal(cached_x, eager_x))
                self.assertTrue(torch.equal(cached_y, eager_y))
            with self.assertRaises(ValueError):
                stream.read(18, 3)
            with self.assertRaises(ValueError):
                check_disjoint(stream, TokenStream(str(root / "a.bin")))

    def test_bad_length_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.bin"
            write_shard(path, np.arange(10))
            with path.open("ab") as handle:
                handle.write(b"x")
            with self.assertRaises(ValueError):
                TokenStream(str(path))

    def test_budget_and_schedule(self):
        config = load_config()
        self.assertEqual(token_budget(config, 101, 512), 2048)
        self.assertAlmostEqual(learning_rate(config, 1, 0, 2e9), config["learning_rate"] / 50)
        self.assertAlmostEqual(learning_rate(config, 100, 1e8, 2e9), config["learning_rate"])
        self.assertAlmostEqual(learning_rate(config, 1800, 1.9e9, 2e9), config["learning_rate"] / 2)
        self.assertEqual(learning_rate(config, 2000, 2e9, 2e9), 0)

    def test_lr_selection_invalid_and_tie(self):
        entries = [{"valid": True, "score": 3., "learning_rate": .002},
                   {"valid": True, "score": 3., "learning_rate": .001},
                   {"valid": False, "score": 0., "learning_rate": .003}]
        self.assertEqual(select_candidate(entries)["learning_rate"], .001)
        with self.assertRaises(ValueError):
            select_candidate([])


class TrainingTests(unittest.TestCase):
    def test_full_model_size_and_panels(self):
        torch.set_num_threads(1)
        model = GPT()
        count = sum(p.numel() for p in model.parameters())
        self.assertTrue(76_000_000 < count < 78_000_000, count)
        self.assertEqual(len(model.measured_parameters()), 24)
        self.assertEqual({n.split(".")[0] for n in model.measured_parameters()},
                         {"block02", "block04", "block06", "block08"})
        self.assertIsNot(model.head.weight, model.embed.weight)

    def test_resume_identical_and_measurement_noninterference(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = tiny_config(root)
            complete = run(config, root / "complete", device="cpu")
            self.assertEqual(complete["tokens"], 176)
            self.assertEqual(complete["step"], 6)
            final_row = json.loads((root / "complete/steps/step000006.json").read_text())
            self.assertEqual(final_row["batch_tokens"], 16)
            self.assertGreater(final_row["lr"], 0)
            run(config, root / "resumed", device="cpu", stop_after=3)
            # Simulate a record written after the durable checkpoint.
            (root / "resumed/steps/step000004.json").write_text('{"uncommitted": true}')
            (root / "resumed/steps/step000004.json.tmp").write_text('{"partial":')
            (root / "resumed/spectra/step000004.tmp").write_bytes(b"partial npz")
            run(config, root / "resumed", device="cpu", resume=True)
            run({**config, "spectra_every": 0}, root / "unmeasured", device="cpu")
            reference = torch.load(root / "complete/checkpoint.pt", weights_only=False)
            for name in ("resumed", "unmeasured"):
                saved = torch.load(root / name / "checkpoint.pt", weights_only=False)
                for key, value in reference["model"].items():
                    self.assertTrue(torch.equal(value, saved["model"][key]), key)
                for param_id, state in reference["optimizer"]["state"].items():
                    for key, value in state.items():
                        self.assertTrue(torch.equal(value, saved["optimizer"]["state"][param_id][key]))
            self.assertEqual(json.loads((root / "complete/steps/step000006.json").read_text())["validation_nll"],
                             json.loads((root / "resumed/steps/step000006.json").read_text())["validation_nll"])
            with self.assertRaises(FileExistsError):
                run(config, root / "complete", device="cpu")

    def test_analysis_and_failure_status(self):
        from .analyze import analyze
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            config = tiny_config(root)
            run(config, root / "run", device="cpu")
            summary = analyze(root / "run")
            self.assertTrue(summary["budget_complete"])
            self.assertEqual(summary["durable_checkpoint"]["step"], 6)
            self.assertEqual(summary["windows"]["1300-1500"]["count"], 0)
            self.assertTrue((root / "run/plots/median.pdf").exists())
            (root / "run/status.json").write_text('{"status":"failed"}')
            summary = analyze(root / "run")
            self.assertFalse(summary["budget_complete"])


if __name__ == "__main__":
    unittest.main()

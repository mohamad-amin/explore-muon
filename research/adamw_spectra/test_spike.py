import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from .model import GPT, ModelConfig
from .spike_diagnostics import (Recorder, bands, batch_gradient, body_layers, mode_projections,
                                ns_scalar, run, state_convention, validation_nll)
from .train import load_config, make_optimizer

TINY = {"vocab_size": 64, "n_layer": 2, "n_embd": 16, "n_head": 2, "seq_len": 8}


def write_shard(path, tokens):
    header = np.zeros(256, dtype="<i4")
    header[:3] = (20240520, 1, len(tokens))
    with open(path, "wb") as handle:
        handle.write(header.tobytes())
        handle.write(np.asarray(tokens, dtype="<u2").tobytes())


class SpikeDiagnosticTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.model = GPT(ModelConfig(**TINY))
        self.config = load_config(overrides={"model": TINY, "precision": "fp32", "compile": False})
        self.layers = body_layers(self.model)
        self.pairs = {}
        for name, layer in self.layers.items():
            out_dim, in_dim = layer.weight.shape
            u, v = torch.randn(out_dim), torch.randn(in_dim)
            self.pairs[name] = (u / u.norm(), v / v.norm())
        self.tokens = torch.randint(0, 64, (4 * 8 + 1,))

    def test_hooks_reconstruct_gradient_and_split_projection(self):
        recorder = Recorder(self.layers, self.pairs, 8, bf16_inputs=False)
        recorder.attach()
        recorder.reconstruct = True
        batch_gradient(self.model, self.tokens, 8, 2, self.config, torch.device("cpu"))
        recorder.end_batch()
        for name, layer in self.layers.items():
            st, grad = recorder.stats[name], layer.weight.grad
            torch.testing.assert_close(st["reconstruction"], grad, rtol=1e-5, atol=1e-7)
            u1, v1 = self.pairs[name]
            self.assertAlmostEqual(float(st["contribution"]), float(u1 @ grad @ v1), places=6)
            self.assertAlmostEqual(float(st["position"].sum()), float(st["contribution"]), places=6)
            # Sum of output gradients is exactly the Linear bias gradient.
            torch.testing.assert_close(st["sum_grad"].float(), layer.bias.grad, rtol=1e-5, atol=1e-7)
            self.assertAlmostEqual(float(u1.double() @ st["pos0_term"] @ v1.double()),
                                   float(st["position"][0]), places=6)
            torch.testing.assert_close(st["batchwise_mean_product"],
                                       torch.outer(st["sum_grad"], st["sum_input"]) / st["tokens"])
            self.assertEqual(st["tokens"], 32)
        recorder.detach()

    def test_mode_projections_and_bands(self):
        state = torch.randn(12, 7, dtype=torch.float64)
        u, s, vh = torch.linalg.svd(state, full_matrices=False)
        torch.testing.assert_close(mode_projections(u, vh, state), s)
        grad = torch.randn(12, 7, dtype=torch.float64)
        self.assertAlmostEqual(float(mode_projections(u, vh, grad).sum()),
                               float(torch.trace(u.T @ grad @ vh.T)), places=10)
        covered = [m for _, lo, hi in bands(512) for m in range(lo, hi + 1)]
        self.assertEqual(covered, list(range(1, 513)))

    def test_conventions(self):
        key, units, noise = state_convention({"optimizer": "muon", "muon_momentum": 0.95})
        self.assertEqual((key, round(units, 12)), ("momentum_buffer", 0.05))
        self.assertAlmostEqual(noise, math.sqrt(0.05 / 1.95))
        key, units, noise = state_convention({"optimizer": "adamw", "betas": [0.9, 0.95]})
        self.assertEqual((key, units), ("exp_avg", 1.0))
        self.assertAlmostEqual(noise, math.sqrt(0.1 / 1.9))
        self.assertAlmostEqual(float(ns_scalar(1e-7) / 1e-7), 492.7496, places=3)
        self.assertAlmostEqual(float(ns_scalar(1e-3)), 0.475, places=3)

    def end_to_end(self, optimizer_name):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            rng = np.random.default_rng(0)
            write_shard(tmp / "fineweb_train_000001.bin", rng.integers(0, 64, 3000))
            write_shard(tmp / "fineweb_val_000000.bin", rng.integers(0, 64, 400))
            config = load_config(overrides={
                "model": TINY, "precision": "fp32", "compile": False, "optimizer": optimizer_name,
                "learning_rate": 0.01, "validation_tokens": 64,
                "train_pattern": str(tmp / "fineweb_train_*.bin"),
                "validation_pattern": str(tmp / "fineweb_val_*.bin")})
            model = GPT(ModelConfig(**TINY))
            optimizer, _ = make_optimizer(model, config, torch.device("cpu"))
            for start in (0, 64):
                tokens = torch.from_numpy(rng.integers(0, 64, 65)).long()
                batch_gradient(model, tokens, 8, 2, config, torch.device("cpu"))
                optimizer.step()
            from .data import TokenStream
            val = TokenStream(config["validation_pattern"]).device_tokens(0, 65, "cpu")
            nll = validation_nll(model, val, config, torch.device("cpu"))
            run_dir = tmp / "run"
            (run_dir / "steps").mkdir(parents=True)
            saved_config = dict(config)
            if optimizer_name == "adamw":  # pre-Muon checkpoints lack these keys
                for key in ("optimizer", "aux_learning_rate", "muon_momentum"):
                    saved_config.pop(key)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": 2, "tokens": 128, "config": saved_config}, run_dir / "checkpoint.pt")
            (run_dir / "steps" / "step000002.json").write_text(json.dumps({"validation_nll": nll}))
            out = tmp / "diag"
            run(run_dir, out, offset=200, batches=4, batch_tokens=64, micro_sequences=2, device="cpu")
            status = json.loads((out / "status.json").read_text())
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(status["status"], "complete")
            self.assertEqual(len(summary["matrices"]), 12)
            for name, entry in summary["matrices"].items():
                rows = entry["bands"]
                arrays = np.load(out / "arrays.npz")
                proj = arrays[f"{name}/projections"]
                self.assertAlmostEqual(sum(r["cross_fitted"] for r in rows), proj.mean(0).sum(), places=8)
                self.assertAlmostEqual(sum(r["in_sample"] for r in rows),
                                       arrays[f"{name}/state_sigma"].sum(), places=8)
                self.assertLess(entry["fresh_mean_gradient"]["hook_vs_gradient_rel_diff"], 1e-5)
                self.assertIn(entry["fresh_mean_gradient"]["spike_reproduced"], (True, False))
                self.assertGreaterEqual(entry["noise"]["edge_batch_1M"], entry["noise"]["edge2_batch_1M"])
            self.assertLess(summary["gates"]["validation_nll"]["abs_diff"], 1e-9)

    def test_end_to_end_muon(self):
        self.end_to_end("muon")

    def test_end_to_end_adamw(self):
        self.end_to_end("adamw")


if __name__ == "__main__":
    unittest.main()

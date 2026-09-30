import json
import random
import unittest
from pathlib import Path

import numpy as np
import torch

from .model import GPT, ModelConfig, RMSNorm
from .spike_diagnostics import forward_structure
from .train import load_config, make_optimizer, model_hash

ROOT = Path(__file__).resolve().parents[2]
TINY = {"vocab_size": 64, "n_layer": 2, "n_embd": 16, "n_head": 2, "seq_len": 8}
VARIANT = {"bias": False, "norm": "rmsnorm", "qk_norm": True}


class ArchitectureVariantTest(unittest.TestCase):
    def test_default_matches_recorded_initial_weights(self):
        for run in ("logs/muon_spectra/depth8_w512_20260925_r2/scientific",
                    "logs/muon_spectra/depth12_w512_20260925_r2/scientific"):
            path = ROOT / run / "metadata.json"
            if not path.exists():
                self.skipTest(f"missing {path}")
            meta = json.loads(path.read_text())
            seed = meta["config"]["seed"]
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            model = GPT(ModelConfig(**meta["config"]["model"]))
            self.assertEqual(model_hash(model), meta["initial_model_sha256"])

    def test_variant_structure(self):
        torch.manual_seed(0)
        model = GPT(ModelConfig(**TINY, **VARIANT))
        names = dict(model.named_parameters())
        self.assertFalse([n for n in names if n.endswith(".bias")])
        for block in model.blocks:
            for norm in (block.ln1, block.ln2):
                self.assertIsInstance(norm, RMSNorm)
            self.assertEqual(tuple(block.attn.q_norm.weight.shape), (8,))
            self.assertEqual(tuple(block.attn.k_norm.weight.shape), (8,))
        self.assertIsInstance(model.norm, RMSNorm)
        self.assertEqual(len(model.measured_parameters()), 12)
        baseline = GPT(ModelConfig(**TINY))
        removed = sum(p.numel() for n, p in baseline.named_parameters() if n.endswith(".bias"))
        added = 2 * 2 * 8  # q/k gains per block
        self.assertEqual(sum(p.numel() for p in model.parameters()),
                         sum(p.numel() for p in baseline.parameters()) - removed + added)

    def test_rmsnorm_formula_and_qk_heads(self):
        torch.manual_seed(1)
        norm = RMSNorm(6)
        with torch.no_grad():
            norm.weight.copy_(torch.linspace(0.5, 1.5, 6))
        x = torch.randn(3, 6)
        expected = x / torch.sqrt((x * x).mean(-1, keepdim=True) + 1e-6) * norm.weight
        torch.testing.assert_close(norm(x), expected)
        model = GPT(ModelConfig(**TINY, **VARIANT))
        q, k, v = model.blocks[0].attn.heads(100 * torch.randn(2, 8, 16))  # epsilon negligible
        for tensor in (q, k):
            torch.testing.assert_close(tensor.pow(2).mean(-1), torch.ones(2, 2, 8), rtol=1e-4, atol=1e-4)
        self.assertGreater(float((v.pow(2).mean(-1) - 1).abs().max()), 1e-1)  # values stay unnormalized

    def test_variant_trains_with_muon_grouping(self):
        torch.manual_seed(2)
        config = load_config(overrides={"model": {**TINY, **VARIANT}, "optimizer": "muon",
                                        "precision": "fp32", "compile": False})
        model = GPT(ModelConfig(**config["model"]))
        optimizer, _ = make_optimizer(model, config, torch.device("cpu"))
        body = optimizer.param_groups[0]["params"]
        self.assertEqual(len(body), 12)
        self.assertTrue(all(p.ndim == 2 for p in body))
        gains = [model.blocks[0].attn.q_norm.weight, model.blocks[0].ln1.weight, model.norm.weight]
        aux = optimizer.param_groups[1]["params"]
        self.assertTrue(all(any(g is p for p in aux) for g in gains))
        tokens = torch.randint(0, 64, (4, 9))
        loss = model(tokens[:, :-1], tokens[:, 1:])
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        before = model.blocks[0].attn.q.weight.detach().clone()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.blocks[0].attn.q.weight))

    def test_forward_structure_uses_qk_norm(self):
        torch.manual_seed(3)
        config = load_config(overrides={"model": {**TINY, **VARIANT}, "precision": "fp32", "compile": False})
        model = GPT(ModelConfig(**config["model"]))
        tokens = torch.randint(0, 64, (4 * 8 + 1,))
        result = forward_structure(model, tokens, config, torch.device("cpu"), sequences=4)
        self.assertEqual(len(result["attention_to_position0"]), 2)
        self.assertEqual(len(result["residual_stream"]), 3)


if __name__ == "__main__":
    unittest.main()

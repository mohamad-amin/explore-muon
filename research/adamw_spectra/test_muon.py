"""Numerical contract checks for the mixed optimizer and captured measurements."""

import copy
import math
import unittest

import torch

from .measure import measure_updates
from .model import GPT, ModelConfig
from .muon import NS_COEFFICIENTS, newton_schulz
from .train import load_config, make_optimizer


def polar_polynomial(matrix):
    """Independent scalar/SVD reference, not the matrix-multiplication code path."""
    u, s, vh = torch.linalg.svd(matrix.double(), full_matrices=False)
    s = s / matrix.double().norm().clamp_min(1e-7)
    for a, b, c in NS_COEFFICIENTS:
        s = a * s + b * s ** 3 + c * s ** 5
    return ((u * s) @ vh).float()


class MuonContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(18)
        self.model = GPT(ModelConfig(vocab_size=32, n_layer=2, n_embd=8, n_head=2, seq_len=8))
        self.cfg = load_config(overrides=dict(optimizer="muon", learning_rate=.01, compile=False))
        self.opt, _ = make_optimizer(self.model, self.cfg, "cpu")

    def test_polynomial_square_tall_wide_zero_and_batches(self):
        for shape in ((8, 8), (32, 8), (8, 32)):
            x = torch.randn(3, *shape)
            x[-1].zero_()
            observed = newton_schulz(x)
            expected = torch.stack([polar_polynomial(v) for v in x])
            torch.testing.assert_close(observed, expected, atol=3e-5, rtol=3e-5)
            self.assertEqual(observed[-1].count_nonzero(), 0)

    def test_parameter_masks_and_auxiliary_reference(self):
        muon_ids = {id(p) for p in self.opt.param_groups[0]["params"]}
        aux_ids = {id(p) for p in self.opt.param_groups[1]["params"]}
        self.assertFalse(muon_ids & aux_ids)
        self.assertEqual(muon_ids | aux_ids, {id(p) for p in self.model.parameters()})
        for name, p in self.model.named_parameters():
            self.assertEqual(id(p) in muon_ids, name.startswith("blocks.") and p.ndim == 2)
        refs = [torch.nn.Parameter(p.detach().clone()) for p in self.opt.param_groups[1]["params"]]
        reference = torch.optim.AdamW(refs, lr=.002, betas=(.9, .95), eps=1e-8, weight_decay=.01,
                                      foreach=True)
        for _ in range(3):
            for p in self.model.parameters():
                p.grad = torch.randn_like(p)
            for p, q in zip(self.opt.param_groups[1]["params"], refs):
                q.grad = p.grad.clone()
            reference.step()
            self.opt.step()
            for p, q in zip(self.opt.param_groups[1]["params"], refs):
                torch.testing.assert_close(p, q, atol=0, rtol=0)

    def test_parameter_write_and_readonly_measurements(self):
        measured = self.model.measured_parameters()
        self.opt.capture_parameters = set(measured.values())
        buffers = {p: torch.zeros_like(p) for p in measured.values()}
        for _ in range(3):
            before = {p: p.detach().clone() for p in self.model.parameters()}
            for p in self.model.parameters():
                p.grad = torch.randn_like(p)
            for p in measured.values():
                buffers[p] = .95 * buffers[p] + p.grad
            self.opt.step()
            for p in measured.values():
                scale = math.sqrt(max(1, p.shape[0] / p.shape[1]))
                expected = before[p] * (1 - .01 * .01) - .01 * scale * polar_polynomial(buffers[p])
                torch.testing.assert_close(p, expected, atol=5e-7, rtol=3e-5)
                torch.testing.assert_close(before[p] * (1 - .01 * .01) - .01 * self.opt.last_updates[p],
                                           p, atol=1e-8, rtol=1e-6)
            state_before = copy.deepcopy(self.opt.state_dict())
            for quantity in ("update", "momentum"):
                rows, values = measure_updates(self.opt, measured, quantity=quantity)
                for name in measured:
                    self.assertLess(abs(rows[name]["normalized_energy_sum"] - 1), 2e-4)
                    self.assertEqual(len(values[name]), min(measured[name].shape))
            for key, state in self.opt.state_dict()["state"].items():
                for field, value in state.items():
                    torch.testing.assert_close(value, state_before["state"][key][field], atol=0, rtol=0)

    def test_resume_preserves_mixed_state_and_group_rates(self):
        for p in self.model.parameters():
            p.grad = torch.randn_like(p)
        self.opt.step()
        restored = copy.deepcopy(self.model)
        restored_opt, _ = make_optimizer(restored, self.cfg, "cpu")
        restored_opt.load_state_dict(copy.deepcopy(self.opt.state_dict()))
        for p, q in zip(self.model.parameters(), restored.parameters()):
            p.grad = torch.randn_like(p)
            q.grad = p.grad.clone()
        for opt in (self.opt, restored_opt):
            for group in opt.param_groups:
                group["lr"] = .007 * group["lr_scale"]
            opt.step()
        self.assertAlmostEqual(restored_opt.param_groups[1]["lr"], .0014)
        for p, q in zip(self.model.parameters(), restored.parameters()):
            torch.testing.assert_close(p, q, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

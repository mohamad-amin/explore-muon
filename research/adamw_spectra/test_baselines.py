"""Ported comparators: Newton-Muon (Track-3 record #15) and PMuon (record #18)."""
import math
import unittest

import torch

from .model import GPT, ModelConfig, StatLinear
from .muon import diagonal_blocks, newton_inverse, newton_schulz, streaming_cov_power
from .test_whitening import TINY, one_step
from .train import load_config, make_optimizer

STATS = {**TINY, "track_input_stats": True, "track_input_cov": True}


def reference_cov_power(C, state, key, gamma, eps=1e-6):
    """Verbatim from the record #18 logfiles (`_streaming_cov_power`)."""
    n = C.size(0)
    Q = state.get(key, None)
    if Q is None or Q.shape != (n, n) or Q.device != C.device:
        Q, _ = torch.linalg.qr(torch.randn(n, n, device=C.device, dtype=C.dtype), mode='reduced')
    Q, _ = torch.linalg.qr(C @ Q, mode='reduced')
    state[key] = Q.detach()
    lam = (Q * (C @ Q)).sum(dim=0).clamp_min(eps)
    d = lam.pow(-gamma)
    res = Q @ torch.diag(d) @ Q.T
    res *= (n ** 0.5 / (d.norm() + eps))
    return res


def reference_inverse(cov, damping=0.2):
    """Record #15 `_refresh_preconditioner`: inverse of cov + (damping * trace / d + 1e-8) I (in FP64)."""
    d = cov.size(-1)
    reg = (cov.diagonal(dim1=-2, dim2=-1).sum(-1) / d * damping + 1e-8)[..., None, None]
    return torch.linalg.inv(cov.double() + reg.double() * torch.eye(d, dtype=torch.float64)).float()


def spd(n, seed, spread=100.0):
    q, _ = torch.linalg.qr(torch.randn(n, n, generator=torch.Generator().manual_seed(seed)))
    return (q * torch.logspace(0, math.log10(spread), n)) @ q.T


def fresh(overrides):
    torch.manual_seed(0)
    config = load_config(overrides={"optimizer": "muon", "precision": "fp32", "compile": False, **overrides})
    model = GPT(ModelConfig(**config["model"]))
    return model, make_optimizer(model, config, torch.device("cpu"))[0]


def backward(model, seed):
    tokens = torch.randint(0, 64, (4, 9), generator=torch.Generator().manual_seed(seed))
    model(tokens[:, :-1], tokens[:, 1:]).backward()


class NewtonMuonTest(unittest.TestCase):
    def test_inverse_and_blocks_match_reference(self):
        cov = torch.stack([spd(12, 1), spd(12, 2, 1e4)])
        torch.testing.assert_close(newton_inverse(cov, 0.2), reference_inverse(cov), rtol=1e-4, atol=1e-5)
        x = torch.randn(50, 4 * 6)
        # The reference accumulates MLP-down statistics as (4, d, d) blocks of x reshaped to (n, 4, d).
        blocks = x.reshape(50, 4, 6).permute(1, 2, 0)
        torch.testing.assert_close(diagonal_blocks(x.T @ x / 50, 4), torch.bmm(blocks, blocks.mT) / 50)

    def test_schedule_ema_and_preconditioned_momentum(self):
        model, optimizer = fresh({"model": STATS, "newton_muon": True, "newton_muon_refresh": 2})
        body = optimizer.param_groups[0]["params"]
        backward(model, 1)
        first_grads = {p: p.grad.detach().clone() for p in body}
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        self.assertFalse(optimizer.newton["ready"])            # step 1 is plain Muon: M1 = G1
        for p in body:
            self.assertTrue(torch.equal(optimizer.state[p]["momentum_buffer"], first_grads[p]))
        # Step 2 refreshes C = lerp(0.001 I, K, 0.05) from the current statistics; G K enters the momentum.
        backward(model, 2)
        modules = {m.weight: m for m in model.modules() if isinstance(m, StatLinear)}
        expected = {}
        for name, p in model.named_parameters():
            if not (name.startswith("blocks.") and p.ndim == 2):
                continue
            second = modules[p].input_cov / modules[p].input_cov_weight
            blocks = 4 if name.split(".")[-2] == "down" else 1
            stat = diagonal_blocks(second, blocks) if blocks > 1 else second[None]
            inverse = reference_inverse(torch.lerp(0.001 * torch.eye(stat.size(-1)).expand_as(stat), stat, 0.05))
            split = p.grad.detach().view(p.shape[0], blocks, -1).transpose(0, 1)
            expected[p] = 0.95 * first_grads[p] + torch.bmm(split, inverse).transpose(0, 1).reshape_as(p)
        optimizer.step()
        self.assertTrue(optimizer.newton["ready"])
        for p in body:
            torch.testing.assert_close(optimizer.state[p]["momentum_buffer"], expected[p], rtol=2e-4, atol=1e-6)

    def test_deterministic_and_validated(self):
        overrides = {"model": STATS, "newton_muon": True, "newton_muon_refresh": 2}
        a, _, _ = one_step(overrides, steps=4)
        b, _, _ = one_step(overrides, steps=4)
        for (_, x), (_, y) in zip(a.named_parameters(), b.named_parameters()):
            self.assertTrue(torch.equal(x, y))
        with self.assertRaises(ValueError):          # needs activation statistics
            load_config(overrides={"model": TINY, "optimizer": "muon", "newton_muon": True})
        for other in ({"pmuon": True}, {"data_norm_alpha": 0.25}, {"soap_precondition": True}):
            with self.assertRaises(ValueError):
                load_config(overrides={"model": STATS, "optimizer": "muon", "newton_muon": True, **other})


class PMuonTest(unittest.TestCase):
    def test_streaming_power_matches_reference(self):
        start = torch.linalg.qr(torch.randn(10, 10, generator=torch.Generator().manual_seed(3)))[0]
        ours, theirs = {"q": start.clone()}, {"q": start.clone()}
        for seed in range(3):
            cov = spd(10, 10 + seed, 1e3)
            torch.testing.assert_close(streaming_cov_power(cov, ours, "q", 0.3, None),
                                       reference_cov_power(cov, theirs, "q", 0.3), rtol=1e-5, atol=1e-6)

    def test_first_step_update_matches_reference_formula(self):
        model, optimizer = fresh({"model": TINY, "pmuon": True})
        body = optimizer.param_groups[0]["params"]
        starts = {}                                     # known starting bases shared with the reference
        for i, p in enumerate(body):
            generator = torch.Generator().manual_seed(100 + i)
            rows, cols = p.shape
            starts[p] = (torch.linalg.qr(torch.randn(cols, cols, generator=generator))[0],
                         torch.linalg.qr(torch.randn(rows, rows, generator=generator))[0])
            optimizer.pmuon["states"][p] = {"right": torch.zeros(cols, cols), "left": torch.zeros(rows, rows),
                                            "generator": None, "q_right": starts[p][0].clone(),
                                            "q_left": starts[p][1].clone()}
        backward(model, 1)
        grads = {p: p.grad.detach().clone() for p in body}
        optimizer.capture_parameters = set(body)
        optimizer.step()
        beta, eps = 0.95, 1e-6
        for p in body:
            u = grads[p] * 0.05                         # first-step EMA momentum (1 - mu) G
            right = (1 + beta) * (u.T @ u) + eps * torch.eye(u.shape[1])
            left = (1 + beta) * (u @ u.T) + eps * torch.eye(u.shape[0])
            c = reference_cov_power(right, {"k": starts[p][0].clone()}, "k", 0.3)
            a = reference_cov_power(left, {"k": starts[p][1].clone()}, "k", 0.3)
            expected = newton_schulz((a @ u @ c)[None])[0] * math.sqrt(max(1.0, p.shape[0] / p.shape[1]))
            torch.testing.assert_close(optimizer.last_updates[p], expected, rtol=1e-4, atol=1e-5)

    def test_deterministic_and_validated(self):
        a, _, _ = one_step({"model": TINY, "pmuon": True}, steps=3)
        b, _, _ = one_step({"model": TINY, "pmuon": True}, steps=3)
        for (_, x), (_, y) in zip(a.named_parameters(), b.named_parameters()):
            self.assertTrue(torch.equal(x, y))
        with self.assertRaises(ValueError):
            load_config(overrides={"model": TINY, "optimizer": "muon", "pmuon": True, "soap_precondition": True})
        with self.assertRaises(ValueError):
            load_config(overrides={"model": TINY, "optimizer": "adamw", "pmuon": True})


if __name__ == "__main__":
    unittest.main()

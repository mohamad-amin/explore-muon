import importlib
import os
import sys
import unittest
from pathlib import Path

import torch

from .model import GPT, ModelConfig
from .muon import (JORDAN_QUINTIC, NS_COEFFICIENTS, deflated_entry, deflated_newton_schulz,
                   newton_schulz, ns_schedule, tracked_entry)
from .train import load_config, make_optimizer

# Local clone of github.com/ComputationalRobotics/Spectral-Deflation (commit 1eb15ce);
# the port-equivalence test skips when it is unavailable.
REFERENCE = os.environ.get("DEFLATION_REFERENCE")
TINY = {"vocab_size": 64, "n_layer": 2, "n_embd": 16, "n_head": 2, "seq_len": 8}


def previous_newton_schulz(momentum):
    """The function body before NS schedules were introduced (verbatim)."""
    x = momentum.float()
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    if x.is_cuda:
        x = x.bfloat16()
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = x.mT
    for a, b, c in NS_COEFFICIENTS:
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.mT if transpose else x).float()


def spiky(batch, n, m, head=(40.0, 12.0), seed=0):
    generator = torch.Generator().manual_seed(seed)
    u = torch.linalg.qr(torch.randn(batch, n, m, generator=generator)).Q
    v = torch.linalg.qr(torch.randn(batch, m, m, generator=generator)).Q
    s = torch.rand(batch, m, generator=generator) * 0.5 + 0.1
    s[:, :len(head)] = torch.tensor(head)
    s = s.sort(dim=1, descending=True).values
    return (u * s[:, None, :]) @ v.mT, s


class DeflationTest(unittest.TestCase):
    def test_default_newton_schulz_is_unchanged(self):
        torch.manual_seed(0)
        for shape in ((3, 24, 16), (3, 16, 24), (2, 16, 16)):
            x = torch.randn(*shape)
            self.assertTrue(torch.equal(newton_schulz(x), previous_newton_schulz(x)))

    def test_schedules_and_validation(self):
        self.assertEqual(ns_schedule(load_config()), NS_COEFFICIENTS)
        config = load_config(overrides={"optimizer": "muon", "ns_polynomial": "jordan", "ns_steps": 3})
        self.assertEqual(ns_schedule(config), (JORDAN_QUINTIC,) * 3)
        with self.assertRaises(ValueError):
            load_config(overrides={"ns_steps": 3})
        with self.assertRaises(ValueError):
            load_config(overrides={"deflation": True})
        with self.assertRaises(ValueError):
            load_config(overrides={"optimizer": "muon", "deflation": True, "deflation_pad": 0.9})

    def test_head_removed_from_divisor_and_restored(self):
        a, s = spiky(1, 64, 40)
        entry, k = deflated_entry(a, generator=torch.Generator().manual_seed(1), window=0.1)
        self.assertEqual(k.tolist(), [2])
        observed = torch.linalg.svdvals(entry)[0]
        rest = s[0, 2:] / s[0, 2:].norm()
        torch.testing.assert_close(observed[:2], torch.full((2,), 1 / 1.01), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(observed[2:], rest, rtol=1e-4, atol=1e-5)
        # Deflation only changes singular values: the (unique) polar factor is preserved.
        def polar(x):
            u, _, vh = torch.linalg.svd(x, full_matrices=False)
            return u @ vh
        torch.testing.assert_close(polar(entry), polar(a), rtol=0, atol=1e-4)

    def test_gate_declines_flat_spectra_and_wide_inputs_transpose(self):
        a = torch.randn(2, 64, 40, generator=torch.Generator().manual_seed(2))
        entry, k = deflated_entry(a, generator=torch.Generator().manual_seed(3), window=0.1)
        self.assertEqual(k.tolist(), [0, 0])
        torch.testing.assert_close(entry, a / (a.norm(dim=(-2, -1), keepdim=True) + 1e-7))
        b, _ = spiky(2, 64, 40, seed=4)
        omega = torch.randn(2, 40, 5, generator=torch.Generator().manual_seed(5))
        tall, k_tall = deflated_entry(b, omega=omega, window=0.1)
        wide, k_wide = deflated_entry(b.mT.contiguous(), omega=omega, window=0.1)
        torch.testing.assert_close(wide, tall.mT)
        self.assertTrue(torch.equal(k_tall, k_wide))

    @unittest.skipUnless(REFERENCE and Path(REFERENCE).exists(), "reference clone unavailable")
    def test_port_matches_reference_code(self):
        os.environ["DEFMUON_COMPILE_DEFL_ITER"] = "0"
        sys.path.insert(0, REFERENCE)
        try:
            reference = importlib.import_module("defmuon.optim.deflation_batched")
        finally:
            sys.path.remove(REFERENCE)
        m = 512
        for seed, head in ((6, (40.0, 12.0, 6.0)), (7, (0.6,))):
            a, _ = spiky(3, 600, m, head=head, seed=seed)
            omega = torch.randn(3, m, 26, generator=torch.Generator().manual_seed(seed))
            ours, k = deflated_entry(a, omega=omega)
            theirs, k_ref, _ = reference.batched_rmfro_clip(a, Omega=omega)
            torch.testing.assert_close(ours, theirs, rtol=0, atol=1e-7)
            self.assertTrue(torch.equal(k, k_ref))

    def test_deflated_muon_step_is_deterministic(self):
        overrides = {"model": TINY, "optimizer": "muon", "precision": "fp32", "compile": False,
                     "ns_polynomial": "jordan", "ns_steps": 3, "deflation": True, "deflation_window": 0.2}
        results = []
        for _ in range(2):
            torch.manual_seed(8)
            config = load_config(overrides=overrides)
            model = GPT(ModelConfig(**config["model"]))
            optimizer, _ = make_optimizer(model, config, torch.device("cpu"))
            tokens = torch.randint(0, 64, (4, 9))
            model(tokens[:, :-1], tokens[:, 1:]).backward()
            before = model.blocks[0].attn.q.weight.detach().clone()
            optimizer.step()
            self.assertFalse(torch.equal(before, model.blocks[0].attn.q.weight))
            self.assertEqual(int(optimizer.deflation_counts[0]), 12)
            results.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
        self.assertTrue(torch.equal(results[0], results[1]))

    def test_deflated_ns_iterates_schedule(self):
        a, _ = spiky(2, 64, 40, seed=9)
        omega = torch.randn(2, 40, 5, generator=torch.Generator().manual_seed(10))
        entry, _ = deflated_entry(a, omega=omega, window=0.1)
        x = entry
        for c1, c2, c3 in (JORDAN_QUINTIC,) * 3:
            g = x.mT @ x
            x = c1 * x + x @ (c2 * g + c3 * g @ g)
        generator = torch.Generator().manual_seed(11)
        direction, _ = deflated_newton_schulz(a, (JORDAN_QUINTIC,) * 3, generator=generator, window=0.1)
        # Head directions (singular value 1/1.01 at entry) are already converged.
        self.assertTrue(bool((torch.linalg.svdvals(direction)[:, :2] > 0.9).all()))
        torch.testing.assert_close(torch.linalg.svdvals(direction), torch.linalg.svdvals(x),
                                   rtol=2e-3, atol=2e-3)


class TrackedDeflationTest(unittest.TestCase):
    def test_restore_and_drop_with_exact_vector(self):
        a, s = spiky(2, 64, 40)
        v1 = torch.linalg.svd(a, full_matrices=False).Vh[:, 0]
        entry, v, energy, _ = tracked_entry(a, v1, head_weight=1.0)
        observed = torch.linalg.svdvals(entry)
        rest = s[:, 1:] / s[:, 1:].norm(dim=1, keepdim=True)
        torch.testing.assert_close(observed[:, 0], torch.full((2,), 1 / 1.01), rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(observed[:, 1:], rest, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(energy, s[:, 0] ** 2 / (s ** 2).sum(dim=1), rtol=1e-5, atol=1e-6)
        dropped, _, _, _ = tracked_entry(a, v1, head_weight=0.0)
        observed = torch.linalg.svdvals(dropped)
        torch.testing.assert_close(observed[:, :-1], rest, rtol=1e-4, atol=1e-5)
        self.assertLess(float(observed[:, -1].max()), 1e-4)

    def test_warm_start_converges_and_wide_inputs_transpose(self):
        a, _ = spiky(1, 64, 40, head=(10.0, 4.0), seed=12)
        v = torch.nn.functional.normalize(torch.randn(1, 40, generator=torch.Generator().manual_seed(13)), dim=-1)
        for _ in range(8):
            _, v, _, _ = tracked_entry(a, v)
        v1 = torch.linalg.svd(a, full_matrices=False).Vh[:, 0]
        self.assertGreater(float((v * v1).sum().abs()), 1 - 1e-6)
        tall, _, _, _ = tracked_entry(a, v)
        wide, _, _, _ = tracked_entry(a.mT.contiguous(), v)
        torch.testing.assert_close(wide, tall.mT)

    def test_track1_step_state_is_deterministic(self):
        overrides = {"model": TINY, "optimizer": "muon", "precision": "fp32", "compile": False,
                     "deflation": True, "deflation_mode": "track1", "deflation_head_weight": 0.0}
        results = []
        for _ in range(2):
            torch.manual_seed(14)
            config = load_config(overrides=overrides)
            model = GPT(ModelConfig(**config["model"]))
            optimizer, _ = make_optimizer(model, config, torch.device("cpu"))
            for _ in range(2):
                tokens = torch.randint(0, 64, (4, 9))
                model(tokens[:, :-1], tokens[:, 1:]).backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            heads = [optimizer.state[p]["head_v"] for p in optimizer.param_groups[0]["params"]]
            self.assertEqual(len(heads), 12)
            for vector in heads:
                self.assertAlmostEqual(float(vector.norm()), 1.0, places=5)
            self.assertEqual(int(optimizer.deflation_counts[0]), 24)
            self.assertGreater(float(optimizer.deflation_energy), 0.0)
            results.append(torch.cat([p.detach().flatten() for p in model.parameters()] + heads))
        self.assertTrue(torch.equal(results[0], results[1]))


if __name__ == "__main__":
    unittest.main()

import math
import unittest

import torch

from .model import GPT, ModelConfig, StatLinear
from .muon import whitening_factors
from .train import load_config, make_optimizer, model_hash

TINY = {"vocab_size": 64, "n_layer": 2, "n_embd": 16, "n_head": 2, "seq_len": 8,
        "bias": False, "norm": "rmsnorm", "qk_norm": True}


def one_step(overrides, seed=0, steps=1):
    torch.manual_seed(seed)
    config = load_config(overrides={"model": TINY, "optimizer": "muon", "precision": "fp32",
                                    "compile": False, **overrides})
    model = GPT(ModelConfig(**config["model"]))
    optimizer, _ = make_optimizer(model, config, torch.device("cpu"))
    generator = torch.Generator().manual_seed(seed + 1)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for _ in range(steps):
        tokens = torch.randint(0, 64, (4, 9), generator=generator)
        model(tokens[:, :-1], tokens[:, 1:]).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return model, optimizer, before


class InputStatsTest(unittest.TestCase):
    def test_statistics_exclude_position0_and_only_update_in_training(self):
        layer = StatLinear(6, 3, bias=False)
        x = torch.randn(4, 5, 6)
        layer(x)
        weight = layer.input_weight
        torch.testing.assert_close(layer.input_mean / weight, x[:, 1:].mean(dim=(0, 1)))
        torch.testing.assert_close(layer.input_sq / weight, x[:, 1:].pow(2).sum(-1).mean())
        with torch.no_grad():
            layer(torch.randn(4, 5, 6))
        layer.eval()
        layer(torch.randn(4, 5, 6))
        self.assertAlmostEqual(float(layer.input_weight), float(weight))

    def test_tracking_leaves_weights_and_state_dict_unchanged(self):
        torch.manual_seed(3)
        plain = GPT(ModelConfig(**TINY))
        torch.manual_seed(3)
        tracked = GPT(ModelConfig(**TINY, track_input_stats=True))
        self.assertEqual(set(plain.state_dict()), set(tracked.state_dict()))
        self.assertEqual(model_hash(plain), model_hash(tracked))

    def test_beta_formula(self):
        layer = StatLinear(4, 2, bias=False)
        mean = torch.tensor([3.0, 0.0, 4.0, 0.0])            # ||xbar||^2 = 25
        layer.input_weight.fill_(0.5)
        layer.input_mean.copy_(0.5 * mean)
        layer.input_sq.fill_(0.5 * (25.0 + 4 * 2.0))          # c0 = 2 per dimension
        v, beta = whitening_factors(layer)
        torch.testing.assert_close(v, mean / 5.0)
        self.assertAlmostEqual(float(beta), math.sqrt(2.0 / 27.0), places=6)
        _, fixed = whitening_factors(layer, 0.25)
        self.assertAlmostEqual(float(fixed), 0.25)
        _, quarter = whitening_factors(layer, power=0.25)
        self.assertAlmostEqual(float(quarter), (2.0 / 27.0) ** 0.25, places=6)


class WhitenedMuonTest(unittest.TestCase):
    def test_beta_one_matches_plain_muon(self):
        plain, _, _ = one_step({"model": {**TINY, "track_input_stats": True}})
        whitened, _, _ = one_step({"model": {**TINY, "track_input_stats": True},
                                   "mean_whitening": True, "mean_whitening_beta": 1.0})
        for (name, a), (_, b) in zip(plain.named_parameters(), whitened.named_parameters()):
            torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=name)

    def test_beta_zero_removes_update_along_mean_input(self):
        model, optimizer, before = one_step({"model": {**TINY, "track_input_stats": True},
                                             "mean_whitening": True, "mean_whitening_beta": 0.0})
        layer = model.blocks[1].attn.q
        v, _ = whitening_factors(layer)
        name = "blocks.1.attn.q.weight"
        group = optimizer.param_groups[0]
        decayed = before[name] * (1 - group["lr"] * group["weight_decay"])   # decoupled weight decay
        update = model.state_dict()[name] - decayed
        self.assertLess(float((update @ v).norm()), 1e-5 * float(update.norm()) + 1e-9)
        self.assertEqual(float(optimizer.whitening_sums[0]), 12.0)

    def test_automatic_beta_is_small_for_shared_input_mean(self):
        _, optimizer, _ = one_step({"model": {**TINY, "track_input_stats": True},
                                    "mean_whitening": True}, steps=2)
        mean_beta = float(optimizer.whitening_sums[1] / optimizer.whitening_sums[0])
        self.assertTrue(0.0 < mean_beta < 1.0)

    def test_validation_and_nesterov(self):
        with self.assertRaises(ValueError):
            load_config(overrides={"model": TINY, "optimizer": "muon", "mean_whitening": True})
        plain, _, _ = one_step({})
        nesterov, _, _ = one_step({"muon_nesterov": True})
        again, _, _ = one_step({"muon_nesterov": True})
        q = "blocks.0.attn.q.weight"
        self.assertFalse(torch.equal(plain.state_dict()[q], nesterov.state_dict()[q]))
        self.assertTrue(torch.equal(nesterov.state_dict()[q], again.state_dict()[q]))

    def test_head_whitening_keeps_weights_and_takes_whitened_adam_step(self):
        from .model import StatLinear
        with self.assertRaises(ValueError):
            load_config(overrides={"model": TINY, "optimizer": "muon", "head_whitening_alpha": 0.5})
        torch.manual_seed(0)
        plain = GPT(ModelConfig(**TINY))
        torch.manual_seed(0)
        tracked = GPT(ModelConfig(**{**TINY, "track_head_cov": True}))
        self.assertIsInstance(tracked.head, StatLinear)
        self.assertEqual(model_hash(plain), model_hash(tracked))      # same parameters and initialization
        torch.manual_seed(0)
        config = load_config(overrides={"model": {**TINY, "track_head_cov": True}, "optimizer": "muon",
                                        "precision": "fp32", "compile": False, "head_whitening_alpha": 0.5})
        model = GPT(ModelConfig(**config["model"]))
        optimizer, _ = make_optimizer(model, config, torch.device("cpu"))
        head = model.head
        basis = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64))[0]
        cov = (basis * torch.logspace(-2, 1, 16, dtype=torch.float64)) @ basis.T
        head.input_cov.copy_(cov.float())
        head.input_cov_weight.fill_(1.0)
        grads = [torch.randn_like(head.weight) for _ in range(2)]
        before = head.weight.detach().clone()
        for g in grads[:1]:
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            head.weight.grad = g.clone()
            optimizer.step()
        values, vectors = torch.linalg.eigh(cov)
        unit = values / values.mean()
        root = ((vectors * (unit + config["data_norm_damping"]).pow(-0.5)) @ vectors.T).float()
        beta1, beta2 = config["betas"]
        m = (1 - beta1) * grads[0]
        v = (1 - beta2) * (grads[0] @ root) ** 2
        step = ((m / (1 - beta1)) @ root) / ((v / (1 - beta2)).sqrt() + config["epsilon"])
        update = step @ root
        update = update * (step.norm() / update.norm())
        lr = optimizer.param_groups[1]["lr"]
        expected = before * (1 - lr * config["weight_decay"]) - lr * update
        self.assertTrue(torch.allclose(head.weight.detach(), expected, atol=1e-6))
        # the shared AdamW did not also move the head
        self.assertNotIn("exp_avg", optimizer.state[head.weight])
        # centered: the root whitens the covariance about the EMA mean
        with self.assertRaises(ValueError):
            load_config(overrides={"model": {**TINY, "track_head_cov": True}, "optimizer": "muon", "head_whitening_center": True})
        centered = load_config(overrides={"model": {**TINY, "track_head_cov": True}, "optimizer": "muon", "precision": "fp32",
                                          "compile": False, "head_whitening_alpha": 0.5, "head_whitening_center": True})
        model2 = GPT(ModelConfig(**centered["model"]))
        optimizer2, _ = make_optimizer(model2, centered, torch.device("cpu"))
        mean = torch.randn(16, dtype=torch.float64)
        model2.head.input_cov.copy_((cov + torch.outer(mean, mean)).float())
        model2.head.input_cov_mean.copy_(mean.float())
        model2.head.input_cov_weight.fill_(1.0)
        for p in model2.parameters():
            p.grad = torch.zeros_like(p)
        model2.head.weight.grad = grads[0].clone()
        optimizer2.step()
        self.assertTrue(torch.allclose(optimizer2.head_white["root"], root, atol=1e-4))
        # norm "none": the plain reparametrized Adam step, without rescaling
        with self.assertRaises(ValueError):
            load_config(overrides={"model": {**TINY, "track_head_cov": True}, "optimizer": "muon", "head_whitening_alpha": 0.5,
                                   "head_whitening_norm": "half"})
        plain_cfg = load_config(overrides={"model": {**TINY, "track_head_cov": True}, "optimizer": "muon", "precision": "fp32",
                                           "compile": False, "head_whitening_alpha": 0.5, "head_whitening_norm": "none"})
        model3 = GPT(ModelConfig(**plain_cfg["model"]))
        optimizer3, _ = make_optimizer(model3, plain_cfg, torch.device("cpu"))
        model3.head.input_cov.copy_(cov.float())
        model3.head.input_cov_weight.fill_(1.0)
        start = model3.head.weight.detach().clone()
        for p in model3.parameters():
            p.grad = torch.zeros_like(p)
        model3.head.weight.grad = grads[0].clone()
        optimizer3.step()
        lr3 = optimizer3.param_groups[1]["lr"]
        expected3 = start * (1 - lr3 * plain_cfg["weight_decay"]) - lr3 * (step @ root)
        self.assertTrue(torch.allclose(model3.head.weight.detach(), expected3, atol=1e-6))

    def test_top_only_root_suppresses_above_mean_directions_only(self):
        from .muon import data_norm_root
        with self.assertRaises(ValueError):
            load_config(overrides={"model": TINY, "optimizer": "muon", "data_norm_top_only": True})
        torch.manual_seed(0)
        layer = StatLinear(16, 8, bias=False, cov=True)
        basis = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64))[0]
        spectrum = torch.logspace(-2, 1.5, 16, dtype=torch.float64)
        layer.input_cov.copy_(((basis * spectrum) @ basis.T).float())
        layer.input_cov_weight.fill_(1.0)
        full = data_norm_root(layer, 0.5, 1e-3).double()
        top = data_norm_root(layer, 0.5, 1e-3, top_only=True).double()
        unit = spectrum / spectrum.mean()
        factors = (unit + 1e-3).pow(-0.5)
        expected = (basis * factors.clamp_max(1.0)) @ basis.T
        self.assertTrue(torch.allclose(top, expected, atol=1e-4))
        eig_top = torch.linalg.eigvalsh(top)
        self.assertLessEqual(float(eig_top.max()), 1.0 + 1e-4)               # nothing is amplified
        self.assertTrue(torch.allclose(torch.linalg.eigvalsh(full).max(), factors.max(), atol=1e-3))   # PD amplifies
        # above-mean directions match the full root exactly
        top_dir = basis[:, -1]
        self.assertAlmostEqual(float(top_dir @ top @ top_dir), float(top_dir @ full @ top_dir), places=4)
        config = load_config(overrides={"model": {**TINY, "track_input_stats": True, "track_input_cov": True},
                                        "optimizer": "muon", "precision": "fp32", "compile": False,
                                        "data_norm_alpha": 0.5, "data_norm_top_only": True})
        self.assertTrue(config["data_norm_top_only"])

    def test_momentum_warmup_schedule(self):
        from .train import muon_momentum
        with self.assertRaises(ValueError):
            load_config(overrides={"model": TINY, "optimizer": "muon", "muon_momentum_start": 0.8})
        with self.assertRaises(ValueError):
            load_config(overrides={"model": TINY, "optimizer": "muon", "muon_momentum_warmup": 0.5})
        plain = load_config(overrides={"model": TINY, "optimizer": "muon", "muon_momentum": 0.9})
        self.assertEqual(muon_momentum(plain, 500, 1000), 0.9)
        warm = load_config(overrides={"model": TINY, "optimizer": "muon", "muon_momentum": 0.9,
                                      "muon_momentum_start": 0.8, "muon_momentum_warmup": 0.5})
        self.assertAlmostEqual(muon_momentum(warm, 0, 1000), 0.8)
        self.assertAlmostEqual(muon_momentum(warm, 250, 1000), 0.85)
        self.assertAlmostEqual(muon_momentum(warm, 500, 1000), 0.9)
        self.assertAlmostEqual(muon_momentum(warm, 900, 1000), 0.9)

    def test_two_tap_prefilter_averages_consecutive_gradients(self):
        with self.assertRaises(ValueError):
            load_config(overrides={"model": TINY, "optimizer": "muon", "muon_prefilter": "two_tap", "muon_nesterov": True})
        with self.assertRaises(ValueError):
            load_config(overrides={"model": TINY, "optimizer": "muon", "muon_prefilter": "three_tap"})
        torch.manual_seed(0)
        config = load_config(overrides={"model": TINY, "optimizer": "muon", "precision": "fp32", "compile": False,
                                        "muon_prefilter": "two_tap", "muon_momentum": 0.9})
        model = GPT(ModelConfig(**config["model"]))
        optimizer, _ = make_optimizer(model, config, torch.device("cpu"))
        weight = model.blocks[0].attn.q.weight
        grads = [torch.randn_like(weight) for _ in range(3)]
        for g in grads:     # hand-set gradients: every parameter needs one for a step
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            weight.grad = g.clone()
            optimizer.step()
        expected = 0.9 * (0.9 * grads[0] + 0.5 * (grads[0] + grads[1])) + 0.5 * (grads[1] + grads[2])
        self.assertTrue(torch.allclose(optimizer.state[weight]["momentum_buffer"], expected, atol=1e-6))
        self.assertTrue(torch.equal(optimizer.state[weight]["previous_grad"], grads[2]))
        q = "blocks.0.attn.q.weight"
        plain, _, _ = one_step({}, steps=2)
        filtered, _, _ = one_step({"muon_prefilter": "two_tap"}, steps=2)
        first, _, _ = one_step({"muon_prefilter": "two_tap"}, steps=1)
        plain_first, _, _ = one_step({}, steps=1)
        self.assertTrue(torch.equal(first.state_dict()[q], plain_first.state_dict()[q]))    # step 1 is unchanged
        self.assertFalse(torch.equal(plain.state_dict()[q], filtered.state_dict()[q]))

    def test_split_momentum_mixes_a_short_ema_in_the_top_input_subspace(self):
        from .muon import data_norm_root
        pd = {"model": {**TINY, "track_input_stats": True, "track_input_cov": True}, "data_norm_alpha": 0.5,
              "muon_momentum": 0.9}
        with self.assertRaises(ValueError):     # needs data norm
            load_config(overrides={"model": TINY, "optimizer": "muon", "momentum_split_top": 0.8})
        with self.assertRaises(ValueError):
            load_config(overrides={"optimizer": "muon", **pd, "momentum_split_top": 1.0})
        with self.assertRaises(ValueError):
            load_config(overrides={"optimizer": "muon", **pd, "momentum_split_top": 0.8, "muon_nesterov": True})
        # the projector spans the above-mean input directions
        torch.manual_seed(0)
        layer = StatLinear(16, 8, bias=False, cov=True)
        basis = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64))[0]
        spectrum = torch.logspace(-2, 1.5, 16, dtype=torch.float64)
        layer.input_cov.copy_(((basis * spectrum) @ basis.T).float())
        layer.input_cov_weight.fill_(1.0)
        root, _, top = data_norm_root(layer, 0.5, 1e-3, with_cov=True, with_top=True)
        above = int((spectrum / spectrum.mean() >= 1).sum())
        self.assertTrue(torch.allclose(top @ top, top, atol=1e-5))
        self.assertAlmostEqual(float(top.trace()), above, places=4)
        self.assertTrue(torch.allclose(root, data_norm_root(layer, 0.5, 1e-3)))
        # with beta_top = beta the preconditioner's input is (1 - beta) M, the same update as plain PD
        q = "blocks.0.attn.q.weight"
        plain, _, _ = one_step(pd, steps=3)
        same, _, _ = one_step({**pd, "momentum_split_top": 0.9}, steps=3)
        split, optimizer, _ = one_step({**pd, "momentum_split_top": 0.5}, steps=3)
        self.assertTrue(torch.allclose(plain.state_dict()[q], same.state_dict()[q], atol=1e-5))
        self.assertFalse(torch.allclose(plain.state_dict()[q], split.state_dict()[q], atol=1e-5))
        # the short buffer is an EMA of the same gradients with beta_top
        config = load_config(overrides={"optimizer": "muon", "precision": "fp32", "compile": False, **pd,
                                        "momentum_split_top": 0.5})
        model = GPT(ModelConfig(**config["model"]))
        optimizer, _ = make_optimizer(model, config, torch.device("cpu"))
        weight = model.blocks[0].attn.q.weight
        grads = [torch.randn_like(weight) for _ in range(3)]
        for g in grads:
            for p in model.parameters():
                p.grad = torch.zeros_like(p)
            weight.grad = g.clone()
            optimizer.step()
        self.assertTrue(torch.allclose(optimizer.state[weight]["momentum_short"],
                                       0.25 * grads[0] + 0.5 * grads[1] + grads[2], atol=1e-6))
        self.assertTrue(torch.allclose(optimizer.state[weight]["momentum_buffer"],
                                       0.81 * grads[0] + 0.9 * grads[1] + grads[2], atol=1e-6))

    def test_snr_gated_root_interpolates_between_pd_top_and_pd(self):
        from .muon import data_norm_root, snr_gated_root
        pd = {"model": {**TINY, "track_input_stats": True, "track_input_cov": True}, "data_norm_alpha": 0.5}
        with self.assertRaises(ValueError):     # needs data norm, and not top-only
            load_config(overrides={"model": TINY, "optimizer": "muon", "data_norm_snr_gate": True})
        with self.assertRaises(ValueError):
            load_config(overrides={"optimizer": "muon", **pd, "data_norm_snr_gate": True, "data_norm_top_only": True})
        with self.assertRaises(ValueError):
            load_config(overrides={"optimizer": "muon", **pd, "data_norm_snr_gate": True, "momentum_split_top": 0.8})
        self.assertTrue(load_config(overrides={"optimizer": "muon", **pd, "data_norm_snr_gate": True})["data_norm_snr_gate"])
        torch.manual_seed(0)
        layer = StatLinear(16, 8, bias=False, cov=True)
        basis = torch.linalg.qr(torch.randn(16, 16, dtype=torch.float64))[0]
        spectrum = torch.logspace(-3, 1.5, 16, dtype=torch.float64)
        layer.input_cov.copy_(((basis * spectrum) @ basis.T).float())
        layer.input_cov_weight.fill_(1.0)
        values, vectors = torch.linalg.eigh(layer.input_cov.double())
        unit = (values.clamp_min(0) / values.clamp_min(0).mean())
        full = data_norm_root(layer, 0.5, 1e-3)
        top = data_norm_root(layer, 0.5, 1e-3, top_only=True)
        self.assertTrue(torch.allclose(snr_gated_root(vectors, unit, 0.5, 1e-3, torch.ones(16)), full, atol=1e-4))
        self.assertTrue(torch.allclose(snr_gated_root(vectors, unit, 0.5, 1e-3, torch.zeros(16)), top, atol=1e-4))
        half = snr_gated_root(vectors, unit, 0.5, 1e-3, torch.full((16,), 0.5)).double()
        factors = (unit + 1e-3).pow(-0.5)
        expected = (vectors * (factors.clamp_max(1) + 0.5 * (factors - 1).clamp_min(0))) @ vectors.T
        self.assertTrue(torch.allclose(half, expected, atol=1e-4))
        top_dir = vectors[:, -1]   # above-mean directions are untouched by the gate
        self.assertAlmostEqual(float(top_dir @ half @ top_dir), float(top_dir @ full.double() @ top_dir), places=4)


if __name__ == "__main__":
    unittest.main()


class SoapPreconditionTest(unittest.TestCase):
    def test_soap_matches_reference_formula_and_whitens_dominant_column(self):
        from .muon import new_soap_state, soap_precondition, soap_update_statistics
        torch.manual_seed(5)
        rows, cols = 12, 10
        mean_input = torch.zeros(cols); mean_input[0] = 1.0
        soap = new_soap_state(torch.zeros(rows, cols))
        grads = [torch.outer(torch.randn(rows), 5 * mean_input) + 0.1 * torch.randn(rows, cols) for _ in range(6)]
        first = soap_precondition(grads[0], soap, 0.9, 0.5)
        self.assertTrue(torch.equal(first, grads[0]))            # no statistics yet
        for g in grads:
            soap_update_statistics(g, soap, 0.9)
        q_col = soap["q_col"]
        self.assertGreater(abs(float(q_col[:, 0] @ mean_input)), 0.99)   # top right-eigenvector = mean input
        momentum = grads[-1]
        before = soap["exp_avg_sq"].clone()
        out = soap_precondition(momentum, soap, 0.9, 0.5)
        projected = soap["q_row"].T @ momentum @ q_col
        expected_sq = 0.9 * before + 0.1 * projected.square()
        expected = soap["q_row"] @ (projected / expected_sq.clamp_min(1e-16).sqrt()) @ q_col.T
        expected = expected * (momentum.norm() / expected.norm())
        torch.testing.assert_close(out, expected)
        share = lambda m: float((m @ mean_input).norm() ** 2 / m.norm() ** 2)
        self.assertLess(share(out), share(momentum))            # the mean-input column is damped

    def test_soap_muon_step_runs_and_is_deterministic(self):
        results = []
        for _ in range(2):
            model, optimizer, _ = one_step({"soap_precondition": True}, steps=3)
            self.assertEqual(len(optimizer.soap["states"]), 12)
            self.assertNotIn("q_row", optimizer.state_dict()["state"][0])
            results.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
        self.assertTrue(torch.equal(results[0], results[1]))
        with self.assertRaises(ValueError):
            load_config(overrides={"optimizer": "muon", "soap_precondition": True, "deflation": True})


class BiasAdamTest(unittest.TestCase):
    def test_implicit_bias_moves_at_aux_rate_and_muon_part_excludes_mean(self):
        overrides = {"model": {**TINY, "track_input_stats": True}, "mean_whitening": True,
                     "mean_whitening_beta": 0.0, "mean_bias_adam": True}
        model, optimizer, before = one_step(overrides)
        zero, _, _ = one_step({**overrides, "mean_bias_adam": False})
        name = "blocks.1.attn.q.weight"
        layer = model.blocks[1].attn.q
        v, _ = whitening_factors(layer)
        xbar_norm = float((layer.input_mean / layer.input_weight).norm())
        group = optimizer.param_groups[0]
        update = model.state_dict()[name] - before[name] * (1 - group["lr"] * group["weight_decay"])
        # First Adam step is m/(sqrt(s)+eps) = g/(|g|+eps): each bias coordinate moves by aux_lr.
        bias_move = (update @ v) * xbar_norm
        aux_lr = optimizer.param_groups[1]["lr"]
        torch.testing.assert_close(bias_move.abs(), torch.full_like(bias_move, aux_lr), rtol=1e-3, atol=1e-7)
        # The rest of the step is exactly beta-0 whitened Muon: the two runs differ only along v.
        difference = model.state_dict()[name] - zero.state_dict()[name]
        torch.testing.assert_close(difference, torch.outer(difference @ v, v), rtol=0, atol=1e-7)
        self.assertGreater(float(difference.norm()), 0.0)
        self.assertEqual(len(optimizer.bias_adam["states"]), 12)
        self.assertEqual(float(optimizer.bias_sums[0]), 12.0)

    def test_bias_adam_validation(self):
        base = {"model": {**TINY, "track_input_stats": True}, "optimizer": "muon", "mean_bias_adam": True}
        with self.assertRaises(ValueError):
            load_config(overrides=base)
        with self.assertRaises(ValueError):
            load_config(overrides={**base, "mean_whitening": True})          # automatic beta
        load_config(overrides={**base, "mean_whitening": True, "mean_whitening_beta": 0.0})

    def test_basis_ablations_use_identity_on_unrotated_sides(self):
        from .muon import new_soap_state, soap_precondition, soap_update_statistics
        torch.manual_seed(6)
        rows, cols = 9, 7
        grads = [torch.randn(rows, cols) for _ in range(4)]
        momentum = torch.randn(rows, cols)
        for basis in ("none", "right", "left"):
            soap = new_soap_state(momentum, basis)
            self.assertTrue(torch.equal(soap_precondition(momentum, soap, 0.9, 0.5), momentum))
            for g in grads:
                soap_update_statistics(g, soap, 0.9)
            q_row = soap["q_row"] if basis == "left" else torch.eye(rows)
            q_col = soap["q_col"] if basis == "right" else torch.eye(cols)
            self.assertEqual(soap["q_row"] is None, basis != "left")
            self.assertEqual(soap["q_col"] is None, basis != "right")
            before = soap["exp_avg_sq"].clone()
            out = soap_precondition(momentum, soap, 0.9, 0.5)
            projected = q_row.T @ momentum @ q_col
            expected_sq = 0.9 * before + 0.1 * projected.square()
            expected = q_row @ (projected / expected_sq.clamp_min(1e-16).sqrt()) @ q_col.T
            torch.testing.assert_close(out, expected * (momentum.norm() / expected.norm()))
        results = []
        for _ in range(2):
            model, optimizer, _ = one_step({"soap_precondition": True, "soap_basis": "right"}, steps=3)
            self.assertNotIn("row_gg", next(iter(optimizer.soap["states"].values())))
            results.append(torch.cat([p.detach().flatten() for p in model.parameters()]))
        self.assertTrue(torch.equal(results[0], results[1]))
        with self.assertRaises(ValueError):
            load_config(overrides={"optimizer": "muon", "soap_precondition": True, "soap_basis": "diag"})

    def test_gradient_second_moment_tracks_projected_gradient(self):
        from .muon import new_soap_state, soap_precondition, soap_update_statistics
        torch.manual_seed(7)
        rows, cols = 8, 6
        soap = new_soap_state(torch.zeros(rows, cols))
        for _ in range(3):
            soap_update_statistics(torch.randn(rows, cols), soap, 0.9)
        momentum, grad = torch.randn(rows, cols), torch.randn(rows, cols)
        before = soap["exp_avg_sq"].clone()
        out = soap_precondition(momentum, soap, 0.9, 0.5, grad=grad)
        q_row, q_col = soap["q_row"], soap["q_col"]
        expected_sq = 0.9 * before + 0.1 * (q_row.T @ grad @ q_col).square()
        torch.testing.assert_close(soap["exp_avg_sq"], expected_sq)
        expected = q_row @ ((q_row.T @ momentum @ q_col) / expected_sq.sqrt()) @ q_col.T
        torch.testing.assert_close(out, expected * (momentum.norm() / expected.norm()))
        model, optimizer, _ = one_step({"soap_precondition": True, "soap_second_moment": "gradient"}, steps=3)
        self.assertTrue(optimizer.soap["gradient_moment"])


class ExactPolarTest(unittest.TestCase):
    def test_exact_polar_has_unit_singular_values_and_runs(self):
        from .muon import newton_schulz, ns_schedule
        x = torch.randn(3, 24, 16) * torch.logspace(0, -4, 16)       # ill-conditioned
        polar = newton_schulz(x, "svd")
        torch.testing.assert_close(torch.linalg.svdvals(polar), torch.ones(3, 16), rtol=1e-4, atol=1e-4)
        u, _, vh = torch.linalg.svd(x, full_matrices=False)
        torch.testing.assert_close(polar, u @ vh)
        self.assertEqual(ns_schedule(load_config(overrides={"optimizer": "muon", "ns_polynomial": "svd"})), "svd")
        with self.assertRaises(ValueError):
            load_config(overrides={"optimizer": "muon", "ns_polynomial": "svd", "deflation": True})
        a, _, _ = one_step({"ns_polynomial": "svd"})
        b, _, _ = one_step({})
        q = "blocks.0.attn.q.weight"
        self.assertFalse(torch.equal(a.state_dict()[q], b.state_dict()[q]))


class DataNormTest(unittest.TestCase):
    def test_covariance_statistic(self):
        layer = StatLinear(6, 3, bias=False, cov=True, cov_stride=2)
        x = torch.randn(4, 9, 6)
        layer(x)
        rows = x[:, 1:][:, ::2].reshape(-1, 6)
        torch.testing.assert_close(layer.input_cov / layer.input_cov_weight, rows.T @ rows / rows.shape[0])
        with self.assertRaises(ValueError):
            ModelConfig(**TINY, track_input_cov=True)

    def test_root_and_steepest_descent_property(self):
        from .muon import data_norm_root, newton_schulz
        torch.manual_seed(9)
        d = 6
        layer = StatLinear(d, 4, bias=False, cov=True)
        basis = torch.linalg.qr(torch.randn(d, d)).Q
        spectrum = torch.tensor([30.0, 5.0, 2.0, 1.0, 0.5, 0.1])
        cov = (basis * spectrum) @ basis.T
        layer.input_cov.copy_(0.5 * cov)
        layer.input_cov_weight.fill_(0.5)
        root = data_norm_root(layer, 0.5, 1e-12)
        scaled = cov / spectrum.mean()
        torch.testing.assert_close(root @ scaled @ root, torch.eye(d), rtol=1e-4, atol=1e-4)
        # polar(G R) R maximizes <G, D> subject to ||D R^-1||_op <= 1.
        g = torch.randn(4, d)
        best = newton_schulz(g @ root, "svd") @ root
        value = float((g * best).sum())
        inverse = torch.linalg.inv(root)
        self.assertAlmostEqual(float(torch.linalg.matrix_norm(best @ inverse, ord=2)), 1.0, places=4)
        for _ in range(50):
            trial = newton_schulz(torch.randn(4, d), "svd") @ root
            self.assertLessEqual(float((g * trial).sum()), value + 1e-5)

    def test_step_is_rms_matched_and_small_alpha_is_muon(self):
        tracked = {**TINY, "track_input_stats": True, "track_input_cov": True}
        plain, optimizer_plain, before = one_step({"model": tracked})
        tiny_alpha, _, _ = one_step({"model": tracked, "data_norm_alpha": 1e-6})
        shaped, optimizer, _ = one_step({"model": tracked, "data_norm_alpha": 0.25})
        name = "blocks.1.mlp.up.weight"
        group = optimizer.param_groups[0]
        decayed = before[name] * (1 - group["lr"] * group["weight_decay"])
        for model in (plain, shaped):
            step = model.state_dict()[name] - decayed
            # Muon's polar has Frobenius norm sqrt(min(m, n)) times the shape scale, at rate lr.
            expected = group["lr"] * math.sqrt(16) * math.sqrt(64 / 16)
            self.assertAlmostEqual(float(step.norm()), expected, delta=0.02 * expected)
        torch.testing.assert_close(tiny_alpha.state_dict()[name], plain.state_dict()[name], rtol=1e-4, atol=1e-6)
        self.assertFalse(torch.allclose(shaped.state_dict()[name], plain.state_dict()[name]))
        self.assertEqual(float(optimizer.data_norm_sums[0]), 12.0)
        with self.assertRaises(ValueError):
            load_config(overrides={"model": {**TINY, "track_input_stats": True}, "optimizer": "muon",
                                   "data_norm_alpha": 0.25})


class SoapVariantTest(unittest.TestCase):
    def test_column_normalization_and_activation_basis(self):
        from .muon import new_soap_state, soap_precondition, soap_update_statistics
        torch.manual_seed(11)
        rows, cols = 7, 5
        soap = new_soap_state(torch.zeros(rows, cols), "right")
        for _ in range(3):
            soap_update_statistics(torch.randn(rows, cols), soap, 0.9)
        momentum = torch.randn(rows, cols)
        before = soap["exp_avg_sq"].clone()
        out = soap_precondition(momentum, soap, 0.9, 0.5, column=True)
        q_col = soap["q_col"]
        projected = momentum @ q_col
        second = (0.9 * before + 0.1 * projected.square()).mean(dim=0, keepdim=True)
        expected = (projected / second.sqrt()) @ q_col.T
        torch.testing.assert_close(out, expected * (momentum.norm() / expected.norm()))
        # right_act: the input basis follows a supplied activation second moment.
        act = new_soap_state(torch.zeros(rows, cols), "right_act")
        basis = torch.linalg.qr(torch.randn(cols, cols)).Q
        gram = (basis * torch.tensor([9.0, 4.0, 2.0, 1.0, 0.5])) @ basis.T
        soap_update_statistics(torch.randn(rows, cols), act, 0.9, col_gram=gram)
        self.assertGreater(abs(float(act["q_col"][:, 0] @ basis[:, 0])), 0.999)
        tracked = {**TINY, "track_input_stats": True, "track_input_cov": True}
        model, optimizer, _ = one_step({"model": tracked, "soap_precondition": True, "soap_basis": "right_act",
                                        "soap_norm": "column"}, steps=3)
        self.assertEqual(len(optimizer.soap["states"]), 12)
        with self.assertRaises(ValueError):
            load_config(overrides={"optimizer": "muon", "soap_precondition": True, "soap_basis": "right_act"})


class DataNormInsideTest(unittest.TestCase):
    def test_inside_only_whitening_is_polar_of_whitened_momentum(self):
        tracked = {**TINY, "track_input_stats": True, "track_input_cov": True}
        exact = {"model": tracked, "data_norm_alpha": 0.5, "ns_polynomial": "svd"}
        inside, optimizer, before = one_step({**exact, "data_norm_post": False})
        both, _, _ = one_step(exact)
        name = "blocks.0.attn.v.weight"
        self.assertFalse(torch.allclose(inside.state_dict()[name], both.state_dict()[name]))
        group = optimizer.param_groups[0]
        step = (before[name] * (1 - group["lr"] * group["weight_decay"]) - inside.state_dict()[name]) / group["lr"]
        # Without post-multiplication the step is the exact polar factor of M R (all singular values equal).
        singular = torch.linalg.svdvals(step)
        self.assertLess(float(singular.max() / singular.min()), 1.001)


class DataNormRowsTest(unittest.TestCase):
    def test_rows_are_equalized_before_the_polar(self):
        tracked = {**TINY, "track_input_stats": True, "track_input_cov": True}
        exact = {"model": tracked, "data_norm_alpha": 0.5, "data_norm_post": False, "ns_polynomial": "svd"}
        rows, optimizer, _ = one_step({**exact, "data_norm_rows": True})
        plain, _, _ = one_step(exact)
        name = "blocks.0.mlp.down.weight"
        self.assertFalse(torch.allclose(rows.state_dict()[name], plain.state_dict()[name]))
        self.assertEqual(len(optimizer.data_norm["row_sq"]), 12)
        state = next(iter(optimizer.data_norm["row_sq"].values()))
        self.assertEqual(state["t"], 1)


class DataNormCenterTest(unittest.TestCase):
    def test_centered_root_ignores_the_mean(self):
        from .muon import data_norm_root
        torch.manual_seed(12)
        layer = StatLinear(5, 3, bias=False, cov=True, cov_stride=1)
        x = torch.randn(64, 9, 5) + torch.tensor([4.0, 0.0, 0.0, 0.0, 0.0])
        layer(x)
        rows = x[:, 1:].reshape(-1, 5)
        weight = layer.input_cov_weight
        torch.testing.assert_close(layer.input_cov_mean / weight, rows.mean(0))
        centered = data_norm_root(layer, 0.5, 1e-9, center=True)
        uncentered = data_norm_root(layer, 0.5, 1e-9)
        cov = torch.cov(rows.T, correction=0).double()
        scaled = cov / torch.linalg.eigvalsh(cov).mean()
        torch.testing.assert_close((centered.double() @ scaled @ centered.double()), torch.eye(5, dtype=torch.float64),
                                   rtol=1e-3, atol=1e-3)
        mean_dir = torch.tensor([1.0, 0, 0, 0, 0])
        self.assertLess(float(mean_dir @ uncentered @ mean_dir), float(mean_dir @ centered @ mean_dir))


class Wave10VariantsTest(unittest.TestCase):
    tracked = {**TINY, "track_input_stats": True, "track_input_cov": True}

    def test_outside_only_and_magnitude_matched(self):
        exact = {"model": self.tracked, "data_norm_alpha": 0.25, "ns_polynomial": "svd"}
        plain, _, before = one_step({"model": self.tracked, "ns_polynomial": "svd"})
        outside, _, _ = one_step({**exact, "data_norm_pre": False})
        matched, optimizer, _ = one_step({**exact, "data_norm_mode": "magnitude_matched"})
        name = "blocks.1.attn.o.weight"
        group = optimizer.param_groups[0]
        step = lambda m: (before[name] * (1 - group["lr"] * group["weight_decay"]) - m.state_dict()[name]) / group["lr"]
        muon = step(plain)
        # Magnitude-matched Muon keeps Muon's direction (a positive multiple of it).
        ratio = step(matched).flatten() @ muon.flatten() / (muon.norm() * step(matched).norm())
        self.assertGreater(float(ratio), 0.9999)
        self.assertFalse(torch.allclose(step(matched), muon))
        # Outside-only differs from Muon and from the sandwich, and is RMS-matched.
        sandwich, _, _ = one_step(exact)
        self.assertFalse(torch.allclose(step(outside), step(sandwich)))
        self.assertAlmostEqual(float(step(outside).norm()), float(muon.norm()), delta=1e-3 * float(muon.norm()))

    def test_soap_in_whitened_coordinates_and_layer_selection(self):
        # Plumbing: with alpha -> 0, SOAP in whitened coordinates equals SOAP up to the data-norm branch's
        # per-matrix RMS rescaling, so each matrix's update keeps its direction. The standard basis and the
        # continuous NS map avoid eigenvector and exact-polar discontinuities under the 1e-7 perturbation.
        config = {"model": self.tracked, "soap_precondition": True, "soap_basis": "none"}
        tiny_alpha, _, before = one_step({**config, "data_norm_alpha": 1e-7}, steps=2)
        soap, _, _ = one_step(config, steps=2)
        for (n, a), (_, b) in zip(tiny_alpha.named_parameters(), soap.named_parameters()):
            if a.ndim == 2 and n.startswith("blocks."):
                da, db = (a - before[n]).flatten(), (b - before[n]).flatten()
                self.assertGreater(float(da @ db / (da.norm() * db.norm())), 0.999, msg=n)
        _, optimizer, _ = one_step({"soap_precondition": True, "soap_layers": "down"}, steps=2)
        self.assertEqual(len(optimizer.soap["states"]), 2)
        self.assertTrue(all(p.shape[0] < p.shape[1] for p in optimizer.soap["states"]))
        _, optimizer, _ = one_step({"soap_precondition": True, "soap_layers": "not_down"}, steps=2)
        self.assertEqual(len(optimizer.soap["states"]), 10)


class DataNormReferenceTest(unittest.TestCase):
    def test_reference_matches_optimizer_first_step(self):
        from .data_norm_muon import data_norm_direction
        tracked = {**TINY, "track_input_stats": True, "track_input_cov": True}
        model, optimizer, before = one_step({"model": tracked, "data_norm_alpha": 0.25})
        name = "blocks.1.mlp.up.weight"
        layer = model.blocks[1].mlp.up
        parameter = layer.weight
        group = optimizer.param_groups[0]
        step = (before[name] * (1 - group["lr"] * group["weight_decay"]) - parameter.detach()) / group["lr"]
        momentum = optimizer.state[parameter]["momentum_buffer"]
        cov = layer.input_cov / layer.input_cov_weight
        torch.testing.assert_close(step, data_norm_direction(momentum, cov, 0.25), rtol=1e-4, atol=1e-5)


class DataNormByKindTest(unittest.TestCase):
    def test_per_kind_alpha_changes_only_the_named_kinds(self):
        tracked = {**TINY, "track_input_stats": True, "track_input_cov": True}
        base = {"model": tracked, "data_norm_alpha": 0.25, "ns_polynomial": "svd"}
        uniform, _, _ = one_step(base)
        by_kind, optimizer, _ = one_step({**base, "data_norm_alpha_by_kind": {"up": 0.4, "q": 0.1}})
        state = uniform.state_dict(), by_kind.state_dict()
        for name in state[0]:
            if name.endswith(("mlp.up.weight", "attn.q.weight")):
                self.assertFalse(torch.allclose(state[0][name], state[1][name]), msg=name)
            elif name.endswith(("attn.k.weight", "attn.v.weight", "attn.o.weight", "mlp.down.weight")):
                torch.testing.assert_close(state[0][name], state[1][name], msg=name)
        alphas = sorted(set(optimizer.data_norm["by_kind"].values()))
        self.assertEqual(alphas, [0.1, 0.25, 0.4])
        with self.assertRaises(ValueError):
            load_config(overrides={"model": tracked, "optimizer": "muon", "data_norm_alpha": 0.25,
                                   "data_norm_alpha_by_kind": {"gate": 0.3}})


class DataNormDecayTest(unittest.TestCase):
    def test_geometry_decay_is_preconditioned_like_the_update(self):
        tracked = {**TINY, "track_input_stats": True, "track_input_cov": True}
        base = {"model": tracked, "data_norm_alpha": 0.25, "ns_polynomial": "svd", "weight_decay": 0.5}
        decoupled, _, before = one_step(base)
        for power in (2, 1):
            geometry, optimizer, _ = one_step({**base, "data_norm_decay": "geometry", "data_norm_decay_power": power})
            group = optimizer.param_groups[0]
            checked = 0
            for name, parameter in geometry.named_parameters():
                if parameter not in optimizer.data_norm["roots"]:
                    continue
                root = optimizer.data_norm["roots"][parameter][0]
                shrink = torch.linalg.matrix_power(root, power)
                shrink = shrink / shrink.diagonal().mean()
                # Same direction in both runs; only the decay differs: W R^p/mean eig(R^p) replaces W.
                expected = -group["lr"] * group["weight_decay"] * (before[name] @ shrink - before[name])
                torch.testing.assert_close(parameter.detach() - decoupled.state_dict()[name], expected,
                                           rtol=1e-4, atol=1e-6, msg=name)
                self.assertGreater(float(expected.norm()), 1e-5)
                checked += 1
            self.assertEqual(checked, 12)
        for bad in ({"data_norm_decay": "geometry"}, {"data_norm_alpha": 0.25, "data_norm_decay": "l2"},
                    {"data_norm_alpha": 0.25, "data_norm_decay": "geometry", "data_norm_decay_power": 3}):
            with self.assertRaises(ValueError):
                load_config(overrides={"model": tracked, "optimizer": "muon", **bad})


class DisplacementTrackingTest(unittest.TestCase):
    """muon_track_displacement records Q <- beta Q + beta/(1-beta) dW and leaves the trajectory unchanged."""

    def run_steps(self, track, steps=3):
        torch.manual_seed(0)
        config = load_config(overrides={"model": TINY, "optimizer": "muon", "precision": "fp32", "compile": False,
                                        "ns_polynomial": "svd", "muon_track_displacement": track})
        model = GPT(ModelConfig(**config["model"]))
        optimizer, _ = make_optimizer(model, config, torch.device("cpu"))
        generator = torch.Generator().manual_seed(1)
        history = [{n: p.detach().clone() for n, p in model.named_parameters()}]
        for _ in range(steps):
            tokens = torch.randint(0, 64, (4, 9), generator=generator)
            optimizer.zero_grad(set_to_none=True)
            model(tokens[:, :-1], tokens[:, 1:]).backward()
            optimizer.step()
            history.append({n: p.detach().clone() for n, p in model.named_parameters()})
        return model, optimizer, history, config

    def test_tracking_is_measurement_only_and_matches_recursion(self):
        plain, _, plain_history, _ = self.run_steps(False)
        tracked, optimizer, history, config = self.run_steps(True)
        for name, parameter in plain.named_parameters():
            torch.testing.assert_close(tracked.state_dict()[name], parameter, rtol=0, atol=0)
        beta = config["muon_momentum"]
        for name, parameter in tracked.named_parameters():
            if not (name.startswith("blocks.") and parameter.ndim == 2):
                continue
            expected = torch.zeros_like(parameter)
            for before, after in zip(history[:-1], history[1:]):
                expected = beta * expected + beta / (1 - beta) * (after[name] - before[name])
            torch.testing.assert_close(optimizer.state[parameter]["displacement_ema"], expected, rtol=1e-5, atol=1e-7)


class TwoSidedDataNormTest(unittest.TestCase):
    tracked = {**TINY, "track_input_stats": True, "track_input_cov": True}

    def run_with_stats(self, overrides, stats_fn, seed=0):
        torch.manual_seed(seed)
        config = load_config(overrides={"model": self.tracked, "optimizer": "muon", "precision": "fp32",
                                        "compile": False, **overrides})
        model = GPT(ModelConfig(**config["model"]))
        optimizer, _ = make_optimizer(model, config, torch.device("cpu"))
        generator = torch.Generator().manual_seed(seed + 1)
        tokens = torch.randint(0, 64, (4, 9), generator=generator)
        before = {n: p.detach().clone() for n, p in model.named_parameters()}
        model(tokens[:, :-1], tokens[:, 1:]).backward()
        if stats_fn is not None:
            optimizer.update_output_statistics(stats_fn(model), 0.8)
        optimizer.step()
        return model, before

    def body(self, model):
        return {n: p for n, p in model.named_parameters() if n.startswith("blocks.") and p.ndim == 2}

    def test_isotropic_output_statistics_reproduce_pd(self):
        base = {"data_norm_alpha": 0.25, "ns_polynomial": "svd"}
        pd, before = self.run_with_stats(base, None)
        identity = lambda m: {p: torch.eye(p.shape[0]) for p in self.body(m).values()}
        two, before2 = self.run_with_stats({**base, "data_norm_out_beta": 0.25}, identity)
        for name in self.body(pd):
            torch.testing.assert_close(two.state_dict()[name] - before2[name], pd.state_dict()[name] - before[name],
                                       rtol=1e-5, atol=1e-7)

    def test_output_root_damps_stiff_output_direction(self):
        base = {"data_norm_alpha": 0.25, "ns_polynomial": "svd"}
        def stiff_first_row(m):
            stats = {}
            for p in self.body(m).values():
                s = torch.eye(p.shape[0]); s[0, 0] = 1e4
                stats[p] = s
            return stats
        pd, before = self.run_with_stats(base, None)
        two, before2 = self.run_with_stats({**base, "data_norm_out_beta": 0.25}, stiff_first_row)
        for name in self.body(pd):
            row_pd = (pd.state_dict()[name] - before[name])[0].norm() / (pd.state_dict()[name] - before[name]).norm()
            row_two = (two.state_dict()[name] - before2[name])[0].norm() / (two.state_dict()[name] - before2[name]).norm()
            self.assertLess(float(row_two), float(row_pd))

    def test_placebo_output_root_keeps_spectrum_but_not_basis(self):
        base = {"data_norm_alpha": 0.25, "ns_polynomial": "svd", "data_norm_out_beta": 0.25}
        def stiff_first_row(m):
            stats = {}
            for p in self.body(m).values():
                s = torch.eye(p.shape[0]); s[0, 0] = 1e4
                stats[p] = s
            return stats
        real, before = self.run_with_stats(base, stiff_first_row)
        placebo, before2 = self.run_with_stats({**base, "data_norm_out_placebo": True}, stiff_first_row)
        for name in self.body(real):
            d_real = real.state_dict()[name] - before[name]
            d_placebo = placebo.state_dict()[name] - before2[name]
            self.assertFalse(torch.allclose(d_real, d_placebo, rtol=1e-3, atol=1e-6))
            # the real root damps the stiff first output row; the placebo spreads that damping over a random basis
            self.assertLess(float(d_real[0].norm() / d_real.norm()), float(d_placebo[0].norm() / d_placebo.norm()))

    def test_output_second_moments_have_no_side_effects(self):
        from .muon import output_second_moments
        config = load_config(overrides={"model": self.tracked, "optimizer": "muon", "precision": "fp32", "compile": False})
        model = GPT(ModelConfig(**config["model"]))
        tokens = torch.randint(0, 64, (3, 9))
        weights_before = {n: b.clone() for n, b in model.named_buffers() if n.endswith("input_cov_weight")}
        for source in ("ef", "gn"):
            stats = output_second_moments(model, tokens[:, :-1], tokens[:, 1:], source, config, torch.device("cpu"),
                                          torch.Generator().manual_seed(0))
            self.assertEqual(len(stats), 12)
            for weight, moment in stats.items():
                self.assertEqual(tuple(moment.shape), (weight.shape[0], weight.shape[0]))
                torch.testing.assert_close(moment, moment.T)
                self.assertGreaterEqual(float(torch.linalg.eigvalsh(moment.double()).min()), -1e-8)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for n, b in model.named_buffers():
            if n.endswith("input_cov_weight"):
                torch.testing.assert_close(b, weights_before[n])
        self.assertTrue(model.training)

    def test_config_validation(self):
        for bad in ({"data_norm_out_beta": -0.1}, {"data_norm_out_source": "x"}, {"data_norm_out_ema": 1.0},
                    {"data_norm_out_refresh": 0}):
            with self.assertRaises(ValueError):
                load_config(overrides=bad)

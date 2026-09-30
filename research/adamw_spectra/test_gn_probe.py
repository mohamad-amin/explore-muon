"""CPU checks of the second-order audit probes (gn_probe.py) on a tiny frontier-norm model."""
import unittest

import torch

from . import gn_probe as P
from .model import GPT, ModelConfig


class GNProbe(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        config = dict(vocab_size=64, n_layer=2, n_embd=32, n_head=2, seq_len=16, bias=False,
                      norm="rmsnorm", qk_norm=True)
        self.model = GPT(ModelConfig(**config)).eval()
        # Larger residual-writer weights than the GPT init, so every path carries signal.
        with torch.no_grad():
            for block in self.model.blocks:
                block.attn.o.weight.mul_(10)
                block.mlp.down.weight.mul_(10)
        self.tokens = torch.randint(0, 64, (4, 16))
        self.targets = torch.randint(0, 64, (4, 16))

    def test_explicit_attention_matches_sdpa(self):
        with torch.no_grad():
            reference = self.model(self.tokens)
            with P.explicit_attention():
                explicit = self.model(self.tokens)
        torch.testing.assert_close(explicit, reference, rtol=1e-5, atol=1e-5)

    def test_per_sequence_gradients_sum_to_batch_gradient(self):
        recorder = P.Recorder(self.model)
        grads, _ = P.gradient_passes(self.model, recorder, self.tokens, self.targets, draws=1)
        self.assertEqual(len(grads), 12)
        for name in recorder.layers:
            self.assertEqual(len(recorder.errors[name]), 2)
            per_sequence = P.per_sequence_gradients(recorder.inputs[name], recorder.errors[name][0])
            torch.testing.assert_close(per_sequence.sum(0), grads[name], rtol=1e-4, atol=1e-6)
        recorder.remove()

    def test_sampled_second_moment_matches_exact_gn(self):
        generator = torch.Generator().manual_seed(1)
        direction = P.random_like({name: layer.weight for name, layer in P.hidden_linears(self.model).items()},
                                  generator)
        exact, _ = P.gn_quadratic(self.model, self.tokens, direction)
        recorder = P.Recorder(self.model)
        draws = 300
        P.gradient_passes(self.model, recorder, self.tokens, self.targets, generator, draws=draws)
        T = self.tokens.shape[1]
        total = torch.zeros(self.tokens.shape[0], draws)
        for name in recorder.layers:
            for k, e in enumerate(recorder.errors[name][1:]):
                g = P.per_sequence_gradients(recorder.inputs[name], e)
                total[:, k] += (g * direction[name]).sum((1, 2))
        estimate = (T * total ** 2).mean()
        self.assertLess(abs(estimate / exact - 1), 0.12)   # ~3.3x the Monte Carlo relative error
        recorder.remove()

    def test_symmetry_directions_have_zero_curvature(self):
        generator = torch.Generator().manual_seed(2)
        for block in (1, 2):
            gauge = P.vo_gauge_direction(self.model, block, head=1, generator=generator)
            null, _ = P.gn_quadratic(self.model, self.tokens, gauge)
            reference, _ = P.gn_quadratic(self.model, self.tokens, P.random_like(gauge, generator))
            self.assertLess(null / reference, 1e-8)
            radial = P.radial_head_direction(self.model, block, "q", head=0)
            null, _ = P.gn_quadratic(self.model, self.tokens, radial)
            reference, _ = P.gn_quadratic(self.model, self.tokens, P.random_like(radial, generator))
            self.assertLess(null / reference, 1e-6)

    def test_more_symmetries_and_non_symmetries(self):
        generator = torch.Generator().manual_seed(4)
        for direction, bound in ((P.norm_gain_direction(self.model, 1, "ln1", generator), 1e-6),
                                 (P.norm_gain_direction(self.model, 2, "ln2", generator), 1e-6),
                                 (P.residual_scale_direction(self.model), 1e-6)):
            null, _ = P.gn_quadratic(self.model, self.tokens, direction)
            reference, _ = P.gn_quadratic(self.model, self.tokens, P.random_like(direction, generator))
            self.assertLess(null / reference, bound)
        # A single writer's radial direction is not a symmetry.
        o = P.hidden_linears(self.model)["block01.o"].weight
        curved, _ = P.gn_quadratic(self.model, self.tokens, {"block01.o": o})
        reference, _ = P.gn_quadratic(self.model, self.tokens, P.random_like({"block01.o": o}, generator))
        self.assertGreater(curved / reference, 1e-3)

    def test_marginal_traces_agree_with_frame(self):
        recorder = P.Recorder(self.model, names=["block02.k", "block01.down"])
        generator = torch.Generator().manual_seed(5)
        P.gradient_passes(self.model, recorder, self.tokens, self.targets, generator, draws=2)
        T = self.tokens.shape[1]
        for name, layer in recorder.layers.items():
            x, errors = recorder.inputs[name], recorder.errors[name]
            m = P.Marginals(layer.weight.shape[0], layer.weight.shape[1], T, "cpu")
            m.add(x, errors[0], errors[1:])
            s = m.summary()
            c_sum, b_sum, count = P.second_moments(x, T * errors[1])
            lam_c, v = P.eigenbasis(c_sum, count)
            lam_b, u = P.eigenbasis(b_sum, count)
            frame = P.Frame(u, v, lam_b, lam_c, T)
            frame.add(x, errors[0], errors[1:])
            f = frame.summary()
            for key in ("exact_in", "exact_out"):
                torch.testing.assert_close(s[key].trace(), f["exact"].sum(), rtol=1e-4, atol=0)
            for key in ("token_in", "token_out"):
                torch.testing.assert_close(s[key].trace(), f["ekfac"].sum(), rtol=1e-4, atol=0)
            torch.testing.assert_close(s["ef_in"].trace(), s["ef_out"].trace(), rtol=1e-5, atol=0)
        recorder.remove()

    def test_directional_terms_match_finite_differences(self):
        generator = torch.Generator().manual_seed(6)
        weights = {name: layer.weight for name, layer in P.hidden_linears(self.model).items()}
        d = P.random_like(weights, generator)
        terms = P.directional_terms(self.model, self.tokens, self.targets, d)
        eps = 1e-3
        def loss_at(scale):
            with torch.no_grad():
                for name, layer in P.hidden_linears(self.model).items():
                    layer.weight.add_(scale * d[name])
                value = P.token_losses(self.model(self.tokens), self.targets).mean().double()
                for name, layer in P.hidden_linears(self.model).items():
                    layer.weight.sub_(scale * d[name])
            return value
        slope = (loss_at(eps) - loss_at(-eps)) / (2 * eps)
        torch.testing.assert_close(terms["first"].mean().double(), slope, rtol=2e-2, atol=1e-6)
        q, _ = P.gn_quadratic(self.model, self.tokens, d)
        torch.testing.assert_close(terms["q"].mean(), q, rtol=1e-6, atol=0)

    def test_key_errors_sum_to_zero_without_qk_norm(self):
        """Softmax ignores a common shift of all of a sequence's keys, so without QK-norm each sequence's
        key errors sum to zero over positions: W_k only sees within-sequence-centered inputs."""
        torch.manual_seed(7)
        config = dict(vocab_size=64, n_layer=2, n_embd=32, n_head=2, seq_len=16, bias=False,
                      norm="rmsnorm", qk_norm=False)
        model = GPT(ModelConfig(**config)).eval()
        recorder = P.Recorder(model, names=["block01.k", "block02.k", "block01.v"])
        P.gradient_passes(model, recorder, self.tokens, self.targets, torch.Generator().manual_seed(8), draws=1)
        for name in ("block01.k", "block02.k"):
            for e in recorder.errors[name]:
                self.assertLess(float(e.sum(1).norm() / e.norm()), 1e-5)
        e = recorder.errors["block01.v"][0]
        self.assertGreater(float(e.sum(1).norm() / e.norm()), 1e-2)
        m = P.Marginals(32, 32, 16, "cpu")
        m.add(recorder.inputs["block01.k"], recorder.errors["block01.k"][0], recorder.errors["block01.k"][1:])
        fit = P.within_between_fit(m.summary())
        self.assertLess(fit["residual_fit"], fit["residual_kfac"])
        recorder.remove()

    def test_frame_bookkeeping(self):
        recorder = P.Recorder(self.model, names=["block01.up", "block02.v"])
        generator = torch.Generator().manual_seed(3)
        P.gradient_passes(self.model, recorder, self.tokens, self.targets, generator, draws=2)
        T = self.tokens.shape[1]
        for name in recorder.layers:
            x, errors = recorder.inputs[name], recorder.errors[name]
            c_sum, b_sum, count = P.second_moments(x, T * errors[1])
            lam_c, v = P.eigenbasis(c_sum, count)
            lam_b, u = P.eigenbasis(b_sum, count)
            frame = P.Frame(u, v, lam_b, lam_c, T)
            frame.add(x, errors[0], errors[1:])
            s = frame.summary()
            # Parseval in an orthonormal frame
            sampled = [P.per_sequence_gradients(x, e) for e in errors[1:]]
            exact_trace = T * torch.stack([(g.double() ** 2).sum((1, 2)) for g in sampled]).mean()
            torch.testing.assert_close(s["exact"].sum(), exact_trace, rtol=1e-4, atol=0)
            token_trace = torch.stack([((T * e).double().pow(2).sum(-1) * x.double().pow(2).sum(-1)).mean()
                                       for e in errors[1:]]).mean()
            torch.testing.assert_close(s["ekfac"].sum(), token_trace, rtol=1e-4, atol=0)
            torch.testing.assert_close(s["kfac"].sum(), lam_b.sum() * lam_c.sum(), rtol=1e-8, atol=0)
            mean_gradient = P.per_sequence_gradients(x, errors[0]).double().mean(0)
            torch.testing.assert_close(s["signal"].pow(2).sum(), mean_gradient.pow(2).sum(), rtol=1e-4, atol=0)
            self.assertTrue(bool((s["noise"] >= -1e-12).all()))
        recorder.remove()


if __name__ == "__main__":
    unittest.main()

"""CPU contracts for optimizer parity, statistics isolation and state accounting."""

import copy
import io
import unittest

import torch

from research.adamw_spectra.model import GPT, ModelConfig
from research.adamw_spectra.muon import output_second_moments as production_moments
from .optim import (METHODS, make_optimizer, output_second_moments,
                    snapshot_optimizer, tensor_memory_breakdown)


class TinyOptimizerContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(417)
        self.model = GPT(ModelConfig(vocab_size=32, n_layer=2, n_embd=8, n_head=2,
                                     seq_len=8, bias=False, norm="rmsnorm", qk_norm=True,
                                     track_input_stats=True, track_input_cov=True))
        self.x = torch.randint(0, 32, (3, 8))
        self.y = torch.randint(0, 32, (3, 8))
        self.cfg = dict(method="muon", lr=.01, aux_lr=.002, momentum=.9, alpha=.25,
                        out_beta=.25, decay=.01, soap_beta2=.9, root_refresh=2)

    def test_group_coverage_and_equal_auxiliary_recipe(self):
        expected_body = {id(p) for n, p in self.model.named_parameters()
                         if n.startswith("blocks.") and p.ndim == 2}
        for method in METHODS:
            with self.subTest(method=method):
                optimizer = make_optimizer(self.model, {**self.cfg, "method": method}, "cpu")
                body, aux = optimizer.param_groups
                self.assertEqual({id(p) for p in body["params"]}, expected_body)
                self.assertFalse({id(p) for p in aux["params"]} & expected_body)
                self.assertEqual({id(p) for g in optimizer.param_groups for p in g["params"]},
                                 {id(p) for p in self.model.parameters()})
                self.assertEqual(body["lr"], .01)
                self.assertEqual(aux["lr"], .002)
                self.assertEqual(aux["betas"], (.9, .95))
                self.assertEqual(aux["eps"], 1e-8)

    def test_gn_and_ef_match_production_without_mutation(self):
        # Existing gradients must survive a statistics call, including nonzero ones.
        self.model(self.x, self.y).backward()
        for source in ("gn", "ef"):
            before_buffers = {name: value.clone() for name, value in self.model.named_buffers()}
            before_gradients = {name: p.grad.clone() for name, p in self.model.named_parameters()}
            before_parameters = {name: p.clone() for name, p in self.model.named_parameters()}
            generator = torch.Generator().manual_seed(1337)
            actual = output_second_moments(self.model, self.x, self.y, source, "fp32", generator)
            expected = production_moments(
                self.model, self.x, self.y, source, {"precision": "fp32"}, torch.device("cpu"),
                torch.Generator().manual_seed(1337))
            self.assertEqual(len(actual), 12)
            for parameter in actual:
                torch.testing.assert_close(actual[parameter], expected[parameter], rtol=0, atol=0)
                self.assertEqual(actual[parameter].dtype, torch.float32)
                self.assertFalse(actual[parameter].requires_grad)
                torch.testing.assert_close(actual[parameter], actual[parameter].T, rtol=0, atol=0)
                self.assertGreaterEqual(float(torch.linalg.eigvalsh(actual[parameter].double()).min()), -1e-8)
            for name, value in self.model.named_buffers():
                torch.testing.assert_close(value, before_buffers[name], rtol=0, atol=0)
            for name, parameter in self.model.named_parameters():
                torch.testing.assert_close(parameter, before_parameters[name], rtol=0, atol=0)
                torch.testing.assert_close(parameter.grad, before_gradients[name], rtol=0, atol=0)
            self.assertTrue(self.model.training)
        self.model.blocks[0].eval()
        modes = {name: module.training for name, module in self.model.named_modules()}
        output_second_moments(self.model, self.x, self.y, "ef", "fp32")
        self.assertEqual({name: module.training for name, module in self.model.named_modules()}, modes)

    def _stepped(self, method, out_beta, identity=False):
        model = copy.deepcopy(self.model)
        optimizer = make_optimizer(model, {**self.cfg, "method": method, "out_beta": out_beta,
                                            "ns_polynomial": "svd"}, "cpu")
        optimizer.capture_parameters = set(optimizer.param_groups[0]["params"])
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            model(self.x, self.y).backward()
            if identity:
                optimizer.update_output_statistics(
                    {p: torch.eye(p.shape[0]) for p in optimizer.param_groups[0]["params"]}, .8)
            optimizer.step()
        return model, optimizer

    def test_ts_zero_output_exponent_equals_pd(self):
        for first, second in (("pd", "ts"), ("spd", "sts")):
            with self.subTest(methods=(first, second)):
                pd, _ = self._stepped(first, 0)
                ts, _ = self._stepped(second, 0)
                for name, parameter in pd.named_parameters():
                    torch.testing.assert_close(parameter, dict(ts.named_parameters())[name], rtol=0, atol=0)

    def test_isotropic_output_factor_matches_pd_exact_polar(self):
        pd, _ = self._stepped("pd", 0)
        ts, _ = self._stepped("ts", .25, identity=True)
        for name, parameter in pd.named_parameters():
            torch.testing.assert_close(parameter, dict(ts.named_parameters())[name],
                                       rtol=2e-4, atol=2e-6)
        # SOAP is deliberately absent here: its rank-deficient gradient Grams
        # have numerically nonunique nullspace bases. A three-step S-PD/S-TS
        # isotropic comparison found a 1.6e-4 weight discrepancy even with exact
        # polar. This is not a warranted invariance contract for that kernel.

    def test_auxiliary_adamw_writes_match_across_methods(self):
        baseline, muon = copy.deepcopy(self.model), copy.deepcopy(self.model)
        first = make_optimizer(baseline, {**self.cfg, "method": "adamw"}, "cpu")
        second = make_optimizer(muon, {**self.cfg, "method": "muon"}, "cpu")
        for _ in range(3):
            for (name, p), (_, q) in zip(baseline.named_parameters(), muon.named_parameters()):
                p.grad = torch.randn_like(p)
                q.grad = p.grad.clone()
            first.step()
            second.step()
            for (name, p), (_, q) in zip(baseline.named_parameters(), muon.named_parameters()):
                if not (name.startswith("blocks.") and p.ndim == 2):
                    torch.testing.assert_close(p, q, rtol=0, atol=0)

    def test_fp32_weights_enforced_and_momentum_preserved(self):
        with self.assertRaisesRegex(ValueError, "FP32"):
            make_optimizer(copy.deepcopy(self.model).to(torch.bfloat16), self.cfg, "cpu")
        _, optimizer = self._stepped("sts", .25, identity=True)
        for parameter in optimizer.param_groups[0]["params"]:
            self.assertEqual(optimizer.state[parameter]["momentum_buffer"].dtype, torch.float32)

    def test_snapshot_is_cpu_data_named_complete_and_detached(self):
        model, optimizer = self._stepped("sts", .25, identity=True)
        snapshot = snapshot_optimizer(model, optimizer)
        names = dict(model.named_parameters())
        self.assertEqual(set(snapshot["state"]), set(names))
        self.assertTrue(snapshot["external"]["soap"]["states"])
        self.assertTrue(snapshot["external"]["data_norm"]["out_stats"])
        self.assertNotIn("modules", snapshot["external"]["data_norm"])
        self.assertTrue(snapshot["model_statistics"])
        for name, state in snapshot["state"].items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    self.assertEqual(value.device.type, "cpu")
                    self.assertFalse(value.requires_grad)
                    torch.testing.assert_close(value, optimizer.state[names[name]][key])
        stream = io.BytesIO()
        torch.save(snapshot, stream)
        stream.seek(0)
        recovered = torch.load(stream, weights_only=True)
        self.assertEqual(recovered["format_version"], 1)
        body_name = optimizer.param_groups[0]["params"][0]
        name = next(name for name, p in model.named_parameters() if p is body_name)
        saved = snapshot["state"][name]["momentum_buffer"].clone()
        optimizer.state[body_name]["momentum_buffer"].zero_()
        torch.testing.assert_close(snapshot["state"][name]["momentum_buffer"], saved, rtol=0, atol=0)

    def test_memory_counts_unique_storages(self):
        model, optimizer = self._stepped("sts", .25, identity=True)
        before = tensor_memory_breakdown(model, optimizer)
        self.assertGreater(before["optimizer_standard"]["bytes"], 0)
        self.assertGreater(before["optimizer_external"]["bytes"], 0)
        self.assertGreater(before["model_statistics"]["bytes"], 0)
        self.assertEqual(before["model_parameters"]["bytes"], sum(p.numel() * 4 for p in model.parameters()))
        parameter = optimizer.param_groups[0]["params"][0]
        optimizer.soap["alias_view"] = optimizer.state[parameter]["momentum_buffer"].view(-1)
        after = tensor_memory_breakdown(model, optimizer)
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()

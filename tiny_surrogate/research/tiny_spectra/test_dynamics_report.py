"""CPU-only scientific contracts for the cross-cohort dynamics report."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from .analyze import (apply_provenance_vetoes, candidate_selections, data_identity_sha256,
                      provenance_groups, read_run)
from .dynamics_report import (EXECUTION_SOURCES, audit_pairing, crossing, load_runs,
                              ratio_interval, ratio_record, recipe, select_lrs, select_momenta)


def run(lr=.01, bank=1.5, full=1.6, status="complete", momentum=.9, seed=1, **config):
    cfg = dict(method="spd", batch_tokens=100, momentum=momentum, lr=lr,
               seed=seed, total_tokens=1000, n_embd=128, eval_every=2, **config)
    uid = f"run-{seed}-{momentum}-{lr}-{cfg.get('selection_metric', 'default')}"
    return dict(uid=uid, config=cfg, family="family_00",
                row=dict(run_id=uid, status=status, issues=[], endpoint_bank_nll=bank, full_validation_nll=full))


class DynamicsContracts(unittest.TestCase):
    def test_crossing_interpolates_only_bracket(self):
        points = [dict(step=0, tokens=0, validation_nll=4),
                  dict(step=10, tokens=1000, validation_nll=3),
                  dict(step=20, tokens=2000, validation_nll=2)]
        c = crossing(points, 2.4)
        self.assertEqual(c["status"], "interpolated")
        self.assertAlmostEqual(c["step"], 16)
        self.assertAlmostEqual(c["tokens"], 1600)
        self.assertEqual(c["bracket_steps"], [10, 20])
        self.assertEqual(c["bracket_tokens"], [1000, 2000])
        self.assertEqual(c["bracket_nll"], [3, 2])
        self.assertEqual(crossing(points, 1.9)["status"], "right_censored")
        self.assertIsNone(crossing(points, 1.9)["step"])
        self.assertEqual(crossing(points, 4.1)["status"], "left_censored")
        self.assertIsNone(crossing(points, 4.1)["step"])
        self.assertEqual(crossing(points, 4)["step"], 0)

    def test_exact_hit_retains_observation_resolution(self):
        points = [dict(step=0, tokens=0, validation_nll=4),
                  dict(step=10, tokens=1000, validation_nll=3),
                  dict(step=20, tokens=2000, validation_nll=2)]
        exact = crossing(points, 2)
        self.assertEqual(exact["status"], "observed")
        self.assertEqual(exact["step"], 20)
        self.assertEqual(exact["bracket_steps"], [10, 20])
        self.assertEqual(crossing(points, 4)["bracket_steps"], [0, 0])

    def test_ratio_resolution_is_conservative_not_interpolation_precision(self):
        muon = dict(evaluations=[dict(step=0, tokens=0, validation_nll=4),
                                 dict(step=10, tokens=100, validation_nll=3),
                                 dict(step=30, tokens=300, validation_nll=2)])
        spd = dict(evaluations=[dict(step=0, tokens=0, validation_nll=4),
                                dict(step=8, tokens=80, validation_nll=3),
                                dict(step=16, tokens=160, validation_nll=2)])
        record = ratio_record(muon, spd, 2.5)
        self.assertAlmostEqual(record["muon_steps_per_spd_step"], 20 / 12)
        interval = record["muon_steps_per_spd_step_interval"]
        self.assertEqual(interval["status"], "bounded")
        self.assertEqual(interval["lower"], 10 / 16)
        self.assertEqual(interval["upper"], 30 / 8)
        reciprocal = record["spd_steps_per_muon_step_interval"]
        self.assertEqual(reciprocal["lower"], 8 / 30)
        self.assertEqual(reciprocal["upper"], 16 / 10)
        # An exact hit at the right endpoint does not collapse the interval.
        exact = ratio_record(muon, spd, 2)
        self.assertEqual(exact["muon_steps_per_spd_step_interval"], interval)

    def test_resolution_interval_zero_and_censored_cases(self):
        m = dict(bracket_steps=[10, 30])
        upper_unbounded = ratio_interval(m, dict(bracket_steps=[0, 8]))
        self.assertEqual(upper_unbounded["lower"], 10 / 8)
        self.assertIsNone(upper_unbounded["upper"])
        self.assertTrue(upper_unbounded["upper_unbounded"])
        zero = ratio_interval(m, dict(bracket_steps=[0, 0]))
        self.assertEqual(zero["status"], "undefined_zero_denominator")
        self.assertIsNone(zero["lower"])
        self.assertFalse(zero["upper_unbounded"])
        for censored in (None, [None, 10], [30, None]):
            result = ratio_interval(m, dict(bracket_steps=censored))
            self.assertEqual(result["status"], "censored_or_missing")
            self.assertIsNone(result["lower"])
            self.assertIsNone(result["upper"])

    def test_first_passage_without_smoothing(self):
        points = [dict(step=i * 10, tokens=i * 100, validation_nll=v)
                  for i, v in enumerate([4, 2, 3, 1])]
        self.assertEqual(crossing(points, 2)["step"], 10)
        self.assertEqual(crossing(points, 2.5)["step"], 7.5)
        self.assertEqual(crossing(points, 1)["step"], 30)

    def test_ratio_zero_and_unreached_censored(self):
        left = dict(evaluations=[dict(step=0, tokens=0, validation_nll=4),
                                 dict(step=20, tokens=200, validation_nll=2)])
        right = dict(evaluations=[dict(step=0, tokens=0, validation_nll=4),
                                  dict(step=10, tokens=100, validation_nll=2)])
        self.assertEqual(ratio_record(left, right, 3)["muon_steps_per_spd_step"], 2)
        self.assertEqual(ratio_record(left, right, 3)["spd_steps_per_muon_step"], .5)
        self.assertIsNone(ratio_record(left, right, 4)["muon_steps_per_spd_step"])
        self.assertIsNone(ratio_record(left, right, 1)["muon_steps_per_spd_step"])
        self.assertEqual(ratio_record(left, right, 1)["muon_steps_per_spd_step_interval"]["status"], "censored_or_missing")
        self.assertEqual(ratio_record(left, right, 4)["muon_steps_per_spd_step_interval"]["status"], "undefined_zero_denominator")

    def test_final_bank_not_full_split_selects_lr(self):
        choices = select_lrs([run(.01, bank=1.4, full=1.9), run(.02, bank=1.5, full=1.3)])
        self.assertEqual(len(choices), 1)
        self.assertEqual(choices[0]["selected"]["lr"], .01)
        self.assertEqual(choices[0]["selected"]["full_nll"], 1.9)
        self.assertTrue(choices[0]["grid_complete"])

    def test_boundary_tie_does_not_qualify_an_interior_selected_rate(self):
        arms = [run(.004, full=2.5, selection_metric="full"),
                run(.008, full=2.3, selection_metric="full"),
                run(.016, full=2.3, selection_metric="full")]
        joint = candidate_selections(arms)[0]
        individual = select_lrs(arms)[0]
        self.assertEqual(joint["selected"]["lr"], .008)
        self.assertEqual(individual["selected"]["lr"], .008)
        self.assertTrue(joint["boundary_lr"])
        self.assertTrue(individual["boundary_lr"])
        self.assertFalse(joint["bracket_qualified"])
        self.assertFalse(individual["bracketed"])

    def test_fineweb_loader_and_shared_runtime_are_both_provenance_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort = Path(directory)
            (cohort / "configs").mkdir()
            paths = (*EXECUTION_SOURCES, "research/tiny_spectra/fineweb.py", "research/tiny_spectra/stories.py")
            manifest = {}
            for relative in paths:
                path = cohort / "frozen" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# synthetic frozen source\n")
                manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            (cohort / "source_manifest.json").write_text(json.dumps(manifest))
            cfg = dict(run()["config"], run_id="fineweb", corpus_kind="fineweb_byte_bpe_v1")
            (cohort / "configs/fineweb.json").write_text(json.dumps(cfg))
            clean = load_runs([cohort])[0][0]
            self.assertFalse(clean["row"]["issues"])
            for relative in paths[-2:]:
                path = cohort / "frozen" / relative
                original = path.read_text()
                path.write_text("# changed source\n")
                bad = load_runs([cohort])[0][0]
                self.assertTrue(any(relative in issue for issue in bad["row"]["issues"]))
                path.write_text(original)

    def test_declared_selection_metric_reverses_lr_winner_in_both_reports(self):
        for policy, expected in ((None, .01), ("bank", .01), ("full", .02)):
            with self.subTest(policy=policy):
                config = {} if policy is None else dict(selection_metric=policy)
                candidates = [run(.01, bank=1.4, full=1.9, **config), run(.02, bank=1.5, full=1.3, **config)]
                dynamics = select_lrs(candidates)[0]
                self.assertEqual(dynamics["selected"]["lr"], expected)
                self.assertEqual(dynamics["selection_metric"], policy or "bank")
                self.assertIn("full-development-split" if policy == "full" else "fixed-bank", dynamics["selection_basis"])
                single = candidate_selections(candidates)[0]
                self.assertEqual(single["selected"]["lr"], expected)
                self.assertEqual(single["selection_policy"], policy or "bank")
                self.assertIn("full-development-split" if policy == "full" else "fixed-bank", single["selection_metric"])
                self.assertEqual(single["selected"]["mean_endpoint_bank_nll"], 1.5 if policy == "full" else 1.4)
                self.assertEqual(single["selected"]["mean_full_validation_nll"], 1.3 if policy == "full" else 1.9)

    def test_declared_metric_reverses_best_momentum(self):
        for policy, expected in ((None, .8), ("bank", .8), ("full", .9)):
            with self.subTest(policy=policy):
                config = {} if policy is None else dict(selection_metric=policy)
                choices = select_momenta(select_lrs([run(momentum=.8, bank=1.4, full=1.8, **config),
                                                     run(momentum=.9, bank=1.5, full=1.3, **config)]))
                self.assertEqual(choices[0]["selected"]["recipe"]["momentum"], expected)
                self.assertEqual(choices[0]["selection_metric"], policy or "bank")

    def test_mixed_selection_policies_split_groups_and_families(self):
        candidates = [run(.01), run(.02, selection_metric="bank"), run(.01, selection_metric="full")]
        self.assertEqual(len(select_lrs(candidates)), 2)
        self.assertEqual(len(candidate_selections(candidates)), 2)
        self.assertEqual(len(select_momenta(select_lrs(candidates))), 2)
        for r in candidates:
            r["metadata"] = dict(initial_parameter_sha256="initial", train_window_sha256="train",
                                 validation_bank_sha256="bank", corpus=dict(splits=dict(val="val")))
            r["provenance"] = dict(source=dict(execution_sha256=dict(train="source")))
        families = audit_pairing(candidates)
        self.assertEqual(len(families), 2)
        self.assertEqual(candidates[0]["family"], candidates[1]["family"])
        self.assertNotEqual(candidates[0]["family"], candidates[2]["family"])

    def test_missing_declared_full_score_cannot_fallback_to_bank(self):
        candidates = [run(.01, bank=1.0, full=None, selection_metric="full"),
                      run(.02, bank=2.0, full=2.1, selection_metric="full")]
        for choice in (select_lrs(candidates)[0], candidate_selections(candidates)[0]):
            self.assertEqual(choice["selected"]["lr"], .02)
            self.assertFalse(choice["grid_complete"])

    def test_unknown_selection_metric_fails_explicitly(self):
        for selector in (select_lrs, candidate_selections):
            with self.assertRaisesRegex(ValueError, "Unknown selection_metric"):
                selector([run(selection_metric="test")])

    def test_data_identity_prefers_manifest_and_retains_text_fallback(self):
        self.assertEqual(data_identity_sha256(dict(data_manifest_sha256="manifest", corpus=dict(raw_utf8_sha256="legacy"))), "manifest")
        self.assertEqual(data_identity_sha256(dict(corpus=dict(raw_utf8_sha256="legacy"))), "legacy")
        self.assertIsNone(data_identity_sha256({}))
        for identity in (dict(data_manifest_sha256="chosen", corpus=dict(raw_utf8_sha256="ignored")),
                         dict(corpus=dict(raw_utf8_sha256="chosen"))):
            with tempfile.TemporaryDirectory() as directory:
                cohort = Path(directory)
                cfg = dict(run()["config"], run_id="sample", data_sha256="chosen")
                config_path = cohort / "sample.json"
                config_path.write_text(json.dumps(cfg))
                run_path = cohort / "runs" / "sample"
                run_path.mkdir(parents=True)
                metadata = dict(initial_parameter_sha256="init", train_window_sha256="train", validation_bank_sha256="val", **identity)
                (run_path / "metadata.json").write_text(json.dumps(metadata))
                (run_path / "summary.json").write_text(json.dumps(dict(status="complete", tokens=1000, final_validation_nll=2.0)))
                (run_path / "status.json").write_text(json.dumps(dict(status="complete")))
                (run_path / "metrics.jsonl").write_text(json.dumps(dict(step=10, tokens=1000, validation_nll=2.0)) + "\n")
                parsed = read_run(cohort, config_path)
                self.assertEqual(parsed["row"]["issues"], [])

    def test_optional_bank_lengths_checked_only_when_recorded(self):
        for lengths in ((None, None), ("same", "same"), ("one", "two"), ("same", None)):
            with self.subTest(lengths=lengths):
                candidates = [run(.01), run(.02)]
                for r, length in zip(candidates, lengths):
                    r["metadata"] = dict(initial_parameter_sha256="initial", train_window_sha256="train",
                        validation_bank_sha256="bank", data_manifest_sha256="data", corpus=dict(splits=dict(val="val")))
                    if length is not None:
                        r["metadata"]["validation_bank_lengths_sha256"] = length
                    r["provenance"] = dict(source=dict(execution_sha256=dict(train="source")))
                single_candidates = copy.deepcopy(candidates)
                groups = provenance_groups(single_candidates)
                apply_provenance_vetoes(single_candidates, groups)
                dynamics_groups = audit_pairing(candidates)
                if lengths == (None, None):
                    self.assertNotIn("validation_bank_lengths_sha256", groups[0]["checks"])
                    self.assertNotIn("validation_bank_lengths_sha256", dynamics_groups[0]["checks"])
                expected = [False, False] if lengths in ((None, None), ("same", "same")) else [True, True] if lengths == ("one", "two") else [False, True]
                for suite in (single_candidates, candidates):
                    self.assertEqual([bool(r["row"]["issues"]) for r in suite], expected)
                self.assertEqual(groups[0]["checks"]["data_identity_sha256"]["status"], "matched")

    def test_manifest_data_mismatch_vetoes_both_reports(self):
        candidates = [run(.01), run(.02)]
        for r, identity in zip(candidates, ("one", "two")):
            r["metadata"] = dict(initial_parameter_sha256="initial", train_window_sha256="train",
                validation_bank_sha256="bank", data_manifest_sha256=identity,
                corpus=dict(raw_utf8_sha256="same-legacy-ignored", splits=dict(val="val")))
            r["provenance"] = dict(source=dict(execution_sha256=dict(train="source")))
        single_candidates = copy.deepcopy(candidates)
        groups = provenance_groups(single_candidates)
        self.assertEqual(groups[0]["checks"]["data_identity_sha256"]["status"], "mismatch")
        apply_provenance_vetoes(single_candidates, groups)
        audit_pairing(candidates)
        for suite in (single_candidates, candidates):
            self.assertTrue(all(r["row"]["issues"] for r in suite))

    def test_conditional_stories_source_hash_and_manifest_change(self):
        with tempfile.TemporaryDirectory() as directory:
            cohort = Path(directory)
            (cohort / "configs").mkdir()
            manifest = {}
            for relative in EXECUTION_SOURCES:
                path = cohort / "frozen" / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("# synthetic unchanged execution file\n")
                manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest_path = cohort / "source_manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            cfg = dict(run()["config"], run_id="legacy")
            (cohort / "configs" / "legacy.json").write_text(json.dumps(cfg))
            legacy, _ = load_runs([cohort])
            self.assertEqual(legacy[0]["row"]["issues"], [])
            story_relative = "research/tiny_spectra/stories.py"
            self.assertNotIn(story_relative, legacy[0]["provenance"]["source"]["execution_sha256"])
            cfg = dict(cfg, run_id="stories", corpus_kind="tiny_stories_byte_bpe_v1")
            (cohort / "configs" / "stories.json").write_text(json.dumps(cfg))
            story_path = cohort / "frozen" / story_relative
            story_path.write_text("# synthetic stories version one\n")
            manifest[story_relative] = hashlib.sha256(story_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            original = next(r for r in load_runs([cohort])[0] if r["config"]["run_id"] == "stories")
            self.assertEqual(original["row"]["issues"], [])
            self.assertIn(story_relative, original["provenance"]["source"]["execution_sha256"])
            self.assertIn(story_relative, original["conditional_source_provenance"]["execution_sha256"])
            # Editing frozen code without updating its manifest is detected.
            story_path.write_text("# synthetic stories version two\n")
            tampered = load_runs([cohort])[0]
            self.assertTrue(any("stories.py" in issue for r in tampered if r["config"]["run_id"] == "stories" for issue in r["row"]["issues"]))
            self.assertEqual(next(r for r in tampered if r["config"]["run_id"] == "legacy")["row"]["issues"], [])
            # Even a correctly updated manifest cannot hide differing executed
            # stories.py versions when two otherwise compatible runs are paired.
            manifest[story_relative] = hashlib.sha256(story_path.read_bytes()).hexdigest()
            manifest_path.write_text(json.dumps(manifest))
            updated = next(r for r in load_runs([cohort])[0] if r["config"]["run_id"] == "stories")
            self.assertEqual(updated["row"]["issues"], [])
            original["uid"], original["row"]["run_id"] = "old-source-run", "old-source-run"
            updated["uid"], updated["row"]["run_id"] = "new-source-run", "new-source-run"
            single = provenance_groups([original, updated])
            self.assertEqual(single[0]["checks"]["conditional_execution_sources"]["status"], "mismatch")
            paired = audit_pairing([original, updated])
            self.assertTrue(paired[0]["checks"]["execution_sources"]["mismatch"])

    def test_incomplete_and_failed_candidates_are_preserved_ineligible(self):
        choices = select_lrs([run(.01), run(.02, bank=1.0, status="running"),
                              run(.03, bank=.9, status="failed")])[0]
        self.assertEqual(len(choices["candidates"]), 3)
        self.assertEqual(choices["selected"]["lr"], .01)
        self.assertTrue(choices["provisional"])
        self.assertFalse(choices["grid_complete"])
        self.assertFalse(choices["candidates"][1]["eligible"])

    def test_provenance_issues_veto_candidate(self):
        bad = run(.01, bank=1.0)
        bad["row"]["issues"].append("hash mismatch")
        choice = select_lrs([bad, run(.02, bank=2)])[0]
        self.assertEqual(choice["selected"]["lr"], .02)
        self.assertTrue(choice["provisional"])

    def test_seed_budget_architecture_separate(self):
        runs = [run(), run(seed=2)]
        other = run()
        other["config"]["n_embd"] = 256
        runs.append(other)
        other = run()
        other["config"]["total_tokens"] = 2000
        runs.append(other)
        self.assertEqual(len(select_lrs(runs)), 4)

    def test_best_momentum_is_endpoint_selected_and_provisional(self):
        selections = select_lrs([run(momentum=.8, bank=1.4, full=1.8),
                                 run(momentum=.9, bank=1.5, full=1.3),
                                 run(momentum=.95, bank=None, full=None, status="running")])
        choice = select_momenta(selections)[0]
        self.assertEqual(choice["selected"]["recipe"]["momentum"], .8)
        self.assertTrue(choice["provisional"])
        self.assertEqual(choice["tested_momenta"], [.8, .9, .95])

    def test_duplicate_same_seed_not_silently_pooled(self):
        first, second = run(), run()
        second["uid"] = "independent-duplicate"
        choice = select_lrs([first, second])[0]
        self.assertIsNone(choice["selected"])

    def test_observation_cadence_is_only_nonscientific_exclusion(self):
        cfg = run()["config"]
        self.assertNotIn("eval_every", recipe(cfg, {"lr"}))
        self.assertIn("seed", recipe(cfg, {"lr"}))
        self.assertIn("total_tokens", recipe(cfg, {"lr"}))


if __name__ == "__main__":
    unittest.main()

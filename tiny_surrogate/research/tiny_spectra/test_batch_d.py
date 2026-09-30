"""Scientific failure-mode tests for the prospective batch readout."""
import copy
import math
import unittest

from .batch_d import BATCHES, RECIPES, SEEDS, TOTAL, analyse, crossing, evaluation_steps, ratio


def family():
    rows = []
    for b, gain in zip(BATCHES, (1.1, 1.3, 1.6)):
        steps = math.ceil(TOTAL / b)
        for seed in SEEDS:
            for method, lr in RECIPES:
                speed = gain if method == "spd" else 1.05 if lr == .02 else 1.
                curve = [dict(step=s, validation_nll=8. - 4. * speed * s / steps)
                         for s in range(steps + 1)]
                rows.append(dict(batch_tokens=b, seed=seed, method=method, lr=lr,
                    status="complete", evaluations=curve, full_development_nll=curve[-1]["validation_nll"]))
    return rows


class BatchReadoutTests(unittest.TestCase):
    def test_resolved_consistent_growth(self):
        result = analyse(family())
        self.assertTrue(result["fixed_recipe_batch_component_passed"])
        self.assertEqual(len(result["all_ratios"]), 2 * 2 * 3 * 56)
        self.assertFalse(result["full_goal_qualified"])
        self.assertTrue(result["five_method_ordering_still_failed"])

    def test_one_seed_reversal_cannot_hide_in_mean(self):
        rows = family()
        for row in rows:
            if row["seed"] == SEEDS[1] and row["method"] == "spd" and row["batch_tokens"] == BATCHES[-1]:
                n = len(row["evaluations"]) - 1
                row["evaluations"] = [dict(step=s, validation_nll=8. - 3.5 * s / n) for s in range(n + 1)]
        result = analyse(rows)
        self.assertFalse(result["fixed_recipe_batch_component_passed"])
        self.assertTrue(all(r["passed"] for r in result["comparisons"] if r["seed"] == SEEDS[0]))

    def test_stronger_muon_control_vetoes_pass(self):
        rows = family()
        for row in rows:
            if row["method"] == "muon" and row["lr"] == .02 and row["batch_tokens"] == BATCHES[-1]:
                n = len(row["evaluations"]) - 1
                row["evaluations"] = [dict(step=s, validation_nll=8. - 8. * s / n) for s in range(n + 1)]
        result = analyse(rows)
        self.assertFalse(result["fixed_recipe_batch_component_passed"])
        self.assertTrue(all(r["passed"] for r in result["comparisons"] if r["muon_lr"] == .01))

    def test_sparse_crossings_remain_unresolved(self):
        rows = family()
        for row in rows:
            row["evaluations"] = [row["evaluations"][0], row["evaluations"][-1]]
        result = analyse(rows)
        self.assertFalse(result["coverage_passed"])
        self.assertFalse(result["fixed_recipe_batch_component_passed"])

    def test_instability_and_missing_trajectories(self):
        rows = family()
        with self.assertRaises(ValueError):
            analyse(rows[:-1])
        rows[-1]["status"] = "numerical_instability"
        rows[-1]["full_development_nll"] = None
        result = analyse(rows)
        self.assertFalse(result["fixed_recipe_batch_component_passed"])
        self.assertTrue(any(r["spd_crossing"]["status"] == "unstable" for r in result["all_ratios"]))

    def test_nonmonotone_first_crossing_and_censoring(self):
        curve = [dict(step=s, validation_nll=v) for s, v in ((0, 8), (8, 6), (16, 7), (24, 5))]
        self.assertEqual(crossing(curve, 6.5), dict(status="reached", left=0, right=8, step=6.))
        self.assertEqual(crossing(curve, 4)["status"], "unreached")
        self.assertEqual(crossing(curve, 9)["status"], "initial")
        self.assertIsNone(ratio(crossing(curve, 6.5), crossing(curve, 4))["value"])

    def test_no_growth_cannot_pass(self):
        rows = family()
        for row in rows:
            n = len(row["evaluations"]) - 1
            speed = 1.3 if row["method"] == "spd" else 1.
            row["evaluations"] = [dict(step=s, validation_nll=8. - 4. * speed * s / n) for s in range(n + 1)]
        self.assertFalse(analyse(rows)["fixed_recipe_batch_component_passed"])

    def test_no_favorable_midpoint_omission(self):
        rows = family()
        for row in rows:
            if row["method"] == "spd" and row["batch_tokens"] == BATCHES[1]:
                n = len(row["evaluations"]) - 1
                row["evaluations"] = [dict(step=s, validation_nll=8. - 8. * s / n) for s in range(n + 1)]
        result = analyse(rows)
        self.assertFalse(result["fixed_recipe_batch_component_passed"])
        self.assertTrue(all(r["gates"]["growth_resolved"] for r in result["comparisons"]))
        self.assertFalse(any(r["gates"]["midpoint_monotone"] for r in result["comparisons"]))

    def test_cadence_includes_partial_endpoint(self):
        self.assertEqual(evaluation_steps(1048576, 1), list(range(93)))
        self.assertEqual(evaluation_steps(65536, 8)[-2:], [1464, 1469])


if __name__ == "__main__":
    unittest.main()

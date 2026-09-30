"""Synthetic readout contracts; no fresh-run results or model operations."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from . import seed_replication_report as report


class SeedReplicationContracts(unittest.TestCase):
    def test_no_summary_opened_before_every_status_is_complete(self):
        with tempfile.TemporaryDirectory() as temporary:
            roots = [Path(temporary) / str(i) for i in range(4)]
            for i, root in enumerate(roots):
                root.mkdir()
                (root / "status.json").write_text(json.dumps(dict(status="complete" if i < 3 else "running")))
                (root / "summary.json").write_text("Should never be read")
            reads = []
            real_read = report.read
            def guarded_read(path):
                reads.append(Path(path).name)
                if Path(path).name != "status.json":
                    self.fail("A scientific summary was opened before all four statuses passed")
                return real_read(path)
            with patch.object(report, "read", side_effect=guarded_read):
                with self.assertRaises(report.IncompleteCohort):
                    report.require_complete(roots)
            self.assertEqual(reads, ["status.json"] * 4)

    def test_window_and_document_weighting_and_compensating_error(self):
        counts = torch.tensor([4, 1, 2, 3, 1])
        owners = torch.tensor([0, 0, 1, 2, 3])
        losses = torch.tensor([1., 3., 2., 4., 5.], dtype=torch.float64)
        identities = [f"{i:02x}" + "0" * 62 for i in range(4)]
        population = dict(starts=torch.arange(5), target_counts=counts, document_indices=owners,
                          targets=11, documents=[dict(identity=name) for name in identities])
        validation = {key: population[key].clone() for key in ("starts", "target_counts", "document_indices")}
        validation["sequence_nll"] = losses
        rows = [dict(identity=name, targets=n, nll=loss) for name, n, loss in
                zip(identities, [5, 2, 3, 1], [1.4, 2., 4., 5.])]
        record = dict(total_targets=11, documents=rows)
        mean, groups = report.reconstruct_evaluation(validation, record, population)
        self.assertAlmostEqual(mean, 28 / 11)
        self.assertEqual(groups[0]["targets"], 5)
        self.assertAlmostEqual(groups[0]["nll"], 1.4)
        # Keep the global weighted mean unchanged while corrupting two documents.
        rows[0]["nll"] += .2
        rows[1]["nll"] -= .5
        with self.assertRaisesRegex(ValueError, "per-document"):
            report.reconstruct_evaluation(validation, record, population)

    def test_same_totals_do_not_accept_changed_window_membership(self):
        counts = torch.ones(4, dtype=torch.long)
        population = dict(starts=torch.arange(4), target_counts=counts,
                          document_indices=torch.arange(4), targets=4, documents=[])
        validation = {key: population[key].clone() for key in ("starts", "target_counts", "document_indices")}
        validation["starts"][0] = 99
        validation["sequence_nll"] = torch.ones(4)
        with self.assertRaisesRegex(ValueError, "frozen population"):
            report.reconstruct_evaluation(validation, {}, population)

    def test_all_steps_evaluation_cadence_and_partial_final_batch(self):
        plan = dict(steps=327, batch_tokens=100992, total_tokens=32939904)
        evaluations = {0, 1, 327, *range(8, 328, 8)}
        rows = [dict(step=step, tokens=min(step * 100992, 32939904),
                     **(dict(validation_nll=3.) if step in evaluations else {})) for step in range(328)]
        curve = report.validate_trajectory(rows, plan)
        self.assertEqual(len(curve), 43)
        self.assertEqual(rows[-1]["tokens"] - rows[-2]["tokens"], 16512)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            report.validate_trajectory(rows + [rows[-1]], plan)
        rows[8].pop("validation_nll")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            report.validate_trajectory(rows, plan)

    def test_primary_uses_each_fresh_pair_and_ignores_selection_seed(self):
        plan = dict(seeds=[1, 2], primary_threshold=-.005, late_steps=[232])
        runs = {}
        for seed, spd in ((0, .5), (1, 1.970), (2, 1.998)):
            for method, loss in (("ts", 2.), ("spd", spd)):
                runs[method, seed] = dict(full_development_nll=loss, curve={232: loss},
                                          groups={i:dict(nll=loss) for i in range(4)})
        pairs = report.paired_results(plan, runs)
        self.assertEqual(set(pairs), {1, 2})
        self.assertTrue(pairs[1]["primary_pass"])
        self.assertFalse(pairs[2]["primary_pass"])
        self.assertLess(sum(row["spd_minus_ts"] for row in pairs.values()) / 2, -.005)
        self.assertFalse(all(row["primary_pass"] for row in pairs.values()))


if __name__ == "__main__":
    unittest.main()

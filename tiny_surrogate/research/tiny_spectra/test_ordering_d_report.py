"""Synthetic ordering rules; no real data, model inference or jobs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import ordering_d_report as report
from . import ordering_d


def rows():
    result=[]
    for i,method in enumerate(report.METHODS):
        for lr in (1.,2.,4.):
            for seed in report.SEEDS:
                result.append(dict(method=method,seed=seed,lr=lr,status="complete",
                    run_id=f"{method}_{lr}_{seed}",full_development_nll=3.-.1*i+(0 if lr==2 else .2)))
    return result


class OrderingContracts(unittest.TestCase):
    def test_joint_lr_does_not_choose_separately_per_seed(self):
        values=rows()
        losses={1.:(1.,3.),2.:(2.1,1.1),4.:(4.,2.)}
        for row in values:
            if row["method"]=="muon":row["full_development_nll"]=losses[row["lr"]][report.SEEDS.index(row["seed"])]
        chosen=report.select_rates(values)["muon"]["selected"]
        self.assertEqual(chosen["lr"],2.)
        self.assertEqual(chosen["by_seed"][str(report.SEEDS[0])],2.1)

    def test_negative_mean_cannot_hide_a_seed_reversal(self):
        values=rows()
        for row in values:
            if row["method"]=="spd" and row["lr"]==2.:
                row["full_development_nll"] = 2.65 if row["seed"]==report.SEEDS[0] else 2.71
        pairs,passed=report.ordering_gates(report.select_rates(values))
        self.assertTrue(pairs[-1]["mean_advantage_at_least_005"])
        self.assertFalse(pairs[-1]["negative_in_both_seeds"])
        self.assertFalse(passed)

    def test_tiny_ordered_gaps_do_not_pass_material_advantage(self):
        values=rows()
        for row in values:
            row["full_development_nll"]=3.-.001*report.METHODS.index(row["method"])+(0 if row["lr"]==2. else .2)
        pairs,passed=report.ordering_gates(report.select_rates(values))
        self.assertTrue(all(p["negative_in_both_seeds"] for p in pairs))
        self.assertFalse(passed)

    def test_exact_boundary_ties_remain_open(self):
        values=rows()
        for row in values:
            if row["method"]=="muon" and row["lr"]==1.:row["full_development_nll"]=2.9
        item=report.select_rates(values)["muon"]
        self.assertFalse(item["bracket_closed"])
        self.assertEqual(item["proposed_single_outward_lr"],.5)
        for row in values:
            if row["method"]=="muon" and row["lr"]==4.:row["full_development_nll"]=2.9
        item=report.select_rates(values)["muon"]
        self.assertTrue(item["ambiguous_both_boundaries"])
        self.assertIsNone(item["proposed_single_outward_lr"])

    def test_unstable_pair_cannot_win_from_its_favorable_seed(self):
        values=rows()
        for row in values:
            if row["method"]=="muon" and row["lr"]==1.:
                if row["seed"]==report.SEEDS[0]:row["full_development_nll"]=-100.
                else:row.update(status="numerical_instability",full_development_nll=None)
        item=report.select_rates(values)["muon"]
        self.assertEqual(item["selected"]["lr"],2.)
        self.assertFalse(item["candidates"][0]["stable_both_seeds"])

    def test_missing_seed_is_not_an_eligible_recipe(self):
        with self.assertRaises(ValueError):report.select_rates(rows()[:-1])

    def test_all_pairs_and_brackets_required_for_success(self):
        choices=report.select_rates(rows())
        self.assertTrue(report.ordering_gates(choices)[1])
        choices["pd"]["bracket_closed"]=False
        self.assertFalse(report.ordering_gates(choices)[1])

    def test_incomplete_family_does_not_open_population_or_scores(self):
        with tempfile.TemporaryDirectory(prefix="ordering-report-") as tmp:
            root=Path(tmp);(root/"configs").mkdir();(root/"runs").mkdir()
            tasks=[]
            for i,row in enumerate(rows()):
                cfg=root/"configs"/f"{i}.json";cfg.write_text(json.dumps(row))
                tasks.append([dict(config=str(cfg))])
                if i<29:
                    out=root/"runs"/row["run_id"];out.mkdir()
                    (out/"ARM_EXECUTION.json").write_text('{"status":"complete"}')
            (root/"PLAN.json").write_text('{}')
            (root/"tasks_initial.json").write_text(json.dumps(tasks))
            with patch.object(report,"verify"),patch.object(report,"population") as population:
                with self.assertRaises(report.NotReady):report.report(root,"initial")
                population.assert_not_called()
            self.assertFalse((root/"report_initial").exists())

    def test_edge_stage_is_exactly_paired_and_passes_worker_identity_checks(self):
        with tempfile.TemporaryDirectory(prefix="ordering-edges-") as tmp:
            root=Path(tmp)
            for name in ("configs","runs","frozen","report_initial"):(root/name).mkdir()
            (root/"PLAN.json").write_text('{"worker_process_seconds":3540}')
            (root/"ordering_driver.py").write_text('# synthetic controller identity\n')
            (root/"frozen/native.py").write_text('# synthetic numerical identity\n')
            (root/"source_manifest.json").write_text(json.dumps({'native.py':ordering_d.sha(root/'frozen/native.py')}))
            tasks=[];hashes={};values=rows()
            for row in values:
                if row["method"]=="adamw" and row["lr"]==4.:row["full_development_nll"]=2.8
                cfg={k:row[k] for k in ("method","seed","lr","run_id")}
                path=root/"configs"/(cfg["run_id"]+".json");path.write_text(json.dumps(cfg))
                hashes[path.name]=ordering_d.sha(path)
                tasks.append([dict(config=str(path))])
                arm=root/"runs"/cfg["run_id"];arm.mkdir()
                (arm/"ARM_EXECUTION.json").write_text('{"status":"complete"}')
                (arm/"config.json").write_text(json.dumps(cfg))
                row["artifact_sha256"]={"config.json":ordering_d.sha(arm/"config.json")}
            (root/"tasks_initial.json").write_text(json.dumps(tasks))
            (root/"PREFLIGHT.json").write_text(json.dumps(dict(plan_sha256=ordering_d.sha(root/"PLAN.json"),
                source_manifest_sha256=ordering_d.sha(root/"source_manifest.json"),
                controller_sha256=ordering_d.sha(root/"ordering_driver.py"),
                tasks_initial_sha256=ordering_d.sha(root/"tasks_initial.json"),initial_config_sha256=hashes)))
            initial=dict(stage="initial",rows=values,plan_sha256=ordering_d.sha(root/"PLAN.json"),
                source_manifest_sha256=ordering_d.sha(root/"source_manifest.json"),boundary_requests={"adamw":8.},
                ordering_viability_passed=False)
            (root/"report_initial/results.json").write_text(json.dumps(initial))
            result=report.prepare_edges(root)
            self.assertEqual(result["arms"],2)
            ordering_d.verify(root,"edges")
            edges=[ordering_d.read(a["config"]) for group in ordering_d.read(root/"tasks_edges.json") for a in group]
            self.assertEqual({c["seed"] for c in edges},set(report.SEEDS))
            self.assertEqual({c["lr"] for c in edges},{8.})
            with self.assertRaises(ValueError):report.prepare_edges(root)


if __name__=="__main__":unittest.main()

"""Synthetic-only Candidate-D qualification contracts; no GPU or real corpus.

All fixtures and receipts live in TemporaryDirectory under ./run's study-local
TMPDIR. Neither the real development data nor any sealed panel is accessed.
"""
from dataclasses import asdict
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from . import qualify_d as qd
from . import cohort as cohort_driver
from .model import ModelConfig


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def recurrence(decay, count):
    """Independent explicit FP32 EMA recurrence, not its real-valued formula."""
    value = torch.tensor(0., dtype=torch.float32)
    for _ in range(count):
        value.mul_(decay).add_(1-decay)
    return value


def synthetic_plan():
    return dict(candidate="reference_directed_D", qualification=dict(updates_each=21, maximum_minutes=20),
        screen_gpu_hour_cap=8., development_seeds=[20261001, 20261002],
        screen_lr_centers=dict(adamw=.0012, muon=.01, pd=.01, ts=.005, spd=.005),
        aux_lr=.002, midpoint_momentum=.9, minimum_fresh_target_capacity=192484352,
        base_batch_tokens=262144, base_total_tokens=96242176,
        model=dict(vocab_size=12576, n_layer=8, n_embd=128, n_head=2, seq_len=512,
                   bias=False, norm="rmsnorm", qk_norm=True),
        microbatch_sequences=4, input_covariance=dict(clock="per_training_microforward", ema=.998, stride=32,
                   gram_precision="FP32"), input_mean_ema=.99, output_statistics=dict(sequences=2, seq_len=512, ema=.8),
        refresh_by_batch={"65536":10,"262144":10,"1048576":2}, geometry_powers=dict(alpha=.25,out_beta=.25),
        soap_beta2=.9, decay=.01, grad_clip=1., cooldown_fraction=.1, warmup_fraction=.034,
        aux_betas=[.9,.95])


class QualificationDContracts(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="qualification-d-contract-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.data = self.root / "synthetic_data"
        self.preparation = self.root / "synthetic_preparation"
        self.data.mkdir()
        self.preparation.mkdir()
        self.plan_path = self.preparation / "PLAN.json"
        write(self.plan_path, synthetic_plan())

    def assert_preparation_rejected_before_load_or_create(self):
        out = self.root / "must_not_create"
        with patch.object(qd, "load_training_corpus") as loader, patch.object(qd, "create") as creator:
            with self.assertRaises(ValueError):
                qd.make_smoke(self.data, self.plan_path, self.preparation, out)
            loader.assert_not_called()
            creator.assert_not_called()
        self.assertFalse(out.exists())

    def receipts(self, *, exit_code=0, infer=False):
        completion = dict(no_model_inference=not infer, training_manifest_sha256="synthetic-training",
                          test_manifest_sha256="synthetic-test")
        write(self.data / "PREPARATION_COMPLETE.json", completion)
        write(self.data / "MANIFESTS_PREPARED.json", completion)
        write(self.preparation / "EXECUTION_COMPLETE.json", dict(result=completion, model_inference=False, gpu_requested=False))
        write(self.preparation / "PROCESS_EXIT.json", dict(exit_code=exit_code))
        return completion

    def test_unfinished_preparation_rejects_before_any_corpus_load(self):
        self.assert_preparation_rejected_before_load_or_create()
        self.receipts()
        (self.preparation / "PROCESS_EXIT.json").unlink()
        self.assert_preparation_rejected_before_load_or_create()

    def test_preserved_failure_vetoes_even_apparent_success_receipts(self):
        self.receipts()
        for path in (self.data / "PREPARATION_FAILED.json", self.preparation / "EXECUTION_FAILURE.json"):
            with self.subTest(path=path.name):
                write(path, dict(error="synthetic preparation failure"))
                self.assert_preparation_rejected_before_load_or_create()
                path.unlink()

    def test_nonzero_exit_inference_or_disagreeing_receipts_reject_before_load(self):
        self.receipts(exit_code=7)
        self.assert_preparation_rejected_before_load_or_create()
        self.receipts(infer=True)
        self.assert_preparation_rejected_before_load_or_create()
        self.receipts()
        write(self.preparation / "EXECUTION_COMPLETE.json", dict(result=dict(no_model_inference=True)))
        self.assert_preparation_rejected_before_load_or_create()

    def fixture(self):
        """Build five saved synthetic arms through the real config constructor."""
        plan = synthetic_plan()
        manifest = dict(corpus_kind=qd.KIND, role="training_and_development_only", vocab_size=12576,
                        seq_len=512, dev_full_windows=2, dev_target_count=3,
                        fresh_target_capacity=192484352)
        write(self.data / "training_manifest.json", manifest)
        completion = dict(training_manifest_sha256=qd.digest(self.data / "training_manifest.json"))
        write(self.data / "PREPARATION_COMPLETE.json", completion)
        self.population = dict(starts=torch.tensor([0,3]), target_counts=torch.tensor([2,1]),
                               document_indices=torch.tensor([0,1]))
        self.corpus = Mock()
        self.corpus.manifest = manifest
        self.corpus.evaluation_starts.return_value = self.population["starts"].clone()
        self.corpus.evaluation_lengths.return_value = self.population["target_counts"].clone()
        self.corpus.evaluation_document_indices.return_value = self.population["document_indices"].clone()

        def fake_create(out, kind, configs):
            cohort = Path(out)
            cohort.mkdir()
            write(cohort / "cohort.json", dict(kind=kind, task_groups=1, max_parallel=1,
                                               gpu_gres="gpu:nvidia_rtx_a4000:1"))
            for cfg in configs:
                write(cohort / "configs" / (cfg["run_id"] + ".json"), cfg)
            sources = {}
            for package, names in (("tiny_spectra", ("train", "model", "optim", "data", "fineweb", "stories")),
                                   ("adamw_spectra", ("model", "muon", "data_norm_muon"))):
                for name in names:
                    relative = f"research/{package}/{name}.py"
                    path = cohort / "frozen" / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("# synthetic qualification source\n")
                    sources[relative] = qd.digest(path)
            write(cohort / "source_manifest.json", sources)
            return cohort

        with patch.object(qd, "prepared_data", return_value=(plan, manifest, completion)), \
                patch.object(qd, "create", side_effect=fake_create):
            cohort = qd.make_smoke(self.data, self.plan_path, self.preparation, self.root / "cohort")
        self.cohort = cohort
        self.runs = {}
        cov_weight, mean_weight = recurrence(.998,2688), recurrence(.99,2688)
        modules = [f"blocks.{block}.{module}" for block in range(8)
                   for module in ("attn.q","attn.k","attn.v","attn.o","mlp.up","mlp.down")]
        hardware = dict(name="NVIDIA RTX A4000", total_memory_bytes=16*1024**3)
        for config_path in (cohort / "configs").glob("*.json"):
            cfg = qd.read(config_path)
            root = cohort / "runs" / cfg["run_id"]
            root.mkdir(parents=True)
            self.runs[cfg["method"]] = root
            geometry = cfg["method"] in ("pd","ts","spd")
            mc = asdict(ModelConfig(vocab_size=12576, n_layer=8, n_embd=128, n_head=2, seq_len=512,
                track_input_stats=geometry, track_input_cov=geometry, stats_clock="microforward",
                stats_decay=.99, cov_decay=.998, cov_stride=32))
            metadata = dict(model_config=mc, n_parameters=4861056, hardware=hardware,
                source_file=str(cohort / "frozen/research/tiny_spectra/train.py"), corpus=manifest,
                covariance_clock="per training microforward", covariance_gram_precision="FP32; autocast disabled",
                initial_parameter_sha256="paired-initial", train_window_sha256="paired-train",
                validation_bank_sha256="paired-validation", validation_bank_lengths_sha256="paired-lengths",
                data_manifest_sha256=completion["training_manifest_sha256"])
            summary = dict(status="complete", steps=21, tokens=21*262144, n_parameters=4861056,
                hardware=hardware, full_validation_nll=2., final_validation_nll=2., train_probe_nll=2.,
                training_seconds=21., total_seconds=25., peak_cuda_allocated_bytes=512*1024**2,
                initial_parameter_sha256=metadata["initial_parameter_sha256"],
                train_window_sha256=metadata["train_window_sha256"], full_development_target_count=3)
            for name,value in (("config.json",cfg),("summary.json",summary),("metadata.json",metadata),
                               ("status.json",dict(status="complete"))):
                write(root / name,value)
            saved = dict(self.population, sequence_nll=torch.tensor([1.,4.],dtype=torch.float64))
            torch.save(saved,root / "final_validation.pt")
            state = dict(_total_forwards=torch.tensor(2688),_step_forwards=torch.tensor(128),
                         input_cov_weight=cov_weight,input_weight=mean_weight)
            statistics = {name:dict(state) for name in modules} if geometry else {}
            roots = {name+".weight":(torch.eye(1),21) for name in modules} if geometry else {}
            external = dict(data_norm=dict(roots=roots,out_stats={name+".weight":torch.eye(1) for name in modules}
                            if cfg["method"]=="ts" else {}),
                            soap=dict(states={name+".weight":dict(v=torch.ones(1)) for name in modules}
                                      if cfg["method"]=="spd" else {}))
            snapshot = dict(config=cfg,metadata=metadata,model=dict(synthetic_weight=torch.ones(1)),
                            optimizer=dict(config=cfg,model_statistics=statistics,external=external))
            torch.save(snapshot,root / "final.pt")
            rows=[]
            for step in range(22):
                row=dict(step=step,tokens=step*262144,elapsed_seconds=float(step),
                         train_nll=2.,step_seconds=1.,gradient_norm_before_clip=.1,lr=cfg["lr"])
                if step in (0,1,21):
                    row.update(validation_nll=2.,train_probe_nll=2.)
                rows.append(row)
            (root / "metrics.jsonl").write_text("".join(json.dumps(row)+"\n" for row in rows))
        return cohort

    def report(self):
        out=self.root / "report"
        with patch.object(qd,"load_training_corpus",return_value=self.corpus):
            result=qd.evaluate_qualification(self.cohort,out)
        return result

    def mutate_json(self,method,name,**changes):
        path=self.runs[method]/name
        value=qd.read(path)
        value.update(changes)
        write(path,value)

    def test_report_uses_real_target_weights_and_requires_three_root_refreshes(self):
        self.fixture()
        result=self.report()
        self.assertTrue(result["numerical_qualification_passed"])
        self.assertFalse(result["scientific_ranking_claim"])
        self.assertFalse(result["test_scored"])
        self.assertEqual(len(result["runs"]),5)
        self.assertEqual({r["full_development_nll"] for r in result["runs"]},{2.})
        self.assertNotEqual(result["runs"][0]["full_development_nll"],2.5)

    def test_report_rejects_missing_or_failed_arm_before_loading_checkpoints(self):
        self.fixture()
        for status in ("running","failed"):
            with self.subTest(status=status):
                self.mutate_json("spd","status.json",status=status)
                with patch.object(qd.torch,"load") as loader:
                    with self.assertRaises(ValueError): self.report()
                    loader.assert_not_called()
                self.assertFalse((self.root / "report").exists())

    def test_report_rejects_missing_method(self):
        self.fixture()
        config=self.cohort / "configs" / (qd.read(self.runs["spd"] / "config.json")["run_id"]+".json")
        config.unlink()
        with self.assertRaises(ValueError): self.report()

    def test_report_rejects_pairing_and_saved_config_mismatches(self):
        self.fixture()
        self.mutate_json("spd","metadata.json",train_window_sha256="unpaired")
        with self.assertRaises(ValueError): self.report()
        self.mutate_json("spd","metadata.json",train_window_sha256="paired-train")
        cfg=qd.read(self.runs["spd"] / "config.json")
        self.mutate_json("spd","config.json",momentum=cfg["momentum"]-.1)
        with self.assertRaises(ValueError): self.report()

    def test_report_rejects_nonfinite_saved_loss_and_checkpoint(self):
        self.fixture()
        path=self.runs["spd"] / "final_validation.pt"
        saved=torch.load(path,weights_only=True)
        saved["sequence_nll"][0]=float("nan")
        torch.save(saved,path)
        with self.assertRaises(ValueError): self.report()
        saved["sequence_nll"][0]=1.
        torch.save(saved,path)
        checkpoint=self.runs["spd"] / "final.pt"
        state=torch.load(checkpoint,weights_only=True)
        state["model"]["synthetic_weight"][0]=float("inf")
        torch.save(state,checkpoint)
        with self.assertRaises(ValueError): self.report()

    def test_report_rejects_wrong_population_clock_and_root_step(self):
        self.fixture()
        path=self.runs["ts"] / "final_validation.pt"
        saved=torch.load(path,weights_only=True)
        saved["document_indices"]=torch.tensor([1,0])
        torch.save(saved,path)
        with self.assertRaises(ValueError): self.report()
        saved["document_indices"]=self.population["document_indices"].clone()
        torch.save(saved,path)
        checkpoint=self.runs["ts"] / "final.pt"
        snapshot=torch.load(checkpoint,weights_only=True)
        first=next(iter(snapshot["optimizer"]["model_statistics"].values()))
        first["_total_forwards"]=torch.tensor(2689)
        torch.save(snapshot,checkpoint)
        with self.assertRaises(ValueError): self.report()
        first["_total_forwards"]=torch.tensor(2688)
        roots=snapshot["optimizer"]["external"]["data_norm"]["roots"]
        key=next(iter(roots))
        roots[key]=(roots[key][0],11)
        torch.save(snapshot,checkpoint)
        with self.assertRaises(ValueError): self.report()

    def test_report_rejects_nonfinite_metrics_and_failed_summary(self):
        self.fixture()
        path=self.runs["spd"] / "metrics.jsonl"
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[3]["train_nll"]=float("nan")
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        with self.assertRaises(ValueError): self.report()
        rows[3]["train_nll"]=2.
        path.write_text("".join(json.dumps(row)+"\n" for row in rows))
        self.mutate_json("spd","summary.json",status="failed")
        with self.assertRaises(ValueError): self.report()

    def test_clock_check_rejects_analytic_weight_in_place_of_fp32_recurrence(self):
        self.fixture()
        checkpoint=self.runs["pd"] / "final.pt"
        snapshot=torch.load(checkpoint,weights_only=True)
        first=next(iter(snapshot["optimizer"]["model_statistics"].values()))
        first["input_weight"]=torch.tensor(1-.99**2688)
        torch.save(snapshot,checkpoint)
        with self.assertRaisesRegex(ValueError,"clock|exclusion"):
            self.report()

    def test_report_checks_pinned_population_even_if_all_methods_share_wrong_ids(self):
        self.fixture()
        for root in self.runs.values():
            path=root / "final_validation.pt"
            saved=torch.load(path,weights_only=True)
            saved["document_indices"]=torch.tensor([1,0])
            torch.save(saved,path)
        with self.assertRaisesRegex(ValueError,"all development targets"):
            self.report()

    def test_report_rejects_changed_frozen_source_and_preserves_existing_output(self):
        self.fixture()
        path=self.cohort / "frozen/research/tiny_spectra/fineweb.py"
        path.write_text("# modified synthetic execution source\n")
        with self.assertRaisesRegex(ValueError,"source changed"):
            self.report()
        out=self.root / "report"
        out.mkdir()
        sentinel=out / "keep.txt"
        sentinel.write_text("preserve previous report")
        with self.assertRaises(FileExistsError): self.report()
        self.assertEqual(sentinel.read_text(),"preserve previous report")

    def test_candidate_d_mocked_submission_keeps_declared_twenty_minutes(self):
        self.fixture()
        # This mock surrounds every submit invocation; no scheduler command can
        # execute even if the implementation changes while this test runs.
        with patch.object(cohort_driver.subprocess,"run",return_value=Mock(stdout="synthetic-job\n")) as submission, \
                contextlib.redirect_stdout(io.StringIO()):
            cohort_driver.submit(self.cohort)
            command=submission.call_args.args[0]
            self.assertIn("--time=00:20:00",command)
            self.assertIn("--gres=gpu:nvidia_rtx_a4000:1",command)
            self.assertIn("--array=0-0%1",command)
            self.assertEqual(qd.read(self.cohort / "submission.json")["job_id"],"synthetic-job")
            with self.assertRaises(FileExistsError):
                cohort_driver.submit(self.cohort)
            submission.assert_called_once()


if __name__=="__main__":
    unittest.main()

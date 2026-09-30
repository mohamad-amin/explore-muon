"""Synthetic-only confirmation contracts. Never load or score the real panel."""
from dataclasses import asdict
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from . import confirm
from .heldout import _write_panel
from .model import GPT, ModelConfig
from .train import save_model_snapshot


class ConfirmationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.panel = _write_panel(self.root / "synthetic_panel",
                                 [dict(slug="one", text="ab" * 6 + "a"),
                                  dict(slug="two", text="a" * 9),
                                  dict(slug="three", text="ba" * 14 + "b")],
                                 ("a", "b"), dict(synthetic=True), seq_len=4,
                                 curve_count=2, role="synthetic_contract_test")
        self.data = self.root / "train.txt"
        self.data.write_text("ababababa")
        source_root = self.root / "frozen"
        project = Path(confirm.__file__).resolve().parents[2]
        manifest = {}
        for relative in confirm.TRAINING_SOURCES:
            target = source_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)
            manifest[relative] = confirm.digest(target)
        source_manifest = self.root / "source_manifest.json"
        confirm.write_new(source_manifest, manifest)
        self.runs = []
        for name in ("first", "second"):
            cfg = dict(run_id=name, keep_model_every=2, total_tokens=16, batch_tokens=8,
                       seed=13, method="muon", momentum=.9, lr=.01,
                       n_layer=1, n_embd=4, n_head=1, seq_len=4,
                       data_path=str(self.data), data_sha256=confirm.digest(self.data))
            config_path = self.root / (name + ".json")
            confirm.write_new(config_path, cfg)
            self.runs.append(dict(run_id=name, run_dir=str(self.root / "runs" / name),
                                  config_path=str(config_path), source_root=str(source_root),
                                  source_manifest=str(source_manifest), checkpoint_steps=[0, 1, 2]))
        self.plan_path = self.root / "plan.json"
        self.lock_path = self.root / "lock.json"
        self.plan = dict(panel=dict(path=str(self.panel.path), sha256=self.panel.manifest_sha256),
                         metrics=dict(primary="character_weighted_nll", secondary="macro_nll"),
                         criteria=dict(keep="all outcomes"), runs=self.runs)
        confirm.write_new(self.plan_path, self.plan)

    def lock(self):
        return confirm.create_lock(self.plan_path, self.lock_path)

    def finish(self, index):
        run = self.runs[index]
        root = Path(run["run_dir"])
        root.mkdir(parents=True)
        cfg = confirm.read(run["config_path"])
        confirm.write_new(root / "config.json", cfg)
        mc = ModelConfig(vocab_size=2, n_layer=1, n_embd=4, n_head=1, seq_len=4)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(123)
            model = GPT(mc)
        confirm.write_new(root / "metadata.json", dict(model_config=asdict(mc),
                          corpus=dict(vocabulary=["a", "b"]),
                          source_file=str(Path(run["source_root"]) / "research/tiny_spectra/train.py")))
        for step in (0, 1, 2):
            save_model_snapshot(model, mc, root / "models" / f"step{step:06d}.pt", step=step, tokens=step * 8)
        for name in ("summary.json", "status.json"):
            confirm.write_new(root / name, dict(status="complete", steps=2, tokens=16))

    def finish_all(self):
        self.lock()
        self.finish(0)
        self.finish(1)

    def score(self, name="first", attempt="first"):
        return confirm.score(self.lock_path, name, self.root / "scores" / attempt, device="cpu")

    def test_lock_before_training_and_no_overwrite(self):
        Path(self.runs[0]["run_dir"]).mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, "precede"):
            self.lock()
        self.assertFalse(self.lock_path.exists())
        Path(self.runs[0]["run_dir"]).rmdir()
        self.lock()
        with self.assertRaises(FileExistsError):
            self.lock()

    def test_lock_rejects_partial_checkpoint_family(self):
        self.plan["runs"][0]["checkpoint_steps"] = [0, 2]
        self.plan_path.write_bytes(confirm.canonical(self.plan))
        with self.assertRaisesRegex(ValueError, "entire saved checkpoint family"):
            self.lock()

    def test_early_scoring_refused_before_inference(self):
        self.lock()
        self.finish(0)
        with patch.object(confirm, "evaluate_document") as inference:
            with self.assertRaisesRegex(ValueError, "Whole training family"):
                self.score()
            inference.assert_not_called()
        self.assertFalse((self.root / "scores").exists())

    def test_bad_other_run_snapshot_refused_before_inference(self):
        self.finish_all()
        path = Path(self.runs[1]["run_dir"]) / "models/step000001.pt"
        saved = torch.load(path, weights_only=True)
        saved["tokens"] = 999
        torch.save(saved, path)
        with patch.object(confirm, "evaluate_document") as inference:
            with self.assertRaisesRegex(ValueError, "step/horizon"):
                self.score()
            inference.assert_not_called()

    def test_changed_config_refused(self):
        self.finish_all()
        path = Path(self.runs[0]["config_path"])
        cfg = confirm.read(path)
        cfg["momentum"] = .1
        path.write_bytes(confirm.canonical(cfg))
        with self.assertRaisesRegex(ValueError, "config/source"):
            self.score()

    def test_changed_panel_refused(self):
        self.finish_all()
        with (self.panel.path / "one.txt").open("a") as stream:
            stream.write("a")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.score()

    def test_changed_frozen_source_refused(self):
        self.finish_all()
        target = Path(self.runs[0]["source_root"]) / "research/tiny_spectra/train.py"
        with target.open("a") as stream:
            stream.write("\n# changed after lock\n")
        with self.assertRaisesRegex(ValueError, "Frozen source"):
            self.score()

    def test_real_panel_role_cannot_use_cpu(self):
        self.finish_all()
        # Keep every byte synthetic; only exercise the role/device guard.
        lock, panel = confirm.load_lock(self.lock_path)
        panel.manifest["role"] = "sealed_confirmation"
        with patch.object(confirm, "load_lock", return_value=(lock, panel)), patch.object(confirm, "complete_family", return_value={}):
            with self.assertRaisesRegex(ValueError, "CPU scoring"):
                self.score()

    def test_weighting_full_and_curve_scores_and_collection_gate(self):
        self.finish_all()
        result = self.score()
        self.assertEqual(result, dict(status="scored_and_sealed", run_id="first"))
        with self.assertRaisesRegex(ValueError, "Every locked run"):
            confirm.collect(self.lock_path, self.root / "scores", self.root / "released")
        self.assertFalse((self.root / "released").exists())
        saved = confirm.read(self.root / "scores/first/sealed_scores.json")
        row = saved["final"]
        weights = {doc.slug: len(doc.full_starts) * 4 for doc in self.panel.documents}
        self.assertEqual(weights, dict(one=12, two=8, three=28))
        manual = sum(row["per_play"][name] * weight for name, weight in weights.items()) / 48
        self.assertAlmostEqual(row["character_weighted_nll"], manual)
        self.assertAlmostEqual(row["macro_nll"], sum(row["per_play"].values()) / 3)
        for curve in saved["curves"]:
            expected = sum(curve["per_play"][name] * weight for name, weight in weights.items()) / 48
            self.assertAlmostEqual(curve["character_weighted_nll"], expected)
            self.assertEqual(curve["full_target_characters"], 48)
        run = confirm.read(self.lock_path)["payload"]["plan"]["runs"][0]
        path = Path(run["run_dir"]) / "models/step000002.pt"
        model = confirm.load_snapshot(run, 2, confirm.digest(path), torch.device("cpu"))
        for doc in self.panel.documents:
            expected = []
            for start in doc.full_starts:
                tokens = torch.tensor(doc.token_ids[int(start):int(start) + 5].copy()).long()[None]
                with torch.no_grad():
                    losses = F.cross_entropy(model(tokens[:, :-1]).flatten(0, 1), tokens[:, 1:].flatten(), reduction="none")
                expected.append(losses.double().mean().item())
            self.assertEqual(row["windows"][doc.slug]["starts"], doc.full_starts.tolist())
            torch.testing.assert_close(torch.tensor(row["windows"][doc.slug]["nll"]), torch.tensor(expected))
        self.score("second", "second")
        confirm.collect(self.lock_path, self.root / "scores", self.root / "released")
        released = confirm.read(self.root / "released/all_results.json")
        self.assertEqual([r["run_id"] for r in released["results"]], ["first", "second"])
        with self.assertRaisesRegex(ValueError, "Only failed attempts"):
            self.score("first", "second_attempt")

    def test_failed_attempt_preserved_and_identical_retry_allowed(self):
        self.finish_all()
        with patch.object(confirm, "evaluate_document", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                self.score()
        self.assertEqual(confirm.read(self.root / "scores/first/status.json")["status"], "failed")
        self.score("first", "retry")
        self.assertTrue((self.root / "scores/first/failure.json").exists())

    def test_retry_rejects_changed_family_weights(self):
        self.finish_all()
        with patch.object(confirm, "evaluate_document", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                self.score()
        path = Path(self.runs[1]["run_dir"]) / "models/step000002.pt"
        saved = torch.load(path, weights_only=True)
        saved["extra_metadata"] = "changes the frozen artifact identity"
        torch.save(saved, path)
        with self.assertRaisesRegex(ValueError, "identical lock/weights"):
            self.score("first", "changed_retry")


if __name__ == "__main__":
    unittest.main()

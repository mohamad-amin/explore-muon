"""Synthetic-only BPE confirmation contracts; never read a real sealed panel."""
from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from . import confirm, stories
from .model import GPT, ModelConfig
from .train import save_model_snapshot


class StoriesConfirmationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        data = self.root / "data"
        data.mkdir()

        def asset(name, raw):
            path = data / name
            path.write_bytes(raw)
            return dict(path=name, bytes=len(raw), sha256=confirm.digest(path))

        vocabulary = ["<eot>", "A", "B"]
        tokenizer = asset("tokenizer.json", confirm.canonical(dict(model=dict(type="BPE", vocab=dict(zip(vocabulary, range(3)))))))
        self.common = dict(corpus_kind=confirm.STORIES_KIND, evaluation_policy_version=2,
                           vocabulary=vocabulary, vocab_size=3, seq_len=4, eot_id=0,
                           tokenizer_sha256=tokenizer["sha256"], tokenizer_fit=dict(train_only=True, fixture=True),
                           membership_sha256="synthetic-document-membership", prefix_hashes=dict(train="synthetic", valid="synthetic"),
                           revision="synthetic-official-source")
        splits = {}
        for role, values in (("train", [1, 2] * 8 + [0]), ("val", [1, 2, 1, 2, 1, 2, 0])):
            tokens = np.asarray(values, dtype="<u2")
            splits[role] = dict(tokens=asset(role + ".u16", tokens.tobytes()),
                                documents=asset(role + ".documents.json", confirm.canonical([dict(start=0, end=len(tokens))])))
        self.data_manifest = dict(self.common, role="training_and_development_only", splits=splits, tokenizer=tokenizer,
                                  default_permutation=asset("permutation.u32", np.arange(4, dtype="<u4").tobytes()),
                                  dev_starts=asset("starts.i64", np.asarray([0, 4], dtype="<i8").tobytes()),
                                  dev_lengths=asset("lengths.i64", np.asarray([4, 2], dtype="<i8").tobytes()),
                                  dev_document_indices=asset("documents.i64", np.asarray([0, 0], dtype="<i8").tobytes()))
        self.data_path = data / "training_manifest_v2.json"
        confirm.write_new(self.data_path, self.data_manifest)
        documents = []
        for slug, ids in (("long", [1, 2, 1, 2, 1, 0]), ("short", [2, 0]), ("tail", [1, 1, 2, 0])):
            ids = np.asarray(ids, dtype="<u2")
            starts, lengths, _ = stories.evaluation_plan_all_targets([dict(start=0, end=len(ids))], 4)
            metadata = dict(counts=dict(full_target_tokens=int(lengths.sum()), curve_target_tokens=int(lengths.sum())))
            documents.append(stories.StoryTestDocument(slug, ids, starts, starts, metadata, lengths, lengths))
        panel_path = self.root / "synthetic_panel"
        panel_path.mkdir()
        panel_manifest = dict(self.common, role="synthetic_contract_test",
                              totals=dict(full_windows=4, curve_windows=4,
                                          full_target_tokens=9, curve_target_tokens=9))
        confirm.write_new(panel_path / "manifest.json", panel_manifest)
        self.panel = stories.StoryTestPanel(panel_path, panel_manifest,
                                            confirm.digest(panel_path / "manifest.json"), tuple(documents))
        # The real loader is never called. All panel bytes and models are fixtures.
        loader = patch.object(confirm.stories, "load_test_panel", return_value=self.panel)
        self.loader = loader.start()
        self.addCleanup(loader.stop)
        source_root = self.root / "frozen"
        project = Path(confirm.__file__).resolve().parents[2]
        source_manifest = {}
        for relative in confirm.TRAINING_SOURCES | {confirm.STORIES_SOURCE}:
            target = source_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, target)
            source_manifest[relative] = confirm.digest(target)
        manifest_path = self.root / "sources.json"
        confirm.write_new(manifest_path, source_manifest)
        self.runs = []
        for name in ("first", "second"):
            config = dict(run_id=name, keep_model_every=2, total_tokens=16, batch_tokens=8,
                          seed=31, method="muon", momentum=.9, lr=.01,
                          n_layer=1, n_embd=4, n_head=1, seq_len=4, corpus_kind=confirm.STORIES_KIND,
                          data_path=str(self.data_path), data_sha256=confirm.digest(self.data_path))
            config_path = self.root / (name + ".json")
            confirm.write_new(config_path, config)
            self.runs.append(dict(run_id=name, run_dir=str(self.root / "runs" / name),
                                  config_path=str(config_path), source_root=str(source_root),
                                  source_manifest=str(manifest_path), checkpoint_steps=[0, 1, 2]))
        self.plan = dict(panel=dict(path=str(self.panel.path), sha256=self.panel.manifest_sha256, kind=confirm.STORIES_KIND),
                         metrics=dict(primary="token_weighted_nll", secondary="macro_document_nll"),
                         criteria=dict(keep="all prespecified outcomes"), runs=self.runs)
        self.plan_path, self.lock_path = self.root / "plan.json", self.root / "lock.json"
        confirm.write_new(self.plan_path, self.plan)

    def lock(self):
        return confirm.create_lock(self.plan_path, self.lock_path)

    def finish(self, index):
        run = self.runs[index]
        root = Path(run["run_dir"])
        root.mkdir(parents=True)
        cfg = confirm.read(run["config_path"])
        confirm.write_new(root / "config.json", cfg)
        mc = ModelConfig(vocab_size=3, n_layer=1, n_embd=4, n_head=1, seq_len=4)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(432)
            model = GPT(mc)
        confirm.write_new(root / "metadata.json", dict(model_config=asdict(mc), corpus=self.data_manifest,
                          data_manifest_sha256=cfg["data_sha256"],
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

    def test_kind_version_role_and_pinned_hash_guards(self):
        spec = self.plan["panel"]
        with self.assertRaisesRegex(ValueError, "panel kind"):
            confirm.load_panel(dict(spec, kind="unknown"))
        with patch.object(confirm.heldout, "load_panel", return_value=self.panel):
            with self.assertRaisesRegex(ValueError, "Character panel kind"):
                confirm.load_panel(dict(spec, kind=confirm.CHARACTER_KIND))
        self.panel.manifest["evaluation_policy_version"] = 1
        with self.assertRaisesRegex(ValueError, "all-target v2"):
            confirm.load_panel(spec)
        self.panel.manifest["evaluation_policy_version"] = 2
        self.panel.manifest["role"] = "sealed_confirmation"
        with self.assertRaisesRegex(ValueError, "pinned external"):
            confirm.load_panel(spec)

    def test_lock_pins_bpe_sources_and_data_identity(self):
        lock = self.lock()
        plan = lock["payload"]["plan"]
        self.assertIn(str(Path(stories.__file__).resolve()), plan["evaluator_sources"])
        self.assertIn(confirm.STORIES_SOURCE, plan["runs"][0]["sources"])
        self.assertEqual(plan["runs"][0]["data_identity"]["tokenizer_sha256"], self.common["tokenizer_sha256"])
        self.assertEqual(plan["runs"][0]["data_identity"]["manifest_sha256"], confirm.digest(self.data_path))

    def test_missing_stories_training_source_rejected(self):
        path = Path(self.runs[0]["source_manifest"])
        manifest = confirm.read(path)
        del manifest[confirm.STORIES_SOURCE]
        path.write_bytes(confirm.canonical(manifest))
        with self.assertRaisesRegex(ValueError, "training implementation"):
            self.lock()

    def test_wrong_tokenizer_or_metric_rejected_at_lock(self):
        self.panel.manifest["tokenizer_sha256"] = "another-tokenizer"
        with self.assertRaisesRegex(ValueError, "data identity differs: tokenizer_sha256"):
            self.lock()
        self.panel.manifest["tokenizer_sha256"] = self.common["tokenizer_sha256"]
        self.plan["metrics"]["primary"] = "character_weighted_nll"
        self.plan_path.write_bytes(confirm.canonical(self.plan))
        with self.assertRaisesRegex(ValueError, "token_weighted_nll"):
            self.lock()

    def test_changed_tokenizer_asset_rejected_before_inference(self):
        self.finish_all()
        with (self.data_path.parent / "tokenizer.json").open("ab") as stream:
            stream.write(b" ")
        with patch.object(confirm, "evaluate_stories_panel") as inference:
            with self.assertRaisesRegex(ValueError, "artifact integrity"):
                self.score()
            inference.assert_not_called()

    def test_changed_metadata_data_identity_rejected_before_inference(self):
        self.finish_all()
        path = Path(self.runs[1]["run_dir"]) / "metadata.json"
        metadata = confirm.read(path)
        metadata["corpus"]["membership_sha256"] = "different-membership"
        path.write_bytes(confirm.canonical(metadata))
        with patch.object(confirm, "evaluate_stories_panel") as inference:
            with self.assertRaisesRegex(ValueError, "metadata/data identity"):
                self.score()
            inference.assert_not_called()

    def test_bpe_family_and_source_gates_still_precede_inference(self):
        self.lock()
        self.finish(0)
        with patch.object(confirm, "evaluate_stories_panel") as inference:
            with self.assertRaisesRegex(ValueError, "Whole training family"):
                self.score()
            inference.assert_not_called()
        self.finish(1)
        source = Path(self.runs[1]["source_root"]) / confirm.STORIES_SOURCE
        with source.open("a") as stream:
            stream.write("\n# invalid changed source\n")
        with patch.object(confirm, "evaluate_stories_panel") as inference:
            with self.assertRaisesRegex(ValueError, "Frozen source"):
                self.score()
            inference.assert_not_called()

    def test_variable_window_actual_cross_entropy_preserves_eos(self):
        class TableModel(torch.nn.Module):
            def forward(self, x):
                return torch.tensor([[-7., 7., 0.], [0., 1., 3.], [4., -1., 0.]])[x]
        model = TableModel()
        all_expected, means = [], {}
        for document in self.panel.documents:
            expected = []
            for start, length in zip(document.full_starts, document.full_lengths):
                tokens = torch.tensor(document.token_ids[int(start):int(start + length) + 1].copy()).long()[None]
                losses = F.cross_entropy(model(tokens[:, :-1]).flatten(0, 1), tokens[:, 1:].flatten(), reduction="none").double()
                all_expected.extend(losses.tolist())
                expected.append(losses.mean().item())
            mean, row = confirm.evaluate_bank(model, document, "full", self.panel, torch.device("cpu"))
            self.assertEqual(row["lengths"], document.full_lengths.tolist())
            torch.testing.assert_close(torch.tensor(row["nll"]), torch.tensor(expected), rtol=0, atol=0)
            means[document.slug] = mean
        result = confirm.aggregate(means, self.panel)
        self.assertAlmostEqual(result["token_weighted_nll"], sum(all_expected) / 9)
        self.assertEqual(result["full_target_tokens"], 9)
        self.assertEqual(result["scored_documents"], 3)
        # The short document contributes its sole EOS target; ignoring it changes the answer.
        self.assertAlmostEqual(means["short"], F.cross_entropy(torch.tensor([[4., -1., 0.]]), torch.tensor([0])).item())
        self.assertNotAlmostEqual(result["token_weighted_nll"], sum(means.values()) / 3)

    def test_bpe_scoring_names_weights_and_whole_family_release(self):
        self.finish_all()
        self.score()
        with self.assertRaisesRegex(ValueError, "Every locked run"):
            confirm.collect(self.lock_path, self.root / "scores", self.root / "released")
        result = confirm.read(self.root / "scores/first/sealed_scores.json")
        text = confirm.canonical(result).decode()
        self.assertNotIn("per_play", text)
        self.assertNotIn("character_weighted", text)
        self.assertNotIn("target_characters", text)
        weights = {doc.slug: len(doc.token_ids)-1 for doc in self.panel.documents}
        self.assertEqual(weights, dict(long=5, short=1, tail=3))
        for row in [result["final"], *result["curves"]]:
            expected = sum(row["per_document"][name] * count for name, count in weights.items()) / 9
            self.assertAlmostEqual(row["token_weighted_nll"], expected)
            self.assertEqual(row["full_target_tokens"], 9)
        self.assertEqual(result["final"]["windows"]["long"]["lengths"], [4, 1])
        self.assertEqual(result["panel_denominators"]["full_target_tokens"], 9)
        self.score("second", "second")
        confirm.collect(self.lock_path, self.root / "scores", self.root / "released")
        self.assertEqual(len(confirm.read(self.root / "released/all_results.json")["results"]), 2)
        with self.assertRaisesRegex(ValueError, "Only failed attempts"):
            self.score("first", "duplicate")

    def test_pooled_document_windows_are_batch_partition_invariant(self):
        class TableModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, x):
                self.calls += 1
                return torch.tensor([[-7., 7., 0.], [0., 1., 3.], [4., -1., 0.]])[x]
        reference = None
        for microbatch in (1, 2, 3, 32):
            model = TableModel()
            values, windows = confirm.evaluate_stories_panel(model, self.panel, torch.device("cpu"), microbatch)
            current = dict(per_document=values, windows=windows, **confirm.aggregate(values, self.panel))
            if reference is None:
                reference = current
            self.assertEqual(current, reference)
            self.assertEqual(model.calls, (4 + microbatch - 1) // microbatch)
        for doc in self.panel.documents:
            mean, row = confirm.evaluate_bank(TableModel(), doc, "full", self.panel, torch.device("cpu"))
            self.assertEqual(mean, reference["per_document"][doc.slug])
            self.assertEqual(row, reference["windows"][doc.slug])

    def test_pooled_window_cannot_borrow_context_from_next_story(self):
        self.panel.documents[0].full_lengths[-1] = 2  # Only one real target remains.
        with patch.object(confirm.stories, "padded_evaluation_batch") as batching:
            with self.assertRaisesRegex(ValueError, "document boundaries"):
                confirm.evaluate_stories_panel(torch.nn.Identity(), self.panel, torch.device("cpu"))
            batching.assert_not_called()

    def test_failed_bpe_attempt_preserves_identical_retry_gate(self):
        self.finish_all()
        with patch.object(confirm, "evaluate_stories_panel", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                self.score()
        self.score("first", "retry")
        self.assertTrue((self.root / "scores/first/failure.json").exists())


if __name__ == "__main__":
    unittest.main()

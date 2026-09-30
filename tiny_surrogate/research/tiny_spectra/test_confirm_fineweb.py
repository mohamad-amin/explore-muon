"""FineWeb confirmation contracts on entirely synthetic prepared artifacts."""
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import confirm, fineweb, fineweb_panel, stories
from . import test_confirm_stories as fixtures


class FineWebConfirmationContracts(unittest.TestCase):
    setUpClass = classmethod(fixtures.StoriesConfirmationContracts.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.StoriesConfirmationContracts.tearDownClass.__func__)
    lock = fixtures.StoriesConfirmationContracts.lock
    finish = fixtures.StoriesConfirmationContracts.finish
    finish_all = fixtures.StoriesConfirmationContracts.finish_all
    score = fixtures.StoriesConfirmationContracts.score

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data_root = self.root / "synthetic_data"
        self.data_root.mkdir()
        panel_root = self.data_root / "sealed_test"
        panel_root.mkdir()

        def write(path, value):
            confirm.write_new(path, value)
            return dict(path=path.name, bytes=path.stat().st_size, sha256=confirm.digest(path))

        def binary(path, values, dtype):
            path.write_bytes(np.asarray(values, dtype=dtype).tobytes())
            return dict(path=path.name, bytes=path.stat().st_size, sha256=confirm.digest(path))

        values = dict(train=[[1, 2] * 8 + [0]], dev=[[1, 2, 1, 2, 1, 2, 0]],
                      test=[[1, 2, 1, 2, 1, 0], [2, 0], [1, 1, 2, 0]])
        splits, offsets, role_assets = {}, {}, {}
        for role, documents in values.items():
            directory = panel_root if role == "test" else self.data_root
            rows, accepted, flat, cursor = [], [], [], 0
            for index, ids in enumerate(documents):
                identity = hashlib.sha256(f"synthetic {role} {index}".encode()).hexdigest()
                rows.append(dict(identity=identity, start=cursor, end=cursor + len(ids),
                                 token_ids_sha256=hashlib.sha256(np.asarray(ids, dtype="<u2").tobytes()).hexdigest()))
                accepted.append(dict(identity=identity, role=role, text=f"synthetic {role} {index}"))
                flat.extend(ids)
                cursor += len(ids)
            splits[role] = dict(tokens=binary(directory / (role + ".u16"), flat, "<u2"),
                                documents=write(directory / (role + ".documents.json"), rows),
                                token_count=len(flat), document_count=len(rows))
            offsets[role] = rows
            accepted_path = self.data_root / (role + ".accepted.jsonl")
            accepted_path.write_text("".join(json.dumps(row) + "\n" for row in accepted))
            role_assets[role] = dict(path=accepted_path.name, bytes=accepted_path.stat().st_size,
                                     sha256=confirm.digest(accepted_path))
        roles = dict(schema_version=1, corpus_kind=confirm.FINEWEB_KIND,
                     roles_frozen_before_tokenizer=True, roles=role_assets,
                     counts={role: len(rows) for role, rows in offsets.items()}, no_model_scores=True,
                     membership=write(self.data_root / "membership.jsonl", {"synthetic": True}),
                     overlap_exclusions=write(self.data_root / "overlap_exclusions.jsonl", {"synthetic": True}))
        write(self.data_root / "ROLES_FROZEN.json", roles)
        roles_hash = confirm.digest(self.data_root / "ROLES_FROZEN.json")
        vocabulary = ["<eot>", "A", "B"]
        tokenizer = write(self.data_root / "tokenizer.json", dict(synthetic=True, vocabulary=vocabulary))
        fit = dict(version=stories.TOKENIZER_VERSION, train_only=True, vocab_size=3, eot_id=0,
                   training_document_count=1, roles_frozen_sha256=roles_hash,
                   accepted_training_sha256=role_assets["train"]["sha256"])
        write(self.data_root / "tokenizer_fit.json", fit)
        decoder_path = self.data_root / "synthetic_decoder.json"
        decoder_asset = write(decoder_path, {"synthetic_decoder": True})
        decoder_asset.update(path=str(decoder_path), revision="synthetic", source_url="synthetic:decoder")
        config = dict(plan={"sha256": "1" * 64}, provenance={"sha256": "2" * 64},
                      prior_use_audit={"sha256": "3" * 64}, decoder=decoder_asset,
                      source_paths={"synthetic-source": str(self.data_root / "unopened_source")})
        write(self.data_root / "PREPARATION_CONFIG_COPY.json", config)
        # The actual preparer hashes its original bytes but canonicalizes this
        # archival copy; these equivalent JSON serializations must both work.
        (self.data_root / "preparation_config.json").write_text(json.dumps(config))
        decoder = dict(asset=decoder_asset, vocab_size=fineweb.SOURCE_VOCAB, eot_id=fineweb.SOURCE_EOT,
                       utf8_and_whitespace_roundtrip=True, literal_eot_is_ordinary_text=True,
                       document_roundtrip_rule="every accepted source document exact IDs",
                       complete_documents_roundtripped=5, complete_tokens_roundtripped=30,
                       tokenizers_version=stories.TOKENIZER_VERSION)
        write(self.data_root / "DECODER_QUALIFIED.json", decoder)
        verified = dict(revision=fineweb.SOURCE_REVISION, provenance_sha256="2" * 64,
                        prior_use_audit_sha256="3" * 64, sources={"synthetic-source": {"synthetic": True}})
        write(self.data_root / "SOURCE_VERIFIED.json", verified)
        preparation = dict(config_sha256=confirm.digest(self.data_root / "preparation_config.json"),
                           plan_sha256="1" * 64, preparation_source_sha256=confirm.digest(fineweb.__file__),
                           source_verification_sha256=confirm.digest(self.data_root / "SOURCE_VERIFIED.json"),
                           decoder_qualification_sha256=confirm.digest(self.data_root / "DECODER_QUALIFIED.json"),
                           no_model_inference=True)
        write(self.data_root / "PREPARATION_STARTED.json", dict(
            config_sha256=preparation["config_sha256"], source_sha256=preparation["preparation_source_sha256"],
            no_model_inference=True))
        common = dict(schema_version=1, corpus_kind=confirm.FINEWEB_KIND,
                      source_repo=fineweb.SOURCE_REPO, revision=fineweb.SOURCE_REVISION,
                      vocabulary=vocabulary, vocab_size=3, seq_len=4, eot_id=0,
                      evaluation_policy_version=2, target_padding_id=-100, input_padding_id=0,
                      evaluation_boundary_rule="every synthetic within-document target, including EOS",
                      tokenizer_sha256=tokenizer["sha256"], tokenizer_fit=fit,
                      roles_frozen_sha256=roles_hash, preparation=preparation)
        starts, lengths, owners = stories.evaluation_plan_all_targets(offsets["dev"], 4)
        self.data_manifest = dict(common, role="training_and_development_only",
            splits={"train": splits["train"], "val": splits["dev"]}, tokenizer=tokenizer,
            default_permutation=binary(self.data_root / "permutation.u32", range(4), "<u4"),
            default_stream_seed=31 + 1729, train_blocks=4, fresh_target_capacity=16,
            dev_starts=binary(self.data_root / "dev.starts.i64", starts, "<i8"),
            dev_lengths=binary(self.data_root / "dev.lengths.i64", lengths, "<i8"),
            dev_document_indices=binary(self.data_root / "dev.owners.i64", owners, "<i8"),
            dev_full_windows=len(starts), dev_target_count=int(lengths.sum()))
        self.data_path = self.data_root / "training_manifest.json"
        write(self.data_path, self.data_manifest)
        test_documents = []
        for doc in offsets["test"]:
            starts, lengths, _ = stories.evaluation_plan_all_targets([dict(start=0, end=doc["end"] - doc["start"])], 4)
            test_documents.append(dict(doc, full_starts=starts.tolist(), curve_starts=starts.tolist(),
                full_lengths=lengths.tolist(), curve_lengths=lengths.tolist(),
                counts=dict(full_windows=len(starts), curve_windows=len(starts),
                            full_target_tokens=int(lengths.sum()), curve_target_tokens=int(lengths.sum()))))
        panel_manifest = dict(common, role="sealed_confirmation", splits={"test": splits["test"]},
            documents=test_documents, document_count=3, full_windows=4,
            totals=dict(full_windows=4, curve_windows=4, full_target_tokens=9, curve_target_tokens=9))
        write(panel_root / "manifest.json", panel_manifest)
        panel_sha = confirm.digest(panel_root / "manifest.json")
        (panel_root / "manifest.sha256").write_text(panel_sha + "\n")
        receipt = dict(training_manifest_sha256=confirm.digest(self.data_path), test_manifest_sha256=panel_sha,
                       fresh_target_capacity=16, counts=roles["counts"], no_model_inference=True)
        write(self.data_root / "MANIFESTS_PREPARED.json", receipt)
        write(self.data_root / "PREPARATION_COMPLETE.json", receipt)
        self.disk_loader = fineweb_panel.load_panel
        self.panel = self.disk_loader(panel_root, expected_manifest_sha256=panel_sha)

        def synthetic_loader(*args, **kwargs):
            panel = self.disk_loader(*args, **kwargs)
            return replace(panel, manifest=dict(panel.manifest, role="synthetic_contract_test"))

        for patcher in (patch.object(confirm, "FINEWEB_PANEL_SHA256", panel_sha),
                        patch.object(confirm, "FINEWEB_TRAINING_SHA256", confirm.digest(self.data_path))):
            patcher.start()
            self.addCleanup(patcher.stop)
        patcher = patch.object(confirm.fineweb_panel, "load_panel", side_effect=synthetic_loader)
        self.loader = patcher.start()
        self.addCleanup(patcher.stop)
        source_root = self.root / "frozen"
        project = Path(confirm.__file__).resolve().parents[2]
        source_manifest = {}
        for relative in confirm.TRAINING_SOURCES | {confirm.STORIES_SOURCE, confirm.FINEWEB_SOURCE}:
            destination = source_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project / relative, destination)
            source_manifest[relative] = confirm.digest(destination)
        source_manifest_path = self.root / "sources.json"
        write(source_manifest_path, source_manifest)
        self.runs = []
        for name in ("first", "second"):
            cfg = dict(run_id=name, keep_model_every=2, total_tokens=16, batch_tokens=8,
                       seed=31, method="muon", momentum=.9, lr=.01, n_layer=1, n_embd=4,
                       n_head=1, seq_len=4, corpus_kind=confirm.FINEWEB_KIND,
                       data_path=str(self.data_path), data_sha256=confirm.digest(self.data_path))
            config_path = self.root / (name + ".json")
            write(config_path, cfg)
            self.runs.append(dict(run_id=name, run_dir=str(self.root / "runs" / name),
                                 config_path=str(config_path), source_root=str(source_root),
                                 source_manifest=str(source_manifest_path), checkpoint_steps=[0, 1, 2]))
        self.plan = dict(panel=dict(path=str(panel_root), sha256=panel_sha, kind=confirm.FINEWEB_KIND),
                         metrics=dict(primary="token_weighted_nll", secondary="macro_document_nll"),
                         criteria=dict(keep="every synthetic result"), runs=self.runs)
        self.plan_path, self.lock_path = self.root / "plan.json", self.root / "lock.json"
        write(self.plan_path, self.plan)

    def assert_refused_before_inference(self, operation, message):
        with patch.object(confirm, "evaluate_bpe_panel") as inference:
            with self.assertRaisesRegex(ValueError, message):
                operation()
            inference.assert_not_called()
        self.assertFalse((self.panel.path / ".confirmation").exists())

    def test_actual_prepared_loader_preserves_kind_all_targets_and_eos(self):
        self.assertEqual(self.panel.manifest["corpus_kind"], confirm.FINEWEB_KIND)
        self.assertEqual(sum(int(doc.full_lengths.sum()) for doc in self.panel.documents), 9)
        self.assertEqual(sorted(int(doc.full_lengths.sum()) for doc in self.panel.documents), [1, 3, 5])
        self.assertTrue(all(int(doc.token_ids[-1]) == 0 for doc in self.panel.documents))
        opened = []
        runtime_loader = fineweb.load_training_corpus

        def capture_runtime(*args, **kwargs):
            corpus = runtime_loader(*args, **kwargs)
            opened.append(corpus)
            return corpus

        with patch.object(fineweb, "load_training_corpus", side_effect=capture_runtime):
            _, identity = fineweb_panel.verify_training_data(confirm.read(self.runs[0]["config_path"]), self.panel)
        self.assertEqual(len(opened), 1)
        self.assertTrue(opened[0]._arrays["train"]._mmap.closed)
        self.assertEqual(identity["corpus_kind"], confirm.FINEWEB_KIND)
        self.assertEqual(identity["roles_frozen_sha256"], self.data_manifest["roles_frozen_sha256"])

    def test_original_config_pin_and_canonical_copy_are_both_checked(self):
        original, copied = (self.data_root / name for name in
                            ("preparation_config.json", "PREPARATION_CONFIG_COPY.json"))
        self.assertNotEqual(confirm.digest(original), confirm.digest(copied))
        self.assertEqual(confirm.read(original), confirm.read(copied))
        changed = confirm.read(copied)
        changed["decoder"]["revision"] = "different synthetic provenance"
        copied.write_bytes(confirm.canonical(changed))
        self.assert_refused_before_inference(self.lock, "configuration/source identity")

    def test_unregistered_real_pins_fail_before_panel_loader(self):
        for name in ("FINEWEB_PANEL_SHA256", "FINEWEB_TRAINING_SHA256"):
            with self.subTest(pin=name), patch.object(confirm, name, None):
                self.loader.reset_mock()
                self.assert_refused_before_inference(self.lock, "pins are not registered")
                self.loader.assert_not_called()

    def test_incomplete_preparation_refused_before_lock_or_inference(self):
        (self.data_root / "PREPARATION_COMPLETE.json").unlink()
        self.assert_refused_before_inference(self.lock, "preparation artifact")
        self.assertFalse(self.lock_path.exists())

    def test_failed_preparation_receipt_refused(self):
        confirm.write_new(self.data_root / "PREPARATION_FAILED.json", dict(error="synthetic failure"))
        self.assert_refused_before_inference(self.lock, "retained failure")

    def test_completion_receipt_must_bind_both_manifests(self):
        receipt_path = self.data_root / "PREPARATION_COMPLETE.json"
        receipt = confirm.read(receipt_path)
        receipt["test_manifest_sha256"] = "0" * 64
        receipt_path.write_bytes(confirm.canonical(receipt))
        self.assert_refused_before_inference(self.lock, "completion does not bind")

    def test_changed_role_freeze_refused_before_scoring(self):
        self.finish_all()
        path = self.data_root / "ROLES_FROZEN.json"
        roles = confirm.read(path)
        roles["roles_frozen_before_tokenizer"] = False
        path.write_bytes(confirm.canonical(roles))
        self.assert_refused_before_inference(self.score, "frozen roles")

    def test_changed_decoder_provenance_refused_before_scoring(self):
        self.finish_all()
        path = self.data_root / "DECODER_QUALIFIED.json"
        decoder = confirm.read(path)
        decoder["utf8_and_whitespace_roundtrip"] = False
        path.write_bytes(confirm.canonical(decoder))
        self.assert_refused_before_inference(self.score, "decoder/source preparation provenance")

    def test_changed_decoder_asset_refused_before_scoring(self):
        self.finish_all()
        (self.data_root / "synthetic_decoder.json").write_text('{"changed": true}')
        self.assert_refused_before_inference(self.score, "Pinned file")

    def test_panel_role_and_tokenizer_identity_cannot_be_mixed(self):
        good = replace(self.panel, manifest=dict(self.panel.manifest, role="synthetic_contract_test"))
        for name, value in (("role", "training_and_development_only"),
                            ("roles_frozen_sha256", "0" * 64),
                            ("tokenizer_sha256", "0" * 64)):
            with self.subTest(field=name):
                self.loader.side_effect = None
                self.loader.return_value = replace(good, manifest=dict(good.manifest, **{name: value}))
                self.assert_refused_before_inference(self.lock, "role|identity differs")

    def test_fineweb_and_stories_sources_required_and_evaluator_pinned(self):
        source_path = Path(self.runs[0]["source_manifest"])
        sources = confirm.read(source_path)
        for missing in (confirm.FINEWEB_SOURCE, confirm.STORIES_SOURCE):
            with self.subTest(source=missing):
                source_path.write_bytes(confirm.canonical({k: v for k, v in sources.items() if k != missing}))
                self.assert_refused_before_inference(self.lock, "cover the training implementation")
        source_path.write_bytes(confirm.canonical(sources))
        lock = self.lock()
        for path in fineweb_panel.EVALUATOR_DEPENDENCIES:
            self.assertIn(str(Path(path).resolve()), lock["payload"]["plan"]["evaluator_sources"])

    def test_whole_family_precedes_any_fineweb_inference(self):
        self.lock()
        self.finish(0)
        self.assert_refused_before_inference(self.score, "Whole training family")

    def test_full_score_labels_weighting_and_complete_family_release(self):
        self.finish_all()
        self.score()
        result = confirm.read(self.root / "scores/first/sealed_scores.json")
        self.assertEqual(result["corpus_kind"], confirm.FINEWEB_KIND)
        self.assertEqual(result["final"]["full_target_tokens"], 9)
        self.assertNotIn("character_weighted_nll", result["final"])
        weights = {doc.slug: int(doc.full_lengths.sum()) for doc in self.panel.documents}
        mean = sum(result["final"]["per_document"][name] * weight for name, weight in weights.items()) / 9
        self.assertAlmostEqual(result["final"]["token_weighted_nll"], mean)
        self.assertEqual(result["final"]["token_weighted_nll"], result["curves"][-1]["token_weighted_nll"])
        with self.assertRaisesRegex(ValueError, "Every locked run"):
            confirm.collect(self.lock_path, self.root / "scores", self.root / "released")
        self.score("second", "second")
        confirm.collect(self.lock_path, self.root / "scores", self.root / "released")
        self.assertEqual(len(confirm.read(self.root / "released/all_results.json")["results"]), 2)
        self.assertEqual(confirm.digest(self.panel.path / "manifest.json"), self.panel.manifest_sha256)

    def test_failed_fineweb_score_keeps_binding_and_identical_retry(self):
        self.finish_all()
        with patch.object(confirm, "evaluate_bpe_panel", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                self.score(attempt="failed")
        receipt = self.panel.path / ".confirmation/family.json"
        before = receipt.read_bytes()
        self.score(attempt="retry")
        self.assertEqual(receipt.read_bytes(), before)
        self.assertEqual(confirm.read(self.root / "scores/failed/status.json")["status"], "failed")
        with patch.object(confirm, "evaluate_bpe_panel") as inference:
            with self.assertRaisesRegex(ValueError, "Only failed attempts"):
                self.score(attempt="duplicate")
            inference.assert_not_called()

    def test_retry_rejects_changed_complete_family_before_inference(self):
        self.finish_all()
        with patch.object(confirm, "evaluate_bpe_panel", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaises(RuntimeError):
                self.score(attempt="failed")
        path = Path(self.runs[1]["run_dir"]) / "models/step000002.pt"
        saved = torch.load(path, weights_only=True)
        saved["extra_metadata"] = "different entire family identity"
        torch.save(saved, path)
        with patch.object(confirm, "evaluate_bpe_panel") as inference:
            with self.assertRaisesRegex(ValueError, "Only identical lock/weights"):
                self.score(attempt="changed_family")
            inference.assert_not_called()


if __name__ == "__main__":
    unittest.main()

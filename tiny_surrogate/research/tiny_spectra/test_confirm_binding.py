"""Synthetic-only contracts for one confirmation family per panel.

The fixture creates its own tiny character panel and CPU model snapshots. No
real panel path, manifest, tokens, or model score is used by these tests.
"""

from concurrent.futures import ThreadPoolExecutor
import copy
from dataclasses import replace
import hashlib
import shutil
import threading
import unittest
from unittest.mock import patch

from . import confirm
from . import test_confirm as fixtures


class PanelBindingContracts(unittest.TestCase):
    # Reuse fixture construction without inheriting or rediscovering its tests.
    setUpClass = classmethod(fixtures.ConfirmationContracts.setUpClass.__func__)
    tearDownClass = classmethod(fixtures.ConfirmationContracts.tearDownClass.__func__)
    setUp = fixtures.ConfirmationContracts.setUp
    lock = fixtures.ConfirmationContracts.lock
    finish = fixtures.ConfirmationContracts.finish
    finish_all = fixtures.ConfirmationContracts.finish_all
    score = fixtures.ConfirmationContracts.score

    @property
    def receipt(self):
        return self.panel.path / ".confirmation" / "family.json"

    def identity(self, family=None):
        if family is None:
            family = {"synthetic": {"models": {"0": "2" * 64}}}
        return confirm.binding_identity({"sha256": "1" * 64}, self.panel, family)

    def assert_panel_manifest_unchanged(self):
        self.assertEqual(confirm.digest(self.panel.path / "manifest.json"),
                         self.panel.manifest_sha256)

    def assert_rejected_without_scoring(self, call, output):
        with patch.object(confirm, "evaluate_document") as inference:
            with self.assertRaises((ValueError, FileExistsError, FileNotFoundError)):
                call()
            inference.assert_not_called()
        self.assertFalse(output.exists())
        self.assert_panel_manifest_unchanged()

    def test_binding_identity_covers_panel_lock_and_complete_family(self):
        family = {"synthetic": {"models": {"0": "2" * 64, "1": "3" * 64}}}
        identity = self.identity(family)
        self.assertEqual(identity, dict(
            schema_version=1,
            panel_manifest_sha256=self.panel.manifest_sha256,
            lock_sha256="1" * 64,
            family_sha256=hashlib.sha256(confirm.canonical(family)).hexdigest()))
        changed = copy.deepcopy(family)
        changed["synthetic"]["models"]["1"] = "4" * 64
        self.assertNotEqual(identity, self.identity(changed))
        self.assertNotEqual(identity, confirm.binding_identity(
            {"sha256": "5" * 64}, self.panel, family))

    def test_canonical_real_path_guard_uses_only_synthetic_panel_objects(self):
        # Patch the role, pin, loader, and canonical-path mapping together. Every
        # path and byte remains synthetic; no actual real-panel loader runs.
        simulated_real = replace(self.panel, manifest=dict(
            self.panel.manifest, role="sealed_confirmation"))
        copied = replace(simulated_real, path=self.root / "copied_synthetic_panel")
        spec = dict(self.plan["panel"])
        with patch.object(confirm, "REAL_PANEL_SHA256", self.panel.manifest_sha256), \
                patch.object(confirm, "REAL_PANEL_PATHS", {confirm.CHARACTER_KIND: self.panel.path}), \
                patch.object(confirm.heldout, "load_panel", return_value=simulated_real) as loader:
            self.assertIs(confirm.load_panel(spec), simulated_real)
            loader.return_value = copied
            with self.assertRaisesRegex(ValueError, "canonical pinned panel path"):
                confirm.load_panel(dict(spec, path=str(copied.path)))
        self.assertFalse(self.receipt.parent.exists())
        self.assert_panel_manifest_unchanged()

    def test_concurrent_identical_binding_is_idempotent(self):
        identity = self.identity()
        barrier = threading.Barrier(8)

        def claim(_):
            barrier.wait(timeout=10)
            return confirm.bind_panel(self.panel, identity)

        with ThreadPoolExecutor(max_workers=8) as workers:
            directories = list(workers.map(claim, range(8)))
        self.assertEqual(directories, [self.receipt.parent] * 8)
        self.assertEqual(confirm.read(self.receipt), identity)
        original = self.receipt.read_bytes()
        self.assertEqual(confirm.bind_panel(self.panel, identity, create=False),
                         self.receipt.parent)
        self.assertEqual(self.receipt.read_bytes(), original)
        self.assert_panel_manifest_unchanged()

    def test_concurrent_conflicting_binding_has_one_winning_identity(self):
        identities = [self.identity(), dict(self.identity(), family_sha256="6" * 64)]
        barrier = threading.Barrier(8)

        def claim(index):
            identity = identities[index % 2]
            barrier.wait(timeout=10)
            try:
                confirm.bind_panel(self.panel, identity)
            except ValueError:
                return identity, False
            return identity, True

        with ThreadPoolExecutor(max_workers=8) as workers:
            outcomes = list(workers.map(claim, range(8)))
        winner = confirm.read(self.receipt)
        self.assertIn(winner, identities)
        self.assertEqual(sum(success for _, success in outcomes), 4)
        for attempted, success in outcomes:
            self.assertEqual(success, attempted == winner)
        self.assert_panel_manifest_unchanged()

    def test_malformed_receipts_fail_closed_without_overwrite(self):
        identity = self.identity()
        self.receipt.parent.mkdir()
        malformed = [b"", b"{", b"null", confirm.canonical({"schema_version": 1}),
                     confirm.canonical(dict(identity, family_sha256="not-a-hash")),
                     confirm.canonical(dict(identity, schema_version=True)),
                     confirm.canonical(dict(identity, schema_version=1.0))]
        for content in malformed:
            with self.subTest(receipt=content):
                self.receipt.write_bytes(content)
                with self.assertRaises(ValueError):
                    confirm.bind_panel(self.panel, identity)
                self.assertEqual(self.receipt.read_bytes(), content)
                self.assert_panel_manifest_unchanged()

    def test_malformed_requested_identity_cannot_claim_panel(self):
        identity = self.identity()
        malformed = [None, {}, dict(identity, schema_version=True),
                     dict(identity, schema_version=1.0), dict(identity, lock_sha256=123),
                     dict(identity, family_sha256="A" * 64),
                     dict(identity, family_sha256="short"),
                     dict(identity, panel_manifest_sha256="0" * 64),
                     dict(identity, unexpected_field=True)]
        for value in malformed:
            with self.subTest(identity=value):
                with self.assertRaises(ValueError):
                    confirm.bind_panel(self.panel, value)
                self.assertFalse(self.receipt.parent.exists())
                self.assert_panel_manifest_unchanged()

    def test_read_only_binding_check_never_creates_a_receipt(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            confirm.bind_panel(self.panel, self.identity(), create=False)
        self.assertFalse(self.receipt.exists())
        self.assert_panel_manifest_unchanged()

    def test_failed_first_score_retains_binding_and_allows_identical_retry(self):
        self.finish_all()
        with patch.object(confirm, "evaluate_document", side_effect=RuntimeError("synthetic failure")):
            with self.assertRaisesRegex(RuntimeError, "synthetic failure"):
                self.score(attempt="failed")
        lock, panel = confirm.load_lock(self.lock_path)
        expected = confirm.binding_identity(lock, panel, confirm.complete_family(lock, panel))
        self.assertEqual(confirm.read(self.receipt), expected)
        receipt_bytes = self.receipt.read_bytes()
        failed = self.root / "scores" / "failed"
        self.assertEqual(confirm.read(failed / "status.json")["status"], "failed")
        self.assertTrue((failed / "failure.json").exists())
        self.assertTrue((self.receipt.parent / "attempts").is_dir())
        # An identical lock copy is the same family, and cannot reset history.
        copied_lock = self.root / "same_lock_elsewhere.json"
        shutil.copyfile(self.lock_path, copied_lock)
        retry = self.root / "scores" / "retry"
        result = confirm.score(copied_lock, "first", retry, device="cpu")
        self.assertEqual(result, dict(status="scored_and_sealed", run_id="first"))
        self.assertEqual(self.receipt.read_bytes(), receipt_bytes)
        self.assertEqual(confirm.read(failed / "status.json")["status"], "failed")
        self.assertTrue((failed / "failure.json").exists())
        self.assert_panel_manifest_unchanged()

    def test_new_lock_for_same_panel_is_rejected_before_inference(self):
        self.lock()
        alternate_plan = copy.deepcopy(self.plan)
        alternate_plan["criteria"]["different_preregistered_family"] = True
        alternate_plan_path = self.root / "alternate_plan.json"
        alternate_lock = self.root / "alternate_lock.json"
        confirm.write_new(alternate_plan_path, alternate_plan)
        # Both prospective locks precede training; only the first scorer binds.
        confirm.create_lock(alternate_plan_path, alternate_lock)
        self.finish(0)
        self.finish(1)
        self.score()
        receipt_bytes = self.receipt.read_bytes()
        output = self.root / "scores" / "other_family"
        self.assert_rejected_without_scoring(
            lambda: confirm.score(alternate_lock, "second", output, device="cpu"), output)
        self.assertEqual(self.receipt.read_bytes(), receipt_bytes)

    def test_copied_identical_lock_cannot_rescore_completed_run(self):
        self.finish_all()
        self.score()
        copied_lock = self.root / "copied_lock.json"
        shutil.copyfile(self.lock_path, copied_lock)
        receipt_bytes = self.receipt.read_bytes()
        output = self.root / "scores" / "copied_lock_attempt"
        self.assert_rejected_without_scoring(
            lambda: confirm.score(copied_lock, "first", output, device="cpu"), output)
        self.assertEqual(self.receipt.read_bytes(), receipt_bytes)

    def test_collect_rechecks_binding_without_recreating_it(self):
        self.finish_all()
        self.score("first", "first")
        self.score("second", "second")
        # Corrupt only this synthetic fixture to emulate missing claim metadata.
        self.receipt.unlink()
        output = self.root / "released"
        self.assert_rejected_without_scoring(
            lambda: confirm.collect(self.lock_path, self.root / "scores", output), output)
        self.assertFalse(self.receipt.exists())

    def test_incomplete_family_does_not_claim_panel(self):
        self.lock()
        self.finish(0)
        output = self.root / "scores" / "too_early"
        self.assert_rejected_without_scoring(lambda: self.score(attempt="too_early"), output)
        self.assertFalse(self.receipt.parent.exists())

    def test_unknown_run_does_not_claim_panel(self):
        self.finish_all()
        output = self.root / "scores" / "unknown_run"
        self.assert_rejected_without_scoring(
            lambda: self.score("outside_family", "unknown_run"), output)
        self.assertFalse(self.receipt.parent.exists())

    def test_invalid_device_does_not_claim_panel(self):
        self.finish_all()
        output = self.root / "scores" / "invalid_device"
        self.assert_rejected_without_scoring(
            lambda: confirm.score(self.lock_path, "first", output, device="meta"), output)
        self.assertFalse(self.receipt.parent.exists())

    def test_existing_output_does_not_claim_panel(self):
        self.finish_all()
        output = self.root / "scores" / "already_exists"
        output.mkdir(parents=True)
        with patch.object(confirm, "evaluate_document") as inference:
            with self.assertRaises((ValueError, FileExistsError)):
                confirm.score(self.lock_path, "first", output, device="cpu")
            inference.assert_not_called()
        self.assertFalse(self.receipt.parent.exists())
        self.assertEqual(list(output.iterdir()), [])
        self.assert_panel_manifest_unchanged()


if __name__ == "__main__":
    unittest.main()

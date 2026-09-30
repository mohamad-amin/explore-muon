"""Synthetic, data-only contracts for sealed confirmation windows and integrity."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from research.tiny_spectra.heldout import (
    _canonical_json, _write_panel, encode_with_oov, extract_mit_speech,
    load_panel, prepare_mit_panel, sha256, window_plan,
)


class HeldoutContracts(unittest.TestCase):
    def test_final_target_and_document_boundary(self):
        tokens = encode_with_oov("ababababa", ("a", "b"))
        full, curve, counts = window_plan(tokens, seq_len=4, curve_count=2)
        self.assertEqual(full.tolist(), [0, 4])
        self.assertEqual(curve.tolist(), [0, 4])
        self.assertEqual(counts["full_target_characters"], 8)
        self.assertEqual(counts["unused_trailing_characters"], 0)
        with self.assertRaises(ValueError):
            window_plan(tokens[:-1], seq_len=4, curve_count=2)

    def test_oov_in_context_and_last_target_excludes_whole_window(self):
        tokens = encode_with_oov("ababXbababababa", ("a", "b"))
        self.assertEqual(int(tokens[4]), -1)
        full, curve, counts = window_plan(tokens, seq_len=4, curve_count=1)
        self.assertEqual(full.tolist(), [8])
        self.assertEqual(curve.tolist(), [8])
        self.assertEqual(counts["excluded_oov_starts"], [0, 4])
        self.assertEqual(counts["oov_characters"], 1)

    def test_even_spread_is_deterministic_subset(self):
        full, curve, _ = window_plan(np.zeros(42), seq_len=4, curve_count=4)
        self.assertEqual(curve.tolist(), [0, 12, 24, 36])
        self.assertTrue(set(curve.tolist()) <= set(full.tolist()))

    def test_extraction_retains_oov_and_removes_only_declared_markup(self):
        raw = (b'<h3>ACT</h3><A NAME=speech1><b>FIRST</b></a><blockquote>'
               b'<A NAME=1.1.1>[Aside] A&amp;B\t2.</A><br><i>Exit</i></blockquote>')
        self.assertEqual(extract_mit_speech(raw), "FIRST:\n A&B 2.\n")

    def make_panel(self, path):
        return _write_panel(path,
                            [dict(slug="one", text="ab" * 25),
                             dict(slug="two", text="a" * 21 + "X" + "b" * 28)],
                            ("a", "b"), dict(synthetic=True), seq_len=4,
                            curve_count=3, role="synthetic_contract_test")

    def test_roundtrip_readonly_and_no_cross_document_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel"
            prepared = self.make_panel(path)
            loaded = load_panel(path, expected_manifest_sha256=prepared.manifest_sha256)
            self.assertEqual(loaded.vocabulary, ("a", "b"))
            self.assertEqual(loaded.seq_len, 4)
            for doc in loaded.documents:
                self.assertFalse(doc.token_ids.flags.writeable)
                for bank in ("full", "curve"):
                    for start in doc.starts(bank):
                        window = doc.token_ids[start:start + loaded.seq_len + 1]
                        self.assertEqual(len(window), 5)
                        self.assertTrue((window >= 0).all())
            with self.assertRaises(FileExistsError):
                self.make_panel(path)
            with self.assertRaises(ValueError):
                load_panel(path, expected_manifest_sha256="0" * 64)

    def test_corrupted_tokens_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel"
            self.make_panel(path)
            target = path / "one.tokens.i16"
            data = bytearray(target.read_bytes())
            data[0] ^= 1
            target.write_bytes(data)
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                load_panel(path)

    def test_incorrect_bank_rejected_even_if_file_hash_updated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "panel"
            self.make_panel(path)
            manifest = json.loads((path / "manifest.json").read_text())
            spec = manifest["documents"][0]["artifacts"]["full_starts"]
            data = bytearray((path / spec["path"]).read_bytes())
            data[0] = 1  # in-bounds but outside the declared stride
            (path / spec["path"]).write_bytes(data)
            spec["sha256"] = sha256(data)
            raw = _canonical_json(manifest)
            (path / "manifest.json").write_bytes(raw)
            (path / "manifest.sha256").write_text(sha256(raw) + "\n")
            with self.assertRaisesRegex(ValueError, "bank or denominator"):
                load_panel(path)

    def test_bad_provenance_creates_no_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "bad.tar.gz"
            archive.write_bytes(b"not the pinned archive")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                prepare_mit_panel(root / "panel", archive_path=archive,
                                  audit_path=root / "unused", original_path=root / "unused")
            self.assertFalse((root / "panel").exists())


if __name__ == "__main__":
    unittest.main()

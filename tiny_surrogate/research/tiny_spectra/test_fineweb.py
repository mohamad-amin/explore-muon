"""Synthetic FineWeb data contracts; no source-corpus processing or model score."""
from contextlib import contextmanager
import builtins
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from . import fineweb as f

ROOT = Path(__file__).resolve().parents[2]
PRIVATE = ROOT / "data/tiny_stories_20260928/_prep_dependencies/tokenizers_0_23_2"


class ASCIIDecoder:
    eot = f.SOURCE_EOT
    def decode_document(self, ids):
        return bytes(np.asarray(ids).tolist()).decode("utf-8", errors="strict")
    def decode_exposure(self, ids):
        return f.decode_boundary_fragment(bytes(np.asarray(ids).tolist()))


def encoded_docs(*texts):
    ids = [f.SOURCE_EOT]
    for text in texts:
        ids.extend(text.encode())
        ids.append(f.SOURCE_EOT)
    return np.asarray(ids, dtype="<u2")


def fixture_doc(text, role, index):
    return dict(role=role, text=text, text_sha256=f.sha(text.encode()), identity=f.document_identity(text),
                source="synthetic", source_index=index, source_token_start=index*1000,
                source_token_end=index*1000+len(text)+1)


@contextmanager
def temporary():
    with tempfile.TemporaryDirectory(prefix="fineweb-contract-", dir=ROOT / "scratch/tmp") as path:
        yield Path(path)


class SourceContracts(unittest.TestCase):
    def write_source(self, path, values, reserved=0):
        header = np.zeros(256, dtype="<i4")
        header[:3] = [f.SOURCE_MAGIC, f.SOURCE_VERSION, len(values)]
        header[10] = reserved
        path.write_bytes(header.tobytes()+np.asarray(values, dtype="<u2").tobytes())
        return f.file_sha256(path)

    def test_strict_header_size_hash_and_token_range(self):
        with temporary() as root:
            path = root / "source.bin"
            digest = self.write_source(path, [f.SOURCE_EOT, 100, 200, f.SOURCE_EOT])
            values, receipt = f.verify_source(path, digest, expected_tokens=4)
            self.assertIsInstance(values, np.memmap)
            self.assertEqual(receipt["token_count"], 4)
            values._mmap.close()
            with self.assertRaises(ValueError):
                f.verify_source(path, "0"*64, expected_tokens=4)
            digest = self.write_source(path, [1, 2, 3, 4], reserved=1)
            with self.assertRaises(ValueError):
                f.verify_source(path, digest, expected_tokens=4)
            digest = self.write_source(path, [1, 2, 65535, 4])
            with self.assertRaises(ValueError):
                f.verify_source(path, digest, expected_tokens=4)
            digest = self.write_source(path, [1, 2, 3, 4])
            path.write_bytes(path.read_bytes()+b"x")
            with self.assertRaises(ValueError):
                f.verify_source(path, f.file_sha256(path), expected_tokens=4)

    def test_only_eot_bounded_interior_documents_survive(self):
        decoder = ASCIIDecoder()
        values = np.asarray([ord('x'), f.SOURCE_EOT, ord('A'), f.SOURCE_EOT,
                             ord('B'), ord('C'), f.SOURCE_EOT, ord('z')], dtype="<u2")
        trimming = []
        documents = list(f.complete_documents(values, decoder, "synthetic", audit=trimming))
        self.assertEqual([doc["text"] for doc in documents], ["A", "BC"])
        self.assertEqual(trimming[0]["prefix_fragment_tokens"], 1)
        self.assertEqual(trimming[0]["suffix_fragment_tokens"], 1)
        self.assertEqual(documents[0]["source_token_start"], 2)
        self.assertEqual(documents[0]["source_token_end"], 4)
        # Cutting through the first complete document discards that fragment.
        self.assertEqual([doc["text"] for doc in f.complete_documents(values, decoder, "s", 2, 8)], ["BC"])
        exposed = list(f.exposure_documents(values, decoder, "old"))
        self.assertEqual([row["text"] for row in exposed], ["x", "A", "BC", "z"])

    def test_test_intervals_are_the_frozen_integer_rule(self):
        starts = f.interval_starts()
        self.assertEqual(len(starts), 16)
        self.assertEqual(starts[0], 0)
        self.assertEqual(starts[-1], 100000000-65536)
        self.assertEqual(starts, [i*(100000000-65536)//15 for i in range(16)])
        self.assertTrue(all(b-a >= 65536 for a, b in zip(starts[:-1], starts[1:])))

    def test_utf8_fragment_trims_only_incomplete_boundary_bytes(self):
        text, audit = f.decode_boundary_fragment(b"\xa9 valid words \xf0\x9f")
        self.assertEqual(text, " valid words ")
        self.assertEqual(audit["utf8_boundary_bytes_trimmed"], 3)
        self.assertEqual(audit["utf8_trimmed_boundary_hex"], "a9f09f")
        self.assertEqual(f.decode_boundary_fragment("😀 café".encode())[1]["utf8_boundary_bytes_trimmed"], 0)
        with self.assertRaises(ValueError):
            f.decode_boundary_fragment(b"valid \xff corruption")
        with self.assertRaises(ValueError):
            list(f.complete_documents(np.array([f.SOURCE_EOT, 0xc3, f.SOURCE_EOT], dtype="<u2"), ASCIIDecoder(), "invalid"))


class OverlapContracts(unittest.TestCase):
    def test_prior_tiny_reconstruction_includes_only_consumed_roles(self):
        with temporary() as root:
            shakespeare = root / "shakespeare.txt"
            shakespeare.write_text("entire earlier character corpus")
            rows, source_specs = [], {}
            for source, texts in (("train", [("old training story", "train")]),
                                  ("valid", [("old development story", "dev"), ("unscored old test", "test")])):
                raw, cursor = b"", 0
                for index, (text, role) in enumerate(texts):
                    body = ("\n"+text+"\n").encode()
                    rows.append(dict(status="accepted", assigned_role=role, source=source,
                        source_index=index, source_byte_start=cursor, source_byte_end=cursor+len(body),
                        identity=f.document_identity(text), text_sha256=f.sha(text.encode())))
                    raw += body+f.EOT.encode()
                    cursor = len(raw)
                path = root / (source+".txt")
                path.write_bytes(raw)
                source_specs[source] = dict(path=str(path), sha256=f.file_sha256(path))
            membership = root / "membership.jsonl"
            with membership.open("x") as stream:
                for row in rows:
                    f.append_jsonl(stream, row)
            manifest = root / "old_manifest.json"
            f.write_json(manifest, dict(membership_sha256=f.file_sha256(membership),
                prefix_hashes={key: spec["sha256"] for key, spec in source_specs.items()},
                splits={"train":dict(document_count=1), "val":dict(document_count=1)}))
            spec = dict(shakespeare=dict(path=str(shakespeare), sha256=f.file_sha256(shakespeare)),
                tiny_stories=dict(training_manifest=dict(path=str(manifest), sha256=f.file_sha256(manifest)),
                    membership=dict(path=str(membership), sha256=f.file_sha256(membership)), sources=source_specs))
            exposed = list(f.iter_prior_tiny_text(spec))
            self.assertEqual([row["text"] for row in exposed],
                             ["entire earlier character corpus", "old training story", "old development story"])
            self.assertFalse(any("unscored" in row["text"] for row in exposed))

    def test_normalized_identity_and_verified_64_word_matches(self):
        text = " ".join(f"word{i}" for i in range(80))
        documents = [fixture_doc(text, "test", 0), fixture_doc("Ｆｕｌｌ Identity", "test", 1)]
        index = f.VerifiedOverlapIndex(documents)
        incoming = fixture_doc("prefix " + " ".join(f"word{i}" for i in range(8, 72)) + " suffix", "train", 2)
        matches = list(index.matches(incoming))
        self.assertEqual([i for i, _ in matches], [0])
        self.assertEqual(matches[0][1]["reason"], "verified_64_word_overlap")
        exact = list(index.matches(fixture_doc("full identity!!!", "train", 3)))
        self.assertEqual(exact[0][0], 1)
        self.assertEqual(exact[0][1]["reason"], "normalized_whole_document_identity")
        no64 = fixture_doc(" ".join(f"word{i}" for i in range(63)), "train", 4)
        self.assertFalse(list(index.matches(no64)))

    def test_rolling_hash_collisions_never_exclude_without_exact_words(self):
        def collisions(words, n=64):
            yield from ((0, start) for start in range(max(0, len(words)-n+1)))
        index = f.VerifiedOverlapIndex([fixture_doc(" ".join(["alpha"]*70), "test", 0)], hash_function=collisions)
        self.assertFalse(list(index.matches(fixture_doc(" ".join(["beta"]*70), "train", 1))))
        self.assertEqual(len(list(index.matches(fixture_doc("prefix " + " ".join(["alpha"]*64), "train", 2)))), 1)

    def test_roles_frozen_before_fit_with_holdout_precedence(self):
        with temporary() as root:
            val = encoded_docs("previously exposed validation")
            test = np.full(1024, ord('x'), dtype="<u2")
            first = encoded_docs("previously exposed validation", "independent heldout text")
            second = encoded_docs("INDEPENDENT HELDOUT TEXT")
            test[:len(first)] = first
            test[768:768+len(second)] = second
            sources = {"fineweb_val_000000.bin": val, "fineweb_train_000072.bin":test,
                       "train":encoded_docs("fresh training material", "previously exposed validation",
                                             "independent heldout text", "FRESH TRAINING MATERIAL")}
            with patch.object(f, "iter_prior_tiny_text", return_value=iter(())):
                roles = f.freeze_roles(root, sources, ASCIIDecoder(), {}, development_prefix=len(val),
                                       test_count=2, test_length=256, train_names=["train"])
            self.assertEqual(roles["counts"], {"dev":1, "test":1, "train":1})
            self.assertTrue((root / "ROLES_FROZEN.json").is_file())
            self.assertFalse((root / "tokenizer.json").exists())
            test_docs = list(f.read_jsonl(root / "test.accepted.jsonl"))
            self.assertEqual(test_docs[0]["text"], "independent heldout text")
            reasons = [row["reason"] for row in f.read_jsonl(root / "membership.jsonl")]
            self.assertIn("within_role_normalized_duplicate", reasons)
            self.assertIn("matches_prior_exposed_text", reasons)
            self.assertIn("matches_fixed_holdout", reasons)
            self.assertIn("within_train_normalized_duplicate", reasons)


class EncodingAndRuntimeContracts(unittest.TestCase):
    def fixture(self, root, *, encode=True):
        documents = {"train":[fixture_doc("alpha beta gamma delta " * 12, "train", 0),
                             fixture_doc("other fresh learning text " * 10, "train", 1)],
                     "dev":[fixture_doc("é", "dev", 2), fixture_doc("development text with final tail", "dev", 3)],
                     "test":[fixture_doc("sealed unique text", "test", 4)]}
        for role, rows in documents.items():
            with (root / (role+".accepted.jsonl")).open("x") as stream:
                for row in rows:
                    f.append_jsonl(stream, row)
        roles = dict(roles_frozen_before_tokenizer=True, counts={k:len(v) for k,v in documents.items()},
                     roles={role:f.artifact(root / (role+".accepted.jsonl"), root) for role in documents})
        f.write_json(root / "ROLES_FROZEN.json", roles)
        tokenizer, fit = f.train_byte_tokenizer(root, roles, PRIVATE, vocab_size=257, threads=1)
        if not encode:
            return documents, roles, (tokenizer, fit)
        receipt = f.build_manifests(root, roles, tokenizer, fit, {}, seq_len=8, min_fresh_targets=32)
        return documents, roles, receipt

    def test_post_fit_role_mutation_rejected_before_any_encoding(self):
        for role in ("train", "dev", "test"):
            with self.subTest(role=role), temporary() as root:
                _, roles, (tokenizer, fit) = self.fixture(root, encode=False)
                path = root / (role+".accepted.jsonl")
                rows = list(f.read_jsonl(path))
                rows[0]["text"] += " illicit change after roles were frozen"
                rows[0]["text_sha256"] = f.sha(rows[0]["text"].encode())
                rows[0]["identity"] = f.document_identity(rows[0]["text"])
                with path.open("w") as stream:
                    for row in rows:
                        f.append_jsonl(stream, row)
                with self.assertRaisesRegex(ValueError, "Pinned file"):
                    f.build_manifests(root, roles, tokenizer, fit, {}, seq_len=8, min_fresh_targets=32)
                self.assertFalse((root / "train.u16").exists())
                self.assertFalse((root / "dev.u16").exists())
                self.assertFalse((root / "sealed_test").exists())
                self.assertFalse((root / "training_manifest.json").exists())

    def test_frozen_role_counts_and_record_roles_are_validated(self):
        for corruption in ("count", "record_role"):
            with self.subTest(corruption=corruption), temporary() as root:
                _, roles, (tokenizer, fit) = self.fixture(root, encode=False)
                if corruption == "count":
                    roles["counts"]["dev"] += 1
                else:
                    path = root / "dev.accepted.jsonl"
                    rows = list(f.read_jsonl(path))
                    rows[0]["role"] = "test"
                    with path.open("w") as stream:
                        for row in rows:
                            f.append_jsonl(stream, row)
                    roles["roles"]["dev"] = f.artifact(path, root)
                # A self-consistent synthetic hash receipt still needs the
                # semantic count/role checks; this is not a real receipt edit.
                (root / "ROLES_FROZEN.json").write_bytes(f.canonical(roles))
                fit["roles_frozen_sha256"] = f.file_sha256(root / "ROLES_FROZEN.json")
                with self.assertRaisesRegex(ValueError, "Document (count|role)"):
                    f.build_manifests(root, roles, tokenizer, fit, {}, seq_len=8, min_fresh_targets=32)
                self.assertFalse((root / "train.u16").exists())

    def test_train_only_fit_mmap_pairing_exhaustion_and_masked_tails(self):
        with temporary() as root:
            documents, roles, receipt = self.fixture(root)
            original_import = builtins.__import__
            def no_tokenizer_import(name, *args, **kwargs):
                if name == "tokenizers" or name.startswith("tokenizers."):
                    raise AssertionError("Runtime loader imported preparation dependency")
                return original_import(name, *args, **kwargs)
            with patch("builtins.__import__", side_effect=no_tokenizer_import):
                corpus = f.load_training_corpus(root, expected_manifest_sha256=receipt["training_manifest_sha256"])
            self.assertEqual(corpus.manifest["corpus_kind"], f.KIND)
            self.assertIsInstance(corpus.tokens("train"), np.memmap)
            self.assertEqual(corpus.tokens("train").dtype, np.uint16)
            a, b = torch.Generator().manual_seed(18), torch.Generator().manual_seed(18)
            small = torch.cat([corpus.batch("train", 2, 8, generator=a)[0] for _ in range(3)])
            big = corpus.batch("train", 6, 8, generator=b)[0]
            torch.testing.assert_close(small, big, atol=0, rtol=0)
            with self.assertRaises(ValueError):
                corpus.batch("train", corpus.manifest["train_blocks"], 8, generator=a)
            starts = corpus.evaluation_starts("val", 8)
            lengths = corpus.evaluation_lengths("val", 8)
            x, y = corpus.evaluation_batch("val", starts, 8)
            self.assertEqual(int((y != -100).sum()), sum(len(doc["text"].encode()) for doc in documents["dev"]))
            self.assertEqual(int((y != -100).sum()), int(lengths.sum()))
            self.assertTrue((lengths < 8).any())
            for row, length in enumerate(lengths.tolist()):
                self.assertTrue((y[row, length:] == -100).all())
                self.assertTrue((x[row, length:] == corpus.manifest["eot_id"]).all())
            with self.assertRaises(ValueError):
                corpus.tokens("test")
            self.assertEqual(corpus.manifest["tokenizer_fit"]["training_document_count"], 2)
            self.assertEqual(corpus.manifest["tokenizer_fit"]["roles_frozen_sha256"], f.file_sha256(root / "ROLES_FROZEN.json"))
            # Corrupted bytes are detected before a runtime corpus is exposed.
            train_path = root / "train.u16"
            with train_path.open("r+b") as stream:
                stream.write(b"\xff\xff")
            with self.assertRaises(ValueError):
                f.load_training_corpus(root)
            corpus.close()

    def test_fitting_rejects_changed_training_membership(self):
        with temporary() as root:
            train = root / "train.accepted.jsonl"
            train.write_text(json.dumps(fixture_doc("tiny training", "train", 0))+"\n")
            roles = dict(roles_frozen_before_tokenizer=True, counts={"train":1}, roles={"train":f.artifact(train, root)})
            f.write_json(root / "ROLES_FROZEN.json", roles)
            train.write_text(train.read_text()+json.dumps(fixture_doc("leaked dev", "dev", 1))+"\n")
            with self.assertRaises(ValueError):
                f.train_byte_tokenizer(root, roles, PRIVATE, vocab_size=257, threads=1)

    def test_insufficient_capacity_never_publishes_preparation_complete(self):
        with temporary() as root:
            self.fixture(root)
            receipt = json.loads((root / "MANIFESTS_PREPARED.json").read_bytes())
            self.assertGreaterEqual(receipt["fresh_target_capacity"], 32)
            # build_manifests publishes only the encoding result. The real
            # orchestrator must qualify the runtime loader before final success.
            self.assertFalse((root / "PREPARATION_COMPLETE.json").exists())
        with temporary() as root:
            documents = {role:[fixture_doc("only a tiny document", role, i)]
                         for i, role in enumerate(("train", "dev", "test"))}
            for role, rows in documents.items():
                with (root / (role+".accepted.jsonl")).open("x") as stream:
                    for row in rows:
                        f.append_jsonl(stream, row)
            roles = dict(roles_frozen_before_tokenizer=True, counts={role:1 for role in documents},
                         roles={role:f.artifact(root / (role+".accepted.jsonl"), root) for role in documents})
            f.write_json(root / "ROLES_FROZEN.json", roles)
            tokenizer, fit = f.train_byte_tokenizer(root, roles, PRIVATE, vocab_size=257, threads=1)
            with self.assertRaisesRegex(ValueError, "Fresh-target capacity"):
                f.build_manifests(root, roles, tokenizer, fit, {}, seq_len=8, min_fresh_targets=1024)
            self.assertFalse((root / "training_manifest.json").exists())
            self.assertFalse((root / "PREPARATION_COMPLETE.json").exists())

    def test_actual_pinned_decoder_eot_unicode_whitespace_and_literal_marker(self):
        path = ROOT / "data/fineweb_d_20260928/source/decoder/manifest.json"
        if not path.exists():
            self.skipTest("Pinned GPT-2 decoder asset not present")
        decoder = f.GPT2Decoder(json.loads(path.read_bytes()), PRIVATE, threads=1)
        for text in (" A\n\n B  ", "café 中文 😀", f.EOT, "can't won't \t 123"):
            ids = decoder.ordinary.encode(text, add_special_tokens=False).ids
            self.assertNotIn(f.SOURCE_EOT, ids)
            self.assertEqual(decoder.decode_document(ids), text)
        self.assertEqual(decoder.roundtrip_documents, 4)
        # Incomplete UTF-8 inside a purported complete document must fail.
        partial = next(i for i, raw in enumerate(decoder.token_bytes) if raw == b"\xc3")
        with self.assertRaises(UnicodeDecodeError):
            decoder.decode_document([partial])


if __name__ == "__main__":
    unittest.main()

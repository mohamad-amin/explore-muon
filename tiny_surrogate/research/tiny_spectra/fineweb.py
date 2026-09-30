"""Data-only Candidate-D FineWeb preparation, with no network or model inference.

Source bytes and GPT-2 decoder assets must already exist and be byte-pinned.
The CLI consumes the reviewed preparation plan, freezes document roles before
fitting its training-only BPE, and never reads or changes earlier sealed panels.
"""
from collections import Counter, defaultdict
from contextlib import contextmanager
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import sys
import time

import numpy as np
import torch

from .stories import (TOKENIZER_VERSION, TokenCorpus, canonical,
                      document_identity, evaluation_plan_all_targets,
                      normalized_words, sha, shingles, write_json)

KIND = "fineweb_byte_bpe_v1"
SOURCE_REVISION = "889765ea1f903759787add96995d81171b632d0c"
SOURCE_REPO = "kjj0/fineweb10B-gpt2"
SOURCE_MAGIC, SOURCE_VERSION = 20240520, 1
SOURCE_VOCAB, SOURCE_EOT = 50257, 50256
EOT = "<|endoftext|>"
CONTEXT, VOCAB_SIZE, MIN_FRESH_TARGETS = 512, 12576, 192484352
HEADER_BYTES = 1024


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            result.update(chunk)
    return result.hexdigest()


def artifact(path, relative_to):
    path = Path(path)
    return dict(path=str(path.relative_to(relative_to)), bytes=path.stat().st_size,
                sha256=file_sha256(path))


def checked_file(spec):
    path = Path(spec["path"]).resolve()
    if "bytes" in spec and path.stat().st_size != spec["bytes"]:
        raise ValueError(f"Pinned file size changed: {path}")
    if file_sha256(path) != spec["sha256"]:
        raise ValueError(f"Pinned file SHA256 changed: {path}")
    return path


def checked_json(spec):
    return json.loads(checked_file(spec).read_bytes())


def read_jsonl(path):
    with Path(path).open() as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def append_jsonl(stream, row):
    stream.write(json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n")


def verify_source(path, expected_sha256, *, expected_tokens=100000000):
    """Validate the entire immutable uint16 shard, including reserved header."""
    path = Path(path).resolve()
    initial = path.stat()
    with path.open("rb") as stream:
        header_raw = stream.read(HEADER_BYTES)
    if len(header_raw) != HEADER_BYTES:
        raise ValueError("Truncated FineWeb header")
    header = np.frombuffer(header_raw, dtype="<i4")
    if (header[0] != SOURCE_MAGIC or header[1] != SOURCE_VERSION
            or header[2] != expected_tokens or (header[3:] != 0).any()):
        raise ValueError("Invalid FineWeb magic/version/count/reserved header")
    if initial.st_size != HEADER_BYTES + 2 * int(header[2]):
        raise ValueError("FineWeb file length disagrees with its token count")
    digest = file_sha256(path)
    if digest != expected_sha256:
        raise ValueError("FineWeb shard hash does not match pinned public LFS identity")
    values = np.memmap(path, mode="r", dtype="<u2", offset=HEADER_BYTES,
                       shape=(int(header[2]),))
    for start in range(0, len(values), 1 << 20):
        if values[start:start+(1 << 20)].max(initial=0) >= SOURCE_VOCAB:
            raise ValueError("Source token outside the GPT-2 vocabulary")
    final = path.stat()
    if (initial.st_ino, initial.st_size, initial.st_mtime_ns) != (final.st_ino, final.st_size, final.st_mtime_ns):
        raise ValueError("Source changed while validating it")
    return values, dict(path=str(path), sha256=digest, bytes=final.st_size,
                        token_count=len(values), magic=SOURCE_MAGIC, version=SOURCE_VERSION,
                        header_sha256=sha(header_raw), reserved_header_zero=True)


def interval_starts(token_count=100000000, count=16, length=65536):
    """Integer floor(linspace(0, token_count-length, count)), without float drift."""
    if count < 2 or length < 1 or token_count < count * length:
        raise ValueError("Invalid or overlapping fixed test intervals")
    return [i * (token_count-length) // (count-1) for i in range(count)]


def load_private_tokenizers(target, *, threads=8):
    target = Path(target).resolve()
    if not 1 <= threads <= 8:
        raise ValueError("Preparation allows one through eight CPU threads")
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["RAYON_NUM_THREADS"] = str(threads)
    sys.path.insert(0, str(target))
    import tokenizers
    if (tokenizers.__version__ != TOKENIZER_VERSION
            or not Path(tokenizers.__file__).resolve().is_relative_to(target)):
        raise ValueError("Wrong private preparation-only tokenizers installation")
    return tokenizers


class GPT2Decoder:
    """Pinned complete-document decoder and exact ordinary-text round-trip gate."""
    def __init__(self, spec, private_target, *, expected_vocab=SOURCE_VOCAB,
                 expected_eot=SOURCE_EOT, threads=8):
        if not spec.get("revision") or not spec.get("source_url"):
            raise ValueError("Decoder must identify a pinned upstream revision and URL")
        path = checked_file(spec)
        library = load_private_tokenizers(private_target, threads=threads)
        raw = path.read_bytes()
        config = json.loads(raw)
        self.special = library.Tokenizer.from_str(raw.decode())
        if (self.special.get_vocab_size() != expected_vocab
                or self.special.token_to_id(EOT) != expected_eot
                or self.special.decode([expected_eot], skip_special_tokens=False) != EOT
                or self.special.encode(EOT, add_special_tokens=False).ids != [expected_eot]):
            raise ValueError("Pinned tokenizer failed GPT-2 vocabulary/EOT qualification")
        # Original source uses ordinary text tokenization. Literal EOT spellings
        # inside documents must not become delimiters via AddedToken matching.
        ordinary_config = dict(config, added_tokens=[], post_processor=None,
                               truncation=None, padding=None)
        self.ordinary = library.Tokenizer.from_str(json.dumps(ordinary_config))
        self.eot, self.vocab_size = expected_eot, expected_vocab
        byte_values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        codepoints = byte_values.copy()
        next_codepoint = 256
        for value in range(256):
            if value not in byte_values:
                byte_values.append(value)
                codepoints.append(next_codepoint)
                next_codepoint += 1
        byte_decoder = {chr(codepoint): value for value, codepoint in zip(byte_values, codepoints)}
        self.token_bytes = [None if index == expected_eot else
                            bytes(byte_decoder[char] for char in self.ordinary.id_to_token(index))
                            for index in range(expected_vocab)]
        self.roundtrip_documents = 0
        self.roundtrip_tokens = 0
        probes = ("hello world", " a\n\n b  ", "café Ελληνικά 中文 😀", EOT,
                  "can't won't 123\tX\r\nend")
        for text in probes:
            ids = self.ordinary.encode(text, add_special_tokens=False).ids
            if expected_eot in ids or self.ordinary.decode(ids, skip_special_tokens=False) != text:
                raise ValueError("Ordinary-text byte/EOT qualification failed")
        self.qualification = dict(asset=dict(spec), vocab_size=expected_vocab, eot_id=expected_eot,
                                  utf8_and_whitespace_roundtrip=True,
                                  literal_eot_is_ordinary_text=True,
                                  document_roundtrip_rule="every accepted source document exact IDs",
                                  tokenizers_version=library.__version__)

    def decode_document(self, ids):
        values = np.asarray(ids, dtype="<u2").tolist()
        if self.eot in values or any(value >= self.vocab_size for value in values):
            raise ValueError("Document body contains a delimiter or invalid token")
        text = b"".join(self.token_bytes[value] for value in values).decode("utf-8", errors="strict")
        if self.ordinary.decode(values, skip_special_tokens=False) != text:
            raise ValueError("GPT-2 byte decoder disagrees with tokenizer decoder")
        if self.ordinary.encode(text, add_special_tokens=False).ids != values:
            raise ValueError("Complete source document failed exact GPT-2 ID round trip")
        self.roundtrip_documents += 1
        self.roundtrip_tokens += len(values)
        return text

    def decode_fragment(self, ids):
        """Exposure-only fragments; never eligible for any new document role."""
        text, _ = self.decode_exposure(ids)
        return text

    def decode_exposure(self, ids):
        raw = b"".join(self.token_bytes[int(value)] for value in ids)
        return decode_boundary_fragment(raw)


def decode_boundary_fragment(raw):
    """Account for incomplete boundary UTF-8 bytes; reject interior corruption."""
    prefix = 0
    while prefix < len(raw) and 0x80 <= raw[prefix] <= 0xBF:
        prefix += 1
    if prefix > 3:
        raise ValueError("More than one partial UTF-8 character at source boundary")
    remaining, suffix = raw[prefix:], 0
    try:
        text = remaining.decode("utf-8", errors="strict")
    except UnicodeDecodeError as error:
        if error.reason != "unexpected end of data" or error.end != len(remaining):
            raise ValueError("Invalid UTF-8 inside a source exposure fragment") from error
        suffix = len(remaining) - error.start
        if suffix > 3:
            raise ValueError("Invalid trailing UTF-8 fragment") from error
        text = remaining[:error.start].decode("utf-8", errors="strict")
    return text, dict(raw_bytes=len(raw), utf8_prefix_bytes_trimmed=prefix,
                      utf8_suffix_bytes_trimmed=suffix,
                      utf8_boundary_bytes_trimmed=prefix+suffix,
                      utf8_trimmed_boundary_hex=(raw[:prefix]+(raw[-suffix:] if suffix else b"")).hex())


def complete_documents(tokens, decoder, source, start=0, end=None, *, audit=None):
    """Yield only documents enclosed by two EOTs wholly inside [start,end)."""
    end = len(tokens) if end is None else end
    if not 0 <= start < end <= len(tokens):
        raise ValueError("Source interval out of bounds")
    boundaries = np.flatnonzero(tokens[start:end] == decoder.eot) + start
    stats = dict(source=source, interval_start=start, interval_end=end,
                 eot_count=len(boundaries), complete_documents=max(0, len(boundaries)-1),
                 prefix_fragment_tokens=(int(boundaries[0])-start if len(boundaries) else end-start),
                 suffix_fragment_tokens=(end-int(boundaries[-1])-1 if len(boundaries) else 0),
                 no_eot_fragment=not bool(len(boundaries)))
    if audit is not None:
        audit.append(stats)
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        left, right = int(left), int(right)
        ids = tokens[left+1:right]
        text = decoder.decode_document(ids)
        yield dict(source=source, source_index=left, source_token_start=left+1,
                   source_token_end=right+1, source_left_eot=left, source_right_eot=right,
                   source_token_ids_sha256=sha(tokens[left+1:right+1].tobytes()),
                   text=text, text_sha256=sha(text.encode()), identity=document_identity(text))


def exposure_documents(tokens, decoder, source):
    """Entire consumed shard, including its otherwise ineligible outer pieces."""
    boundaries = np.flatnonzero(tokens == decoder.eot)
    spans = [(0, int(boundaries[0]))] if len(boundaries) else [(0, len(tokens))]
    spans += [(int(left)+1, int(right)) for left, right in zip(boundaries[:-1], boundaries[1:])]
    if len(boundaries):
        spans.append((int(boundaries[-1])+1, len(tokens)))
    for index, (left, right) in enumerate(spans):
        if right <= left:
            continue
        text, byte_audit = decoder.decode_exposure(tokens[left:right])
        outer = index in (0, len(spans)-1)
        if not outer and byte_audit["utf8_boundary_bytes_trimmed"]:
            raise ValueError("Complete prior-exposure document has incomplete UTF-8")
        yield dict(source=source, source_index=left, source_token_start=left,
                   source_token_end=right, text=text, identity=document_identity(text),
                   exposure_outer_fragment=outer, **byte_audit)


class VerifiedOverlapIndex:
    """Candidate-sized index; 64-bit keys accelerate, exact word tuples decide."""
    def __init__(self, documents, n=64, hash_function=shingles):
        self.documents = list(documents)
        self.words = [normalized_words(doc["text"]) for doc in self.documents]
        self.n, self.hash_function = n, hash_function
        self.identities, self.index = defaultdict(list), defaultdict(list)
        for index, (doc, words) in enumerate(zip(self.documents, self.words)):
            self.identities[doc["identity"]].append(index)
            for key, offset in hash_function(words, n):
                self.index[key].append((index, offset))

    def matches(self, document, *, excluded=(), first_only=False):
        excluded = set(excluded)
        found = set()
        for index in self.identities.get(document["identity"], ()):
            if index not in excluded:
                found.add(index)
                yield index, dict(reason="normalized_whole_document_identity", identity=document["identity"])
                if first_only:
                    return
        words = normalized_words(document["text"])
        for key, offset in self.hash_function(words, self.n):
            for index, other_offset in self.index.get(key, ()):
                if index in excluded or index in found:
                    continue
                if words[offset:offset+self.n] == self.words[index][other_offset:other_offset+self.n]:
                    found.add(index)
                    yield index, dict(reason="verified_64_word_overlap", incoming_word_offset=offset,
                                      protected_word_offset=other_offset,
                                      normalized_shingle_sha256=sha(" ".join(words[offset:offset+self.n]).encode()))
                    if first_only:
                        return


def iter_prior_tiny_text(spec):
    """Previously consumed train/dev only, verified against original byte offsets."""
    shakespeare = checked_file(spec["shakespeare"])
    text = shakespeare.read_bytes().decode("utf-8")
    yield dict(source="prior_tiny_shakespeare", source_index=0, text=text, identity=document_identity(text))
    previous = spec["tiny_stories"]
    manifest = checked_json(previous["training_manifest"])
    membership_path = checked_file(previous["membership"])
    if file_sha256(membership_path) != manifest["membership_sha256"]:
        raise ValueError("Prior TinyStories membership differs from its consumed manifest")
    source_paths = {role: checked_file(value) for role, value in previous["sources"].items()}
    if set(source_paths) != {"train", "valid"}:
        raise ValueError("Need both previously consumed TinyStories prefixes")
    for role, path in source_paths.items():
        if file_sha256(path) != manifest["prefix_hashes"][role]:
            raise ValueError("Prior TinyStories source differs from its manifest")
    streams = {role: path.open("rb") for role, path in source_paths.items()}
    counts = Counter()
    try:
        for row in read_jsonl(membership_path):
            if row["status"] != "accepted" or row["assigned_role"] not in ("train", "dev"):
                continue
            stream = streams[row["source"]]
            stream.seek(row["source_byte_start"])
            text = stream.read(row["source_byte_end"]-row["source_byte_start"]).decode("utf-8").strip()
            if sha(text.encode()) != row["text_sha256"] or document_identity(text) != row["identity"]:
                raise ValueError("Previously consumed TinyStories document reconstruction changed")
            counts[row["assigned_role"]] += 1
            yield dict(source="prior_tiny_stories_"+row["assigned_role"], source_index=row["source_index"],
                       text=text, identity=row["identity"])
    finally:
        for stream in streams.values():
            stream.close()
    if (counts["train"] != manifest["splits"]["train"]["document_count"]
            or counts["dev"] != manifest["splits"]["val"]["document_count"]):
        raise ValueError("Prior TinyStories accepted-role count mismatch")


def unique_documents(documents, role, audit_stream):
    """First normalized identity wins in fixed source order within a holdout."""
    accepted, seen = [], {}
    for document in documents:
        row = {key: value for key, value in document.items() if key != "text"}
        row["assigned_role"] = role
        if not normalized_words(document["text"]):
            row.update(status="excluded", reason="empty_normalized_document")
        elif document["identity"] in seen:
            row.update(status="excluded", reason="within_role_normalized_duplicate",
                       matched_source=seen[document["identity"]])
        else:
            seen[document["identity"]] = [document["source"], document["source_index"]]
            accepted.append(dict(document, role=role))
            row.update(status="candidate", reason="first_normalized_identity_in_role")
        append_jsonl(audit_stream, row)
    return accepted


def freeze_roles(output, sources, decoder, prior_spec, *, development_prefix=1048576,
                 test_count=16, test_length=65536, train_names=None):
    """Streaming overlap audit; accepted role text files are frozen before BPE."""
    output = Path(output)
    role_paths = {role: output / (role + ".accepted.jsonl") for role in ("train", "dev", "test")}
    train_names = train_names or [f"fineweb_train_{index:06d}.bin" for index in (1, 2, 3)]
    val_name, test_name = "fineweb_val_000000.bin", "fineweb_train_000072.bin"
    trimming, counts, exposure_counts = [], Counter(), Counter()
    with (output / "membership.jsonl").open("x") as audit_stream, (output / "overlap_exclusions.jsonl").open("x") as evidence_stream:
        dev = unique_documents(complete_documents(sources[val_name], decoder, val_name, 0,
                               development_prefix, audit=trimming), "dev", audit_stream)
        test_source_documents = (document
            for start in interval_starts(len(sources[test_name]), test_count, test_length)
            for document in complete_documents(sources[test_name], decoder, test_name, start,
                                                start+test_length, audit=trimming))
        test_candidates = unique_documents(test_source_documents, "test", audit_stream)
        index = VerifiedOverlapIndex(test_candidates)
        excluded = set()
        # Every old validation token is exposure evidence, including outer
        # fragments, irrespective of which subset happened to be evaluated.
        def previous_documents():
            yield from exposure_documents(sources[val_name], decoder, "prior_reference_validation")
            yield from iter_prior_tiny_text(prior_spec)
        for document in previous_documents():
            exposure_counts[document["source"]] += 1
            for candidate, match in index.matches(document, excluded=excluded):
                excluded.add(candidate)
                append_jsonl(evidence_stream, dict(stage="test_prior_exposure", candidate_source=test_candidates[candidate]["source"],
                    candidate_source_index=test_candidates[candidate]["source_index"],
                    exposed_source=document["source"], exposed_source_index=document["source_index"], **match))
            if "utf8_boundary_bytes_trimmed" in document:
                exposure_counts["utf8_boundary_bytes_trimmed"] += document["utf8_boundary_bytes_trimmed"]
                if document["utf8_boundary_bytes_trimmed"]:
                    append_jsonl(evidence_stream, dict(stage="prior_exposure_utf8_boundary",
                        **{key: value for key, value in document.items() if key not in ("text", "identity")}))
        test = []
        for i, document in enumerate(test_candidates):
            accepted = i not in excluded
            row = {key: value for key, value in document.items() if key != "text"}
            row.update(assigned_role="test", status="accepted" if accepted else "excluded",
                       reason="independent_complete_document" if accepted else "matches_prior_exposed_text")
            append_jsonl(audit_stream, row)
            if accepted:
                test.append(document)
        if not dev or not test:
            raise ValueError("No eligible development or independent test documents")
        del index, test_candidates
        for document in dev:
            append_jsonl(audit_stream, dict(**{key: value for key, value in document.items() if key != "text"},
                                            assigned_role="dev", status="accepted", reason="fixed_development_document"))
        for role, documents in (("dev", dev), ("test", test)):
            with role_paths[role].open("x") as stream:
                for document in documents:
                    append_jsonl(stream, document)
            counts[role] = len(documents)
        # Tiny heldouts determine exclusion; the large train corpus is streamed.
        protected = VerifiedOverlapIndex(dev + test)
        seen = {}
        with role_paths["train"].open("x") as stream:
            for name in train_names:
                for document in complete_documents(sources[name], decoder, name, audit=trimming):
                    row = {key: value for key, value in document.items() if key != "text"}
                    row["assigned_role"] = "train"
                    identity = document["identity"]
                    match = next(protected.matches(document, first_only=True), None)
                    if not normalized_words(document["text"]):
                        row.update(status="excluded", reason="empty_normalized_document")
                    elif identity in seen:
                        row.update(status="excluded", reason="within_train_normalized_duplicate",
                                   matched_source=seen[identity])
                    elif match is not None:
                        matched_index, evidence = match
                        other = protected.documents[matched_index]
                        row.update(status="excluded", reason="matches_fixed_holdout")
                        append_jsonl(evidence_stream, dict(stage="train_holdout", source=name,
                            source_index=document["source_index"], protected_role=other["role"],
                            protected_source=other["source"], protected_source_index=other["source_index"], **evidence))
                    else:
                        seen[identity] = [name, document["source_index"]]
                        row.update(status="accepted", reason="first_training_identity_without_holdout_match")
                        append_jsonl(stream, dict(document, role="train"))
                        counts["train"] += 1
                    append_jsonl(audit_stream, row)
                print(json.dumps(dict(stage="role_assignment", source=name, accepted=dict(counts))), flush=True)
    if counts["train"] == 0:
        raise ValueError("No training documents survived holdout protection")
    frozen = dict(schema_version=1, corpus_kind=KIND, roles_frozen_before_tokenizer=True,
                  roles={role: artifact(path, output) for role, path in role_paths.items()}, counts=dict(counts),
                  membership=artifact(output / "membership.jsonl", output),
                  overlap_exclusions=artifact(output / "overlap_exclusions.jsonl", output),
                  source_interval_trimming=trimming, prior_exposure_documents=dict(exposure_counts),
                  normalized_rule="NFKC,casefold,Unicode\\w+,single-space words",
                  precedence="prior exposure excludes test; both holdouts exclude incoming training",
                  within_role_dedup="first normalized identity in fixed source/interval/start order",
                  overlap_rule="verified contiguous 64 normalized words; rolling hash never decides alone",
                  no_model_scores=True, frozen_unix=time.time())
    write_json(output / "ROLES_FROZEN.json", frozen)
    return frozen


def train_byte_tokenizer(output, roles, private_target, *, vocab_size=VOCAB_SIZE, threads=8):
    """Fit exactly once from byte-pinned accepted training docs after role freeze."""
    output = Path(output)
    raw = (output / "ROLES_FROZEN.json").read_bytes()
    if json.loads(raw) != roles or roles.get("roles_frozen_before_tokenizer") is not True:
        raise ValueError("Roles must be frozen on disk before tokenizer fitting")
    spec = dict(roles["roles"]["train"], path=str(output / roles["roles"]["train"]["path"]))
    path = checked_file(spec)
    library = load_private_tokenizers(private_target, threads=threads)
    tokenizer = library.Tokenizer(library.models.BPE())
    tokenizer.pre_tokenizer = library.pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tokenizer.decoder = library.decoders.ByteLevel()
    trainer = library.trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, show_progress=False,
        special_tokens=[EOT], initial_alphabet=sorted(library.pre_tokenizers.ByteLevel.alphabet()))
    identities = hashlib.sha256()
    consumed = 0
    def iterator():
        nonlocal consumed
        for document in read_jsonl(path):
            if document["role"] != "train":
                raise ValueError("Tokenizer received a nontraining document")
            if consumed:
                identities.update(b"\n")
            identities.update(document["identity"].encode())
            consumed += 1
            yield document["text"]
    tokenizer.train_from_iterator(iterator(), trainer=trainer, length=roles["counts"]["train"])
    if consumed != roles["counts"]["train"] or tokenizer.get_vocab_size() != vocab_size:
        raise ValueError("Training-only tokenizer count or exact vocabulary-size gate failed")
    tokenizer.save(str(output / "tokenizer.json"))
    fit = dict(version=library.__version__, train_only=True, vocab_size=vocab_size,
               training_document_count=consumed, training_identity_sequence_sha256=identities.hexdigest(),
               roles_frozen_sha256=sha(raw), accepted_training_sha256=spec["sha256"],
               initial_alphabet="all256bytes", normalization="none; decoded source text unchanged",
               pretokenizer="ByteLevel(add_prefix_space=False,use_regex=True)",
               eot_id=tokenizer.token_to_id(EOT), threads=threads)
    write_json(output / "tokenizer_fit.json", fit)
    return tokenizer, fit


def verify_frozen_roles(output, roles):
    """Recheck every role before encoding any bytes after tokenizer fitting."""
    output = Path(output).resolve()
    raw = (output / "ROLES_FROZEN.json").read_bytes()
    frozen = json.loads(raw)
    if (frozen != roles or frozen.get("roles_frozen_before_tokenizer") is not True
            or set(frozen["roles"]) != {"train", "dev", "test"}
            or set(frozen["counts"]) != {"train", "dev", "test"}):
        raise ValueError("Frozen document-role receipt changed or is incomplete")
    for role, artifact_spec in frozen["roles"].items():
        path = (output / artifact_spec["path"]).resolve()
        if path != output / (role + ".accepted.jsonl"):
            raise ValueError("Frozen role artifact escaped its declared role path")
        checked_file(dict(artifact_spec, path=str(path)))
        count = 0
        for document in read_jsonl(path):
            if document.get("role") != role:
                raise ValueError("Document role disagrees with its frozen role artifact")
            count += 1
        if type(frozen["counts"][role]) is not int or count != frozen["counts"][role]:
            raise ValueError("Document count disagrees with its frozen role receipt")
    return sha(raw)


def encode_role(output, role, tokenizer, *, frozen_spec):
    output = Path(output)
    accepted_path = checked_file(dict(frozen_spec, path=str(output / (role + ".accepted.jsonl"))))
    destination = output / "sealed_test" if role == "test" else output
    if role == "test":
        destination.mkdir()
    token_path = destination / (role + ".u16")
    offsets, cursor = [], 0
    eot = tokenizer.token_to_id(EOT)
    # Ordinary text may contain the literal EOT spelling, without becoming a
    # structural delimiter. Only the appended ID is the document terminator.
    library = sys.modules["tokenizers"]
    ordinary_config = json.loads(tokenizer.to_str())
    ordinary = library.Tokenizer.from_str(json.dumps(dict(ordinary_config, added_tokens=[], post_processor=None)))
    with token_path.open("xb") as stream:
        for document in read_jsonl(accepted_path):
            ids = ordinary.encode(document["text"], add_special_tokens=False).ids
            if eot in ids or ordinary.decode(ids, skip_special_tokens=False) != document["text"]:
                raise ValueError("Fitted byte-BPE failed text/EOT round trip")
            ids = np.asarray(ids + [eot], dtype="<u2")
            stream.write(ids.tobytes())
            offsets.append(dict(identity=document["identity"], source=document["source"],
                source_index=document["source_index"], source_token_start=document["source_token_start"],
                source_token_end=document["source_token_end"], text_sha256=document["text_sha256"],
                start=cursor, end=cursor+len(ids), token_ids_sha256=sha(ids.tobytes())))
            cursor += len(ids)
    document_path = destination / (role + ".documents.json")
    write_json(document_path, offsets)
    return dict(tokens=artifact(token_path, destination), documents=artifact(document_path, destination),
                token_count=cursor, document_count=len(offsets)), offsets


def build_manifests(output, roles, tokenizer, fit, preparation, *, seq_len=CONTEXT,
                    min_fresh_targets=MIN_FRESH_TARGETS, default_stream_seed=20261001+1729):
    output = Path(output)
    frozen_hash = verify_frozen_roles(output, roles)
    if fit.get("roles_frozen_sha256") != frozen_hash:
        raise ValueError("Tokenizer fit names a different frozen document-role receipt")
    splits, offsets = {}, {}
    for role in ("train", "dev", "test"):
        splits[role], offsets[role] = encode_role(output, role, tokenizer, frozen_spec=roles["roles"][role])
        print(json.dumps(dict(stage="encoded", role=role, tokens=splits[role]["token_count"],
                              documents=splits[role]["document_count"])), flush=True)
    blocks = (splits["train"]["token_count"]-1) // seq_len
    if blocks * seq_len < min_fresh_targets:
        raise ValueError(f"Fresh-target capacity {blocks * seq_len} is below predeclared {min_fresh_targets}")
    permutation = torch.randperm(blocks, generator=torch.Generator().manual_seed(default_stream_seed)).numpy().astype("<u4")
    (output / "train.default_block_permutation.u32").write_bytes(permutation.tobytes())
    starts, lengths, indices = evaluation_plan_all_targets(offsets["dev"], seq_len)
    for name, values in (("starts", starts), ("lengths", lengths), ("document_indices", indices)):
        (output / ("dev." + name + ".i64")).write_bytes(values.tobytes())
    common = dict(schema_version=1, corpus_kind=KIND, revision=SOURCE_REVISION, source_repo=SOURCE_REPO,
        vocabulary=[tokenizer.id_to_token(i) for i in range(tokenizer.get_vocab_size())],
        vocab_size=tokenizer.get_vocab_size(), seq_len=seq_len, token_dtype="little-endian uint16",
        tokenizer_sha256=file_sha256(output / "tokenizer.json"), tokenizer_fit=fit, eot_id=fit["eot_id"],
        roles_frozen_sha256=file_sha256(output / "ROLES_FROZEN.json"),
        preparation=preparation, evaluation_policy_version=2,
        evaluation_boundary_rule="Every within-document target after first token, including EOS; tails/short docs right padded",
        target_padding_id=-100, input_padding_id=fit["eot_id"], target_count_unit="BPE_tokens")
    training = dict(common, role="training_and_development_only", splits={"train":splits["train"], "val":splits["dev"]},
        tokenizer=artifact(output / "tokenizer.json", output),
        dev_starts=artifact(output / "dev.starts.i64", output),
        dev_lengths=artifact(output / "dev.lengths.i64", output),
        dev_document_indices=artifact(output / "dev.document_indices.i64", output),
        default_permutation=artifact(output / "train.default_block_permutation.u32", output),
        default_stream_seed=default_stream_seed, torch_version=str(torch.__version__),
        train_blocks=blocks, fresh_target_capacity=blocks*seq_len,
        training_boundary_rule="original documents+EOS packed into disjoint target blocks; blocks may cross documents",
        train_stream_rule="torch.randperm(entire block population); common prefixes across batches/horizons; no recycling",
        dev_full_windows=len(starts), dev_target_count=int(lengths.sum()),
        dev_scored_documents=len(set(indices.tolist())), dev_zero_target_documents=len(offsets["dev"])-len(set(indices.tolist())))
    write_json(output / "training_manifest.json", training)
    documents = []
    for document in offsets["test"]:
        ss, ll, _ = evaluation_plan_all_targets([dict(start=0, end=document["end"]-document["start"])], seq_len)
        documents.append(dict(document, full_starts=ss.tolist(), curve_starts=ss.tolist(),
            full_lengths=ll.tolist(), curve_lengths=ll.tolist(),
            counts=dict(full_windows=len(ss), curve_windows=len(ss), full_target_tokens=int(ll.sum()), curve_target_tokens=int(ll.sum())),
            inclusion_reason="Every first-identity complete interval document passing prior-exposure protection"))
    windows = sum(row["counts"]["full_windows"] for row in documents)
    targets = sum(row["counts"]["full_target_tokens"] for row in documents)
    test = dict(common, role="sealed_confirmation", scoring_authorization="requires separate frozen complete comparison family",
        splits={"test":splits["test"]}, documents=documents, document_count=len(documents), full_windows=windows,
        scored_document_count=sum(bool(row["full_starts"]) for row in documents),
        zero_target_documents=sum(not row["full_starts"] for row in documents),
        curve_rule="all within-document targets; no selected subset",
        totals=dict(full_windows=windows, curve_windows=windows, full_target_tokens=targets, curve_target_tokens=targets))
    write_json(output / "sealed_test/manifest.json", test)
    (output / "sealed_test/manifest.sha256").write_text(sha(canonical(test))+"\n")
    receipt = dict(training_manifest_sha256=sha(canonical(training)), test_manifest_sha256=sha(canonical(test)),
        fresh_target_capacity=blocks*seq_len, counts=roles["counts"], no_model_inference=True, finished_unix=time.time())
    write_json(output / "MANIFESTS_PREPARED.json", receipt)
    return receipt


@contextmanager
def preparation_limits(threads=8, memory_gib=32, seconds=7200):
    """Hard process-local ceiling for explicit preparation, never set at import."""
    if not 1 <= threads <= 8 or not 0 < memory_gib <= 32 or not 0 < seconds <= 7200:
        raise ValueError("Preparation exceeds the reviewed CPU/memory/time cap")
    previous_limit = resource.getrlimit(resource.RLIMIT_AS)
    cap = int(memory_gib * 1024**3)
    soft = min(cap, previous_limit[0]) if previous_limit[0] != resource.RLIM_INFINITY else cap
    resource.setrlimit(resource.RLIMIT_AS, (soft, previous_limit[1]))
    previous_handler = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError("Reviewed data-preparation time budget exhausted")
    signal.signal(signal.SIGALRM, expired)
    signal.alarm(int(seconds))
    torch.set_num_threads(threads)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        resource.setrlimit(resource.RLIMIT_AS, previous_limit)


def validate_preparation_config(config):
    """Independent gates remain binding even if a caller edits runtime config."""
    plan = checked_json(config["plan"])
    data = plan["data"]
    if (plan["candidate"] != "reference_directed_D"
            or plan["status"] != "data_and_numerical_qualification_only"
            or data["revision"] != SOURCE_REVISION or data["source_repo"] != SOURCE_REPO
            or data["training_shards"] != [1, 2, 3]
            or data["maximum_training_source_tokens"] != 300000000
            or data["development_prefix_tokens"] != 1048576
            or data["test_intervals"] != 16 or data["test_interval_source_tokens"] != 65536
            or data["vocab_size"] != VOCAB_SIZE or data["minimum_fresh_targets"] != MIN_FRESH_TARGETS
            or data["document_roles_frozen_before_tokenizer"] is not True
            or data["decoder_roundtrip_qualification"] is not True
            or plan["no_model_inference_during_preparation"] is not True):
        raise ValueError("Configuration does not match reviewed Candidate-D data contract")
    provenance = checked_json(config["provenance"])
    metadata = provenance["metadata"]
    if metadata["repo_sha"] != SOURCE_REVISION:
        raise ValueError("Public source metadata revision mismatch")
    names = [f"fineweb_train_{index:06d}.bin" for index in (1, 2, 3, 72)] + ["fineweb_val_000000.bin"]
    public = {row["rfilename"]: row for row in metadata["files"]}
    if set(config["source_paths"]) != set(names) or any(name not in public for name in names):
        raise ValueError("Exactly the five declared source shards are required")
    for name in names:
        if public[name]["size"] != 200001024 or public[name]["lfs"]["size"] != 200001024:
            raise ValueError("Public source size mismatch")
    if sum(public[name]["size"] for name in names) > data["maximum_source_bytes"]:
        raise ValueError("Declared source byte cap exceeded")
    audit = checked_json(config["prior_use_audit"])
    if (audit.get("shard72_not_previously_used_for_model_evaluation") is not True
            or audit.get("gate_passed") is not True
            or audit.get("source_revision") != SOURCE_REVISION
            or audit.get("source_sha256") != public["fineweb_train_000072.bin"]["lfs"]["sha256"]):
        raise ValueError("Independent shard-72 prior-use audit has not passed")
    output = Path(config["output_root"]).resolve()
    study_root = Path.cwd().resolve()
    if not (study_root / "run").is_file() or not output.is_relative_to(study_root):
        raise ValueError("Run through ./run; all preparation outputs must stay in this study")
    if not Path(config["private_tokenizers_target"]).resolve().is_relative_to(study_root):
        raise ValueError("Preparation dependencies must be private to this study")
    return plan, provenance, public, audit, output


def prepare(config_path):
    """Explicit orchestration; never downloads data, launches a job or scores a model."""
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_bytes())
    plan, provenance, public, prior_audit, output = validate_preparation_config(config)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "PREPARATION_STARTED.json", dict(config_sha256=file_sha256(config_path),
        source_sha256=file_sha256(__file__), started_unix=time.time(), no_model_inference=True))
    write_json(output / "PREPARATION_CONFIG_COPY.json", config)
    try:
        limits = plan["cpu_preparation"]
        with preparation_limits(limits["maximum_threads"], limits["maximum_memory_gib"], limits["maximum_hours"]*3600):
            sources, source_receipts = {}, {}
            for name, path in config["source_paths"].items():
                sources[name], source_receipts[name] = verify_source(path, public[name]["lfs"]["sha256"])
            write_json(output / "SOURCE_VERIFIED.json", dict(revision=SOURCE_REVISION, sources=source_receipts,
                provenance_sha256=config["provenance"]["sha256"], prior_use_audit_sha256=config["prior_use_audit"]["sha256"]))
            decoder = GPT2Decoder(config["decoder"], config["private_tokenizers_target"], threads=limits["maximum_threads"])
            roles = freeze_roles(output, sources, decoder, config["prior_exposure"])
            decoder_receipt = dict(decoder.qualification, complete_documents_roundtripped=decoder.roundtrip_documents,
                                   complete_tokens_roundtripped=decoder.roundtrip_tokens)
            write_json(output / "DECODER_QUALIFIED.json", decoder_receipt)
            # Releasing source mappings and the GPT-2 tokenizer bounds fit memory.
            sources.clear()
            del decoder
            tokenizer, fit = train_byte_tokenizer(output, roles, config["private_tokenizers_target"],
                                                  threads=limits["maximum_threads"])
            provenance_record = dict(plan_sha256=config["plan"]["sha256"], config_sha256=file_sha256(config_path),
                source_verification_sha256=file_sha256(output / "SOURCE_VERIFIED.json"),
                decoder_qualification_sha256=file_sha256(output / "DECODER_QUALIFIED.json"),
                prior_exposure=config["prior_exposure"], preparation_source_sha256=file_sha256(__file__),
                resource_limits=limits, no_model_inference=True)
            receipt = build_manifests(output, roles, tokenizer, fit, provenance_record)
            # Final loader qualification validates all training/eval boundaries;
            # it does not open the sealed confirmation panel or perform inference.
            load_training_corpus(output, expected_manifest_sha256=receipt["training_manifest_sha256"])
            write_json(output / "PREPARATION_COMPLETE.json", receipt)
            return receipt
    except BaseException as error:
        import traceback
        write_json(output / "PREPARATION_FAILED.json", dict(error=repr(error), traceback=traceback.format_exc(),
                   failed_unix=time.time(), no_model_inference=True))
        raise


def verified_mmap(root, spec, dtype):
    root = Path(root).resolve()
    path = (root / spec["path"]).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Artifact escaped data directory")
    checked_file(dict(spec, path=str(path)))
    return np.memmap(path, mode="r", dtype=dtype)


class FineWebCorpus(TokenCorpus):
    """Generic token-corpus contract with compact mmap storage for train tokens."""
    def __init__(self, train_ids, val_ids, vocabulary, manifest, dev_starts, permutation,
                 dev_documents, dev_lengths, dev_document_indices):
        # TokenCorpus only converts the tiny placeholder; the immutable uint16
        # training map is installed afterward, avoiding a full int64 pool copy.
        super().__init__({"train": np.zeros(1, dtype=np.uint16), "val": val_ids}, vocabulary,
                         manifest, dev_starts, permutation, dev_documents, dev_lengths, dev_document_indices)
        self._arrays["train"] = train_ids
        self.boundaries["train"] = (0, len(train_ids))

    def close(self):
        """Release the source mapping before fixture deletion or explicit teardown."""
        self._arrays["train"]._mmap.close()

    def batch(self, split, batch_size, seq_len, *, generator=None, positions=None,
              device="cpu", return_positions=False):
        if split != "train":
            return super().batch(split, batch_size, seq_len, generator=generator, positions=positions,
                                 device=device, return_positions=return_positions)
        if seq_len != self.manifest["seq_len"] or batch_size < 1:
            raise ValueError("Positive batch and frozen context required")
        if positions is not None and generator is not None:
            raise ValueError("Positions or generator, not both")
        starts = (self.window_positions(split, batch_size, seq_len, generator=generator)
                  if positions is None else torch.as_tensor(positions, device="cpu"))
        if starts.dtype not in (torch.int32, torch.int64) or starts.shape != (batch_size,):
            raise ValueError("Integer canonical starts required")
        if ((starts < 0).any() or (starts % seq_len != 0).any()
                or (starts + seq_len >= len(self._arrays["train"])).any()):
            raise ValueError("Training start outside frozen complete-block stream")
        indices = starts.numpy()[:, None] + np.arange(seq_len+1)[None, :]
        values = torch.from_numpy(self._arrays["train"][indices].astype(np.int64))
        x, y = values[:, :-1].contiguous().to(device), values[:, 1:].contiguous().to(device)
        return (x, y, starts.clone()) if return_positions else (x, y)


def load_training_corpus(path, *, expected_manifest_sha256=None):
    """No tokenizer dependency, full train-pool copies, or sealed-panel access."""
    path = Path(path)
    manifest_path = path / "training_manifest.json" if path.is_dir() else path
    if manifest_path.name != "training_manifest.json":
        raise ValueError("Expected the explicit FineWeb training_manifest.json")
    root = manifest_path.parent
    raw = manifest_path.read_bytes()
    if expected_manifest_sha256 is not None and sha(raw) != expected_manifest_sha256:
        raise ValueError("FineWeb training manifest hash mismatch")
    manifest = json.loads(raw)
    if (manifest["corpus_kind"] != KIND or manifest["role"] != "training_and_development_only"
            or set(manifest["splits"]) != {"train", "val"} or manifest["evaluation_policy_version"] != 2):
        raise ValueError("Wrong FineWeb corpus kind, role or evaluation policy")
    if len(manifest["vocabulary"]) != manifest["vocab_size"]:
        raise ValueError("Vocabulary size mismatch")
    checked_file(dict(manifest["tokenizer"], path=str(root / manifest["tokenizer"]["path"])))
    arrays, offsets = {}, {}
    for split, spec in manifest["splits"].items():
        values = verified_mmap(root, spec["tokens"], "<u2")
        documents_path = checked_file(dict(spec["documents"], path=str(root / spec["documents"]["path"])))
        documents = json.loads(documents_path.read_bytes())
        if len(values) != spec["token_count"] or len(documents) != spec["document_count"]:
            raise ValueError("Token/document count mismatch")
        cursor = 0
        for document in documents:
            end = document["end"]
            if (document["start"] != cursor or not cursor < end <= len(values)
                    or sha(values[cursor:end].tobytes()) != document["token_ids_sha256"]
                    or values[end-1] != manifest["eot_id"]):
                raise ValueError("Document token integrity or EOS boundary failure")
            cursor = end
        if cursor != len(values):
            raise ValueError("Document boundaries do not cover token stream")
        for start in range(0, len(values), 1 << 20):
            if values[start:start+(1 << 20)].max(initial=0) >= manifest["vocab_size"]:
                raise ValueError("Token outside fitted vocabulary")
        arrays[split], offsets[split] = values, documents
    starts = verified_mmap(root, manifest["dev_starts"], "<i8")
    lengths = verified_mmap(root, manifest["dev_lengths"], "<i8")
    indices = verified_mmap(root, manifest["dev_document_indices"], "<i8")
    expected = evaluation_plan_all_targets(offsets["val"], manifest["seq_len"])
    if any(not np.array_equal(actual, wanted) for actual, wanted in zip((starts, lengths, indices), expected)):
        raise ValueError("FineWeb complete-target development coverage mismatch")
    if len(starts) != manifest["dev_full_windows"] or int(lengths.sum()) != manifest["dev_target_count"]:
        raise ValueError("FineWeb development denominator mismatch")
    permutation = verified_mmap(root, manifest["default_permutation"], "<u4")
    blocks = (len(arrays["train"])-1) // manifest["seq_len"]
    if (manifest["train_blocks"] != blocks or manifest["fresh_target_capacity"] != blocks*manifest["seq_len"]
            or not np.array_equal(np.sort(permutation), np.arange(blocks))):
        raise ValueError("Fresh capacity or bijective block permutation mismatch")
    return FineWebCorpus(arrays["train"], arrays["val"], manifest["vocabulary"], manifest, starts,
                         permutation, offsets["val"], lengths, indices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare"])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.config), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

"""Immutable, data-only Shakespeare confirmation panel; no model or scoring API.

Preparation reads only the pinned MIT archive, original vocabulary, and data
audit. It never downloads, imports training code, or computes model losses.
The panel's sealed role requires a separate recipe lock before model scoring.
"""

from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import re
import tarfile

import numpy as np


ORIGINAL_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
ARCHIVE_SHA256 = "cff31766fca4cb49e017b3c523a104c7174e70d150801195ebab4bb9291fb218"
AUDIT_SHA256 = "a6255fffcc1b3f92d20c0815fdbddd06348b42ca7f9300f3d8af4dc2d7663ec4"
SOURCE_COMMIT = "6b82db852c7322dd33e95db347f0ddfc812c6409"
SEQ_LEN = 128
CURVE_WINDOWS_PER_PLAY = 32


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def _canonical_json(value):
    return (json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def _verify_bytes(path, expected):
    raw = Path(path).read_bytes()
    if sha256(raw) != expected:
        raise ValueError(f"SHA256 mismatch: {path}")
    return raw


def extract_mit_speech(raw):
    """Exactly the extraction frozen in audit_mit.py; unsupported text survives."""
    chunks = []
    source = raw.decode("latin1")
    for name, body in re.findall(r"<A NAME=([^ >]+)>(.*?)</A>", source, re.I | re.S):
        text = html.unescape(re.sub("<[^>]+>", "", body))
        if name.startswith("speech"):
            chunks.append("\n" + text + ":\n")
        elif re.fullmatch(r"\d+\.\d+\.\d+", name):
            chunks.append(text + "\n")
    extracted = "".join(chunks).lstrip("\n")
    return re.sub(r"\[[^\]]*\]", "", extracted).replace("\t", " ")


def encode_with_oov(text, vocabulary):
    """Keep the trained ID mapping; -1 records unsupported characters unchanged."""
    if len(vocabulary) != len(set(vocabulary)) or len(vocabulary) > 32767:
        raise ValueError("Vocabulary must be unique and fit positive int16 IDs")
    mapping = {character: index for index, character in enumerate(vocabulary)}
    return np.asarray([mapping.get(character, -1) for character in text], dtype="<i2")


def window_plan(tokens, seq_len=SEQ_LEN, curve_count=CURVE_WINDOWS_PER_PLAY):
    """128-stride complete windows; remove ANY containing OOV, including target.

    Curve starts are a deterministic evenly spaced subset of valid full starts.
    Integer floor spacing includes the first and last valid starts. No resampling,
    random seed, model score, or cross-document concatenation is involved.
    """
    if seq_len < 1 or curve_count < 1:
        raise ValueError("Positive sequence length and curve count required")
    tokens = np.asarray(tokens)
    if tokens.ndim != 1:
        raise ValueError("Token array must be one-dimensional")
    candidates = np.arange(0, max(0, len(tokens) - seq_len), seq_len, dtype="<i8")
    invalid_prefix = np.concatenate(([0], np.cumsum(tokens < 0, dtype=np.int64)))
    valid = invalid_prefix[candidates + seq_len + 1] == invalid_prefix[candidates]
    full = candidates[valid]
    if len(full) < curve_count:
        raise ValueError("Document has fewer valid full windows than curve_count")
    if curve_count == 1:
        indices = np.array([0], dtype=np.int64)
    else:
        indices = np.arange(curve_count, dtype=np.int64) * (len(full) - 1) // (curve_count - 1)
    curve = full[indices]
    last_candidate_end = int(candidates[-1]) + seq_len + 1 if len(candidates) else 0
    counts = dict(
        candidate_full_windows=len(candidates), full_windows=len(full),
        curve_windows=len(curve), excluded_oov_windows=int((~valid).sum()),
        excluded_oov_starts=candidates[~valid].tolist(),
        full_target_characters=len(full) * seq_len,
        curve_target_characters=len(curve) * seq_len,
        unscored_initial_characters=1 if len(candidates) else 0,
        unused_trailing_characters=len(tokens) - last_candidate_end,
        oov_characters=int((tokens < 0).sum()),
    )
    return full, curve, counts


@dataclass(frozen=True)
class ConfirmationDocument:
    slug: str
    text: str
    token_ids: np.ndarray
    full_starts: np.ndarray
    curve_starts: np.ndarray
    manifest: dict

    def starts(self, bank):
        if bank == "full":
            return self.full_starts
        if bank == "curve":
            return self.curve_starts
        raise ValueError("Bank must be full or curve")


@dataclass(frozen=True)
class ConfirmationPanel:
    path: Path
    manifest: dict
    manifest_sha256: str
    documents: tuple

    @property
    def vocabulary(self):
        return tuple(self.manifest["vocabulary"])

    @property
    def seq_len(self):
        return self.manifest["seq_len"]


def _write_panel(destination, documents, vocabulary, provenance, *, seq_len=SEQ_LEN,
                 curve_count=CURVE_WINDOWS_PER_PLAY, role="sealed_confirmation"):
    """Write once; existing directories and partial artifacts are never reused."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite panel: {destination}")
    # Qualify all inputs before creating any output directory.
    planned = []
    seen = set()
    for source in documents:
        slug, text = source["slug"], source["text"]
        if not re.fullmatch(r"[a-z0-9_]+", slug) or slug in seen:
            raise ValueError("Unique safe document slugs required")
        seen.add(slug)
        tokens = encode_with_oov(text, vocabulary)
        full, curve, counts = window_plan(tokens, seq_len, curve_count)
        artifacts = {"text": (f"{slug}.txt", text.encode()),
                     "tokens": (f"{slug}.tokens.i16", tokens.astype("<i2").tobytes()),
                     "full_starts": (f"{slug}.full_starts.i64", full.astype("<i8").tobytes()),
                     "curve_starts": (f"{slug}.curve_starts.i64", curve.astype("<i8").tobytes())}
        row = {key: value for key, value in source.items() if key != "text"}
        row.update(characters=len(text), counts=counts,
                   oov_counts={char: text.count(char) for char in sorted(set(text) - set(vocabulary))},
                   artifacts={key: dict(path=name, bytes=len(raw), sha256=sha256(raw))
                              for key, (name, raw) in artifacts.items()})
        planned.append((row, artifacts))
    if not planned:
        raise ValueError("Panel requires documents")
    manifest = dict(
        schema_version=1, role=role, scoring_authorization="requires_separate_frozen_recipe_and_cohort_lock",
        seq_len=seq_len, curve_windows_per_play=curve_count,
        vocabulary=list(vocabulary), vocab_size=len(vocabulary),
        vocabulary_sha256=sha256("".join(vocabulary).encode()),
        vocabulary_rule="Original trained sorted65 alphabet; no remapping or extension",
        token_dtype="little-endian signed int16; OOV=-1", starts_dtype="little-endian signed int64",
        full_window_rule="starts range(0, len(document)-seq_len, seq_len); retain iff all seq_len+1 IDs >= 0",
        curve_window_rule="full_starts[floor(i*(n_full-1)/(curve_count-1))], i=0..curve_count-1",
        oov_rule="Preserve text and encode OOV=-1; exclude entire input+target window, no substitution",
        boundary_rule="No window crosses a play boundary; each document has independent starts",
        aggregation_plan="Report token-weighted NLL plus all per-play NLLs and equal-play mean; paired seeds",
        provenance=provenance, documents=[row for row, _ in planned],
        totals={key: sum(row["counts"][key] for row, _ in planned)
                for key in ("candidate_full_windows", "full_windows", "curve_windows", "excluded_oov_windows",
                            "full_target_characters", "curve_target_characters", "oov_characters",
                            "unscored_initial_characters", "unused_trailing_characters")},
        document_count=len(planned), characters=sum(row["characters"] for row, _ in planned),
    )
    destination.mkdir(parents=True, exist_ok=False)
    for _, artifacts in planned:
        for name, raw in artifacts.values():
            with (destination / name).open("xb") as stream:
                stream.write(raw)
    manifest_bytes = _canonical_json(manifest)
    with (destination / "manifest.json").open("xb") as stream:
        stream.write(manifest_bytes)
    with (destination / "manifest.sha256").open("x") as stream:
        stream.write(sha256(manifest_bytes) + "\n")
    return load_panel(destination, expected_manifest_sha256=sha256(manifest_bytes))


def prepare_mit_panel(destination, *, archive_path, audit_path, original_path):
    """Prepare every eligible audited complete play, with all source hashes pinned."""
    if Path(destination).exists():
        raise FileExistsError(f"Refusing to overwrite panel: {destination}")
    _verify_bytes(archive_path, ARCHIVE_SHA256)
    audit = json.loads(_verify_bytes(audit_path, AUDIT_SHA256))
    original = _verify_bytes(original_path, ORIGINAL_SHA256).decode()
    vocabulary = tuple(sorted(set(original)))
    if len(vocabulary) != 65:
        raise ValueError("Expected the existing 65-character vocabulary")
    if audit["archive_sha256"] != ARCHIVE_SHA256 or audit["original_corpus_sha256"] != ORIGINAL_SHA256:
        raise ValueError("Audit provenance mismatch")
    eligible = [row for row in audit["documents"] if row["eligible"]]
    if len(eligible) != 28 or [r["slug"] for r in eligible] != audit["eligible_slugs"]:
        raise ValueError("Expected all 28 eligible plays in original audit order")
    documents = []
    with tarfile.open(archive_path) as archive:
        for row in eligible:
            raw = archive.extractfile(row["archive_member"]).read()
            text = extract_mit_speech(raw)
            if sha256(raw) != row["html_sha256"] or sha256(text.encode()) != row["normalized_sha256"]:
                raise ValueError(f"Audited extraction/hash mismatch: {row['slug']}")
            oov = {c: text.count(c) for c in sorted(set(text) - set(vocabulary))}
            if len(text) != row["normalized_characters"] or oov != row["normalized_oov"]:
                raise ValueError(f"Audited character/OOV mismatch: {row['slug']}")
            if any(row["overlapping_ngram_occurrences"][str(n)] for n in (12, 20, 32)) or row["overlapping_100letter_occurrences"]:
                raise ValueError("Eligible document failed the recorded overlap gate")
            documents.append(dict(slug=row["slug"], title=row["title"], text=text,
                                  source_url=row["source_url"], html_sha256=row["html_sha256"],
                                  inclusion_reason="All eligible canonical plays; zero shared20word or100letter sequence vs entire old corpus",
                                  overlap_audit={k: row[k] for k in ("overlapping_ngram_occurrences", "overlapping_100letter_occurrences")}))
    provenance = dict(archive_sha256=ARCHIVE_SHA256, original_corpus_sha256=ORIGINAL_SHA256,
                      audit_sha256=AUDIT_SHA256, github_commit=SOURCE_COMMIT,
                      source_url="https://shakespeare.mit.edu/",
                      archive_url=f"https://api.github.com/repos/TheMITTech/shakespeare/tarball/{SOURCE_COMMIT}",
                      preparation_source_sha256=sha256(Path(__file__).read_bytes()),
                      excluded_play_slugs=audit["excluded_slugs"],
                      excluded_reason="Whole plays excluded if any part shares normalized20word or100letter sequence with entire original corpus",
                      normalization="Audited numeric/speaker anchors, entity decoding, [] stage-span removal, tabs to spaces")
    return _write_panel(destination, documents, vocabulary, provenance)


def load_panel(path, *, expected_manifest_sha256=None):
    """Pure local data load; verify every byte, encoding, boundary, and bank rule.

    Caller scoring code must additionally enforce its frozen recipe lock. Loading
    a data panel here is not permission to evaluate an unlocked model against it.
    Arrays returned are read-only; this API has no torch or model dependency.
    """
    path = Path(path)
    raw = (path / "manifest.json").read_bytes()
    digest = sha256(raw)
    if digest != (path / "manifest.sha256").read_text().strip():
        raise ValueError("Manifest checksum mismatch")
    if expected_manifest_sha256 is not None and digest != expected_manifest_sha256:
        raise ValueError("Panel differs from externally pinned manifest")
    manifest = json.loads(raw)
    if manifest["schema_version"] != 1 or manifest["role"] not in ("sealed_confirmation", "synthetic_contract_test"):
        raise ValueError("Unsupported panel schema or role")
    vocabulary = tuple(manifest["vocabulary"])
    if len(vocabulary) != manifest["vocab_size"] or sha256("".join(vocabulary).encode()) != manifest["vocabulary_sha256"]:
        raise ValueError("Vocabulary integrity failure")
    if manifest["role"] == "sealed_confirmation" and (
        len(vocabulary) != 65 or manifest["document_count"] != 28 or manifest["seq_len"] != SEQ_LEN
        or manifest["curve_windows_per_play"] != CURVE_WINDOWS_PER_PLAY
    ):
        raise ValueError("Sealed panel dimensions differ from declaration")
    documents = []
    seen = set()
    for row in manifest["documents"]:
        slug = row["slug"]
        if slug in seen or not re.fullmatch(r"[a-z0-9_]+", slug):
            raise ValueError("Invalid document identity")
        seen.add(slug)
        artifacts = {}
        for key, spec in row["artifacts"].items():
            if Path(spec["path"]).name != spec["path"]:
                raise ValueError("Artifact paths must be local basenames")
            blob = _verify_bytes(path / spec["path"], spec["sha256"])
            if len(blob) != spec["bytes"]:
                raise ValueError("Artifact length mismatch")
            artifacts[key] = blob
        text = artifacts["text"].decode()
        tokens = np.frombuffer(artifacts["tokens"], dtype="<i2")
        full = np.frombuffer(artifacts["full_starts"], dtype="<i8")
        curve = np.frombuffer(artifacts["curve_starts"], dtype="<i8")
        encoded = encode_with_oov(text, vocabulary)
        expected_full, expected_curve, counts = window_plan(tokens, manifest["seq_len"], manifest["curve_windows_per_play"])
        if not np.array_equal(tokens, encoded) or len(text) != row["characters"]:
            raise ValueError("Text/token integrity failure")
        if not np.array_equal(full, expected_full) or not np.array_equal(curve, expected_curve) or counts != row["counts"]:
            raise ValueError("Evaluation bank or denominator integrity failure")
        if {c: text.count(c) for c in sorted(set(text) - set(vocabulary))} != row["oov_counts"]:
            raise ValueError("OOV accounting integrity failure")
        documents.append(ConfirmationDocument(slug, text, tokens, full, curve, row))
    if len(documents) != manifest["document_count"] or sum(len(d.text) for d in documents) != manifest["characters"]:
        raise ValueError("Panel document/character accounting failure")
    for key, total in manifest["totals"].items():
        if sum(d.manifest["counts"][key] for d in documents) != total:
            raise ValueError(f"Panel total mismatch: {key}")
    return ConfirmationPanel(path.resolve(), manifest, digest, tuple(documents))

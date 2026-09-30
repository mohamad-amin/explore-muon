"""Pinned Tiny Shakespeare source and deterministic character-window sampling.

Nothing downloads at import. ``prepare_tiny_shakespeare`` is the only download
entry point. Splits are contiguous, exclusive character ranges; even the target
of the last permissible training window stays inside its training split.
"""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import urllib.request

import torch


SOURCE_COMMIT = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
SOURCE_URL = (f"https://raw.githubusercontent.com/karpathy/char-rnn/{SOURCE_COMMIT}"
              "/data/tinyshakespeare/input.txt")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def prepare_tiny_shakespeare(directory, *, expected_sha256=None):
    """Explicitly fetch the commit-pinned original corpus, or verify its cache.

Returns ``directory/input.txt``. Existing bytes are never silently overwritten.
If a prior source manifest exists, its checksum must match the bytes on disk.
Callers can additionally require a predeclared ``expected_sha256``.
    """
    directory = Path(directory)
    destination = directory / "input.txt"
    manifest_path = directory / "source.json"
    prior = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if destination.exists():
        raw = destination.read_bytes()
        if prior is None and expected_sha256 is None:
            raise ValueError("Unmanifested cached input needs an explicit expected_sha256")
    else:
        if prior is not None:
            raise ValueError("Source manifest exists but input.txt is missing")
        with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
            raw = response.read()
    digest = sha256_bytes(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Source SHA256 mismatch: {digest}")
    if prior is not None and (prior["sha256"] != digest or prior["source_url"] != SOURCE_URL):
        raise ValueError("Cached source differs from its pinned manifest")
    decoded = raw.decode("utf-8")
    if len(decoded) < 2:
        raise ValueError("Corpus must contain at least two characters")
    source = dict(source_url=SOURCE_URL, source_commit=SOURCE_COMMIT,
                  sha256=digest, bytes=len(raw), characters=len(decoded))
    directory.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        with destination.open("xb") as stream:
            stream.write(raw)
    if not manifest_path.exists():
        with manifest_path.open("x") as stream:
            json.dump(source, stream, indent=2, sort_keys=True)
            stream.write("\n")
    return destination


@dataclass
class CharacterCorpus:
    """CPU character IDs with immutable split boundaries and sorted vocabulary.

Vocabulary is defined from the whole corpus, as in Karpathy's character setup;
this shares the alphabet, never training windows or targets, across splits.
``positions`` passed to :meth:`batch` are local to the chosen split.
    """

    token_ids: torch.Tensor
    vocabulary: tuple
    boundaries: dict
    manifest: dict

    @property
    def vocab_size(self):
        return len(self.vocabulary)

    def encode(self, text):
        mapping = {character: index for index, character in enumerate(self.vocabulary)}
        return torch.tensor([mapping[character] for character in text], dtype=torch.long)

    def decode(self, ids):
        return "".join(self.vocabulary[int(index)] for index in ids)

    def tokens(self, split):
        start, end = self.boundaries[split]
        return self.token_ids[start:end]

    def window_positions(self, split, batch_size, seq_len, *, generator):
        if batch_size < 1 or seq_len < 1:
            raise ValueError("batch_size and seq_len must be positive")
        possibilities = len(self.tokens(split)) - seq_len
        if possibilities < 1:
            raise ValueError("Split is too short for a complete input/target window")
        if generator is None or generator.device.type != "cpu":
            raise ValueError("Provide an explicit CPU torch.Generator for paired sampling")
        return torch.randint(possibilities, (batch_size,), generator=generator)

    def batch(self, split, batch_size, seq_len, *, generator=None, positions=None,
              device="cpu", return_positions=False):
        """Sample paired next-character inputs/targets with explicit RNG or starts.

Passing explicit starts consumes no random numbers. Returned tensors can be
copied to CUDA, while sampling is always on CPU and independent of CUDA RNG.
        """
        if batch_size < 1 or seq_len < 1:
            raise ValueError("batch_size and seq_len must be positive")
        if positions is not None and generator is not None:
            raise ValueError("Specify positions or a generator, not both")
        if positions is None:
            positions = self.window_positions(split, batch_size, seq_len, generator=generator)
        else:
            positions = torch.as_tensor(positions, device="cpu")
            if positions.dtype not in (torch.int32, torch.int64):
                raise ValueError("Window positions must be integer indices")
            positions = positions.long()
        source = self.tokens(split)
        if positions.shape != (batch_size,):
            raise ValueError("positions must contain exactly batch_size starts")
        if (positions < 0).any() or (positions + seq_len >= len(source)).any():
            raise ValueError("A window or its final target crosses the split boundary")
        indices = positions[:, None] + torch.arange(seq_len + 1)[None, :]
        windows = source[indices]
        x, y = windows[:, :-1].contiguous().to(device), windows[:, 1:].contiguous().to(device)
        return (x, y, positions.clone()) if return_positions else (x, y)


def corpus_from_text(text, *, train_fraction=0.9, test_fraction=0.0, source=None):
    """Build deterministic IDs/splits; optional final test tail stays separate."""
    if not 0 < train_fraction < 1 or not 0 <= test_fraction < 1 - train_fraction:
        raise ValueError("Fractions need train > 0, validation > 0 and test >= 0")
    vocabulary = tuple(sorted(set(text)))
    mapping = {character: index for index, character in enumerate(vocabulary)}
    ids = torch.tensor([mapping[character] for character in text], dtype=torch.long)
    train_end = int(len(text) * train_fraction)
    val_end = int(len(text) * (1 - test_fraction))
    boundaries = {"train": (0, train_end), "val": (train_end, val_end)}
    if test_fraction > 0:
        boundaries["test"] = (val_end, len(text))
    if any(end - start < 2 for start, end in boundaries.values()):
        raise ValueError("Every requested split needs at least two characters")
    manifest = dict(
        schema_version=1, source=source, raw_utf8_sha256=sha256_bytes(text.encode("utf-8")),
        characters=len(text), vocabulary=list(vocabulary), vocab_size=len(vocabulary),
        vocabulary_rule="sorted unique Unicode characters of entire source",
        train_fraction=train_fraction, test_fraction=test_fraction,
        split_rule="contiguous, exclusive character offsets; windows include one next-token target",
        splits={name: dict(start=start, end=end, characters=end-start,
                           utf8_sha256=sha256_bytes(text[start:end].encode("utf-8")),
                           token_ids_sha256=sha256_bytes(ids[start:end].numpy().astype("<i8").tobytes()))
                for name, (start, end) in boundaries.items()},
    )
    return CharacterCorpus(ids, vocabulary, boundaries, manifest)


def load_corpus(path, *, train_fraction=0.9, test_fraction=0.0, expected_sha256=None):
    """Load local bytes only; never downloads or modifies a source file."""
    path = Path(path)
    if path.is_dir():
        path = path / "input.txt"
    raw = path.read_bytes()
    digest = sha256_bytes(raw)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Source SHA256 mismatch: {digest}")
    manifest_path = path.with_name("source.json")
    source = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"path": str(path.resolve())}
    if "sha256" in source and source["sha256"] != digest:
        raise ValueError("Corpus bytes differ from source manifest")
    return corpus_from_text(raw.decode("utf-8"), train_fraction=train_fraction,
                            test_fraction=test_fraction, source=source)

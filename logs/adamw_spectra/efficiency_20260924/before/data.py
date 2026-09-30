"""Deterministic, non-wrapping access to the repository's FineWeb token shards."""

import bisect
import glob
from pathlib import Path

import numpy as np
import torch


class TokenStream:
    def __init__(self, pattern):
        self.paths = [Path(p).resolve() for p in sorted(glob.glob(pattern))]
        if not self.paths:
            raise FileNotFoundError(f"No token shards match {pattern}")
        self.arrays, self.ends, self.manifest = [], [], []
        total = 0
        for path in self.paths:
            header = np.fromfile(path, dtype="<i4", count=256)
            if len(header) != 256 or tuple(header[:2]) != (20240520, 1):
                raise ValueError(f"Invalid FineWeb header: {path}")
            count = int(header[2])
            if count < 2 or path.stat().st_size != 1024 + 2 * count:
                raise ValueError(f"Invalid FineWeb shard length: {path}")
            self.arrays.append(np.memmap(path, dtype="<u2", mode="r", offset=1024, shape=(count,)))
            total += count
            self.ends.append(total)
            self.manifest.append({"path": str(path), "tokens": count,
                                  "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
        self.total = total

    def read(self, offset, count):
        if offset < 0 or count < 1 or offset + count > self.total:
            raise ValueError("Token stream exhausted; refusing to wrap or reuse training data")
        chunks = []
        remaining = count
        while remaining:
            index = bisect.bisect_right(self.ends, offset)
            start = self.ends[index - 1] if index else 0
            take = min(remaining, self.ends[index] - offset)
            chunks.append(np.asarray(self.arrays[index][offset - start:offset - start + take]))
            offset += take
            remaining -= take
        return np.concatenate(chunks).astype(np.int64)

    def batch(self, offset, sequences, seq_len, device):
        tokens = torch.from_numpy(self.read(offset, sequences * seq_len + 1)).to(device)
        return tokens[:-1].view(sequences, seq_len), tokens[1:].view(sequences, seq_len)


def check_disjoint(train, validation):
    train_ids = {(p.stat().st_dev, p.stat().st_ino) for p in train.paths}
    val_ids = {(p.stat().st_dev, p.stat().st_ino) for p in validation.paths}
    if train_ids & val_ids:
        raise ValueError("Training and validation shards overlap (including links)")

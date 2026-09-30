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
        # Copy/cast straight into the final allocation, including shard crossings.
        result = np.empty(count, dtype=np.int64)
        written = 0
        remaining = count
        while remaining:
            index = bisect.bisect_right(self.ends, offset)
            start = self.ends[index - 1] if index else 0
            take = min(remaining, self.ends[index] - offset)
            result[written:written + take] = self.arrays[index][offset - start:offset - start + take]
            offset += take
            written += take
            remaining -= take
        return result

    def device_tokens(self, offset, count, device):
        """Transfer one accumulated batch, then take microbatch views on device."""
        device = torch.device(device)
        tokens = torch.from_numpy(self.read(offset, count))
        if device.type == "cuda":
            return tokens.pin_memory().to(device, non_blocking=True)
        return tokens.to(device)

    def batch(self, offset, sequences, seq_len, device):
        tokens = self.device_tokens(offset, sequences * seq_len + 1, device)
        return token_views(tokens, 0, sequences * seq_len, seq_len)


def token_views(tokens, offset, count, seq_len):
    if count % seq_len or offset < 0 or offset + count + 1 > tokens.numel():
        raise ValueError("Invalid microbatch slice")
    return (tokens[offset:offset + count].view(-1, seq_len),
            tokens[offset + 1:offset + count + 1].view(-1, seq_len))


def check_disjoint(train, validation):
    train_ids = {(p.stat().st_dev, p.stat().st_ino) for p in train.paths}
    val_ids = {(p.stat().st_dev, p.stat().st_ino) for p in validation.paths}
    if train_ids & val_ids:
        raise ValueError("Training and validation shards overlap (including links)")

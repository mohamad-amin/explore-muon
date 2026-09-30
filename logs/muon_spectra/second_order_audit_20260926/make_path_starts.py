"""Start states for the step-vs-path probe (2026-09-30 11:40 CDT, MUON_CASE): a kept 16M checkpoint re-labelled to a
smaller batch, so newton_train.py reads the same 16M batch split into K sequential sub-batches.

A 16M run's step s+1 reads tokens [s * 16M, (s+1) * 16M). With batch_tokens = 16M / K and the start step relabelled to
s * K, steps s*K + 1 … s*K + K read the same range (each rank a different slice of each sub-batch; the union is the same
sequences). The weights, momentum buffers and auxiliary AdamW state are unchanged. The LR schedule depends on the step
only through the warmup (long over at s*K) and on tokens otherwise, so the LR factor is the same as the 16M step's.

usage: make_path_starts.py SOURCE_ARM:STEP OUT_ROOT K [K ...] [--offset B]
  (writes OUT_ROOT/start_k{K}/scientific/kept/step{(s+B)*K:06d}.pt; with --offset B the same weights read the batch B
   steps later, a replicate on other data (K = 1 then also needs its own relabelled start))
"""
import sys
from pathlib import Path

import torch


def main():
    args = sys.argv[1:]
    offset = 0
    if "--offset" in args:
        i = args.index("--offset")
        offset = int(args[i + 1])
        del args[i:i + 2]
    item, out_root = args[0], Path(args[1])
    arm, step = item.rsplit(":", 1)
    step = int(step)
    saved = torch.load(Path(arm) / "scientific" / "kept" / f"step{step:06d}.pt", map_location="cpu", weights_only=False)
    for k in (int(v) for v in args[2:]):
        config = dict(saved["config"])
        assert config["batch_tokens"] % k == 0
        config["batch_tokens"] = config["batch_tokens"] // k
        label = (step + offset) * k
        relabelled = {**saved, "config": config, "step": label,
                      "path_start_of": f"{item} re-labelled to batch_tokens / {k} (step {step} -> {label}, offset {offset})"}
        kept = out_root / f"start_k{k}" / "scientific" / "kept"
        kept.mkdir(parents=True, exist_ok=True)
        torch.save(relabelled, kept / f"step{label:06d}.pt")
        print("wrote", kept / f"step{label:06d}.pt", "batch_tokens", config["batch_tokens"])


if __name__ == "__main__":
    main()

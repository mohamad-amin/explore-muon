"""Read-only CPU mmap/schema/hash audit; does not instantiate a model."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
ROOT = Path(__file__).resolve().parents[3]
COHORT = ROOT / "logs/muon_spectra/soaudit_batch16m_20260927"
RUN = COHORT / "TS_a0.5_b0.5gn_b16M_lr0.028_mom0.9_s260925_l40s/scientific"
CHECKPOINT = RUN / "kept/step000046.pt"
STATS = RUN / "kept/step000046_input_stats_rank0.pt"
metadata = json.loads((RUN / "metadata.json").read_text())
source_hashes = {}
for name in ("muon.py", "model.py", "distributed.py", "train.py"):
    path = COHORT / "frozen/adamw_spectra" / name
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == metadata["source_sha256"][name], name
    source_hashes[name] = digest
saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=False, mmap=True)
stats = torch.load(STATS, map_location="cpu", weights_only=False, mmap=True)
banks = defaultdict(list)
for name, tensor in saved["model"].items():
    if name.startswith("blocks.") and tensor.ndim == 2:
        banks[tuple(tensor.shape)].append(name)
owners = {name: index % metadata["world_size"]
          for names in banks.values() for index, name in enumerate(names)}
state_keys = Counter(tuple(value) for value in saved["optimizer"]["state"].values())
report = {
    "checkpoint": str(CHECKPOINT), "rank0_stats": str(STATS),
    "step": saved["step"], "tokens": saved["tokens"],
    "world_size": metadata["world_size"], "source_sha256_verified": source_hashes,
    "checkpoint_keys": list(saved), "optimizer_keys": list(saved["optimizer"]),
    "optimizer_state_key_counts": {",".join(k): v for k, v in state_keys.items()},
    "owners": owners, "rank0_stats_count": len(stats),
    "stats_schema": {name: {key: {"shape": list(t.shape), "dtype": str(t.dtype)}
                            for key, t in values.items()} for name, values in stats.items()},
    "model_nonparameter_stat_keys": [k for k in saved["model"]
                                     if any(v in k for v in ("input_", "out_stats", "root"))],
    "limits": ["B/L/R absent", "rank0 input moments do not replace other owners' moments",
               "C saved after46 is not cached R last refreshed45",
               "no model or GPU computation; checkpoint payload hashes not computed"],
}
target = Path(__file__).with_name("schema.json")
target.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({"output": str(target), "step": report["step"],
                  "source_hashes_verified": len(source_hashes),
                  "body_matrices": len(owners), "rank0_stats_count": len(stats)}, indent=2))

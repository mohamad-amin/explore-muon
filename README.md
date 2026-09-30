# Explore Muon

Muon / AdamW update-spectrum experiments, moved from `../last_layer` at the
user's request. This directory is the canonical home for the code, protocols,
experiments, checkpoints, plots, comparisons and research notes.

## Start here

- [Research state](RESEARCH_STATE.md) and [research guide](RESEARCH_GUIDE.md).
- [Code and protocols](research/adamw_spectra/README.md). The historical Python
  package name `research.adamw_spectra` is retained for both optimizers.
- [Eight-layer comparison](logs/muon_spectra/comparison_depth8_20260925/README.md),
  [complete comparison PDF](logs/muon_spectra/comparison_depth8_20260925/comparison.pdf).
- `logs/adamw_spectra/`: AdamW pilots, qualification, depth experiments and results.
- `logs/muon_spectra/`: Muon experiments, qualification, comparisons and results.
- `data/fineweb10B_scaling/`: unused data staging from the superseded width-scaling
  proposal; moved with its original manifests and links. Current runs use `fineweb10B`.

## Runs

| Optimizer | Layers | State at migration | Slurm job/step | Results |
|---|---:|---|---|---|
| AdamW | 8 | Complete | 2618555.3 | [Run](logs/adamw_spectra/g20_20260924_212246_r2/README.md) |
| AdamW | 12 | Complete | 2620279 | [Run](logs/adamw_spectra/depth12_w512_20260924/README.md) |
| AdamW | 20 | Queued, 8 × 96 GB | 2620278 | [Run](logs/adamw_spectra/depth20_w512_20260924/README.md) |
| Muon | 8 | Complete | 2618555.9 | [Run](logs/muon_spectra/depth8_w512_20260925_r2/README.md) |
| Muon | 12 | Complete | 2620620 | [Run](logs/muon_spectra/depth12_w512_20260925_r2/README.md) |
| Muon | 20 | Queued, 4 × 96 GB | 2620623 | [Run](logs/muon_spectra/depth20_w512_20260925_r2/README.md) |

All depth cells keep width 512, 8 heads, context 512, global batch 1,048,576
and total 1,539,870,720 tokens. See protocols for optimizer, initialization,
measurement and comparison limits. Queue status is a dated snapshot.

## Running the tools

Run commands from this directory. The existing `.venv` and original FineWeb
shards are shared through links to `../last_layer`; versions and exact targets
are recorded in [environment.json](environment.json).

```bash
.venv/bin/python -m research.adamw_spectra.train --help
.venv/bin/python -m research.adamw_spectra.distributed --help
.venv/bin/python -m research.adamw_spectra.compare --help
.venv/bin/python -m unittest research.adamw_spectra.test_study research.adamw_spectra.test_muon
```

No package installation is needed in this workspace. Training launchers live
with each frozen experiment. Analyses consume saved spectra and do not rerun
training. New output directories are required to preserve previous results.

## Migration and historical paths

The original four directories were physically moved here on the same
filesystem. Compatibility symlinks in `../last_layer` preserve existing
submission scripts, absolute paths in metadata and old result links. The two
pending job IDs were retained and released after the move. Their controllers
resolve the new project root and write their outputs here.

Frozen sources, configs, results and checkpoints were not edited. See
[migration record](migration/move_report.json) and
[file inventory](migration/inventory_before.json). Current notes were moved
from the old research index/state; unrelated last-layer studies remain there.

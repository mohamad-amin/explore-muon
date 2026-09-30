# Depth 12, width 512: eight-GPU AdamW spectrum study

**Completed successfully.** Slurm 2620279 ran on g22 with eight RTX A6000
GPUs and exited 0:0; total allocation elapsed 00:27:45. All 1,469 updates and
1,539,870,720 tokens completed. Final validation NLL **3.8396469429135323**;
all 60 spectral snapshots and plots are present. Main training invocation
took 1474.68 seconds (24.6 minutes). See `scientific/summary.json` and
`scientific/plots/median.png`. Verified 2026-09-25 01:23 UTC. Submission
details below are retained as history.

Submitted Slurm job **2620279**, `gpu` partition, one node, **8 GPUs**, 16 CPU
cores, 128 GiB host memory, eight-hour limit. Hardware: **48 GB GPUs or 96 GB RTX Pro**.
At submission verification the job was pending for scheduler priority.
Use `squeue -j 2620279` for current status; queued does not mean training has started.

Depth is 12; width 512, eight attention heads, context 512, global batch
1,048,576 tokens, seed 260924, and total 1,539,870,720 training tokens match
the original eight-layer comparison. The fixed peak LR is 0.0012, taken from
`baseline_lr_selection.json`; no depth-specific rate sweep is run. Both
new jobs use 1,469 updates and 60 spectral samples (every 25 plus endpoints).
The 24 matrices are Q/K/V/O and MLP up/down at four relative depths.

Execution is automatic: a tiny eight-GPU numerical/collective check, five
full-size qualification updates with replica parameter/moment hashes, the
full scientific run, then PNG/PDF plots and descriptive summaries. Frozen
sources and configs are under `frozen/`, `config.json`, and
`frozen_manifest.json`. `job.sbatch` is the submitted script. No additional
condition is launched on failure.

`pipeline.json` will appear when the allocation starts. `slurm-2620279.log`
contains controller output, with stage-specific logs alongside it. Main
outputs go under `scientific/`: checkpoints, per-step metrics, singular-value
archives and final plots. The controller records execution failures explicitly.

The initialization recipe remains depth-dependent: residual-projection std
is 0.02/sqrt(2*depth). Same seed does not imply identical shared-layer weights.
The eight-layer reference used one GPU; distributed reductions/hardware can
change rounding. CPU checks verified replica state identity, correct token
weighting, dummy synchronization and numerical reference/resume agreement;
resume was not bit-identical (maximum observed parameter difference 1.86e-9).
See the frozen `SCALE_UP_CASE.md` and `DDP_CPU_QUALIFIED.json` for details.

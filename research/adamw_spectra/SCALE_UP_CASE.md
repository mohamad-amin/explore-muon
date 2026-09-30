# Eight-GPU depth-only AdamW spectrum runs

User request, 2026-09-24: submit two jobs using eight GPUs each, depths 12
and 20; prefer 96 GB RTX Pro for depth 20 and allow 48 GB or RTX Pro for 12.
Correction: **width and batch stay fixed; study depth only**. No jobs from
the earlier width-scaling interpretation were submitted. That proposal is
preserved under `logs/adamw_spectra/scaleup_20260924/superseded_width_scaling_proposal/`.

## Decision argument

The user asks how AdamW adaptive-update spectral trajectories depend on
depth while width and global batch remain fixed. Use 12- and 20-layer
versions of the existing width-512, eight-head model. Hold context, objective,
optimizer hyperparameters, data order and training horizon fixed as well,
so extra training or separate learning-rate selection does not become a
second treatment. The eight-layer pilot selected peak LR 0.0012 by its
predeclared validation criterion; reuse that rate and its schedule for both
depths, with no new per-depth pilots. The eight-layer run continues independently.

This describes update geometry; differences do not themselves establish a
learning limitation or an optimizer-specific effect. Parameter counts and
compute grow with depth: fixed data exposure is neither compute-matched nor
20N for all models. Retain the initialization rule, including residual
projection std=0.02/sqrt(2*depth), and report this depth-dependent
parameterization. A common seed does not imply identical shared-layer weights.

Use eight DDP replicas without changing the global batch. Weight gradients
by actual tokens, including the final partial batch; use graph-connected
zero-weight dummy forwards only to synchronize ranks that finish their local
microbatches earlier. Qualify single-device equivalence, optimizer replica
agreement and resume, then run bounded full-size CUDA qualification inside
each allocation. Measure the same 24 relative-depth matrices every 25
updates plus endpoints, with distributed independent exact SVDs. Each job
has an eight-hour cap, frozen sources/configs and checkpoints. No extra
seed, rate sweep, all-parameter panel or Muon arm is added.

## Fixed choices

| Setting | Depth 12 | Depth 20 |
|---|---:|---:|
| Width / heads / context | 512 / 8 / 512 | 512 / 8 / 512 |
| Parameters | 89,603,072 | 114,822,144 |
| Global tokens per update | 1,048,576 | 1,048,576 |
| Total training tokens | 1,539,870,720 | 1,539,870,720 |
| Updates / spectral samples | 1,469 / 60 | 1,469 / 60 |
| Sampled blocks (one-based) | 3, 6, 9, 12 | 5, 10, 15, 20 |

Both use seed 260924, LR 0.0012, betas (0.9,0.95), epsilon 1e-8,
weight decay 0.01, clipping 1, 50-update warmup and final-10%-token cooldown.
The final batch has 561,152 tokens, matching the eight-layer run. These
horizons correspond to 17.19 and 13.41 tokens/parameter. The common rate
comes from `logs/adamw_spectra/g20_20260924_212246_r2/selection.json`; it is
not asserted to be individually optimal at the larger depths.

Use the original `data/fineweb10B` stream and fixed validation bank. The
7.2B-token directory staged before the correction is unnecessary and will
not be used. The original manifest remains unchanged. The shared
initialization rule gives residual std 0.005/0.004082/0.003162 at depths
8/12/20; this study includes that recipe's depth dependence.

Slurm 22.05 lacks --prefer: request `96g` for depth 20 and `48g` for depth
12 (the 96g nodes also advertise 48g). Each job uses one node, eight GPUs,
16 CPU cores and eight hours. DDP and hardware can change reduction
rounding; global mean loss and update equations remain fixed. Independent
review supported this corrected design and requested the numerical/reference/
resume checks and explicit horizon/initialization disclosures above.

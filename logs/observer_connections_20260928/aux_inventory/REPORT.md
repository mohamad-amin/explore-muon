# Saved auxiliary displacements have different dimensions and radial behavior

2026-09-28. Completed in 8.38 seconds on CPU, with at most two numerical
threads, mmap, and no model construction, forward/backward pass, optimizer,
GPU, or training. This inventories actual checkpoint differences, not
optimizer directions or functional contributions. All 13 selected pairs and
468 auxiliary tensor records are complete.

## Exact partition

The original body_aux partition is exhaustive: 48 hidden Q/K/V/O/up/down
matrices have 25,165,824 parameters; 36 auxiliary tensors have 51,783,168.
Every before/after state has exactly these 84 keys. Architecture fields match
across the 13 pairs; PD/SOAP-PD additionally enable nonpersistent input-statistic
tracking. Exact keys, shapes, dtypes and tensor hashes are in `result.json`.

| Auxiliary group | Exact keys (i=0,…,7) | Tensor dimensions | Parameters |
|---|---|---|---:|
| Head | `head.weight` | 50,304×512 | 25,755,648 |
| Token embedding | `embed.weight` | 50,304×512 | 25,755,648 |
| Position embedding | `position.weight` | 512×512 | 262,144 |
| Body gains | `blocks.i.ln1.weight`, `blocks.i.ln2.weight` | 16×[512] | 8,192 |
| Final gain | `norm.weight` | [512] | 512 |
| Q/K gains | `blocks.i.attn.q_norm.weight`, `blocks.i.attn.k_norm.weight` | 16×[64] | 1,024 |

These are untied head and token-embedding matrices. There are no bias tensors.
The Q/K gains act in the 64-dimensional per-head coordinates; each Q or K
normalizer's gain vector is shared across the 8 heads in its block. Frozen
`model.py` and `muon.py` establish the partition and optimizer assignment.

## Original Muon 4M 183→184

Let w be the old saved tensor and d=w_next−w. Group norms concatenate the
listed tensors; radial change means w·d/||w||². Reductions use FP64 on saved
FP32 values. This is the same pair as the completed body_aux loss probe.

| Group | Old norm | Next norm | Actual displacement norm | ||d||/||w|| | Per-coordinate RMS(d) | Radial change |
|---|---:|---:|---:|---:|---:|---:|
| Head | 250.4694 | 250.9142 | 2.220961 | .008867 | .00043763 | +.00173811 |
| Token embedding | 187.8058 | 188.0774 | 2.195737 | .011692 | .00043266 | +.00137884 |
| Position embedding | 12.0224 | 12.0293 | .083361 | .006934 | .00016281 | +.00054891 |
| Body gains | 90.03325 | 90.03137 | .018856 | .0002094 | .00020833 | −.00002088 |
| Final gain | 26.33170 | 26.34995 | .019121 | .0007262 | .00084504 | +.00069313 |
| Q/K gains | 35.20097 | 35.22542 | .027402 | .0007784 | .00085630 | +.00069474 |

The head and token embedding have comparable absolute displacement norms.
All norm gains together have displacement norm .038367, but the final and
Q/K gains' per-coordinate movement is roughly twice the head's. Their
displacement/old-parameter cosines are +.9545 and +.8925; body gains have −.0997.
Small absolute norm therefore cannot exclude a normalization group from a
functional counterfactual. These coordinate scales do not identify which
group contributes to the positive body–auxiliary loss interaction.

All groups above use plain auxiliary AdamW at the same scheduled LR .002,
betas (.9, .95), epsilon 1e−8, and decoupled weight decay .01. The actual effective
decay coefficient is .00002 for every auxiliary parameter, including gains
and embeddings; there is no no-decay gain group. The body uses LR .014 and
Muon momentum .9 here. All selected steps are outside warmup/cooldown.

For context, the ideal decay norms for head/token embedding/position/body
gains/final gain/QK gains are .005009/.003756/.000240/.001801/.000527/.000704.
Ideal decay is 9.55% of the body-gain displacement norm, versus .23% of the
head norm. This is a norm ratio, not an additive signed contribution: decay
can oppose the adaptive write. `adaptive_plus_rounding_norm` records
||d+lr·wd·w||. It includes parameter-write rounding and is **not** the stored
or reconstructed Adam direction. Optimizer moments were not read.

## Original 1M own-trajectory checks

The saved M/PD/S/S∘PD pairs 200→201, 500→501, 900→901 all share the same
architecture, data-exposure convention, auxiliary LR .002 and decay .01.
The body differs: M/S use LR .007 on L40S, PD/S∘PD use LR .01 on Ada; all use
body momentum .95. Thus these are state-matched checkpoints on different
co-adapted trajectories, not identical-state causal comparisons.

The table lists actual displacement norms for head / token+position
embeddings / all normalization gains. Full group/tensor metrics are in the
CSV and JSON artifacts.

| State | Muon | PD | S | S∘PD |
|---|---|---|---|---|
| 200 | 2.2291 / 2.2398 / .05184 | 2.2426 / 2.2314 / .04592 | 2.2626 / 2.2830 / .05409 | 2.2790 / 2.2591 / .04754 |
| 500 | 2.1464 / 2.1582 / .02801 | 2.1438 / 2.1648 / .02579 | 2.1425 / 2.1607 / .02645 | 2.2294 / 2.2493 / .02629 |
| 900 | 2.1533 / 2.1676 / .02669 | 2.1537 / 2.1738 / .02522 | 2.1519 / 2.1714 / .02688 | 2.0790 / 2.1049 / .02420 |

At a fixed stage, head/embedding displacement norms span at most 4.3% across
these methods; the normalization total spans 8.6–17.8%. This establishes
roughly comparable auxiliary Euclidean scale under the common auxiliary
recipe, not comparable functional behavior or a rate explanation.

## Current head evidence and limits

The current main protocol's 13:50 review pauses the head-whitening line after
the norm-matching and mean-suppression confounds. It calls for missing plain
Adam head-only LR controls ×⅓ and ×3 before interpreting that line. The
uncentered α¼ endpoint gain and early deficit are historical evidence, not
an authorization or conclusion for this inventory.

The stopped unmatched α½ arm still has a stale `scientific/status.json`
reporting running at step 0. Its launcher log records termination, its README
records the pause, and live Slurm accounting confirms 2567578.542 CANCELLED,
ending 2026-09-28 13:49:38 CDT. The α¼ arm never started according to that
README. No head-LR-named record was found in the bounded filename search.
No main process, queue, checkpoint, code, protocol, or state file was changed.

## Reproduction and retained failure

Run `PYTHONDONTWRITEBYTECODE=1 .venv/bin/python logs/observer_connections_20260928/aux_inventory/inventory.py`
from the project root. The script itself caps numerical threads and disables
CUDA visibility; only PyTorch tensor loading/reduction is used.

`inventory.py`, `inventory.log`, `status.json`, `tensors.csv`, `groups.csv`,
and `result.json` preserve the completed computation. Provenance includes
both checkpoint paths, file size/mtime checks, source SHA256 verified against
frozen manifests, full configs and group metadata, and hashes of every
accessed auxiliary before/after tensor. No whole-checkpoint hashing or
optimizer-moment/body-value reads were needed. The maximum tensor norm
polarization error was 4.48e−16 relative to the old/next squared norms.

An initial overly strict model-config equality check stopped when it reached
PD's additional `track_input_stats`/`track_input_cov` flags. The original
script and traceback are retained as `inventory_initial_failed.py` and
`inventory_initial_failed.log`. The correction compares all architecture
fields, permits only those two tracking flags, and verifies every saved
parameter shape and the complete key partition. No measurement definition
or scientific criterion changed.

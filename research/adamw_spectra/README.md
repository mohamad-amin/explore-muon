# AdamW update spectra

Standalone implementation of the user's 77M AdamW study inspired by
[Spectral Scaling Laws of Muon](https://arxiv.org/pdf/2606.04058v2).
Read [PROTOCOL.md](PROTOCOL.md) for the frozen question, reconstruction choices,
rate-selection rule and interpretation limits. This does not modify Track 3.

The measured matrix is the **actual adaptive AdamW direction**
`m_hat / (sqrt(v_hat) + eps)` reconstructed after each optimizer step.
The spectral panel excludes LR and decoupled weight decay; their step norms
are recorded separately. It covers Q/K/V/O and MLP up/down in blocks 2/4/6/8.
Full direct-SVD singular values are retained every 25 updates, plus the first
and final updates. No NS, custom
preconditioning, rank truncation, or spectral feedback changes training.

## Eight-layer comparison

Both runs are complete. [Comparison report](../../logs/muon_spectra/comparison_depth8_20260925/README.md)
contains validation curves, all 24 spectral trajectories, final full spectra,
step magnitudes, CSV exports and a combined PDF. Generate a read-only report
using `python -m research.adamw_spectra.compare --adamw <run> --muon <run> --out <new-directory>`.
The comparison verifies shared initialization/data/token controls and full
spectral archive coverage before plotting. Muon momentum and its actual
post-NS update remain separate observables.

## Muon extension (2026-09-25)

The authorized Muon depth cohort is documented in [MUON_CASE.md](MUON_CASE.md)
and [MUON_REVIEW.md](MUON_REVIEW.md). Configs `muon_depth{8,12,20}_width512.json`
use the same architecture/data/horizon and Muon body matrices plus auxiliary
AdamW. Use `distributed.py` for Muon, even for single-rank checks; it records
both `spectra/` (captured post-NS update) and `spectra_momentum/` (momentum).
Analyze with `analyze.py` and `--quantity momentum` for the second channel.
`run_muon.py` is the bounded frozen-job controller. The historical AdamW
instructions below describe the unchanged original experiment.

## Setup

Use the repository `.venv` (PyTorch, NumPy, Matplotlib). There are no new
runtime dependencies. This runner uses one selected GPU with gradient
accumulation; CPU mode is for qualification. Launch commands below are run
from the repository root. Check live GPU occupancy before choosing a device.

The default model is a conventional pre-LayerNorm GPT with 8 blocks, width
512, 8 attention heads, context 512 and untied 50,304-row vocabulary matrices.
It uses GELU, learned positions and ordinary uncapped cross-entropy. The
paper does not identify an exact model revision: these architecture and
initialization choices are documented reconstruction assumptions. This is
paper-comparable descriptive work, not an exact reproduction or a controlled
Muon-versus-AdamW result.

FineWeb GPT-2-tokenized shards already live in `data/fineweb10B`. The runner
validates file headers/lengths, rejects overlapping train/validation files,
and refuses data recycling. Missing data can be obtained using the existing
`data/cached_fineweb10B.py` script. Actual parameter count, token budget,
data-file manifests, full config, environment, source hashes and copies of
source files are saved to each output directory.

## Hyperparameters and rate selection

Standard PyTorch AdamW throughout: betas `(0.9, 0.95)`, epsilon `1e-8`, decay
`0.01`; FP32 parameters/moments, BF16 autocast on GPU, gradient clipping at
1.0. Warm up for 50 updates, hold constant, then linearly cool down across
the final 10% of tokens. Batch size is 1,048,576 tokens. The training budget
is 20 times the actual parameter count, rounded to a sequence; the final
batch is shortened to meet that budget. Full parameter count and planned
update count are saved in `metadata.json`.

`6e-4` is a provisional, conventional GPT LR; it is **not empirically tuned
yet**. The small sweep compares `3e-4`, `6e-4`, `1.2e-3` using identical
initialization/data and 300-step prefixes of the full schedule. It picks the
lowest mean held-out NLL at steps 200/250/300, with lower LR breaking exact
ties, and reports boundary winners. Spectra are disabled for pilots. The
selected config restores full measurement and uses a separate scientific
seed. Pilot seed is `main_seed - 1`. No automatic grid expansion or full run
follows the sweep.

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m research.adamw_spectra.sweep \
  --config research/adamw_spectra/configs/77m.json \
  --out logs/adamw_spectra/lr_pilots

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m research.adamw_spectra.train \
  --config logs/adamw_spectra/lr_pilots/selected_config.json \
  --out logs/adamw_spectra/77m_seed260924

.venv/bin/python -m research.adamw_spectra.analyze \
  logs/adamw_spectra/77m_seed260924
```

For a second seed, use the same selected config with `--seed 260925` and a
new output directory. A second seed tests reproducibility without retuning.
The proposed first-stage budget is 900 pilot updates plus one full run;
pilot work must be counted separately. No scientific training is launched
by importing this package, running tests, or rendering results.

## Instrument qualification and continuation

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python \
  -m unittest research.adamw_spectra.test_study -v
```

Tests check actual AdamW writes across multiple steps, non-mutation, known
spectra/quantile indices, zero matrices, data boundaries, schedule/budget,
77M parameter count, and bitwise CPU resume and measurement non-interference.

Use `--stop-after 5` with a fresh output directory for a bounded full-model
device qualification; this retains the intended full-run schedule. Resume
that same run with the same command/config and `--resume`, removing the
stop limit when ready. Checkpoints are atomic and include model, optimizer,
token position and RNG. Resume rejects changed config, source, data manifest,
device or precision. Records newer than the durable checkpoint are preserved
in an `uncommitted_*` directory before replay. Existing runs are never
silently overwritten. Checkpoints are trusted local pickle files; do not
load checkpoints from untrusted sources.

The default microbatch is 16 sequences; change `microbatch_sequences` in a
copied config before a run if needed. The global batch stays fixed. Exact
SVD runs on CPU with four PyTorch threads by default; `svd_device: "cuda"`
uses direct CUDA SVD. Record timing to assess measurement overhead before a
full run. `spectra_every: 25` is the user-selected default, giving 60 samples
over the planned 1,469 updates, including first and final. This is sparser
than the paper's every-update sampling; use `1` if that cadence is needed.
Loss is still logged every update.

## Efficient execution

The default CUDA path now uses `torch.compile` for both training and
validation, plus PyTorch's fused AdamW (`optimizer_impl: "auto"`). CPU
qualification stays eager and uses foreach AdamW. These are implementations
of the same AdamW equations, not a different optimizer. The resolved choices
are recorded in metadata and checked on resume. Set `compile: false` and
`optimizer_impl: "single"` in a copied config for the original execution path.
Compiled execution selects the native C/C++ compilers recorded by the running
Python interpreter and logs their paths; an inherited Conda cross-compiler
was incompatible with this `.venv`'s system Python headers.

Each accumulated batch is read once and sent to the GPU in one pinned-memory
transfer; microbatches are device views. At the default batch/microbatch sizes
this reduces host-to-device transfers from 128 to one per training update.
The fixed validation bank is cached on device (8 MiB) and reused. Source-token
order, context boundaries and partial-final-batch weighting are preserved.
Optimizer counter reads and pre-step norm reads are batched, and measurement
arithmetic reuses private temporary buffers without mutating optimizer state.
The spectra still use exact direct SVD at the requested sampling times.

Compilation has an initial cost and the shortened final microbatch can create
another graph. `status.json` reports setup time and total invocation time;
the first training step also includes any lazy backward compilation. Compare
steady-state timings separately from startup. The CUDA smoke test is tiny;
an uncontended full-size throughput benchmark is still required to quantify
the training speedup. See [EFFICIENCY.md](EFFICIENCY.md) for numerical checks
and the measured scope of the performance work.

## Outputs and interpretation

- `steps/stepNNNNNN.json`: NLL, LR, token count, clipping, norms, spectral
  diagnostics and separate training/measurement/validation timings.
- `spectra/stepNNNNNN.npz`: all normalized singular values of every measured
  matrix, in descending order. Quantile q is index `ceil(q*r)-1`.
- `checkpoint.pt`, `checkpoint.json`, `metadata.json`, `status.json`, `source/`: continuation
  and provenance. `stopped_at_requested_step` is not a complete scientific run.
- `plots/`: PNG/PDF quantile trajectories against steps and tokens, median
  panels, full/bulk histograms, and training health. Bulk plots remove the
  leading value without renormalizing; histogram bins are common across panels.
- `summary.json`: descriptive windows 1100–1300 and 1300–1500, exact coverage,
  cooldown overlap, clipping and timing. Missing windows remain empty.
  Run status and the durable-checkpoint sidecar are shown separately: records
  after that checkpoint can be valid observations without being resumable.

Use `analyze --snapshot-step 1450 RUN_DIRECTORY` to choose a recorded
snapshot (the default is the latest). A normalized-spectrum plateau alone
does not prove improved learning, an optimizer-specific mechanism or a
scaling law. These AdamW update spectra differ in measurement stage from
the paper's **pre-NS Muon momentum** spectra. A causal comparison would need
a matched Muon run and post-NS measurements; that is outside this implementation.

# Independent six-endpoint pairing and source review

2026-09-28. Read configurations, scientific metadata/status, checkpoint
sidecars, recorded final validation rows, frozen/scientific source files,
and token-shard headers. No model scoring, checkpoint-tensor load, optimizer,
GPU or scheduler action. This review is the only new file.

**Verdict:** the six endpoints are suitable for one common frozen CPU
evaluation helper. The initial proposed score offsets were out of range;
the parent corrected and fixed them before any scoring, preserving the
original proposal. With that correction, no pairing/source blocker remains.

## Completed endpoints

All six `scientific/status.json` records say complete. The final step rows
and `checkpoint.json` sidecars agree on steps/tokens, and each binary final
checkpoint exists. The scoring script should additionally assert those
step/token values when it loads the binary checkpoint.

| Horizon | Schedule | Final step | Tokens | Original validation NLL |
|---|---|---:|---:|---:|
| 1x | beta .8 | 92 | 1,539,870,720 | 4.472392700613 |
| 1x | beta .9 | 92 | 1,539,870,720 | 4.574189670384 |
| 1x | .8→.9 warmup | 92 | 1,539,870,720 | 4.515205945820 |
| 2x | beta .8 | 184 | 3,079,741,440 | 3.974049357697 |
| 2x | beta .9 | 184 | 3,079,741,440 | 3.984007218853 |
| 2x | .8→.9 warmup | 184 | 3,079,741,440 | 3.962814012542 |

The primary original warmup-minus-.8 gaps are therefore **+.042813245207**
and **-.011235345155**. Beta .8 is the better constant at both horizons,
so fixing it as the primary comparator does not depend on the new panel.
The secondary warmup-minus-.9 gaps are -.058983724564 and -.021193206310.
Use these full-precision records rather than differences of rounded numbers.

Paths are the six declared in `next_direction_peer/HISTORY_REVIEW.md`:
the short constants come from `soaudit_prefilter16m_20260928` and
`soaudit_batch16m_20260927`; the long constants from
`soaudit_horizon16m_20260928` and `soaudit_b16mlong_20260928`; both warmups
from `soaudit_momwarm16m_20260928`.

## Common scientific/model metadata

All six share initialization hash
`f8efae98b3d9e0dd5e9cd2318ea2b56a8b01358742af2d91e8c86ea9caf41d19`.
I checked exact agreement of training/validation manifests, architecture,
parameter count, world size, GPU type, precision, compile choice, effective
optimizer implementation, NS coefficients and parameter/optimizer assignment.
All use four ranks and NVIDIA RTX 6000 Ada GPUs.

The common active recipe includes alpha .5, body LR .028, auxiliary AdamW
LR .002, SOAP options, clipping, root/statistic clocks, three-step LR warmup,
and final-10%-token cooldown. Config differences are the intended momentum
choice/schedule, total horizon, kept-checkpoint lists and newer disabled
defaults. No new head whitening or gradient prefilter is active.

Within each horizon this is a schedule comparison. Across horizons the
total exposure and cooldown timing differ; the short and long runs are
not claimed to be bytewise continuations of one another. Those differences
do not prevent evaluating their final models with one inference helper.

## Source qualification and common CPU forward

Verified **300 source-file comparisons**: both the per-run scientific copy
and cohort frozen copy against the execution metadata hashes, for every
recorded available source in all six endpoints. There were zero mismatches.
These are integrity checks, not 300 independent scientific replications.

There are two model-source revisions. Their only relevant difference adds
`track_head_cov`, default false, and selects a StatLinear head only when
that option is enabled. It is not enabled in this family. The newer
`model.py` is byte-identical to the qualified observer helper. The older
version has the same active Linear-head forward, normalization, positions,
attention and MLP computation. `data.py` is identical across all six.

The later optimizer/runner revisions add:

- a disabled two-tap prefilter;
- disabled head-whitening options;
- an output-factor multiplication in SOAP statistics, inactive with zero
  output factor;
- momentum scheduling, active only for the intended warmup treatment.

For constant runs the scheduling function returns the configured constant
beta. No unintended active algorithm change was found in the inspected
differences. This establishes compatible active recipes and forward code,
not bit-identical compiled training executions across revisions.

The qualified CPU builder can safely turn off nonpersistent input-statistic
tracking and strictly load every endpoint into the same FP32 eval model.
Optimizer states, cached roots and SOAP statistics are irrelevant to this
forward-only readout. I directly compared all four copied files in
`confidence_calibration/source/adamw_spectra/` with
`dose_function/source/adamw_spectra/`: each is byte-identical. Those qualified
helpers were also independently compared with the body-auxiliary source.
The scoring script should retain its direct copy hash checks and actual
checkpoint tensor hashes.

## Data-range correction before scoring

Independent parsing of the current 32 FineWeb training shard headers found
exactly **3,200,000,000 tokens**, matching the recorded manifests. The
initial offsets 3,230,131,072 and 3,245,131,072 were outside that stream.
This was detected before any input or model score was produced.

The parent fixed the new banks to:

- [3,140,131,072, 3,140,139,265), including the extra token needed for
  8,192 next-token targets;
- [3,150,131,072, 3,150,139,265), with the same coverage.

Both lie inside the available stream, beyond the largest selected model's
3,079,741,440-token training prefix, and apart from the earlier observer
score banks. The original proposal is preserved as
`PROTOCOL_before_offset_check.md`; the corrected protocol records the reason.
No data staging, wrapping, new training or changed model/readout criterion
was needed. Runtime TokenStream range/manifest checks remain appropriate.

## Interpretation boundary

Pairing supports the fixed cross-fitted scalar-temperature diagnostic.
Every model must receive the same grid, opposite-bank selection/scoring,
tie rule and retained scale-one baseline. The primary comparison remains
warmup versus beta .8 within each horizon. Grid-boundary or raw-order
failure stays unresolved under the declared rule, without expansion.

Even a material calibration change would concern these saved endpoint
predictions. It would not rewrite the original training NLL, establish a
causal step clock, or authorize the paused head-LR/whitening line.

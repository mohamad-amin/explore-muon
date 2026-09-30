# Does global prediction confidence account for schedule NLL differences?

2026-09-28. Fixed before new model scoring. The independent high-level
proposal is `../next_direction_peer/HISTORY_REVIEW.md`; this protocol accepts
its six-endpoint discriminator. The angular comparison and failed early
mixed-response branch remain closed. No training or head-LR repair is proposed.

## Decision argument

The original SOAP-PD momentum warmup beats constant beta .8 at the 2×
horizon by .01124 NLL, but loses by .04281 at 1×. Existing held-out dose
outputs also have large, phase-dependent values of NLL−predictive entropy,
the exact derivative of CE(a*z,y) with respect to a at a=1. Therefore
models with similar loss can still differ substantially in global confidence.
That derivative does not give the available finite calibration gain.

Possible explanation: some of the schedule gap is a difference in global
confidence, which can be corrected after training without changing token
rankings. Competitor: their predictive distribution shapes/progress differ,
and giving each model the same confidence correction leaves the gap intact.
The comparison below measures that finite correction independently, without
confusing in-sample fitting with out-of-sample improvement.

This is a diagnostic of saved endpoints, not a reliability-calibration
benchmark, a training-rate estimate or proof of a causal phase clock. Even
a large correction would not show how calibrated training would evolve.
Temperature scaling is established prior art, not a proposed novel method.

## Fixed model family, data and scoring

Only final SOAP-PD alpha .5, LR .028, seed 260925 on Ada: beta .8, beta .9,
and beta .8→.9 warmup over the first half, at 1× and 2× horizons. Use the
six exact paths in `HISTORY_REVIEW.md`; recheck completion, final steps
92/184, token counts, initial hash, data and active settings. Compare
schedules within horizons. Keep any inactive-default source differences.

Use two fresh 16-sequence banks, context 512, at FineWeb training-stream
offsets 3,140,131,072 and 3,150,131,072. Each reads 8193 tokens (8192 targets),
strictly beyond the largest selected training exposure of 3,079,741,440.
These intervals are disjoint from previously used observer score positions.
Use identical inputs for all six models. They are contiguous sequence banks,
not independent training seeds or guaranteed independent documents.

The fixed inverse-temperature/logit-scale family is a={.8,.9,1,1.1,1.2}.
Every scale receives a per-token CE score from the same unmodified logits.
Record all scores, base entropy, scale derivative and scale curvature.
Do not save full-vocabulary logits or modify checkpoints.

For each model choose the scale with minimum mean NLL on bank A, then score
that frozen choice on bank B; also fit B and score A. Exact ties choose
nearest to one, then the lower scale. The pooled cross-fitted result weights
both scoring banks equally. Models can choose different scales, but all
receive the same fixed family. Selection and reported scoring use opposite
banks. The same-scale-one scores remain the raw comparison.

## Comparisons and declared readings

Primary contrast at each horizon is warmup minus constant beta .8, the
better constant in the original recorded validation. Warmup minus beta .9
is secondary and cannot replace a failed primary. For each held-out fold:

    raw_gap = L_warm(1) − L_constant(1)
    calibrated_gap = L_warm(selected_elsewhere) − L_constant(selected_elsewhere)
    gap_change = calibrated_gap − raw_gap.

Retain both folds, pooled values, raw/calibrated model scores, all grid
scores and selected scales. Required qualification for a decisive primary
interpretation, separately at each horizon:

- Original raw ordering reproduces in both fresh scoring banks: warmup
  worse at 1×, better at 2×.
- Neither compared model's fitted scale is a grid boundary in either fold.
- The relevant differential correction has consistent sign in both folds.

**Material confidence component:** the absolute primary gap shrinks in
both folds, with pooled gap_change ≤ −.005 at 1× or ≥ +.005 at 2×. If the
gap crosses zero, retain the reversal; require its absolute magnitude still
shrinks to call this attenuation. A reversal with larger magnitude is a
different outcome. The .005 threshold is a prospective materiality flag,
not a significance bound. Also require the originally losing model's own
held-out calibrated NLL to improve in both folds. Gap attenuation caused
only by harming the original winner is procedure-sensitive, not evidence
that the loser's confidence was responsible. Report each model's correction
regardless. This clarification was adopted on design review before scoring.

**Ranking survives this modest calibration family:** original ordering
persists in both calibrated folds and abs(pooled gap_change)<.005, with
interior selections. This only weakens a material differential effect in
this grid/panel; it is not equivalence or exclusion of finer/global scales.
If all models select one, the coarse grid has resolved no finite correction.

Other outcomes, raw-order failure, disagreement between folds or boundary
selection are unresolved at the declared scope. Do not enlarge the data,
temperature grid, checkpoint family or comparator set. No percentages of
explained gap near zero, and no post-hoc clock/rare-token/group search.
After this panel, close the fixed test and decide a distinct next question
from the complete evidence, not from whichever endpoint is favorable.

## Qualification and resource cost

Use the already qualified CPU FP32 eager model/data helpers copied byte
identically from `../dose_function/source/adamw_spectra/`. Two numerical
threads; no CUDA, gradients or optimizer. Verify strict parameter loading,
84 model tensors, parameter count and source/config compatibility. Record
full consumed model-tensor hashes and token/source/JSON hashes.

For every sequence, FP64 direct CE at a=1 must agree with standard FP32 CE
within 5e−5 per token. Compute the scale derivative directly as E_p[z]−z_y,
and check it equals NLL−H(p) within 1e−10 absolute. The nonnegative scale
curvature is Var_p(z), computed as a centered second moment. A synthetic
fixed logit example checks the derivative by central finite difference,
shift invariance and temperature indexing before scientific scoring.
Repeat the first sequence once to verify deterministic scale-one scores.
This is one qualification forward beyond the 192 scientific forwards.

Forecast from the first complete sequence including all five vocabulary
reductions, loading/hash cost and the extra qualification forward; add 25%
margin and 20 seconds. Stop if projected total exceeds 600 seconds and
check the wall boundary after every sequence. Preserve original failures
and partial arrays. All new files stay here; main training and sealed
surrogate panels remain untouched.

## Data-availability correction before scoring

Independent header checks found only 3,200,000,000 training-stream tokens,
so the initial proposed offsets 3,230,131,072 and 3,245,131,072 were outside
the available stream. No inputs or model outcomes were scored. The corrected
fixed offsets above are available, beyond all six training exposures and
disjoint from the observer's previous score intervals. The original protocol
is retained as `PROTOCOL_before_offset_check.md`. No data was staged, no
stream wrapping was allowed, and no model/readout criterion changed.

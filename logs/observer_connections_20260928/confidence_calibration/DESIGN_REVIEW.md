# Independent design review: finite global-confidence correction at six endpoints

2026-09-28. Read the independent history proposal, current observer/main
summaries, the fixed protocol, original training manifest/data access rules,
and prior observer scoring offsets. No model call, temperature outcome,
checkpoint-tensor result or new scientific score was read. Only this review
was written.

**Verdict: go with the already accepted pre-scoring availability and
interpretation corrections.** This is a distinct, answerable endpoint
question. It is not a demonstration of the momentum mechanism or a reason
to reopen the paused head-whitening line.

## Identification and fixed readout

The six models are the three declared schedules at each of two horizons.
Keeping warmup minus beta .8 primary at both horizons is appropriate because
that comparator was chosen from the original validation, before the fresh
panel. Beta .9 remains secondary. Do not select a new best constant or
combine the two horizons as though extra training tokens were calibration.

Giving each fixed model the same five positive logit scales and selecting
on the opposite bank separates in-sample fitting from out-of-sample score.
An exact tie-break by distance to one, then lower scale, is deterministic.
Use per-model selections and retain all five per-sequence losses, including
scale one. Positive scaling preserves token rankings algebraically; the
question concerns probability magnitudes, not an argmax correction.

The two banks have equal token counts, so equal fold weighting is also
pooled token weighting. Each comparison must reuse the same scored sequences
and targets. Fold outcomes are not independent model or training replicates;
contiguous sequences need not be independent documents. Sign agreement is a
robustness condition for this panel, not a significance test.

The .005 threshold is explicitly **pooled differential correction**, with
the intended correction direction and attenuation in both folds. Required
qualification is separately assessed for each horizon and compared pair:

1. Both raw bank gaps reproduce the original ordering.
2. Neither model selected a boundary scale in either fitting bank.
3. Held-out correction direction agrees across the two folds.
4. For a material confidence interpretation, absolute gap attenuation occurs
   in both folds and the original **losing** model's held-out NLL improves
   in both folds. The root accepted this fourth clarification before scoring.

The last condition matters: damaging the original winner can shrink a gap
without showing that a correctable confidence defect hurt the loser. Always
report each model's held-out correction separately. If this condition fails,
report the procedure-sensitive outcome rather than a confidence explanation.

A gap reversal is retained, not silently clipped to zero. Under the declared
attenuation reading its absolute magnitude must still shrink. The threshold
is scientifically material relative to the original margins, but it is not
a confidence interval, and crossing it is not statistical confirmation.

## What this can and cannot establish

If independently fitted scaling preferentially improves the loser and
attenuates the gap consistently, the selected endpoint comparison contains
a finite global-confidence component. That is stronger than correlating
head norms or entropy with loss. It still does not establish why training
created that component, what fraction of the whole learning-rate advantage
it mediates, or whether changing calibration during training would help.

If order and gap survive this finite opportunity, report that restricted
result. Interior grid selections do not prove a global continuous optimum,
and every scale selecting one means this coarse grid resolved no useful
finite correction. It does **not** show confidence is irrelevant. Derivative
and curvature diagnostics may reveal an unresolved smaller-scale opportunity;
they must not be converted into an added fitted scale or an unmeasured loss
improvement after seeing the panel.

Boundary selection, unreproduced raw ordering or disagreeing folds is
unresolved and closes the fixed panel. No extra bank, endpoint, temperature,
window, token-frequency split or favorable comparator follows automatically.

## Correct derivative, curvature and numerical checks

For z fixed, p_a = softmax(a z), and per-token

    L(a) = logsumexp(a z) − a z_y,
    L'(a) = E_{p_a}[z] − z_y,
    L''(a) = Var_{p_a}(z) >= 0.

At a=1, entropy H(p)=logsumexp(z)−E_p[z], so **L'(1)=L(1)−H(p)**.
At arbitrary a the identity is L'(a)=[L(a)−H(p_a)]/a. Curvature Var_p(z)
is with respect to a. Curvature with respect to log(a) at a=1 instead equals
L'(1)+Var_p(z); those two parameterizations must not be mixed.

FP64 direct logsumexp/target subtraction should agree with the independently
reduced scale-one FP32 CE within the specified per-token tolerance. The
1e−10 absolute identity gate is appropriate for the FP64 scalar identity.
Use centered second moments for curvature, not cancellation-prone E[z²]−E[z]².
Compute entropy from finite log-probabilities rather than log of probabilities
that may underflow. A common per-token logit shift must not change CE,
derivative or curvature; the synthetic test should exercise this explicitly.

The proposed synthetic finite-difference derivative check, scale indexing
check, deterministic first-sequence repeat and fixed parameter/source/token
integrity checks are sufficient for the declared readout. Scale-one
reproduction here means the same saved logits/input calculation; the fresh
panel is not expected to equal the original large validation bank's score.

## Availability correction and provenance

The originally suggested banks at 3,230,131,072 and 3,245,131,072 were outside
the manifest's 32×100,000,000 = 3,200,000,000 token stream. TokenStream's
non-wrapping exhaustion check would correctly reject them. This was caught
before any model score and is not a scientific failed outcome.

The replacement offsets **3,140,131,072 and 3,150,131,072** each consume 8,193
source tokens for 16×512 targets. They are inside the stream, beyond every
selected checkpoint's 3,079,741,440-token maximum exposure, and disjoint from
the examined prior observer score intervals. The original protocol is retained
as `PROTOCOL_before_offset_check.md`; no data was staged or wrapped. Validate
and hash the actual extracted x/y tensors before scoring all models identically.

The input source shares historical compatibility paths with the relocated
project's original data; reading this documented shared corpus does not
require reading the separate last-layer study or its sealed results.

## Resource and scope boundary

192 scientific sequence forwards plus one deterministic repeat are the fixed
cost. Five full-vocabulary FP64 reductions reuse each forward's logits;
include those reductions and loading/hashing in the first-sequence forecast.
Stream sequence outputs rather than accumulating full vocabulary tensors.
The two-thread, CPU-only, 600-second forecast/wall cap and preservation of
partial/failure artifacts are appropriate. No gradients, optimizer steps,
GPU allocation, checkpoint mutation or main validation rewrite is warranted.

The root has accepted the availability correction, the a-coordinate formulas,
the pooled threshold with fold sign requirements, the losing-model improvement
guard, and the all-one-grid limitation. No further permission or review cycle
is required before the bounded implementation is qualified and run.

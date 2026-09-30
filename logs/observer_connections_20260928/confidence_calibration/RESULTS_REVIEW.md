# Independent confidence-calibration result review

2026-09-28. Read the completed producer record, executed sources, analyzer,
full per-token grid and analysis outputs. Independently recomputed bank
minima, opposite-bank mapping, primary contrasts and integrity checks.
No new model forwards, scales, checkpoints or samples. This is the only
new file.

## Verdict

**All six models select a=1 on both calibration banks.** This is the
protocol's explicitly declared coarse-grid outcome: the tested family
resolves no finite confidence correction. Close the comparison without
a finer grid or larger panel. Do not upgrade it to proof that calibration
cannot affect the gap at finer scales, or to an identified training
mechanism/phase clock.

The fresh scale-one comparisons nevertheless reproduce both original
primary schedule orderings: warmup loses to constant beta .8 at 1x and
wins at 2x. This is consistency on new inputs for these same trained
endpoints, not independent training-seed replication.

## Execution, integrity and mapping

The producer completed 192 scientific forwards plus the one qualification
repeat in 227.91 seconds, against a 405.67-second forecast and 600-second
cap. The repeat logits match exactly. Maximum direct derivative-identity
error is 5.45e-14; maximum FP64/FP32 scale-one CE difference is 3.50e-5,
within the declared 5e-5 gate. Synthetic derivative finite difference and
common-shift checks also passed.

The executed and current probe hashes both equal the reviewed
`b48af6367d42690a89506afcd68914c9b386ae59faac672ce4ea5d68decfaf06`.
The executed protocol matches its recorded hash. All four helper hashes
and both saved token hashes match. Array shape `(6,32,5,512)` and the
model/scale ordering agree with the producer and analyzer.

The analyzer fits on A and scores B, then fits B and scores A. It never
selects a model's scale from the same scoring fold. Equal scoring-bank
sizes justify the pooled mean. Integer-tenths distance implements the
declared nearest-one/lower-scale exact-tie rule without binary-decimal
distance ambiguity.

All twelve bank minima are uniquely at scale index 2, a=1. The smallest
penalty for any nonunit candidate relative to a=1 is **.00931264 NLL**.
Thus the unit selections are genuine fixed-grid results, not approximate
ties or numerical accidents. All grid outcomes remain retained.

## Primary held-out contrasts

All numbers are warmup minus constant beta .8:

| Horizon | Fit A, score B | Fit B, score A | Pooled |
|---|---:|---:|---:|
| 1x | +.03299113 | +.05290068 | +.04294590 |
| 2x | -.01294883 | -.01178284 | -.01236584 |

Because every selected scale is one, calibrated scores are the identical
score arrays as raw scores. Their correction and every paired gap change
are exactly zero under an identical arithmetic reduction path. There is
no material confidence component measured in this grid.

However, the protocol reserves the stronger "ranking survives this modest
family" classification for an informative nontrivial calibration comparison.
The present result is specifically **coarse_grid_resolves_no_finite_correction**.
The fact that the visible ranking remains unchanged is not an equivalence
test or evidence against a finer correction.

## Arithmetic nuisance in the initial analyzer

The initial implementation formed the raw bank mean through a reduction
over all scale columns, then formed the selected-scale mean from a separate
slice. Although the a=1 observations are identical, different summation
orders produced corrections around 1e-15. Testing their signs could attach
a spurious "fold corrections disagree" reason.

The proper fix is to use the same scalar reduction path for the raw and
selected scale-one vectors, not to add a scientific tolerance or change
the selection rule. I independently used a common path and obtained exact
zero corrections. The parent preserved the original code and six outputs
under `initial_reduction/` and regenerated the bookkeeping fields. I then
verified all twelve corrections and all comparison gap changes are exactly
zero, both primary labels are `coarse_grid_resolves_no_finite_correction`,
and the spurious sign-disagreement reason is gone. Secondary labels remain
descriptive only. Selected scales, grid data, figure PNG and scientific
classification are unchanged; preservation checks are in
`run1/reduction_repair.json`.

Verified final analyzer SHA256:
`8043217f83a92a274297636c8de08a627315bedd7f4676c16a896328b9147235`.
Verified final analysis SHA256:
`d8b2f022d84004ee2350205761aebd641f0d6f1cfcbd27b1aab05449485e0270`.

## Derivatives are retained observations, not an unrun finer fit

Mean scale derivatives remain negative: approximately -.194 to -.213 at
1x and -.087 to -.106 at 2x. Mean scale curvatures are approximately 8.17
and 7.34–7.38, respectively. These are valid derivatives at a=1, and the
identity `g=NLL-H(p)` holds. A nonzero derivative can coexist with a unit
minimum on a coarse discrete grid.

Do not turn these values into a Newton-estimated temperature, an estimated
unmeasured loss gain, or a reason to extend the grid. Neither local
derivatives nor all-one selection answer whether the **differential**
schedule gap would change under a finer continuous calibration.

## Decision

Preserve the fresh-input reproduction of the schedule ordering and the
entire fixed grid. The bounded confidence discriminator is complete but
did not resolve a finite correction, exactly as anticipated in its stop
rule. No extra scale, state, sample, head-LR adjustment or calibration-based
training claim follows. Main training results remain at their original
ordinary validation endpoints.

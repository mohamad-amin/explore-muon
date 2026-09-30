# Schedule rankings reproduce; the fixed confidence grid resolves no correction

2026-09-28. All six completed SOAP-PD endpoints select the original logit
scale a=1 on both fixed banks. The short-horizon warmup disadvantage and
long-horizon advantage both reproduce on fresh inputs. The declared coarse
confidence family therefore resolves no finite correction; it does not
establish that finer confidence adjustments are irrelevant. Close this
panel without another scale, input bank, endpoint or fitted temperature.

The 192 scientific forwards plus one deterministic qualification repeat
completed in 227.91 seconds on two CPU threads. No training, gradient,
optimizer or GPU operation. Tool session 55327 exited 0.

## Fixed experiment

Models are SOAP-PD alpha .5, LR .028, seed 260925, trained on Ada, with
constant beta .8, constant beta .9, or beta .8→.9 warmup over the first
half. Compare them within the 92- and 184-update horizons. This is a
one-seed endpoint comparison, not a training-rate or causal-clock test.

Two disjoint fresh banks each contain 16 sequences of 512 targets, beyond
every selected checkpoint's training exposure. Each model receives all
five predeclared positive logit multipliers {.8,.9,1,1.1,1.2}. Select a
model's multiplier on A and score it on B; then reverse the roles. Every
scale-one baseline and every per-token grid score is retained.

The primary comparator is warmup minus beta .8 at both horizons, fixed
from the original validation results. Beta .9 is secondary. Neither fresh
scores nor the calibration procedure select a new comparator.

## Fresh rankings and cross-fitted readout

| Horizon | Schedule | Raw fresh NLL | Selected a, fit A / fit B | Cross-fitted correction |
|---|---|---:|---:|---:|
| 1× | beta .8 | 4.577816 | 1 / 1 | 0 |
| 1× | beta .9 | 4.692352 | 1 / 1 | 0 |
| 1× | warmup | 4.620762 | 1 / 1 | 0 |
| 2× | beta .8 | 4.081724 | 1 / 1 | 0 |
| 2× | beta .9 | 4.086286 | 1 / 1 | 0 |
| 2× | warmup | 4.069358 | 1 / 1 | 0 |

The same inputs are scored raw and after opposite-bank selection. Selecting
a=1 applies no transformation, so the mathematical correction is exactly
zero. The raw and calibrated primary gaps are:

| Horizon | Score bank A | Score bank B | Pooled fresh gap | Original validation gap |
|---|---:|---:|---:|---:|
| 1× | +.052901 | +.032991 | +.042946 | +.042813 |
| 2× | −.011783 | −.012949 | −.012366 | −.011235 |

Both original orderings reproduce in both fresh banks. The shared absolute
NLL shift on this different text panel does not change those paired signs.
The secondary warmup-minus-beta-.9 means are −.071590 and −.016928 at 1×
and 2×, also negative in both banks. Secondary results do not replace the
primary readings.

Every nontrivial candidate is worse than a=1 on each fitting bank. Pooled
changes from the nearest grid points are:

| Model | L(.9)−L(1) | L(1.1)−L(1) |
|---|---:|---:|
| 1× beta .8 | +.066518 | +.015663 |
| 1× beta .9 | +.064988 | +.017155 |
| 1× warmup | +.064531 | +.017605 |
| 2× beta .8 | +.051170 | +.023263 |
| 2× beta .9 | +.051753 | +.022510 |
| 2× warmup | +.049741 | +.024250 |

This is not a boundary or numerical tie. It is the predeclared all-one
outcome: **coarse grid resolves no finite correction**. No .005 material
differential correction was observed. The result is not promoted to the
stronger "confidence does not explain the ranking" claim.

## Derivative evidence explains the resolution limit

The scale derivative at a=1 is exactly g=L−H(p), and its curvature with
respect to a is Var_p(z). Mean retained values are:

| Model | g | Scale curvature |
|---|---:|---:|
| 1× beta .8 | −.213372 | 8.174431 |
| 1× beta .9 | −.199711 | 8.176893 |
| 1× warmup | −.194227 | 8.172395 |
| 2× beta .8 | −.099057 | 7.382875 |
| 2× beta .9 | −.106102 | 7.367460 |
| 2× warmup | −.087069 | 7.338759 |

A negative mean derivative favors an infinitesimal sharpening direction,
yet the tested a=1.1 step increases loss. Thus the identity choice does not
establish continuous stationarity. These are derivatives of each fixed
model's held-out loss, not learned temperatures, verified finite gains or
an estimate of how much of a training gain calibration mediates. No Taylor
optimum, added scale or inferred unmeasured loss decrease is substituted
for the fixed grid after seeing it.

The earlier dose-state confidence differences remain valid at their selected
early/mid states. They did not imply that the completed schedule endpoints
would have a materially different finite calibration opportunity.

## Qualification, provenance and retained corrections

Independent pairing review checked all six completed endpoints, shared
initialization/data/architecture, intended schedule differences and 300
frozen/scientific source comparisons. The copied FP32 eager helpers are
byte-identical to the previously qualified observer helper. Actual loaded
model hashes, all 84 parameter tensors, step/token metadata, score tokens
and source/JSON inputs are recorded in `run1/result.json`.

The initial proposed offsets exceeded the available 3.2B-token stream.
Independent header checks caught this before scoring. The corrected fixed
offsets 3,140,131,072 and 3,150,131,072 are inside the stream, beyond the
3,079,741,440 maximum selected exposure, and separate from previous observer
score intervals. The original protocol and its correction are preserved.
No data staging, wrapping or outcome-dependent sample selection occurred.

The repeated first forward is exact. Maximum FP64 scale-derivative identity
error is 5.45e−14, under 1e−10. Maximum FP64-versus-FP32 scale-one CE error
is 3.50e−5 per token, under the fixed 5e−5 gate. All losses and curvatures
are finite, and centered variances are nonnegative. Forecast 405.67 seconds,
actual 227.91, within the 600-second boundary.

The initial analyzer used different floating-point reduction paths for
identical raw and selected-a=1 arrays, producing spurious corrections of
order 1e−15 and misleading sign flags. Original analyzer and outputs are
preserved in `run1/initial_reduction/`. The corrected analyzer uses identical
reductions; selection, scientific data, thresholds and both all-one verdicts
are unchanged. This is an arithmetic-reporting correction, not a scientific
gate relaxation or a second model evaluation.

## Connection and decision

This fresh panel strengthens the state-local observation that warmup's
ordering depends on the training horizon. It supplies no measured finite
confidence remedy and leaves smaller confidence differences unresolved.
The previously measured Q/K radius/turn differences remain a distinct
geometric finding, not an explanation established by this calibration test.

Close the six-endpoint/five-scale panel. Do not expand it to obtain a positive
calibration result, replace original validation scores, or restart the paused
head-whitening line. No robust optimizer/architecture improvement or novelty
is claimed. The observer goal remains broader than either local diagnostic.

Evidence: [protocol](PROTOCOL.md), [pairing review](PAIRING_REVIEW.md),
[design review](DESIGN_REVIEW.md), [implementation review](IMPLEMENTATION_REVIEW.md),
[results review](RESULTS_REVIEW.md), [gap figure](run1/confidence_gaps.png),
`run1/per_token.npz`, `run1/analysis.json`, `run1/model_scores.csv`,
`run1/contrasts.csv`, `run1/grid_curves.csv` and `run1/result.json`.

# Larger Q/K turns reach predictions, but their benefit depends on the rest of the update

2026-09-28. The fixed functional test is complete. Both predictions about
movement pass: shorter momentum enlarges Q/K predictive changes, and the
higher LR produces less amplification than nominal squared-step scaling.
This holds with Q/K applied alone and after the actual complement update.
The predeclared **Q/K-last helpfulness** condition fails. Q/K helps alone,
but overlaps with the useful changes made by the rest of the model.

Keep that positive functional connection and its limitation together.
The result does not show Q/K is generally harmful, does not identify the
mediator of the training gain, and does not justify a radius/step intervention.
Close the fixed four-state test without more states, heads or fitted scales.

## Fixed comparison

Four SOAP-PD own states, 46→47, cross LR {.028,.04} with beta {.9,.8},
alpha .5, 16M batch, seed 260925, Ada. Score the same two fresh eight-sequence
banks at base, Q/K weights only, all other parameters only, and the full
actual next model. Q/K is exactly 16 weight tensors; the 68-tensor complement
includes Q/K/RMS gains, other body weights, embeddings and head.

These are saved writes, including decay and rounding. No gradients,
optimizer steps, gauge canonicalization, radius matching or rescaling are
used. The full corner includes all 84 actual next parameter tensors.
The 256 scientific forwards plus one restoration check completed in
287.14 seconds on two CPU threads; session 92765 exited 0. No GPU work.

## The angular factorial pattern reaches predictive distributions

Primary movement is exact forward KL, averaged over the fixed targets.
Q-alone uses KL(p_base || p_Q); Q-last uses KL(p_rest || p_full).
These have different reference distributions and are not interchangeable
or additive shares of total movement.

| Contrast | Q-alone, pooled | Q-alone, bank 0 / 1 | Q-last, pooled | Q-last, bank 0 / 1 |
|---|---:|---:|---:|---:|
| beta .8/.9 at LR .028 | 1.725 | 1.773 / 1.692 | 1.453 | 1.425 / 1.474 |
| beta .8/.9 at LR .04 | 1.755 | 1.246 / 2.222 | 1.679 | 1.208 / 2.084 |
| LR .04/.028 at beta .9 | 1.397 | 1.612 / 1.244 | 1.252 | 1.343 / 1.183 |
| LR .04/.028 at beta .8 | 1.421 | 1.133 / 1.634 | 1.447 | 1.139 / 1.672 |

Every beta ratio exceeds one. Every LR ratio is below the fixed nominal
squared-step ratio 2.040816. Therefore P1 and P2 each pass all eight required
bank-level cells. The high-LR beta contrast differs considerably between
banks; the pooled value is not universal across text.

Absolute movements are retained so these ratios do not conceal scale:

| LR / beta | Q-alone KL | Rest-alone KL | Full KL | Q-last KL |
|---|---:|---:|---:|---:|
| .028 / .9 | .000769 | .071920 | .081294 | .000733 |
| .028 / .8 | .001328 | .097584 | .111879 | .001065 |
| .04 / .9 | .001075 | .074271 | .084296 | .000918 |
| .04 / .8 | .001886 | .140026 | .160734 | .001541 |

All required mean KLs are well resolved; the smallest primary denominator
is .000632, far above the 1e−10 numerical tolerance. There are no unresolved
ratios. Descriptive finite-response RMS ratios also rise with shorter beta:
pooled 1.316/1.326 at the two LRs; higher-LR RMS ratios are 1.187/1.195.
They are not substituted for the predeclared KL criteria.

This is a real prediction-level counterpart of the weight-space observation,
but it remains an own-state comparison. Inputs, gains, trained representations
and reference probability distributions co-evolve with beta and LR. No
common-state radius treatment or causal mediation has been isolated.

## Helpfulness is strongly order dependent

Negative entries below mean lower NLL. Symmetric credit averages the two
possible group orders and is positive for improvement; it is an explicit
descriptive convention, not unique causal attribution.

| LR / beta | Q-alone change | Q-last change | Full change | Symmetric Q credit |
|---|---:|---:|---:|---:|
| .028 / .9 | −.011529 | −.002161 | −.050069 | +.006845 |
| .028 / .8 | −.014710 | +.001102 | −.035174 | +.006804 |
| .04 / .9 | −.010986 | −.000334 | −.047339 | +.005660 |
| .04 / .8 | −.014124 | +.007604 | −.018490 | +.003260 |

Q/K helps alone in every state and both banks. At beta .8, however, its
last-added effect is +.002530/−.000327 at LR .028 and +.005111/+.010097
at LR .04. Thus P3 fails both its sign-consistency and pooled −.005
conditions. At high LR the last-added Q/K effect is adverse in both banks;
at low LR it is inconsistent and adverse on average.

The symmetric Q credit remains positive in every state and bank. The
alternative readout suggested by one independent review would therefore
give a different usefulness description. The root explicitly chose the
actual-background margin before scoring and retains the symmetric credits
without replacing the failed criterion. These observations do not warrant
calling Q/K useless or deleting its updates.

Likewise, the better-trained beta-.8 states have smaller immediate full-step
loss decreases on this panel than beta .9. That repeats the established
warning that a selected state's one-step change does not rank accumulated
training progress. No trajectory-rate conclusion follows from this table.

## Ordinary prediction overlap explains the order dependence

The positive loss interaction is I=LF−LQ−LR+L0. Using additive logits
zadd=zQ+zR−z0 gives an exact split into ordinary additive prediction overlap
and the finite mixed-output effect:

| LR / beta | Total interaction | Additive overlap | Finite mixed loss |
|---|---:|---:|---:|
| .028 / .9 | +.009369 | +.009551 | −.000182 |
| .028 / .8 | +.015812 | +.015753 | +.000059 |
| .04 / .9 | +.010652 | +.010423 | +.000229 |
| .04 / .8 | +.021728 | +.021636 | +.000092 |

The mixed effect is small beside the additive interaction at each pooled
state. Base-probability-centered finite logit responses have positive Q/rest
covariance correlations .650, .707, .605 and .696 in the same order. Q's own
response variance is .00153–.00380, while twice its cross-covariance with the
rest is .01954–.04496. A small isolated response can therefore have a material
interaction with a larger response pointing toward overlapping corrections.

These are finite-response covariances, not a tangent GN matrix or a fraction
of the eventual training gain. The exact CE split supplies the finite-loss
interpretation. It is consistent with the earlier body/auxiliary overlap
findings, now for a different parameter partition and optimizer family.
It does not identify a disposable common direction or reopen the failed
helpful-nonlinearity branch.

## Qualification and preserved decisions

Configuration/data/init pairing matches the earlier factorial; all 100
scientific-source hashes passed. Actual 46/47 checkpoint indices, token
counts, strict masks and restored-base output passed. All 48 consumed body
tensors in each before/after pair match the angular analysis's recorded
hashes; full-model and token hashes are retained. Sources and original
checkpoints were untouched.

The maximum FP64/FP32 corner CE difference is 4.80e−5 per token, within the
unchanged 5e−5 gate. Exact CE/KL and finite-loss identities have maximum
error 4.54e−15; directly checked covariance expansions are within 1.67e−15
in the declared relative units. Covariances are symmetric/PSD to tolerance.
The first-sequence forecast was 392.23 seconds; actual 287.14, under 600.

An independent structural check finds exactly zero Q/K-only and Q/K-last
CE/KL change at the first context position, as required when causal
attention has only one available key. This checks the partition/forward
semantics; it is not an additional scientific prediction or selected bin.

The primary readout, alternative-review choice and scope were fixed before
scoring. During execution but before outcome inspection, the analyzer's
handling of numerically unresolved KL ratios was clarified using the
already fixed 1e−10 qualification tolerance. The original executed protocol
is unchanged; the annotated copy, timestamp and frozen analyzer are retained.
No epsilon, ratio clipping, sample/scale change or outcome-based criterion
was introduced. All observed primary denominators are millions of times
larger than that tolerance, so this clarification affects no decision.

## Decision

The weight-space factorial pattern has a finite predictive counterpart in
both tested backgrounds. Its enlarged short-momentum Q/K response helps
alone but fails the chosen actual-background helpfulness condition, largely
because of ordinary prediction overlap. Keep that distinction as the result.

Close the four-state bridge. Do not add Q/K heads, time points, smaller steps
or a complement factorial to rescue a conditional-benefit story. The result
does not authorize a radius constraint, momentum rule or removal of radial
or overlapping movement. A broader priority discussion should decide the
next question; the observer goal remains open and no new optimizer or
architecture improvement is claimed.

Evidence: [protocol](PROTOCOL.md), [implementation review](IMPLEMENTATION_REVIEW.md),
[independent result review](RESULTS_REVIEW.md), [figure](run1/qk_function.png),
`run1/scalars.npz`, `run1/analysis.json`, `run1/means.csv`, `run1/ratios.csv`,
`readout_clarification.json` and `PROTOCOL_with_readout_clarification.md`.

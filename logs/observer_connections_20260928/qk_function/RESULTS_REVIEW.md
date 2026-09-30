# Independent result review: larger Q/K prediction changes, but the chosen conditional-usefulness test fails

2026-09-28. Recomputed the fixed readout directly from `run1/scalars.npz`
with an independent NumPy reduction, checked per-sequence/bank summaries,
all factorial ratios, numerical identities, and available provenance.
No extra model, gradient, checkpoint-tensor, optimizer or GPU call was made.
The only new file is this review.

**Verdict: P1 and P2 pass; P3 fails. Close the fixed bridge as specified.**
The positive result is substantive: the angular factorial pattern reaches
actual predictive distributions in both backgrounds. The stronger claim
that the enlarged Q/K write supplies material helpful loss change *after
all other actual parameter changes* does not pass. Neither statement should
be erased by the other.

## Independent recomputation and provenance

- Arrays have exactly four states × sixteen sequences × 512 targets, with
  four CE corners, three forward KLs, two conditional KLs, the full 3×3
  centered finite-response covariance, six split/diagnostic terms and
  three finite label-linear terms. All stored values are finite.
- Independently reconstructed all **432 metric summaries** and every
  per-sequence metric, including means, dispersion, signs and extrema:
  discrepancy with the analysis is zero. Recomputed all **96 ratios**:
  discrepancy zero. The two factorial predictions pass all eight bank
  conditions each; the conditional usefulness prediction fails unchanged.
- CE/KL/linear and direct interaction identities reproduce exactly in the
  saved arrays. Maximum overlap/mixed split error is 4.00e−15; maximum
  mixed-loss identity error is 4.54e−15; symmetric-credit identity error
  is 4.44e−16. Covariances are exactly symmetric in storage and satisfy
  the declared PSD check; the minimum scaled eigenvalue is zero.
- The producer's directly reduced mixed-covariance check has maximum
  error 1.66e−15. This direct check cannot be regenerated from scalars alone;
  it is retained as producer qualification, whose code was reviewed.
- A separate structural check in the existing arrays passes exactly:
  at the first causal token, Q/K-only changes cannot affect attention with
  one available key. All 64 sequences' QK-alone and QK-last CE differences,
  the corresponding KLs, and Q/K covariance entries are exactly zero.
- Independently rehashed **100 scientific source files, 12 source JSONs,
  four helper files**, all analysis inputs, and saved x/y tokens. Current
  metadata equals the prior qualified factorial. Target shifts and
  within-bank adjacent-sequence boundaries are consistent.
- Current/executed probe and protocol hashes agree with the producer;
  current/precommitted analyzer hashes agree with the analysis commitment.
  The original/annotated numerical-clarification protocol hashes agree.
- All eight checkpoint files retain the recorded size and mtime. No tensor
  bytes were reread in this review: the producer's full-model hashes and
  exact body-hash comparisons to the angular audit remain its recorded
  qualification, not an independently repeated tensor check.
- Producer execution is complete: **256 scientific forwards + 1 restoration
  forward, 287.14 seconds**, below the 392.23-second forecast and 600-second
  cap. Restored base logits agree exactly. Maximum per-token FP64-versus-FP32
  CE difference is 4.7974e−5, within the unchanged 5e−5 gate. This checks loss
  reduction on the same FP32 logits, not an FP64 network forward.

The separate ratio-resolution clarification retains raw KL values but does
not let means at or below 1e−10 establish a primary ratio. It was recorded
as fixed before outcome inspection, with preserved original protocol and
analyzer commitment. Its effect is independently immaterial here: required
bank-level KL means range from **.00063191 to .00249143**, over six million
times the tolerance at the low end. Recomputing the predicates directly
without that extra resolution condition yields the same P1/P2 outcomes.

## Functional transmission passes in both backgrounds

Pooled ratios (the actual acceptance tests use each bank separately):

| Factorial contrast | Q/K alone KL | Q/K-last KL |
|---|---:|---:|
| beta .8/.9, LR .028 | 1.72543 | 1.45325 |
| beta .8/.9, LR .04 | 1.75498 | 1.67908 |
| LR .04/.028, beta .9 | 1.39679 | 1.25200 |
| LR .04/.028, beta .8 | 1.42072 | 1.44655 |

All individual bank beta ratios exceed one. Their ranges are 1.246–2.222
alone and 1.208–2.084 conditional. All bank LR ratios remain below the
predeclared nominal squared-step multiplier 2.040816: 1.133–1.634 alone,
1.139–1.672 conditional. These are not numerical boundary passes.

Thus the extra weight-space Q/K turn is not simply absorbed by an irrelevant
parameter scale: larger and partly LR-compensated finite predictive changes
are measured in this selected family. This does not isolate radius as their
cause. Each arm has its own representation, gains and base distribution.

Absolute pooled Q/K-alone KL is .000769/.001328/.001075/.001886 in the fixed
arm order; complement-alone KL is .071920/.097584/.074271/.140026. Retain
these magnitudes rather than inventing a Q/K fraction of full KL: interactions
and different divergence backgrounds prevent that decomposition.

## Local usefulness depends on the order, and the chosen order fails

Negative CE changes help. Arm order is LR .028 beta .9/.8, then LR .04
beta .9/.8:

| Arm | Q/K alone | Q/K last, bank 0 / bank 1 | Q/K last, pooled | Symmetric improvement credit |
|---|---:|---:|---:|---:|
| .028 / .9 | −.011529 | −.001652 / −.002669 | −.002161 | +.006845 |
| .028 / .8 | −.014710 | +.002530 / −.000327 | **+.001102** | +.006804 |
| .04 / .9 | −.010986 | +.000135 / −.000803 | −.000334 | +.005660 |
| .04 / .8 | −.014124 | +.005111 / +.010097 | **+.007604** | +.003260 |

P3 required each short-beta arm's Q/K-last change to be negative in both
banks and pooled at most −.005. The low-LR arm disagrees in sign and has
an adverse pooled effect. The high-LR arm is adverse in both banks. Neither
passes the material floor. This is not a near-threshold conclusion.

Q/K alone nevertheless helps in **every bank of every arm**, and the
symmetric two-order credit is positive in every bank. These are legitimate
partial positives, not alternate acceptance criteria. The root explicitly
chose Q/K-last before scoring and preserved that choice in the protocol.
My earlier suggested symmetric credit therefore must remain descriptive;
it cannot rescue P3 after its failure.

The complete step also still improves mean loss in all four pooled states
(−.05007/−.03517/−.04734/−.01849). An adverse Q/K-last margin does not mean
the entire actual step is adverse, nor imply that freezing Q/K would help
a subsequent training trajectory.

## The interaction is primarily additive prediction overlap

Pooled total loss interactions are +.009369/+.015812/+.010652/+.021728.
The corresponding finite mixed-loss terms are only
−.000182/+.000059/+.000229/+.000092. The signed difference is almost entirely
the exact additive-logit overlap term. All terms are retained; the mixed
term is not identically zero and has bank-dependent signs in some arms.

This explains the local alone-versus-last reversal algebraically without
identifying an expendable feature route. A helpful isolated direction can
become locally costly after a large useful complement update, just as the
older body/auxiliary findings showed. The small mixed term does not license
another nonlinear subgroup search. The covariance is a centered *finite*
logit-response covariance, not a measured tangent GN or parameter Hessian.

## Decision and limits

The appropriate update is two-part:

1. Preserve the positive structural-to-functional connection. The measured
   Q/K radius/turn pattern has a real next-step predictive counterpart,
   both alone and on the updated background.
2. Close the proposed **material, actual-background local-usefulness**
   bridge. It failed at the fixed selected states. Do not move its gate to
   the alone margin or symmetric credit, choose a favorable bank/head,
   adjust scales, or add states in response.

A single selected step at four own states cannot establish a training-rate
mediator, universal phase clock, radius-control rule, or causal superiority
of shorter memory. One-step group deletion would also repeat the old greedy-
loss mistake: finite local improvement need not preserve future dynamics.
Banks are contiguous data, not independent model seeds. No new optimizer,
architecture intervention, per-head factorial or training run follows from
this review.

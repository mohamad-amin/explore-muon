# Saved-factor calibration: interpretation and next measurement

2026-09-28. Read the executed calibration and its result; independently
summarized its rows with Python's standard library. No model or GPU calls.

The reconstruction repair is appropriate. FP32 matrix multiplication is not
associative, and this instrument must reproduce the frozen right-first map.
The original failure and source are preserved, the tolerance was unchanged,
and every retained A0/B0 direction now reconstructs with zero relative error.

The calibration qualifies the original informal amplification story. For
the A0/A1 label-only pair, aggregate stage cosines are
`.83754 -> .89904 -> .74057` for pre-NS input, NS output, and final direction.
In fact, the NS-stage cosine increases for all 48 matrices in both examined
pairs. Therefore the observed final sensitivity should not be attributed
to the polar approximation creating extra angular disagreement. The
post-map inverse-root weighting and its interaction with the changed inner
map are consequential. The one-sided hybrid sensitivities are diagnostics;
they do not establish that a one-sided optimizer would train better.

Relative factor differences are large in the independent C/D inverse metric,
especially for V. This is a finite-sample reference geometry, not knowledge
of the exact population factor. In the label-only comparison the bottom
three-quarters-by-rank square block accounts for a median .09365 of raw
factor-difference energy and .70309 after inverse weighting. The block
occupies 56.25% of matrix coordinates: the meaningful observation is the
large redistribution under weighting, not concentration in a tiny subspace.

**Reporting detail:** `torch.median` returns the lower middle value for these
48 entries. Conventional medians, averaging the two middle entries, are
.36194 raw and 1.76914 inverse-weighted for the label-only pair, versus
.64454 and 2.33045 for the disjoint-bank pair. The archived summary's
.36043/1.72050 and .60711/2.30999 remain valid *lower medians*. Identify the
convention rather than silently replacing the preserved summary.

## Next: establish functional relevance before an estimator candidate

I recommend a bounded held-out predictive-GN measurement of the existing
direction differences before comparing a new factor estimator. Inverse
roots deliberately enlarge parameter movement in directions with low
estimated output curvature. Such enlargement can make parameter-space
variation look much larger while leaving model predictions relatively
unchanged. This is particularly relevant after the earlier V/O gauge
qualification and the strong within-/cross-layer coupling measurements.

The most direct statistic is

```
Q(D_a - D_b) / [Q(D_a - D_PD) + Q(D_b - D_PD)],
Q(d) = mean[(Jd)^T (diag(p) - p p^T) (Jd)].
```

Use new, fixed held-out inputs and the full model Jacobian. The logit-space
quadratic form can be evaluated exactly conditional on logits/Jd; it does
not require predictive-label sampling, which would reintroduce the very
Monte Carlo ambiguity under examination. Preselect at least one raw-bank
pair and the independent pooled pair, and report held-out bank agreement.
Do not choose directions by their loss score. Keep the same complete body
scope and normalization as the existing diagnostic. This is a measurement
of prediction sensitivity, not a descent, rate, or improvement claim.

If functional variation is tiny relative to the added TS geometry, close
the argument that the observed parameter angles alone justify repairing the
estimator. If it remains appreciable and repeatable, a bounded same-target
estimator comparison becomes well motivated. A square-root-Fisher random
probe changes fourth moments while preserving its conditional second-moment
target, but has no universal variance advantage over categorical labels.
It should be evaluated as an estimator comparison, with an independent
reference and functional metric, rather than selected solely because its
directions look smoother. No new model run is launched by this review.

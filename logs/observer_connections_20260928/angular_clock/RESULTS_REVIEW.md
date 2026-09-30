# Independent results review: the angular-step compensation is real in the fixed family

2026-09-28. Reviewed the completed `run1/` CSVs/JSONs, reran all contrasts
from the per-head records with an independent standard-library reduction,
and checked source/metadata/scalar provenance. No additional checkpoint
tensors were loaded, and no model, gradient, optimizer or GPU was called.
The only new file is this review.

**Verdict: both predeclared material readings pass exactly as implemented.
The useful conclusion is a missing relative-motion variable, not a discovered
causal clock or a new optimizer.** The fixed comparison should now close.

## Independent arithmetic and integrity

- 12 complete pairs, 576 unique body-matrix records, 1,536 unique Q/K-head
  records, 1,536 matched head-ratio records, 24 contrast cells, and 480 scalar
  radius records. Every pair contains 48 matrices, eight of each kind, and
  128 Q/K heads. All specified states and next-step indices are present.
- Recomputed every paired chord/radius/tangent/step-norm ratio from `heads.csv`,
  all 24 median chord contrasts, extrema, and head-threshold counts. Maximum
  discrepancy with the exports is **zero**. The JSON and CSV contrasts agree.
- Both P1 and P2 pass all eight later cells apiece: two steps × two kinds ×
  the two fixed beta or LR values. Step 9 is retained and does not satisfy
  either late reading; it was never an acceptance test.
- Recomputed all 480 spectral-radius inversions from the original scalar
  step JSONs: maximum discrepancy zero; incoming-weight indices remain t−1.
- Independently rehashed 100 frozen source files and 40 scalar/metadata/status
  JSON inputs, all matching the producer manifest. The current probe/protocol
  hashes match the prepared/result records. Metadata for all twelve accesses
  equals the earlier qualified four-arm factorial metadata.
- The 24 checkpoint files retain the recorded size/mtime, and each consumed-
  tensor manifest contains the expected 48 hashes. **I did not independently
  rehash tensor bytes**, since this review was explicitly to avoid additional
  checkpoint reads. Those hashes and input finiteness checks are supplied by
  the executed producer, whose code was reviewed before execution.
- Maximum recomputed relative radius-identity error: 3.28e−14. Maximum absolute
  chord-squared identity error: 2.65e−13, below the predeclared absolute 1e−10
  tolerance. Head-to-matrix norm/energy sums agree to 2.88e−12 relative or
  better; decay-corrected radial arithmetic agrees to 3.11e−15 absolute.
- Runtime was 36.67 seconds, safely below 180 seconds. The first-pair forecast
  of 22.62 seconds underestimated the actual reads; do not describe that
  forecast as conservative or exact. No scientific gate was changed.

## What the fixed contrasts say

Primary numbers below are medians of paired head ratios, separately for Q/K:

| Step | High/low LR, beta .9: Q / K | High/low LR, beta .8: Q / K | Short/long beta, LR .028: Q / K | Short/long beta, LR .04: Q / K |
|---|---:|---:|---:|---:|
| 9 | 1.3424 / 1.3565 | 1.3471 / 1.3606 | 1.0087 / 1.0084 | 1.0118 / 1.0119 |
| 46 | 1.1097 / 1.1140 | 1.1612 / 1.1854 | 1.1494 / 1.1741 | 1.1977 / 1.2466 |
| 83 | 1.0915 / 1.0894 | 1.1372 / 1.1595 | 1.1539 / 1.1707 | 1.2183 / 1.2457 |

A 42.86% increase in configured LR produces only a 8.94–18.54% increase in
these later median angular turns. This is substantial compensation, not
complete invariance. The early LR response remains much closer to nominal.

Shortening beta produces 14.94–24.66% larger later median turns. Each of the
512 matched later head ratios is above one, although only 53–63 of 64 heads
per cell exceed the declared 1.10 threshold. For P1, 46–60 of 64 heads per
cell are below 1.20: individual counterexamples must remain visible. Heads,
layers and checkpoints are dependent observations within one training seed.

Within the later beta comparisons:

- actual head-step norm ratio: .99869–1.00278;
- tangent norm ratio: 1.00026–1.01573;
- incoming radius ratio: .80179–.87440;
- next radius ratio: .80175–.87317.

The larger normalized turns are therefore not a hidden larger head update
norm. Their size is largely accounted for by the smaller weight radii. This
is an exact geometric accounting of the observed writes, not an assumption
that relative angles equal predictive progress.

For each matched head I independently checked

    chord_hi/chord_lo
      = (tangent_hi/tangent_lo) * (radius_next_lo/radius_next_hi)
        * sqrt[(1+cos_lo)/(1+cos_hi)].

Maximum relative discrepancy is 5.27e−15. This avoids multiplying aggregate
medians as though medians distributed over products. The exported raw heads
are the appropriate level for attributing the chord magnitude algebraically.

## Radial accounting rules out the simplest diffusion story

The exact squared-radius increment is

    Delta r² = 2 <W,Delta W> + ||Delta W||².

Summing over all Q/K heads, the ratio of the signed radial cross term to the
squared-step term is:

| Step | LR .028, beta .9 / .8 | LR .04, beta .9 / .8 |
|---|---:|---:|
| 9 | 1.849 / .862 | 3.483 / 2.575 |
| 46 | 5.917 / 2.691 | 7.163 / 2.160 |
| 83 | 4.710 / 3.182 | 2.961 / 1.948 |

Outward coherent radial drift dominates the squared-step contribution in
every later sampled cell. The longer-beta arm has a larger summed ratio in
all six LR×step pairs, but individual heads can have negative radial motion.
Removing the small inward scalar-decay component makes the adaptive radial
cosines more positive; scalar weight decay is not the source of this outward
motion. The FP32 rounding qualification remains.

Thus r≈eta sqrt(t) is **not established** and its pure diffusion premise is
poor at these observed states. The archive supports an optimizer/map-dependent
radial effect. It does not decompose the history of accumulated radius into
stale momentum, the polar map, whitening, gains, or earlier state changes.
Three isolated writes cannot supply that missing historical decomposition.

## Inference limits and the next decision

The earlier momentum×LR result weakened a nominal-body-distance clock. This
new result shows why that was not equivalent to testing the amount of
relative feature motion: normalization and radius adaptation substantially
compress the LR change, and beta changes the angular step under the same
prescribed norm. It is legitimate to update the explanatory picture in that
specific way.

It is **not** legitimate to conclude:

- that Q/K turning causes the loss advantage or its late reversal;
- that momentum warmup works by controlling these radii;
- that the same mechanism applies to non-normalized body matrices;
- that a per-head normalized-weight chord is function-space distance, or
  removes every possible model symmetry;
- that optimizer steps are a universal phase clock;
- that radial drift is wasted update energy to delete.

The two beta arms differ in direction and state as well as radius. Q/K is
only part of the model; RMS gains, auxiliary Adam state, whitening statistics,
functional geometry and other representations still co-evolve. Even exact
scale symmetry at epsilon zero does not show that Q/K is the group carrying
the training improvement. The previous normalization/Track-3 negatives and
closed early mixed-response branch remain applicable constraints.

Close this bounded audit as a positive structural finding. A future causal
comparison would have to distinguish memory from angular amplitude under a
matched condition, rather than infer causality from current-state correlation.
That is a separate design question; this review authorizes no new checkpoint
family, time warp, model probe or training intervention.

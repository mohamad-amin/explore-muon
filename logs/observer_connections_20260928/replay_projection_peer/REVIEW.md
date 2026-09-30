# Independent review of the finite-history replay decomposition

2026-09-28. Read the replay protocol, scalar implementation and results,
original replay source/JSONs, metadata shard manifests and existing norm
logs. Independent standard-library reductions only; no model, checkpoint
tensor, GPU, scheduler or token-content search. This is the only new file.

## Decomposition and tail check

The decomposition is correct. The rows are newest-first: row k contains
batch `500-k`, evaluated at one common full model state W500. The stored
historical clipping factors are all one for both runs. Weights are therefore
`.95^k`, with total mass 19.8815894156, and

    Mstar_100 = A*gbar + sum_k w_k*(p_k-gbar)

holds in the retained 16-coordinate frame. Independent reconstruction of
the saved Mstar projections gives relative errors 1.45e-7 and 2.40e-7,
consistent with the recorded FP32 accumulation/projection tolerance.
The mean, signed cross terms and ratios use the proper common coordinates.

The historical-tail bound is indexed correctly. Starting from zero
momentum, the full-model clipped-gradient norm recursion through step 400
upper-bounds the norm of the body momentum M400; multiplication by .95^100
bounds its contribution to M500. The actual logs are complete. The
reported bounds .0488455 and .0924261 agree with independent reductions
(the explicit clip epsilon accounts for a small difference from simply
using `min(raw_norm,1)`). As a bound on the ideal logged recurrence it is
adequate for the stated scale comparison; it is not a formal machine-
rounding certificate.

The resulting conclusion is correctly scoped:

| Quantity | Muon | PD |
|---|---:|---:|
| Weighted-data residual / constant part | .0548 | .0775 |
| Weighted-data residual / actual projected M | 2.71 | 2.79 |
| Actual old-tail bound / actual-minus-replay difference | .00883 | .00943 |
| Actual old-tail bound / actual projected M | .456 | .361 |

The older actual tail cannot explain the large actual/replay difference.
It is not negligible relative to the tiny remaining actual momentum, so
avoid using the same bound to declare that momentum's precise residual
direction tail-free. The bound says nothing about re-evaluating all older
batches at W500, whose gradient norms were not measured.

The near-unit cosine of actual-minus-replay with the negative constant
part is a helpful descriptor, but largely follows from the displayed norm
separation. It is not an independent mechanistic confirmation. The strong
observation is that weighting recent data at fixed W cannot explain the
large discrepancy, within these retained directions; model-history effects
dominate that discrepancy. Those effects include changing body and auxiliary
parameters, plus the experiment's numerical conventions, not an isolated
body-only feedback mechanism. The smaller weighted-data residual remains
large enough to matter relative to actual M.

## The lag pattern is real as a finite-array descriptor

The code centers each of the 16 coordinates over all 100 rows, then reports
pooled lagged inner products and cosines. It does not subtract vectors from
different eigenframes or reverse the historical weights. The original
positive lag1–10 pattern is therefore not an obvious indexing/sign error.
It is a descriptor of neighboring corpus batches scored at fixed weights.

It is **not** a period-two or period-three dynamical measurement. W does
not change between these replay evaluations. The common model was trained
on the same data and the Ritz frame is selected at its endpoint, so even
an initially randomized corpus would not make these conditioned gradients
iid. Global centering removes one constant vector but does not remove a
smooth age trend, corpus-composition blocks, heteroscedasticity or the
model's differential fit to more-recent training material. The lag-three
maximum is not sufficient reason to fit a period or investigate a new
momentum mechanism.

## One useful, bounded metadata qualification

There is a concrete pre-existing corpus boundary, verified from both
archived manifests, without inspecting token content. The replay spans
input offsets 419,430,400 through 524,287,999 and crosses the 500,000,000
boundary between two 100M-token shard positions in the ordered stream.
Batch 477 straddles it:

- start input offset: 499,122,176;
- end-exclusive input offset: 500,170,752;
- proportion from the next shard: 170,752 / 1,048,576 = .162841796875.

Thus a single metadata regressor q has value 1 for newest-first rows 0–22,
`.162841796875` for row 23, and 0 for rows 24–99. These are the fifth and
sixth shard positions in the manifest, not a claim that numeric filenames
or arbitrary corpus topics define a scientific population.

The parent's proposed **single intercept-plus-q fit to every coordinate**
is the cleanest small qualification. It uses all 100 rows, handles the
straddling batch correctly, and avoids a selected changepoint or dropping
the awkward observation. Write

    p_k = mean(p) + (q_k-mean(q))*b + e_k.

The weighted-data residual then splits exactly into

    b * sum_k w_k*(q_k-mean(q)) + sum_k w_k*e_k.

Report the explained centered Frobenius variation, both component norms,
their signed inner product, and their signed projections on the original
weighted residual. Those projection contributions sum to one but may be
negative; squared norm ratios do not form an additive percentage.
Retain all ten lag descriptors before and after the fit, using both the
remaining and original variance denominators so variance removal cannot
artificially inflate a correlation. The own-variance view describes shape;
the original-denominator view describes how much covariance remains.

As an initial exploratory check, I excluded the one straddling row and
demeaned the two resulting contiguous shard portions. That removed only
about 6.5% of variation in each model and left a lag-three peak in both.
This was not the final declared qualification: it motivated preferring the
fractional all-row regression above. Do not select it over that regression
if their results differ, and do not add further strata or trend fits.

## Interpretation and stopping point

If the q fit removes much of the lag structure, the right wording is that
the structure is *associated with the known shard boundary*. It cannot be
assigned causally to corpus composition: the boundary is also a division
between recent and older exposure at the selected trained state. If most
structure remains, the single boundary-mean explanation is insufficient;
finer corpus structure, trained-model recency and other conditional-history
effects remain observationally mixed. Neither result warrants a shuffle,
replay, centering or momentum intervention.

The required new information to distinguish causal corpus composition from
model-conditioned recency would vary exposure history or evaluate fixed
material at an independently specified model state. That is outside this
saved-array qualification. Do not launch it merely because the lag-three
coincidence is interesting. Preserve the positive ordered correlations as
an unresolved descriptive feature, close the finite-history discriminator,
and keep the main conclusion at the level the artifacts establish.

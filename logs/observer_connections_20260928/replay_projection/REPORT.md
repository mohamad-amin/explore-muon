# Recent-data weighting is small beside the common gradient and large beside momentum

2026-09-28. Arithmetic on the saved100-batch replay at two selected1Mstep500
states, Muon and PDalpha.25, beta.95. No new replay, model, GPU or training.
All findings concern the16retainedRitz coordinates, not the full space.

## Exact finite-history reduction

At fixed W500 the original probe evaluated batches500…401, newest first.
Let p_k be their stored gradient coordinates, w_k=beta^k f_k the historical
clipping weights, A=sum(w)=19.881589, and gbar=mean(p). Exactly:

    Mstar = A*gbar + weighted_data_residual,
    weighted_data_residual = sum_k w_k(p_k−gbar).

All100clip factors equal1 in both cases. Reconstructed Mstar and mean
coordinates match the saved projections within2.40e−7relative error; the
decomposition identity holds to roundoff. No fitted scale or signal threshold
enters this reduction.

| Norm in retained coordinates | Muon state | PD state |
|---|---:|---:|
| Actual momentum M | 0.10716 | 0.25628 |
| Common component A*gbar | 5.30489 | 9.23625 |
| Weighted data residual | 0.29055 | 0.71600 |
| Replayed momentum Mstar | 5.44591 | 9.63459 |
| Actual minus replayed momentum | 5.53450 | 9.79825 |

The data residual is5.48%/7.75% of the common component, and5.25%/7.31%
of the large actual/replay discrepancy. Recent-data weighting cannot explain
that discrepancy in this span. Yet the residual is **2.71×/2.79× actual
momentum**. A perturbation small relative to the fixed-state mean need not
be small relative to the optimizer's remaining input after cancellation and
model-history effects.

Actual momentum opposes the common component (cosines−.834/−.636). The
actual-minus-replay vector is nearly opposite it (.9990/.9980), largely an
algebraic consequence of the scale separation, not independent mechanistic
confirmation. Model-history effects include body and auxiliary evolution;
this is not isolated body feedback.

## Finite-history tail and conditioning

Mstar replays100terms; saved M also contains beta^100M400, with coefficient
.0059205. A triangle bound using all400existing clipped global gradient-norm
logs puts this actual-momentum tail below0.04885/0.09243 in full body norm.
These are about0.9% of the retained-coordinate discrepancy under the probe's
intended orthonormality, but45.6%/36.1% of the small actual momentum. Do not
infer its precise residual direction from a negligible-tail assumption.

This does not bound an unmeasured full stale-free replay tail: older gradients
at W500 can have different norms. The model was trained on these same batches
and historical clips are held fixed. The analysis is conditional on a finite
history, not iid sampling of a population gradient.

## Order structure and a known source boundary

Centered fixed-weight gradients have positive correlations at lags1…10in
both raw replays (ranges.209–.396 for Muon,.024–.322 for PD). These describe
input order at fixed weights, not optimizer oscillation along a trajectory.
Lags are not independent replications.

The manifest supplies one boundary: this history covers419,430,400…524,288,000
input tokens and crosses500M inside batch477. Let q_k be the fraction of a
batch beyond the boundary. Newest-first rows0…22have q=1, row23has q=.1628418,
and rows24…99have q=0. No change point was searched in gradient traces.

A post-hoc intercept-plus-q regression explains6.36%/6.43% of total centered
projection variance. Residual order structure remains: lag3=.340/.266 after
regression, and most other lags remain positive. The single coarse mean jump
does not explain the correlation pattern. No further boundary/token-feature/
periodicity search follows.

The fitted component contributes95.32%/94.40% of the weighted residual's
*signed parallel component*. The remaining weighted norm is0.04967/0.12788,
still46.4%/49.9% of actual projected momentum. This does not mean the shard
causes95% of noise: the regressor and recency weights are strongly aligned,
and fitting/scoring reuse the same rows.

## Calibrate the attribution before interpreting it

Uniformly permuting the100observed centered rows gives an exact finite-set
reference without assuming the trained-model rows were exchangeable. For
centered data Z, centered weights wc and centered regressor qc:

    E||wc^T Z_perm||² = ||wc||² ||Z||_F²/(K−1),
    E[fitted_delta · delta] / E||delta||² = cos²(wc,qc).

Here cos²=.77655, so a large fitted weighted share is partly expected just
from alignment. This is a ratio of expectations, **not** the expectation of
the observed signed share; individual shares can exceed1 or be negative.
The expected unweighted explained fraction is1/99=1.01%.

The observed weighted residual's squared norm is **5.23×/5.37×** the exact
random-row-order expectation. This calibrated finite-array fact shows the
observed order strongly aligns variation with recency weights. It is not a
p-value, population noise estimate, or evidence that shuffling training would
help. Corpus composition and trained-model recency remain confounded. The
pre-calibration output is preserved separately.

## Connection and decision

The SNR audit warned that raw gradient energy does not characterize the
normalized update's signal budget. This is a temporal counterpart: a small
ordered correction relative to a strong common gradient can be large relative
to momentum after model history cancels most of that common component.
Ordinary variance summaries and iid filtering formulas can miss this
distinction. This connects measurements; it is not a new optimizer.

Close the finite-history discriminator with both conclusions intact:
model-history differences dominate the large actual/replay gap, while ordered
data reweighting remains substantial on the scale of actual momentum. Neither
mean removal, shuffle intervention nor a different beta is justified by these
selected states. Separating corpus composition from model-conditioned age
needs an independently chosen comparison, not another regressor on these rows.

`analyze.py`/`result.json` retain all modes, lags, errors, tail logs and hashes.
`shard_check.py`/`shard_result.json` retain the single boundary regression,
residual identity and permutation calibration. Original JSONs are in`inputs/`;
independent reviews are in`../replay_projection_peer/`.

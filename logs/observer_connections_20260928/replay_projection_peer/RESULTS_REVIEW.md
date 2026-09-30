# Fractional boundary qualification: independent result review

2026-09-28. Read `shard_check.py`, `shard_result.json` and the declared
protocol addendum. Independently recomputed the fit, weighted decomposition
and all ten residual lag descriptors from the original replay JSONs using
standard-library scalar arithmetic. No model, GPU, token-content inspection,
new regressor or changed sample. This is the only new file.

## Verification

The implementation is correct. q is the exact input-token fraction beyond
the one manifest boundary: 23 ones, .162841796875, and 76 zeros. Centering
q and every projection before fitting gives an intercept-plus-q regression;
the fit and remaining residual are orthogonal in the unweighted 100-row
metric. That orthogonality is not assumed after EMA weighting.

All checked explained-variance, weighted-norm, signed parallel-share,
cross-term and lag statistics agree with independent computation to at
most 5.6e-16. The retained identity errors are at floating-point roundoff.
Both own-residual and original-variance lag denominators are implemented
correctly. The original pre-fit lag records remain available separately.

| Descriptor | Muon | PD |
|---|---:|---:|
| Centered variation captured by q | 6.36% | 6.43% |
| Fitted component's signed parallel share of EMA residual | 95.32% | 94.40% |
| Original weighted residual norm | .29055 | .71600 |
| Remaining weighted residual norm | .04967 | .12788 |
| Remaining/original weighted norm | 17.09% | 17.86% |
| Remaining/actual projected momentum norm | 46.35% | 49.90% |

The 95% figures are signed projection contributions onto the original
weighted residual, not fractions of raw variance or an additive energy
partition. The fitted and remaining weighted vectors have positive inner
products (.001486/.012343). Keep their norms and cross terms together.

## Why the two percentages differ

This is a genuine distinction between total variation and variation selected
by a particular weighting rule, not a contradiction or numerical anomaly.
The deterministic geometry of this history makes it transparent:

- Uniform mean of q across 100 batches: **.231628**.
- Mean of q under the existing beta .95 weights: **.699286**.
- Cosine between centered q and centered EMA weights: **.881219**.

Consequently the fitted contribution to the original weighted residual is

    b * sum_k w_k (q_k-mean(q)) = 9.297772 * b,

where b is the fitted 16-vector of boundary coefficients. Much of the
remaining per-batch variation cancels under that same sum. A modest
unweighted component can therefore dominate the *recency-weighted shift*.
No new beta or fitted weighting was used to obtain that result.

A precise wording is: **Ordered variation associated with the known
boundary accounts for little total centered variation, but strongly aligns
with the existing recency weights and accounts for most of their signed
shift relative to uniform averaging.**

"Low-frequency variation matters after averaging" is a reasonable general
intuition, but this experiment fits a coarse ordered step contrast, not a
Fourier spectrum or frequency-response model. Prefer the concrete wording
above for the measured result.

## Limits and decision

The known boundary is also an age division in data already used to train
the evaluated model. Its coefficient can contain corpus composition,
model-conditioned recency/forgetting or a correlated age trend. The
regression does not identify which. Do not say that the shard *causes*
95% of noise, that the remainder is iid, or that this discovers a new
momentum mechanism.

The lag structure is only partly reduced: lag three remains .340/.266;
Muon retains positive pooled lags 1–10, while PD's lag two becomes negative.
Preserve those descriptors without interpreting the largest lag as a
period. There is no reason to search more changepoints, age regressors or
token features in this pass.

The main replay conclusion stands. The large difference between actual
momentum and fixed-weight replay is still dominated by model-history
effects in the retained span. This new qualification explains why a small
part of raw projected variation can nevertheless matter to recency
weighting. Even after the fit, the remaining weighted norm is roughly
half the actual projected momentum norm, so it is not negligible at that
scale. Close the bounded analysis with these distinctions; no data-order,
centering, replay or optimizer intervention follows.

## Addendum: same-row fitting creates a substantial calibration baseline

The parent identified an important qualification to the 95% parallel-share
reading. It uses coefficients fitted and scored on the same projection
array. Because centered q is already strongly aligned with centered EMA
weights, a large fitted parallel contribution is partly mechanical even
without ordered structure in the rows.

Let Z be the row-centered K-by-16 array, S its squared Frobenius norm,
`C=I-11^T/K`, x centered q, and wc centered weights. Under a uniform random
permutation pi of these fixed rows,

    E[Z_pi Z_pi^T] = S*C/(K-1).

This gives the exact finite-permutation expectations

    E[explained centered variation fraction] = 1/(K-1),
    E[||delta||^2] = ||wc||^2*S/(K-1),
    E[delta_fit dot delta] / E[||delta||^2]
        = (wc dot x)^2/(||wc||^2 ||x||^2).

For this archive the first value is 1/99, about 1.01%, and the last is
about **.77655**. The observed 6.36%/6.43% explained variation and the
actual/expected squared weighted-residual ratio are useful additional
descriptors beside the parallel shares.

The final expression is a **ratio of expectations**, not the expectation
of the observed signed share. Individual signed shares can exceed one or
be negative. Neither dividing the observed 95% by .77655 nor subtracting
them creates a calibrated significance test or causal attribution.

These are algebraic baselines for statistic behavior under a hypothetical
permutation, not an assumption that trained-model batch gradients are
exchangeable. No simulation, p-value or new sample is needed. The observed
decomposition remains valid, but the high parallel share should not be
presented alone as strong evidence that a special ordered component was
discovered; substantial alignment is built into fitting and scoring on
the same array. Preserve the raw-variation, weighted-energy and
same-row-calibration views together.

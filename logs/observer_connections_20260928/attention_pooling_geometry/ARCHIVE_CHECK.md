# Fixed scalar consistency check before any attention capture

2026-09-28, before reducing the archived fit coefficients. The analytical
identities have passed deterministic numerical qualification. This is a
descriptive check of the **retained fitted estimators**, not a new gradient,
curvature sample, optimizer direction or training comparison.

Read all 32 marginal JSONs (M/PD/S/SPD at 10/50/100/200/500/900/1300/1469),
all eight V matrices in each. Verify recorded arm/step, 2048 sequences, and
context 512 from each frozen training metadata file. Preserve hashes and
all raw `fit_exact` values, including fit/K-FAC residuals. Do not load the
tensor archives or fit new coefficients.

For every cell compute the necessary ensemble-model constraint

    r = ((T-1)*a + b)/T - 1.

The equal-query independent-error specialization additionally predicts
b in [1,T/H_T] and a in [(T-T/H_T)/(T-1),1] for causal attention. General
downstream temporal covariance need not satisfy this narrower b range;
the finite-T coefficient relation still needs iid residual inputs
independent of routing/error covariance. Do not mix these assumptions.

Report every layer and fixed per-method/per-step summaries. Fixed reference
bands |r|≤.05/.10/.25 and matrix-fit residual≤.05/.10/.25 are descriptive
scales only, not significance tests or acceptance gates. No layer/stage is
dropped when the two-matrix fit is poor. Include medians and full ranges
over eight layers; layers and checkpoints are not independent seed trials.

Large systematic disagreement would make the scalar forward-concentration
account unsupported as a quantitative description of these saved fits.
It would not identify which assumption failed, prove population rejection
without estimator uncertainty, or license fitting a modified coefficient.
Agreement would only satisfy a necessary condition; q has not been measured
and must not be inferred from b and then called a successful prediction.
There is no automatic new model capture or training arm after either outcome.

Cost: standard-library JSON arithmetic and a static plot if useful, CPU at
most two threads, no model/tensor calls, expected seconds. Preserve the
complete check; close it without state or factor expansion. The older
source-coverage limitations remain, and no historical sample nesting is
inferred by this within-file scalar calculation.

# Functional relevance of retained direction variation

After completed repeatability and saved-factor calibration, before any new
model evaluation. Independent review in `../ts_repeatability_peer/CALIBRATION_REVIEW.md`
recommends this premise check before an estimator candidate or larger sample.

Fresh TS directions vary substantially in Euclidean coordinates, but inverse
roots expand weak-curvature directions. Therefore direction cosine alone
does not establish that the variation changes predictions. Keep W46, M46,
fixed rank0-C proxy, saved directions and all scalar hyperparameters fixed.

On two disjoint banks of four sequences at offsets2,800,098,304 and
2,900,098,304, measure logit JVPs for the retained A0/B0/AB/CD/ABCD/PD and
pooled-BF16 directions. Reconstruct A1 from its saved B and fixed R using the
frozen CPU map, and verify its parameter-space comparisons against the prior
result. Eight directions total; all48 body matrices, auxiliary parameters fixed.
Apply the common factor−.028 to put them in a step-sized unit (all ratios
are invariant to that factor). These are hypothetical directions with stored
lagged momentum, not actual replayed next updates.

Compute the full8×8 predictive-GN Gram from probability-centered logit
tangents, without sampled labels. Compare functional difference energy for
independent8, repeated labels, independent16, pooling and numerical control
with TS-versus-PD functional differences. Both per-bank values and pooled
values are retained. No line search, true-loss optimization, new rate or
population confidence claim follows.

Qualification: one diagonal and one difference Gram quadratic must match
the frozen direct GN helper within1e−4 relative error (absolute floor1e−8),
with finite/PSD checks. This independently checks reduction, scaling and
difference formation. A timed first sequence forecasts all eight sequences;
stop if projection exceeds300s, with a live300s cap. CPU FP32 forward AD,
FP64 scalar Gram accumulation, explicit math attention, two threads, no GPU.
All outputs under `run2/functional/`; all original artifacts remain untouched.

If functional variation is much smaller relative to the added TS geometry,
the noise bottleneck premise weakens despite low Euclidean cosines. If it
remains material in both banks, a same-target estimator-variance control is
worth considering next. No threshold is tuned to this new metric, and
greater repeatability alone is not equated with optimization progress.

# Independent persistence-analysis review

2026-09-28. Read-only review of the study guide/state, observer synthesis,
`measure_persistence.py`, `measure_frame.py`, `gn_probe.Frame`, the historical
persistence reports, and the two step-500 saved probes. CPU only, two threads;
no model calls, GPUs, training, or writes outside this peer directory.

## Assessment

Fixed input/output rank bins are a reasonable first retrospective analysis.
Their membership comes from separate basis sequences, avoiding selection on
the measured gradient noise. Keep a signed cross-time numerator and signed
debiased squared norms, including negative estimates. A broad rank cell can
contain persistent and reversing components simultaneously; neither a positive
aggregate nor cancellation identifies a stable set of individual coordinates.

For a fixed orthogonal projector P, estimated mean gradients a and b from
independent sequence sets have E[aᵀPb] = μₐᵀPμᵦ. Under independent sequences,
Sₐ = ||Pa||² − Σᵢ noiseₐᵢ/N is an unbiased squared-gradient norm. Correlation
X/sqrt(SₐSᵦ) is a ratio estimator, not unbiased and not bounded by one in a
finite sample. Report X, both S, both subtracted noise traces, coordinate count,
and normalization failure flags; do not clamp correlations or treat cells as
independent replications. A debiased drift norm Sₐ+Sᵦ−2X is an optional useful
companion, subject to the same assumptions.

The contemplated comparison of persistence rank groups with applied-change
energy rank groups is descriptive. It cannot yet show that the individual
coordinates emphasized by an optimizer carry persistent signal. The frame
update includes learning rate, decay, parameter-write effects, and the fresh
training gradient. Its stored momentum is from the preceding update. These
objects must retain their distinct labels.

## Concrete source checks

Full numbers are preserved in `source_checks.json`.

* The persistence probes each use 4096 sequences per side. The frame probes
  each use 8192 sequences and 512 independent basis sequences.
* The frame per-matrix median eight-sequence block-variance ratios have median
  1.1044 for Muon and 1.0975 for PD. Their ranges are 1.056–1.247 and
  1.043–1.247 respectively. The ideal independent-sequence ratio is one.
  Thus exact 1/N debiasing is an assumption already showing mild empirical
  tension. These medians are not an energy-weighted variance correction.
* The frame and persistence eigenvalue vectors agree to at most 2.6e-8
  relative norm error. The generation code uses the same basis data and label
  seed. This supports similar frames, but does not prove identical eigenvectors;
  eigenbases were not retained. Median neighboring eigenvalue ratios at rank
  64 are about 1.008 on the output side and 1.014 on the input side, so small
  boundary movements between nearby eigendirections remain plausible.
* The frame measurement range contains both persistence sequence ranges.
  A selection based on its measured means would therefore share data with
  the persistence calculation. Fixed rank bins avoid that selection issue.
* All later persistence states use the same sequence set B, paired against
  set A at the base state. Base-to-later numerators can be estimated under
  the independent-set assumption; errors across lags are correlated. Products
  between two later-state estimates are not independently noise corrected.

## Controls and interpretation

1. Keep input and output rank separate. These are Kronecker-factor ranks, not
   exact GN spectral projectors and not established mean/spike routes.
2. Show displacement share relative to coordinate count, as well as relative
   to raw-gradient norm share. The tail contains most parameters, so large
   total tail energy alone is unsurprising. Compare within matrix kind;
   rectangular up/down groups differ in size and interpretation.
3. As a sensitivity check, recompute norm debiasing with noise multipliers
   1, 1.1 and 1.25. Do not call these calibrated confidence intervals or a
   correction. Persistence data lack groupwise block covariances, and even
   the frame medians cannot determine covariance at long sequence separations.
4. Show per-layer heterogeneity behind kind totals. Coordinate counts are not
   statistical sample sizes: coordinates share sequence and model effects.
   The saved diagonal variances cannot supply a justified full covariance
   standard error for a high-dimensional dot product.
5. Low cosine means directional change relative to estimated norm. It does
   not distinguish deterministic rotation, oscillation, nonstationary useful
   signal, and sampling error. Negative lag-one correlation alone does not
   establish period two. Conversely, positive lag-one correlation does not
   establish persistence over the optimizer's memory window.

The existing dense-lag probes are a promising subsequent check if step-500
rank groups show a coherent pattern, because their sign reversals can separate
some short-lived alignment from longer-lived structure. They are fresh rerun
trajectories, with a single base time, and should be analyzed on their own
terms. This review supports the planned bounded first analysis, not a new
optimizer claim or a training experiment.

# Raw gradient signal is not the optimizer's signal budget

2026-09-28. CPU-only retrospective analysis, no model forwards or training.
All results and scripts are in this directory. This is a constraint on the
mechanism inferred from the experiments, not a challenge to their recorded
validation losses.

## 1. The near-100% SNR-energy statistic has a near-100% pure-noise baseline

The existing `frame_snr_probe.py` takes 64 microgradients totaling 1M tokens.
Per entry it estimates signal as `q=max(mean^2 - variance/64,0)` and computes
`SNR_B=q*B/(variance*micro_tokens)`. The reported energy fraction uses q both
as a weight and to select whether the entry clears SNR=1.

For iid zero-mean Gaussian microgradients in a fixed frame, this is equivalent
to `t_63^2 > 1 + 1M/B`. Exact finite-sample calculation, independently checked
by a peer and by two million simulated entries, gives:

| Extrapolated batch | Fraction of pure-noise entries above threshold | Fraction of positive estimated signal energy above threshold |
|---|---:|---:|
| 1M | 16.22% | 86.11% |
| 4M | 26.78% | 98.69% |
| 16M | 30.66% | 99.91% |

The real raw-basis entry fractions exceed this null, and their 1M energy
fractions do as well. Actual gradients clearly need not be pure noise. The
result establishes a narrower logical point: **a 99% energy reading from this
estimator at 16M cannot by itself show that the relevant update is signal
dominated**. Increasing the extrapolated batch does not improve the mean
estimate measured on 1M tokens. Positive clipping followed by selection puts
nearly all retained mass above the shrinking threshold even under the null.

`signal_noise_audit.py` regenerates `signal_noise_audit.json`, the aggregate
CSV, and `signal_noise_null.png/.pdf`. Input JSON and source SHA256s are saved.
The Gaussian null is a calibration, not a fitted error model for the network.
The probe's fitted SOAP bases also depend on the same samples; the raw basis
avoids this extra dependence, hence its use in the figure.

## 2. Existing real measurements expose a separate weighting mismatch

The older `frame/` tensors retain independent-basis measurements: a Kronecker
frame learned on 512 sequences, per-sequence means/variances on 8192 other
sequences, and the actual next training step in that frame. This is a 4M-token
measurement of each 1M-batch run, with no extrapolation beyond its sample
budget. We reweighted its entries by actual displacement, not just gradient
signal. Results at step 500:

| Method | Raw positive signal energy in estimated SNR>1 entries | Actual body-update energy in those entries | Estimated slope share outside those entries |
|---|---:|---:|---:|
| Muon | 99.20% | 22.94% | 6.00% |
| PD | 99.59% | 15.60% | 10.57% |
| SOAP-Muon | 99.29% | 17.23% | 13.20% |
| SOAP composed with PD | 99.71% | 13.18% | 16.25% |

Including weight decay or subtracting its archived estimate barely changes
these fractions. The tables retain both. The masks use the same estimated
SNR definition as the raw-energy comparison, with training batch 2048
sequences. They are descriptive classifications, not statistical discoveries.

The pattern survives the predeclared extension to steps 200 and 900:

| Method | Estimated weak-coordinate update energy, steps 200 / 500 / 900 | Their estimated slope share, steps 200 / 500 / 900 |
|---|---|---|
| Muon | 69.1 / 77.1 / 77.8% | 5.0 / 6.0 / 5.8% |
| PD | 78.9 / 84.4 / 83.3% | 8.3 / 10.6 / 10.0% |
| SOAP-Muon | 74.1 / 82.8 / 81.2% | 8.7 / 13.2 / 10.0% |
| SOAP composed with PD | 82.9 / 86.8 / 86.6% | 14.3 / 16.3 / 14.5% |

The gradient signal remains >99% in the estimated high-SNR group throughout.
At step 500, the weak complement has positive measured slope for all six
matrix kinds in all four methods. This observation does not prove each weak
coordinate has useful signal, or that its slope is statistically different
from zero after mask selection. It demonstrates why raw-energy dominance
cannot rule out noise sensitivity in the directions the optimizer actually
emphasizes. Nor does high weak-coordinate update energy imply those updates
are wasteful: their raw signal is smaller by definition, and the polar map
redistributes magnitude across coordinates.

Scope limits matter. These are different optimizers' own states in independent
Kronecker frames, not actual SOAP frames or exact GN eigenbands. The probes'
eight-sequence block-variance medians are 1.04–1.27 at step 500, so sequence
independence is approximate. SNR masks are chosen on the same data used to
estimate slope and signal; no confidence claim is attached. Aggregate signed
signal, unclipped mean energy, variance, full and decay-subtracted displacement,
and per-kind sums are retained in `saved_frame_reweighting*.json`.

## 3. The connection to the dynamics evidence

The recent dynamics measurements describe strong, oscillating gradient
components and much flatter directions carrying most displacement. Static
SNR calls a deterministic period-two gradient component signal, even though
its sign reverses after the next step. Thus three distinct quantities must
remain separate:

1. Mean gradient at a fixed state versus data-sampling noise.
2. Gradient components that persist as the model moves.
3. Their weight after the optimizer's normalization and polar map.

The current argument for sign-like SOAP improvement goes from (1), weighted
by raw mean squared, to a claim about (3), while useful progress also depends
on (2). The null calibration and actual-update reweighting expose two missing
premises in that jump. A useful hypothesis is that whitening suppresses large
oscillatory components and makes numerous weak persistent components more
important, where noise normalization can still matter. This is a connection
to test, not a new mechanism claim or an argument to truncate weak directions.

The existing gradient- versus momentum-second-moment runs have similar losses
and the output-side ablations are real evidence. They are compatible with
signal equalization, noise control, temporal effects, or some combination.
They do not uniquely select among them.

## 4. What can and cannot be recovered from the archive

The SNR JSONs do not retain per-entry gradients, variance or optimizer bases,
so their estimator cannot be repaired retrospectively. The larger `frame/`
artifacts support the complementary analysis above, but do not hold the
actual time-dependent SOAP bases/denominators either.

A tempting independent-split reconstruction would subtract the first-4096
sequence mean in `persistence/` from twice the 8192-sequence mean in `frame/`.
The sample ranges and Muon step-500 checkpoint match, but the output-factor
eigenvalues are not bitwise identical: only 59 of 96 factor arrays match,
with maximum relative eigenvalue difference 7.94e-8. Small eigenvalue changes
do not bound signs or rotations in nearly degenerate eigenspaces. U and V
were not retained. We therefore did **not** treat the coordinate subtraction
as an independent data split. This is a data-retention limit, not a failure
of the underlying model.

The smallest future discriminator is a saved actual optimizer basis, two
independent gradient splits, and measurement of raw and post-normalization
direction energy plus independent held-out slope. Cross-fit mask selection
and scoring; retain signed group signal estimates and uncertainty rather
than clipping every entry into an apparent positive signal. A fixed-state
comparison of total-, signal-, and noise-second-moment maps comes before any
new training arm. A temporal comparison on consecutive saved states must
also distinguish whole-model updates from body-only counterfactuals.

Independent conceptual and analytic review: `signal_noise_peer/PEER_REVIEW.md`.

# Fresh factors are noisy in weights, much quieter in predictions

2026-09-28. Fixed-state CPU study, all48 hidden matrices of TS16M atstep46.
It asks whether the small predictive-label output-factor sample is reproducible;
it does not replay online training or compare training rates. The predeclared
factor test, independent review, numerical controls and subsequent functional
qualification are all retained.

**The Euclidean instability is real, but it does not establish an optimization
bottleneck.** Much of it lies in directions with low predictive curvature at
this state. Moreover, the TS-versus-PD functional difference contains a large
amplitude effect, so even the reassuring functional ratio needs qualification.
This immediate premise check is closed without a new estimator or larger-data
run. Alternative variance-reduction ideas remain proposals.

## What was held fixed

- Model W46 and stored M46; no optimizer call. M46 was used in the preceding
  step and is lagged input here, not a reconstructed next-step momentum.
- One input root R from the retained rank0 post46 covariance sidecar. It is
  a proxy for other matrix owners and roots cached at45, not their actual
  historical R. Historical B/L/EMA state was not retained.
- Alpha=beta=.5, relative damping=.001, frozen five-polynomial CPU FP32 NS,
  per-matrix norm and shape conventions. LR/decay are absent from the direction
  comparisons; the functional probe applies a common−.028 scale.
- Four disjoint8-sequence banks A/B/C/D, with independent second label draws
  on A and B. Labels come from the fixed model, all512 positions per sequence.
  Raw B factors are pooled before normalization. Four independent-anchor
  `.8 B_-j + .2 B_j` updates give a single-refresh proxy, not historical EMA.

CPU FP32 statistics match the frozen implementation's CPU B and gradient
reconstruction exactly. The GPU experiment used BF16 forward/NS; explicit
CPU BF16 NS is a numerical control, not a GPU replay. All actual factors,
roots, retained directions, tokens and source versions are archived.

## Parameter-space result

| Comparison | Whole-body direction cosine | Squared variation / summed squared TS–PD differences |
|---|---:|---:|
| Disjoint8-sequence factors, all six pairs | .6965–.7021 | .442–.449 |
| Same A inputs, different labels | .7406 | .3846 |
| Same B inputs, different labels | .7391 | .3882 |
| Independent16-sequence pools AB versus CD | .8221 | .3098 |
| One-refresh EMA proxies | .9836–.9869 | .0246–.0309 |
| Pooled32 CPU FP32 versus BF16 NS | .999845 | .000298 |

These meet the prospective material-sensitivity flag, including the clarified
requirement that independent16-bank cosine exceed the mean of all six8-bank
cosines. The six pairs share banks and are not six independent experiments.
Repeated labels account for a substantial part of the observed variation,
but two repeated banks do not identify a precise variance decomposition.

The effect is heterogeneous. Mean cosines across the independent8 pairs are
q .834, k .790, v .700, o .915, up .529, down .938. Up and V also remain the
least reproducible when only labels change. Every checked output factor has
zero eigenvalues below the relative .001 damping threshold in the recorded
fraction statistic; this is not an exact population noise-floor diagnosis.

Fresh8 is deliberately not equated to the smoothed online estimate. The
stationary weight ESS of its .8 EMA is about72 sequences, and changing model
states/nonlinear inverse roots prevent deriving online variance from this
single-state result. The much smaller one-refresh effects reinforce that limit.

## Where the map changes the variation

Saved factors were recalibrated using independent CD inverse-root coordinates.
For A0/A1, conventional median relative B difference is .362 in ordinary
coordinates and1.769 in that finite reference metric. The bottom-three-quarter
eigenvalue-pair block's share rises from9.4% to70.3%; that block already has
56.25% of coordinates. CD is an estimate, not a population reference.

With identical per-matrix weights at every comparison stage:

| Pair | LMR input cosine | After NS | Final L NS(LMR) R |
|---|---:|---:|---:|
| A0/A1, same inputs/new labels | .8375 | .8990 | .7406 |
| A0/B0, disjoint inputs | .7948 | .8781 | .7021 |

NS increases cosine for every matrix in these two comparisons. The large
final disagreement is therefore not explained as an instability introduced
by NS alone. Holding either pre- or post-L at A0 gives label-pair cosines
.9037/.8980, versus .7406 when both use the second factor. These hybrid paths
are sensitivity diagnostics; their effects are nonlinear and not additive
noise percentages. Fixed R also determines the metric of the final map.

## Independent prediction-space measurement changes the interpretation

On two new banks of four inputs, with no sampled labels, form the exact
finite-input predictive-GN Gram of eight retained/reconstructed directions.
The body Jacobian and the softmax loss Hessian are used; both a diagonal and
a difference quadratic agree with the frozen direct helper to <8e−8 relative
error. There are eight scoring inputs, shared by all directions.

| Pair | Parameter cosine | Predictive-GN cosine | Variation / TS–PD difference energy in GN |
|---|---:|---:|---:|
| A0/B0 | .7021 | .98843 | .00583 |
| A0/A1 | .7406 | .99313 | .00331 |
| AB/CD | .8221 | .99339 | .00376 |

The A0/B0 functional ratio is .00574/.00592 in the two banks; its per-input
range is .00476–.00781. The label-only ratio is likewise small in both banks.
Thus low parameter-space cosine alone substantially exaggerates the variation
in this local predictive metric.

This is not disappearance of the effect. The functional A0/B0 difference
norm is15.5% of A0's own response norm; label-only is11.7%. Also, PD's response
norm is2.27–2.41× the TS variants', despite matching parameter norms, and
TS–PD GN cosine is .979–.981. The denominator is dominated by amplitude.

As a **post-hoc interpretation check only**, normalizing each response to
unit GN norm gives variation-to-TS–PD ratios .286/.165/.170 for the three
rows above. These do not replace the original endpoint or reopen its
predeclared flag; they prevent promoting the tiny fixed-norm ratio into a
claim that shape uncertainty has vanished. The selected response Gram is
nearly one-dimensional, but this is the span of eight closely related maps,
not the spectrum or rank of the full GN operator.

No true-loss slope, finite-step cross entropy, rare-target effect, historical
EMA path or later representation change was measured. Predictive-GN closeness
does not prove harmless noise or exclude an effect on eventual optimization.

## Decision and connection

Do not launch a new factor estimator or enlarge this point sample merely
because the weight-space directions looked unstable. The more relevant
measurement weakens that immediate bottleneck premise, while preserving
uncertainty about shape and dynamics. Large coordinate changes need not be
large changes to predictions; parameter-norm matching can conceal functional
amplitude differences. This is consistent with the earlier V/O gauge and
body/auxiliary findings, without making them the same mechanism.

An independent prior-art review identified Fisher half-factor propagation
with random signs as a same-expected-B alternative to sampled labels. Its
raw-factor variance is no worse than Gaussian half-factor noise for linear
statistics, but neither has a universal advantage over categorical labels or
after inverse roots/NS. It is documented in `../gaussian_fisher_peer/NOTE.md`
and **has not been run**. Its priority falls after the functional check.

The next distinct question is actual joint-step allocation: which auxiliary
group produced the previously measured positive body–auxiliary loss
interaction? That is an independently observed finite-loss effect, not a
continuation of this covariance-sampling probe. Begin with saved displacements
and current head/clipping records before designing another counterfactual.

## Integrity, failures and cost

Run1 stopped before scientific banks at an overly conservative1303s forecast
that priced all48 factors as2048-dimensional. Run2 timed actual shapes with
30% margin, preserving all matrices/comparisons and the900s cap; it completed
in392.30s. The original executable and cost stop remain intact.

The first saved-factor calibration stopped at its unchanged1e−5 reconstruction
gate because FP32 products had been reassociated. The retry restored the
frozen right-factor-first order, reproducing retained directions exactly;
it completed in65.70s. Original source/failure/log are retained. The functional
probe completed in110.13s under its300s cap. All computation used two CPU
threads, no GPU, no training, and no change to main-study artifacts.

Reproduction: `probe.py --run <new-directory>`, `analyze.py --run run2`,
`stage_calibration.py --out <new-directory>`, `functional_probe.py`, and
`functional_analyze.py`; preserve existing directories rather than overwriting.
Numerical summaries are `run2/analysis.json`, `run2/calibration2/result.json`
and `run2/functional/analysis.json`. The calibration's stored torch.median
summaries are lower medians; conventional medians quoted here were independently
recomputed from all matrix rows. Independent design, implementation, output
and functional reviews are under `../ts_repeatability_peer/`.

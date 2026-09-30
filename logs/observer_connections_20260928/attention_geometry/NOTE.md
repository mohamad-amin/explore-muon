# Attention transports a useful-looking shared mean differently under each optimizer

**Later qualification, same observer session:** the V-coordinate amplitude
ratios below are not invariant under the exact per-head V/O gauge. The
post-O calculation in `../value_gauge/REPORT.md` removes the universal
learned-route magnitude ordering (SOAP-PD is below Muon at step500). Smaller
current joint V/O perturbations survive. The fixed functional probe in
`../value_split/REPORT.md` finds no repeatable extra conditional gain for PD's
constant component. Original measurements below remain intact; they must not
be read as an invariant functional amplitude or causal advantage.

2026-09-28. Retrospective CPU analysis of the four existing 1M-batch, seed-260925
frontier trajectories; eight checkpoints and eight layers each. These models
use bias-free projections, RMSNorm and per-head Q/K RMSNorm. No model was
constructed or run, no GPU or job was used, and no main-study artifact changed.

The new connection is **upstream of attention averaging**: Muon's learned value
projection progressively suppresses its shared input mean relative to centered
input variation, whereas PD and SOAP-PD preserve that route much more strongly.
This helps distinguish two interpretations of the known growing activation
mean: a useless direction that whitening happens to allow to grow, versus a
shared value/bias route that the architecture and optimizer jointly regulate.
The saved evidence supports the existence of the route and different treatment;
it does not establish its usefulness or causal contribution to training gains.

## Evidence across the full trajectories

Let x be the normalized attention input, μ = E[x], C = E[xxᵀ], V = W_v x,
and z = A(x)V be the input of the output projection. Compute the pre-attention
moments directly from the saved C, μ and W_v. The post-attention moments are
already measured and saved for the output projection. No replay is required.

| Final checkpoint; median across layers | Muon | PD | SOAP-Muon | SOAP-PD |
|---|---:|---:|---:|---:|
| Input mean energy / total input energy | .270 | .538 | .508 | .590 |
| Value mean energy / total, before attention | .029 | .293 | .143 | .429 |
| Measured post-attention mean energy / total | .108 | .614 | .430 | .732 |
| V gain on mean / gain on centered variation | .077 | .446 | .134 | .603 |
| Centered energy after / before attention | .194 | .339 | .193 | .421 |

The fourth row is

    [||W_v μ||² / ||μ||²]
    ------------------------------------------ .
    [tr(W_v (C − μμᵀ) W_vᵀ) / tr(C − μμᵀ)]

It corrects for the larger input means: the difference is not merely that PD
presents a larger μ to an otherwise equivalent value map. This ratio starts
near 1 for all four methods at step 10. By step 200 it is .160/.617/.429/.800,
and by step 500 it is .110/.482/.239/.632. For both PD and SOAP-PD it exceeds
Muon's in all eight layers at steps 200, 500, 900 and 1469. SOAP-Muon's smaller
late excess is not universal (six of eight layers at the endpoint). These are
paired layer descriptions, not 32 or 256 independent replicates.

The measured pre- and post-attention mean fractions exceed Muon's in every
layer of all three other trajectories at those four times. The coexistence of
higher mean fractions and better training outcomes makes a simple "less mean
is better" explanation untenable; it does not reverse that claim into "more
mean causes better learning." Their ranking is not monotonic in NLL: PD has a
larger shared mean than SOAP-Muon but worse final loss.

See [the trajectory figure](attention_mean_transport.png) and [all tables](TABLES.md).

## What attention guarantees, and what it does not

Each causal softmax row sums to one, so within each sequence

    A W_v x = W_v μ + A W_v (x − μ).

Thus a constant value component passes through attention exactly. But attention
is data dependent: E[A W_v(x−μ)] need not vanish. **The actual post-attention
mean is not identified with W_v μ.** The last stage of the table uses its own
measured mean, and centered energy is centered separately at each stage. No
sum-of-channel-energy interpretation drops their cross term.

This distinction is empirically material. At the endpoint, the median cosine
between the measured post-attention mean and W_v μ is .744/.972/.944/.978;
||μ_z−W_v μ||/||μ_z|| is .793/.248/.336/.222. The constant input route is more
aligned with the actual mean under the better geometries, while Muon's post
mean is much more influenced by data-dependent selection. The energy-transport
ratio is not an estimate of attention entropy: it combines data correlations,
value directions, and learned attention.

## The within/between split does not isolate a document-specific channel

The archived statistic satisfies

    C = C_within + Cov(sequence means) + μμᵀ.

At the endpoint, the covariance of sequence means contributes only
.0285/.0137/.0179/.0094 of tr(C). Most of the recorded between-sequence second
moment is the shared global mean. Adding a separate coefficient for the
sequence-specific covariance barely helps: the median absolute Frobenius
residual reduction across all 768 Q/K/V marginals is .00052; maximum .04084.
Normalized Gram condition numbers are modest (median 2.05, maximum 5.38), but
this component has small absolute energy, and its fitted coefficient often
becomes negative. Do not treat those coefficients as a physical positive
mixture, or infer a special document-mean optimizer from them.

The existing "mean share" field is uᵀCu/tr(C), where u = μ/||μ||; the mean
energy fraction used here is ||μ||²/tr(C). They are close late, but logically
different. No substantive earlier result is invalidated by that distinction.

The old architecture prediction also needs its residual retained: the
within/between fit was predeclared to halve K-FAC's residual for K and V at
every checkpoint from 50. It does not do so universally. For Muon's V, fitted
residual / K-FAC residual rises from .397 at step 50 to .847 at the endpoint;
SOAP-Muon's final ratio is .535. The corresponding absolute residuals are .723
and .445. PD and SOAP-PD retain stronger fits (.292/.304 absolute endpoint
residuals). Explaining the familiar mean-direction ratio is not the same as
explaining the full marginal orientation. These fit differences can themselves
reflect the larger rank-one mean component in the better methods.

## Connections, prior failures, and competing explanations

- The head-whitening experiment shows that a large shared-mean direction can
  carry useful early token statistics. Attention has an exact constant-value
  route too. The analogy is architectural, not evidence that its contents or
  optimal update rule match the head's.
- The archived spike is often a mean-product gradient, but top-spike deflation,
  mean-only whitening, dropping that step, and capping it did not beat tuned
  Muon. Centering the whole body's C reduced PD's gain from roughly .0172 to
  .0089 NLL. These failures discourage deleting or isolating the mean as a
  cure; both shared and centered structure matter.
- Per-kind alpha scans gave no extra gain, and token-weighted C produced very
  small one-step improvements. This observation concerns the learned value map
  and the route attention preserves, rather than a proposal to change K/V's
  scalar whitening exponents.
- A simpler account is co-adaptation: differently weighted optimizer steps
  lead W_v to suppress different channels, while large means contribute little
  to useful descent. The fixed norm, decay, QK normalization, and attention
  selection all change along the trajectories. Only one seed is present.
- Rank-one activation energy and low-rank raw-gradient energy can coexist with
  useful movement in a broad persistent centered tail. Neither a large mean nor
  the raw spectrum says which channel earns the loss decrease.

## Smallest discriminator

At one retained mid-training state of Muon and PD, take the *actual saved V
parameter displacement* D_v. Its local functional perturbation splits exactly
as D_v μ + D_v(x−μ). Score these two components on independent held-out sequence
sets, retaining their signed slope and full 2×2 GN quadratic including the
cross term. An explicit constant value perturbation implements the first term,
because A1=1; a parameter projector onto μ alone is **not** the same operation
when x also varies along μ. Include the measured data-dependent selection mean.

If the common component has no repeatable productive descent after weighting
by the actual update, the route is primarily co-adaptation and should not drive
an optimizer/architecture branch. If it is productive and is specifically
mis-scaled under Muon across both states, the next question is a controlled
shared-offset versus centered-feature parameterization, keeping its initial
function and update norms matched. A frozen one-step score still does not
establish a sustained training improvement. This measurement is proposed only;
no replay or training has been launched.

## Reproduction and integrity

Run from the project root:

    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python logs/observer_connections_20260928/attention_geometry/analyze.py
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python logs/observer_connections_20260928/attention_geometry/attention_transport.py
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 .venv/bin/python logs/observer_connections_20260928/attention_geometry/summarize.py

`analyze.py` uses all 32 JSON+tensor marginal archives (1536 matrix reports,
768 Q/K/V full marginals), hashes their bytes before and after, and reproduces
the archived two-component residuals with maximum absolute error 4.90e−9.
`attention_transport.py` mmap-loads the 32 original checkpoints and reads only
eight V-weight tensors each, hash-checking those tensors and checkpoint
size/mtime. It checks the checkpoint step and retains the model config. The
large embeddings, other weights, and optimizer tensors are not traversed.
The marginal archive includes every sequence position, whereas the online PD
factor excludes position zero and uses a subsample; this is a description of
the model's held-out activations, not a reconstruction of its optimizer factor.

Probe source: `logs/muon_spectra/second_order_audit_20260926/measure_marginals.py`
and `research/adamw_spectra/gn_probe.py`. They use 2048 paired held-out sequences
at every checkpoint and one model-sampled label draw per sequence. The saved
"exact" GN marginals are stochastic estimates of the full-sequence marginal,
not exact population tensors. They have no independent split-sample errors.
The primary activation/weight transport calculation does not use those sampled
labels; the curvature and fit claims do. Retain that distinction.

# The project already tested an instantaneous differential edge

2026-09-28. Read-only source/artifact audit with a small analytic qualification.
No model or GPU. This corrects an omission in the preceding observer synthesis:
the current16Mprofiles lack a faithful dynamic-feedback reconstruction, but
the project already measured differential gain at older1M/4Mstates on
September27. The derivative concept is not new to this project.

## Existing evidence constrains the next question

`eos_linearized.py` differentiated a smooth FP32finite-NS map at constructed
next momentum beta*M+g, including input normalization and PD's final per-matrix
norm matching. Arnoldi was applied to JG. This differs from the newer static
profile maps of lagged M. All completed records are retained:

| State | LR×leading real eigenvalue | Ratio to frozen scalar bound |
|---|---:|---:|
| Muon beta.95,1M500 | 3.7836 | 0.9702 |
| Muon beta.95,1M900 | 4.1560 | 1.0656 |
| Muon beta.95,4M183 | 4.2518 | 1.0902 |
| Muon beta.9,4M183,LR.014 | 6.6106 | 1.7396 |
| Muon beta.9,4M183,LR.02 | 7.0538 | 1.8563 |
| Muon beta.81,4M183 | 12.1585 | 3.3587 |
| Muon Nesterov beta.95,4M183 | 3.7414 | 2.7820 |
| PD alpha.25,beta.95,1M500 | 3.2785 | 0.8406 |
| PD alpha.25,beta.95,4M183 | 4.9158 | 1.2605 |
| PD alpha.25,beta.9,4M183 | 5.9711 | 1.5713 |

The initially promising near-one examples did not generalize. The universal
instantaneous-edge prediction failed within this instrument. The original
two smoke results are retained separately and do not rescue that prediction.

The construction uses an unclipped fresh validation-stream gradient, sampled
GN, a fixed fresh PD root and FP32NS products. It is not the actual incoming
training gradient, true-Hessian feedback, online cached root or BF16rounding
map. However, actual step184preclip norms for the decisive beta.81, Nesterov
and beta.9Muon exceptions are all below1. Active local training clipping does
not explain those exceptions. Source/data scope differences remain; the
failed prediction should not be explained away.

## Two stronger interpretations are unproven

The random-pair symmetry check does not establish J is symmetric/PSD. On
diagonal input diag(1,.5), the exact five paper-NS polynomials give this
chain-rule Jacobian on the diagonal subspace:

    J = [[−1.1300175,  2.2600350],
         [ 1.0112489, −2.0224979]].

It is nonsymmetric, with eigenvalues0 and−3.1525154. Central differences
agree to1.22e−9relative error. This disproves a general guarantee for finite
NS plus input normalization; it does not show that a trained buffer has
this negative mode. Exact polar and finite NS have different differentials.
PD's momentum-dependent norm matching can also affect symmetry.

An above-bound instantaneous eigenvalue also does not uniquely establish
nonlinear saturation as the stabilizing cause. Stability of a moving/periodic
trajectory involves products of changing joint parameter/state Jacobians.
GN is not the full loss Hessian; evolving roots/SOAP, auxiliaries and decay
were omitted. These scalar records do not identify which effect matters.

An accurate positive-real eigenvalue above the scalar bound still supplies
an unstable mode of that frozen model, even without symmetry. Conversely,
a near-bound leading positive mode does not certify stability: negative or
complex modes were not ruled out. Only six leading-real eigenvalues survive,
without residuals, full Hessenberg matrix or bases. No complete spectral
radius or retrospective convergence certificate is available.

## Decision

The new same-state16Mstatic-map suppression remains valid, as does the
distinction between map value and derivative. A useful next dynamic question
must nevertheless incorporate the old failed instantaneous-edge test.
Repeating it at another selected state is not automatically progress. Do not
repair/relaunch EOS to recover a preferred principle.

The saved fixed-weight replay supplies a cheaper, different discriminator:
recent-data weighting versus model-history differences. Its reduction is
in`../replay_projection/REPORT.md`.

Evidence: original JSON/source snapshots in`inputs/`, `audit.py`,
`result.json`, and independent source review in
`../linearized_edge_peer/REVIEW.md`. The analytic coefficients agree with
the archived paper5recipe/current coefficient table. The old measurement
JSONs have no historical source hash; a current snapshot is not retroactively
such a hash.

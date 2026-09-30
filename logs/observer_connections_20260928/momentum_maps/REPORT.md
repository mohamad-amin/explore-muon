# Whitening changes the frozen map; static suppression is not feedback gain

2026-09-28. Arithmetic audit of 18 saved profiles, no models or GPUs. Inputs
and the probe source are copied with hashes under `inputs/` and `source/`.
The scope is Muon/PD/SOAP-PD states at 16M, beta .9/.8, steps 9/46/83.
Independent scientific/source review: `../momentum_map_peer/REVIEW.md`.

Subsequent historical correction: an older1M/4Mdifferential experiment already
exists and failed its universal instantaneous-edge prediction. The missing
current16Mfeedback reconstruction must not be described as the project's
first derivative measurement. See`../linearized_edge/REPORT.md`.

## A stronger comparison already existed in the archive

The main study's 15:43 interpretation compares each optimizer's actual step
at its own state: whitening methods put much less step energy in measured
stiff directions, and benefit more from shorter momentum. Different states
alone leave operator effects and co-adaptation entangled.

Every saved JSON also scores ideal Muon and full alpha-.5 PD maps of the
**identical lagged momentum at the identical state**, against the same sampled
curvature. Extracting these gives a direct conditional map comparison:

| Same-state PD / Muon ratio | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Fraction of energy in top 16 Ritz vectors | 0.000189 | 0.001774 | 0.006907 |
| Absolute squared norm in those vectors | 0.02466 | 0.04315 | 0.09693 |
| Curvature / squared norm, separate held-out set | 0.000249 | 0.006655 | 0.018842 |

PD's top-16 fraction is 145–5,305 times smaller at all 18 states, including
Muon states. Its raw map norm is 3.52–12.09 times larger, but even absolute
stiff movement is smaller. The effect is therefore not just a denominator
change or optimizer-state co-adaptation. Lower whole-direction curvature per
norm also survives on a separate scoring set. This is positive evidence for
additional covariance-dependent suppression in the frozen map.

The top-four fraction is also lower everywhere. The top-one claim is not
universal: one profile has ratio 1.493, so no claim of suppressing every stiff
eigenvector follows. All references here are to the measured Ritz directions;
no missing eigenspace is reconstructed from JSON summaries.

## What is, and is not, the input to these maps

The checkpoint holds M_s after step s. Both counterfactual maps consume M_s,
but the recorded actual displacement W_(s+1)−W_s uses M_(s+1), incorporating
the fresh training gradient. Thus the reported momentum/actual fractions
are not an input/output pair for a single application of the optimizer.

This has a visible consequence: negative lagged momentum is uphill on the
separate held-out loss at **18/18** states, its polar map at **17/18**, and
its PD map at **16/18**. Every actual next body step descends. The archive's
`quality=a²/(2q)` is positive even for uphill directions because it allows
a negative optimal multiplier. `result.json` adds the nonnegative-step
quantity `max(-a,0)²/(2q)` without changing the original values. It would be
incorrect to rank useful forward steps by the sign-erasing quality alone.

Further scope differences are material: the probe uses exact SVD polar, a
fresh root from the curvature bank, and no per-matrix norm matching. Online
PD uses NS, cached/EMA statistics and per-matrix norm matching; SOAP adds
further changing state. The original profiler's next gradient applies only
the saved body displacement, leaving auxiliaries fixed, as already identified
by the earlier observer. None is an exact replay of the full next training
state or a beta intervention at a common state.

## Muon already redistributes momentum spatially

The polar map itself reduces the normalized top-16 fraction relative to
the same raw momentum to 0.00132–0.00820 of its original value: a 122–757-fold
reduction. Consequently, “Muon has only a temporal filter” is too literal.
The supported distinction is **extra covariance-dependent suppression beyond
the polar map's existing redistribution**. This does not invalidate the
recorded lower-beta training advantage of PD/SOAP-PD.

Shorter beta does not raise every descriptive fraction either. PD's lagged
momentum fraction is slightly lower at steps 46 and 83, while its actual-step
fraction rises 3.52-fold at step 83. Its absolute fraction remains much smaller
than Muon's. These within-method comparisons still involve different states,
GN operators and trajectories, not a measured transfer function.

## The missing dynamical object is a derivative

Tiny stiff projection of phi(M) does not bound the response of phi to an
incoming stiff perturbation. The latter depends on A=Dphi(M). Degree-zero
homogeneity gives A[M]=0, while weak transverse singular directions can have
arbitrarily large gain. `DIFFERENTIAL.md` gives an exact fixed-momentum-norm,
fixed-output-norm counterexample and the source-faithful joint Jacobian.

This turns the loose “spatial filtering substitutes for temporal averaging”
story into a specific question: does whitening lower the relevant differential
feedback gain while retaining useful prediction response? The current archive
establishes a frozen-map value contrast, not this derivative, and cannot answer
that question by reweighting its saved scalar projections.

The body/auxiliary result connects as a constraint, not a shared mechanism.
Most positive interaction there is useful prediction overlap; algebraically
removing it discards most of the embedding update's label-linear descent.
Here low curvature also fails to imply useful descent for lagged maps.
Both require signed progress and dynamic response alongside energy, rather
than minimizing energy/curvature or deleting overlap as an objective.

## Decision and next boundary

Retain the strengthened same-state suppression result. Do not infer that
whitening has no mechanism merely because the causal derivative is absent,
or that the new analysis supplies an optimizer improvement. No new model
probe, training arm, prefilter or eigenvector-deflation method is launched.

A future causal test should compare actual maps at a common state and common
incoming momentum/fresh gradient, include clipping and normalization, measure
the response to a prescribed perturbation, and distinguish true-Hessian
feedback from a GN approximation. Nonstationary feedback requires successive
Jacobians, not a single raw sharpness threshold. That would be a separate
bounded decision, not an automatic continuation of this scalar audit.

`audit.py`, `result.json`, `table.csv`, and `TABLE.md` preserve every declared
state and signs, as well as top-1/4/16 and both curvature sets. The observed
training curves and original scientific artifacts are unchanged.

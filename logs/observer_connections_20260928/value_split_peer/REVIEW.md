# Independent review: constant and centered value-update components

2026-09-28. Scope: proposed CPU-only measurement of the eight actual saved V
displacements from step 500 to 501 at the original 1M Muon and PD states.
Read `RESEARCH_GUIDE.md`, `RESEARCH_STATE.md`, observer `STATE.md`,
`attention_geometry/NOTE.md`, `body_aux/REPORT.md`, and relevant frozen/current
model and probe source. Slurm currently shows the main allocations 2567578
and 2618555 running. No model, training job, or GPU was used for this review.

## Recommendation

Proceed with the bounded premise check, with the interpretation limits below.
The exact split and direct finite-loss comparison answer a worthwhile missing
question: does the component of this saved V displacement associated with the
fixed shared input mean provide local descent on independently selected data?
This is not yet a discriminator between a useful learned route and
co-adaptation, nor an optimizer-specific defect or a training-rate claim.
Co-adaptation and local usefulness can coexist.

Keep the proposed two states, two banks of four sequences per state, and fixed
finite coefficients. Do not add cross-state direction transports, optimize
coefficients, or enlarge the sample in response to an interesting outcome.

## Exact identities

At every V projection, write D = W501 − W500, and freeze the archived input
mean mu from the matching step-500 state. The proposed map is

    W x + a D mu + b D (x − mu)
        = (W + b D) x + (a − b) D mu.

The latter form is a useful implementation/checking identity. It exposes the
intervention as a weight change plus an explicit value bias. These two pieces
are functional interventions; the constant term is not generally a projector
of the original weight matrix onto mu. Such a projector also changes variable
input components parallel to mu.

When a = b = c, every layer equals (W + cD)x. This remains true when earlier
V layers change the live input to later layers. Therefore the full network is
exactly the direct V-only parameter interpolation, provided x stays live in
the centered piece and mu remains a detached constant. Detaching or reusing
the baseline x would break the finite identity. Re-estimating mu during each
forward would define a different intervention.

The recorded attention implementation has no value normalization and has
dropout_p=0. Its causal softmax rows sum to one. Consequently A Dmu = Dmu
within each sequence, including the first position. At a nonzero intervention,
earlier layers may also change later Q, K and attention matrices; that is part
of this exact counterfactual and must not be silently frozen.

At (a,b)=(0,0), logit tangents obey J_const + J_center = J_V-only. This is the
derivative of the full network, so all cross-layer effects are retained.

## What to measure and how to interpret it

Use signed first derivatives s_i = dL/dc_i, where negative is descent. For
softmax predictive probabilities p, retain

    G_ij = mean_tokens [sum_v p_v J_i,v J_j,v
                       − (sum_v p_v J_i,v)(sum_v p_v J_j,v)].

The predictive GN loss model is s^T c + 0.5 c^T G c. Its mixed term is
G_12 a b; do not drop it, double it, or assign additive curvature shares.
Check G symmetry/PSD within floating-point tolerance. Accumulate these scalar
inner products in FP64 where inexpensive. The scope is the joint eight-layer
V displacement; layers are not independent replicates.

GN is not the true Hessian of this coefficient path. The network has nonlinear
logit derivatives, including cross-layer changes in live x, and the loss
Hessian also contains a residual-weighted logit second derivative. Therefore
the exact finite interaction

    I = L(1,1) − L(1,0) − L(0,1) + L(0,0)

need not equal G_12. Disagreement does not by itself invalidate JVP or GN.
The proposed true-loss corners and half step are an appropriate safeguard.
Report both pure and conditional effects. In particular,
L(1,1)−L(0,1) measures the finite constant contribution when the centered
component is already present; L(1,0)−L(0,0) measures it alone. They can have
opposite signs, as the earlier body/auxiliary example cautions.

Negative slopes repeated in both banks support local useful movement in this
operationally defined route. A poor c=1 but better c=0.5 full step is evidence
of local scaling/nonlinearity only. Zero or positive constant slopes close
the proposed fixed local premise check; they do not establish that the learned
mean route is useless. The saved step may participate in oscillatory stability,
or its benefit may depend on Q/K/O/MLP or auxiliary updates held fixed here.

## State, mean, and scale limits

- Muon and PD are each measured at their own states with their own saved
  displacements, LRs, weight decay and write rounding. This is a comparison of
  actual local behavior, not an isolated geometric map comparison. Calling
  one method's route specifically mis-scaled across states would require a
  distinct common-state, common-scale design that this probe does not supply.
- The step is a parameter displacement, not the pre-NS momentum, a fresh
  gradient, or a decay-free optimizer direction. Preserve that terminology.
- The split depends on the selected mean. Changing mu to mu+delta transfers
  Ddelta from the centered component to the constant component, leaving their
  sum invariant. The independently archived 2048-sequence mean is a sensible
  predeclared choice; do not describe the attribution as unique or invariant.
- These means use all sequence positions on a held-out validation sample;
  the online PD factor uses a different position/subsampling convention.
  This probe studies an activation route, not reconstruction of that factor.
  The new never-trained training-stream banks may also differ from the mean
  estimation distribution; disclose the two sources and exact offsets.
- A centered V perturbation need not have zero mean after data-dependent
  attention. If logging it is cheap within an existing pass, retain the
  per-layer average A D(x−mu), Dmu, and their signed inner product. At minimum
  keep this caveat explicit; the two labels mean algebraic components, not
  zero-mean versus all-mean output channels.
- Do not rescale the two pieces for the primary test: the actual saved step
  supplies the relevant coefficients. Parameter Frobenius norm matching is
  not directly defined for an explicit value-bias intervention versus a
  weight intervention. Report each component's logit/GN norm and signed slope
  alongside its raw loss effect. A scale-invariant slope/sqrt(G_ii), when
  numerically resolved, is descriptive; it does not remove state confounding.
- Four sequences per bank are sufficient for a cheap premise screen, not
  precise population estimates. Preserve each sequence and both bank means.
  Contiguous sequence banks need not be independent documents. Treat a bank
  disagreement as uncertainty, not permission to select a favorable average.

## Minimal qualification and stop rule

On a predeclared qualification sequence from each already fixed state, check
baseline reconstruction and diagonal a=b=1 (preferably also 0.5) against
direct V-only weights, using the same explicit-attention execution path.
Check J_const+J_center against a direct V-weight JVP and against central
finite differences at two declared small scales that remain resolved in FP32.
Use scale-relative logit errors with a sensible absolute floor; very small
finite differences can fail only because of cancellation. Preserve failed
attempts and distinguish a numerical repair from a scientific design change.

The planned baseline, (1,0), (0,1), (1,1), and (0.5,0.5), plus two base
tangents per sequence, are the smallest sufficient scientific measurements.
No local coefficient optimization or cross-fitting is needed because the
direction, means and scales were fixed before viewing the new labels. Both
banks should use identical samples across the two methods for paired
description, without treating the different models as independent replications.

Retain the two-thread CPU cap, CUDA disabled, source/input provenance, and
projected 15-minute cost gate. Failure of qualification or the cost gate should
stop this bounded probe with an honest preserved record. Do not turn it into
an architecture or optimizer training branch without the next decision argument.

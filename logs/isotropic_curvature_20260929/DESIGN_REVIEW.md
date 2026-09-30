# Independent design review: isotropic-curvature trajectory atlas

2026-09-29. Advisory design review before scientific model calls. Read the
current research guide/state, the source and inventory, and the Su paper.
No checkpoints, outcome tensors, models, or jobs were evaluated for this review.
This review recommends a design; it does not assert that every recommendation
has been adopted. All subsequent decisions belong in the frozen protocol.

## Decision and purpose

Proceed with a bounded longitudinal atlas. Six original 1M checkpoints
(Muon/PD at 10, 500, 1300) and three fixed MLP-up sites provide a useful first
view of how finite-displacement curvature changes with training and depth.
The saved adjacent writes give these radii operational meaning. This is a
measurement study, not an optimizer competition or a requirement that each
observed mechanism immediately improve a local loss.

The most valuable result would be a picture of where a radial curvature-cost
description works, where it fails, and how the discrepancy evolves. Preserve
the discrepancies, signed responses, and full curves. Do not reduce the
study to whether two random rotations pass an isotropy threshold. Muon and
PD followed distinct trajectories, learning rates, and hardware; common
initialization and exposures support a matched descriptive comparison,
not attribution of every difference to their map alone.

## Separate the two isotropy assumptions

[Su's model](https://arxiv.org/pdf/2511.00674) combines a radial description of
the finite activation-space Taylor remainder with a spherical input model.
Whitening a second moment addresses part of the latter assumption. It does
not establish the former, remove a nonzero mean, or make the entire input
distribution spherical. The paper's finite-radius construction should not
be collapsed into a scalar Hessian assumption.

Generic two-sided rotations of a matrix D preserve its singular values but
change the distribution of activation displacement lengths on actual inputs.
Consequently, unequal remainders can reflect either changed radii or changed
curvature orientation. Separate these effects within the proposed five
directions per panel:

1. Actual adjacent write D.
2. Two independently seeded left-only rotations O_L D.
3. One right-only rotation D O_R^T.
4. Its matched whitened-frame rotation D S O_R^T S^-1, where
   S = (C + .001 mean(eig(C)) I)^(1/2).

This preserves the proposed direction count. Left-only rotations satisfy
||O_L D u_t|| = ||D u_t|| for every token, without any assumption about C.
They therefore provide a particularly clean output-orientation comparison.
The matched right pair addresses how input geometry changes an orbit with
fixed raw or whitened singular values. A left rotation of P = D S maps back
to O_L D: raw and whitened versions are identical, not independent evidence.

Use fixed shared operators across methods, times, and same-shaped sites.
Save the actual operators or their fully reproducible construction. Two
output rotations and one input orientation are exploratory samples; neither
agreement nor disagreement identifies an entire angular distribution. A
cheap structured orthogonal operator is acceptable if named accurately;
do not describe a few Householder factors as a Haar draw.

## What is the displacement radius?

The last MLP-up site is a useful architectural control. Its downstream
MLP-down, residual addition, final normalization, and head are positionwise.
Its per-token loss can therefore be paired directly with that token's
preactivation displacement norm. At blocks 1 and 4, subsequent attention
mixes positions: a token's loss depends on many perturbed activations, not
only its own ||D u_t||. A token-level h-versus-local-radius scatter there is
descriptive, not a clean test of the same pointwise radial model.

Retain both token maps and sequence-level loss/remainder against the whole
activation displacement tensor's Frobenius radius, with any RMS convention
explicit. Left-only rotations preserve that radius too. Depth-dependent
failure can then be distinguished from an indexing mismatch or silently
ignored sequence mixing.

The signed multiples ±{.25,.5,1,2} answer the useful question of curvature
over the actual update neighborhood. They may all lie in the quadratic
regime. In that event the conclusion is “quadratic over these observed
radii,” not that the paper's larger-radius growth is absent. If discovering
a broader H(r) is also a first-pass objective, predeclare a single limited
wider-radius anchor at a fixed state/site, or a fixed activation-RMS radius
grid, before reading results. Do not extend an individual curve repeatedly
until a desired knee appears. Mark λ=1 on both optimizer-relative and
activation-radius plots.

## Derivatives, finite responses, and interactions

For each token or explicitly aggregated unit retain

    h(λ) = L(W + λD) − L(W) − λ a,
    a = dL(W + λD)/dλ at λ=0,
    κ = d²L(W + λD)/dλ² at λ=0.

Plot h against .5 λ² κ, and preserve both signs of λ. The even remainder
and the odd remainder after subtracting the linear term expose different
departures from a quadratic model; averaging them immediately discards
useful asymmetry. True Hessian curvature and h may be negative. Do not clip
them or require a convex radial fit before reporting the observations.

Differentiating scalar mean CE supplies the aggregate slope, not every
token's slope. Per-token h requires the corresponding per-token directional
derivative, for example from a qualified JVP of the token-loss vector.
If per-token second derivatives are too expensive, retain exact aggregate
κ and label that scope. Never broadcast a mean derivative into a purported
token-level remainder. Existing GN helpers are not a true-Hessian estimator.

The joint selected-up response adds valuable dynamical context. Its slope
must equal the sum of single-site slopes; its Hessian minus their sum gives
the selected cross terms. Finite interaction is the joint loss change minus
the sum of single loss changes. These are observations about simultaneous
writes, not automatic evidence that positive overlap should be removed.
For hook implementations use λ D x_current at each site. Cached baseline
inputs are exact for an isolated site, but at joint sites they omit the
changes in later-layer inputs caused by earlier perturbations and measure
a different intervention.

## Numerical and data qualification

Simply converting the model to FP64 is insufficient. The frozen RMSNorm
casts inputs/gains to FP32, the usual CE helper casts logits to FP32, and
the GN helpers contain additional casts. The diagnostic copy must preserve
the architecture, epsilon, GELU approximation and causal masking while
explicitly changing those precision conventions. Qualify its base predictions
against the frozen FP32 interpretation, then qualify derivatives and finite
responses on the actual FP64 function. Agreement need not be bitwise.
Use an attention implementation supporting the chosen higher-order AD;
archive every intentional source change and its hash.

Estimate the uncentered C on the separately fixed calibration inputs, with
the position convention stated. Save C, input mean, centered covariance,
spectrum, regularization scale, and score-bank displacement norm summaries.
Second-moment whitening does not certify out-of-sample isotropy; its score
bank residual is itself useful evidence. Unit-sphere versus identity-second-
moment conventions differ by an input-dimension factor in squared radius.
Keep that scale explicit when relating plots to the paper.

Clarify whether “two score sequences, two banks” means two total or two per
bank. Two total independent contexts are a small exploratory sample, even
though each contains 512 correlated tokens. Shared fixed contexts are good
for comparing evolution; tokens are not independent context replications.

Keep W_s, stored M_s, and D_s = W_(s+1) − W_s separate. The stored momentum
was used for the completed update s; it is not the next write's optimizer
input. Preserve decay/rounding in D, source and tensor hashes, exact token
arrays or hashes, and all numerical failures. C and its coordinate frame
change across checkpoints: compare physical activation effects and invariant
spectra rather than subtracting coordinates in separately fitted bases.

## Bounded execution and first deliverable

The first complete panel should forecast total wall time, memory and disk,
including calibration, orthogonal operators, AD, all signed scales, and
joint extras. Keep the two-thread CPU bound and archive qualification
failures before repair. No optimizer reconstruction or new training is
needed. A full 2048-square random QR is a nontrivial part of this budget.

The first deliverable should be the raw joined measurements and an interpreted
visual atlas: radius coverage, quadratic/finite response curves, matched
orientation comparisons in both frames, depth/time changes, and selected
joint interactions. An unsuccessful radial description is still useful
if the atlas shows which assumption fails and in which regime. Optimizer
design and independent confirmation can follow this empirical model-building
step; they should not determine which curves are retained.

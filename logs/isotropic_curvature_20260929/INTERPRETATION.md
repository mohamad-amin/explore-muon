# How the atlas bears on the isotropic-curvature model

Written before the full atlas finishes. These are definitions and interpretation
boundaries, not numerical findings or an optimizer prescription.

## The model correspondence

Su's paper (https://arxiv.org/pdf/2511.00674, Sections2–3) analyzes

    min_Q -<G,Q> + E_z h(||Q z||),

with isotropic z and a scalar finite-remainder cost h. Under its stated
conditions, the optimizer preserves singular spaces and homogenizes the
spectrum; exact polar requires the stronger sharp-kink conditions. h is not
a matrix Hessian, and the model is an average single-iteration approximation.

Our algebraic extension replaces the input law by u=C^(1/2)z. With
P=Q C^(1/2), the gradient in P coordinates is G C^(-1/2). The polar case is

    Q = c polar(G C^(-1/2)) C^(-1/2),

the ideal half-power PD shape before its practical norm matching. If z is
unit-spherical, C is a scatter rather than the literal second moment; use
sqrt(n) C^(1/2)z and absorb the scalar into h for a literal second moment.
This correspondence does not derive quarter power, damping, SOAP, finiteNS
or momentum. It motivates measuring whether the loss cost is more nearly
radial in an appropriate coordinate system.

## What is measured exactly, conditional on the diagnostic objective

Each score context has a full512-token, FP64 cross-entropy vector. For a
fixed parameter ray W+sD, automatic differentiation supplies the logit
derivatives z', z''. For token t:

    a_t = <p_t-e_y,t, z'_t>,
    q_GN,t = Var_(p_t)(z'_t),
    q_H,t = q_GN,t + <p_t-e_y,t, z''_t>,
    R_t(s) = loss_t(s) - loss_t(0) - s*a_t.

Thus q_H is the true directional second derivative of this smooth diagnostic
loss, while q_GN is its positive predictive Gauss–Newton component. Their
difference is measured explicitly. The remainder can be negative; a negative
observation is not clipped to satisfy the convex scalar surrogate.

There are TWO distinct departures here. H−GN is still part of the quadratic
Taylor term: it arises from the second derivative of the model's logits with
respect to parameters. It must not be called third/higher-order loss behavior.
Only R(s)−.5*s²*q_H measures departure from the true local quadratic. The first
completed early panel illustrates why both are needed: its actual-direction
H is about10times GN, while its s=1 remainder is within1% of the H quadratic.

The local quadratic predicts R_t(s)=.5*s²*q_H,t. Departures with s can reflect
higher derivatives and changing curvature along the ray. Positive/negative
asymmetry reveals effects hidden by averaging the two signs. The full radius
grid is retained even when no apparent power-law region or takeoff exists.

These are exact finite-context derivatives within floating-point qualification,
not population curvature, and not derivatives of the discrete BF16 optimizer.
Training/checkpoint writes stay distinct from the FP64 measuring function.

## The left rotations are the strongest controlled radial test

For D_L=O_L D with O_L orthogonal,

    ||D_L x|| = ||D x||

for every input, not only on average. A difference in their finite remainders
therefore cannot be explained by a difference in the activation-kick norm
distribution. It exposes orientation/baseline dependence neglected by a
shared scalar radial-cost approximation. Two signed-permutation rotations
give a deliberately limited panel, not a full rotation-population estimate.

At the final MLP-up site, each token's subsequent computation is local, so
the pairing of that token's kick norm and its loss remainder is direct. At
earlier sites, later causal attention mixes positions: the natural scalar
example is the whole context, whose activation perturbation is a matrix.
Left rotations preserve its Frobenius norm as well, but tokenwise plots
must not erase cross-position dependencies.

Actual directions were produced by training. Their orientation can be correlated
with curvature through the optimizer and learned state. A difference from
rotations is evidence about that conditional structure; it is not a theorem
that a distributional isotropy approximation is never useful.

There is also an exact connection to input curvature marginals. Let F_in be
the output-index partial trace of the weight Hessian,
F_in[i,j]=sum_a H[(a,i),(a,j)]. Averaging uniformly over left signed-permutation
matrices gives

    E_O q_H(O D) = tr(F_in D^T D) / m,

because E[(O D)[a,i](O D)[b,j]]=delta_ab*(D^T D)[i,j]/m. The same identity
holds for GN. Our two rotations are a small panel, not a certified estimate
of this entire average. The identity nevertheless clarifies why agreement
of a one-sided marginal with C need not predict the actual trained update's
cost: that update can have privileged output alignment lost by the averaging.
The local-transmission follow-up tests one specific architectural source of
that alignment, without fitting a correction to the measured Hessian.

## What the raw and whitened right rotations separate

The raw right rotation preserves the singular values of D but changes which
input directions it acts on. The whitened construction preserves those of
D S, S=(C+delta I)^(1/2), before mapping back. No final norm matching is added.

If output cost were radial and inputs genuinely elliptical under the correct
C, this would motivate orientation invariance in whitened coordinates.
In this finite panel, damping, a nonzero mean, non-spherical higher moments,
finite calibration data, and input–downstream-curvature dependence remain.
Record the actual score-input radii and both parameter/metric norms so a
lower remainder is not automatically called better isotropy or optimization.

## What the curvature matrices do and do not represent

The5x5 Hessian is the Hessian with respect to coefficients on the five recorded
parameter directions. The5x5 GN Gram uses their logit tangents. This basis
is not orthonormal; parameter and input-metric Gram matrices are retained.
Eigenvalues of the raw coefficient matrix are not the parameter Hessian's
eigenvalues. Directional forms and actual coefficient changes remain valid.

The joint3x3 matrices use the actual up changes at blocks1/4/8. The finite
interaction is joint remainder minus the sum of the individual remainders.
Its quadratic prediction is .5*s²*(sum_ij H_ij - trace(H)), including both
orders of off-diagonal entries. This describes these three changes with
everything else fixed, not the whole84-parameter-tensor training update.
Later layers always use their current perturbed inputs.

Joint curves are indexed by their actual-write multiplier. Their changing
intermediate activation radii are not inferred from the separately cached
individual-input radii.

## Why the architectural transmission check normalizes GN, not all of H

For a nonlinear local map r(z)=W_down gelu(z), GN pulls back as
J_r^T GN_r J_r. The true Hessian instead is

    H_z = J_r^T H_r J_r + sum_a (grad_r L)_a * Hessian_z(r_a).

Consequently the measured residual tangent J_r*dz provides a natural
infinitesimal GN normalization, but it does not remove every contribution
to the true Hessian. At finite radius, changing coordinates also changes
the subtracted first-order term: R_z(s) equals the downstream remainder
at Delta r(s), plus <grad_r L, Delta r(s)−s J_r dz>. Simply plotting the
old remainder against a new norm would mix these terms. This is why the
transmission analysis is explicitly limited to tangent GN and retains
the separate H−GN term rather than presenting a fitted cure for it.

## The intended trajectory comparison

We compare early/middle/late geometry within each completed recipe and show
both recipes' own states at common exposure. This can reveal evolving radial
profiles, orientation dependence, and coupling. It does not by itself assign
causality to the optimizer or demonstrate a sustained rate benefit from a
different update. The measurements are intended to constrain an explanatory
model before selecting an optimizer intervention.

The reference write D and its rotated span change with the checkpoint. A
change in a measured Rayleigh quotient therefore reflects both the Hessian
and the direction being probed. Report it as curvature along current writes
or their controls, not as an isolated change in the whole Hessian spectrum.
The saved restricted spectra are not fixed-subspace trajectories or global
sharpness measurements. This scope does not prevent comparing how well the
local Hessian predicts the actual finite step at each state.

## Why mean H–GN agreement need not mean a linear model

For a fixed state, input and direction, the exact model-curvature contribution
is (p−e_y)·z''. If a hypothetical target y were drawn from that state's own
predictive distribution p, its conditional expectation would be zero, even
when z'' is large. Approximate agreement of the true target distribution and
p can therefore produce mean H≈GN without removing network nonlinearity.
This is a mathematical example, not a claim that the evaluated model is
calibrated or that this mechanism explains the observed trajectory.

Our retained mean absolute token difference checks this distinction directly:
a small signed mean and large absolute mean indicate cancellation. The
remaining explanatory question concerns the joint behavior of prediction
residuals and logit curvature, rather than whether every individual logit
has become linear in the parameter direction. The atlas keeps the terms
needed to state that question but does not estimate the unknown conditional
target distribution or identify calibration as the cause.

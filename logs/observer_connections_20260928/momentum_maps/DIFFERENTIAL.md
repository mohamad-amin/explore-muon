# Why a quiet step can still have large feedback sensitivity

Historical addendum: the project already measured conditional finite-NS
Jacobians at older1M/4Mstates. Their universal instantaneous-edge prediction
failed. See`../linearized_edge/REPORT.md`; the calculus here does not establish
a new project mechanism or make an unperformed current16Mmeasurement the
first differential experiment.

This is elementary local calculus, not a new optimizer or empirical claim.
It sharpens the pre-existing observer literature note's warning about using
raw GN sharpness for nonlinear normalized optimizers. The main saved profiles
measure map values; the object below is not retained in those scalar files.

Take 0<epsilon<1/sqrt(2), and define

    M = diag(sqrt(1−2 epsilon²), epsilon, epsilon),   ||M||_F = 1
    E = (e_23 − e_32)/sqrt(2),                      ||E||_F = 1.

The exact polar map is phi(M)=I. Its output norm is sqrt(3), and its projection
on E is zero for every epsilon. Nevertheless,

    Dphi(M)[E] = E / epsilon.

To derive this, at a positive diagonal M the skew component of the polar
differential is (dM_ij−dM_ji)/(sigma_i+sigma_j); the relevant denominator is
2 epsilon. A rank-one curvature model G[D]=lambda<E,D>E therefore gives zero
static stiff energy but differential feedback gain lambda/epsilon, arbitrarily
large at fixed momentum and output norm. There is no contradiction: a function
value does not bound its derivative.

For a fixed root R=diag(r1,r,r), the PD map polar(MR)R also has zero projection
on E. Its derivative along E is (r/epsilon)E. Matching its output norm to
sqrt(3) changes this to sqrt(3)r/(||R||_F epsilon) E. The derivative of the
normalizing denominator vanishes in this direction because its diagonal
output is orthogonal to E. Thus norm matching does not eliminate the issue.
This example also shows how a root can reduce differential gain; it does
not establish that the network's roots do so in relevant directions.

## Source-faithful local dynamics

The main trainer uses unnormalized momentum, M_plus=beta M+g(w), followed by
w_plus=w−eta phi(M_plus). With H=Dg(w), A=Dphi(M_plus), the local Jacobian on
(delta_w,delta_M) is

    J = [[I−eta A H, −eta beta A],
         [    H,         beta I]].

Under frozen symmetric PSD A and H, positive coupled eigenmodes have
characteristic equation z²−(1+beta−eta kappa)z+beta=0, where kappa belongs to
A^(1/2) H A^(1/2). For 0<=beta<1 their strict scalar stability condition is
0<eta kappa<2(1+beta). This conditional result concerns the derivative, not
the Rayleigh quotient of phi(M) or its top-mode energy.

Writing m=(1−beta)M gives the equivalent EMA convention, with an explicit
(1−beta) in the gradient coupling but A_m=A/(1−beta). Those factors cancel
at consistently rescaled operating points. One must not credit beta with a
stability benefit by changing the momentum convention without its derivative.

For the actual training system H is the derivative of the supplied, clipped
gradient; it need not be symmetric or equal to GN. Root refresh, SOAP state,
auxiliary parameters, weight decay and approximate NS add other derivatives.
The exact polar derivative is PSD at smooth full-rank points, and so is the
unnormalized frozen symmetric-root sandwich. Momentum-dependent Frobenius
matching can destroy that symmetry/PSD in general. No PSD stability theorem
for the full optimizer follows from the simplified block.

Finally, a nonzero normalized step is not an ordinary fixed point, and polar
is nonsmooth at zero/rank loss. The block is a local derivative along a
trajectory; stability of time-varying or periodic motion involves products
of successive blocks. Instantaneous eigenvalues alone are insufficient.

## Qualification

`check_differential.py` checks one epsilon=.02 example and a declared fixed
root, including the full 18-dimensional joint Jacobian, against central
finite differences. `differential_check.json` retains all errors and gates.
It does not train a toy, fit a favorable regime, or call a network/GPU.
Independent review and substantive qualifications are in
`../momentum_map_peer/REVIEW.md`.

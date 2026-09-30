# What PD-top can establish after polar mapping and norm matching

2026-09-28. Bounded algebra/source review of the recorded PD-top decision
and preflight, `muon.py:data_norm_root` and the active update/norm-matching
path. No training outcome beyond the published preflight was read. No model,
checkpoint tensor, numerical map, GPU, job action or new experiment was used.
Only this note is written.

**PD-top is an informative ablation of the uncapped input-root recipe. It
does not hold the final stiff-direction update fixed while independently
turning off a tail contribution.** That qualification does not invalidate
the experiment; it determines which outcome statements are justified.

## The coefficient operation is clean; its downstream effects are coupled

Write C=U diag(lambda_i) U^T, c_i=lambda_i/mean(lambda), and

    r_i=(c_i+epsilon)^(-alpha),   r_i_top=min(r_i,1).

The code changes exactly this root-factor branch. The suppressed factors
are identical between the two roots; factors above one become one. Strictly
the cutoff is c_i=1−epsilon, not exactly one, because damping is included.
That small distinction is definitional, not a substantive defect.

For the ideal exact-polar version of one matrix, omitting the descent sign,

    F_R = P(MR) R,
    D_R = k F_R / ||F_R||_F,

where k includes the target Frobenius norm and shape factor. Online execution
uses finite NS, cached/EMA roots and the next momentum, but the same
pre-root/post-root/matching order. If H projects onto retained high-variance
input directions, R_top H=R H, yet

    D_top H = c_top P(M R_top) R H

need not equal c_full P(M R) R H. The tail change generally changes the polar
orientation and the final scalar. The two pieces are not independent additive
updates. Finite NS adds singular-value-dependent response of its own.

A stronger scale qualification is exact for this ideal normalized map: replacing
R by any positive scalar cR leaves D_R unchanged. One could rescale full
PD's root until every factor is below one without changing its update.
Thus absolute root "amplification" is not itself an invariant mechanism;
the meaningful treatment is the changed **relative spectral profile**.
Clamping is nonuniform, so it is a real change rather than that trivial
rescaling. The implemented finite-NS normalizer is analogous away from
its numerical floor, with ordinary precision qualifications.

There is a useful exact allocation identity for full-column-rank square/tall
matrices under exact polar: P(MR)^T P(MR)=I, so

    D_R^T D_R = k² R² / tr(R²).

This covers the ideal shape of Q/K/V/O and MLP-up, 40 of the 48 body matrices.
It shows that capping tail factors increases the normalized budget allocated
to every retained high direction, even though its root coefficient is
unchanged. Polar orientation still matters to alignment with output curvature
and gradients. For wide MLP-down, the identity becomes a normalized
R Pi_(MR) R, with Pi the row-space projector; that projector also changes.
These exact-polar identities are explanatory references, not assertions that
five-step NS produces an exact isometry.

A simple valid undamped example makes the normalization effect explicit.
Take M=I and normalized C=diag(1.9,.1), alpha=1/2, k=sqrt(2). Squared column
energies are:

- Muon: (1,1);
- full PD: (.1,1.9);
- PD-top: (20/29,38/29), approximately (.690,1.310).

Thus the tail root is capped at one, yet its final parameter energy still
exceeds Muon's through redistribution; the retained high-direction energy
is almost seven times full PD's. “No factor above one” is true of R_top,
not a statement that every final weak-direction step equals or lies below
Muon. “Isotropic bulk” should refer to the chosen input metric, not unchanged
bulk updates.

## What the preflight establishes

The published 15–22× reduction versus Muon's map is positive same-state,
same-momentum evidence of extra covariance-dependent static suppression.
Full PD's 226–888× reduction shows the capped map is a substantially weaker
dose by this measurement. The main note now acknowledges that graded dose.

Source scope matters: `step_profile_probe.py` constructs exact-SVD polar
maps of lagged M_s with fresh roots and no per-matrix norm matching or shape
factor. Online training uses NS, cached roots, per-matrix matching, shape
factors and M_(s+1). A global scalar leaves an energy fraction unchanged,
but 48 separate matching factors alter relative matrix allocation and can
alter its projection onto global GN vectors. Therefore the raw-profile
threshold is not a certificate for the precise online normalized recipe,
much less its differential feedback or trajectory stability.

This connects directly to `../momentum_maps/REPORT.md`: Muon already
redistributes momentum spatially, and extra static suppression is distinct
from damping the response to future perturbations. The failed instantaneous-
edge law remains a constraint; it cannot be repaired by renaming this
fraction a stability margin.

## Informative positive and negative outcomes

**If PD-top retains much of PD's gain and the shorter-beta benefit:** the
uncapped r_i>1 factors are not necessary for those outcomes in the tested
recipe. A bounded input-metric modification can suffice. This would be
practically useful and consistent with a suppression account, but would
not uniquely identify it: polar orientation, norm redistribution, radii,
clipping and the learned state also change.

**If PD-top loses the beta benefit or much of PD's gain:** the particular
capped recipe is insufficient at those settings. It would establish the
value of something removed or altered by clamping—not prove that useful
signal/noise treatment in the tail, rather than the weaker final suppression
or different functional step size, is the mediator. The preflight's 15–40×
difference from full PD makes a suppression-strength threshold an obvious
competing explanation. A zero beta gain does not logically refute every
spatial-suppression account. Intermediate outcomes remain intermediate.

**If it retains more relative benefit at 4M than 16M:** that is a batch-
dependent recipe result. It does not by itself establish that low-input-
variance directions are noise or that their amplification only pays at
large batch. Earlier observer work specifically separated low raw energy,
noise and productive update allocation.

All such conclusions remain conditional on the tested LRs/schedules. Equal
parameter norm does not equal equal function-space rate. No additional sweep
is proposed here. The implemented capped root also retains the full covariance
estimation, eigendecomposition and NS work; it is not already cheaper. A
future low-rank implementation would need separate evidence and engineering.

## Coordinate meaning and a genuinely cheap diagnostic

PD-top can be viewed as a floored proxy metric:
R_top=[max(C/mean+epsilon I,I)]^(-alpha), with max defined spectrally.
It preserves orthogonal-coordinate equivariance but does not restore general
affine covariance. The identity reference, trace normalization, damping and
Euclidean norm matching retain coordinate choices, as the prior gauge audit
already explained. This is not a new objection to its empirical utility;
more covariance is not automatically better training.

A small archive-only check could clarify the preflight without changing a
run: use its saved `norm` and `energy_top16` to retain **absolute projected
energy** as well as its fraction, and `curvature/norm²` for the whole
conditional direction. For exact polar and a contracting R_top, the raw
PD-top Frobenius norm cannot exceed the raw Muon norm, so its lower fraction
cannot arise solely from inflating the denominator. This strengthens that
limited static statement. Preserve signed slope; the saved positive
`a²/(2q)` is not evidence that a lagged direction is downhill.

The scalar archive cannot reconstruct how per-matrix online matching changes
the global Ritz projection: per-matrix projections/root data and Ritz vectors
are not retained there. Do not fabricate that reconstruction from rank-0
sidecars or older frames, and do not require a new model replay to keep this
interpretation honest. The present experiment remains a meaningful
necessity question about a capped metric recipe; its outcomes should not be
forced into a unique suppression-versus-tail mechanism verdict.

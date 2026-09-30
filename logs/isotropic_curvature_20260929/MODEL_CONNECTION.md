# From singular values to displacement-norm concentration

2026-09-29. Conceptual connection for the trajectory atlas; no new model
evaluations or numerical analysis. Source: Weijie Su, *Isotropic Curvature
Model for Understanding Deep Learning Optimization: Is Gradient
Orthogonalization Optimal?* ([paper](https://arxiv.org/pdf/2511.00674),
CC BY 4.0). The identities below explain a measurable mechanism; they do not
establish that its assumptions hold in these transformers.

## Exact spherical-input identity

Let z be uniform on the unit sphere in R^n, A=Q^T Q, and s=||Qz||².
Spherical second and fourth moments give exactly

    E[s] = tr(A)/n,
    Var(s) = 2/[n(n+2)] {tr(A²) − tr(A)²/n}.

At fixed Frobenius norm of Q, the mean squared displacement is fixed.
Flattening its squared singular values reduces the variance of that
displacement. For square or tall Q, equal singular values make Q^T Q a
multiple of identity: every spherical input receives exactly the same
displacement norm. For a wide matrix, rank constraints leave unavoidable
variation even when its nonzero singular values are equal. Our selected
MLP-up matrices are tall.

Writing r_eff=tr(A)²/tr(A²), the relative variance is

    Var(s)/E[s]² = 2/(n+2) {n/r_eff − 1}.

Thus high dimension and an effective rank proportional to n already produce
concentration. Literal full rank is insufficient: a few dominant singular
values can still make r_eff small. This distinction can be tracked through
training using the archived direction spectra.

## Why concentration can reduce curvature cost

Write q(s)=h(sqrt(s)). If q is convex, Jensen gives E[q(s)]≥q(E[s]). The
constant-displacement tall-matrix case attains that lower bound. More
generally, spherical symmetry makes the expected convex cost a symmetric
convex function of the eigenvalues of A; averaging those eigenvalues cannot
increase it. Variance alone does not order every arbitrary pair of spectra
for every convex q.

For smooth q and sufficiently concentrated s, a second-order approximation is

    E[q(s)] ≈ q(μ) + .5 q''(μ) Var(s),   μ=E[s].

This is an approximation with higher-moment remainder, not Jensen's exact
identity. For h(r)=k r^4, k≥0, it is exact:

    E[h(||Qz||)] = k {tr(A)² + 2 tr(A²)}/[n(n+2)].

The excess cost at fixed Frobenius norm is therefore a dispersion penalty.
For purely quadratic h, q is linear and this penalty vanishes. When
effective rank is large, a substantial benefit from further flattening
requires sufficiently strong growth of q or a consequential tail region.
The gradient-alignment term also changes with Q, so reduced curvature cost
alone does not imply that complete flattening is the optimal update.

## What this atlas can connect

The archived token displacements let us compare observed squared-norm
variation with the spherical prediction from each direction's spectrum,
then examine whether greater variation accompanies greater finite remainder
at comparable mean squared radius within a state and site. Across checkpoints
the radial cost itself may change, so pooling these associations would not
isolate dispersion. This separates three questions over
depth and training: whether input geometry predicts the radius distribution,
whether radius predicts the remainder, and whether that remainder grows
strongly enough for dispersion to matter. Signed responses and the measured
true quadratic term expose where a convex radial approximation is unsuitable.

For actual inputs x with second moment C,
E||Dx||²=tr(D C D^T) exactly. Fourth moments determine its variance.
Whitening C therefore controls mean quadratic energy, not angular fourth
moments, heavy tails, varying input lengths, or nonzero means. An uncentered
whitened second moment is not a zero-mean spherical distribution. Moreover,
our regularized frame preserves ||D S||_F², which equals calibration mean
squared displacement plus the damping times ||D||_F²; it does not preserve
the undamped energy exactly. The recorded norms make this distinction visible.

The left rotations are especially informative: every token's radius is
identical before and after rotation. A difference in the linear-subtracted
remainder therefore cannot be attributed to a changed radius distribution.
It shows that radius alone misses relevant structure on this comparison.
At block8 this is a clean token-local test. At earlier depths, later
attention allows cross-position interactions; the discrepancy can reflect
that structure as well as output orientation, rather than identifying one
specific source of anisotropy.

Together these measurements can reveal an evolving regime in which spectral
concentration is useful, already sufficient, or overwhelmed by directional
structure. That is a substantive empirical model-building result even
without an optimizer prescription or a universal isotropy claim.

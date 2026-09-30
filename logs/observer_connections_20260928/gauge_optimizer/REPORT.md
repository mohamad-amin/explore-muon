# Invariance does not explain the geometry-decay improvement

2026-09-28. A bounded conceptual branch prompted by the observer's V/O gauge
result. Independent review recommended closing a broad saved-matrix stress
test before execution. Only one fixed2×2algebraic example was evaluated on
CPU; no checkpoint, network, optimizer trajectory or GPU was used.

The possible explanation considered here was that activation whitening,
normalization and geometry decay improve optimization by repairing coordinate
dependence. This was an observer hypothesis, not a claim made by the main
program. The algebra and existing controls do not support that account.

## Distinguish one-sided geometry from a whole circuit

For the exact within-head symmetry V'=SV,O'=OS^-1, the attention circuit
function is unchanged. Gradient/momentum covectors transform as

    Mv'=S^-T Mv,    Mo'=Mo S^T,    Co'=S Co S^T.

For raw SPD input moment C and exact polar, the ideal half-power map

    F(M,C)=polar(M C^-1/2) C^-1/2

satisfies F(M S^T,S C S^T)=F(M,C)S^-1. Whitening the transformed input reduces
the coordinate change inside polar to an orthogonal rotation. This gives
O-side covariance, not V/O-pair covariance: V's gradient has a left transform
and input-only PD supplies no corresponding output metric. An output or
functional scalar normalization cannot generally repair that shape defect.

In the fixed example, O-side covariance holds to5.1e−16relative error, while
V-side error is.892 and the composed tangent error is.364. These are
equivariance errors for an illustrative input, not optimizer performance.

The full ideal undamped half-power sandwich with consistently transformed
left and right metrics preserves both sides (pair error3.2e−16). This is a
conditional geometric identity. It is not the practical onlineTSrecipe and
does not establish a training advantage.

## Scalar normalization, damping and direction are different

Rescaling C by its mean eigenvalue changes the raw half-power map by a
scalar. Under a nonorthogonal input transformation, that scalar changes:
the example's O-side multiplier is1.74256, with ray error below3e−16 after
allowing that scalar. Adding the code's.001isotropic damping changes the ray
as well, here by.00468relative error. A matrix-function identity for raw C
therefore cannot simply be assigned to the regularized implementation.

Per-matrix parameter-Frobenius matching introduces another coordinate-dependent
scalar. It preserves a ray within a chart, but its value changes under the
gauge. Even the fully covariant ideal two-sided map loses pair covariance
after separate leg matching (example pair error.419). Normalization in an
already covariant functional/metric norm could preserve an existing
equivariance, but cannot create the missing V-side transformation of an
input-only map. None of this identifies a better learning-rate rule.

## The useful decay is not the more equivariant decay

Ordinary scalar decay commutes exactly with the fixed gauge:

    rho*(SV)=S*(rho V),    rho*(OS^-1)=(rho O)S^-1.

The resulting composed product rho²OV is the same in either chart. In the
example its relative discrepancy is9.1e−17. Thus fixed-gauge covariance was
not missing from ordinary decoupled decay.

The project's shaped decay W R^p / mean(diag(R^p)) generally does not commute
with a nonorthogonal transformation. At alpha.25,p=2 in the example, O-side
error is.501 and composed-tangent error.257. This is enough to reject the
proposed explanation that geometry decay's empirical benefit comes from
restoring this covariance. It does not invalidate the benefit itself or the
main study's relative-step/equilibrium explanation.

Global Euclidean gradient clipping is also coordinate dependent. Transforming
a saved momentum covariantly defines a conditional map comparison; it does
not reproduce a gauge-transformed training history whose clip factors would
have changed. Moving roots, SOAP state, damping and finiteNS add further
qualifications beyond the exact-polar example.

## Existing controls and prior art constrain a remedy

The main protocol already tested magnitude-matched Muon, retaining Muon's
direction while matching PD's local output energy. It did not recover PD's
gain. PD on the norm-pinned hyperball baseline retained only about.0011NLL
improvement, and lowering scalar decay did not preserve the large final
advantage. These controls do not test every possible normalization, but rule
against presenting a single scalar amplitude correction as an established
explanation or remedy. See MUON_CASE's wave10, Track3and second-order entries.

The literature also distinguishes these objects. Basic K-FAC has a qualified
affine-invariance theorem; GO-MUON explicitly separates its weighted spectral
oracle from the subsequent Frobenius rescaling; Circuit-Muon already couples
V/O updates and addresses gauge balance. Primary-source details and limits
are in`LITERATURE.md`. This branch makes no novelty claim.

## Decision

Close the broad gauge-stress/normalization-remedy branch. The meaningful
qualification is that coordinate covariance, update geometry and beneficial
regularization are different properties. A more invariant map is not thereby
a better trainer, and the successful decay result is not evidence that
invariance was restored. A new empirical branch would need a distinct
observed limitation and a discriminating comparison beyond these identities.

`check.py`/`result.json` preserve the fixed matrices, map variants, errors and
positive controls (<1e−10). `../gauge_optimizer_peer/REVIEW.md` preserves the
independent direction review. Main code/results/queues remain untouched.

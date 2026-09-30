# Review: V/O gauge versus PD normalization and decay

2026-09-28. Read current research guide/state, the observer's value-gauge
identity/report, PD magnitude controls, Track 3 hyperball and geometry-decay
results, and the implemented root/normalization/decay code. No model, GPU,
job, checkpoint-tensor computation or new optimizer experiment. This review
is the only new file.

## Recommendation

**Do not open a broad saved-matrix V/O gauge stress branch now.** Its likely
result, parameterization-dependent updates, follows already from the
operator algebra and would not identify why the trained methods differ.
The valuable result here is a precise separation of three claims:

1. ideal input-PD has a right-coordinate equivariance at alpha 1/2;
2. the complete V/O gauge changes an output coordinate system as well;
3. the implemented normalization, damping, clipping and regularization
   do not constitute a gauge-invariant full optimizer.

A short algebraic qualification is worthwhile because it prevents calling
the Track 3 decay fix a restoration of gauge invariance. An invariant
normalizer remains a possible design choice, but there is no demonstrated
observable failure that singles it out for a new analysis or intervention.

## Correct transforms and the limited ideal identity

Use column activations. Under an invertible within-head block-diagonal S,

    Wv' = S Wv,                 Wo' = Wo S^-1,
    Mv' = S^-T Mv,              Mo' = Mo S^T,
    Cv' = Cv,                   Co' = S Co S^T.

Momentum transforms as a gradient covector, not as a parameter displacement.
The desired displacement transforms are instead `Dv'=S Dv` and
`Do'=Do S^-1`. Covariance EMA means and covariances must transform along
with their samples; scalar EMA weights do not change. Simply applying S
to a gradient buffer would invalidate the proposed comparison.

For positive-definite **raw** C and exact polar, define

    F(M,C) = polar(M C^-1/2) C^-1/2.

Then

    F(M S^T, S C S^T) = F(M,C) S^-1.

One proof writes `S^T (S C S^T)^-1/2 = C^-1/2 Q` for an orthogonal Q and
uses right-orthogonal equivariance of polar. This is the O-side input
coordinate identity. It assumes a fixed coordinate transform, a coherent
transformation of the raw statistic and momentum, no damping, and a
well-defined full-rank polar factor.

It does **not** establish equivariance of the V/O pair. V receives the
left covector transform `S^-T Mv`, while its input covariance is unchanged.
Input-only PD has no output-side metric that would generally turn this
into `S Dv`. Thus the ideal undamped alpha-1/2 pair already fails under
a general nonorthogonal gauge before introducing Frobenius matching.
This missing left transformation must not be blamed on the normalizer.

## What each implementation choice changes

- **Partial power.** Alpha 1/4 or 1/8 generally lacks the right-GL identity
  even before damping/normalization. The empirical successful power is
  therefore not the ideal alpha-1/2 symmetry case by construction.
- **Covariance scale convention.** `muon.py:255–260` divides C by its mean
  eigenvalue. For alpha 1/2 without damping this introduces a scalar factor
  `sqrt(mean_eig(C))` into F. A nonorthogonal S changes that scalar, so
  even the raw update has a scalar equivariance defect before its final
  Frobenius matching. The matching cancels that particular common factor
  but introduces its own coordinate-dependent norm.
- **Damping.** `C/mean_eig(C)+delta I` chooses a Euclidean identity metric.
  It does not transform by congruence under general S. This is a deliberate
  conditioning/regularization convention, not evidence of a bug.
- **Weight-space matching.** Even if `Do'=Do S^-1` held before matching,
  the norms `||Do S^-1||_F` and `||Do||_F` generally differ. Rescaling both
  to the same numerical Frobenius norm changes their pulled-back magnitudes.
- **Finite NS.** For an ideal raw root the transformed NS inputs differ by
  right orthogonal Q, so the Frobenius-normalized polynomial itself is
  orthogonally equivariant in exact arithmetic. Finite NS is not necessarily
  an additional right-GL failure in that particular ideal construction.
  BF16/FP32 rounding and root approximation are separate qualifications.
- **History/clipping.** A fixed linear transform commutes with an ordinary
  unnormalized momentum EMA if its input covectors transform consistently.
  Global Euclidean gradient clipping does not commute with a general S.
  Transforming one saved buffer therefore defines a conditional map test,
  not a transformed history of the complete online algorithm. Owner-local
  statistics, cached roots and missing SOAP state add further qualifications.

Orthogonal S is an essential algebraic control, because the Euclidean
norms, relative damping and exact matrix functions should respect it.
Failure there would be an implementation/convention problem. Failure for
nonorthogonal S is expected and is not by itself a scientific result.

## A scalar counterexample already resolves the broad stress-test question

Choose `S=s I` on all value channels, a permitted special case. For the
implemented input-only PD with covariance normalized by its mean eigenvalue,
the root of O is unchanged by `Co -> s^2 Co`. Positive scalar rescaling of
either momentum is removed by NS input normalization. Consequently its
adaptive Dv and Do stay unchanged under this transformed saved-state test,
whereas equivariance would require multiplying them by s and 1/s.

For the attention branch's linearized product response, the result is

    Wo' Dv' + Do' Wv' = (Wo Dv)/s + s(Do Wv).

The two contributions trade magnitude while the base model function remains
identical. This does not need a new checkpoint computation to establish.
Ordinary scalar step matching or the measured V-only route cannot decide
which resulting function change is useful.

An O-side local output norm `tr(Do Co Do^T)` is invariant when Do transforms
correctly, so it can preserve the ideal O-side identity. But V's local
output norm is in the **internal value coordinates**, which S changes.
Calling that norm "function-space" would repeat the original V-amplitude
identification error. A branch-level V metric needs O and attention mixing;
even a scalar invariant norm cannot repair general missing left-side
orientation. Normalizing two different directions to an invariant magnitude
does not make the directions equivariant.

## Decay: empirical compatibility is different from gauge covariance

Ordinary scalar decoupled decay is already equivariant under any fixed
linear reparameterization: `W -> (1-eta*lambda)W` commutes with S or S^-1.
For both members of the pair it changes their product by the same scalar
factor in either gauge. Its interaction with a non-equivariant adaptive
map can still alter the learned equilibrium.

The code's geometry decay is `W R^p / mean(diag(R^p))`. Even for raw
`R^2=C^-1`, transforming C gives `C'^-1=S^-T C^-1 S^-1`, so multiplying
`W S^-1` by it generally does not yield `(W C^-1) S^-1`. Trace matching
does not repair this. Thus the successful geometry decay should be read
as compatibility between a particular update rule and regularizer in the
chosen parameterization, not restored right-GL or V/O gauge covariance.
Gauge invariance is not a necessary condition for the empirical gain.

## Existing controls constrain the motivation

- **Magnitude-matched Muon:** at the original selection state/recipe,
  matching Muon's per-layer input-weighted output energy to PD gave only
  -.0017 against tuned Muon, versus PD's -.0172. This rejects the simple
  explanation that PD's gain is just a scalar per-layer LR reduction
  (`MUON_CASE.md:716–730`). It does not directly test PD normalized by an
  invariant functional norm; do not claim that unrun variant failed.
- **Hyperball PD:** the direction-only comparison made PD's internal
  Frobenius matching irrelevant, yet its Track 3 gain was only -.0011
  versus the same-GPU MuonH control. Fixing every weight matrix on a
  Frobenius sphere did not restore the larger PD gain; top-input weight
  depletion remained (`:1195–1214`, `:1302–1351`). That sphere is itself
  coordinate dependent, so this is not an equivariance test either.
- **Decay controls:** matching PD's LR/decay shrink recovered only a small
  part of the strong-decay deficit, while geometry decay restored much
  more. Conversely, Muon plus geometry decay lost +.0097 in Track 3, while
  PD plus that decay helped (`:1338–1378`, `:1497–1537`). The interaction
  is not a universal better-decay or scalar-budget effect.
- **Functional route checks:** the larger internal V-mean story weakened
  after O was included; smaller actual joint V/O perturbations survived.
  The local constant-route usefulness test did not establish a repeatable
  PD-specific benefit. Those observations do not supply a missing premise
  for a new gauge-normalization remedy.

These controls leave many conceivable normalizers untested, but they remove
the strongest simple reasons to expect a normalization-only explanation.
An arbitrary coordinate stress is not a replacement for evidence that one
of those normalizers fixes an observed limitation.

## Smallest worthwhile work and stopping boundary

The worthwhile work in this pass is the algebraic/source separation above.
If an explicit equivariance claim must be validated for documentation, a
small synthetic positive-definite matrix identity with orthogonal and one
nonorthogonal S is enough: ideal raw alpha-1/2 O-side mapping should commute;
relative-C scaling, damping and parameter-norm matching can then be labeled
as separate departures. That would be an implementation qualification,
not a new research result, and would require no saved network state.

A saved-matrix analysis becomes scientifically useful only with a narrower
question whose answer could change a decision, such as whether a specific
observed finite functional defect disappears under one predeclared scale
control while the update direction is held fixed. The current archive has
no such identified defect, and earlier controls make it unsafe to invent
one from gauge sensitivity. Do not expand this into missing-covariance
reconstruction, SOAP-state repair, a gauge sweep, or a training proposal.
Retain parameterization dependence as a limitation, and close this branch
unless a distinct empirical premise appears.

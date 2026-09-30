# PD-top changes relative geometry; the raw preflight exposes why normalization matters

2026-09-28. The main PD-top training intervention is ongoing. This note reads
no PD-top training outcome. It combines an independent source/algebra
[review](../progress_lead/PD_TOP_INTERPRETATION.md), tiny deterministic numerical
qualification, and all four existing same-state preflight JSONs. No model,
tensor, gradient, new map, GPU or training call. Scalar/algebra runtime .014s.
The inspected [optimizer](source/muon.py) and [probe](source/step_profile_probe.py)
are copied unchanged with [snapshot hashes](source_snapshot.json). These are
snapshots taken during this audit, not a claim that the original producer
recorded an execution-source manifest. “Top16” below means the producer's
retained leading GN Ritz directions, not a newly recomputed exact eigenspace.

**Finding:** in these four raw map probes, PD-top and full PD have nearly the
same absolute energy in the top16GN directions: their ratio is .87–1.10.
Here energy means squared parameter projection, not curvature-weighted energy.
Yet PD-top's *fraction* of energy there is15–58times larger. Full PD has a
much larger raw norm, allocating far more energy outside that subspace.
This is a substantive distinction between removing a projected component
and redistributing a fixed update budget. Once globally normalized, the
fraction difference is a real difference in projected step energy. Online
training instead normalizes every matrix separately, which these scalars
cannot reconstruct.

## The root operation and the final update are different objects

The implementation changes r_i=(lambda_i/mean(lambda)+epsilon)^(-alpha)
to min(r_i,1). It keeps all contracting root coefficients and caps the
others. With exact polar P, the matched per-matrix map is

    F_R = P(MR) R,       D_R = k F_R / ||F_R||_F.

Two consequences clarify the intervention:

1. **Root scale itself disappears:** D_(cR)=D_R for every c>0. A full-PD
   root could be scaled until none of its eigenvalues exceeds1 without
   changing this ideal matched update. The meaningful intervention changes
   the relative spectral profile; an absolute “factor above1” depends on
   the chosen convention.
2. **Capping changes the allocation even in untouched root directions.**
   For full-column-rank square/tall matrices,

       D_R^T D_R = k² R² / tr(R²).

   Reducing tail coefficients lowers the denominator and raises the matched
   energy in every retained high-variance input direction. This exact-polar
   identity applies to the shapes of Q/K/V/O and MLP-up (40/48body matrices),
   subject to rank. For wide MLP-down the input Gram contains the changing
   row-space projector: k² R Pi_(MR) R / tr(R Pi_(MR) R).

The polar orientation also changes in general, affecting gradient/curvature
alignment. Online finiteNS, cached roots and fresh momentum add further
differences; the exact-polar formulas are qualified reference identities,
not a claim of exact online orthogonality.

A two-dimensional example makes the point concrete. Let M=I,
C=diag(1.9,.1), alpha=.5, zero damping, and target squared norm2:

| Map | High-input-variance column energy | Low-input-variance column energy |
|---|---:|---:|
| Muon |1|1|
| Full PD |.1|1.9|
| PD-top |.689655|1.310345|

The weak-direction root is capped at1, yet its final matched column energy
still exceeds Muon's. The unchanged high-direction root now produces almost
seven times full PD's column energy. Root contraction and final motion cannot
be treated as independent additive pieces.

Deterministic6×6,12×6 and6×12 matrices verify scale invariance below8e−16
relative error and the appropriate Gram identity below3e−15. The simplified
tall identity fails on the wide example as intended (error .187), while the
projector formula passes. The2D values match their exact fractions. These
are mathematical qualifications, not model experiments.

## Four saved preflight states, without selecting a favorable view

The recorded maps use the same lagged momentum at Muon and PD16M states46/83,
with fresh roots and exact polar. They omit online per-matrix norm matching
and shape factors. We retain signed slopes, norms, fraction f16,
absolute projected energy ||D||² f16, and full GN curvature/||D||².

| State | PD-top norm / Muon norm | Full PD norm / Muon norm | PD-top / full PD fraction f16 | PD-top / full PD absolute top16energy |
|---|---:|---:|---:|---:|
| Muon46 |.955|6.913|57.77|1.102|
| Muon83 |.943|3.580|15.08|1.046|
| PD46 |.953|4.486|19.40|.877|
| PD83 |.949|3.948|15.05|.870|

The near equality of absolute projections coexists with a large fractional
difference. It does not imply equal projected vectors, equal direction,
equal suppression after normalization, or equal training behavior. It does
show that much of the raw comparison concerns the denominator: full PD
has substantially more raw energy outside the retained top16subspace. Main's rounded
“15–40× further reduction” misses the ~58×case; all four are shown here.
These raw absolute comparisons condition on the producer's root-scale
convention. Rescaling one root would change its raw energy while preserving
the ideal matched update. This is no statistical equivalence claim.

PD-top's own extra suppression versus Muon survives the absolute view:
absolute top16energy is **16.86–23.62×lower**, and raw norm is slightly
smaller, so that comparison cannot be explained solely by denominator
inflation. Its whole GN curvature per squared norm is **.101–.189 of Muon's**.
Full PD has **17.65–20.71×lower absolute top16energy** than Muon. These are
conditional map observations at the same states, not population estimates.

Every one of the12mapped directions has a positive held-out slope under
the producer's descent-sign convention: all are uphill at that state.
They map saved M_s, not the next update's M_(s+1). Their positive squared
quality a²/(2q) therefore cannot be read as available forward descent.
This reproduces the input-timing distinction already established in the
observer's [momentum-map audit](../momentum_maps/REPORT.md).

## What the running intervention can teach

PD-top is naturally a **floored proxy input metric**:
R_top=[max(C/mean+epsilon I,I)]^(-alpha), with a spectral maximum.
It tests whether the uncapped relative profile is necessary in the tested
training recipe. If much of the gain remains, that is useful evidence that
a bounded profile suffices. It would motivate studying a simpler metric,
without establishing why it works. The current implementation still computes
fullC and its eigendecomposition; it has no demonstrated cost saving.

If the gain or the short-momentum interaction disappears, the capped recipe
is insufficient there. That could reflect reduced final stiff suppression,
changed tail allocation, polar orientation, functional scale or adapted
radii. Its weaker normalized suppression remains a competing explanation;
a negative result does not uniquely identify tail noise/signal as the cause
or refute every spatial-filter mechanism. Both positive and negative outcomes
remain valuable recipe evidence without a forced binary causal reading.

The raw absolute and fractional comparisons must both remain visible. Neither
is a substitute for the online map. Per-matrix normalization can change a
global GN projection, including cross-matrix interference. The JSONs do not
contain the per-matrix projections needed to reconstruct it; no estimate is
fabricated from these aggregate norms. Nor does static suppression measure
the derivative of the update map or oscillation feedback.

This algebra connects the current intervention with the earlier equal-norm
functional-amplitude and gauge findings: a direction's internal coefficient,
its Euclidean allocation, its prediction response and its trajectory benefit
are separate quantities. The new preflight arithmetic makes that distinction
observable in the very experiment now being run. No new training arm or
repair sweep follows from this note.

Artifacts: [protocol](PROTOCOL.md), [code](check.py), [complete results and
hashes](result.json), [all12map records](preflight.csv), [all4ratio records](ratios.csv),
and [independent result review](RESULTS_REVIEW.md).

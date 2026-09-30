# PD-top: relative metric shape, normalization, and the retained preflight

2026-09-28. A bounded observer follow-up to the independent high-level
[interpretation review](../progress_lead/PD_TOP_INTERPRETATION.md).
Main's new intervention is live; no PD-top training outcomes are read here.
No main source, protocol, queue or run is changed.

## Decision argument

Main's root clamp leaves high-variance root coefficients unchanged while
removing factors above1. Its same-momentum preflight reports15–22× less
top16GN energy fraction than Muon, versus226–888× for fullPD. The peer
review notes that polar orientation and per-matrix norm matching couple
the two operations, and that the preflight did not apply online norm matching.
Thus this can test whether the uncapped spectral profile is necessary for
a recipe, but a negative outcome cannot uniquely refute spatial suppression.

A precise missing connection is between the root's scale convention and
the final allocated update. For ideal polar, the matched map is invariant
under R→cR. For a full-column-rank square/tall matrix its input Gram is
D^T D=k²R²/tr(R²). Absolute root factors above1 therefore do not themselves
identify absolute amplification of an applied parameter direction. For wide
matrices an additional row-space projector remains.

Qualify these identities with deterministic tiny matrices, including the
peer's exact two-dimensional counterexample. Separately read all4 existing
PD-top preflight JSONs and expose norm, signed slope, top16energy fraction,
absolute top16energy=norm²*fraction, and whole-GN curvature/norm² for
Muon/fullPD/PD-top. Retain all methods/states, with ratios to the same-state
Muon and betweenfullPD/PD-top. This is scalar recombination, not a new map
probe. The first M46JSON was inspected for schema before this note and
showed very different raw map norms (~157,1084,150); this motivation is
explicitly retrospective. No training score has been selected or inspected.

## What the result can change

If PD-top's absolute projected energy is also lower, its lower energy fraction
is not solely dilution by a larger raw norm. If fullPD/PD-top fraction and
absolute comparisons differ, preserve both: global normalization and48separate
matrix normalizations are different operations. Neither statistic measures
feedback damping, online next-momentum behavior, or a training rate. Signed
uphill slopes remain uphill even when a²/(2q)>0. Do not infer per-matrix
online projections from missing vectors/scalars.

No threshold is used to choose a newoptimizer or launch a run. This check
clarifies what the current intervention changes, so positive/negative main
outcomes can be interpreted without a false suppression-versus-tail dichotomy.
The practical alternative view is a floored input metric, whose relative
spectrum can be bounded while retaining the full-covariance information.
It is not yet cheaper: the implementation still estimates/decomposes fullC.

## Resource and stop

CPU NumPy only, ≤2threads, <30seconds. FourJSONs plus source text; no tensor,
model, GPU, optimizer, scheduler mutation, extra evaluation or training.
All inputs/source hashes and complete outputs retained here. Root will
independently inspect the algebra and ask the reviewer to assess the derived
scope. No expansion to another checkpoint family or new map reconstruction.

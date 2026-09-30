# Independent live-runner review

2026-09-29. Reviewed `run_atlas.py`, its executed copy, the protocol,
instrument, source manifest, and output schema while the producer continued.
Read only small JSON/token/NPZ files from completed outputs; no model calls,
checkpoint loads, large tensor archives, or producer edits. The first system
Python lacked NumPy; the schema checks then ran in the project's existing
venv with numerical thread limits set to two. No scientific result was
selected or excluded by this review.

**Verdict:** no scientific blocker found in the runner. The completed panel's
saved units and identities agree with the source. The final completion check
should explicitly verify finite joint arrays and cross-panel coverage; the
report must distinguish coefficient curvature from norm-normalized curvature.
Neither requires changing the live producer or repeating measurements.

## Freeze and provenance

Current files exactly match their `run1/executed_*` copies at review time:

| File | Executed SHA256 |
|---|---|
| PROTOCOL.md | `969fa9bdc1c95f6afc006fd42969e92124f10df07fc557273b37042a1f8eeea5` |
| instrument.py | `f8e373c52f8cc699e744e449e2cbff52dd36d3e2572427777c3ce3c84bd34314` |
| run_atlas.py | `ddf48aff77479b9d50845b18e5228139f89ebf8c65064239e1e8fee528d83270` |
| source_manifest.json | `9f8d6f94f96d5757fcb40c652107d0d845642ebbcc5ad5f7b2dfcf3ba7a90daf` |

All three copied diagnostic package files match the executed manifest's
hashes. The original source hashes, paired initialization, parameter count,
and train/validation manifests are checked across trajectories. Each state
retains full metadata, original-source checks, actual next-step row, model
tensor hashes, and checkpoint file size/time. Data arrays are saved once;
the same fixed offsets and checked stream manifest are reused at every state.

The protocol's three-hour amendment is preserved with its earlier version,
and the live cap is 10800 seconds. The forecast is made after the first full
state (three depth panels plus joint), rather than immediately after its
first depth panel. This is an execution-granularity difference, not a changed
scientific panel. Progress checks impose a boundary between work units;
they are not a preemptive timeout during an individual derivative operation.

## Units and completed-array checks

The score array has shape [4,512]. Rows0–1 are bank A and rows2–3 bank B;
each row is a separate length512 context. Tokens/targets have the intended
one-token shift within each contiguous bank. Calibration has [8,512] inputs
and targets. Equal context lengths make the mean over four sequence means
equal to the pooled token mean; they remain four context observations, not
2048 independent replications.

The completed `Muon_000010/block01_up/per_token.npz` contains:

- loss [4,5,15,512], with the declared direction labels and signed scales;
- slope, true H, GN, and activation radius [4,5,512];
- projected H [4,5,5], already averaged over tokens within each context;
- projected GN [4,512,5,5], retaining the token axis.

All numeric fields were finite. Maximum absolute discrepancies were:
zero-scale baselines across directions 0; projected-H diagonal versus mean
token-H 2.71e−20; projected-GN diagonal versus token-GN 3.39e−21; left-rotation
token radii 5.55e−17. Summary H/GN means reproduce their arrays exactly.
Equal-bank versus direct pooled H differed only by 1.69e−21 reduction rounding.
These checks concern schema and arithmetic, not the scientific interpretation
of this one panel. Other panels and the joint output were not yet complete
when these bounded checks ran.

## Hessian and joint interpretation

The five directions are not normalized or orthogonal in parameter space.
Their 5×5 matrices are Hessian/GN restrictions in the declared coefficient
coordinates. Eigenvalues of those matrices alone are not Hessian eigenvalues
per unit weight norm. The archived parameter Gram supports that conversion
on its independent span; the input-metric Gram supports a different, explicitly
named norm. Directional H/GN comparisons along the identical vector already
have a consistent denominator.

The joint function uses current layer inputs and the three actual disjoint
weight changes. Its scalar ray corresponds to coefficient vector s·(1,1,1).
The runner checks per-token slope additivity, the agreement of summed 3×3 H
with the joint scalar curvature, and the agreement of summed predictive Gram
with joint GN. Using the separate zero-point actual tangents for that Gram
is correct; coefficient derivatives agree with individual interventions at
zero even though later finite inputs interact.

For three sites, the finite loss interaction is

    I(s) = L_joint(s) − sum_i L_i(s) + 2 L_0.

Equivalently use joint remainder minus the three individual remainders;
that retains the tiny measured slope-additivity residual explicitly. The
quadratic prediction is .5 s²[H_joint − sum_i H_i], equal to the off-diagonal
sum of the joint coefficient Hessian when its diagonal matches individual H.
The saved arrays contain everything needed to check that diagonal identity
without new forwards. Do not omit the restored baseline terms or subtract
three uncentered losses directly.

## Archive coverage and final checks

Stored momentum is M_s from the completed update s; actual D is the FP64
difference of saved weights s+1 and s, including decay/rounding. Labels and
the original mapping preserve this distinction. Tensors retain the five
directions, full C/factors, calibration inputs, score inputs/preactivations/
residuals, sequence gradients, summed-loss output adjoints, momenta, and
before/after weights. C means, centered covariances, nonlinear H−GN, finite
remainders, relative radii, and sequence radii are recoverable from those
saved quantities; not every derived quantity is a standalone field.

Before final reporting, check all 18 final `per_token.npz` files, six
`joint.npz` files, labels/scales/context ordering, and finiteness throughout.
The producer asserts finite individual finite-radius losses but lacks an
explicit analogous assertion on every joint finite-radius loss. NPZ permits
NaNs, so the final analysis should supply that check rather than rely solely
on `status=complete`. Also check joint-H diagonals against individual H.
Use final `per_token.npz` as the complete panel: `partial.npz` is a recovery
snapshot written before the current context's last H/GN matrices are filled.

The final input-manifest rehash and source rehash strengthen completed-run
provenance. If execution stops early, report the partial coverage and preserved
failure instead; the root result/manifest is intentionally written only at
successful completion. No issue found warrants interrupting the live study.

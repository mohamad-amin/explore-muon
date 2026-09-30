# Streaming observations from the fixed atlas

The full atlas is now complete. See [REPORT.md](REPORT.md) for complete results;
the partial observations below are preserved as their original dated record.

These entries are dated partial-panel observations. Final conclusions require
the complete18panel dataset and independent result review. No observation
changes the declared states, depths, directions, radii, or data.

## First complete state: Muon at update10

All three selected depths and their joint panel completed. Two score banks
with two full contexts each are retained. Full projection/rotation/derivative
checks pass. Source: run1/Muon_000010/.

| Up matrix | True H / GN along actual write | Actual H / mean H of two left rotations |
|---|---:|---:|
| Block1 |10.203|20.141|
| Block4 |9.112|9.340|
| Block8 |7.614|6.403|

Left rotations preserve every position's activation-displacement norm. At
block1 the actual/left ratio is19.855 and20.439 in the two separate banks.
The difference therefore is not caused by changing the measured kick-radius
distribution. At block8 the post-up suffix is token-local; earlier blocks
include later attention and cross-position effects.

Lower curvature is not automatically a better direction. In block1 the
actual slope is−.0026301, while the two left slopes are−.00002058and−.00002551.
Rotation reduces the measured curvature but also destroys most first-order
alignment. The observation tests the radial cost approximation; it does not
recommend replacing the actual direction by a rotation.

The first state already separates two issues. Its actual block1 remainder
at s=1 is.99198times the true local quadratic, while the left-rotation
remainders are also close to their own quadratics. Thus strong orientation
dependence exists inside a locally quadratic neighborhood. At s=+16 and−16,
the actual remainder is.88059and1.13785times that quadratic; the two left
rotations remain within about.3%. The signs reveal an asymmetric departure,
not an established universal superquadratic radial law.

The three-up joint Hessian cost is1.697times the sum of its individual costs;
the GN ratio is1.579. At s=1, the finite interaction is3.20368e−5NLL, versus
3.25042e−5from the true-Hessian cross terms. At +16/−16 the interactions are
.0065983/+.0104100 against the common quadratic prediction.0083211.
This is the selected three-up interaction, not the all-layer/full-update
coherence measured in the main program.

These observations are conditional on this early state and these contexts.
They motivate reading the later states carefully: the true-Hessian model
nonlinearity term is material here, and a common norm-only cost does not
describe the tested orientations exactly. They do not establish population
isotropy failure at every state, a new optimizer, or a training-rate mechanism.

## Middle Muon state: agreement in means conceals cancellation

Muon500 completed all three depth panels. Along its current actual writes,
H/GN is .98765/.96663/.96904 at blocks1/4/8, compared with10.20/9.11/7.61
at update10. Actual/mean-left H remains9.29/6.44/5.06, so the strong
orientation dependence has not disappeared with mean GN–H agreement.

In block1, mean|H−GN|/meanGN is1.6424, while14.0% of the early state's token
directional Hessians were negative and27.7% are negative at500. The midpoint
H/GN means are.99794and.97683 in the two banks; their mean-absolute differences
are1.56386and1.72490times meanGN. Agreement after averaging therefore is not
evidence that the model-curvature contribution is negligible for each token.

For block1's actual ray at |s|=16, the even remainder divided by its true
quadratic is1.00922at10 and1.03152at500. The signed odd/even remainder ratios
are−.12746and−.06616. These are within-state decompositions. Actual write
norms and activation radii differ across checkpoints; these numbers do not
isolate a change in a global Hessian spectrum or in a fixed physical ray.
The complete atlas will show all remaining states rather than extrapolating
these first completed temporal comparisons.

# Independent review of the bounded gauge qualification

2026-09-28. Read the gauge protocol, report, `check.py` and `result.json`.
Checked the transformations algebraically and independently verified the
reported scalar factors using the fixed 2x2 matrices. No checkpoint, model,
GPU, gauge sweep or optimizer trajectory. This is the only new file.

**Verdict:** the toy and report support the intended conceptual closure.
The coordinate-invariance explanation was considered by this observer; it
was not a claim made by the main research program. Nothing here establishes
an empirical failure, new optimizer or performance ordering.

## Transformations and positive controls

For product P=OV and covector K with respect to P, the script correctly
sets `Mv=O^T K` and `Mo=K V^T`. Under `V'=SV`, `O'=OS^-1`, this gives
`Mv'=S^-T Mv`, `Mo'=Mo S^T`. These preserve the dual gradient/displacement
pairings; direct independent calculations give .282 in both charts.

External input covariance C is unchanged, while `Co=V C V^T` transforms
as `Co'=S Co S^T`. The left metric `Bv=O^T O` transforms as
`Bv'=S^-T Bv S^-1`. Bo remains identity because the circuit's output
coordinates do not change. All matrices requiring inverse roots are
positive definite, and S is invertible (determinant 1). The saved script
hash matches the current `check.py` exactly.

The ideal raw half-power O-map positive control is sound: the transformed
whitened momentum differs by a right orthogonal factor, so its pulled-back
direction equals the original. The reported error 5.1e-16 is appropriate.

The full two-sided control is also sound. Under the left covector/metric
change, `Bv'^-1/2 S^-T = Q Bv^-1/2` for an orthogonal Q, and the outer
left root satisfies `Bv'^-1/2 Q = S Bv^-1/2`. Together with the right-side
identity this yields the desired displacement transformation on both
legs. The near-zero pair tangent discrepancy is therefore a valid positive
control, not an accidental use of displacement transforms for covectors.

The pair tangent is `O Dv + Do V`; its sign convention is immaterial to
these relative equivariance errors. It is an infinitesimal product response,
not a finite network update or loss measurement.

## Ray, map magnitude and grafting

The report correctly separates direction from scale. Independent arithmetic
gives the expected mean-normalization multiplier

    sqrt(tr(Co')/tr(Co)) = 1.742556942633008,

matching the reported 1.7425569426330103. It changes the O-map's magnitude
without changing its pulled-back ray. For raw Frobenius grafting, the
expected multiplier is

    ||Do||_F / ||Do S^-1||_F = .5738693385187158,

matching the saved .5738693385187157. Grafting therefore preserves the ray
within each chart while breaking equality of the full mapped displacement
across charts. Relative isotropic damping changes the ray too; its .00468
error is an illustrative value, not a bound or performance metric.

The full two-sided raw map losing pair covariance after separate leg
grafting is consistent with the same argument. No single invariant scalar
can repair the missing left-side orientation of input-only PD in general.

One precision qualification: finite Newton–Schulz by itself should not be
described as necessarily breaking the ideal covariance identity. Its
Frobenius-normalized matrix polynomial is equivariant to the orthogonal
left/right factors induced by ideal transformed roots, in exact arithmetic.
The present report merely lists finite NS among implementation qualifications,
which is acceptable; the measured failures here arise from the explicitly
tested metric/normalization choices, not an untested NS defect.

## Decay and conclusion

Scalar decay commutes with a fixed gauge on both legs, yielding the same
rho-squared product in either chart. The shaped-decay code matches the
project's alpha-quarter, p=2 convention and its trace normalization.
V-side covariance holds in this example because its input metric is fixed;
O-side covariance fails because the metric changes by congruence while the
decay multiplier is applied on the right. The distinction is algebraically
correct.

Consequently the observer's proposed explanation that geometry decay helps
by restoring this covariance is unsupported: ordinary scalar decay already
has it, while shaped decay generally lacks it. That does not challenge the
main program's observed update/decay interaction or prove that invariance
is undesirable. Existing magnitude-matched Muon and hyperball controls
constrain simple scalar remedies, but do not claim that every invariant
normalization has been tested.

No further untouched metric is needed to close this conceptual branch.
The fixed example qualifies the identities and limitations it was designed
to check. A new empirical investigation would require a separate observed
limitation and discriminating comparison, not more coordinate stress.

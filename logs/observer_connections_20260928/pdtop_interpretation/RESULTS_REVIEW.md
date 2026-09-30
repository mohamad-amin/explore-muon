# Independent review: raw projected magnitude and normalized allocation tell different stories

2026-09-28. Read the fixed protocol, numerical-qualification source/results,
all four original preflight JSONs, the profiler's field definitions, and the
root report. Independently recomputed every scalar map record and ratio.
No new map call, model, checkpoint tensor, gradient, GPU, training outcome
or training proposal. Only this review is written.

**Verdict: the arithmetic and the report's qualified interpretation are sound.**
The raw PD-top/full-PD comparison is largely a change in energy allocation,
with much greater full-PD norm outside the measured stiff directions. This
must not be turned into "their normalized suppression is the same": after
one common global norm convention, the large fractional difference is real.
The online 48-matrix normalization still cannot be reconstructed here.

## Field definitions and independent reproduction

The inspected `step_profile_probe.py` computes

    norm2 = v @ v,
    norm = sqrt(norm2),
    proj = ritz @ v,
    energy_top16 = sum(proj[:16]²) / norm2.

Consequently `norm² * energy_top16` reconstructs the numerator used by the
probe, up to saved floating-point rounding. It is a parameter-space squared
projection magnitude, not GN curvature energy and not a fresh verification
of Ritz convergence, exact orthogonality, or the population GN eigenspace.
The curvature-per-norm field correctly uses the independently scored whole-
direction `curvature` divided by norm², not `curvature_curv_set` by accident.

The four states are exactly Muon and PD at 46/83 in the specified 16M runs;
all three methods are retained at every state. All twelve scalar records
and four ratio records reproduce from the original JSONs, with maximum
scalar difference 1.11e−16 from harmless multiplication ordering. Both CSV
exports match their JSON counterparts exactly. All **nine input hashes**
match the recorded protocol/source/preflight manifest. No outcome from the
running PD-top training arms was accessed.

| State | PD-top/full-PD top16 fraction | PD-top/full-PD raw absolute magnitude² | (Full-PD norm / PD-top norm)² |
|---|---:|---:|---:|
| Muon46 | 57.7742 | 1.10189 | 52.4321 |
| Muon83 | 15.0796 | 1.04632 | 14.4121 |
| PD46 | 19.4038 | .87668 | 22.1332 |
| PD83 | 15.0475 | .87013 | 17.2934 |

Each row satisfies the exact scalar accounting

    fraction_ratio = absolute_ratio * squared_norm_ratio.

The absolute magnitudes are within roughly thirteen percent in this small
panel. That is a descriptive same-order statement, not statistical
equivalence, equal projected vectors, equal orientation or equal damping.
The large fractional difference comes predominantly from the different
whole-direction norm in the chosen raw-root convention.

PD-top's comparison with Muon remains positive in the absolute view:
its raw measured top16 magnitude² is 16.86–23.62 times smaller, with norm
only .943–.955 of Muon's. Full PD's corresponding absolute reduction is
17.65–20.71 times. PD-top's whole-direction GN curvature per squared norm
is .101–.189 of Muon's. These same-state scalar observations cannot be
explained solely by inflating PD-top's total norm.

## Do not dismiss the fractional result as a denominator artifact

For each whole-body direction rescaled to the same global norm k, its
reported projected magnitude² is k² times its energy fraction. Therefore
the PD-top/full-PD difference after **equal global norm** is still 15–58×.
Moving more of a fixed budget outside a measured stiff subspace is a real
geometric change; direct removal of the raw projected component is not the
only way to accomplish it.

Conversely, raw absolute magnitudes depend on the root-scale convention.
The ideal matched per-matrix map is invariant to R→cR, while its unnormalized
output and absolute projections are not. The newly exposed raw comparison
is useful accounting under the recorded convention, not a uniquely physical
notion of how much each optimizer suppresses a component.

Only the complement of the **measured top16 Ritz directions** is identified
by these numbers. Do not rename all remaining movement "flat," a specific
input-eigenvalue tail, or useful descent. No such decomposition is retained.

## Algebraic qualification and its scope

The source correctly tests the ideal exact-polar map

    D_R = k P(MR)R / ||P(MR)R||_F.

Positive scalar homogeneity of exact polar gives D_(cR)=D_R. For full-column-
rank square/tall matrices P(MR)^T P(MR)=I, giving the stated normalized
R² Gram. For wide matrices the row-space projector P^T P remains. These
identities are established algebraically; I reviewed the supplied numerical
qualification rather than running additional map calls.

The recorded tiny cases support the intended checks: scale errors below
8e−16, appropriate Gram errors below 3e−15, and a .187 failure of the simplified
tall expression in the wide example. The two-dimensional energies independently
match the exact fractions (.1,1.9) and (20/29,38/29). The diagonal elementwise
minimum used in that illustrative case is valid because its root is diagonal;
the general qualification constructs the clamp spectrally. The toy target
norm omits the common shape multiplier, which can be included in k without
changing the identities. None certifies exact online finite-NS behavior.

## Timing, signs and the unobservable online normalization

Every one of the twelve mapped directions has positive signed held-out
slope and negative c_star. The source already includes the nominal descent
sign. They are uphill in that scored orientation, so a positive a²/(2q)
cannot be read as useful forward descent. This concerns maps of lagged M_s;
it does not establish that the actual next training update, incorporating
the incoming gradient, is uphill.

Online updates also apply separate matrix normalizers, shape factors,
finite NS and cached statistics. If d_i is one matrix's raw direction,
a global Ritz projection after grafting depends on sum_i s_i P_ritz d_i.
The archive retains only the combined projection magnitude and combined
norm, not each matrix's projected vector and their cross terms. Even knowing
all s_i would not recover that sum from the stored scalar alone. Thus the
equal-global-norm interpretation is valid but is not the online map.

I agree with `REPORT.md`: the intervention can determine whether the capped
relative spectral profile suffices for a tested training recipe. A positive
result need not identify its mechanism; a negative result does not uniquely
prove tail signal/noise or falsify every suppression account. The implemented
full covariance/eigendecomposition cost is unchanged. Static map observations
do not measure differential feedback, stability or a training rate.

For wording, "full PD has substantially more raw energy outside the subspace"
is safer than implying that it adds an independent tail vector to PD-top.
The report already correctly warns against equal-vector and online-map
interpretations. Close this scalar/algebra clarification without another
map reconstruction, new state, outcome-dependent normalization or sweep.

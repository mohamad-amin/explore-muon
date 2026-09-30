# Independent review: momentum-window mechanism and retained map contrasts

2026-09-28. Read-only review of the main protocol's 15:43 CDT entry, current
guide/state, observer STATE/SYNTHESIS, actual probe and optimizer/checkpoint
sources, all 18 retained 16M profiles for Muon/PD/SOAP-PD at momentum .9/.8
and steps 9/46/83, and the completed auxiliary attribution. No model call,
GPU, scheduler action, checkpoint mutation or main-study edit. This file is
the only new artifact from this review. Scalar reductions used Python's
standard library.

## Judgment

There is stronger evidence for a spatial map effect than the main entry's
own-state table alone: **the saved profiles already contain polar and PD
maps of the same saved momentum at the same model state, scored in the same
curvature geometry.** Their difference cannot be explained solely by comparing
different optimizers' co-adapted states. The PD map suppresses the measured
stiff subspace on Muon's states too.

That establishes a conditional map property, not the causal chain
"shorter momentum leaks oscillation, the online whitening map removes it,
and this removal causes faster training." The retained momentum and actual
next step are time-misaligned, the counterfactual map differs from the online
map, and most counterfactual directions point uphill. The best next action is
to report the existing same-state contrasts with these qualifications; they
do not justify a new training/filter intervention.

## What the instrument actually measures

The relevant source is
`logs/muon_spectra/second_order_audit_20260926/step_profile_probe.py`, reviewed
SHA256 `a2786ce23d23fee1ef1dee07d94fbdfbc398a3f663c78af9f038c45108c632a2`.

1. **Time indexing.** Lines 59–73 load `step_s.pt` and the next weights,
   then compare the saved buffer `M_s` with `W_(s+1)-W_s`. The frozen trainer
   calls `optimizer.step()` before saving `step_s.pt`; frozen
   `distributed.py:433–446` saves the subsequent file as weights only.
   Frozen `muon.py:664` implements `M <- beta*M + clipped_gradient`.
   For these non-Nesterov, unprefiltered runs, `M_s` drove the preceding
   `W_(s-1)->W_s` update. The measured next displacement instead uses
   `M_(s+1)=beta*M_s+g_s^clip`, with the fresh training batch at `W_s`.
   Thus energy ratios between `momentum` and `actual` are not measured
   transmission coefficients of one optimizer application.

2. **Three distinct maps.** Probe lines 128–133 define `actual` as the
   retained body displacement including decay/write rounding, `muon` as
   `-polar(M_s)`, and `pd` as `-polar(M_s R) R`. The polar is a direct FP32
   SVD. R is a fresh full input-second-moment root with alpha 1/2 and
   damping .001, estimated from the 128 curvature sequences, excluding
   position zero (lines 89–103). It is not the run's cached owner-local
   EMA root. There is no SOAP counterfactual in these files, even on
   SOAP-PD's states.

3. **Normalization changes the interpretation.** Neither counterfactual
   receives the online aspect factor or PD's per-matrix RMS matching.
   The actual code applies both (frozen `muon.py:704,747–767`). The probe's
   PD maps have much larger raw global norms than its polar maps. Fractions
   and Rayleigh quotients remove a global multiplier but do not correct
   the different allocation between matrices. They identify the complete
   unnormalized frozen map's geometry, not a faithful replay of the online
   PD operator. The JSON has insufficient per-matrix information to apply
   the online norm rule retrospectively.

4. **Which geometry and samples.** The GN is restricted to 48 body matrices.
   A 48-step random-start Lanczos run produces the top 16 Ritz vectors on
   the curvature validation bank. Slope and `curvature` use a distinct
   128-sequence training-stream bank beyond the 1x training budget.
   `curvature_curv_set` uses the bank that estimated R and the Ritz frame.
   The independent-bank quadratic therefore provides a useful check
   against selecting a favorable measurement geometry. Ritz residuals or
   convergence checks are not retained; call the projected quantities
   top-Ritz-subspace estimates, not exact spectral projectors.

5. **The reported next gradient is body-only.** Probe lines 158–165 add only
   the hidden-weight displacement to the base model, then evaluate a new
   body gradient. Head, embeddings and norm gains remain at step s. The
   gradient-pair record is a body-only counterfactual on a common held-out
   bank, not the actual full `W_(s+1)` gradient. Its sign flips are evidence
   of local reversal under that displacement; one pair is not a frequency
   spectrum or by itself proof of a sustained period-two mode. The existing
   `body_aux/REPORT.md` found that auxiliary scope was not the main cause
   of a gradient-cosine sign difference at Muon4M183; sample pairing was.
   Do not turn this source finding into the opposite untested assertion.

## The smallest saved-artifact audit already answers one premise

Include exactly the nine original and nine new Muon/PD/SOAP-PD profiles at
16M, steps 9/46/83. For each state compare `muon` and `pd`, never subtract
eigenvectors across states. Let `e=energy_top16`, `n=norm`, and `q=curvature`
on the independent scoring bank. The directly recoverable quantities are:

- matched-global-norm stiff ratio: `e_pd/e_muon`;
- raw projected squared-norm ratio: `(n_pd/n_muon)^2 * e_pd/e_muon`;
- matched-global-norm independent curvature ratio:
  `(q_pd/n_pd^2)/(q_muon/n_muon^2)`;
- signed slope, signed `c_star`, and positive-step feasible model decrease:
  `max(0,-slope)^2/(2*q)`.

Across all 18 existing profiles:

| PD / polar ratio | Minimum | Median | Maximum |
|---|---:|---:|---:|
| Top-16 energy fraction | .000189 | .001774 | .006907 |
| Raw top-16 squared norm | .02466 | .04315 | .09693 |
| Independent-bank curvature per squared norm | .000249 | .006655 | .01884 |

The energy-fraction contrast is roughly 145–5,300-fold. Its direction is not
just a large denominator: raw projected squared norm also falls, by about
10–41-fold. Independent-bank curvature per squared norm falls at every
state. These are deterministic summaries across correlated measurements,
not 18 independent replications or a statistical population claim.

For example, on **Muon's own step-46 state at beta .8**, the polar/PD maps
have top-16 fractions `.0016015 / .0000015279`, norm `156.78 / 1155.14`, and
independent curvatures `297.02 / 42.67`. The fixed-state PD map lowers the
stiff projected fraction by about 1,048-fold and its absolute projected
squared norm by 19.3-fold. At beta .9 the analogous energy-fraction ratio
is .001127. The spatial map property exists without training under PD.

**But preserve the signs:** 17/18 polar-map slopes and 16/18 PD-map slopes
are positive, whereas all 18 actual next-body-step slopes are negative.
At Muon beta .8 step 46 the slopes are `+3.904` and `+.647`, respectively.
The displayed `quality=a^2/(2q)` assigns positive quality even to these
uphill directions because it allows a negative step coefficient. The
positive-step feasible decrease is zero for those rows. This is a concrete
reason not to equate low stiff energy/curvature with a productive step.
The lagged momentum's phase relative to the current gradient matters.

The same reductions should retain top-1/top-4 views and both curvature
banks, without requiring all three prefixes to decrease on every state:
PD beta .9 step 9 has an extremely small polar top-1 projection and its
PD/polar top-1 fraction ratio is about 1.49, while top-4/top-16 both fall
strongly. Do not select a favorable k after inspecting the answer.

## Which part of the 15:43 interpretation needs narrowing

- "More oscillation reaches every momentum" does not hold literally in
  the table. PD's top-16 momentum fraction at step 46 is .3415 -> .3381,
  and at step 83 .3353 -> .3320. It rises early, and rises at all three
  sampled Muon and SOAP-PD states. Nor does a static energy fraction
  distinguish oscillation amplitude from changes in persistent signal.
- "Only Muon's map passes it on" is too categorical. PD's actual stiff
  fraction rises by 1.27x, 1.79x and 3.52x; Muon's by 1.20x, 3.03x and
  2.67x. PD remains much lower in absolute fraction. The robust contrast
  is a much smaller level, not uniquely zero transmission of the change.
- "Muon has only temporal filtering" is also too strong. Polar mapping
  itself dramatically redistributes energy relative to raw momentum:
  at Muon beta .8 step 46 the fractions go .3428 -> .0016015. Muon lacks
  the additional activation-covariance-dependent map; it is not a spatial
  identity map in the GN eigenbasis.
- In a fixed-input linear EMA, the alternating/DC gain ratio is
  `(1-beta)/(1+beta)`: .9 -> .8 raises it 2.11x in amplitude (4.46x in
  squared amplitude). This predicts a tendency under fixed forcing, not
  the observed ratios in changing model states, clipped gradient streams
  and differently adapted curvature. Use it as a testable algebraic
  premise rather than as an inference from the own-state energy table.

A supportable wording is: *The saved-state PD map has much lower stiff
projection and curvature per norm than polar mapping of the same lagged
momentum, even at Muon's states. Actual whitening-method steps likewise
remain weakly coupled to each state's stiff directions at shorter momentum.
This is consistent with spatial geometry relaxing a temporal-averaging
tradeoff, but the time-matched online transmission and its contribution to
the training-rate gain have not been identified.*

## Missing information and stopping boundary

These JSONs retain scalar metrics, Ritz values and gradient projections,
not the Ritz vectors, directions, per-matrix contributions or fresh roots.
The next weights file does not retain `M_(s+1)` or its incoming training
gradient. The optimizer's owner-local SOAP states/root caches are outside
the ordinary parameter state dictionary; the trainer saves rank-zero input
statistics separately, which do not reconstruct every owner's map. A
matched online input/output replay therefore cannot be manufactured from
these summaries by renaming `M_s` as the next momentum.

Do not open a replay/transport repair cascade to obtain that missing causal
claim. First complete the compact JSON audit above, with source hashes and
scope labels. It answers whether spatial suppression is solely an artifact
of comparing different learned states: it is not, for this frozen map. It
does not answer whether filtering oscillatory versus useful parts causes
the shorter-window training gain. An eventual stronger design would need
one common state, one common fresh gradient/history, time-matched momentum
inputs, faithful per-matrix normalization, and both signed descent and
functional response. None of that is authorized or executed by this review.

## Relation to the completed auxiliary attribution

The meaningful connection is that **shared predictive movement can be
useful despite positive curvature or finite-loss interaction**. It is not
evidence that the auxiliary result identifies the same stiff oscillation.

At Muon4M183, `aux_partition/run1/validation.json` attributes 98.4% of the
discovery and 96.2% of the fresh body–auxiliary interaction to ordinary
finite-logit overlap, with body/embedding and body/head pair shares about
65–66% and 34–35%. Post-hoc projection of an auxiliary finite response off
the body's response leaves only 12.1% / 7.5% of embeddings' label-linear
descent, though 57.8% / 55.5% of its predictive metric energy remains.
The corresponding head fractions are 36.2% / 30.4% descent and 85.1% /
86.5% energy. Thus suppressing the overlapping response would remove much
of the useful prediction change; it is not a free redundancy removal.

Those are finite-logit response-span calculations at one 4M state; the
step profiles measure body-only infinitesimal GN geometry at 16M. The
auxiliary four-direction Gram does not locate its overlap in the body's
top-16 GN eigenspace and contains no temporal spectrum. The profile JSONs
contain no auxiliary projections. The missing cross-object measurements
cannot be inferred from similarly positive quadratic terms. Preserve the
connection as a constraint on interpretation: any proposed spatial filter
must retain useful signed descent, not merely reduce overlap or curvature.
No staged, head-LR, centering or additional model probe follows from it.

## Addendum: output suppression is not the feedback derivative

The parent reviewer identified a sharper causal gap than co-adaptation alone.
The saved fractions concern the value of `phi(M_s)`. Local momentum feedback
depends instead on the derivative `A = D phi(M)` composed with the derivative
of the incoming gradient. Tiny projection of one output onto a stiff vector
does not bound sensitivity to a perturbation in that vector.

For a differentiable degree-zero map, positive scaling leaves the output
unchanged, so Euler's identity gives `D phi(M)[M] = 0`. Exact polar and a
frozen-root PD sandwich have this property. It says that the radial
perturbation is invisible; it leaves transverse sensitivities unrestricted.
Weak singular values can make those sensitivities large.

An exact 3x3 counterexample, refined in discussion with the parent reviewer,
also holds the momentum norm fixed. For `0 < epsilon < 1/sqrt(2)`, let

    M = diag(sqrt(1-2 epsilon^2), epsilon, epsilon),   ||M||_F = 1
    K = (e_23 - e_32) / sqrt(2),                      ||K||_F = 1.

The polar output is I and has zero projection on K at every epsilon, while

    D polar(M)[K] = K / epsilon.

For a GN model `G[D] = lambda <K,D> K`, the static output has exactly zero
stiff energy, but the differential gain along K is `lambda/epsilon` and
can be arbitrarily large. Conversely, radial perturbations have zero gain.
This is a local mathematical demonstration, not a model of the measured
network or proof that whitening fails to control feedback. For a frozen
root `R=diag(r1,r,r)`, the PD-sandwich derivative in this direction is
`(r/epsilon)K`. If it is subsequently norm-matched to sqrt(3), the derivative
is `sqrt(3) r/(||R||_F epsilon) K`: the norm derivative's radial correction
vanishes here because the diagonal output is orthogonal to K. The divergence
therefore survives fixed momentum norm and fixed output norm. This is useful
because it rules out a specific inference from the archive while leaving
its strong same-state suppression intact.

For the code's unnormalized momentum, write

    M_plus = beta M + g(w)
    w_plus = w - eta phi(M_plus).

At one operating point, with `H = Dg(w)` and `A = Dphi(M_plus)`, the local
Jacobian on perturbations ordered as `(delta_w, delta_M)` is

    [[I - eta A H,  -eta beta A],
     [      H,          beta I]].

The equivalent EMA-normalized parameterization `m=(1-beta)M` gives
`m_plus=beta m+(1-beta)g(w)`. With `A_m=Dphi(m_plus)`, its block matrix is
`[[I-eta(1-beta)A_m H,-eta beta A_m],[(1-beta)H,beta I]]`.

This identity is exact for that simplified smooth update. With a frozen
positive-semidefinite H and A, the nonzero coupled modes reduce to

    z^2 - (1+beta-eta kappa) z + beta = 0,

where kappa is an eigenvalue of `A^(1/2) H A^(1/2)` in the unnormalized
parameterization. The stable interval for an isolated positive mode is
`0 < eta kappa < 2(1+beta)` (equivalently
`0 < eta(1-beta) kappa_m < 2(1+beta)` in the EMA parameterization). This
derives why **differential** spatial gain can trade against temporal
averaging in a frozen approximation. It does not establish that the
measured static energy ratios are estimates of kappa.

The qualifications here are substantive, not optional technical details:

- The gradient derivative is the true loss Hessian for ordinary gradients.
  Substituting GN is a positive-semidefinite approximation. During active
  clipping it is the clipping Jacobian composed with the Hessian, including
  dependence on the global norm; the simple PSD argument need not hold.
- Exact polar is the gradient of the nuclear norm and therefore has a PSD
  derivative at full-rank smooth points. For fixed symmetric R, the
  unnormalized PD sandwich `polar(M R) R` is the gradient of `||M R||_*`
  and shares that property. Online PD then rescales by its M-dependent
  per-matrix Frobenius norm. That additional derivative generally destroys
  self-adjointness/PSD; it must not be silently omitted. SOAP's evolving
  bases/denominators and changing input roots add further state variables.
- The code stores unnormalized momentum. For fixed beta, `m=(1-beta)M`
  preserves the degree-zero output but rescales its derivative:
  `Dphi(m)=Dphi(M)/(1-beta)`. The explicit `(1-beta)` factor is not an
  automatic stability improvement unless the reference operating point
  and normalization convention are held consistently.
- A nonzero normalized update generally has no ordinary stationary point
  with nonzero M, while exact polar is nonsmooth at M=0. Treat this as a
  locally frozen Jacobian along the actual trajectory, or as one factor
  in a periodic-orbit Jacobian product. A constant-coefficient stability
  threshold is not a claim about the nonstationary training system.

**Recommendation.** Include this small analytic counterexample and the
Jacobian distinction as an interpretation note alongside the scalar audit.
They are sound, inexpensive and resolve a real category error, but are not
new empirical evidence or a separate research branch. A small finite-
difference check of the analytic derivative/block is useful qualification;
retain its numerical errors. Do not launch a toy parameter sweep that
merely rediscovers the identity. The eventual causal
question is whether whitening reduces the relevant feedback derivative,
retains useful response, and changes the allowable momentum tradeoff on its
own trajectory. The existing static maps establish only the first map-value
premise; they do not yet measure that derivative.

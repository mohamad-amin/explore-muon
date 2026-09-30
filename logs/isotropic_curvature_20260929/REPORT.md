# Curvature along retained Muon and PD trajectories

**Complete measurements and numerical verification, 2026-09-29.** All 18
matrix panels and six joint states finished. The strongest result is that
curvature along actual updates remains strongly dependent on orientation,
even where the finite loss remainder is nearly quadratic. The known MLP
Jacobian accounts for a substantial factor in the GN contrast, and the
remaining contrast changes with training.

This is the bounded empirical follow-up to the user's request to examine
[Su's isotropic-curvature model](https://arxiv.org/pdf/2511.00674). It provides
a completed measurement atlas and an explanation of its supported findings.
The previous observer goal remains closed; no new optimizer was trained.

Start with the [trajectory overview](run1/report_tables/trajectory_overview.png),
[finite-remainder overview](run1/report_tables/finite_remainders_overview.png),
and [complete 18-panel table](run1/report_tables/TABLES.md).

## Measured trajectory results

The following numbers use four-context means for each individual panel;
ranges span the two recipes and three depths. H and GN below denote the
directional quadratic forms along the actual saved write, not matrix division.

| Checkpoint | H / GN | Actual H / mean H of two left rotations | Mean token abs(H−GN) / mean GN |
|---|---:|---:|---:|
| 10 | 6.132–10.658 | 4.163–20.141 | 5.856–11.379 |
| 500 | 0.857–0.996 | 4.758–9.289 | 0.766–1.995 |
| 1300 | 0.900–0.993 | 4.938–8.988 | 0.495–1.960 |

**Orientation dependence persists across the sampled trajectory.** Actual
writes have substantially greater curvature than left rotations with exactly
the same per-position activation norms. This holds at the final block too,
where the suffix is token-local. A common cost depending only on the up-layer
activation-kick norm omits material structure in this conditional comparison.
It is not enough to repair only the input's spherical-distribution assumption:
the left control does not require that assumption in the first place.

**The relationship between true Hessian and GN changes strongly.** Early
true curvature is 6–11 times GN; middle and late mean curvature is much closer
to it. But the large absolute token differences show that this later agreement
conceals cancellation. At the late checkpoints, 13.4–32.0% of token directional
Hessians are negative even though all aggregate curvature rows are positive.
These are descriptive dependent-token distributions. Mean agreement cannot
establish that the logits are locally linear in the parameter direction.

**Finite growth and orientation dependence are different findings.** At the
actual single-matrix write (s=1), R/Q is 0.99063–1.00242; at s=−1 it is
0.99746–1.00948. At |s|=16, the even part R_even/Q ranges from 0.98372 to
1.03152 across all 18 actual-write panels. Thus the sign-averaged response
remains nearly quadratic over this range despite the large orientation gap.
The corresponding activation RMS radii at 16× range from 0.695 to 5.341.
There is nevertheless asymmetry: R_odd/R_even at 16× ranges from −0.14883 to
0.03773. The largest case is early PD block 1. A symmetric radial fit would
hide that component. The controls also matter: the largest even departure
among raw-right rotations is +18.3%, so the actual-write result must not be
generalized to every direction. No universal growth exponent or absence of
a more distant nonlinear regime follows.

**The three selected layers interact.** Joint Hessian cost divided by the
sum of individual costs is:

| Method | Update 10 | Update 500 | Update 1300 |
|---|---:|---:|---:|
| Muon | 1.697 | 1.477 | 1.395 |
| PD | 1.572 | 1.714 | 1.620 |

At the actual joint step, the measured interaction is 98.1–99.8% of the
prediction from the Hessian cross terms. At ±16×, interaction divided by that
prediction ranges from 0.745 to 1.362, with the strongest asymmetry early.
These are genuine cross terms of the measured three-up perturbation; they
do not establish that coupling is the cause of an optimizer's training gain.

Curvature by itself is not direction quality. For example, early Muon block 1
has first-order slope −0.002630 along its actual write, versus −0.0000206 and
−0.0000255 for the rotations. Removing the curvature also removes most of the
descent alignment in this example. The experiment constrains an explanatory
model; it does not prescribe the rotated directions as an optimizer.

## Where the orientation contrast comes from

The tensor-only transmission calculation passed its derivative, tensor-hash,
radius-reproduction and pooling checks. Across all 36 actual-versus-left
comparisons, GN per unit preactivation energy is 4.925–11.536 times greater
for the actual write. Passing each perturbation through the checkpoint's
GELU/down-projection Jacobian contributes a transmission contrast of
1.726–5.627. After normalizing by that residual-tangent energy, the remaining
GN contrast is 1.072–3.692. The transmission factor exceeds one and the
remaining contrast exceeds one in both separate score banks as well.

The final block shows a clear change in this decomposition. At update 10,
its remaining contrast is only 1.07–1.09 for Muon and 1.16 for PD: much of
the GN difference in up-preactivation coordinates corresponds to the known
local projection. At update 1300, the remaining factors are 3.33–3.42 and
3.13–3.28, while the local transmission factors have fallen to about 1.8.
Thus the total contrast does not have a single fixed source throughout
training. [Transmission figure](run1/transmission_analysis/transmission_contrasts.png).

This is a descriptive decomposition under a specified architectural metric,
not a percentage of causal optimizer benefit. Residual-normalized differences
can include output orientation, distributions of tangent energy over positions,
and their association with downstream curvature. The normalization does not
match every token's residual norm. It also does not explain all of H−GN or
the finite-radius response.

## Input geometry and the right-rotation controls

The spherical prediction for displacement moments depends on both the update
spectrum and the input coordinates. For PD's actual writes, observed angular
squared-displacement relative variance is 13.3–127.0 times the raw spherical
prediction. In the regularized whitened frame that ratio is 0.34–11.65; the
observed-to-spherical mean also moves closer to one in every PD panel.
Whitening substantially changes this comparison, but does not make the
measured inputs spherical. For example, late PD block 1 still has 5.93 times
the predicted relative variance in the whitened angular coordinates.
These are finite-context moment comparisons, not population tests.

Muon supplies an important qualification. Its actual raw write's effective
rank rises from 170–489 early to about 510–511 later, out of a maximum 512.
At the late states, raw angular squared-kick relative variance is only about
0.000015–0.000348. A nearly flat tall matrix can make displacement norms
nearly constant even for strongly non-spherical inputs. Small norm dispersion
alone therefore cannot validate the input-law assumption.
[Moment figure](run1/geometry_analysis/norm_dispersion.png),
[gradient/momentum/write spectra](run1/geometry_analysis/gradient_momentum_write_spectra.pdf).

The two right rotations also demonstrate why actual radius must be retained.
At late PD block 1, the raw rotation has 10.87 times the preactivation energy
and 7.55 times the GN of the whitened rotation; their GN-per-residual-energy
ratio is only 0.94. A smaller unnormalized cost is not by itself stronger
evidence for isotropy. Across all 18 panels, raw-right versus white-right
GN-per-preactivation-energy ratios are 0.555–0.958; residual-energy-normalized
ratios range from 0.604 to 1.237. The controls preserve different metrics and
are not parameter-norm matched after mapping back.
[Every comparison and both banks](run1/report_tables/raw_vs_white.csv).

The 5-dimensional parameter spans remain full rank after Gram
orthonormalization and have positive restricted Hessian eigenvalues in all
18 four-context averages. This does not imply a positive full Hessian:
unmeasured directions and the negative individual-token curvature remain.
[Restricted curvature](run1/geometry_analysis/restricted_curvature.json).

## What this changes in the explanatory model

The measurements support separating input displacement statistics, the
architecture's transmission of those displacements, residual downstream
curvature, and finite-radius asymmetry. A universal scalar cost of the raw
up-preactivation norm merges these distinct, changing quantities. Input
whitening alone cannot resolve the exact left-rotation discrepancy, and a
known local output metric removes only part of the GN discrepancy.

An elliptical-input extension of the paper still gives a principled connection
to ideal half-power PD: with P=Q C^(1/2), the gradient in those coordinates is
G C^(−1/2), and its polar case gives

    Q ∝ polar(G C^(−1/2)) C^(−1/2).

This is an algebraic connection under the model's assumptions, not a derivation
of the practical quarter-power recipe, damping, momentum or schedules.
The atlas now shows which additional conditional structure an improved model
would need to explain. It does not identify an optimal optimizer or establish
that a more elaborate curvature model improves training.

## Completion, checks, and usable artifacts

The producer completed in 6,024.69 seconds (100.4 minutes), within the fixed
three-hour bound. Main reductions/figures took 181.4 seconds, geometry 14.4
seconds, and transmission 18.0 seconds. All three execution sessions exited
successfully. No partial panel replaced a missing result.

The independent NumPy-only checker recomputed all 18 panels and six joint
states from raw arrays and compared all five main CSV tables. It passed.
Maximum left-radius discrepancy was 2.78e−16; projected-Hessian diagonal versus
separate directional AD differed by at most 5.96e−19. The joint GN matrix sum
agreed with its independently computed directional GN to 1.26e−17.
Transmission reproduced archived squared radii to 5.00e−16 and its pooled
factorization to 4.34e−19. Its analytic GELU derivative agreed with autograd
to 5.11e−15. Numerical tolerances and all input hashes are retained in the
[verification record](run1/verification.json) and
[transmission record](run1/transmission_analysis/result.json).

- [Compact data tables](run1/report_tables/TABLES.md) and adjacent CSVs include
  all panels and both score banks.
- [Full reduction inventory](run1/analysis/README.md) links the 9,450 curve
  rows, 630 curvature rows, 3,528 projected-curvature entries, 630 joint rows
  and 4,410 paired-radius rows, including per-context readouts.
- [Signed profiles](run1/analysis/figures/signed_radius_profiles_all_contexts.pdf)
  retain all directions, contexts, signs and nonzero radii.
- [Joint positive profiles](run1/analysis/figures/joint_interaction_positive.png)
  and [negative profiles](run1/analysis/figures/joint_interaction_negative.png)
  show the complete selected-layer interaction.
- `run1/*/block*_up/tensors.pt` and `per_token.npz` retain the measurement
  tensors; executed source and protocol copies are in `run1/`.
- [Interpretation notes](INTERPRETATION.md), [design review](DESIGN_REVIEW.md),
  [post-fix transmission source review](TRANSMISSION_POSTFIX_REVIEW.md), and
  [final independent interpretation review](RESULT_REVIEW.md) preserve the
  reasoning and review boundaries.

## The question the measurements answer

Does the size of an activation perturbation determine its loss cost along
trained updates, and how does that relationship change with training?
The paper motivates a scalar remainder cost and, under additional assumptions,
spectral homogenization. Our experiment examines its application to retained
updates; it is not a reproduction of the paper's random-direction experiment
or a test of its conditional mathematical theorems.

For a saved matrix write D at a fixed state W, the measured remainder is

    R(s) = L(W+sD) − L(W) − s <gradient L(W), D>.

The true Hessian predicts Q(s)=s² DᵀHD/2. Predictive Gauss–Newton (GN) uses
the first derivative of the logits and the curvature of cross-entropy. The
difference H−GN contains the logits' second derivatives: it is still part
of the second-order loss term. R−Q is a separate departure beyond the true
local quadratic. The report keeps these quantities distinct.

Left rotation is the strongest radius control: replacing D by O D preserves
the norm of every activation kick D x. Any difference in the linear-subtracted
loss cost therefore cannot come from a changed kick-radius distribution.
Positive and negative scales add a separate test: a norm-only cost is even,
whereas R_even=(R(s)+R(−s))/2 and R_odd=(R(s)−R(−s))/2 expose its measured
symmetric and asymmetric components.

## Fixed coverage and what is retained

The atlas covers Muon and PD at saved updates 10, 500 and 1300, with MLP-up
matrices in blocks 1, 4 and 8: 18 panels. Each contains the actual next saved
write, two left rotations, a raw right rotation and a right rotation in a
regularized input-whitened frame. Fifteen signed scales span −16 to +16 times
the recorded write, including zero. The same four fresh 512-token contexts
are used everywhere, divided into two banks. Eight separate fresh contexts
calibrate the input covariance. All positions and both banks are retained.

Each panel includes per-token losses, exact directional slopes, true-Hessian
and GN terms, the full 5×5 Hessian/GN in its chosen direction span, and
parameter/input-metric Gram matrices. Six joint panels apply the three actual
up writes together and retain the full 3×3 curvature and finite interactions.
Later perturbed layers always use their current inputs.

The archives also retain the complete selected gradient matrices, lagged
momentum, actual saved writes, their singular values, input covariances and
factors, calibration and score activations, output adjoints, and source and
checkpoint hashes. The output adjoints at earlier sites include downstream
token dependencies. They are derivatives of the summed sequence loss, not
isolated derivatives of a token's own loss.

The smooth diagnostic model runs in FP64 at the saved FP32 weights. It uses
a separate source copy with explicit attention and without hidden FP32
normalization/loss casts. Original training sources and results are preserved.
Qualification compared the original and diagnostic forward functions, finite
differences and AD, cached suffixes and full forwards, and joint functional
perturbations against direct parameter substitution. Details are in
[qualification/result.json](qualification/result.json) and [PROTOCOL.md](PROTOCOL.md).

## Architectural transmission and spherical-input comparisons

A retrospective, independently reviewed tensor-only calculation asks whether
the known MLP geometry explains some output-orientation dependence. For up
preactivation Z and its tangent dz=D x, the immediate residual-branch tangent is

    dr = W_down [gelu'(Z) ⊙ dz].

This induces an input-dependent metric in the 2048-dimensional up-output
space, of rank at most 512. We compare the exact pooled decomposition

    GN / E||dz||² = (E||dr||² / E||dz||²) × (GN / E||dr||²).

It introduces no fitted metric or extra network evaluation. It describes
infinitesimal GN, not every part of the true Hessian or a finite-step response.
At earlier blocks, later attention mixes positions; the energy and curvature
comparison is pooled over whole contexts rather than presented as a causal
tokenwise decomposition. Both banks and every direction are retained.

The spherical-input calculation compares measured angular displacement
moments with an exact identity. For a unit-spherical input z, A=QᵀQ and
r_eff=tr(A)²/tr(A²),

    E||Qz||² = tr(A)/n,
    Var(||Qz||²) / E(||Qz||²)² = 2/(n+2) × (n/r_eff − 1).

Flattening a tall matrix's spectrum reduces displacement variation at fixed
Frobenius norm. Turning that into an optimization benefit additionally needs
the appropriate curvature cost and first-order alignment. Whitening an
uncentered second moment alone does not produce spherical directions or
control fourth moments. [MODEL_CONNECTION.md](MODEL_CONNECTION.md) records
the derivation and qualifications.

## Boundaries of the conclusions

- These are true Hessians in selected finite direction spans, not the full
  million-dimensional matrix Hessians. Restricted eigenvalues use the
  parameter Gram to orthonormalize the span; raw coefficient eigenvalues
  are not called parameter-Hessian eigenvalues.
- The saved write and its control directions change between checkpoints.
  Their curvature evolution combines changes in the state and in the
  directions. Three snapshots do not establish a continuous spectral law.
- Four contexts are four contexts, not 2048 independent replications. The
  two signed-permutation rotations are a small conditional orientation panel.
- Block 8 has a token-local suffix. Blocks 1 and 4 feed later attention and
  require context-level interpretation. Joint results concern only three
  selected up matrices, not the complete training update.
- The recipes share initialization and data, but use different optimizer/LR
  recipes and training hardware. Own-state comparisons do not isolate an
  optimizer's causal effect. PD here uses quarter-power whitening; the
  ideal half-power connection to an elliptical version of the paper's model
  does not make these runs an exact realization of that theory.
- The paper studies an average/distributional approximation. Its experimental
  activation kicks are individually normalized; ours arise from shared weight
  rays selected by training and retain their natural token radii. Conditional
  discrepancies are not a general falsification of its model.
- Lower curvature alone is not a better direction: rotation also changes
  first-order descent. This task does not fit an exponent, select an optimizer,
  infer a training-rate gain, or extend the radius grid after inspecting it.

All new work remains in this directory. The numerical producer uses two CPU
threads and the reductions one; no training or GPU job is launched by this
study. The sealed surrogate study and main research sources remain untouched.

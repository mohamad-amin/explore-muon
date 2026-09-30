# Longitudinal finite-curvature atlas motivated by Su (2511.00674)

2026-09-29. The user authorized proceeding after the paper discussion.
This is a new bounded measurement task, not a resumption of the cleared
open-ended observer goal. All new artifacts remain in this directory.
Root read RESEARCH_GUIDE.md and RESEARCH_STATE.md and verified main allocations
2567578/2618555 live at07:59UTC. CPU only, at most two numerical threads;
no training, GPU use, scheduler mutation, or sealed-surrogate data.

## Decision argument and scientific scope

The paper's scalar finite-activation remainder connects matrix update spectra
to nonlinear loss growth. Our prior archive contains local GN measurements
and selected finite steps, but no located panel jointly comparing orientation,
radius, and training time with true Hessian and GN distinguished. A curvature
spectrum alone cannot establish that nonlinear loss cost is radial. Conversely,
failure of exact isotropy at a selected state is not a proof that a distributional
model is useless. The objective here is an exploratory trajectory atlas, not
another local-usefulness gate or a new optimizer claim.

The independently reviewed design is in DESIGN_REVIEW.md; source/coverage in
INVENTORY.md. Both preceded any scientific model measurement. We adopt the
reviewer's distinct left/right rotation controls, full-context scoring and
the current-input rule for joint parameter perturbations. We extend the signed
radius grid before scoring, so it covers beyond the immediate actual step.

## Fixed panel

Original completed1M-token-batch trajectories under
logs/muon_spectra/soaudit_traj_20260926:

- Muon: M_lr0.007_s260925_l40s.
- PD: PD_a0.25_lr0.01_s260925_ada.

Use kept states10/500/1300 and actual next weights11/501/1301. Study MLP-up
matrices in blocks1/4/8 (zero-based0/3/7), giving18panels. All methods, times,
depths and directions remain in output. Initial state/data/source pairing is
verified; hardware and optimizer/LR recipes differ and remain explicit.
This is a comparison of each recipe's own states, not isolated optimizer causality.
The architecture, context512 and complete next-token loss are retained.

Block8 is the clean token-local suffix after its up matrix. Earlier blocks
feed later attention: their per-token losses can depend on perturbations at
many positions. Retain per-token AND whole-sequence displacement/remainder
quantities and do not identify a token's input kick with its entire loss response.

## Data and coordinates

Fresh calibration inputs: eight sequences starting at training-stream offset
2,900,262,144. Score inputs: two sequences at each of2,910,262,144 and
2,920,262,144. All are beyond the selected models' maximum training exposure,
within the existing3.2B-token source, mutually disjoint, and fixed before scoring.
Manifest/range checks precede loading. These are four contexts, not2048
independent experimental replicates. Same inputs are used at every state.

For each panel, collect all512positions of the normalized input x on the
eight calibration sequences; form uncentered C=mean(xx^T), including position0.
Save full C, its spectrum, mean, calibration activations, and original norms.
Define C_reg=C+1e-3*mean_eigenvalue(C)*I and S=C_reg^(1/2). This is a regularized
coordinate transform, not the historical online root and not proof of spherical
inputs. Save whitening residuals and score-bank displacement distributions.

Let D be the FP64 difference between the saved next and current up weights.
The five directions are:

1. actual D;
2–3. two independently seeded left signed-permutation rotations O_L D;
4. one right signed-permutation rotation D O_R^T;
5. D S O_R^T S^-1, using the SAME right rotation as4.

Seeds/operators are fixed by block, independent of method/time/outcome, and
archived. Left rotations preserve every individual ||Dx|| exactly and test
output-orientation dependence without changing the radius distribution.
Raw right rotation preserves weight-space singular values. Whitened right
rotation preserves singular values of D S, with regularization qualifications.
Do not Frobenius-match direction5 after transforming back; that would change
the comparison. Signed permutations are a bounded orientation panel, not a
Haar population test. Save the actual matrices and invariant checks.

## Numerical function and measurements

Use a separate copied FP64 diagnostic model. Remove the original RMSNorm and
loss FP32 casts; use explicit causal attention for higher derivatives. No
training source is edited. This measures the smooth FP64 loss at saved FP32
weights, not BF16/FP32 update rounding. Compare the diagnostic baseline with
the unchanged CPU FP32 model, and verify cached-prefix/suffix outputs against
the full FP64 forward before scientific reduction.

For signed multipliers
{-16,-8,-4,-2,-1,-.5,-.25,0,.25,.5,1,2,4,8,16}, save every token loss and
R(s,D)=L(W+sD)-L(W)-s<g,D>. Positive and negative sides remain separate.
Negative remainders/curvatures are valid observations; never force a convex
radial fit. Record actual activation displacements and relative-to-base norms
so coverage is expressed in physical as well as actual-write units. A range
without a nonlinear takeoff is reported as such, without chasing a knee.

At zero, compute exact per-token directional slopes and true Hessian quadratic
terms through automatic differentiation. Separately compute predictive GN
from logit JVPs, retaining their difference (the model's second-derivative term).
If an aggregate projected Hessian is computed, label its finite direction span
explicitly; it is not the full million-dimensional parameter Hessian. Preserve
full selected gradients and stored momentum alongside writes for later
interpretation, without silently substituting M_s for M_(s+1).

The joint panel applies the three ACTUAL up-weight changes simultaneously.
Every later perturbation uses its CURRENT forward input, including changes
caused by earlier matrices. Compare joint remainder with the sum of the three
individual remainders. Retain the full3x3 coefficient-space Hessian/GN when
qualified, including signed cross terms. This measures selected-up interaction,
not all body/auxiliary or all cross-layer coupling.

## Analysis fixed before scoring

- Plot true Hessian versus GN curvature by method, depth and training time.
- Plot both signed finite remainder profiles against the local quadratic,
  with per-context curves and actual/activation-radius axes.
- Compare actual versus left rotations at identical activation norm per token.
- Compare raw/whitened right-rotation departures, retaining the changed
  activation norm distribution and both denominator norms. No selected best
  frame or percentage improvement is a success gate.
- Plot selected joint interaction and its quadratic prediction over time.
- Save numeric tables, full direction/activation/provenance artifacts, and
  an explanatory report with supported observations and unresolved questions.

No optimizer ranking, fitted exponent, stable-phase threshold, or microscopic
loss margin is used to decide whether this atlas counts as completed. Its
value is the measured structure and how that structure changes over time.

## Qualification, resource bound, and preservation

Before the full panel, qualify the FP64 function, AD derivatives against
finite differences, cached suffixes, source/config/tensor pairing, left/right
invariants, and joint current-input implementation. Preserve failures and
changes with their reasons. First engineering timing may change execution
chunking but not the declared scientific panel after results are inspected.

Use at most two CPU numerical threads, no CUDA. Forecast full cost from a
completed representative panel before continuing; intended upper bound two
hours of numerical execution, with progress saved after every direction.
If execution exceeds that bound, retain the partial atlas and diagnose the
cost without dropping unfavorable cells. No training follows automatically.
Status and decisions go in this directory's STATE.md because the user requires
observer files separate from main research notes. The previous observer's
closed reports remain unchanged.

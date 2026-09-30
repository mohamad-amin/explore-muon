# Independent priority review: make interaction a discriminator, not a destination

2026-09-28. Read-only scientific review of existing project evidence. No model
execution, optimizer construction, GPU use, training, or scheduler change.
The new files from this review are confined to this directory. Sealed external
confirmation data and `../last_layer` were not inspected. The guide, current
state, observer synthesis/state, underlying reports, and relevant main-protocol
entries were read. Live `squeue` showed allocations 2567578/2618555 running and
2626862_[1-9] pending; those records are not authorization for new work.

## Judgment

The body–auxiliary interaction is a legitimate missing part of the current
body-only geometry description. **Attributing its positive sign to a parameter
group is not yet a mechanism or an optimization opportunity.** The next probe
is worthwhile only if it distinguishes ordinary useful predictive overlap,
relative step-scale calibration, and nonlinear representation/readout transport.
Its conclusion must change which of those stories remains plausible.

Otherwise this becomes another one-step audit whose endpoint is "the parts
interact," already established for body pieces and value-route components.
There is no reason to expect a positive interaction to disappear in a good
optimizer. The archive's actual advances come from whole trajectories, while
several very large one-step opportunities failed to become rates.

The broader question that survives the observer's corrections is: **how does
the optimizer allocate changes in predictions across interacting parameter
groups and over time?** Parameter energy, internal activation amplitude, raw
gradient SNR, and selected-direction coherence each lost a proposed causal
interpretation when checked in the relevant metric or on a trajectory. That
is a useful narrowing of the mechanism, not an indictment of measurement.

## Three competing explanations

### 1. Ordinary overlap of useful prediction responses

Two updates can individually reduce the same prediction error, then jointly
overshoot it. The resulting positive loss interaction does not identify an
expendable channel. Even the elementary quadratic

    L(b,a) = (b+a-1)^2 / 2

at b=a=0, with body displacement .9 and auxiliary displacement .4, has
body-only change −.495, auxiliary-only change −.320, interaction +.360, and
full change −.455. The auxiliary update helps alone but hurts conditional on
the body update, exactly the qualitative pattern in the observer probe.

This explanation has unusually strong local prior support:

- Every one of 1,128 off-diagonal entries was positive in each archived
  Muon/PD/K-FAC selected-body-direction Gram. Their dominant normalized mode
  was broad and aligned with the descent-slope vector at squared cosine
  .968–.993 (`../coupling_archive/REPORT.md`). Removing shared curvature can
  therefore remove useful descent in almost the same proportion.
- Actual value constant/centered steps have GN correlation about .56 under
  both Muon and PD. Their finite interactions agree with the predictive-GN
  cross term to factors 1.063 and 1.018 (`../value_split/REPORT.md`).
- Actual own steps often minimize local loss near half their full scale;
  one-step conditional losses are therefore observed inside an oscillating,
  co-adapted trajectory, not independent greedy descent moves.

**Prediction:** an additive-logit counterfactual explains most body–aux loss
interaction; the effect follows the magnitude and alignment of useful
responses, without a large mixed change in logits. A concentrated head share
would still be compatible with this null because the head touches every logit.
This outcome closes the special-channel-remedy story at the tested state.

### 2. Relative functional step size and training phase are miscalibrated

The body and auxiliary parameters use different optimizers and scales. The
same auxiliary recipe across body methods does not imply the same auxiliary
change in predictions: the learned representations and weights differ. A
one-step overlap may be ordinary yet become unusually costly because one
group's response is large relative to the current joint descent.

The head-whitening branch directly exposes this confound. Its initial
whitened-coordinate norm matching changed the effective head step; alpha 1/2
lost .295, alpha 1/4 gained .040, and centered matched alpha 1/2 still lost
.217. The independent 13:50 review paused the repair cascade because the
plain-head Adam learning-rate control was missing (`MUON_CASE.md`,
13:27/13:50 entries). None establishes a whitened-head advantage.

The horizon evidence makes a fixed-scale/fixed-lag explanation plausible but
not complete. SOAP-PD's beta .8 advantage changes from roughly .10 on the
short horizon to .010 on the long one, while the scalar momentum ages remain
different. In 4M PD, the shorter-momentum early lead reversed at completion
(`../clipping_connection/COMPLETION_ADDENDUM.md`). Training phase matters.

**Prediction:** the additive response accounts for interaction, but the
conditional auxiliary slope/curvature and functional response ratios differ
systematically across already saved early/late states. The relevant change
tracks training phase or body geometry, not merely parameter-count or norm.
A one-state fitted head LR cannot test this explanation. Before any future
head-geometry training claim, a scalar head-only LR control is mandatory for
scientific identification; it is not permission to run that control now.

### 3. Representation/readout transport creates a material mixed logit term

For a linear head z = W h with all other auxiliary parameters fixed, changing
the body and head together gives an exact mixed logit response

    z11 − z10 − z01 + z00 = (Delta W) (Delta h).

This is distinct from the softmax-loss curvature applied to two additive
responses. The head update acts on a changed representation; analogously,
embedding and normalization updates can change the inputs or scales seen by
many body matrices. A body-only GN model does not include these cross-group
terms. If they matter repeatedly, the mechanism points toward co-adaptation
or transport of optimizer state, not necessarily lowering an auxiliary LR.

Existing evidence does not demonstrate this explanation. In the completed
body/aux probe, body-only and full body-gradient changes have cosine .9979,
and auxiliary scope was not the source of the cross-bank sign reversal.
That constrains strong "auxiliaries drive all body rotation" stories, but it
does not decompose the +.01809 finite loss interaction into predictive overlap
and nonlinear response transport.

**Prediction:** a mixed logit response makes a material, reproducible signed
contribution after additive prediction overlap is removed. For head-versus-body
it must satisfy the exact bilinear identity, and its loss contribution should
recur at a prespecified independent saved state before motivating a dynamic
architecture or optimizer explanation. A large raw mixed-logit norm alone is
insufficient: softmax gauge directions and loss-insensitive responses matter.

## The exact discriminator available to a future bounded forward probe

For one token, let z00,z10,z01,z11 be the four corner logits, with fixed true
label y. Define a=z10−z00, b=z01−z00, c=z11−z10−z01+z00. Then exactly

    I = L(z11) − L(z10) − L(z01) + L(z00)
      = I_overlap + I_transport,

    I_overlap = L(z00+a+b) − L(z00+a) − L(z00+b) + L(z00),
    I_transport = L(z00+a+b+c) − L(z00+a+b).

For softmax cross entropy L(z)=logsumexp(z)−z_y, the explicit label term
cancels from I_overlap. It measures the finite loss interaction of additive
prediction responses, without a GN approximation. I_transport retains the
true label and isolates the additional mixed response along this specified
path. Neither term is generally nonnegative. At small scales, the first tends
to the predictive-GN cross term; the second includes the residual term missing
from GN. Their finite split is exact but asymmetric in the chosen path: it
adds c last and is not a unique causal attribution.

This avoids making another coefficient eigenspectrum or optimizing a learning
rate on the diagnostic bank. It needs only the already proposed corner
forwards plus CE at the synthetic additive logits. Corner logits can be
processed one sequence at a time, retaining per-sequence sufficient scalars.
No backward pass is required for this exact decomposition. The old body_aux
probe did not retain logits, so no numerical claim about this split can be
recovered from its saved losses and gradients alone.

`verify_identities.py` checks these identities and the toy counterexample using
small synthetic arrays and the Python standard library; it calls no model.

## What the existing interventions already rule against

| Simple proposed remedy | Existing constraint |
|---|---|
| Delete or center the mean route because its gradient/activation is large | Mean-only interventions did not recover full PD gains; the value-route conditional gain changes sign across banks, and the internal V amplitude ordering is not V/O-gauge invariant. |
| Suppress positive coupling or stage independent body updates | Common response is closely aligned with descent. Staged Muon/PD help on fresh inputs but lose on actual momentum (−29%/−37% local model decrease). |
| Delete the stiff shared modes | Top-32 global deflation moves PD's reference fraction from .739 to .411 in the saved test. Coherent selected-piece Grams are not the full GN eigenspace. |
| Whiten the head and call any gain geometric | Existing norm matching confounded scale; the plain-head LR control is absent. The branch was paused after failed predictions, not confirmed. |
| Shorten momentum because clipping normalizes gradients at large batch | Muon clip .1, beta .8 versus .9 gives +.00341 final NLL; normalized input alone does not reproduce PD/SOAP-PD's large short-run gain. |
| Improve curvature-estimator weight-space repeatability and infer a rate gain | TS fresh maps have cosine around .70 but predictive-GN cosine around .99; response amplitude changes the apparent noise ratio. No rate bottleneck was established. |
| Treat a large exact-GN local opportunity as a training prescription | GN-PD's gain was front-loaded; the 16M exact-geometry preflight was sample-sensitive and worse than Kronecker. Greedy Muon also lost under the greedy GN step rule. |

These are scoped counterexamples, not universal impossibility theorems. They
prevent silently recycling a failed premise as a new mechanism.

## Bounded decision tree

1. **Inventory first.** Read the actual saved displacements, parameter groups,
   optimizer scales, and prior head/clipping controls. Root is doing this
   separately. Parameter norms locate movement but cannot identify its loss
   effect. No new model call is necessary at this stage.
2. **Choose one explanatory split before execution.** Prefer body versus head
   versus the remaining auxiliaries because the head has an exact mixed-logit
   identity and a documented LR confound. Eight corners give a complete
   three-group factorial and reveal third-order interactions; a cheaper set
   of isolated subgroup contrasts must explicitly retain unattributed
   higher-order interactions. Do not force subgroup effects to sum when they
   do not. Keep the completed 16-sequence banks fixed for diagnosis, set a
   CPU time cap before execution, and do not increase the sample on a weak
   result. This is a proposal requiring the observer's own protocol decision,
   not an instruction from this review to execute.
3. **Use the exact logit split to decide the branch.** Dominant additive overlap
   without a special residual favors explanation 1 and ends channel-removal
   work. A distinctive functional scale imbalance motivates checking the
   existing phase/scale inventory, explanation 2. A robust material transport
   contribution motivates explanation 3. Failure to resolve the split under
   the cap is an unresolved premise, not permission for a repair cascade.
4. **Require a second saved-state prediction, not another descriptive ranking.**
   If step 3 selects a specific mechanism, prerecord its sign or scaling
   prediction at one already saved state chosen for a scientific reason
   (for example a late phase, not the largest effect found by scanning).
   Freeze the same functional decomposition. A negative result closes the
   claim's scope; it does not trigger a larger bank or more subgroup splits.
5. **Only a repeatable trajectory prediction can motivate a future method.**
   Explanation 2 calls first for the missing scalar control; explanation 3
   calls for a transport-aware hypothesis with a scalar-matched competitor.
   Explanation 1 offers no new remedy. Any eventual rate test needs frozen
   criteria, the method's own trajectory, matched relevant resources and
   full learning-process controls. No training or GPU test is requested now.

The natural successful endpoint for this observer pass may be a precise
negative conclusion: ordinary response overlap explains the anomaly and no
new optimizer branch is justified. Conversely, a nontrivial mixed functional
term with a correct held-out-state prediction would advance an architecture
mechanism beyond the existing descriptive audits, without claiming novelty
or a robust improvement prematurely.

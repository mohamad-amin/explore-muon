# Observer synthesis: which signal does an optimizer need?

2026-09-28, updated after progress-lead and normalized-metric interpretation.
This is an intermediate result in the open observer goal. CPU only; no
training, GPU use, or main-study changes. The previous synthesis is preserved
in `SYNTHESIS_pass16.md`.

## A retained advantage is clear; a precise rate trend is not

The completed three-pair warmup reduction places the shrinking vertical gap
in context. After both arms use beta.9 and before LR cooldown, all fixed
training-curve anchors show a positive apparent lead. With31-row means it
stays near6updates, changing only.16–.43over states110→140. With11-row means
the same endpoint change is1.03–1.79updates and the interior lead rises and
falls. This describes preserved progress with possible erosion, not a
uniquely identified constant translation. Horizontal training comparisons
use different batches on the same ordered stream; agreement across seeds
does not remove that nuisance. See `progress_lead/REPORT.md`.

The fixed-bank archive supplies only100/150anchors inside the shared phase.
Every declared inverse loss target uses the same secant. Constructive monotone
curves with constant5, eroding10→2andgrowing1→9update leads all reproduce
the actual four anchors exactly in each pair. That proof of nonidentification
is stronger than treating interpolation sensitivity as an error bar. It
does not invalidate the retained endpoint gains, nor show that all constructed
curves are feasible optimizer trajectories. It closes the scalar archive route
without choosing a smoother or adding evaluations to force a rate story.

## A capped metric changes allocation, even where root coefficients stay fixed

The main's new PD-top intervention asks which part of PD's root profile is
necessary. The independent interpretation review exposes a useful connection
to prior equal-norm and gauge results: the normalized exact map
D=k polar(MR)R/||polar(MR)R|| is invariant underR→cR. Therefore calling a
root factor an amplification because it exceeds1 is partly a scale convention.
The nonuniform clamp is meaningful because it changes the relative spectrum.
For square/tall full-column-rank matrices, DᵀD=k²R²/trR². Capping low-input-
variance factors raises final energy on unchanged high-input-variance factors.
The metric can be viewed as spectrally floored, but suppression and tail
allocation are not independent pieces of the matched update.

Allfour new preflight JSONs make the issue tangible. FullPD andPD-top put
almost equal absolute raw energy in the top16GNsubspace (.87–1.10ratio),
although their fractions differ15–58times. FullPD's much larger raw norm
accounts for most of that discrepancy. PD-top still suppresses absolute
projected energy17–24times versusMuon, so its own result is not denominator
inflation. Once globally matched, the fractional difference becomes an actual
projected-energy difference. Online matching is per matrix and not recoverable
from aggregate scalars. Allrawmapsuse laggedM and are uphill here; none is
the next applied training step. See `pdtop_interpretation/REPORT.md`.

This connects three levels of evidence: coefficients of a root, allocation
of a normalized parameter update, and useful finite prediction change. The
forthcoming training outcome adds a fourth. IfPD-topworks, the uncapped
profile is unnecessary in that recipe; ifitfails, the capped recipe is
insufficient. Either result is useful without uniquely identifying spatial
feedback versus weak-direction signal. The present fullmatriximplementation
is not yet cheaper. No additional training or map reconstruction follows
from this clarification; the next discriminator should add scientific
information rather than compensate for a preferred mechanism reading.

## Routing, errors and sequence weighting are different parts of value curvature

The old statement that diffuse attention can amplify value coherence now
has a precise conditional form: F_in=X^T A^T R A X/T. R is downstream
temporal error covariance after tracing channels; attention concentration
alone does not determine it. Only under independent equal-query errors does
this reduce to a pooled-input second moment, and only with iid residual
inputs independent of routing does it reduce further to two activation
moments. The archive's empirical sequence centering changes the prediction:
b=1/q but a=(Tq−1)/(q(T−1)); at 512 uniform causal positions, b≈75 and b/a≈88.
The exact and approximate objects are qualified separately, with prior art
acknowledged, in `attention_pooling_geometry/REPORT.md`.

The complete saved scalar check is not encouraging for that closure. Only
one of 256 V fits has its necessary normalized coefficient relation within
.10, and every two-component matrix fit has residual above .10. The early
fits include nonpositive within coefficients; later within coefficients can
be 3–13, rather than near the idealized range. These are descriptive
incompatibilities of retained Monte Carlo fits, not a population test or an
identified failed independence. No new q measurement or fitted rescue follows.

A useful interpretation correction survives: absolute b mixes overall scale
and relative sequence weighting. Final b differs roughly twofold between M
and PD/SPD, while median per-layer b/a is about 1.63–1.72 across methods.
That does not make the component matrices equal or the fit exact, but it
prevents treating b alone as an effective attention width or normalized
preconditioner shape. Forward routing, input correlations and downstream
metric dependence remain distinct targets.

## The schedule result strengthens while the mechanism remains open

The second SOAP-PD seed now gains .023944 from warmup versus constant beta
.9, agreeing with the original .021193 gain. Alongside PD's .026671 transfer,
both planned cohort pairs pass their original .01 threshold. Sources, data,
initialization and per-step clocks match within pairs. This is two SOAP-PD
seeds plus one PD seed, not a crossed multi-method replication or a universal
schedule. The short-horizon failure remains. See `warmup_transfer/`.

The new `dynamics_synthesis/PEER_REVIEW.md` sharpens the methodological
lesson: a component-last loss failure is not a necessary condition for its
long-run role; greedy-Muon already supplied that warning. Three accounts
remain live—an early progress head start, an endogenous relative-rate effect
from radii, and richer moving coupled feedback. Shrinking vertical loss gaps
alone do not distinguish them. The next trajectory check must retain actual
measurement cadence and interpolation uncertainty rather than fit a time
warp or treat a local score as a training rate.

## Trajectory transfer is positive; the conditional-input archive does not qualify

The distinct cancellation hypothesis has reached a provenance limit before
real tensor arithmetic. Current code suggests nested g1M/g4M samples, but
the old GN2 outputs do not freeze the executed producer or sampling calls;
that source was edited later. The proposed complementary 3M mean therefore
cannot be called a verified disjoint sample. `conditional_input/REPORT.md`
records the bounded no-go. BF16 interval propagation and synthetic checks
are retained as a future method, not evidence that cancellation occurred
or amplified relative disagreement in these actual gradients.

The conceptual connection remains valid: adding the same history leaves
the absolute gradient difference unchanged while potentially changing its
size relative to the remaining input. Raw signal energy, history-conditioned
input, mapped parameter motion and prediction changes remain distinct. No
iid-noise, population-SNR or actual clipped-input conclusion is supported
by the unqualified de-mixing.

Meanwhile, a completed main PD pair adds direct trajectory evidence. At
the doubled 16M horizon, warmup ends .026671 below constant beta .9, meeting
the original .01 criterion. Its early validation lead of .116697 narrows
but persists after the schedules share beta .9 from update 93 onward.
The matched pair differs only in momentum scheduling; sources and per-step
clocks match. See `warmup_transfer/README.md` and its independent review.
It is one PD seed and has no constant-.8 PD reference. The new SOAP-PD seed's
reference was unfinished at that review and has since completed as recorded
above; the original PD-only artifacts remain intact.

This positive transfer and the failed Q/K-last usefulness test can both be
true. They concern different objects and horizons. The evidence supports a
useful schedule in two selected whitening recipes, but not a causal step
clock, a particular normalized group, or noise cancellation as its mechanism.
The next synthesis should account for the evolving state and gradient
history rather than replacing trajectory progress with another local score.

## The angular pattern reaches predictions; local benefit remains order dependent

The missing functional premise is now measured in `qk_function/REPORT.md`.
At the fixed four SOAP-PD 46→47 states, shorter momentum enlarges Q/K-only
predictive KL by pooled factors 1.725/1.755 at the two LRs; after the rest
of the actual update, the factors are 1.453/1.679. Higher LR increases KL
substantially less than the nominal squared-step ratio in both backgrounds.
All declared bank-level pattern conditions pass. The positive angular
observation therefore has a genuine finite predictive counterpart at these
own states; it was not merely invisible parameter-scale motion.

The chosen helpfulness bridge fails. Q/K helps when applied alone, but
at shorter beta its last-added mean loss effect is +.001102 and +.007604,
with bank disagreement at low LR and both banks adverse at high LR. The
symmetric two-order improvement credit remains positive everywhere. It is
an alternative attribution convention retained as descriptive evidence,
not a replacement for the actual-background criterion chosen before scoring.
One must not turn this order dependence into "Q/K is useless."

The loss interaction is predominantly ordinary additive prediction overlap:
.00937–.02173 total interaction versus −.000182 to +.000229 finite mixed loss.
Positive Q/rest response correlations (.605–.707) connect this example to
the earlier broad useful-coupling and body/auxiliary findings. Small isolated
movement can interact substantially with a larger complementary group; its
loss value cannot be assigned by a parameter-norm fraction or by the most
favorable application order. These finite covariances are not tangent GN.

The better-training beta-.8 states also have smaller immediate full-step
decreases on this selected panel. The result again separates actual learning
progress from the value of one local finite step. It does not identify
radius, memory or overlap as a causal mediator, or justify freezing Q/K,
removing radial motion, or fitting a smaller step. Close the fixed bridge.
The next priority should come from a broader comparison of remaining
anomalies and artifacts, not automatic subdivision of the 68-tensor rest.

The [priority reset](priority_reset_after_qk/REVIEW.md) identifies a distinct
candidate at the optimizer's input: compare repeatability of fresh gradients
with that of the input left after the same momentum history is added. This
connects the replay cancellation and raw-signal/update-weighting findings
without another component ablation. The two old `gn2` states may retain
the necessary g1M/g4M tensors, but historical sample nesting failed
qualification in the bounded review above. The precision design is retained
without numerical gradient interpretation. No predictor outcome or new
method is established.

## Endpoint confidence test: ranking reproduces, finite correction unresolved

The proposed direct confidence discriminator is now complete in
`confidence_calibration/REPORT.md`. All six schedule endpoints select a=1
on both fixed fitting banks, so opposite-bank calibrated scores equal raw
scores. Warmup minus constant beta .8 is +.042946 at 1× and −.012366 at 2×,
with original signs on both fresh banks. This is a useful consistency check
for the same trained endpoints, not an independent seed replication.

The declared all-one outcome means the coarse family resolved no finite
correction. Negative mean scale derivatives remain, while a=1.1 increases
loss. Therefore a discrete unit minimum is not continuous stationarity;
the experiment cannot exclude finer confidence differences. No inferred
temperature or unmeasured loss gain is substituted for the fixed panel.
Original unavailable-offset planning and a harmless reduction-path repair
are preserved, and independent results review confirms the final zero
corrections. Close without grid/sample/state expansion.

This keeps two observations distinct. The Q/K comparison shows a real
relative weight-motion difference under equal nominal budgets. The endpoint
test reproduces a schedule ranking but does not determine its confidence
component or identify Q/K motion as its cause. Before another intervention,
the useful missing premise is whether existing actual-step functional
artifacts tell how much those normalized Q/K motions contribute to the
model's prediction change. Weight-space geometry alone is insufficient.
The inventory in `angular_clock/FUNCTIONAL_AVAILABILITY.md` found whole-body
totals at six low-LR states but no kindwise decomposition, and no matching
profiles for the six high-LR states. Older direction Grams could not fill
that gap. The separately reviewed four-state test above supplied new
information without inferring a share from parameter energy.

## A new connection: momentum alters the relative step through weight radius

The prior LR factorial held nominal body norms fixed within beta pairs and
increased total nominal movement 43% across LRs. That was a useful control,
but it did not hold or measure relative motion of normalized feature weights.
The new twelve-state saved-weight comparison in `angular_clock/REPORT.md`
fills that gap. Per-head Q/K normalized chords grow only 9–19% late when
LR grows 43%, alongside radii 21–31% larger. Shorter beta gives 15–25%
larger chords, mainly through 13–20% smaller radii, despite essentially
equal head-step norms. Both reviewed materiality criteria pass in every
later Q/K × setting cell. Early-state contrasts are much smaller/different.

This connects architecture and optimizer memory: normalization makes a
weight's radius irrelevant to its output under ideal positive rescaling,
but that radius affects how much a fixed-size future write rotates the
weight direction. Momentum, the nonlinear map and the evolving state can
therefore change relative motion even when configured LR/norm budgets are
identical. The exact radial cross terms are strongly positive, so a pure
diffusion eta*sqrt(t) explanation is not warranted. The sample does not
attribute the drift uniquely to any one of those causes.

The earlier nominal-arc result is preserved; its exclusion of a generic
"more prescribed movement" account does not exclude a relative-motion
account. The main conclusion that phase should be scheduled in optimizer
steps is stronger than the available comparisons establish. Q/K chords are
not actual prediction distances, all arms occupy their own states, and the
other body kinds lack this individual scaling symmetry. No radius control,
momentum rule or training-rate benefit follows yet.

Both the nonlinear-channel branch and this fixed necessary-condition test
are closed at their declared scope. Independent review then proposed the
distinct scalar-temperature discriminator completed above. Existing
loss-minus-entropy values revealed confidence differences but did not
quantify that finite opportunity. Its all-one outcome remains separate
from the angular observation and from the paused head-whitening repair.

## Finite usefulness changes the decision: close the nonlinear-channel lead

The latest exact split resolves the missing premise from the dose probe.
Early half-power PD has a negative total body–auxiliary interaction on fresh
inputs, −.047696, but this is additive overlap −.053006 plus a slightly
adverse finite mixed effect +.005310. Mixed-effect signs disagree between
the two fresh banks. The prospective absolute-helpfulness and phase tests
fail; the half-minus-quarter comparison passes as relative improvement,
not as absolute helpfulness. See `early_split/REPORT.md` and the independent
result review in `early_coupling_peer/`.

The older inputs expose the distinction particularly clearly: negative J
on every sequence coexists with positive finite mixed loss on every same
sequence. The additional KL correction reverses the tempting interpretation.
This is a precise connection between the broad concern about one-step
linear scores and a measured whole-model example. It does not invalidate
J's algebra or the presence of label-relevant output nonadditivity.

The prior late-Muon overlap result now has a useful extension: ordinary
additive prediction interactions can be complementary as well as redundant.
Their sign changes across phase and own state. Neither sign makes overlap
or a parameter group an intrinsic defect. A subgroup factorial is therefore
unwarranted under the declared decision; this branch closes without expansion.

Separately, both original warmup horizons are now complete. The positive
2× result is retained, but the 1× run ends at 4.515206, +.0428 above constant
beta .8 and outside its declared .02 target. This supports horizon-sensitive
momentum choices. Existing comparisons do not identify steps, parameter
distance, estimator age or function-space progress as the causal clock.
Independent discussion is considering distinct saved-evidence questions;
the broad observer goal remains active.

## Preceding lead, now qualified above: early whole-model nonadditivity

The main dose experiment supports its predicted alpha×momentum interaction:
shortening beta improves quarter-power PD by .086 and half-power PD by .124
at 16M. The observer's actual-step probe adds functional information at
9→10 and 46→47. Equal parameter budgets do not fix prediction changes:
stronger whitening reduces body KL later, but increases it early at beta .8.
Auxiliary functional steps also change despite identical auxiliary settings.
This keeps functional amplitude and state co-adaptation distinct from the
proposed stiff-oscillation mediator. See `dose_function/REPORT.md`.

At step 9, quarter-power body/auxiliary updates have positive mean loss
interaction (+.087/.061), while half-power has negative interaction
(−.047/−.035); the signs hold on all eight inputs at both momenta. At step 46
all four interaction means are positive. The interaction depends on phase
and geometry, not merely the parameter-group name.

Retained CE and KL permit a post-hoc exact identity:
`J = I_CE − (KL_full−KL_body−KL_aux)`
`  = mean[(p0−e_y)·(z_full−z_body−z_aux+z0)]`.
Early half-power J is −.03275/−.02970, negative on every input. It vanishes
for exactly additive logits, so it establishes a label-relevant mixed model
response. It is not the previous CE(full)−CE(additive logits) term; without
the additive logits' KL, no finite overlap fraction can be reconstructed.
Do not compare these as percentages with the late-Muon 96–98% overlap result
or call J a measured residual Hessian.

This additional early-phase phenomenon is not yet the training mediator:
its sign change appears at both betas before the strongest training
interaction develops, and half-power loses at beta .9 despite its favorable
early joint-step effect. A targeted future comparison must distinguish the
full mixed-loss effect from prediction overlap and localize it before any
subgroup remedy. No extra model calls or rescalings were added to this probe.

Separately, the main 2× momentum-warmup run completed at 3.962814, meeting
its numerical target and beating both constants in this seed. That supports
a practical phase-dependent window in the tested recipe; it does not identify
the observer's early mixed response as its cause.

## A direct predictive test: frequency profiles depend on learning phase

The priority reset chose new forward-only measurement over another scalar
geometry audit. Ten saved1Mcheckpoints were evaluated on32fixed fresh
sequences, with target-frequency bins defined from the original data-only
countbank. Existing validation scores fixed a Muon-progress reference before
new losses were read. Both own-Muon interior-anchor controls were retained.
The complete136-second CPUprobe is in`prediction_profile/REPORT.md`.

The expected extra rare-target SOAP signature did not appear. Same-step
rare-leaning gains at500became common-leaning at900; after the independent
progress correction, all12method×stage×bank contrasts were positive, opposite
the predicted sign. Actual overall predictive gains remained. Ordinary Muon
progress itself changes its rare/common profile strongly:−.311 over500→900
and−.043 over900→1300. Its interpolation controls reach−.065, comparable to
or larger than the method residuals, so a common-token mechanism is not
identified either. The directnearlymatchedS500/PD500bank signs disagree.

The older final-checkpoint frequency table remains a valid observation in
its scope. The new result shows why that table alone cannot identify a
heavy-tail normalization mechanism independently of phase. Target unigram
frequency is also distinct from hidden-feature-coordinate signal/noise.
No sample/bin/phase-warp extension follows this bounded screen; a32-sequence
result does not prove the absence of regime-specific learning differences.

## A useful rejected connection: geometry decay does not restore gauge covariance

The V/O gauge result raised a possible link between activation preconditioning,
normalization and successful geometry decay. Independent review and exact
algebra close the broad version of that explanation. Input-only half-power
PD has a raw O-side covariance property, but lacks the output-side metric
needed for V; per-leg scalar normalization cannot repair that missing shape.
Practical damping, factor scaling and Frobenius matching further distinguish
an equivariant ray from an equivariant step.

Most decisively, ordinary scalar weight decay already commutes with the
fixedV/Ogauge, whereas geometry-shaped decay generally does not. Its empirical
benefit cannot therefore be credited to restoring this covariance. Existing
magnitude-matchedMuon and hyperball controls also weaken a normalization-only
account. The project never claimed otherwise; this was an observer hypothesis
checked before launching a broad saved-matrix stress test.

The one fixed2×2qualification and primary-source precedents are in
`gauge_optimizer/REPORT.md` and`LITERATURE.md`. K-FAC's qualified affine
invariance, GO-MUON's distinction between oracle ray and grafted radius, and
Circuit-Muon's existing V/Ocoupling prevent a novelty claim. Close this branch
without a gauge-balancing or normalization intervention. Useful regularization,
coordinate covariance and training rate remain different properties.

## Current connection: the residual optimizer input has a different signal scale

The transport archive retained100batch-gradient projections at the same
weights. Their exact decomposition separates an equally weighted common
component from the effect of giving recent batches more weight. In the16
retained coordinates at Muon/PD1M500, that data-reweighting term is only
5–8% of the common component yet2.7–2.8times actual momentum. Model-history
changes dominate the much larger actual/replayed gap. This is a temporal
counterpart to the raw-SNR/update-weighting mismatch: variation can be small
relative to a large fixed-state mean and substantial after the model's
history cancels most of that mean. It does not identify a new optimizer or
equate conditional replay variation with independent sampling noise.

The observed chronological ordering makes the weighted residual's squared
norm5.23/5.37times its exact uniform-row-permutation expectation. A known
500M-token shard boundary aligns with this recency weighting, while removing
its fitted mean jump explains only6.4% of raw centered variance and leaves
positive lag structure. The tempting94–95%weighted-attribution statistic is
partly geometric: fitting/scoring the same rows already produces a.77655
ratio-of-expectations reference from weight/regressor alignment. All these
quantities are retained and calibrated in`replay_projection/REPORT.md`.

This leaves a legitimate data/dynamics question, not a shard-causality claim.
Those batches trained the checkpoint; content, age and model conditioning
are entangled. Another regression on the same100rows cannot identify them.
No shuffle or mean-removal intervention is justified by this result.

## Historical correction: differential gain was already measured

The earlier observer inventory was restricted to reconstruction tensors and
new16Mprofiles. It missed the September27`eos_linearized`JSONs. Those records
already measured a conditional FP32finite-NS Jacobian times sampled GN at
10olderstates. Near-one stability ratios at beta.95did not generalize;
beta.81and Nesterov had ratios3.36and2.78. Actual clipping was inactive at
the decisive exceptions, so it does not rescue the old universal prediction.
See`linearized_edge/REPORT.md` and the independent source review.

The derivative/value distinction remains correct, but it is not a new project
proposal. Moreover finiteNS plus input normalization is not generally
symmetric/PSD (an exact source-polynomial example passes finite differences).
Only leading-real eigenvalues were retained. The old data refute a universal
instantaneous edge within that surrogate; they do not establish saturation
as the unique stabilizing cause or measure products of changing full-state
Jacobians. Any future feedback experiment must address this earlier failure.

## Latest trajectory constraint: the useful window is not a body-norm budget

The feedback inventory found enough tensors for conditional matrix-map work
at older states, but not an exact reconstruction of the new16Mfeedback
measurement. Instead of launching a replay, a fresh independent review
identified an existing completed2×2: SOAP-PD at beta{.8,.9} and body
LR{.028,.04}, same seed,92steps,Ada, initialization/data and auxiliary clocks.
The beta.8validation benefit is~.108atstep50 and~.102–.105atcompletion at both
rates. The LRinteraction is small there, while a declared early training
window retains a−.021interaction. No significance or equivalence claim is
possible from one paired seed. See`momentum_lr/REPORT.md`.

The exact online norm matching makes nominal body arc192Σeta, the same within
each beta pair. Raising LR changes it from466to666, yet does not materially
change the late short-window benefit. The broad trough is also more aligned
by steps than by this movement proxy. Thus a simple accumulated-body-distance
account is weakened; shorter momentum is using the same prescribed norm
budget differently. This does not identify a unique clock: actual functional
movement, auxiliary directions, statistical transients and evolving geometry
remain distinct, and the92-step family ends before the long-horizon reversal.

The newly completed4Mmain trajectory independently retains the same early
short-window lead and ends~.002worse, strengthening the phase qualification.
It does not itself establish that steps are causal, since cross-batch recipes
change several clocks. The new2×2 is a cleaner existing discriminator and
is now closed without another sweep. The derivative question remains useful,
but the retained information and trajectory evidence should determine whether
its next measurement is worth doing.

## Current synthesis: shared progress and differential feedback

Two new results separate questions that were being conflated. The full
body/head/embedding/norm factorial attributes most positive joint-step loss
interaction to embeddings and head, then explains96–98% of it through the
overlap of their separate prediction changes. The mixed representation/readout
term is small at this selected checkpoint. A bank with no input rows absent
from the next training batch still has the embedding interaction, excluding
absent-row momentum as its explanation. The special-channel-removal premise
is closed; see `aux_partition/REPORT.md` and `embedding_clock/REPORT.md`.

That result gives the earlier coupling/value observations a direct finite-loss
meaning. Multiple groups can move toward the same correction and incur a
positive curvature cross term. Removing overlap may remove useful progress:
the post-hoc embedding projection retains55.5% of response energy but only
7.5% of label-linear descent. This is not a tested parameter update and it
does not argue that all coupling is beneficial. It makes reducing coupling
an insufficient objective.

The new momentum profiles meanwhile contain unused same-state map comparisons.
At all18 declared16Mstates, PD maps of the identical lagged momentum suppress
top16Ritz energy much more than ideal Muon maps. Independent-held curvature
per squared norm is also lower everywhere. This is a stronger operator-level
observation than comparing each method's step at its different trained state.
The successful geometry cannot be dismissed as co-adaptation alone.

But the logged M_s is not the input to the actual s→s+1 update. Its Muon/PD
maps are mostly uphill, while the actual fresh-gradient next steps descend.
The saved positive squared-slope quality hides that distinction. Polar itself
also reduces the raw momentum's stiff fraction substantially. The supported
story is extra covariance-dependent spatial redistribution, not an absence
of spatial redistribution under Muon.

Most consequentially, a quiet output in a stiff direction does not measure
the response to a perturbation in that direction. A fixed-norm3×3 example has
zero stiff output projection and arbitrarily large polar derivative. The
actual local feedback object is the joint weight/momentum update Jacobian,
which involves Dphi(M_plus) times the derivative of the incoming gradient.
Clipping, per-matrix norm matching, moving roots and SOAP state matter there;
replacing the gradient derivative by GN is an approximation. The exact
counterexample and source-faithful simplified Jacobian have passed numerical
qualification and independent review in `momentum_maps/`.

This is a falsifiable connection, not a new optimizer discovery: whitening
may permit shorter averaging by reducing *differential* feedback gain while
preserving useful response. Current scalar profiles show output suppression
but cannot establish that causal premise. Auxiliary overlap constrains what
must be preserved if one eventually changes the response. Neither result
licenses removing stiff components, freezing an auxiliary group, or tuning a
local step to a one-step diagnostic. Future work must add genuinely missing
dynamic information or move to a distinct promising artifact, rather than
repeat static energy comparisons.

## Fourth pass: distinguish parameter variability, prediction shape and amplitude

The output-factor sampling premise has now been tested. Four fresh8-sequence
factors produce TS map cosines~.70; repeated labels on identical inputs give
~.74. Pooling helps. Yet the retained directions have predictive-GN cosines
~.99 on independent inputs. The final inverse-root map, not NS alone, expands
their parameter differences. A large Euclidean discrepancy was therefore
insufficient evidence of an optimization bottleneck.

There is a second trap in the reassuring result. At matched parameter norm,
the PD reference produces a functional response2.27–2.41× larger than TS's.
The small GN variation-to-TS–PD ratio (<.006) partly uses this amplitude as its
denominator. Equalizing functional amplitudes post hoc leaves relative shape
uncertainty~.17–.29, not zero. Those extra normalizations were not training
experiments or replacement endpoints. Preserve all three views: weight-space
variation, prediction-space variation, and response magnitude. See
`ts_repeatability/REPORT.md` and `run2/metric_comparison.png` within it.

This closes the immediate covariance-sampling branch without declaring its
noise harmless. A known Fisher half-factor/Rademacher alternative was examined
algebraically and in the literature, but not run. Improving raw estimator
variance alone would not establish a useful nonlinear optimizer change.

Independent scalar reading also matters. The completed clipping control does
not reproduce the whitening methods' momentum benefit. Shorter momentum helps
early in an older4M run but loses by its endpoint, and its advantage narrows in
the newly completed longer16M run. Flatter loss curves account for part of this;
do not confuse loss-gap attenuation with an equal loss of speedup. Records and
source checks are in `clipping_connection/COMPLETION_ADDENDUM.md`.

The new tiny-character artifacts provide a separate sampling example: the
selected validation windows and their complement favor different methods,
verified from saved individual losses. This says where the reversal comes
from, not that it shares a physical mechanism with gradient noise. The old
scores are now development evidence and the main candidate failed its full
qualification. Its new confirmation data remain untouched by this observer.

The next distinct question is attribution of the measured body–auxiliary
finite-loss interaction. It is a more direct observed effect than the abandoned
instantaneous noise-bottleneck story; it still must be separated from generic
useful descent overlap. Current next-step boundaries are in `STATE.md`.

## Third pass: qualify the route, retain the quieter movement

The actual value displacement has now been scored in constant and centered
functional components on two fixed held-out banks. The extra constant benefit
after the centered step changes sign between banks. There is no repeatable
PD-specific local advantage to justify another centering/bias intervention.
The predictive GN cross term explains the positive finite interaction closely;
the two components have almost identical GN correlation (~.56) under Muon
and PD. See `value_split/REPORT.md`.

The saved-weight contractions also expose a necessary invariance correction.
Larger V-coordinate means do not necessarily mean larger functional routes:
an exact within-head V/O change of basis preserves the model function. After
O, the shared constant branch at step500 has summed energies93.96/101.24/
122.44/74.98 for M/PD/S/SPD, removing the old universal amplitude ordering.
Smaller actual joint V/O perturbations survive: PD's energy is5.2× smaller
than Muon's, and SOAP-PD's is9.7× smaller than SOAP-Muon's. These are fixed-mean,
own-state descriptors, not rates. See `value_gauge/REPORT.md` and the exact
counterexample in `value_gauge/IDENTITY.md`. Earlier amplitude claims below
refer to internal coordinates and are superseded as functional interpretations.

The small direction-Gram archive supplies a broader explanation of positive
interference. Its leading normalized mode spans many matrices and closely
aligns with descent slopes; it is repeatable across score halves. It does
not identify a disposable mean channel or the full parameter GN spectrum.
Prior block-GN/deflation/staging results remain constraints. See
`coupling_archive/REPORT.md`.

This closes the fixed architectural premise check without declaring the route
useless. The next distinct question is output-curvature estimator repeatability:
TS used a fixed small local sample while SOAP used global gradients. That
unmeasured estimator difference remains a plausible, falsifiable explanation
to examine before a new optimizer prescription. Current resource/decision
details are in `STATE.md`.

## Follow-up that changes the initial picture

The initial premise below has now been tested further. Fixed independent
input-rank groups show positive weak-tail alignment after one step, but dense
lag measurements change sign within a few steps. The tail is not a steady
slow gradient. See `PERSISTENCE_FINDINGS.md`.

A new architecture connection appears in the learned value projections:
relative gain on the shared input mean versus centered variation ends at
.077 for Muon, .446 for PD, and .603 for SOAP-PD. The larger mean route is
therefore not just a larger input mean; the learned value map treats it
differently. Attention preserves a constant value component exactly, but
data-dependent selection also changes its measured mean. Local usefulness
is still unknown. See `attention_geometry/NOTE.md`.

A controlled80-second CPU experiment then resolved a specific premise.
At Muon4M183→184, body-only and full updates both have positive same-bank
but negative cross-bank body-gradient alignment. Sample-specific components
explain the sign much better than auxiliary scope. Yet a separate finite-step
loss interaction is positive on all16 measured sequences: the auxiliary
step helps alone but makes the joint step worse than body-only. These are
distinct findings, not a mandate to freeze auxiliaries or alter training.
See `body_aux/REPORT.md` and independent review in `persistence_peer/`.

The original first-pass synthesis is preserved as `SYNTHESIS_pass1.md`.
The current next decision is in `STATE.md`: independently score the actual
value update's constant and centered functional components before proposing
a new parameterization or optimizer change.

The most useful connection across the saved work is that **large raw gradient
components, persistent gradient components, and components emphasized by the
actual update are three different objects**. Several current explanations
implicitly identify them. The archive already gives reasons to separate them.

## Evidence that changes the next question

### Raw signal does not describe where the update spends its energy

The new retrospective analysis reads twelve saved frame tensors, covering
Muon, PD, SOAP-Muon and SOAP composed with PD at steps 200/500/900. These are
1M-batch trajectories with independently fitted Kronecker bases and 8192
measurement sequences. Estimated high-SNR coordinates hold >99% of raw
positive signal energy, yet their complement holds 69–87% of body-displacement
energy. The complement also has positive measured slope in all 72
method/kind/checkpoint groups, though the mask-selection dependence prevents
treating those signs as independent statistical tests.

This makes the weak coordinates worth understanding. It does not establish
that they are noise, that all their movement is useful, or that their energy
explains the optimizer ranking. Noise can matter after normalization even
when it barely changes a raw-energy summary.

Separately, the newer SNR probe's estimator has a strong null effect. With
zero-mean Gaussian noise it produces 98.69% and 99.91% estimated signal
energy above SNR=1 at extrapolated 4M and 16M batches. Analytic calculation,
two million simulated entries, and an independent derivation agree. This
invalidates neither the measured training gains nor all gradient signal;
it means that particular statistic cannot resolve the mechanism.

Read `SIGNAL_NOISE_FINDINGS.md`; data in `saved_frame_reweighting*.json`;
calibration in `signal_noise_audit.json`; independent reviews in
`signal_noise_peer/`. The plotted null is `signal_noise_null.png`.

### An output-curvature comparison is also an estimator comparison

The frozen TS implementation estimates each output factor from eight local
sequences per refresh, not from the whole training batch or all ranks. Its
nominal stationary weight ESS is 72 sequences at every batch. SOAP's Gram
uses full globally averaged gradients, whose sampling error falls as batch
grows. Therefore the finding "this SOAP recipe beats this TS recipe at 16M"
does not by itself establish an intrinsic superiority of gradient statistics
over curvature statistics.

There is also useful negative evidence. Input C's clock is largely constant
in tokens, and at 16M TS's factor age is comparable to SOAP's, so simple
stories about a hidden 16-fold C acceleration or uniquely stale B are not
supported. A small/disjoint versus pooled B direction-repeatability check
would isolate the sampling question before another training comparison.
Its outcome is unknown.

Read `estimator_clocks/NOTE.md` and `TABLE.md`. Twenty-seven frozen-source
hashes were checked against nine completed runs.

### Local model improvement survives, but the flat-band explanation is weaker

Across 37 saved states, total step slope and curvature are reproduced to
about 1e-6 relative error. Those are well qualified measurements. A band
split built from 48 Lanczos nodes is a split into feasible Krylov vectors,
with valid within-subspace quadratic scores; it is not an exact projector
onto the full GN spectrum. The inferred flat-band length multiplier still
changes considerably as the prefix grows. Flat c*>1 survives prefixes
32/40/48, so the local opportunity should not simply be dismissed.

More directly, its relative slope fraction fails as a universal training
ranking: at 16M, TS has a larger flat fraction than the better-training
SOAP-PD, and fresher SOAP-PD momentum improves training while reducing that
fraction. Its absolute total model quality, however, improves by about
45–49%. That result does not need a spectral-band interpretation. Ratios
and numerators need to stay visible together.

Read `spectral_claims/NOTE.md` and `TABLES.md`. All 76 audited inputs were
hash-checked unchanged. This audit does not claim full48 is wrong, or that
one-step model quality establishes a training rate.

## A coherent hypothesis, and what would refute it

A candidate picture is that whitening reduces coupling to large oscillatory
gradient components and increases the relative importance of numerous weak
components; successful temporal averaging then needs to distinguish their
sampling noise from their useful persistence. The current SOAP mechanism
could involve signal equalization, noise control, temporal effects, or a
combination. This hypothesis connects the spike, preconditioner, batch and
momentum observations, but does not uniquely predict them yet.

An immediate falsifier of the simplest version would be weak-coordinate
groups that have no independently measured persistence or descent after the
actual update weighting is applied. Conversely, a stable population of weak,
persistent, productive groups would make a raw-energy-based dismissal of
noise control especially implausible. Neither outcome alone establishes a
new optimizer.

## Initial next analysis from the archive (now completed)

Use `persistence/*.pt` internally, without subtracting vectors from separately
measured eigenframes. Those files contain independent sequence sets for
the base and later states in the same saved measurement frame. Group entries
by predeclared input/output rank bins defined from the separate basis data;
retain signed cross-time products, debiased norms and cancellation warnings.
Relate those coarse groups to the existing frame-displacement energy by
matching rank groups only, not signed vectors. The groupwise association is
descriptive across probes/own states, and eigenbasis ambiguity is retained
as a limit. Start with Muon and PD at step500, interpret it, then decide
whether extension to200/900 is warranted. CPU reads only, <=2 threads, no
model calls, and a projected bound of two minutes.

The reason to do this next is substantive: does the portion of the model
movement hidden by the raw-signal summary track persistent structure, or
mostly transient/noisy structure? The previous analysis already establishes
that this portion is large enough to matter. It avoids proposing a method
from a measurement failure alone.

## Status and limits

The observer goal remains active. No robust optimization/architecture
improvement or novelty is claimed. The persistence analysis and CPU
counterfactual described above have completed. The separate pre-existing observer notebook
continues to hold the body–auxiliary premise-check proposal. Its files,
the main code, research state, protocols and scheduler queues remain untouched
by this pass. All new scripts, plots, data and notes are under this directory.

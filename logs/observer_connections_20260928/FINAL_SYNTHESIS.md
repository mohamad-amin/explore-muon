# Observer work: findings, contributions, and limits

Prepared 2026-09-28 after the user cleared the observer goal. The goal service
reports no active goal. This is a retrospective synthesis, not a continuation
plan or a claim that the discovery objective was achieved.

**Outcome:** the observer established several useful connections, qualified
important measurements, and narrowed explanations for the main program's
optimizer gains. It did **not** establish a new robust optimizer or architecture
improvement, a unique causal mechanism, or a novelty claim. PD, SOAP-PD, TS,
their training comparisons, and the momentum-warmup runs belong to the main
research program. This observer analyzed and independently checked that
evidence and performed bounded CPU experiments on saved states.

The central synthesis is that five objects must remain distinct:

    fixed-state gradient → history-conditioned optimizer input
    → normalized update and learned parameter scales
    → change in predictions → progress over a trajectory.

Several tempting explanations jumped across one or more of these links.
The observer's strongest results identify where those jumps fail and where
a structural effect really does reach the next level of observation.

## 1. Raw gradient signal does not describe the update's information budget

The main SNR probe used a positive-clipped estimate of signal energy and
extrapolated its threshold to larger training batches. Analytic null
calibration, simulation, and an independent derivation showed that **pure
zero-mean Gaussian noise** already produces 98.69% and 99.91% of positive
estimated signal energy above the declared threshold at extrapolated4M/16M
batches. The gradients need not be pure noise; the statistic cannot establish
signal dominance from a near-100% reading alone.

A different saved archive supplied an independent, complementary observation:
coordinates holding more than99% of estimated raw signal energy held only a
minority of actual update energy. The estimated weak complement held
**69–87% of body displacement energy** across the selected methods/states.
Those classifications do not certify that individual coordinates are noisy
or useless. They show why raw-gradient energy is the wrong weighting for
dismissing uncertainty after normalization.

Temporal checks then prevented a convenient replacement story. Weak input
groups could align positively at one lag but alternate at denser lags; they
did not form a uniformly persistent slow gradient. Fixed-state signal,
temporal persistence, and update allocation are separate measurements.

The estimator-clock audit also found that TS's output factor used eight
owner-local sequences per refresh, with nominal stationary EMA weight ESS72,
while SOAP used globally averaged training gradients. Larger training batches
improve the latter observation without automatically enlarging the former.
This qualifies an intrinsic “gradient statistics beat curvature statistics”
interpretation. Conversely, input covariance's clock was largely fixed in
tokens; an alleged hidden16-fold acceleration was not supported.

Evidence: [SNR and displacement](SIGNAL_NOISE_FINDINGS.md),
[persistence](PERSISTENCE_FINDINGS.md), [estimator clocks](estimator_clocks/NOTE.md).

## 2. Parameter-space changes can overstate or conceal functional changes

Fresh TS output-factor estimates produced directions with parameter cosine
about.70. On independent inputs their predictive-GN cosine was about.99.
The initial appearance of a large estimator bottleneck therefore weakened
when measured in predictions. It did not disappear: PD's response at matched
parameter norm was2.27–2.41times larger than TS's, and amplitude-normalized
comparisons retained meaningful shape uncertainty. The outcome justified
neither a new estimator nor a claim that estimator noise is harmless.

The same distinction appeared in attention. V-coordinate mean amplitudes
change under an exact V/O change of basis that preserves the model function.
Accounting for O removed the simple universal amplitude ordering, while
smaller joint V/O perturbations under PD survived. A separate finite-loss
test found no repeatable PD-specific extra benefit from the constant value
route. This closed the selected centering/bias-route explanation without
declaring mean structure irrelevant.

The observer's own proposed invariance explanation for geometry-aware decay
also failed algebraically. Ordinary scalar decay already respects the fixed
V/O gauge, while the helpful shaped decay generally does not. Better training
and greater coordinate covariance are different properties.

Evidence: [TS repeatability](ts_repeatability/REPORT.md),
[value gauge](value_gauge/REPORT.md), [value split](value_split/REPORT.md),
[geometry and decay](gauge_optimizer/REPORT.md).

## 3. Apparent interference is often overlap of useful prediction changes

A selected Muon step supplied an initially striking anomaly: auxiliary
updates helped alone but worsened loss when added after the body update.
The full body/head/embedding/norm factorial attributed roughly two thirds
of the interaction to embeddings, one third to the head, and very little
to normalization gains. An exact finite-logit decomposition explained
**96–98% by overlap of separate prediction changes**. A large nonlinear
transport defect was unnecessary at this state.

The overlap was not an identified disposable component. Algebraically
removing the embedding response parallel to the body retained55.5% of its
response energy but only7.5% of its label-linear descent. This was a diagnostic
projection, not a realizable tested parameter update. It illustrates why
minimizing interaction alone can remove useful progress.

Related checks reinforced and qualified that account:

- A broad selected-direction coupling mode aligned with descent; positive
  cross terms did not identify a special harmful channel.
- Absent-row embedding momentum could not explain the interaction on a bank
  with no such absent input rows.
- Early half-power PD had negative, beneficial total interaction, but its
  mixed nonlinear contribution was slightly adverse on fresh inputs. The
  benefit came from complementary additive changes. A favorable base-linear
  score of mixed logits did not imply favorable finite cross-entropy.
- Q/K changes helped alone and had positive symmetric attribution, while
  their last-added margin could be adverse. Attribution order and overlapping
  corrections matter; an adverse last-added score does not justify freezing
  the component or vetoing its possible long-run role.

Evidence: [body/auxiliary premise](body_aux/REPORT.md),
[factorial attribution](aux_partition/REPORT.md),
[coupling](coupling_archive/REPORT.md), [embedding support](embedding_clock/REPORT.md),
[early finite split](early_split/REPORT.md), [Q/K functional test](qk_function/REPORT.md).

## 4. Normalized weights create an endogenous relative step size

The existing SOAP-PD momentum×LR factorial retained a similar short-momentum
benefit after a43% increase in nominal body movement. That initially appeared
compatible with a simple optimizer-step clock. Saved-weight analysis exposed
another state variable: Q/K weight radii.

Across twelve saved states, later beta.8 Q/K radii were13–20% smaller than
beta.9 radii, with almost identical head-step norms. Their normalized turns
were **15–25% larger**. Raising LR43% grew the radii enough that later turns
increased only **9–19%**. Coherent outward radial motion was material, often
larger than squared-step growth, so a simple random-walk radius law was not
established.

A new CPU functional probe showed that this difference reaches predictions:
shorter beta increased isolated Q/K predictive KL by approximately1.7times
at the selected states, while higher LR had a smaller-than-naive squared-step
effect. This is a positive structural-to-functional connection. Its selected
Q/K-last usefulness prediction failed, and no causal share of the eventual
training improvement was identified. Later review correctly rejected using
that local failure as a universal veto of a radius-mediated history effect.

This is the clearest constructive alternative supplied by the observer:
**momentum and preconditioning may change effective feature movement through
learned scale, even when nominal LR and parameter-step norms are fixed.**
The general normalization principle is known; the measurements and their
connection to this project's factorial are the contribution here.

Evidence: [momentum×LR](momentum_lr/REPORT.md),
[angular/radial geometry](angular_clock/REPORT.md),
[prediction changes](qk_function/REPORT.md),
[dynamical synthesis](dynamics_synthesis/PEER_REVIEW.md).

## 5. Static suppression, feedback stability, and trajectory progress differ

The observer extracted a stronger comparison already present in eighteen
profiles: apply Muon and PD maps to the same lagged momentum at the same
state. PD's measured top16Ritz energy fraction was145–5,305times smaller;
absolute projected magnitude was also smaller. Thus additional covariance-
dependent redistribution is real within that conditional map comparison.

Two qualifications changed its interpretation. Muon's polar map already
reduced the stiff fraction122–757times relative to raw momentum. And the
saved lagged input M_s was not the fresh M_(s+1) used by the actual next
step: lagged maps were mostly uphill while actual next body steps descended.
The positive score a²/(2q) erased that sign distinction.

An exact counterexample showed that a small stiff output component does not
bound the derivative of the nonlinear map in that direction. The old archive
also already contained a differential stability experiment whose universal
instantaneous-edge prediction failed. Finite Newton–Schulz plus normalization
need not have a symmetric positive-semidefinite derivative. These results
prevent assigning static energy fractions a causal feedback interpretation.

The spectral audit preserved accurate total quadratic forms while showing
that48Lanczos nodes did not justify exact hard projectors onto a flat band.
The finite-history replay added another scale distinction: ordered-data
reweighting was small beside the common fixed-state gradient, but large beside
the much smaller actual momentum after cancellation. Its corpus/age cause
remained confounded.

Evidence: [same-state maps](momentum_maps/REPORT.md),
[map derivative](momentum_maps/DIFFERENTIAL.md),
[earlier stability experiment](linearized_edge/REPORT.md),
[spectral resolution](spectral_claims/NOTE.md), [replay](replay_projection/REPORT.md).

## 6. Successful schedule transfer survives; a precise rate law does not

The observer independently verified main-program controls rather than
claiming their training results as its own. Normalizing Muon's gradients
did not reproduce the whitening methods' large early short-momentum gain.
The longer-horizon records showed why that gain is phase dependent.

Momentum warmup improved the doubled-horizon endpoints by.021193 and.023944
for SOAP-PD and.026671 for PD against each own beta.9 reference. These are
two SOAP seeds plus one PD seed, not a fully crossed replicated study or a
universal schedule. The short-horizon warmup failure remains.

The completed progress audit found positive apparent training-curve leads
after beta matches. Estimated catch-up over states110→140 ranged.16–1.79
updates across the fixed smoothers. Only validation100/150 lies in the
common-beta, constant-LR interval. Explicit monotone curves with constant,
shrinking, and growing leads all reproduce those same four observations.
The endpoint benefit is clear; a precise constant head start or continuing
rate trend is not identified by those fixed-bank anchors.

Evidence: [clipping/horizon controls](clipping_connection/COMPLETION_ADDENDUM.md),
[warmup transfer and replication](warmup_transfer/README.md),
[progress lead](progress_lead/REPORT.md).

## 7. PD-top made the allocation issue concrete

Before reading its training outcomes, the observer examined main's new
root-clamping intervention. The ideal matched map is unchanged byR→cR;
absolute root factors above1 are partly a scale convention. For square/tall
full-column-rank matrices its input Gram is k²R²/tr(R²). Capping the tail
therefore changes final motion even where the root coefficients are unchanged.

In all four raw preflight states, PD-top/full-PD absolute top16 projection
energy ratios were.87–1.10, while their fractions differed15–58times. Full PD
had much greater total raw norm. After equal global normalization that
fractional difference is a real allocation difference, not an artifact.
Online per-matrix normalization cannot be reconstructed from the scalar file.
The intervention tests a relative metric profile, equivalently a floored
proxy metric; its outcome alone cannot uniquely separate “suppression” from
“tail amplification.”

The last interrupted turn saw one16M beta.8 PD-top arm complete at4.600305,
with the beta.9 counterpart not yet located and4M still running. No completed
paired PD-top training interpretation was made by this observer. That partial
external snapshot is not a new observer achievement.

Evidence: [PD-top interpretation and numerical checks](pdtop_interpretation/REPORT.md).

## Additional completed findings and bounded closures

These results are part of the full observer record, not omitted successes
or silently abandoned failures:

| Question | Finding and limit | Evidence |
|---|---|---|
| Does a simple attention-pooling account explain V curvature? | Derived and qualified F_in=XᵀAᵀRAX/T, separating routing from downstream error covariance. The simplified iid two-moment model failed its retained-fit check: only1/256cells met its coefficient relation within.10; none had fit residual≤.10. Absolute between coefficient b is not relative b/a. No population rejection or replacement preconditioner was claimed. | [Report](attention_pooling_geometry/REPORT.md) |
| Are larger internal V means an amplified useful constant route? | Gauge-aware output accounting weakened that amplitude story; smaller joint V/O movement survived. The selected constant-route usefulness effect did not replicate across banks. | [Geometry](value_step_geometry/REPORT.md), [gauge](value_gauge/REPORT.md), [finite split](value_split/REPORT.md) |
| Does SOAP have an extra rare-target signature? | The declared signature failed after independent progress calibration. Large interpolation controls prevented replacing it with a common-token mechanism. Global predictive gains remained. | [Report](prediction_profile/REPORT.md) |
| Does confidence explain the warmup ranking? | Every choice on the fixed coarse logit-scale grid was1. Fresh-bank short/long rankings reproduced, but no calibration correction was resolved. Finer effects were not excluded. | [Report](confidence_calibration/REPORT.md) |
| Does the whitening×momentum dose identify the mediator? | Main's interaction is real, but equal body norms conceal changed body and auxiliary prediction amplitudes. A scalar-shrink explanation was inadequate; a unique stiff-filter mechanism was not isolated. | [Report](dose_function/REPORT.md) |
| Can historical gradients supply a disjoint conditional-input test? | The proposed g1M/g4M remainder stopped before real arithmetic because historical sample nesting/source provenance was insufficient. BF16 interval methods were checked only synthetically. No cancellation result was inferred. | [Report](conditional_input/REPORT.md) |
| Can full input-curvature marginals be used retrospectively? | Full matrices survive for Q/K/V, while important O/up/down cases retain diagonals/scalars. The prior token-weighting and frame-diagonal tests do not exhaust full marginal geometry, but the archive does not support the desired comparison. | [Inventory](curvature_marginal_reset/HISTORY_REVIEW.md) |
| Can current dynamic feedback be reconstructed? | Older tensors and scalar stability experiments exist, but current16Mprofiles lack the state/operator sequence needed for a faithful online feedback reconstruction. | [Inventory](feedback_inventory/REPORT.md) |
| Why did a tiny-surrogate ordering reverse? | The fixed-bank subset and its complement favored different methods; archived per-window losses reconstruct the reversal. No shared physical mechanism with the SNR issue was established. This is development evidence; sealed confirmation data stayed unread. | [Note](surrogate_observation/NOTE.md) |
| Is Fisher half-factor propagation a justified replacement estimator? | Prior art and algebra supplied alternatives, but no universal variance advantage after inverse roots/polar mapping was established. No replacement was run. | [Note](gaussian_fisher_peer/NOTE.md) |

The [README](README.md) indexes the full source, numerical, literature,
qualification, failure, and independent-review record. Individual reports
preserve selected-state, sample, precision, and provenance limits.

## How the work pursued the goal

The work progressed from auditing explanatory claims to testing connections:

1. Read frozen execution sources alongside reports, checking what quantity
   each artifact actually contains and what data/clock generated it.
2. Reanalyze whole declared panels instead of selecting favorable layers,
   seeds, or checkpoints. Keep signs, norms, denominators, missing coverage,
   and failed predictions visible.
3. Use algebraic counterexamples and null calibrations for logical claims.
   These are cheap ways to distinguish an unsupported implication from an
   empirical question needing new measurements.
4. Where an archive lacked a consequential observable, perform bounded CPU
   measurements at saved states: fixed fresh inputs, actual saved writes,
   finite-loss decompositions, or predictive-curvature comparisons. These
   were experiments, not merely a rereading of summaries, but they were not
   training runs or new trajectories.
5. Use independent high-level/design/result reviews for major branches,
   including reviews that changed or closed the observer's own hypotheses.
   Preserve original failures and predeclared readings. Qualify engineered
   corrections explicitly rather than silently repairing evidence.

All new observer material stayed in `logs/observer_connections_20260928/`.
Observer computations used CPU only, with at most two numerical threads per
process. No observer training or GPU job was launched. Main training and its
queues were not modified, the separate `../last_layer` study was untouched,
and tiny-surrogate confirmation panels were not scored. Individual scripts
record runtimes; this synthesis does not claim a newly audited aggregate
compute total. Read-only scheduler checks distinguished live from historical
status without treating idle resources as permission for a new run.

## What changed in the main program, and what remains observer-only

The main journal's19:56entry explicitly cites the observer's angular-clock
and same-state momentum-map reports. It accepts that the whitening-dose
experiment demonstrates an alpha×beta interaction without identifying the
mechanism, and corrects the claim that only Muon passes stiff oscillation
into the step. Those are documented impacts of this observer work.

Other main corrections, including earlier spectral and greedy-step-rule
revisions, arose through the broader research/review process. The observer
checked and connected them; this synthesis does not claim sole authorship.
Likewise, main PD/SOAP/TS gains, warmup results and PD-top experiments must
not be credited as observer inventions. Some observer qualifications—especially
SNR, functional amplitude and interpolation resolution—remain documented
here without evidence that every main summary has incorporated them.

## Achievement and self-assessment

The scientific achievement is a more defensible explanatory framework,
several positive cross-artifact connections, and a smaller set of plausible
mechanisms. The strongest constructive link is learned radius affecting
relative Q/K motion and reaching predictions. The strongest measurement
correction is the SNR null. The strongest interpretation changes concern
useful overlap, map value versus feedback, and normalized allocation.

The remaining gap is causal and trajectory-level: no observer intervention
has shown that modifying one of these mechanisms yields a robust sustained
gain across seeds, horizons, or model regimes. The proposed mechanisms can
coexist, and selected-state evidence does not rank their contributions.

A limitation of the approach was spending many passes on related local
attribution questions. These produced useful constraints but diminishing
returns toward a new method. I also initially treated some component-last
usefulness checks too strongly; later peer discussion corrected that. The
work was stronger at rejecting explanations than at constructing and testing
a coherent replacement. It should be evaluated as scientific clarification
and hypothesis narrowing, not as delivery of the ultimate improvement goal.

At interruption, [post_switch_geometry](post_switch_geometry/HISTORY_INVENTORY.md)
contained only an inventory and design review for relating direct momentum-
buffer carry to persistent geometry. No protocol execution, tensor reduction,
model call or result was produced for it. Its source-level recurrence is
not an empirical finding about negligible memory or functional influence.
It remains explicitly unexecuted, with no automatic continuation after the
user cleared the goal.

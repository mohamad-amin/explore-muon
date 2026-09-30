# Observer: connections across saved evidence

**Stopped at the user's request on 2026-09-28 after the goal was cleared.**
Start with [the final synthesis](FINAL_SYNTHESIS.md) for what was learned,
how the work was done, its documented impact, and what remains unproven.
No robust new optimizer or architecture improvement was established.
The final post-switch proposal remained an inventory/design, without execution.
Historical next-step language below records the research sequence and does
not authorize continuation.

Started 2026-09-28. This directory belongs to the observer goal in Codex chat
01a0e93f-b944-7830-a874-5f300405abb8. All files created by this pass stay here.
The pre-existing `../observer_20260928/` notebook is separate work, read as
evidence and left intact. The main training code, queue and protocols are not
modified by this observer pass.

## Scope and resource contract

Understand phenomena across saved artifacts, investigate anomalies and competing
explanations, and develop falsifiable connections that might support robust
optimizer or architecture improvements. No improvement or novelty is presumed.
Initial computations use existing JSON, source and logs on CPU only, at most
two numerical threads per process. No GPU or scheduler allocation is used.
Main training allocations 2567578 (priv-g14) and 2618555 (g20) were confirmed
live at 2026-09-28 13:26 CDT and remain untouched.

## Initial questions

1. Do batch-size changes alter the statistical estimators' time scales enough
   to confound the interpretation of momentum and whitening gains?
2. Which conclusions about the flat curvature band are supported by saved
   Krylov measurements, as opposed to exact total quadratic forms?
3. Does the saved per-entry SNR evidence distinguish gradient noise from
   signal equalization as the mechanism of SOAP's improvement?

These are bounded retrospective analyses, not new training branches. Each
analysis records source paths, the scope of its claims, alternative
explanations, and a smallest discriminating measurement. Any substantial new
experiment needs its own decision argument and independent peer discussion.

## Progress accounting

The prior visible turn completed the historical project survey. This turn
continues the broader observer objective and does not count that survey as
completion of the research goal.

## First-pass results

Latest completed work: [warmup progress lead](progress_lead/REPORT.md) qualifies
allthree pairs and shows that sparse fixed-bank anchors admit constant,
eroding and increasing leads, while dense training means retain a positive
several-update advantage. [PD-top interpretation](pdtop_interpretation/REPORT.md)
connects the new main intervention to normalization: four raw preflight maps
have nearly equal absolute stiff energy underPD-top/fullPD despite15–58times
different fractions. All checks use CPU scalar/algebra work; both have
independent reviews. See STATE/SYNTHESIS for current decisions and limits.

Start with [STATE.md](STATE.md) and [SYNTHESIS.md](SYNTHESIS.md). The broad observer objective remains
active; no optimizer improvement or novelty is claimed.

- [Signal/noise and actual displacement](SIGNAL_NOISE_FINDINGS.md): analytic
  estimator calibration, independently checked simulation, and twelve saved
  frame tensors. [Figure](signal_noise_null.png).
- [Estimator clocks](estimator_clocks/NOTE.md): nine frozen runs; sampling
  budget distinguished from estimator age.
- [Spectral claims](spectral_claims/NOTE.md): 37 states; direct totals,
  Krylov-band stability and counterexamples to ranking by a fraction.
- [Independent review](signal_noise_peer/PEER_REVIEW.md) and
  [reweighting review](signal_noise_peer/REWEIGHTING_REVIEW.md).

The second pass completed [persistence analysis](PERSISTENCE_FINDINGS.md),
an [attention value-route analysis](attention_geometry/NOTE.md), a
[literature connection](dynamics_literature/NOTE.md), and a
[CPU four-corner counterfactual](body_aux/REPORT.md). All computations are
complete. The third pass completed [functional value scoring](value_split/REPORT.md),
[actual-step geometry](value_step_geometry/REPORT.md), a necessary
[V/O gauge qualification](value_gauge/REPORT.md), and the broader
[coupling archive analysis](coupling_archive/REPORT.md). The proposed next
discriminator concerned output-curvature estimator repeatability; the fourth
pass has now completed that [conditional and functional study](ts_repeatability/REPORT.md).
It also added [clipping/horizon outcomes](clipping_connection/COMPLETION_ADDENDUM.md)
and a [saved-window surrogate observation](surrogate_observation/NOTE.md).
Current next question and execution state are in STATE.md. These notes launch
no training intervention.

The latest pass closes the [body/auxiliary factorial](aux_partition/REPORT.md):
embeddings account for most interaction, and ordinary prediction overlap
explains96–98% of it. The [input-support inventory](embedding_clock/REPORT.md)
excludes absent-row momentum as the cause on one fresh bank. A new
[same-state momentum-map audit](momentum_maps/REPORT.md) strengthens the
evidence for PD's additional stiff suppression while separating map values
from the [feedback derivative](momentum_maps/DIFFERENTIAL.md). All18states,
source snapshots, numerical checks and an independent review are retained.
These computations are complete, CPU-only, and establish no new training
improvement. The full observer objective remains active.

The next pass inventories[retained feedback tensors](feedback_inventory/REPORT.md)
and analyzes an existing[momentum×body-LR factorial](momentum_lr/REPORT.md).
The~.10short-momentum advantage survives43%more nominal body movement, with
an early interaction retained and no causal phase-clock claim. Its
[trajectory figure](momentum_lr/contrasts.png), source checks, original
qualification stop and independent results review are preserved. No replay,
new training or GPU work followed the inventory.

The latest pass restores an overlooked[prior differential experiment](linearized_edge/REPORT.md)
and reduces the[saved fixed-weight replay](replay_projection/REPORT.md).
Recent-data weighting is small beside the common gradient but large beside
actual momentum in the retained coordinates. Ordered variation and a known
data boundary are analyzed with an exact finite-set calibration; causal
corpus/recency claims remain open. Existing observer notes now explicitly
acknowledge the old failed instantaneous-edge prediction. CPU arithmetic only.

A further[coordinate-covariance qualification](gauge_optimizer/REPORT.md)
connects the V/Ogauge to PD normalization and decay, then rejects a broad
invariance-restoration explanation before running a saved-model stress test.
One fixed algebraic example and primary precedents are retained; no new
normalization, balancing or training method follows.

The latest[direct prediction-profile probe](prediction_profile/REPORT.md)
scores ten saved checkpoints on a fixed fresh panel, using independently
calibrated Muon-progress references. The declared extra rare-target SOAP
signature fails; substantial interpolation controls leave the mechanism
unresolved. All token losses and[phase-contrast figures](prediction_profile/run1/phase_contrasts.png)
are retained. The320score forwards completed in136seconds onCPUonly.

The latest [dose-function probe](dose_function/REPORT.md) scores base/body/
auxiliary/full corners at eight quarter-/half-power PD states. Equal parameter
norms conceal predictive-amplitude changes, and early body–auxiliary loss
interaction changes sign. An exact post-hoc CE/KL identity establishes a
label-relevant mixed-logit response, with explicit limits and independent
review. The 256 score forwards completed in 139 seconds on CPU. Main dose
and warmup results are external context, not observer runs.

The follow-up [exact finite split](early_split/REPORT.md) is complete in
207 seconds on CPU. Negative early half-power interaction reproduces, but
complementary additive prediction changes explain its mean benefit; the
finite mixed response is slightly adverse and inconsistent across fresh
banks. P1 and P3 fail, while P2 passes only as relative improvement over
quarter power. The proposed useful-nonlinearity branch closes without a
subgroup factorial or sample expansion. The independently checked KL
correction also reverses the tempting J interpretation on all eight original
inputs. See [results review](early_coupling_peer/RESULTS_REVIEW.md).

The new [Q/K angular-step comparison](angular_clock/REPORT.md) connects
normalization, weight-radius growth and momentum. Across the fixed twelve
saved states of the earlier LR×beta factorial, larger LR is substantially
absorbed by growing radii; shorter momentum produces larger normalized Q/K
turns under equal nominal norms. Both declared readings pass, with every
head retained and no causal rate claim. The saved-weight pass took 36.67
seconds on two CPU threads, with no model/GPU call. See its
[figure](angular_clock/run1/angular_ratios.png),
[independent review](angular_clock/RESULTS_REVIEW.md), and the two distinct
[next-direction discussions](next_direction_peer/DECISION.md).

The separate [confidence discriminator](confidence_calibration/REPORT.md)
is complete: six saved schedule endpoints, two fixed banks, five logit
scales, opposite-bank selection/scoring. Every selection is a=1. The fresh
short-/long-horizon schedule ranking reproduces, but the coarse grid resolves
no finite confidence correction and does not exclude finer effects. The
193 total forwards finished in 227.91 seconds on CPU. Original preflight
and arithmetic-reduction corrections are preserved; the fixed panel is
closed without expansion. See its [review](confidence_calibration/RESULTS_REVIEW.md)
and [figure](confidence_calibration/run1/confidence_gaps.png).

A [functional-availability inventory](angular_clock/FUNCTIONAL_AVAILABILITY.md)
then checked whether retained profiles can attribute the angular result to
Q/K prediction changes. They cannot: the low-LR states retain whole-body
totals, the high-LR states lack matching profiles, and older Grams describe
different directions/states. No model/checkpoint tensor call or new
functional experiment followed that inventory.

After a fresh priority discussion, the [Q/K functional bridge](qk_function/REPORT.md)
measured the missing actual-update terms at four fixed 46→47 states. Both
factorial movement patterns pass in isolated and updated backgrounds, but
the selected Q/K-last usefulness condition fails. Q/K remains helpful alone
and has positive symmetric credit; ordinary additive prediction overlap
explains the order dependence. The 257 total forwards took 287.14 seconds
on two CPU threads. The fixed bridge is closed without expansion; see its
[independent review](qk_function/RESULTS_REVIEW.md) and
[figure](qk_function/run1/qk_function.png).

The subsequent [priority reset](priority_reset_after_qk/REVIEW.md) recommends
stepping back from further component localization. It proposes a gated
saved-artifact question about gradient repeatability after addition of
momentum history, with explicit historical-membership, clipping and BF16
limits. No predictor outcome, new scoring or optimizer intervention followed
that discussion.

The [conditional-input qualification](conditional_input/REPORT.md) stopped
before real gradient arithmetic: the historical GN2 sampling/source contract
is not preserved well enough to certify the proposed disjoint remainder.
The frozen training recurrence and a synthetic BF16 interval design are
retained, without empirical cancellation or SNR claims. No replay repaired it.

Separately, the [completed main PD warmup pair](warmup_transfer/README.md)
passes its original .01-gain criterion by −.026671 NLL at the doubled
horizon. Sources and per-step clocks match; the early lead narrows but
persists after beta matches. The observer only read and plotted saved
scalars. Its [review](warmup_transfer/REVIEW.md), [trajectory](warmup_transfer/trajectory.png)
and reproducible extraction remain separate from the unfinished fresh-seed
SOAP-PD pair and from the unqualified gradient-membership question.

The [dynamical synthesis](dynamics_synthesis/PEER_REVIEW.md) separates a
retained head start, radius-mediated relative rates and moving coupled
feedback. A [marginal inventory](curvature_marginal_reset/HISTORY_REVIEW.md)
clarifies which full input factors were actually saved and which approximations
were tested. The [attention-pooling analysis](attention_pooling_geometry/REPORT.md)
then qualifies an exact routing/error identity and its finite-context
coefficient correction, but finds the simple iid closure unsupported by all
256 archived V fits. No attention/model capture or optimizer arm followed.

The [SOAP-PD warmup replication](warmup_transfer/SOAP_REPLICATION_REVIEW.md)
has now completed at −.023944 NLL relative to its constant-.9 reference.
Together with the PD transfer, both predeclared cohort pairs pass. New
replication files preserve the earlier PD-only extraction and its dated
incomplete-pair status rather than rewriting those records.

[The resulting priority decision](dynamics_synthesis/DECISION.md) accepts
that local component helpfulness cannot veto a long-run mechanism. The next
candidate is a trajectory-lead comparison with an explicit cadence and
interpolation-resolution check; it has not been executed or turned into a
new training/evaluation commitment.

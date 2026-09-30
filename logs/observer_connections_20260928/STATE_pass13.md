# Observer current map

Updated 2026-09-28 after the completed six-endpoint confidence discriminator.
The broad research goal remains active. No robust new optimizer or architecture
improvement is claimed. Earlier maps and interpretations remain in numbered
`STATE_pass*.md` and `SYNTHESIS_pass*.md` snapshots.

## Current objective and boundary

Connect the saved dynamics, data, activation and curvature evidence; test
anomalies that could support a robust improvement. All observer output stays
in this directory. Main code, protocols, frozen results, queues and the separate
`../last_layer` study remain untouched. Use CPU only, at most two numerical
threads; the tiny-surrogate branch also uses smaller GPUs, so they are not
presumed uncontested. Its sealed confirmation panels remain unread.

## Latest decisive result

[confidence_calibration/REPORT.md](confidence_calibration/REPORT.md) records
the completed fixed six-endpoint/five-scale test. Every model chooses a=1
on both fitting banks. The short- and long-horizon warmup rankings reproduce
on fresh inputs: warmup minus beta .8 is +.042946 and −.012366 NLL, with
the expected sign on each bank. All cross-fitted corrections are exactly zero.

This is the predeclared **coarse grid resolves no finite correction** outcome,
not proof that finer confidence adjustments are irrelevant. Mean derivatives
at a=1 remain negative; no fitted scale or unmeasured gain is inferred from
them. Close this panel without finer scales, added endpoints or more inputs.

The run completed 192 scientific forwards and one repeat in 227.91 seconds
on two CPU threads, under its 600-second cap; session 55327 exited 0.
Independent design, source/pairing, implementation and result reviews pass.
An unavailable initial offset proposal was corrected before scoring, and an
arithmetic-only analyzer correction removed ~1e−15 false differences between
identical a=1 arrays. Originals, hashes and unchanged scientific verdicts
are retained. No optimizer/architecture improvement follows.

## Latest positive structural connection

[angular_clock/REPORT.md](angular_clock/REPORT.md) supplies a new connection
between normalization, momentum and relative step size. In the existing
SOAP-PD beta {.8,.9} × LR {.028,.04} factorial, all twelve retained adjacent
state pairs were analyzed without model calls. Both declared readings pass
in every later Q/K × setting cell:

- A 43% LR increase raises median matched-head normalized turns only 9–19%
  at steps 46/83. Head radii grow 21–31%, while tangent writes grow ~42–43%.
- Shorter momentum raises those turns 15–25% despite head-step norms within
  about .3%; its radii are 13–20% smaller and tangent writes nearly unchanged.
- At step 9 these differences are not developed: LR changes turns 34–36%,
  while the beta contrast is only ~1%. Coherent outward radial drift is
  material; an eta*sqrt(t) diffusion account is not justified.

The primary readout is median paired-head chord ratio, separately for all
64 Q and 64 K heads. All individual exceptions are retained. This measured
weight-space motion is not a prediction-space distance or a causal mediator.
It explains why the previous nominal-arc comparison does not exclude an
endogenous relative-step explanation of the LR-insensitive phase. It does
not establish a new optimizer, a universal step clock or an intervention.

The 36.67-second CPU pass used two threads, no model/GPU calls, and session
73509 exited 0. Numerical identities, source/scalar hashes and paired
contrasts passed independent review. Close this fixed comparison without
more states, fitted power laws or a clock sweep.

## Preceding decisive result: helpful mixed nonlinearity fails

[early_split/REPORT.md](early_split/REPORT.md) closes the proposed helpful
mixed-response premise. At early half-power PD on eight fresh inputs,
total interaction −.047696 splits into additive overlap −.053006 and finite
mixed loss +.005310. The mixed effect changes sign between fresh banks.
The predeclared absolute-helpfulness and phase predictions fail; the
half-minus-quarter comparison passes as a smaller adverse mean effect.

On the original eight inputs, J is negative on every sequence but finite
mixed loss is positive on every sequence. A positive predictive-KL correction
reverses the tempting finite-benefit interpretation of the base-linear J.
All old corner values reproduce exactly, all scalar identities pass to
roughly 5e−15, and independent result review agrees. The probe completed
256 forwards in 207.47 seconds on two CPU threads; session 97830 exited 0.
No subgroup factorial, added inputs, states or fitted rescaling follows.

This extends the earlier positive-overlap finding: additive prediction
interaction can also be complementary. Neither local sign identifies the
mediator of the main training gain. No new training method is established.

## Preceding dose result, preserved with its finite-effect qualification

[dose_function/REPORT.md](dose_function/REPORT.md) records eight own states:
PD alpha {1/4, 1/2} × beta {.9, .8}, steps 9→10 and 46→47, with eight shared
inputs and four base/body/auxiliary/full corners at every state.

- The main training dose prediction holds: beta gains are −.08627 and −.12360,
  with final interaction −.03733. This supports the hyperparameter interaction,
  without uniquely identifying stiff-oscillation filtering as the mediator.
- Half/quarter body predictive-KL ratios are .845/1.085 early and .716/.684
  midway (beta .9/.8). Body parameter norms differ by less than .04% between
  powers. Uniform functional shrinkage fails in the early beta-.8 cell;
  later amplitude effects remain possible.
- Auxiliary KL ratios are .461/.400 early and 1.586/.793 midway despite
  identical auxiliary settings. Changed functional auxiliary steps are part
  of the changed trajectory, not an invalidation of the matched treatment.
- At step 9, body–auxiliary loss interaction is positive for quarter power
  (+.087/.061) and negative for half power (−.047/−.035), each sign on every
  input at both betas. At step 46 all four interactions are positive.
- A post-hoc exact CE/KL identity gives
  `J = I_CE − (KL_full − KL_body − KL_aux)`
  `  = mean[(p0−e_y)·(z_full−z_body−z_aux+z0)]`.
  Early half-power J is −.03275/−.02970, negative on all eight inputs; quarter
  power gives +.01563/+.03153. This is label-relevant model-output nonadditivity.
  It is NOT the earlier finite mixed-loss term CE(full)−CE(additive logits).
  No overlap fraction, residual-Hessian magnitude, subgroup attribution or
  training mediator follows from J alone.

The early sign change precedes the strongest training alpha×beta interaction
and exists at both betas; it cannot by itself explain the final ordering.
The prior late-Muon result remains valid in its state-local scope.

## Next scientific decision

The early mixed-response, fixed angular comparison and confidence panel are
closed at their declared scopes. Independent discussions and the priority
decision are in `next_direction_peer/`. The angular result establishes a
relative weight-motion difference, but not its functional importance.
[The read-only inventory](angular_clock/FUNCTIONAL_AVAILABILITY.md) found
that the saved LR-.028 profiles contain
actual next-body totals but not Q/K-specific slopes, response tensors or
coupling blocks; the fixed LR-.04 arms lack those profile records. Historical
direction Grams concern other states/maps. Therefore the actual Q/K
functional contribution cannot be reconstructed from the retained totals,
and parameter-energy shares must not be substituted for it. A separately
designed measurement would be needed before a memory/radius intervention.
No new scoring or training is live.
The all-one confidence outcome is not permission to make its grid finer.
The latest short-horizon warmup has completed at 4.515206, missing its
within-.02-of-beta-.8 prediction; the longer warmup remains positive at
3.962814. This is evidence of horizon sensitivity, not a causal identification
of optimizer steps as the relevant clock.

Do not simply enlarge the completed eight-state panel, fit a step rescaling,
freeze a group, or equate a favorable one-step interaction with a training
rate. Any new factorial/phase test needs its own bounded decision, numerical
checks and explicit prediction. The sealed head-initialization study is out
of scope. Earlier closed branches remain closed unless a distinct premise
and comparison are recorded.

## Decision-relevant constraints from completed work

| Evidence | Consequence |
|---|---|
| [SNR audit](SIGNAL_NOISE_FINDINGS.md) and saved-frame reweighting | The clipped/extrapolated statistic can report near-100% signal energy under pure noise. Raw gradient energy does not characterize where the actual update moves. |
| [Spectral audit](spectral_claims/NOTE.md) and [older EOS audit](linearized_edge/REPORT.md) | Exact totals do not resolve hard low-curvature bands. A prior differential measurement already failed a universal instantaneous-edge prediction; finite NS need not have a symmetric/PSD derivative. |
| [Persistence](PERSISTENCE_FINDINGS.md) | Weak input groups align at one lag but alternate at denser lags; they are not a demonstrated steady slow gradient. |
| [Body/auxiliary probe](body_aux/REPORT.md) | Same-bank versus cross-bank sampling explained the gradient-sign discrepancy better than auxiliary scope. |
| [Value split](value_split/REPORT.md) and [gauge qualification](value_gauge/REPORT.md) | Larger internal value means are not invariant functional routes. No repeatable extra PD benefit from the constant route was established. |
| [Coupling archive](coupling_archive/REPORT.md) | Positive selected-direction coupling is broad and aligned with useful descent; it does not identify an expendable channel. |
| [TS repeatability](ts_repeatability/REPORT.md) | Parameter-map variation (~.70 cosine) is much smaller functionally (~.99 cosine), with an important amplitude qualification. No estimator bottleneck was established. |
| [Late auxiliary factorial](aux_partition/REPORT.md) | At Muon 4M183→184, embeddings account for about two thirds of interaction and exact additive prediction overlap explains 96–98%. This does not generalize automatically to early PD states. |
| [Embedding support](embedding_clock/REPORT.md) | Absent-row Adam movement cannot explain the effect on a fresh bank with none of those input rows. |
| [Momentum maps](momentum_maps/REPORT.md) | Same-state PD maps strongly suppress measured stiff energy, but lagged maps and actual next steps have different time indices and often opposite slope signs. |
| [Momentum×LR](momentum_lr/REPORT.md) | A ~.10 short-momentum advantage survived 43% more nominal body movement; an early interaction remains and no causal phase clock was identified. |
| [Q/K angular comparison](angular_clock/REPORT.md) | Radius growth absorbs much of the higher LR; shorter beta yields larger normalized-weight turns under equal step norms. This supplies an endogenous-rate alternative, not a training mediator. |
| [Confidence discriminator](confidence_calibration/REPORT.md) | All six endpoints choose a=1 on both banks; original horizon-dependent ranking reproduces. The coarse grid resolves no correction and does not exclude finer confidence effects. |
| [Fixed-weight replay](replay_projection/REPORT.md) | Ordered-data reweighting is small beside the common gradient but large beside actual momentum. Exact permutation calibration prevents an inflated same-row attribution claim. Corpus and trained-model age remain confounded. |
| [Gauge/decay](gauge_optimizer/REPORT.md) | Scalar decay already commutes with the fixed gauge; helpful shaped decay need not. Restored gauge covariance is not an explanation of its benefit. |
| [Prediction profiles](prediction_profile/REPORT.md) | The declared extra rare-target SOAP signature failed after independent progress calibration; Muon interpolation controls and bank variation are material. Target frequency is not hidden-coordinate SNR. |

Further provenance, estimator clocks, literature notes and original failure
records are indexed in [README.md](README.md) and [SYNTHESIS.md](SYNTHESIS.md).
The broad objective has not been reduced to any one of these diagnostics.

## Execution state

All observer model computations are terminal. Latest completed probe:

- `confidence_calibration/run1`, tool session **55327**, exit 0, CPU two
  threads. 192 scientific forwards plus one repeat, 227.91 seconds against
  forecast 405.67 and cap 600. Repeat exact, scale-derivative identity max
  5.45e−14, FP64/FP32 CE max 3.50e−5 within 5e−5. All six models select one.
  Crossfit analysis/review complete; original differing-reduction outputs
  preserved under `run1/initial_reduction/`, correction proof in
  `run1/reduction_repair.json`. No fresh model evaluation for the repair.

Earlier completed computations:

- `angular_clock/run1`, tool session **73509**, exit 0. Saved-weight arithmetic
  only: 12 pairs, 576 matrix rows, 1536 head rows, 36.67 seconds, two threads.
  First-pair forecast 22.62 seconds underestimated later cold reads; actual
  remained below the 180-second cap. Both declared readings pass. Independent
  results review reproduces all 24 contrast medians and checks 100 source and
  40 scalar-input hashes; no additional checkpoint tensor load was needed.
- `early_split/run1`, tool session **97830**, exit 0. 256 score forwards,
  207.47 seconds, CPU two threads. Forecast 315.91 seconds; cap 600 seconds.
  Original corner reproduction errors are zero; maximum scalar-identity
  error 5.14e−15. Declared P1/P3 fail; P2 passes only as relative improvement.
  Implementation and independent result review are in `early_coupling_peer/`.

Previous completed model probe:

- `dose_function/run1`: complete, 139.32 s, 256 score forwards plus one
  restoration check, two CPU threads; session 79074, PID 1950734, exit 0.
- Forecast 287.82 s under the 600 s boundary. Strict corner equality and
  restoration passed; FP64/FP32 CE max error 2.32e−6. One hundred archived
  scientific-source hashes matched metadata. Independent implementation and
  results reviews are in `dose_interaction_peer/`.
- The later mixed-logit reduction used saved scalars only. No extra model
  call, sample, fitted rescaling or endpoint was added.
- Previous `prediction_profile/run1` completed in 136.03 s, 320 score forwards,
  session 28000, exit 0. All its original samples/controls remain unchanged.

Current scheduler lookup initially failed because sandbox DNS could not
resolve the Slurm host. A read-only query outside that restriction succeeded:
allocations 2567578 (priv-g14) and 2618555 (g20) are running. This was a lookup
failure, not evidence of terminated jobs, and no job was restarted or changed.

Main-program external evidence: the quarter-power dose pair and both original
warmup arms are complete. The 2× run ends at 3.962814, below constant beta .8
(3.9740) and .9 (3.9840), meeting its numerical target in one seed. The 1×
run ends at 4.515206, +.0428 above beta .8, failing its target. At 17:39 CDT,
scheduler steps and current pipeline records show PD warmup replication
training on priv-g14 (about 120/184) and second-seed SOAP-PD qualifying on
g20, followed by their beta-.9 references. SSH authentication failed; scheduler
and continuously written per-step records supplied the live evidence instead.
The small-surrogate branch has no jobs; its latest seed test failed and its
confirmation panels remain unscored. These are external, not observer runs.

## Progress and completion audit

The preceding goal turn made progress through the completed twelve-state
angular discriminator. This continuation completed a separately reviewed
direct predictive confidence test: it reproduces the horizon-dependent
ranking on fresh inputs, resolves no finite calibration correction in the
fixed family, and closes that panel. The next action is to qualify the
functional relevance of the positive angular observation from existing
artifacts before another intervention. The full goal remains incomplete;
neither completion nor a research blocker is claimed.

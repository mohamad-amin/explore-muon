# Observer current map

Updated 2026-09-28 after the completed exact early-interaction split.
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

The early mixed-response branch is closed. Independent high-level discussions
in `next_direction_peer/` are reviewing distinct retained-evidence questions,
including normalization/weight evolution and the new momentum phase result.
No new model probe, training arm or resource commitment has been made.
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
| [Fixed-weight replay](replay_projection/REPORT.md) | Ordered-data reweighting is small beside the common gradient but large beside actual momentum. Exact permutation calibration prevents an inflated same-row attribution claim. Corpus and trained-model age remain confounded. |
| [Gauge/decay](gauge_optimizer/REPORT.md) | Scalar decay already commutes with the fixed gauge; helpful shaped decay need not. Restored gauge covariance is not an explanation of its benefit. |
| [Prediction profiles](prediction_profile/REPORT.md) | The declared extra rare-target SOAP signature failed after independent progress calibration; Muon interpolation controls and bank variation are material. Target frequency is not hidden-coordinate SNR. |

Further provenance, estimator clocks, literature notes and original failure
records are indexed in [README.md](README.md) and [SYNTHESIS.md](SYNTHESIS.md).
The broad objective has not been reduced to any one of these diagnostics.

## Execution state

All observer model computations are terminal. Latest completed probe:

- `early_split/run1`, tool session **97830**, exit 0. 256 score forwards,
  207.47 seconds, CPU two threads. Forecast 315.91 seconds; cap 600 seconds.
  Original corner reproduction errors are zero; maximum scalar-identity
  error 5.14e−15. Declared P1/P3 fail; P2 passes only as relative improvement.
  Implementation and independent result review are in `early_coupling_peer/`.

Latest completed model probe:

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

The preceding orientation turn supplied new evidence: the 1× warmup had
completed and failed its target, changing the phase-schedule interpretation;
replication and surrogate states were verified. This continuation records the
already completed early-split result and changes the next action by closing
its useful-nonlinearity premise. The full open goal remains incomplete;
neither completion nor a research blocker is claimed.

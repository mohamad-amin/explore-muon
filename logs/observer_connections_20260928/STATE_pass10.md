# Observer current map

Updated 2026-09-28 after the completed whitening-dose functional probe.
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

The strongest new lead is early nonlinear body–auxiliary coupling. A useful
follow-up must distinguish ordinary additive prediction overlap from the
full mixed-logit loss effect, and localize the response under a stated
hypothesis before any optimizer prescription. The exact four-corner J does
not settle either question. A fresh independent review is requested in
`early_coupling_peer/`. The accepted follow-up is now running in `early_split/`:
the exact additive-logit/mixed-loss split at beta .9, alpha 1/4 and 1/2,
steps 9 and 46. It reproduces eight old inputs and scores eight fixed fresh
inputs. Subgroup attribution remains deferred until finite usefulness is known.

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

Earlier observer computations are terminal. Current live probe:

- `early_split/run1`, tool session **97830**. Reported PID 3 is inside the
  sandbox PID namespace, not a host PID. Poll this tool handle before treating
  an observation timeout as termination; never restart from a stale status.
- 256 planned score forwards, CPU two threads, 600-second checked boundary.
  First-sequence forecast 315.91 seconds. Initial old-corner reproduction
  errors are zero and scalar identity errors are below 3e−15. Fresh scientific
  results have not been interpreted. Protocol and independent review are saved.

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

Main-program external evidence: the quarter-power dose pair is complete.
The 2× momentum-warmup run completed at 3.962814, below constant beta .8
(3.9740) and .9 (3.9840), meeting its recorded numerical target in one seed.
The 1× arm must be checked live before using its results; old notes do not
establish its state. These are main-program results, not observer runs.

## Progress and completion audit

The preceding research turn made progress: a qualified eight-state functional
probe exposed an early label-relevant mixed response and changed the next
scientific question. The interruption affected bookkeeping, not execution;
the probe was already complete. The full open goal remains incomplete, and
neither completion nor a research blocker is claimed.

# Observer current map

Updated 2026-09-28 after the conditional-input provenance gate and PD warmup transfer.
The broad observer goal remains active. No robust new optimizer or architecture
improvement is claimed. Earlier maps/interpretations remain in numbered
`STATE_pass*.md` and `SYNTHESIS_pass*.md` snapshots.

## Objective and boundary

Connect saved dynamics, data, activations, curvature and intervention evidence;
investigate anomalies that could support a robust improvement. All new output
stays in this observer directory. Main code/protocols, frozen results, queues
and the separate `../last_layer` study remain untouched. CPU only, at most two
numerical threads; no training or GPU jobs. Tiny-surrogate confirmation panels
remain unread. Historical launches and idle resources authorize nothing new.

## Latest decisions and completed external evidence

[conditional_input/REPORT.md](conditional_input/REPORT.md) closes the proposed
saved disjoint-gradient construction at its historical membership gate.
The GN2 records identify arms/step/input labels but do not freeze the original
sampling calls or executed producer; the available producer was edited later.
Current-code nesting is plausible but unqualified. No real gradient tensors
were compared, no cancellation statistic was computed, and no model replay
was used to repair the archive. Frozen training sources confirm the intended
unnormalized beta=.95 recurrence, but do not resolve diagnostic membership
or full-model clipping. The precision note and synthetic interval checks
remain a mathematical design for future qualified data, not an empirical result.

[warmup_transfer/README.md](warmup_transfer/README.md) independently records a
new completed **main-program** result: PD warmup ends at 4.017883 versus
constant beta .9 at 4.044554, difference −.026671, passing the original ≥.01
gain criterion. The pair is matched in initialization, data, source bytes,
per-step LRs/tokens and L40S hardware; only momentum scheduling differs.
The validation lead narrows from −.116697 at update 50 to −.026671 at 184,
persisting after both schedules use beta .9 from update 93. This is one
paired PD seed at 2× horizon, not evidence of superiority to a PD beta-.8
reference (absent), a universal schedule or an identified mediator. Fresh-seed
SOAP-PD still awaits its completed constant reference; the joint two-pair
prediction is unresolved. No new observer training was launched.

## Previous functional result: angular pattern reaches predictions, conditional helpfulness fails

[qk_function/REPORT.md](qk_function/REPORT.md) covers the four existing SOAP-PD
LR {.028,.04} × beta {.9,.8} own states at 46→47. Two fixed fresh eight-sequence
banks receive base/QK16-weights/rest68/full-next corners. Rest includes Q/K/RMS
gains, other body matrices, embeddings and head. No direction reconstruction,
weight canonicalization, gradient, optimizer or fitted scale is used.

- **P1/P2 pass:** shorter beta increases isolated Q/K KL by pooled 1.725/1.755
  at the two LRs, and conditional KL after the rest by 1.453/1.679. Higher LR
  increases isolated KL 1.397/1.421 and conditional KL 1.252/1.447, below the
  nominal squared-step multiplier 2.040816. All required bank signs/bounds hold.
- **P3 fails:** at beta .8, adding Q/K after the rest changes NLL by +.001102
  at LR .028 (bank signs differ) and +.007604 at LR .04 (both adverse). Neither
  meets the declared both-bank-negative, pooled ≤−.005 condition.
- **Partial positive preserved:** Q/K helps alone in every state/bank. Its
  symmetric two-order credit is also positive everywhere. That alternative
  attribution convention was retained descriptively, not substituted for P3.
- Positive loss interaction is almost entirely ordinary additive prediction
  overlap. Pooled interaction is .00937–.02173; mixed loss is −.000182 to
  +.000229. Finite Q/rest response correlations are .605–.707. These are finite
  logit covariances, not tangent GN or causal shares of a training gain.

The structural-to-functional connection is real at these own states; radius
is not isolated as its cause. The chosen actual-background usefulness premise
fails. Close this fixed bridge without more heads/states/banks/rescaling or a
complement factorial. An adverse last-added margin does not make Q/K useless
or imply that freezing it would improve a trajectory. Full steps still improve
pooled loss at every selected state. Independent results review agrees.

## Next scientific decision

A broad independent [priority reset](priority_reset_after_qk/REVIEW.md) is complete.
The purpose is to compare substantive remaining anomalies/artifact families,
not continue local four-corner attribution by default. The Q/K result adds a
real functional constraint but no optimizer prescription. The fixed confidence
grid, useful early mixed-response lead and earlier auxiliary/mean-route panels
remain closed at their scopes. No new model scoring or training is live.

Its conditional-input candidate remains an open hypothesis, but the selected
two-file archive route is now unqualified and closed without numerical
interpretation. Do not silently substitute overlapping gradients, infer
independent samples from labels, add an EMA factor, or replay fresh model
gradients to repair it. A contemporaneous execution/sample record could
reopen that specific gate. Absolute perturbations are unchanged by adding
the same fixed history; relative/directional repeatability can change.

The next high-level task is an evidence synthesis across distinct levels:
raw fixed-state gradients, conditioned optimizer input, nonlinear maps and
adaptive radii/statistics, finite predictive movement, and trajectory progress.
The positive PD schedule transfer coexists with failed selected-step
component-usefulness tests and does not identify their mediator. No strong
new optimizer intervention is currently qualified. This is a priority decision,
not a blocker of the broader observer goal or a reason to repeat closed panels.

Any substantial new branch needs its own missing premise, competing account,
bounded comparison and independent high-level discussion. No extra LR, smaller
Q/K step, finer confidence grid or chosen layer follows from the failed local
criterion. The full goal remains broader than these selected-state tests.

## Recent completed connections

| Evidence | What it established and what remains open |
|---|---|
| [Q/K angular comparison](angular_clock/REPORT.md) | In twelve saved states, larger LR is partly absorbed by larger radii; shorter beta gives 15–25% larger late normalized Q/K turns under equal step norms. Coherent radial drift makes a simple diffusion law inadequate. |
| [Functional availability](angular_clock/FUNCTIONAL_AVAILABILITY.md) | Existing low-LR profiles have only whole-body totals; high-LR profiles and Q/K-specific response terms were absent. The new four-state probe supplied distinct missing information. |
| [Confidence discriminator](confidence_calibration/REPORT.md) | All six schedule endpoints choose a=1 on both banks. Fresh short/long warmup gaps +.042946/−.012366 reproduce, but the coarse grid resolves no correction and does not exclude finer effects. Closed without expansion. |
| [Early exact split](early_split/REPORT.md) | Negative early half-power interaction is additive overlap −.053006 plus mixed loss +.005310 on fresh inputs. Helpful finite nonlinearity fails its declared premise; negative base-linear J was insufficient. |
| [Dose functional probe](dose_function/REPORT.md) | The main alpha×beta training interaction is positive evidence, but equal body norms conceal changing body and auxiliary functional amplitudes. No unique stiff-filtering mediator follows. |
| [Momentum×LR](momentum_lr/REPORT.md) | A ~.10 short-window advantage survives 43% more nominal body movement. The angular and functional results now show why this is not an equivalent test of relative or predictive movement. |
| [Conditional-input gate](conditional_input/REPORT.md) | Historical nesting is not established for the proposed g1M/g4M remainder. Precision bounds are qualified synthetically, but no real cancellation statistic is inferred. |
| [PD warmup transfer](warmup_transfer/README.md) | A clean existing main-program pair gains −.026671 at 2× horizon. The early benefit persists after beta matches; one seed, no constant-.8 PD comparison, no causal phase clock. |

## Constraints from earlier completed work

| Evidence | Decision-relevant constraint |
|---|---|
| [SNR audit](SIGNAL_NOISE_FINDINGS.md) | The clipped/extrapolated statistic can give near-100% signal energy under pure noise. Raw gradient energy does not describe where normalized updates move. |
| [Spectral audit](spectral_claims/NOTE.md) | Accurate total quadratic forms do not resolve exact hard low-curvature projectors from 48 Lanczos nodes. One-step quality is not a training rate. |
| [Older EOS audit](linearized_edge/REPORT.md) | An earlier differential experiment already failed a universal instantaneous-edge prediction. Finite NS need not have a symmetric/PSD derivative. |
| [Persistence](PERSISTENCE_FINDINGS.md) | Weak input groups align at one lag but alternate at denser lags; a steady slow-gradient story is unsupported. |
| [Body/auxiliary probe](body_aux/REPORT.md) | Same-bank versus cross-bank sampling explained the gradient-sign discrepancy better than auxiliary scope. |
| [Value split](value_split/REPORT.md), [gauge qualification](value_gauge/REPORT.md) | Internal value means are not invariant functional amplitudes. No repeatable extra PD benefit from the constant route was found. |
| [Coupling archive](coupling_archive/REPORT.md) | Positive selected-direction coupling is broad and aligned with useful descent, not an identified expendable channel. These Grams are not the full GN spectrum. |
| [TS repeatability](ts_repeatability/REPORT.md) | Large parameter-map variation is much smaller functionally, with an amplitude qualification. No estimator bottleneck was established. |
| [Late auxiliary factorial](aux_partition/REPORT.md) | At Muon4M183→184, embeddings account for much of interaction and additive prediction overlap explains 96–98%. State-local, not a universal harmful-group claim. |
| [Embedding support](embedding_clock/REPORT.md) | Absent-row Adam movement cannot explain interaction on a fresh bank with none of those input rows. |
| [Momentum maps](momentum_maps/REPORT.md) | Same-state PD maps suppress measured stiff energy, but lagged M maps and actual next steps can have opposite slope signs. Output suppression does not measure differential feedback gain. |
| [Replay](replay_projection/REPORT.md) | Ordered-data reweighting is small beside the common gradient but large beside actual momentum. Corpus and trained-model age remain confounded. No feature/lag hunt. |
| [Gauge/decay](gauge_optimizer/REPORT.md) | Scalar decay already commutes with the fixed gauge; helpful shaped decay need not. Restored covariance is not an explanation of its gain. |
| [Prediction profiles](prediction_profile/REPORT.md) | The extra rare-target SOAP signature failed after independent progress calibration. Frequency is not hidden-coordinate SNR. |

Further provenance, original failures, estimator-clock/literature notes and
historical interpretations are indexed in [README.md](README.md) and
[SYNTHESIS.md](SYNTHESIS.md).

## Execution and provenance

All observer computations are terminal.

- Current pass: source/metadata reads and synthetic-only precision arithmetic;
  no empirical gradient/predictor computation or model call. Independent
  provenance and precision notes are in `conditional_input/`.
- `warmup_transfer/extract.py` reads the completed main PD pair only; scalar
  extraction/plotting completed in .76 seconds. Original criterion, all
  184 updates, five validation points, source hashes and paired clocks are
  retained. No inference was made from the unfinished SOAP-PD reference.

- `qk_function/run1`: session **92765**, exit 0. 256 scientific forwards plus
  one restoration, 287.14 s, two CPU threads. Forecast 392.23 s, cap 600 s.
  Max FP64/FP32 CE error 4.80e−5 < fixed 5e−5; scalar/covariance identities
  about 1e−15. Strict corners, restoration, source/tensor/token hashes pass.
  Independent recomputation matches 432 metric summaries and 96 ratios.
- A numerical ratio-resolution clarification was fixed during execution,
  before inspecting outcomes, using the existing 1e−10 tolerance. Original
  executed protocol, annotated copy, timestamp and committed analyzer are
  retained. All observed required KL means exceed .00063, so it changes no
  decision. It is not an added epsilon or a scientific threshold change.
- `confidence_calibration/run1`: session **55327**, exit 0, 192 scientific
  forwards plus one repeat, 227.91 s. All-one outcome; preserved preflight
  offset correction and arithmetic-only same-reduction repair.
- `angular_clock/run1`: session **73509**, exit 0, saved-weight arithmetic,
  36.67 s. `early_split/run1`: session **97830**, exit 0, 207.47 s.
- Earlier `dose_function` and `prediction_profile` probes are complete.
  Reported PID 3 in newer processes is a sandbox-namespace PID, not a host PID.
  Do not restart work from dated notes or an observation timeout.

External main-program context: scheduler check at 18:53 CDT confirmed running
allocations 2567578/2618555 on priv-g14/g20. The original two warmup horizons
are complete (2× 3.962814; 1× 4.515206). In the later saved progress check,
the PD pair is complete and passes, while fresh-seed SOAP-PD warmup is complete
at 3.964582 and its constant reference remains in progress. No paired
SOAP-PD replication conclusion is drawn from that unfinished comparison.
These are main-program runs, not observer computations.

## Progress/completion audit

The preceding turn completed the Q/K bridge, preserving positive transmission
and its conditional-usefulness failure. This turn establishes that the next
proposed archive discriminator lacks required historical membership evidence,
preserves a qualified precision design without empirical claims, and verifies
a newly completed PD schedule-transfer pair. These results change the next
action toward synthesis of trajectory-level evidence rather than unqualified
gradient subtraction or another local component split. The full objective
remains incomplete; neither completion nor a research blocker is claimed.

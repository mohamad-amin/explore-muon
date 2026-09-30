# Observer current map

Updated 2026-09-28 after the completed Q/K functional bridge test.
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

## Latest result: the angular pattern reaches predictions, but conditional helpfulness fails

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

Its distinct candidate is conditional optimizer-input repeatability: a
stable raw gradient may become less stable relative to the input left after
addition to opposing momentum history. The proposed initial family is the
two old 1M step-500 Muon/PD `gn2/*directions.pt` files. Before numerical
interpretation, qualify historical source/sample nesting, negative-gd sign,
unnormalized beta*M+g convention, clipping scope and BF16 interval bounds.
Current code alone does not certify historical nesting, and body-only gd
does not reconstruct full-model clipping. No new predictor outcomes have
been computed. This is a candidate/gating note, not an execution protocol
or a substitute for the actual next training step. Absolute differences
are unchanged by adding a common history; only relative/directional
repeatability can change. No SNR extrapolation or iid-noise claim follows.

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

External main-program context: scheduler check at 18:20 CDT confirmed running
allocations 2567578/2618555 on priv-g14/g20. The original two warmup horizons
are complete (2× 3.962814; 1× 4.515206). In the later saved progress check,
PD warmup replication is complete at 4.017883 but its constant-beta reference
is still collecting; fresh-seed SOAP-PD warmup is also still collecting.
No paired replication conclusion is drawn from an unfinished comparison.
These are main-program runs, not observer computations.

## Progress/completion audit

The preceding turn completed the confidence discriminator and identified the
missing Q/K functional measurements. This turn supplied a separately reviewed
four-state experiment: the angular pattern reaches predictions, while the
chosen actual-background helpfulness condition fails. That changes the next
action by closing an intervention premise and requesting a broader priority
reset. The full observer objective remains incomplete; neither completion
nor a research blocker is claimed.

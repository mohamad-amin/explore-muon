# Priority reset: test whether optimizer gains change the prediction profile

2026-09-28. Read the current guide/state and observer synthesis, the original
frequency probe and its outputs, the displacement-efficiency source, and
saved scalar validation records/available checkpoints. No model, checkpoint
tensor, GPU or scheduler call. This is the only new file.

## Select one substantive probe

**Run a bounded forward-only, phase-corrected frequency-profile comparison
on fresh FineWeb inputs.** This asks whether the optimizers improve a
different prediction regime, beyond looking like Muon farther along its
training trajectory. It uses actual predictive loss, is invariant to the
internal gauge issues, and has a clear hypothesis suggested by earlier
evidence without replaying another local curvature argument.

The original September 25 `frequency_probe_20260925` already evaluated
final Muon/PD/SOAP/SOAP-PD checkpoints on 512 sequences. It found a stronger
SOAP advantage on mid-rare targets and a broader PD advantage. Merely
repeating a final-checkpoint frequency table is therefore not the new
question. The missing control is ordinary progress: rare targets may show
larger NLL changes simply because every method learns them later, and a
better optimizer is farther along that common progression.

A future-versus-past gradient replay is currently lower priority. It would
require many expensive backward passes and still needs independent control
of exposure history/content. The old 100-row scalar line is closed. More
parameter-displacement arithmetic also lacks a direct connection to useful
predictions and faces gauge/path conventions already identified. The
proposed prediction probe addresses a genuinely missing outcome-level
comparison at bounded CPU cost.

## Frozen cells and independent progress calibration

Use the completed original 1M trajectories in
`logs/muon_spectra/soaudit_traj_20260926/`:

- `M_lr0.007_s260925_l40s`: steps 200, 500, 900, 1300;
- `PD_a0.25_lr0.01_s260925_ada`: steps 500, 900;
- `S_lr0.007_s260925_l40s`: steps 500, 900;
- `SPD_a0.25_lr0.01_s260925_ada`: steps 500, 900.

That is ten fixed states. Include every one; no checkpoint selection from
the new score bank. The existing model/data pairing should be verified,
and the two hardware types retained as a limitation of these old recipes.
This is not an independent training-seed confirmation.

Before the new forwards, set each candidate's interpolation coefficient
using only its **existing ordinary validation** NLL. At candidate step 500,
use the Muon500→900 bracket; at candidate step 900 use Muon900→1300.
For candidate X at t and later Muon anchor u,

    a_X = (old_L_M(t)-old_L_X(t)) / (old_L_M(t)-old_L_M(u)).

The six coefficients are interior. From the inspected records:

| Candidate | Step 500 bracket weight | Step 900 bracket weight |
|---|---:|---:|
| PD | .194668 | .228536 |
| SOAP | .204835 | .350554 |
| SOAP-PD | .329063 | .433857 |

At each fresh target, compare its candidate NLL to the fixed secant profile
`(1-a_X)*NLL_M(t)+a_X*NLL_M(u)`. This is a trajectory-profile reference,
not an actual intermediate trained model or an ensemble prediction.

The parent proposed adding Muon200 as a tenth state. Endorse this: report
two leave-interior-anchor-out sensitivity profiles, predicting Muon500 from
Muon200/900 and Muon900 from Muon500/1300 with the same old-validation
calibration. They expose ordinary nonlinearity of profile-versus-progress.
**They are descriptive sensitivity controls, not rigorous error bounds or
additional quantities to subtract until a preferred result appears.**
Keep both, even if they disagree.

## Prediction and strongest alternative

Use target-token frequency from a fixed 32M-token training-prefix count,
before scoring any candidate. Keep the six historical edges
`[0,1e-6,1e-5,1e-4,1e-3,1e-2,1]`. The different count-bank choice from the
older 50M future-training count must be documented; this is not an exact
reproduction of the old bin membership.

The primary broad contrast is the secant-residual NLL on
`1e-6 <= frequency < 1e-4` minus that on `frequency >= 1e-3`.
Negative values mean extra mid-rare improvement relative to common targets
after the independently calibrated progress correction.

**Prediction:** SOAP has a more negative contrast than PD, and adding SOAP
to PD likewise makes the contrast more negative, at both fixed stages.
PD's gains are closer to a broad progress shift. This connects the earlier
stacking dissection to what the models actually predict, without identifying
target rarity with parameter-coordinate SNR or claiming a heavy-tail cause.

**Strongest alternative:** all the apparent frequency differences follow
ordinary Muon progress, including nonlinear changes along that trajectory.
If candidate residuals resemble the two Muon interpolation controls and
are small relative to fresh-input paired dispersion, the specific
rare-target explanation is unsupported by this probe. Do not then add
new frequency thresholds, difficulty strata or context features.

Frequency bins are external to candidate errors. Never stratify by Muon's
new per-token loss, which would introduce regression-to-the-mean and
selection effects. The weights are calibrated on the old bank, not fitted
to the same new score observations. Report fresh overall NLL residual as
well: old-bank global matching need not remain exact on a new finite panel.

## Bounded size, gates and adequate interpretation

Endorse the parent's two fixed fresh banks of 16 sequences each, context
512: **320 total sequence-forwards across ten states**, CPU FP32, two
threads, no gradients. This is a meaningful descriptive premise check,
not a powered absence test. The primary broad bins should cover thousands
of targets (the old proportions were roughly 36% and 41%). The rarest
under-1e-6 bin would contain only about 90 targets at the historical share
and must remain descriptive.

Use reused qualified model/loading/CE helpers where possible. Before
committing, time the first two declared score sequences, including relevant
model-load overhead, and forecast all 320 forwards plus ten loads with
a conservative margin against the fixed **600-second** wall cap. If the
forecast fails, stop and preserve qualification; do not silently reduce
the panel or replace the test. Qualification score rows belong to the
fixed panel and cannot be discarded because of their values.

Required checks are modest and consequential:

- identical input/target arrays across every model, with recorded offsets
  outside each evaluated model's training prefix and prior observer banks;
- frozen checkpoint/config/source identifiers and evaluation mode;
- finite per-token FP32 outputs, with FP64 accumulation for differences;
- every target assigned once, and frequency-bin sums/counts reconstruct
  total NLL exactly to the chosen accumulation tolerance;
- all ten states and both banks retained, with no tuning on these scores.

Report each bank separately plus the pooled estimate. Retain per-sequence
paired losses/contrast sums and show paired sequence-level dispersion.
Tokens, strata, stages and two adjacent corpus banks are not independent
training replications; do not manufacture large n from token counts.

With only 32 sequences, an unresolved contrast remains unresolved. A lack
of significance cannot close all prediction-regime hypotheses. Conversely,
a large consistent profile difference that survives the progress-sensitivity
controls is useful enough to change the next mechanistic focus. The
original proposal of a larger panel would cost considerably more; do not
expand to it after seeing an ambiguous result.

## What changes next

If the specified SOAP-specific mid-rare profile is clearly present at both
stages and on both banks beyond the visible progress-control behavior,
study output-side normalization through its effect on those predictions,
instead of continuing generic stiff-energy or gauge stories. That result
still would not justify a new optimizer recipe by itself.

If it collapses onto ordinary progress or reverses, retire the old simple
frequency-specific interpretation and preserve the negative result. If the
panel is inconclusive, close this bounded probe without another feature
search; use the completed main trajectory interventions, including momentum
warmup, to set the next question. This is a substantive experiment worth
running under the stated gate, not another bookkeeping exercise.

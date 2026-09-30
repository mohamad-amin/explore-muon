# Prediction-regime differences after an independent progress calibration

2026-09-28, before fresh model scoring. Independent high-level review:
`../priority_reset_peer/REVIEW.md`. Previous observer turn was progress,
closing an uninformative invariance-restoration branch. This is a new bounded
forward-only scientific probe, not another static-metric audit.

## Missing premise

The Sep25frequency probe already found final SOAP gains leaning toward
mid-rare target tokens and PD gains spread more broadly. Faster ordinary
learning could create that pattern if rare/common predictions improve at
different rates. The unanswered question is whether the profile differs
after correcting for progress, using information independent of the score
panel. A positive result would identify a prediction regime for a later
mechanism test; it would not establish output-gradient causality.

## Fixed states, calibration and score panel

Use the original1M`soaudit_traj_20260926`seed260925states: Muon at200/500/900/1300,
and PD/SOAP-Muon/SOAP-PD at500/900. Ten checkpoints, same architecture and
initial weights/data; hardware differs by original paired trajectory and is
recorded. Main/tiny-surrogate jobs and sealed panels remain untouched.

Freeze the reference weights from original logged validation NLL only:
for each non-M500state interpolate Muon500→900loss profiles, and for each
non-M900state interpolate Muon900→1300. For target old-bank loss L*, weight
t=(L_left−L*)/(L_left−L_right). Require0<=t<=1; no extrapolation or fresh-panel
refit. The result is a secant of loss profiles, not an actual interpolated
model or proof of exact phase matching.

Both own-Muon interpolation controls are mandatory: predict M500fromM200/900,
and M900fromM500/1300using the same old-bank formula. They measure sensitivity
to nonlinear learning profiles, not a rigorous bound or subtraction target.
Also retain direct S500−PD500, whose old-bank losses differ by only.001908,
as a nearly matched-mean comparison without a Muon secant.

Score exactly two banks of16sequences, context512, starting at training-stream
offsets3,120,000,000 and3,160,000,000. These positions exceed every selected
checkpoint's training exposure. Save exact input/target tokens and hashes.
No sequence, bin or checkpoint is selected by fresh loss. These are in-domain
held-out positions, not a newly sealed independent test population.

Reproduce the old unigram reference:50Mtokens at offset2,500,000,000 from the
training partition, with the six original frequency bins
[0,1e−6,1e−5,1e−4,1e−3,1e−2,1]. Counts depend on data only, not model loss.
Process counts in chunks to bound RAM and retain the exact count vector.
Primary shape contrast pools mid-rare[1e−6,1e−4) versus common>=1e−3. The
rarest bin is descriptive only; report its support and retain its losses.

## Readout and predictions

For each checkpoint/method define residual loss=method−frozenMuonreference.
The primary contrast is mean residual(mid-rare)−mean residual(common).
Negative means an extra rare-target advantage relative to the progress
reference. Retain raw same-step differences, residual global/binned means,
both banks separately, and all paired per-sequence sums/counts. Ordinary
progress predicts similar profiles after calibration. A SOAP-specific
signature predicts negative residual contrasts for S/SPD at both stages,
with consistent bank signs, beyond the own-Muon control patterns.

Report sequence-level paired dispersion and nominal cluster SE for the ratio
of grouped sums, explicitly conditional on this panel. Adjacent sequences
may share documents; no seed-level significance or population equivalence
claim follows. A useful lead needs consistency across banks/stages and clear
separation from BOTH interpolation-control magnitudes and paired dispersion.
Otherwise the regime-specific mechanism is unresolved, not disproven. Do
not enlarge the sample, add bins/features, fit a favorable phase warp, or
launch a training remedy after a weak result. Context-position effects are
not a second searched endpoint in this pass.

## Execution qualification and boundary

Use frozen helpers from the qualified observer body_auxsource, CPU FP32eval,
no input-statistic mutation, gradients, optimizer or CUDA. Check source hashes,
architecture/initialization pairing, checkpoint steps, finite token losses,
and exact token/bin coverage. Reconstruct total loss from bin sums. On the
first sequence, per-token mean must agree with the model's scalar CE within
2e−6; a repeated forward must reproduce per-token loss within2e−6.

Time two actual score sequences and the first checkpoint load before the
remaining work. Extrapolate all320forwards and10loads with25%margin plus
20seconds overhead; if this exceeds600seconds, stop and preserve qualification.
Also enforce a600-second wall boundary. Never start an optimizer or GPU job.
All per-token NLLs, tokens, calibration, timings, source/checkpoint provenance,
failures and final analysis stay in this directory. A cost stop is incomplete
measurement, not a negative scientific outcome.

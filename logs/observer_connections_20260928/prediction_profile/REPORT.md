# The rare-target signature is not stable after accounting for learning progress

2026-09-28. Completed forward-only CPU probe of ten saved1Mcheckpoints on
32fresh512-token sequences. No training or GPU. Runtime136.03seconds for
320score forwards, plus the two declared repeat/scalar qualification calls.
The independent old-bank calibration was fixed before fresh scoring.

## What this adds to the earlier frequency probe

The Sep25final-checkpoint probe found SOAP's gain leaning toward mid-rare
targets, while PD's gain was broader. That observation remains valid for
its particular final checkpoints and score panel. It did not distinguish
optimizer-specific prediction preferences from ordinary learning progress.

This probe tests that missing premise at steps500and900, before cooldown.
Muon200/500/900/1300supply a reference trajectory; PD, SOAP-Muon(S) and
SOAP-PD(SPD)are evaluated at500/900. Reference interpolation weights use
only the original logged validation NLL. No weight, bin, state or input
was selected or refitted on the fresh scores.

For each target state, the reference is a linear interpolation of Muon
per-token losses, calibrated to its old-bank mean. This is a secant of loss
profiles, not a real interpolated model or proof of exact phase matching.
Own-Muon500and900interior-anchor controls expose that limitation.

The two new banks start at3.12B/3.16B training-stream positions, beyond all
selected states' exposure. Frequency bins use the original50M-token reference
at2.5B, independent of model scores. Mid-rare means1e−6<=frequency<1e−4;
common meansfrequency>=1e−3. The primary contrast is

    mean(method−progress_reference | mid-rare)
      − mean(method−progress_reference | common).

A negative contrast would be an extra rare-target advantage. The six bins
contain97,1966,3945,3619,2962,3795targets respectively. The extreme rarest bin
is descriptive only. All16,384targets and their token losses are retained.

## The declared signature did not appear

| Method/state | Raw same-step rare−common contrast | Progress-adjusted contrast | Nominal sequence SE |
|---|---:|---:|---:|
| PD500 | −0.00693 | +0.05353 | 0.01471 |
| PD900 | +0.04909 | +0.05884 | 0.01787 |
| S500 | −0.02645 | +0.03717 | 0.01647 |
| S900 | +0.00820 | +0.02315 | 0.02134 |
| SPD500 | −0.04952 | +0.05269 | 0.01897 |
| SPD900 | +0.02487 | +0.04338 | 0.01940 |

All twelve method×stage×bank adjusted contrasts are positive, opposite the
predicted negative SOAP signature. The raw same-step signature itself changes
with stage: all three methods are more rare-leaning at500and more common-leaning
at900on this panel. These statements concern the declared contrasts, not a
claim that any method increases rare-token NLL in every bin.

The paired global improvements remain. At500, PD/S/SPDminusMuon are
−.02413/−.04075/−.06759; at900they are−.02290/−.02668/−.03510. Thus failure
of the rare-target explanation does not erase the observed predictive gains.

## Interpolation and bank controls are material

Muon itself changes its rare/common profile strongly and nonlinearly:

| Muon progress interval | Change in rare−common NLL difference |
|---|---:|
| 500→900 | −0.31061 |
| 900→1300 | −0.04266 |

The M500control, predicted from M200/M900, has adjusted contrast+.01828
(bank0−.00727,bank1+.04500). The M900control, predicted from M500/M1300,
has−.06521(bank0−.06002,bank1−.07032). These controls are sensitivity
profiles, not rigorous bounds or quantities to subtract selectively. Their
scale prevents promoting the opposite sign into a new common-token mechanism.

The direct S500−PD500comparison has nearly matched old-bank means
(difference−.001908)and avoids a Muon secant. Its fresh rare−common contrast
is−.01952, with nominalSE.01795and bank values−.04713/+ .00961. It therefore
does not supply a stable independent rare-target discriminator either.
Its fresh global mean difference is−.01662, showing that old-bank mean
matching need not transfer exactly to a small fresh panel.

Sequence SEs are paired ratio-of-sums calculations, conditional on fixed
calibration weights and this panel. Adjacent sequences can share documents;
these are not seed-level significance tests and do not account for all
calibration or data-population uncertainty. The32-sequence panel can expose
a large consistent effect but cannot prove an absence of regime differences.

## Qualification and provenance

All ten models share the original initialization hash, architecture, data
manifests and budget. The original hardware differs by trajectory (Muon/S
on L40S, PD/SPD on Ada), as recorded; this is not a new hardware-matched
training replication. CPUhelpers are byte-identical to the prior qualified
observer source. Strict parameter loads, checkpoint steps, finite per-token
losses and exact shared-token/bin coverage pass. Model scalarCEequals the
first sequence's token mean, and the repeated forward is identical (both
errors0). Bin sums reconstruct each total NLL.

The first two score sequences forecast199.98seconds including a25%margin;
actual execution finished at136.03seconds, below the600-second limit. Runtime
checks occur every eight sequences. No time gate, sample count or analysis
definition was changed. Checkpoint tensor hashes, source hashes, frozen old
calibration, exact tokens/counts, all per-token losses and timings are retained.
Independent implementation and result reviews passed; scalar recomputation
agrees within1.4e−17.

## Decision

The declared extra rare-target SOAP signature is unsupported at these states
under this comparison. The broader regime-specific mechanism remains
unresolved because phase interpolation and score-bank variation are material.
Do not conclude equivalence to ordinary Muon progress, a universal common-token
advantage, or a falsification of the original final-checkpoint observation.

Close the bounded probe without more samples, bins, contexts or fitted phase
warps. The useful connection is that prediction-group improvements have their
own nonlinear learning curves; a rare-target gain at one endpoint cannot by
itself identify a heavy-tail normalization mechanism. Any later mechanism
test must retain a progress comparison with adequate phase resolution.

Evidence:`run1/analysis.json`,`primary_contrasts.csv`,`phase_contrasts.png`/
`.pdf`, `per_token_nll.npz`,`tokens_frequency.npz`,`prepared.json`, and
`result.json`. Scripts:`probe.py`,`analyze.py`. Reviews:
`../priority_reset_peer/{REVIEW,IMPLEMENTATION_REVIEW,RESULTS_REVIEW}.md`.

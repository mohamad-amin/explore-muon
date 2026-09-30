# Existing clipping and horizon controls completed

Verified from raw saved records at2026-09-28 15:00 CDT; numerical completion
audit recorded15:02:55 CDT. This addendum preserves the14:17 snapshot and its
then-correct partial-run statements in `REPORT.md`, `audit.json`, CSV and logs.
Their hashes are recorded and confirmed unchanged in `completion_audit.json`.
No training, model call, GPU use, job submission, or main-note edit occurred.

**The simple clipping-frequency explanation fails this control. The large
absolute SOAP-PD momentum benefit attenuates on the longer trajectory, even
before cooldown.** Neither outcome identifies an optimal beta or proves a
noise-versus-lag mechanism.

## Matched Muon clipping control

Both runs are complete at92 steps and1,539,870,720 tokens. The pipeline's
scientific workers and full pipelines exited successfully. All25 recorded
source hashes match exactly between the pair; the **only configuration
difference is body momentum beta**. Initial weights, data manifests, hardware
(L40S), per-step body LR and auxiliary LR match.

| Muon clip.1 | beta.9 | beta.8 | beta.8 minus.9 |
|---|---:|---:|---:|
| Validation step50 | 5.6759361 | 5.6787930 | +.0028569 |
| Final validation92 | 4.8881850 | 4.8915906 | +.0034056 |
| Clipped steps | 91/92 | 92/92 | — |
| Mean train NLL47–83 | 5.3323401 | 5.3452463 | +.0129062 |
| Mean raw-gradient coefficient age47–83 | 8.1069 | 3.8935 | −4.2133 steps |
| Mean global-unit-gradient coefficient age47–83 | 8.8962 | 3.9998 | −4.8964 steps |

Thus clipping every step and halving the effective history does **not**
reproduce PD's−.124 or SOAP-PD's−.102 short-run improvement. The observed
positive.0034 final difference is a single-seed difference, not a significant
harm claim; the relevant result is that the large gain is absent.

For the ordinary clip1 Muon pair, beta.8 minus.9 is−.02030. The clipping × beta
endpoint interaction is therefore+.02370 NLL, opposite to the explanation that
normalizing Muon's gradients should unlock the large whitening-sized beta
benefit. This control rules against clipping incidence/normalized input alone
as a sufficient explanation at the tested rate. It does **not** establish
that clipping is irrelevant in general, nor isolate all norm/angle and
auxiliary-state effects across different optimizer geometries. The stronger
main-notebook wording “the clipping confound is ruled out” should be read with
this scope.

## SOAP-PD at twice the token horizon

The new beta.8 run and existing beta.9 reference both complete184 updates and
3,079,741,440 tokens on RTX6000 Ada. Their initial-model hash, data-manifest
fingerprints, hardware, and every per-step body/auxiliary LR match. Config
differences are beta and newly recorded disabled defaults (`muon_prefilter:
none`, head-whitening alpha0/centerFalse/norm-match). Six frozen files differ;
their actual execution-source hashes were checked and unified diffs retained.
Inspection finds newly added head-whitening/filter code inactive in this
pair, plus associated logging/configuration/tests. This is matched active
behavior, not byte-identical execution sources.

| Validation step | beta.9 | beta.8 | beta.8 minus.9 |
|---|---:|---:|---:|
| 50 | 5.3467095 | 5.2501685 | −.0965409 |
| 100 | 4.4694454 | 4.4221032 | −.0473423 |
| 150 | 4.1434990 | 4.1300337 | −.0134654 |
| 184 | 3.9840072 | 3.9740494 | −.0099579 |

For comparison, the short92-step beta pair finishes4.5741897 versus4.4723927,
a−.1017970 difference. The longer-horizon endpoint gap is9.78% as large.
That is a comparison of **loss gaps**, not evidence for a90% reduction in
training speedup. Curves are flatter later.

The long pair also loses most of its absolute gap **before** cooldown starts
at167. The dense paired training records show:

| Window | Mean beta.8 minus.9 train NLL | Local beta.9 slope, NLL/step | Approximate local step lead |
|---|---:|---:|---:|
| 47–83 | −.083816 | −.020005 | 4.19 |
| 84–92 | −.078082 | −.012732 | 6.13 |
| 93–120 | −.041113 | −.008618 | 4.77 |
| 121–150 | −.018101 | −.005432 | 3.33 |
| 151–166 | −.007635 | −.003272 | 2.33 |
| 167–184, cooldown | −.007413 | −.005035 | 1.47 |

The last column divides the window's mean paired loss difference by the
reference's least-squares slope on the same window. This is an explicitly
exploratory **local linear conversion**, not an observed validation crossing
or a globally valid step-equivalent rate. It shows why the shrinking raw gap
overstates attenuation, while suggesting the late lead itself also decreases.
Independent validation is only recorded at50,100,150,184 and is too sparse
for a precise late crossing claim. Training windows are correlated and do
not provide independent replications or confidence intervals.

This is not solely a shortening of a fixed memory relative to the run:

| Constant-LR window | beta.9 raw age | beta.8 raw age | beta.9 global-unit age | beta.8 global-unit age |
|---|---:|---:|---:|---:|
| 47–83 | 8.4129 | 3.9490 | 8.8962 | 3.9998 |
| 93–120 | 9.0924 | 3.9519 | 8.9981 | 4.0000 |
| 121–150 | 8.8430 | 3.9380 | 8.9999 | 4.0000 |
| 151–166 | 8.8492 | 3.9717 | 9.0000 | 4.0000 |

Both methods clip every step through166 and178/184 steps overall. The late
constant-LR raw ages remain148.5M versus66.6M tokens, and the normalized-unit
ages remain151.0M versus67.1M tokens. The scalar memory difference persists
as the gain attenuates. Coefficient ages remain scalar history descriptors;
they do not reconstruct per-matrix directions, useful descent or noise.

## Consequence for the interpretation

The available evidence favors an **early-trajectory advantage at the tested
recipe**, with a smaller residual advantage later. This connects to the
previously preserved4M PD beta.81 result: a~.057 early gain reversed to+.0118
at completion. It is increasingly hard to infer a durable optimizer principle
from the large92-step endpoint gap alone.

The batch is fixed at16M across the horizon control, and clipping/mean ages
are almost unchanged late. A simple static story that large batch means little
noise, therefore shorter averaging remains much better, is incomplete. The
gradient field, curvature, useful drift, optimizer state or noise relative to
signal can all evolve. The observer's earlier SNR audit established that the
clipped/extrapolated statistic cannot certify near-universal reliable signal;
this horizon result provides no replacement measurement of noise. These
findings are compatible constraints, not a demonstrated common mechanism.

Important limits:

- One seed for the long pair; only the shorter SOAP-PD beta.8 improvement has
  a second-seed replication. The long residual.010 could require replication.
- The comparison fixes LR.028, geometry alpha.5, auxiliary recipe and schedule.
  It is not a beta/LR optimum or a complete horizon bracket.
- The two horizon cohorts have matching initial hashes/data manifests and
  warmup3 but separate executions and different kept-checkpoint schedules.
  The beta.9 short/long source difference adds a two-sided SOAP gradient
  branch inactive when output power is0. Their observed early paths differ
  slightly (step50 beta.9 validation differs.01056), so the long experiment
  is not presented as a bytewise continuation of the short trajectory.
- Cooldown moves from84–92 to167–184. The preceding constant-LR attenuation
  is real in the saved trajectory, but the final two-horizon comparison does
  not isolate exposure, training stage and schedule as separate causes.
- Geometry and global clipping remain coupled to auxiliaries. The controls
  reject a sufficient explanation; they do not prove the positive mechanism.

No new scientific run follows from this addendum. Existing main queues remain
unchanged. The warranted update is to carry “early/transient versus sustained
benefit” as a live distinction when interpreting momentum, spectrum and SNR
artifacts, and to avoid calling beta.8 the large-batch optimum from these runs.

## Reproducibility and preservation

`completion.py` reads eight completed arms (920 positive-step records), verifies
all200 source-file hashes against execution metadata, checks contiguous steps,
full token budgets and successful scientific worker exits, and writes only
new `completion_*` artifacts here. It takes under a second of stdlib CPU
arithmetic, with no torch or model import. `completion_steps.csv` contains all
reconstructed ages and raw norm/LR/loss fields. `completion_audit.json` contains
source/config/init/data comparisons, input hashes, exact differences and
the unchanged prior-artifact hashes. The original14:17 report remains intact.

# Momentum changes Q/K turns through weight-radius adaptation

2026-09-28. Both predeclared readings pass in the existing one-seed SOAP-PD
learning-rate × momentum factorial. All twelve saved adjacent-state pairs
were analyzed in 36.67 seconds on two CPU threads, with no model forwards,
gradients, optimizer reconstruction or GPU use.

**Finding:** equal prescribed parameter-step norms do not imply equal
relative motion of normalized query/key weights. Shorter momentum has
smaller Q/K radii and larger normalized turns. Increasing LR grows the
radii and produces a much smaller late turn increase than its 43% nominal
increase. This supplies a concrete alternative to interpreting the earlier
LR-insensitive momentum benefit as evidence of a causal optimizer-step clock.
It does not identify the cause of that training benefit.

## Why this is a distinct connection

The previous `momentum_lr` analysis found a roughly .10 NLL benefit from
beta .8 at both LR .028 and .04. Its nominal body arc rose 43%, while the
benefit and broad timing changed little late in the run. Separately, the
main warmup test succeeds at 184 updates and misses its target at 92.
Neither comparison measured how growing parameter radii affect relative
motion of normalized maps.

The architecture normalizes Q and K separately within each head. In the
zero-epsilon limit, multiplying one head's Q or K weight block by a positive
scalar leaves its normalized output unchanged at a fixed input. Therefore
the per-head weight direction is a more relevant geometric descriptor than
its raw step norm. This is a known normalization effect, not a new theory:
[auto rate-tuning](https://arxiv.org/abs/1812.03981) and
[intrinsic learning rates](https://arxiv.org/abs/2010.02916) provide precedents.
Their SGD/normalization results are not proofs for this nonlinear SOAP-PD map.

## Fixed comparison and observable

Use beta {.9,.8} × body LR {.028,.04}, 16M batch, 92 updates, seed 260925,
Ada. The four arms were previously paired and source-qualified. Their
own states 9→10, 46→47 and 83→84 are scored in full: 48 body matrices per
state, and 8 blocks × 8 heads separately for Q and K. Headwise results are
primary; the other kinds are context and have no asserted individual scale
symmetry. All 576 matrix rows and 1,536 head rows are retained.

For each saved head weight w and its actual next write w', the normalized
turn is the chord ||w'/||w'|| − w/||w||||. It is a weight-space quotient
descriptor, not a measured feature-output or prediction-space distance.
RMSNorm epsilon, input changes, gains, and other parameter updates remain
outside its functional interpretation.

Primary contrasts are medians of matched head ratios, Q and K separately.
The initial root draft's ratio of RMS chords was changed to this paired
readout on independent review before the twelve-state extraction; the draft
is preserved. Thresholds, states and arms were not changed after extraction.

## Both necessary-condition readings pass

| Contrast | State | Q | K |
|---|---:|---:|---:|
| LR .04/.028 at beta .9 | 9 | 1.342 | 1.356 |
| LR .04/.028 at beta .8 | 9 | 1.347 | 1.361 |
| LR .04/.028 at beta .9 | 46 | 1.110 | 1.114 |
| LR .04/.028 at beta .8 | 46 | 1.161 | 1.185 |
| LR .04/.028 at beta .9 | 83 | 1.091 | 1.089 |
| LR .04/.028 at beta .8 | 83 | 1.137 | 1.160 |
| Beta .8/.9 at LR .028 | 9 | 1.009 | 1.008 |
| Beta .8/.9 at LR .04 | 9 | 1.012 | 1.012 |
| Beta .8/.9 at LR .028 | 46 | 1.149 | 1.174 |
| Beta .8/.9 at LR .04 | 46 | 1.198 | 1.247 |
| Beta .8/.9 at LR .028 | 83 | 1.154 | 1.171 |
| Beta .8/.9 at LR .04 | 83 | 1.218 | 1.246 |

P1 required LR ratios below 1.20 in every later Q/K × beta cell; all eight
pass, versus nominal LR ratio 1.4286. Later head-radius ratios are 1.206–1.314,
while tangent-displacement ratios remain about 1.419–1.433. Larger radii
substantially absorb the larger nominal parameter step. Compensation is
partial, not exact: relative turns still increase 9–19%.

P2 required beta ratios above 1.10 in every later Q/K × LR cell; all eight
pass. Later beta-.8/.9 radius ratios are .802–.874, while actual head-step
norm ratios are .999–1.003 and tangent-norm ratios 1.000–1.016. The larger
relative turns mainly accompany smaller radii, rather than larger absolute
head steps. These ranges summarize separate paired medians; they are not
multiplied together as an exact median decomposition.

The early state is different: LR still changes turns by 34–36%, and the
momentum contrast is about 1%. Thus the radius/turn separation develops
during training. It is not an initial hidden setting difference.

The effect is broad but not universal across heads. In the later LR cells,
46–60 of 64 heads meet the <1.20 threshold; in the later momentum cells,
53–63 of 64 meet >1.10. Every exception remains in `head_ratios.csv`.
Heads and checkpoints are not independent seed replications.

[Figure](run1/angular_ratios.png) shows paired medians and head quartiles;
the shading is descriptive dispersion, not a confidence interval.

## Radius growth is not simply a random walk

The exact accounting is

    ||w'||² − ||w||² = 2<w,d> + ||d||²,
    d = w'−w.

For every pooled Q/K cell the radial cross term is positive. At state 46,
its ratio to squared-step energy is 5.91–7.18 at beta .9 and 2.00–2.85 at
beta .8. At state 83 the ranges are 2.53–5.32 and 1.88–3.31 respectively.
The coherent outward radial term is therefore material, often much larger
than the squared-step term. A radius law eta*sqrt(t) based only on diffusion
is not justified. Current radial terms do not reconstruct all past growth.

The report retains signed radial/tangent components of the actual write,
and of d+eta*wd*w. The latter removes ideal scalar decay but still contains
FP32 write rounding; it is not relabeled as a captured optimizer direction.
Nor does the observed radial drift establish whether momentum history,
the nonlinear map, the learned state, or their interaction caused it.

Whole-matrix summaries show smaller beta-.8 radii in the other body kinds
at the last retained state too. Their normalized-weight chords are useful
descriptors, but individual V/O/MLP rescalings are not the Q/K symmetry.
No all-body functional claim follows from those rows.

## Qualification and stopping decision

All consumed metadata and scalar records match the prior factorial's
recorded hashes. The active model, Muon/SOAP, PD and trainer sources match
their saved metadata; SOAP lives in `muon.py` in these revisions. All 1,152
consumed before/after body-tensor hashes are recorded. Checkpoint steps,
token counts, shapes and file metadata passed. Original files were untouched.

The maximum radius identity relative error is 3.27e−14; chord-squared
identity absolute error is 2.65e−13, within the fixed 1e−10 gates. The first
pair forecast was 22.62 seconds; colder later reads made actual time 36.67
seconds, still under the 180-second boundary. Session 73509 exited 0.
The 24-matrix scalar radius curves are explicitly indexed to incoming
weights, separately from after-update kept states.

Close this fixed twelve-state comparison with both readings supported.
The earlier nominal-arc result remains true, but does not exclude an
endogenous relative-step explanation. A causal claim that phase is governed
by step count is premature. No normalization change, momentum schedule,
radius constraint or new training improvement is inferred.

A causal continuation would have to separate memory from relative motion
and show which components matter to loss; the present own-state observations
do neither. The independent output-confidence proposal remains a distinct
direct predictive question. This positive geometry result does not authorize
merging the two explanations or reopening the failed mixed-response branch.

Evidence: [protocol](PROTOCOL.md), [implementation review](IMPLEMENTATION_REVIEW.md),
[independent results review](RESULTS_REVIEW.md),
`run1/result.json`, `matrices.csv`, `heads.csv`, `head_ratios.csv`,
`summaries.csv`, `scalar_radii.csv`; all CSVs are inside `run1/`.

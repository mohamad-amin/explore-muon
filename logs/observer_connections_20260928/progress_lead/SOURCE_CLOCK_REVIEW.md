# Independent source, clock, and resolution review

2026-09-28. Read frozen source, configurations, metadata, statuses, and raw
JSON timing fields for the three completed 184-update warmup/.9 pairs.
Read the earlier `warmup_transfer/` and `confidence_calibration/PAIRING_REVIEW.md`
qualifications. No model, checkpoint tensor, horizontal-lead reduction,
new scientific score, or job. Only this review is written.

**Verdict:** the proposed conservative training-record support **94–166**
is correct: those losses measure model states **93–165**, after the first
common-beta update and before any cooldown update. Entire smoothing windows
and both endpoints of each reference crossing must obey that restriction.

## Exact runs and source integrity

Paths are relative to `logs/muon_spectra/`; each below uses its own
`scientific/metadata.json`, `source/`, `status.json`, and `steps/`:

| Label | Arm |
|---|---|
| S25 warm | `soaudit_momwarm16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada` |
| S25 reference | `soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada` |
| S26 warm | `soaudit_warmrep_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260926_ada` |
| S26 reference | `soaudit_warmrep_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260926_ada` |
| P25 warm | `soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_l40s` |
| P25 reference | `soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_l40s` |

All six statuses are complete. Each has exactly 185 step JSONs, with IDs
0–184. All 25 recorded frozen source files per arm match their SHA256
manifests: **150 checks, zero mismatches**.

Five arms share the newer source revision:

- `train.py`: `a73307e7dfb51e5f17a65cf5114fcedbda25b52df639e98460baf5fda1fc5025`
- `distributed.py`: `7312fc88c8654f26b67a87046d4a530dbb46db657e65b72402a2a686212d522c`
- `model.py`: `034c0f37a986b994bb78078d4f9002e21b9c5737566f8fb05d9e56ce84168b4f`
- `muon.py`: `9178cd3b3d0b132067282952a9fc4faf151d8f382bd95c170248b2014f459f1d`

The original S25 reference has the older revision:

- `train.py`: `b637265f9fe47ebfecdc5615d29c94b0f8c8242068ef1d1cba6afdb7c3acab5f`
- `distributed.py`: `8eb97eddc413c0ca578afb663b0a239d9d26b27abfd5c80c098ca39a7f27ca5c`
- `model.py`: `7bb63819666a4917ac7919a6ac16540293a9ef074a6015e1729e21ea4f6fb25a`
- `muon.py`: `c6e29f41b035988bf80d1544c5e5369c3804ba7fc869afdb765b0ba97d3e321e`

For S25, the recorded config differences are the intended momentum schedule
and later inactive defaults: prefilter=`none`, head whitening alpha=0,
head centering=false, and head normalization=`match` within the disabled
head-whitening feature. The prior confidence pairing review qualified the
broader active-source differences, including the inactive SOAP output-factor
addition. This pass independently verifies timing control flow and hashes;
it does not claim bit-identical compiled executions across revisions.

## Loss and update indexing

In the newer frozen `distributed.py`:

- Lines 295–302 increment the update counter and choose LR/beta using
  **tokens consumed before** that update.
- Lines 323–335 run all microbatches and accumulate their token-weighted
  detached losses while weights remain W[t−1].
- Line 347 performs `optimizer.step()`, producing W[t].
- Lines 352–357 reduce the already accumulated loss, advance the token
  counter, and put that loss into record t.
- Lines 429–430 evaluate validation after the update, on W[t].

The older reference has the same order at lines 285–344 and validation at
418; it lacks only the new momentum-schedule assignment. Therefore:

    record t: train_nll      = loss of W[t−1] on training batch t
              validation_nll = loss of W[t] on the fixed validation bank
              tokens         = cumulative tokens AFTER update t
              lr             = LR applied DURING update t.

Training-loss state exposure is `tokens−batch_tokens`, not the record's
post-update `tokens`. Subtracting one from both arms' step axes leaves a
lead difference unchanged but matters for regime boundaries. Validation
records exist only at 0,50,100,150,184 in all six arms.

## Exact schedule boundaries

The shared nominal batch is 16,777,216 tokens and budget is 3,079,741,440.
New `train.py:217–224` implements the beta ramp; lines 227–232 implement LR.
Using each arm's raw `tokens−batch_tokens` reproduces:

| Update | Tokens before update | Warm beta | Applied LR |
|---:|---:|---:|---:|
| 92 | 1,526,726,656 | .8991464176940777 | .028 |
| 93 | 1,543,503,872 | .9 | .028 |
| 166 | 2,768,240,640 | .9 | .028 |
| 167 | 2,785,017,856 | .9 | .02679530250435569 |

References use beta .9 throughout. Thus update93 is the first equal-beta
update; training record93 still measures W92, before that update. Record94
is the first training loss after it. Update167 is the first cooldown update;
record167 itself still measures W166, but excluding it is a conservative
choice that preserves both pre-update and update-label conventions.

For records94–166 inclusive, exact full-window centers are:

| Width | Half-width | Record centers | Model-state centers |
|---:|---:|---|---|
| 11 | 5 | 99–161 | 98–160 |
| 21 | 10 | 104–156 | 103–155 |
| 31 | 15 | 109–151 | 108–150 |

Use no padding and no shortened endpoint windows. Interpolated crossings
must be bracketed by two valid reference centers, each with its entire raw
support inside94–166. Queries outside the supported curve or losses outside
its range should be explicitly censored/unresolved, not extrapolated.
Changing widths changes the eligible centers; it must not silently change
the scientific evaluation anchors or favor whichever width yields crossings.

## Partial final batch

Only update184 is partial: 9,510,912 tokens, versus 16,777,216 at every
update1–183. It lies outside the proposed common-regime analysis and therefore
needs no additional restriction there. The final training row still measures
W183, with a properly token-normalized loss over the smaller batch; the final
validation row measures W184 on the unchanged fixed bank and remains a valid
endpoint outcome.

If any supplementary full-run training average includes184, distinguish an
equal-update mean from a token-weighted mean. Do not treat the last update
as another full16M unit when converting steps to exposure. Its smaller batch
also changes its sampling precision. No endpoint training/validation losses
should be silently substituted for one another.

## Mathematical resolution witness

The proposed construction is sound **conditional on checking the strict
four-anchor order** R100>W100>R150>W150. It concerns validation-state indices,
so there is no training-row off-by-one correction here.

For each fixed delta family—constant5, linear10→2, linear1→9—construct a
strictly decreasing piecewise-linear reference through

    (100,R100), (100+delta(100),W100),
    (150,R150), (150+delta(150),W150),

then set Lwarm(t)=Lref(t+delta(t)) on100≤t≤150. The time-map slopes are
1, .84, and1.16, respectively; all are positive. All reference knots are
ordered, all constructed states are at most159, and each family exactly
matches the four observed anchors. Strict monotonicity gives precisely its
specified inverse lead. No favorable family needs selection.

This establishes nonidentification of the lead trend from **those four
validation anchors alone**. It is not an empirical evaluation at the invented
intermediate states, not an interpolation estimate of the actual trajectories,
and not a claim that all three curves are realizable optimizer dynamics.
It also does not satisfy or negate the independent dense training records;
those are the stated secondary evidence. Keep this proof of limited anchor
resolution separate from scientific outcome tables.

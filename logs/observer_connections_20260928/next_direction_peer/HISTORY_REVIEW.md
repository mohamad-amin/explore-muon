# Next direction: distinguish prediction confidence from the schedule's NLL gain

2026-09-28. Read current guide/state, observer STATE/SYNTHESIS, completed
functional reports and the latest 1x/2x momentum-warmup entries. Inspected
retained scalar values and final-checkpoint/config availability only. No
model, checkpoint-tensor, GPU, training or scheduler call. This is the only
new file. The early-split P1/P3 failures remain closed.

## Recommendation

**Prioritize a scalar output-calibration discriminator over a broad weight-
norm/gain inventory.** It tests whether the newly measured schedule NLL
advantage remains after giving each completed model the same independently
scored opportunity to correct global confidence. This is a distinct
functional question, not another local interaction, frequency-bin, replay
or stiff-energy analysis.

The completed 1x warmup fails its original target and the 2x warmup passes.
That is useful horizon-dependent schedule evidence. It does not identify
steps as a causal phase clock: optimizer statistics, auxiliary adaptation,
relative parameter changes and prediction confidence all evolve together.
Raw head/weight/gain norms would add correlations without determining which
changes matter to the loss. A confidence-sensitive derivative already exists
in retained function-level outputs and provides a concrete alternative.

## A new observable from retained scores

For a fixed model's logits z, define a global multiplicative scale a and
loss `L(a)=CE(a z,y)`. At a=1,

    dL/da = E_p[z] - z_y = CE(z,y) - H(p).

The completed `dose_function` probe retained both CE and predictive entropy,
so this is an exact scalar extraction with no new model call. The derivative
with respect to log(a) is the same at a=1. Positive values favor locally
softening logits; negative values favor sharpening them. This is a
temperature-direction diagnostic on this panel, not a complete reliability-
calibration assessment or the value of a fitted temperature improvement.

| State | Mean NLL | Mean entropy | Scale derivative L-H | Bank 0 / bank 1 derivative |
|---|---:|---:|---:|---:|
| Quarter, beta .9, step 9 | 7.055135 | 6.816074 | +.239060 | +.257504 / +.220616 |
| Quarter, beta .8, step 9 | 7.069564 | 7.002881 | +.066683 | +.073587 / +.059778 |
| Half, beta .9, step 9 | 7.092148 | 6.155042 | +.937106 | +.910704 / +.963508 |
| Half, beta .8, step 9 | 7.082994 | 6.247017 | +.835978 | +.810868 / +.861087 |
| Quarter, beta .9, step 46 | 5.418790 | 5.898458 | -.479668 | -.339875 / -.619460 |
| Quarter, beta .8, step 46 | 5.333492 | 5.855709 | -.522217 | -.379341 / -.665093 |
| Half, beta .9, step 46 | 5.417636 | 5.674041 | -.256405 | -.164478 / -.348332 |
| Half, beta .8, step 46 | 5.271925 | 5.369333 | -.097408 | +.034367 / -.229183 |

Early models with nearly equal NLL can differ greatly in this confidence
direction. Later models generally favor the opposite change. This is an
actual loss-sensitive distinction that weight norms cannot establish.
It also prevents a blanket confidence explanation: quarter-power beta .8
has better late NLL while its sharpening derivative becomes more negative.
The amount of recoverable loss depends on curvature along the scale and
cannot be inferred from the derivative alone.

These are exploratory observations from the already used eight-input dose
panel. They motivate a different fixed-family test; they are not new
confirmation of the dose or momentum mechanism.

## Proposed fixed-family test: six completed schedule endpoints

Use only the final SOAP-PD alpha .5, body LR .028, seed260925 Ada models:
constant beta .8, constant beta .9, and the declared .8→.9 warmup, each at
1x and 2x horizon. All six final checkpoints exist. Relative paths under
`logs/muon_spectra/` are:

- `soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada`;
- `soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada`;
- `soaudit_momwarm16m_20260928/SPD_a0.5_b16M_lr0.028_momwarm0.8to0.9_s260925_ada`;
- `soaudit_horizon16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.8_s260925_ada`;
- `soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada`;
- `soaudit_momwarm16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada`.

No intermediate-checkpoint hunt, alpha extension, LR selection or second
model family. Compare schedules **within each horizon**; do not confuse
the extra 2x training exposure with a calibration treatment.

Fix exactly two fresh 16-sequence banks before scoring, beyond 3.08B
training tokens and disjoint from previous observer score positions. Use
the same 512-token contexts for every model. Evaluate the fixed logit-scale
family `a in {.8,.9,1,1.1,1.2}`. The scale-one score is mandatory.

Use two-fold cross-fitting: for each model, choose its scale by minimum
mean CE on bank A and score that frozen choice on B; then choose on B
and score on A. An exact tie should choose the scale nearest one, with
a fixed secondary tie rule. All models receive the same candidate family
and opportunity. Report both held-out directions and their pooled estimate.
Do not choose a global scale by minimizing the same rows on which its
improvement is reported.

Retain every scale's per-sequence score, selected scales, raw and calibrated
NLLs, and paired schedule contrasts. A boundary winner is unresolved beyond
this fixed modest family; do not expand the grid. This is a finite
calibration opportunity, not a claim to have found the global temperature
optimum. Because only logits change after inference, no training or
head-optimizer repair is involved.

## Primary comparison and discriminating outcomes

The question is not "can any model be calibrated?" Many can. It is:

**Does the warmup-versus-constant NLL gap persist after independently
selecting confidence correction for both models?**

Fix warmup versus constant beta .8 as the primary contrast at both horizons:
beta .8 was the better constant by the original recorded validation scores.
Retain warmup versus beta .9 as the specified secondary comparison. Never
choose the comparator from a favorable new-bank result.

For each comparison report

    raw_gap = L_warm(1) - L_constant(1),
    calibrated_gap = heldout_L_warm(a_warm) - heldout_L_constant(a_constant),
    gap_change = calibrated_gap - raw_gap.

**Confidence explanation:** calibration preferentially helps the currently
losing schedule, consistently across both held-out directions, reducing
the original 1x disadvantage and/or the 2x advantage. An absolute
differential correction of .005 NLL is a reasonable prospective material
flag: it is substantial relative to the observed .0112 warmup-over-.8
2x margin. Report absolute values rather than unstable "percent explained"
when the new panel's raw gap is close to zero or changes sign.

**Prediction-shape/progress explanation:** models may all gain from
calibration, but their paired gaps remain substantially unchanged and
retain the original ordering in both held-out directions. That weakens
global confidence as the explanation of the schedule ranking, while
preserving all other forms of co-adaptation or temporal dynamics.

If the raw gap does not reproduce on the fresh panel, fold signs disagree,
or calibration remains boundary-limited, call the discriminator unresolved
and stop. A 32-sequence panel is not an absence/equivalence test. Do not
add inputs, temperatures or candidate checkpoints after seeing it.

## Cost and interpretation boundary

Six models x 32 inputs require **192 forward sequences**. The five scale
CEs reuse each sequence's logits; the additional full-vocabulary reductions
must be included in a first-sequence cost forecast. Use the already
qualified CPU FP32 forward, FP64 scalar reductions, two numerical threads,
and a 600-second forecast/wall boundary with margin. Stream logits rather
than retaining six models' vocabulary tensors. All predicted benefits are
measured with fixed checkpoints; no model/optimizer job is proposed here.

Validate scale-one reproduction, input/checkpoint/source hashes, finite CE,
fixed token coverage and the unscaled `dL/da=L-H` identity from logits.
Retain original scientific checkpoints unchanged and save any calibration
parameters only as analysis artifacts. Do not modify main validation
records or replace their benchmark endpoint with calibrated scores.

Even a large gap reduction would establish a confidence component of the
selected endpoint comparison, not an explanation of training-rate gains,
a causal step clock or a recommendation to change head LR. The calibrated
scale is an output-only intervention after training; it does not replay
how gradients, representations or momentum would have evolved. Generic
norm/gain inventories can wait until this functional distinction gives
them a specific role. The paused head-whitening line remains paused.

## Retained-input provenance for this review's scalar observation

Files under `logs/observer_connections_20260928/dose_function/run1/` were
read only:

- `analysis.json`: SHA256
  `2df39c0c4873d80c0f63a7cb35ffcca10e6b3068ea008b041e2d71bca5d7ae43`;
- `scalar_tokens.npz`: SHA256
  `6deebb82f67c72093fc10174d06e2aa3daef496ed7bc056183cdf6fe5038b2e8`;
- `result.json`: SHA256
  `285bab4d3d808924eca4ad36ae7fd65cdf90ed7d861c09e2b4b5d2b430ca329d`;
- `tokens.npz`: SHA256
  `7cb3eb00dac332fdf9a2d85b460f567da41860dcfd6750f97cab9886966d95fa`.

The new temperature study is a proposal requiring the parent's decision
argument and pairing/resource qualification; no such scoring was run by
this review. The candidate is meaningful because it tests a specific
alternative explanation of a completed training intervention, not because
it supplies another statistic to plot against steps.

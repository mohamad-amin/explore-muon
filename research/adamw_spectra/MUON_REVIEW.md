# Independent review of the Muon depth comparison

Reviewed 2026-09-24 from a fresh factual brief, AGENTS.md, RESEARCH_GUIDE.md,
the current state/protocol, model and measurement code, and the paper.
No sealed initialization results were read. No code or jobs were changed.
Live review check: no active goal object; allocation 2618555 remains on g20;
AdamW depth-20 job 2620278 is pending for Resources. The dated state records
the completed depth-8/depth-12 baselines; those results were not reanalyzed.

**Recommendation: proceed after the numerical prerequisites below.** The
three requested runs are useful as a bounded, paper-inspired descriptive
comparison of optimizer recipes at fixed width and data exposure. They
cannot isolate orthogonalization as the cause of any loss difference, establish
a general scaling law, or diagnose a head limitation. No additional scientific
sweep, seed, control arm, or depth is necessary for this descriptive objective.

The paper measures normalized momentum before NS, uses Muon LR 0.01 for its
spectral experiments and auxiliary AdamW LR 0.002 with betas (0.9, 0.95), and
presents the proposed five polynomial stages in Appendix A.2. Its Algorithm 1
uses plain momentum. The momentum coefficient, precision details and
experiment-specific implementation are not fully established by these
statements. Treat this as a documented reconstruction, not exact reproduction.
[Paper, sections 2–3 and Appendix A](https://arxiv.org/html/2606.04058v2).

## What the comparison can decide

At constant width, body matrix ranks/shapes stay fixed, avoiding the automatic
1/sqrt(rank) change in normalized singular values that occurs when width
changes. The paired AdamW/Muon recipes can show whether pre-NS concentration
and its depth dependence accompany different applied-update geometry under
the same architecture, data, horizon and initialization rule. NLL establishes
whether each trajectory describes useful training. Fixed token exposure is
not compute matching or equal tokens/parameter across depths; retain those
existing disclosures and compare against each matching-depth baseline.

My prediction before Muon outcomes: post-NS spectra will be substantially
flatter and may plateau even when momentum continues to evolve. An ideal
full-rank polar update has every Frobenius-normalized singular value equal
to 1/sqrt(rank), about 0.0442 here. Thus post-NS stabilization alone is largely
expected from the algorithm. Momentum stabilization across both optimizers
would be consistent with ordinary training dynamics; geometric differences
without NLL benefit remain descriptive. A weak Muon result at this one
operating point cannot reject Muon generally.

## Choices to freeze explicitly

- Momentum 0.95 without Nesterov is reasonable for this reconstruction.
  Specify `M <- 0.95*M + clipped_gradient`, zero initialization, FP32 state,
  and no momentum schedule. Do not silently substitute an EMA or Nesterov
  implementation. Such substitutions may preserve some scale-normalized
  quantities while changing the state or direction being measured.
- Use the six linear weight matrices per transformer block for Muon. Route
  token/position embeddings, head, LayerNorm parameters and all biases to
  auxiliary AdamW. A blanket `ndim == 2` rule would be wrong for this model.
  Assert exhaustive, disjoint parameter coverage.
- The proposed `sqrt(max(1, rows/cols))` multiplier is an implementation
  convention, distinct from the paper's initialization-scaling statement.
  Apply it using the original parameter orientation: factors 1 for attention,
  2 for MLP-up, and 1 for MLP-down. Preserve the existing model initialization.
  This choice is acceptable when labeled; it is not a recovered paper detail.
- Keep auxiliary LR 0.002 if prioritizing the requested paper-inspired Muon
  recipe, but call the comparison an optimizer-package intervention. The
  baseline auxiliary LR is 0.0012, and embeddings/head hold most parameters
  in the 8-layer model. Their changed learning rate is a major alternative
  explanation for NLL differences. Equal weight-decay coefficient also does
  not equal matched shrinkage: per-step decay is `1 - group_lr*0.01`.
  Record both group rates and decay norms; do not claim matched updates.

## Prerequisites, bounded to implementation qualification

1. **Measure what was actually used.** Preserve the exact normalized FP32
   momentum spectrum for paper comparison, but distinguish it from the
   rounded/epsilon-adjusted tensor entering BF16 NS. Document cast order,
   normalization precision/epsilon and polynomial order. Measure or capture
   the actual computed post-NS direction including aspect multiplier, before
   LR and decay. Recomputing with different precision would create a different
   instrument. Direct SVD of that direction is the required adaptive panel;
   a scalar polynomial applied to input quantiles is not equivalent because
   the polynomial can reorder singular values and BF16 introduces error.
2. **Verify the update and state contracts.** On square, tall, wide and zero
   matrices, check a clear reference polynomial implementation, measured
   direction against the actual parameter write after subtracting decay from
   pre-step weights, and no measurement mutation. Check both optimizer states
   on checkpoint/resume and all DDP replicas, including uneven final batches.
   Existing Adam-specific replica audits cannot simply be reused unchanged.
3. **Exercise the intended operating point.** Five updates with 50-update
   warmup reach only 10% of peak LR. Use the bounded full-size qualification
   at peak rate, explicitly as an engineering override, and exercise schedule
   transitions in the tiny check. This tests execution, finiteness and update
   scale, not LR quality or scientific loss ranking. Freeze the scientific
   schedule unchanged; do not turn qualification into a tuning search.
4. **Preserve the interpretation and resource bounds.** Retain 24 matrices,
   first/every-25/final sampling, full singular values and both norms, fixed
   validation, and the pre-cooldown summary window. Include doubled spectral
   collection in runtime forecasting and fit each run within its existing
   eight-hour/allocation limit. Report one-seed observations and hardware
   differences; wall-clock comparisons across different GPU counts/types
   are not optimizer speedup evidence.

These are finite implementation checks, not a request for user approval or
an expanded research program. If they pass, launch the already-authorized
three Muon runs and leave completed/pending AdamW baselines intact.

Implementation response accepted: use `warmup_steps=1` only for the five-step
full-size engineering qualification, reaching peak rates 0.01/0.002; retain
50 for science. Capture the FP32 direction after the actual BF16 polynomial
and shape scaling, and archive momentum separately. These resolve the two
principal design concerns. There is no further scientific prerequisite once
the stated implementation, replica/resume, finiteness and resource checks pass.

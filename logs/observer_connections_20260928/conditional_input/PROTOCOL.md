# Conditional-input candidate: qualification before numerical interpretation

2026-09-28. The preceding priority reset proposes a distinct question:
whether addition of a fixed opposing momentum history makes two fresh
gradient estimates much less repeatable relative to the resulting input.
This note records the scoped qualification decision, not an executed
gradient experiment. No real gradient or predictor arithmetic has been run.

## Missing premise and competing explanation

The replay and map audits distinguish a large raw fixed-state gradient,
stored history and the actual/constructed next input. If history nearly
cancels the common part of two gradient estimates, their unchanged absolute
difference can become large relative to the remaining input. The competitor
is that the relevant sums remain comparably repeatable, or that apparent
cancellation is below storage/accumulation resolution.

For conditioned history H and estimates g_A,g_B,

    b_A = H + g_A, b_B = H + g_B,
    b_A − b_B = g_A − g_B.

No new absolute perturbation or stochastic variance is created by H. The
question concerns relative magnitude/direction. It does not imply iid noise,
population SNR, a useful mapped direction or a training-rate mechanism.

## Proposed retained family and gating sequence

Only the two original 1M step-500 Muon/PD `gn2/*directions.pt` files named
in `../priority_reset_after_qk/REVIEW.md`. The proposed disjoint pair would
be g_A=g1M and g_B=(4*g4M−g1M)/3 **only after** historical same-offset
nesting, sample counts and loss units are established. Inputs marked `gd`
must have their stored negative sign accounted for. The matched frozen
training recipe uses unnormalized beta*M+g, beta .95, not an EMA factor.

Before reading cancellation statistics:

1. Qualify historical executable/dependencies, model state and sample
   membership from retained execution records, not current source alone.
2. Qualify sign, averaging units, parameter ownership, momentum convention
   and clipping scope. Body-only diagnostic gradients do not reconstruct
   full-model clipping, so a future qualified computation would still be
   a conditional unclipped predictor unless more information existed.
3. Propagate BF16 rounding intervals through the nested subtraction and
   history addition, retaining earlier FP32 accumulation uncertainty as a
   separate limitation. Zero/uncertain residual norms make angles unresolved.
4. Only after these pass, fix numerical readout/materiality/resource limits
   and consider a two-thread saved-tensor reduction with no model/GPU calls.

## Result of the first gate

The independent bounded provenance search is complete and the historical
membership gate **does not pass**. The available producer was edited after
the two outputs. Logs and JSONs bind arm/step/input labels, but omit executed
source hashes and the sampling calls. Current code supports the intended
nesting; it cannot certify those historical arguments. This is missing
evidence, not proof that the samples were nonnested or the outputs wrong.

Accordingly do not perform the proposed de-mixing, calculate cancellation
statistics, quietly replace the pair by an overlapping comparison, infer a
gradient from a polar direction, or replay new model gradients to repair the
archive. A later contemporaneous source/command/sample record could reopen
this gate. The remaining precision design is mathematical qualification,
not evidence that the empirical cancellation hypothesis holds.

The broad observer objective remains open. Closing this particular archive
interpretation is not a research blocker or an optimizer discovery. Retain
the exact distinction between raw gradients, conditioned history, nonlinear
maps, finite prediction changes and accumulated training progress when
choosing the next question. Main training outcomes continue to be ordinary
validation NLL; this qualification changes none of them.

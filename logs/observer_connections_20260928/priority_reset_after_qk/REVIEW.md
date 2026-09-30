# Priority after the Q/K functional test

2026-09-28. Read guide/state, observer synthesis, the completed Q/K result,
and source/schema information for older GN-direction archives. No model
call, new scoring, gradient-tensor arithmetic or optimizer experiment.
This is a candidate/gating discussion, not an execution protocol. Only
this review file is written.

## Do not continue component localization

The Q/K result is positive about transmission: the normalized-turn
factorial pattern reaches both isolated and conditional predictive KL.
Its declared conditional-usefulness premise fails. The larger motion
does not license a Q/K intervention at this selected point, but neither
does the conditional failure refute a long-run role for radius adaptation.
The same actual step can help alone, meet useful overlapping changes from
the complement, and add little or harm when applied last.

A new four-corner split of the remaining tensors would mainly repeat that
problem. Nor should the positive symmetric credit replace the chosen
criterion after the result. Close the fixed Q/K test with both its
positive transmission and conditional failure intact.

## The most worthwhile remaining candidate

**Does history cancellation make the next optimizer input substantially
less reproducible than the fresh gradient that enters it?**

This shifts attention from another parameter group's share to the object
the optimizer actually transforms. Several existing findings motivate it:

- Raw-gradient signal energy and the coordinates emphasized by the update
  are different objects; the earlier clipped SNR statistic was not a valid
  resolution of this issue.
- Fixed-weight replay has a large common component that model history
  largely cancels. Its recency-weighting term can be small compared with
  the common component and large compared with actual momentum.
- The new maps made explicit that the relevant incoming object is
  `beta*M_s+g_s`, not the stored lagged M_s alone.
- Strong spatial suppression and larger functional movement have now been
  measured without identifying a productive local component. Another
  instantaneous component score is unlikely to settle the rate mechanism.

The candidate is **conditional next-input sensitivity**, not a claim that
noise has already been shown to cause the gains, and not a new denoising
optimizer. If it is supported, a later method would need to reason about
uncertainty at the input remaining after history is added, rather than
dismissing uncertainty from raw gradient energy. If it is unsupported,
that particular extrapolation from the replay observation should close.

## Exact retained-artifact question

The existing `feedback_inventory/result.json` records that the later
`gn2/` files retain an unnormalized `gd` direction for `g1M`, `g4M` and
constructed `momentum`. The older `gn/` files do not retain gd and must
not be used to reconstruct a gradient by inverting polar.

The smallest candidate family is two original 1M step-500 states:

- `logs/muon_spectra/second_order_audit_20260926/gn2/`
  `M_lr0.007_s260925_l40s_step000500_directions.pt`;
- the same directory's
  `PD_a0.25_lr0.01_s260925_ada_step000500_directions.pt`.

Their JSONs identify the matching original `soaudit_traj_20260926` arms.
The corresponding saved optimizer checkpoint supplies the exact FP32
momentum buffer. The inventory confirms the direction keys/dtypes, but
this review has **not** loaded or numerically compared these tensors.

The current `one_step_gn.py:455–468` computes g1 from 2,048 sequences and
g4 from 8,192 sequences at the same `GRAD` offset and model state. If that
nesting is established for the historical outputs, the complementary
3M-token mean is

    g_B = (4*g4 - g1)/3,

with g_A=g1. The code's stored `gd` is the **negative** of each input.
Then compare the raw pair `(g_A,g_B)` with the conditional predictor pair

    b_A = beta*M_s + g_A,
    b_B = beta*M_s + g_B.

Use the recorded beta .95 for these selected states and the unnormalized
buffer convention. Do not add an EMA `(1-beta)` factor. Retain all 48 body
matrices, whole-body norms, signed inner products and cosines; any fixed
matrix-kind summaries are context, not an opportunity to choose whichever
group supports the hypothesis. No eigenframe reconstruction, SVD, mapped
update, GN percentage or predictive scoring is needed in the first check.

The exact difference identity is

    b_A - b_B = g_A - g_B.

History therefore does not amplify the *absolute* incoming perturbation.
It can change its size relative to the resulting input or rotate the two
predictors differently. Report raw differences and both denominator norms
together. A normalization-only rise in relative disagreement must not be
relabeled as newly created noise.

## Gates that must precede any numerical interpretation

1. **Historical source/membership.** Establish the same frozen model,
   token source, offset, sequence length and nested sample membership for
   the historical g1/g4 files. Current code and labels alone are not a
   complete execution provenance certificate: the inspected result JSONs
   record arm/step and input labels but not the complete sample/command
   contract. Qualification of that history remains outstanding. If nesting
   cannot be established, do not call the algebraic remainder a disjoint
   batch or silently substitute an overlapping comparison.
2. **Loss normalization and momentum units.** Verify gd is the raw mean
   token-loss gradient, not a per-matrix normalized/scaled direction.
   The current helper sums per-sequence mean-loss gradients and divides
   by sequence count. Verify actual buffer ownership/order and the .95
   recurrence from the matched checkpoint/config. The retained constructed
   `momentum:gd` offers an algebraic cross-check after accounting for its
   selected fresh-gradient input and quantization.
3. **Clipping/precision scope.** The diagnostic fresh gradients are body-
   only FP32 gradients; the stored buffer comes from the training history.
   Full-model clipping cannot be reconstructed from body-only gd without
   the auxiliary gradient norm. Even if nearby actual training steps were
   unclipped, do not assume a different held-out gradient was unclipped.
   Unless exact full clipping is qualified, label this a **conditional
   unclipped predictor**, not the actual next training step. Historical
   BF16 training versus diagnostic FP32 is another retained convention.
4. **Quantization/cancellation.** The gd tensors were saved in BF16.
   Propagate fixed BF16 rounding intervals through `4*g4-g1` and through
   addition of exact beta*M before reading small residuals or angles.
   Preserve signed lower/upper ambiguity instead of clipping a negative
   signal estimate. A BF16 interval is not a certificate for every earlier
   FP32 accumulation error; distinguish those sources. If subtraction or
   predictor cancellation is unresolved at the archive's precision, stop
   with that limitation rather than repairing it by a new model replay.

These are real outstanding gates. No numerical gradient/predictor outcome
has been inspected in this review. Complete source coverage can be addressed
in a later bounded pass before deciding whether the actual reduction is
worth executing.

## What the result could change—and what it cannot

The discriminating observation would be raw gradients agreeing closely
while the two b inputs disagree substantially more in relative magnitude
or direction, robust to their quantization intervals. That would expose
a concrete weakness in using raw-gradient repeatability to describe the
optimizer's conditional input. If both pairs agree similarly or history
improves agreement, the simple amplification picture is unsupported.
Mixed-state outcomes remain mixed; do not select a favorable state.

There are only two contiguous unequal-sized data blocks. Their difference
includes content variation, not certified iid noise. Conditioning on one
saved M also excludes uncertainty in that history: the test does not
measure the unconditional stochastic variance of momentum. No extrapolation
to 16M, a population SNR, a useful polar direction or a training-rate gain
follows. In particular, a noisy parameter-space input can still be quiet
in predictions, as the earlier TS study showed.

Only memory-mapped reads of the few relevant gradient/momentum tensors
and FP64 reductions would be needed if the gates pass: two CPU threads,
no model, with an explicit short resource cap and no expansion to all
archived states. This is much smaller than another forward factorial.

## High-level reset

There is **no strong new optimizer intervention yet**. The most useful
theory synthesis now separates:

- population/fixed-state gradients;
- stored history and constructed next input;
- nonlinear normalized maps and adaptive radii/statistics;
- finite predictive movement;
- slow accumulated progress along the trained trajectory.

The project has positive evidence at several of those levels, but their
causal links do not follow from substituting one local score for another.
Any future dynamics account must accommodate the failed instantaneous-edge
law, phase-dependent momentum benefit, functional transmission without
conditional usefulness, and the observed cancellation of fixed-state mean
gradients by history. A frozen stationary quadratic or globally clean
period-two model is not already established by the retained 1M lag data.

The conditional-input discriminator above is one concrete next artifact
question serving that synthesis. It is not permission to restart a closed
Q/K, nonlinear-channel, confidence or frequency panel. If its source or
precision gates fail, report that no strong empirical successor is yet
qualified and continue the theoretical/evidence synthesis; this is a
research-priority conclusion, not a declaration that the broad goal is
blocked or complete.

# Next early-coupling discriminator: finite loss before subgroup attribution

2026-09-28. Read the current guide/state, latest dose/warmup entries,
completed dose-function report, and the qualified auxiliary-factorial
protocol/source. No model, GPU, scheduler or checkpoint-tensor call.
This is the only new file.

## Recommendation

**First measure the exact additive-logit/mixed-loss split, with a stage
control; defer the 16-corner subgroup factorial.** The missing premise is
whether the newly observed negative base-residual mixed term actually
improves finite loss when applied after the additive body/auxiliary
prediction. Group attribution is premature until that is known.

The current evidence does establish a real early difference: half-power
PD has negative total body–auxiliary interaction at step 9, with negative
J on every scored sequence; quarter-power has positive interaction. By
step 46 the interactions are positive. But J is not the finite mixed-loss
effect. A full subgroup factorial could identify a component of J or of a
total interaction while leaving its useful finite-response interpretation
ambiguous. Four selected states with the exact split are the cleaner next
discriminator at similar or lower CPU cost.

This is a new narrow function-level question, not an expansion intended
to recover the failed uniform-amplitude story. The original dose-function
probe remains complete with its declared conclusion intact.

## Precise competing explanations

At a fixed saved state let z0,zB,zA,zF be base/body/auxiliary/full logits,
and let `zadd=zB+zA-z0`. Measure

    I_overlap = CE(zadd)-CE(zB)-CE(zA)+CE(z0),
    I_mixed   = CE(zF)-CE(zadd),
    I_total   = I_overlap + I_mixed.

The first term is the interaction of additive prediction changes; its
label terms cancel. The second is the actual finite loss effect of adding
the model's mixed-output response along this specified logit path.

**Working hypothesis:** early half-power PD has a materially helpful
finite mixed-output term, stronger than quarter-power, and that benefit
weakens by step 46. This would give the early J observation a concrete
finite-loss meaning and establish a stage-dependent whole-model phenomenon.

**Strong competitor:** the negative early total interaction mainly comes
from complementary/anticorrelated additive prediction changes. J is
negative at the base loss residual, but its finite benefit is canceled
by the accompanying change in KL when the mixed response is added after
zadd. Then localization of a "helpful nonlinear transport" route would
be pursuing the wrong object.

Both explanations can produce the existing J/I signs. The retained scalar
KLs cannot distinguish them because `KL(p0||padd)` was not measured.
This is a genuinely missing observable, not another ratio of old numbers.

## Smallest complete comparison

Fix beta .9 and use quarter/half-power PD at steps **9 and 46**: four base
states, each with its actual saved next-step weights. Beta .9 is the
declared reference recipe; both betas already show the early sign pattern,
so duplicating beta .8 adds less information than the later-stage control.
Do not add step 83, a new momentum or a selected loss-matched checkpoint.

Use all original eight dose-function inputs for reproduction/discovery,
plus exactly two new four-sequence banks selected before scoring. Every
state receives the same inputs. On old inputs, reproduce the original
four CE corners and J/I values before interpreting the new zadd quantity.
Fresh inputs are a consistency check at selected states, not an independent
training-seed or causal confirmation. Keep their results separate from
the old inputs that motivated the question.

This gives four states x 16 inputs x four parameter corners = **256 model
forwards**. zadd and its CE/KL use already computed logits and require no
fifth model forward. The previous 256-forward dose probe took 139 seconds;
extra FP64 logsumexp/mixed reductions add cost, but a 600-second cap with
a first-complete-sequence forecast is plausible on two CPU threads.
Preserve any cost stop rather than reducing the sample after starting.

At similar cost, an early-only 16-corner factorial would give more subgroup
labels but no within-arm stage control. Reusing its qualified head shortcut
is not a reason to ask the larger attribution question first.

## Reading fixed before the new score panel

For the fresh inputs, keep the prediction simple and directional:

1. Half-power step-9 `I_mixed` is negative in both banks, with pooled value
   at most **-.01 NLL** to count as a material helpful term.
2. The paired half-minus-quarter `I_mixed` contrast at step 9 is negative
   in both banks, with pooled value at most **-.01**.
3. Half-power `I_mixed` becomes less favorable at step 46 in both banks.

The .01 material floor is prospective for this follow-up, well above the
numerical gate and a meaningful part of the previously observed .035–.047
negative total interaction. It is not a significance threshold. Report all
values even when these criteria fail; no bank/stage may be dropped.

Retain `I_total`, `I_overlap`, `I_mixed`, directly computed J, and
`KL(p0||padd)`. Check both exact identities:

    I_total = I_overlap + I_mixed,
    I_mixed = J + KL_full - KL_add.

Keep absolute values and signed terms. Do not report a fraction where the
total is near zero or where cancellations make the percentage misleading.
Base/full/body/auxiliary NLL changes remain useful context. No extra
normalization, response rescaling or optimizer quality score is needed.

Use the same frozen CPU FP32 model helpers and FP64 reductions. Preserve
the existing old-corner reproduction tolerances, exhaustive body/auxiliary
partition and fixed input hashes. Require the scalar split identities to
the previously qualified FP64 tolerance. The newer observable is CE of
an algebraically constructed logit response; zadd is not a trained model
or a realizable parameter update by assumption.

## Decision after this one comparison

If the material fresh-bank predictions hold, report a helpful early mixed
model-output contribution along the specified finite-logit path, with its
stage qualification. That would justify considering a later, separately
designed subgroup-attribution experiment if it serves a concrete question;
it does not automatically launch the 16-corner factorial.

If the mixed term is small, adverse or inconsistent while additive overlap
accounts for the negative interaction, close the helpful-mixed-response
premise. Preserve J as a correct base-linear descriptor whose sign did
not identify finite benefit. If results are ambiguous, close this fixed
panel without more states, samples or group combinations.

No outcome makes this term the mediator of the momentum rate gain. The
favorable early half-power interaction exists at beta .9 even where its
final loss loses to quarter-power. The new main warmup result, 3.962814,
is separate positive single-case evidence for the practical schedule
prediction; it should not be used to assign causality to an unrelated
early local term. The head-whitening repair line remains paused, and
neither this design nor subgroup attribution would authorize reopening it.

This narrowly completes the finite functional interpretation before
localization. It avoids turning one interesting sign reversal into an
unbounded local experimental cascade.

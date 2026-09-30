# Is the positive Q/K angular result functionally consequential?

2026-09-28. Read current guide/state, `angular_clock/REPORT.md`, its
`FUNCTIONAL_AVAILABILITY.md`, the closed confidence report and recent main
schedule notes. No model scoring, checkpoint-tensor load, GPU or job call.
This is the only new file.

## Decision

**The proposed Q/K functional probe is worth doing, with a factorial
transmission prediction fixed before scoring.** Merely measuring nonzero
Q/K KL would be too weak: any nontrivial update may do that. The useful
question is whether the *specific* momentum/LR pattern found in normalized
head turns also appears in actual predictions, including the background
change made by the rest of the update.

This is a better next use of bounded CPU work than another search for a
phase clock. The angular result supplied a positive structural premise;
the exact relevant functional quantities are absent from the archive.
Other candidate lines are presently less compelling: the fixed confidence
family selected the identity everywhere; useful early mixed nonlinearity
failed; rare-target specificity did not survive its progress controls;
and the original replay rows cannot disentangle corpus age/content by
another regression. None should be reopened to obtain a favorable result.

## What the angular evidence predicts—and what could defeat it

At step 46 the observed beta .8/.9 matched-head turn ratios are about
1.15–1.25 at the two LRs. Increasing LR by 43% raises turns only about
11–19%, because radii are larger. Absolute tangent writes tell a different
story from relative turns. The effects are broad across Q/K heads and
develop after the early state, so this is more than an isolated norm datum.

**Working functional prediction:** the larger beta-.8 Q/K turns produce
larger Q/K prediction changes, while radius adaptation partially absorbs
the functional effect of the higher LR.

**Simple competitor:** the extra weight-direction motion falls in weakly
excited input directions, is attenuated by attention/normalization, or
changes significance when the simultaneously updated gains and remaining
network are present. Then the parameter-space pattern need not transmit
to predictions. A further possibility is that Q/K produces substantial
predictive motion whose finite conditional loss effect is adverse at the
actual step. Neither large motion nor low motion alone establishes useful
optimization.

The completed TS study is a precedent for taking this competitor seriously:
large parameter-space differences became much smaller predictive changes.
Here the objects are actual Q/K writes in the matched LR-by-momentum
family, so the new measurement would not repeat that old estimator test.

## Fixed family and four corners

Use only the four already paired SOAP-PD states at **46→47**:
beta {.9,.8} crossed with body LR {.028,.04}, alpha .5, 16M batch,
seed260925, Ada. No extra checkpoint or head selection. Step 46 is an
existing declared midpoint with a developed angular effect; it is not
chosen by future functional scores.

Partition the actual saved update into:

- QK: the 16 query/key weight matrices across eight blocks;
- rest: all 68 remaining tensors, including Q/K normalization gains,
  V/O/MLP matrices, embeddings, head and other gains.

Score base, QK-only, rest-only and full next model on exactly two fixed
fresh banks of eight sequences each, context 512. All four states see
the same inputs. The gains deliberately remain in the complement: this
tests the role of the Q/K *weight* motion singled out by the angular
analysis and whether it survives the actual changing background.

Exact endpoint copying must account for every parameter once. Q/K norm
gains must not be accidentally included by matching a broad `attn.q`
prefix. No radial-only, tangent-only, headwise or gain-only extra corner
is needed in this pass.

## Minimum functional readout

Retain all four NLLs and base-to-corner forward KLs. Also compute, from
the same four forwards,

    KL(p_rest || p_full),
    Delta_Q|rest = L_full - L_rest.

The first measures the actual Q/K addition after the rest has moved;
the second is its signed conditional loss effect. Both are needed to
avoid deciding from an isolated Q/K step at the old background alone.
Retain Q/K-only `L_QK-L_base` as the corresponding unconditional effect.

Report exact finite-loss interaction
`L_full-L_QK-L_rest+L_base`. It need not vanish and its sign does not
identify waste. The previous probes already established that additive
prediction interaction can be useful or costly. Do not reopen synthetic
additive-logit or subgroup attribution here.

Predictive cancellation must be described carefully. A small full KL
beside larger isolated KLs does not by itself provide an additive energy
decomposition. These divergences have different pairs of distributions;
retain their absolute values and the explicit conditional quantity rather
than assigning a "fraction of prediction change" to Q/K. No GN, full
Hessian or mediation claim follows from this finite experiment.

## Direction-changing criteria without an arbitrary component share

Predeclare the following structural-to-functional predictions, tested
separately on both banks:

1. **Momentum translation:** beta-.8/.9 Q/K-only predictive KL is greater
   than one at both LRs. Repeat the same contrast using the conditional
   `KL(p_rest||p_full)` rather than replacing the first endpoint with it.
2. **LR compensation:** LR-.04/.028 Q/K-only KL is below the nominal
   squared-step multiplier `(.04/.028)^2 = 100/49`, at both betas. This
   is a dimensional prediction from partial compensation, not an exact
   law or a fitted target. Retain the analogous conditional ratios too.

These tests ask whether the observed ordering/compensation reaches
predictions; they do not require Q/K to exceed an invented fraction of
the whole-model KL. Report absolute KLs, all individual sequence readings
and paired bank values, so a numerically tiny denominator cannot manufacture
a pass. Fixed numerical repeat/error checks set the resolution floor;
do not create a post-hoc "material share" cutoff.

Then keep the signed loss reading distinct. If `L_full-L_rest` is negative
in both banks for each beta-.8 arm, the enlarged Q/K response is locally
helpful at the actual step. If its effect is positive, mixed or unchanged
while movement increases, retain a functional-motion result without calling
it productive. Comparing its beta contrasts is descriptive because the
base states are different; it cannot rank accumulated training progress.

**If both functional pattern tests transmit in both backgrounds**, the
radius/turn observation becomes a real prediction-level link and remains
a candidate for a later experiment separating memory from relative motion.
It still does not authorize a norm constraint or new momentum schedule.

**If the weight ordering fails to transmit or depends strongly on the
updated complement**, do not continue toward a Q/K-radius intervention
on the strength of angular statistics. Report that the structural proxy
has not met its functional premise. This is not a declaration that Q/K
is generally irrelevant; it is a reason to stop this specific account.

**If input-bank disagreement or numerical resolution prevents a decision**,
close this fixed test unresolved. No more heads, states, input banks or
response rescalings follow merely to rescue the pattern.

## Cost, source and scope

Four states x 16 inputs x four corners gives **256 forwards**. Earlier
qualified probes of this size took roughly 140–207 seconds; computing
the extra conditional KL needs reductions, not more model passes. Time
one complete four-corner sequence plus paired-load/hash overhead and
forecast against a fixed 600-second cap, CPU FP32/FP64 reductions and
two threads. Available scoring offsets must be checked against the 3.2B
stream and recorded prior intervals before any model scoring.

Use the qualified common helper and actual saved weights, with source and
token hashes, strict corner reconstruction, restored-base repeat checks,
finite/nonnegative KL tolerances and ordinary FP32 CE qualification.
Preserve all four states, both banks and every outcome. A cost stop is an
execution outcome, not a scientific null.

The experiment does not measure the causal benefit of radius changes,
an intrinsic learning rate, gradient persistence or the optimizer's
differential feedback. It supplies the missing functional premise for
one positive, architecture-grounded observation. That is a meaningful
bounded step; repeating parameter-space statistics would not supply it.

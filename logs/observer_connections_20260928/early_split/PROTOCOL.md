# Does the early mixed-output response actually help finite loss?

2026-09-28, before new model scoring. The previous dose-function probe is
complete and its conclusions remain intact. Independent high-level review:
`../early_coupling_peer/REVIEW.md`. This is a distinct missing-observable
test, not an extension to recover the uniform-amplitude hypothesis.

## Missing premise and alternatives

Early half-power PD has negative total body/auxiliary loss interaction and
negative J=(p0−e_y)·mixed_logits. But J scores the mixed response at the base
residual. Its actual finite benefit after the additive prediction changes
may be canceled by an accompanying predictive-KL change.

At each four-corner state define zadd=zB+zA−z0 and measure exactly:

    I_overlap = CE(zadd)−CE(zB)−CE(zA)+CE(z0),
    I_mixed = CE(zF)−CE(zadd),
    I_total = I_overlap + I_mixed.

The first term is label independent. The second is the finite loss effect
of adding the mixed output response last, along this specified logit path.
zadd is a synthetic response, not assumed to be a realizable parameter update.

Working hypothesis: early half-power PD has a materially helpful I_mixed,
stronger than quarter-power, and the benefit weakens later. Competitor:
negative additive overlap explains the early interaction, while J's apparent
benefit is canceled in the finite mixed-loss effect. Group attribution is
premature until these alternatives are distinguished.

## Fixed states, inputs and predictions

Use only beta .9, quarter/half-power PD at saved 9→10 and 46→47: four states.
Use all original eight `dose_function/run1/tokens.npz` inputs for reproduction,
then two new four-sequence banks at training offsets 3,170,131,072 and
3,195,131,072. Context is 512. All states receive exactly the same 16 inputs.
Fresh inputs check consistency at selected states, not independent training
seeds or causal confirmation. No beta .8, step 83 or subgroup factorial.

Fresh-panel readings fixed before scoring:

1. Half-power step-9 I_mixed is negative in both fresh banks, with pooled
   mean <= −.01 NLL for a material helpful effect.
2. Half-minus-quarter I_mixed at step 9 is negative in both banks, with
   pooled mean <= −.01.
3. Half-power I_mixed becomes less favorable at step 46 in both banks.

The .01 floor is a prospective materiality criterion, not significance.
Keep all values, absolute signs and cancellations; do not choose a fraction
when the total is small. If these fail or are ambiguous, close the fixed
comparison without more inputs, states, groups or fitted rescaling. If they
hold, a later separately designed localization test may be considered; no
training remedy follows automatically. No result identifies the mediator
of the main alpha×beta training gain.

## Qualification and implementation

Frozen CPU FP32 helpers from the completed dose probe; two threads, no CUDA,
gradients or optimizer. Preserve exact full/body/aux corner loading and
checkpoint/token hashes. On the original inputs reproduce every corner's
per-token FP64 NLL and forward KL within 2e−5 maximum and 2e−6 per-sequence
mean. These gates are set before execution and will not be silently relaxed.

Compute log probabilities in FP64. The synthetic additive response can be
formed as lpB+lpA−lp0, followed by normalization: it differs from zadd only
by a per-token common shift. Retain I_total/I_overlap/I_mixed, KL(p0||padd),
and directly computed J using the four log-probability vectors. Check per
token, to 1e−10:

    I_total = I_overlap + I_mixed,
    I_mixed = J + KL_full − KL_add,
    J = I_total − (KL_full−KL_body−KL_aux).

KL must be finite and >= −1e−10; no probability or loss clipping. Retain
all per-token scalar terms, but not enormous full-vocabulary logits.

Cost: four states × 16 inputs × four corners = 256 score forwards. The
additive-logit CE/KL needs no fifth model call. Time the first complete
sequence and load; forecast the remaining 63 sequences and three loads
with 25% margin plus 20 seconds. Stop before the remaining work if forecast
exceeds 600 seconds. Check the 600-second wall boundary after each sequence.
All outputs/failures stay here; no model or scheduler action elsewhere.

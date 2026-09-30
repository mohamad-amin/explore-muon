# Does the whitening×momentum interaction conceal a functional step-size change?

2026-09-28, before new model scoring. Independent high-level discussion:
`../dose_interaction_peer/REVIEW.md`. The main quarter-power dose experiment
has completed; this observer will not manage or duplicate its training.

## Positive evidence and missing mediator

At16M/commonLR.028, changing beta.9→.8 improves PDalpha.25 by.08627 and
alpha.5 by.12360NLL. The predicted intermediate gain and alpha×beta interaction
are supported in this one-seed recipe. Changing alpha also changes allocation,
functional step size, state adaptation and auxiliary gradients, so the
intervention does not uniquely identify stiff-oscillation filtering.

The bounded new question is whether matched nominal body norms hide a
systematic predictive-amplitude difference. Uniformly smaller body KL under
stronger whitening would keep a scalar functional-amplitude competitor alive.
Similar/larger KL would weaken that blanket account. Neither result proves
the proposed directional mediator. Identical auxiliary settings need not
give identical auxiliary prediction changes; include their full interaction.

## Fixed comparison

Four complete PDarms: alpha{.25,.5}×beta{.9,.8}, allLR.028,seed260925,L40S,
16Mbatch and92steps. Retain their paired full loss trajectories and source/
config checks. Use saved step9→10 and46→47 from every arm: eight own states.
These are actual adjacent-checkpoint displacements, including decay and write
rounding; they are not lagged-momentum proxy maps or a common-state intervention.

At every state score exactly four corners: base,48body matrices updated,
all36auxiliary tensors updated, full next weights. All84parameters must be
covered once. Score two fixed banks of4sequences each,512context, offsets
3,180,000,000 and3,190,000,000 in the training stream. These are beyond each
selected run's1×training budget. Retain exact tokens; no score-guided selection.

Primary amplitude is mean forwardKL(p_base||p_corner), label independent,
for body/aux/full. Also retain reverseKL and their symmetric mean as fixed
checks of finite-step asymmetry; never choose a favorable divergence after
seeing results. Record base predictive entropy, per-sequence NLL at every
corner, and exact finite loss interaction L_full−L_body−L_aux+L_base.
Record full/body/aux parameter-displacement norms separately.

Report alpha.5/alpha.25ratios at each beta/stage/bank, and beta.8/beta.9ratios
at each alpha/stage. Ratios are descriptive across different own states,
base distributions and geometries. Keep all signs and banks; no inference
of mediation, optimal LR or rate from a KL/NLL ratio. A uniform amplitude
story requires a consistent direction across both stages/betas/banks. Mixed
ratios close that simple reading; they do not refute every amplitude effect.
Do not extend states/samples, tune a rescaling or launch a remedy.

## Qualification and cost

CPU FP32eager,2threads, no CUDA/gradients/optimizer. Use the already qualified
observer helpers. Require matched initialization, data, hardware, architecture
and active settings apart from alpha/beta. Preserve newer disabled-option
source differences. Strict corner loads must equal the corresponding saved
parameters. On the first input, restored-base loss/logits must reproduce
within2e−6; FP64CE against ordinaryFP32CE within5e−5. KL must be finite and
nonnegative up to1e−10; identical logits give zero. Analyze log probabilities
and divergences inFP64, retainingFP32losses as well. All KLs average over the
same real tokens; no logit gauge normalization or probability clipping.

Time one complete four-corner sequence and one paired checkpoint load.
Forecast remaining63four-corner sequences plus7loads with25%margin and
20seconds overhead. If over600seconds, stop and retain qualification. Check
the600-second boundary after every sequence; no newGPUor sample reduction.
All256score forwards, raw scalar arrays, sources, input/checkpoint hashes,
cost stops/failures and interpretations remain here.

## Post-hoc scalar identity after the completed primary readout

The early body–auxiliary loss interaction changed sign with alpha. Without
new model calls, derive J=I_NLL−(KL_full−KL_body−KL_aux) from the retained
per-token scalars. Exactly J=mean[(p0−onehot_y)·(z_full−z_body−z_aux+z0)].
Retain all8states, all8inputs and both banks. This measures a label-relevant
mixed-logit response at the base distribution; it vanishes for additive
logits. It is invariant to per-corner common logit shifts.

This is not the earlier aux_partition finite mixed-loss term CE(z_full)−
CE(z_add); that also requires KL(p0||p_add), which is not retained here.
Do not infer an overlap fraction, a measured residual Hessian, an auxiliary
subgroup, or training mediation. The identity-based reduction is explicitly
post-hoc and does not replace the predeclared amplitude endpoints. No extra
state, input, fitted rescaling, forward or remedy is added.

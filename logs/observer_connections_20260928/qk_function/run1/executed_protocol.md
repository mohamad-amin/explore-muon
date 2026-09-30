# Does the angular-step factorial pattern reach useful Q/K prediction changes?

2026-09-28. Fixed before new scoring. Independent high-level discussions:
`../qk_function_peer/GEOMETRY_REVIEW.md` and `HISTORY_REVIEW.md`. This is a
missing-functional-premise test of the positive angular observation. The
confidence grid, early mixed-response hypothesis and earlier auxiliary
factorials remain closed. No training intervention is proposed.

## Decision argument and competing explanations

At the four SOAP-PD LR×beta states, shorter beta creates larger normalized
Q/K weight turns at equal head-step norms, while growing radii absorb much
of a 43% LR increase. That is not a measured prediction change. Existing
profiles retain only whole-body totals for the low-LR states and no matching
functional decomposition for the high-LR states.

The useful lead predicts that the same factorial pattern reaches Q/K's
actual finite predictive movement and that its contribution to the realized
next model helps held-out loss. Competing explanations are that state/input
geometry absorbs the angular difference, or that the Q/K response is
cancelled, background-dependent or not locally useful. A nonzero Q/K KL
alone would not discriminate these accounts.

A fixed four-corner comparison supplies the missing terms without another
training run. It can decide whether a further loss-directed Q/K question is
motivated at these selected states; it cannot prove Q/K mediates the rate
gain, explain the whole trajectory, or derive a universal phase clock.

## Fixed states, groups and score inputs

Use only 46→47 in the existing one-seed, Ada, 16M SOAP-PD alpha-.5
factorial from `../momentum_lr/analyze.py`: LR {.028,.04} × beta {.9,.8}.
Step 46 is the predeclared middle, constant-LR retained state, before
cooldown and after the angular contrast has developed. Do not add steps
9/83, another seed, optimizer, head or chosen loss-matched state.

QK comprises all 16 query/key weight tensors. Its complement R comprises
the other 68 model parameter tensors, including all RMS/QK gains, embeddings,
head and the other 32 body matrices. Corners are:

0. All base weights W46.
1. QK at W47, R at W46.
2. R at W47, QK at W46.
3. All actual next weights W47.

These are the actual saved writes, including decay/rounding. No per-head
canonicalization, radius matching or step rescaling. Full is the actual
next model, not a body-only approximation.

Two fresh banks of eight 512-token sequences start at training-stream
offsets 3,100,131,072 and 3,110,131,072. Each reads 4097 source tokens for
4096 targets. Both are inside the verified 3.2B-token stream and beyond
the 1.5399B-token training budget of all four selected runs. They are
disjoint from previous observer scoring intervals and identical across
states. Banks are contiguous text; they are not independent training seeds.

## Exact finite readouts

Let z0,zQ,zR,zF and corresponding probabilities denote the four corners.
Primary predictive amplitude is forward KL. Retain:

- K_Q = KL(p0 || pQ), K_R = KL(p0 || pR), K_F = KL(p0 || pF).
- K_Q_last = KL(pR || pF), K_R_last = KL(pQ || pF).
- QK-alone loss LQ−L0 and QK-last loss LF−LR; both complement margins;
  full loss LF−L0 and I=LF−LQ−LR+L0.

For each isolated corner retain the exact label-linear term
E_p0[delta_z]−delta_z_y, so its CE change is this term plus its forward KL.
This is linear in a finite logit response, not a weight-gradient/JVP score.

Using p0 as a common within-state metric, center log-probability differences
vQ=log(pQ)−log(p0), vR and vF by their p0-weighted vocabulary means.
Retain their 3×3 covariance per target. This is a finite logit-response
covariance, **not** a tangent GN matrix. It preserves signed Q/R overlap,
the actual full response and the mixed response vF−vQ−vR. Cross-state base
distributions differ, so this is an own-state comparison, not a common-state
curvature intervention. Do not report a fraction of full movement by
discarding cross terms or dividing by a small/cancelling full quantity.

The same logits also provide the exact additive-logit loss split using
zadd=zQ+zR−z0: I_overlap=CE(zadd)−LQ−LR+L0 and
I_mixed=LF−CE(zadd). Retain them to distinguish additive interaction from
finite nonlinearity; no new helpful-nonlinearity hypothesis or group search
is introduced. No scale is fitted to a direction or to predictions.

## Predictions and stopping readings

These are retrospective structural predictions fixed before the new score
panel, not significance tests or necessary conditions for all training gains.

1. **Angular pattern reaches isolated Q/K predictions:** K_Q(beta .8)/
   K_Q(beta .9)>1 at both LRs in both banks, and K_Q(LR .04)/K_Q(LR .028)
   <(.04/.028)^2=2.040816 at both betas in both banks. Report all absolute
   K_Q values, ratios and their distance from these bounds; merely passing
   by a tiny amount is not a large functional compensation claim.
2. **The pattern survives the realized background:** the same two ratio
   conditions hold for K_Q_last, with all other parameters at their next
   values. A contradiction between alone and last is background dependence,
   not a reason to select the favorable order.
3. **Local usefulness at shorter beta:** LF−LR is negative in both banks
   at each LR, with pooled value ≤ −.005 NLL at each LR. Retain the alone
   margin and both longer-beta arms as context. The .005 floor is a declared
   materiality flag, not a confidence threshold or an optimizer-ranking test.

If all three pass, the positive weight-space observation has a functional
and locally useful counterpart worth a separately designed causal question.
It still does not prove mediation, or justify changing radius/memory.
If any fail or are ambiguous, close this fixed bridge test without more
states, heads, input banks, fitted scales or a revised criterion. Preserve
partial positive results, bank disagreement and signed interactions.
Negative local usefulness does not prove a component was useless throughout
training; it prevents this selected-state result from licensing a remedy.

## Qualification and bounded execution

Same qualified frozen FP32 eager CPU helpers as `confidence_calibration/`;
FP64 reductions, two numerical threads, no gradients/optimizer/CUDA.
Recheck four-arm configuration/data/init pairing, completed status and
scientific source hashes. Actual checkpoint steps/tokens and strict tensor
partitions must match; record consumed full-model hashes and compare the
48 body-tensor hashes with the preceding angular probe at step 46.

Every corner's FP64 CE must match standard FP32 CE within 5e−5 per token.
KLs must be finite and ≥−1e−10. Exact CE/KL and interaction-split identities
must hold to 1e−10 absolute. Response covariances must be symmetric/PSD
within 1e−10 times max(1, largest diagonal), and exact covariance expansion
must match the directly reduced mixed response. A fixed synthetic example
qualifies vocabulary-shift invariance and the covariance algebra before
real scoring. Strict corner tensor equality is checked on each state's
first input; restore base and reproduce its logits exactly once.

Cost: four states × 16 sequences × four corners = 256 scientific forwards,
plus one restoration check. Forecast after the first complete sequence,
including all covariance/KL/additive-logit reductions and load/hash costs,
with 25% margin plus 20 seconds. Stop if projected total exceeds 600 seconds;
check the wall boundary after each sequence and preserve every partial/failure.
All files remain here. No main GPU job, training source, original checkpoint,
validation record or sealed-surrogate data is changed.

## Choice between the independent suggested readouts

Both reviews endorse one functional-premise test, but suggest different
primary summaries. Before scoring, the root adopts the history review's
two-factor KL transmission pattern and actual-background loss margin.
The geometry review's finite-response RMS ratio and symmetric two-order
loss credit remain descriptive exports, not substitute acceptance criteria.

KL directly measures changed predictive distributions and lets the LR
contrast be compared with the fixed nominal squared-step ratio. The QK-last
margin asks what those exact writes add when every other parameter already
has its actual next value; it is explicitly conditional, not a unique
attribution of loss. The alone margin and symmetric credit retain the
order dependence rather than hiding it. The .005 material floor is fixed
before results and is not a test of a training-rate mediator. No review
proposal is silently substituted after an unfavorable outcome.

# Priority review: test the functional relevance of larger Q/K turns once

2026-09-28. Independent high-level discussion using the completed angular
comparison, its functional-availability inventory, and the closed confidence
and early-mixed-response results. No checkpoint tensors, model calls, scores
for the proposed probe, GPU or training jobs were read or run. Only this
review is written.

**Verdict: one bounded functional premise test is worth doing.** The angular
comparison supplied a positive, reproducible structural difference in the
existing LR×beta family; whether that difference reaches useful predictions
is exactly the missing premise. The proposed four actual-update corners are
a direct way to test it. This is preferable now to another norm inventory,
a finer confidence grid, resurrecting mean-route attribution, or a generic
stiff-mode remedy. It does not warrant a per-head factorial or an intervention
that fixes angular step sizes.

## Keep the scope small and physically well defined

Use only the four SOAP-PD own states at **46→47**, LR {.028,.04} × beta
{.9,.8}, and the same two fixed fresh banks of eight sequences each. This
is the already measured point at which the angular difference is present;
it is a selected local premise test, not independent confirmation of a
trajectory mechanism. The two LRs make the existing compensation connection
testable without adding another family or time point.

The four corners are base, Q/K-weight write only, every other parameter's
write only, and the complete actual next state. Q/K means all 16 attention
Q and K weight tensors. The 68-tensor complement includes Q/K RMS gains,
other normalization gains, embeddings, head, V/O and MLP weights. Do not
silently put the gains into Q/K or hold them fixed at the full corner.
The full corner must exactly reconstruct all 84 saved next tensors.

Using actual saved writes avoids the M_s versus M_(s+1) ambiguity. Nothing
should recompute a gradient, moment, root or optimizer step. Own-state
comparisons remain co-adapted; they are not the same-state effect of changing
beta or LR.

## Do not add a gauge-standardization treatment

Keep the saved weights and their actual updates. Headwise normalization was
appropriate for describing one architectural scale symmetry, but separately
canonicalizing endpoints or rescaling update pieces would define a different
functional experiment. A Q/K-only finite corner also contains the joint Q/K
nonlinearity inside attention; this is part of its real response, not an
error that requires separating Q and K or individual heads.

The *output* readout should remove the harmless vocabulary-constant logit
shift. For delta_Q=z_Q−z_0 and delta_R=z_R−z_0, center each response under the
base distribution independently at every token, then retain

    V_Q = mean_tokens Var_(p0)(delta_Q),
    V_R = mean_tokens Var_(p0)(delta_R),
    C_QR = mean_tokens Cov_(p0)(delta_Q,delta_R).

These are finite-response moments. They are **not** infinitesimal GN
quadratics, even if their formulas resemble one. The relation
V_add=V_Q+V_R+2 C_QR is exact for additive finite logits. It does not identify
V_full unless the mixed model response is zero. Centered differences of
log-probabilities give the same moments as raw logits and can avoid storing
irrelevant offsets. A covariance cosine is descriptive, not a groupwise
causal attribution.

Forward KL(p0||p_corner) and finite CE changes retain the nonlinear size and
usefulness. Positive covariance or large isolated variance alone is not a
reason to suppress a group. The optional additive-logit/full split is useful
because it costs no extra forward, but it must stay descriptive and must not
reopen the failed "helpful mixed response" hypothesis without new evidence.

## One falsifiable local premise and a conservative material readout

My proposed premise is deliberately narrower than "Q/K explains the beta
training gain":

**The larger normalized Q/K turn under beta .8 produces a materially larger
Q/K-only predictive response that is also locally useful in the full update
context, at both existing learning rates.**

Use the following fixed two-part reading before outcomes are inspected:

1. The finite Q/K response-RMS ratio
   `sqrt(V_Q(beta .8)/V_Q(beta .9))` exceeds **1.10 in both banks at both LRs**.
   This tests translation of the 15–25% weight-chord difference, without
   assuming that the function-level multiplier equals the weight multiplier.
   If a denominator is zero or numerically unresolved, declare it unresolved;
   do not stabilize the conclusion with an outcome-dependent epsilon.
2. Retain both signed Q/K marginal effects,

       d_alone = L_Q − L_0,
       d_last  = L_full − L_R,

   and define the symmetric two-order loss credit

       S_Q = −(d_alone + d_last)/2.

   Require **S_Q(beta .8)>0 in each bank and pooled S_Q(beta .8)>=.001 NLL**
   at both LRs. The .001 cutoff is a prospective local materiality flag well
   above forward-rounding scale, not a confidence interval or a claim that
   .001 will accumulate over subsequent steps. Report S_Q for beta .9 and
   their differences as well; do not choose whichever attribution order
   makes beta .8 look better.

This symmetric credit is an explicit averaging convention for two possible
orders of applying the groups. It is not unique causal attribution. Its
advantage is that Q/K and complement credits sum to the actual finite loss
change while both conditional signs remain visible. In particular a group
can help alone and hurt last through ordinary predictive overlap. The
usefulness gate should therefore not be replaced by a favorable base-linear
logit pairing, nor should one order be silently discarded.

Do not divide S_Q or the variance by a near-zero whole-step loss or quadratic
and announce a percentage of the training gain. A small net update can hide
large cancelling or overlapping pieces. The finite four-corner and covariance
identities keep those possibilities explicit.

If the root adopts a different materiality number or readout from another
independent review, fix that choice and its reason before scoring. Do not
silently substitute it after outcomes for the concrete version given here.

## Outcomes that actually change the next decision

- **Both conditions hold:** the observed weight-space effect has a material,
  locally productive function-space counterpart at this selected point.
  Keep the Q/K/angular connection as a candidate worth a later causal design;
  do not claim it mediates the accumulated beta benefit, recommend a radius
  intervention, or expand to more states automatically.
- **Movement amplification fails:** the chord difference does not transfer
  in the anticipated way. Input anisotropy, RMS gains, attention invariances,
  or the local Jacobian can absorb it. Stop using the weight-space magnitude
  as functional evidence; do not rescue it with chosen heads or a new norm.
- **Movement grows but finite credit fails:** Q/K turns are a real functional
  difference but have no established local productive counterpart here.
  This directly weakens the tempting schedule explanation. It does not show
  Q/K is useless over training or justify removing its updates.
- **Banks disagree, effects are tiny, or qualifying identities fail:** record
  unresolved or numerical failure as appropriate, preserve all scores, and
  stop this fixed test. No extra bank, time point, rescaling, per-head split,
  subgroup search or finer calibration follows.

The complement may carry the main finite response, but a favorable complement
result does not authorize immediately decomposing its 68 tensors. The earlier
body/auxiliary work already demonstrates how easily overlap and finite-step
nonlinearity complicate that attribution.

## Qualification and bounded cost

Four states × sixteen sequences × four corners is 256 scientific forwards.
Use CPU with at most two numerical threads and the proposed 600-second cap;
forecast after a complete sequence including all four forwards and vocabulary
reductions. Two original banks are needed to expose sign instability; they
are not two training replications. Use available, previously unscored offsets
and include the one-token target overlap in disjointness checks.

Require strict base/full tensor reconstruction, restoration, matching inputs,
finite FP64 CE/KL/covariances, nonnegative variances within a fixed rounding
tolerance, Cauchy–Schwarz for the centered covariance, additive-variance and
finite CE split identities, and a repeated base output. Retain all labels,
per-sequence results, source/checkpoint/token provenance and failed attempts.
No singular-direction, matrix-power or derivative estimator is needed.

This is a bounded functional premise check on completed runs, not a new
scientific optimizer branch or permission to use training GPUs. Its value is
that a negative result changes the explanation before more effort is spent
on a weight-space effect whose predictive relevance was unmeasured.

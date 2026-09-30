# Independent review: TS output-factor repeatability premise

2026-09-28. This is a prospective methodological review, before seeing the
proposed measurement. Read-only source/metadata inspection and this note only;
no checkpoint load, model evaluation, GPU, job, or external state mutation.
The observer's `STATE.md` and `estimator_clocks/NOTE.md`, plus
`RESEARCH_GUIDE.md` and `RESEARCH_STATE.md`, supplied the current context.

## Verdict and exact claim

Proceed with a bounded CPU sensitivity measurement at the retained TS 16M
step46 state, conditional on its frozen stored momentum and one independent,
fixed input root. It directly addresses the missing premise: does changing
the small output-curvature sample change the resulting normalized TS body
direction appreciably? The TS state is preferable to a PD state for this
question because it avoids transplanting a direction into another optimizer's
co-adapted landscape. Neither choice can establish an optimization rate.

The stored M46 was formed before the weight state W46. It can be a common
fixed test vector but is not the newly computed step47 momentum. Roots are
also reconstructed, not the missing historical online R/B. State those facts
in the title/summary of results. Do not call the reconstructed direction the
actual next training update.

Single8 factors are not the online EMA. At decay .8, the stationary weight
ESS is9 observations, nominally72 sequences. At23 refreshes (through step45),
the first observation's coefficient is .8^22, about .00738, and the weight
ESS is approximately9. Thus even a pooled32 sample is smaller than the
stationary nominal EMA sample. Nonstationarity and sequence dependence can
reduce the online information content, but this experiment will not measure
that reduction or the historical lag. Large single8 variability would be
evidence of sensitivity to a fresh observation, not proof that the online
training estimator was unreliable. Small single8 variability would more
strongly weaken the proposed small-sample bottleneck at this frozen state.

## Minimal design improvements

1. Retain four disjoint8-sequence banks A/B/C/D, with positions fixed before
   results. Evaluate all48 body matrices. Report aggregate and per-kind/per-
   matrix distributions; do not qualify only the favorable or cheap layers.
   Bank comparisons have only four sampled groups, not thousands of
   independent tokens. Document contiguous-token/document dependence.
2. Repeat predictive-label sampling on A and ideally B as well. One repeated
   bank gives a conditional label-Monte-Carlo comparison, not a reliable
   decomposition of global sequence and label variances. With A-only repeats,
   describe that limitation and do not subtract a conditional variance from
   the across-bank variance to claim an identified sequence-noise fraction.
3. Compare D(AB) with D(CD): this is the available independent16-versus16
   stability comparison. Compare D(AB) and D(CD) with D(ABCD) as *nested*
   convergence checks. Shared data necessarily increases similarity, so
   16-versus32 agreement alone does not establish32-sample repeatability.
4. Add the algebra-only EMA-refresh sensitivity check. For each j, construct
   B_-j from the other three banks, then compare
   D(B_-j) with D(.8 B_-j + .2 B_j). This uses a24-sequence current-state
   anchor independent of the new8-sequence observation. It measures the
   nonlinear sensitivity to one correctly weighted refresh around a proxy
   anchor. It is not a reconstruction of a72-sequence historical EMA.
   One preregistered split suffices if all four cannot pass the cost gate.
5. Pool/mix **raw** B before its eigenvalue-mean normalization. Averaging
   individually normalized banks changes the trainer's estimator. Preserve
   each bank's trace and norm to expose this scale information.

No synthetic bootstrap can create independent72-sequence evidence out of
the original32 sequences. A weighted four-bank pseudo-EMA with the first
factor copied has only about2.91 observations of weight ESS, so it also
cannot be presented as the stationary online EMA.

## Correct map and numerics

The frozen `muon.py` uses:

```
L = (B / mean_eigenvalue(B) + .001 I)^(-.5)
Z = L @ M46 @ R
U = L @ NS_paper5(Z) @ R
D = U / ||U||_F * sqrt(min(out,in)) * sqrt(max(1,out/in))
```

Each matrix is normalized separately. A single global normalization would
change the direction of the concatenated body vector. Use the frozen
five-polynomial map rather than substituting exact SVD polar: the experiment
concerns the implemented nonlinear map, and approximation sensitivity may
be part of that implementation. CPU arithmetic uses FP32 products while
the original CUDA map uses BF16 products; record this as a controlled
numerical approximation, not a bitwise training replay. Input R estimation
must be shared across B comparisons and sampled independently of A/B/C/D.

The PD control is the same construction with L=I, identical M/R and
per-matrix norm. Retain trace-normalized B distances, L distances and final
D distances. If cheap, retain the fraction of B eigenvalues below the
relative damping level .001; root amplification can make a small B change
matter, while a large raw-B change can disappear in normalized D.

Frozen source anchors:
`logs/muon_spectra/soaudit_batch16m_20260927/frozen/adamw_spectra/muon.py`
lines36–65 (NS precision),472–498 (raw EMA then normalized root),
690–739 (two-sided map and per-matrix normalization),832–868 (sampled labels).
The chosen run's `scientific/metadata.json` confirms nesterov=false,
alpha=.5, output beta=.5, damping=.001, output refresh2, samples8, EMA=.8.

## Preregistered interpretation

Use continuous measurements first. At least record all six single8 pairwise
cosines, the independent16 pair, nested16-to32 distances, PD distance, and
the EMA-refresh sensitivity. A whole-body cosine can conceal concentrated
instability in one projection family, so the per-kind results remain part
of the decision.

Suggested diagnostic flags, to be adopted or revised **before** data:

- Stable at this diagnostic scale: all six single8 aggregate cosines >=.99
  and independent16 cosine >=.995, with no projection-family pair <.95.
- Material sample sensitivity: at least one single8 aggregate cosine <=.90
  and the independent16 comparison is more stable than the mean single8
  comparison. A lower cosine without improvement under pooling is unstable
  but leaves the sample-size remedy unestablished.
- Intermediate/mixed findings remain intermediate/mixed. These are
  descriptive thresholds, not universal optimization boundaries or
  statistically calibrated hypothesis tests. A few groups cannot provide
  a high-confidence population stability claim.

Also compare estimation variation with the size of the geometry being
added. For two norm-matched draws, report

```
||D_i-D_j||^2 / (||D_i-D_PD||^2 + ||D_j-D_PD||^2).
```

This prevents a cosine such as .99 being declared harmless when TS itself
only differs slightly from PD. It is a descriptive ratio with finite-sample
noise and possible tiny denominators, not an unbiased signal/noise estimate.
The mean/cross-products relative to PD can also be retained if inexpensive.

## Independent score and next decision

Independent loss or GN scoring is unnecessary to answer the restricted
repeatability question. Do not expand the first probe merely to get an
optimization story. If it finds material instability and a subsequent claim
is that pooling improves the direction, that claim needs an independent
held-out bank and either a matched-norm true-loss profile or validated
slope-and-curvature measurements. A held-out gradient dot product alone
does not establish loss improvement, especially at an oscillatory own-state
checkpoint with a lagged fixed M. A parameter-space angle does not by
itself establish a function-space impact.

If the initial factors/directions are already stable, close this particular
sample-instability premise at this state without spending the remaining
budget on favorable scoring or more checkpoints. If single8 is unstable
but the .2-refresh perturbation is small, qualify the noise explanation
and prioritize actual historical-state retention before training changes.
If both are material and pooling improves repeatability, an independently
scored follow-up becomes justified, still without a rate or mechanism claim.

The proposed two-thread CPU limit and15-minute cost gate are appropriate.
Qualification should establish the complete48-matrix workload forecast;
an over-budget outcome is a useful recorded limit, not permission to switch
to a selected layer set or compete with the training jobs for a GPU.

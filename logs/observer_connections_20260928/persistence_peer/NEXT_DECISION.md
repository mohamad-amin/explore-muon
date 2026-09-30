# Next bounded diagnostic: resolve body–auxiliary attribution first

2026-09-28. High-level peer discussion only; no computation or new experiment.

**Priority: A, the four-corner body/auxiliary check, initially at the original
4M Muon step 183.** That checkpoint already has a specific disagreement:
body-only next-step probing reports gradient cosine +0.095, while loading the
actual full next state reports −0.526. Data amount, data identity, precision,
and parameter scope all differ. A matched check can distinguish those
explanations without adding a proposed optimizer or a new functional split.

The new dense-lag result makes this question more relevant. The PD input tail
is positive at lag one but reverses at later lags, including a broadly shared
negative lag-six product. Those are body-gradient observations at states where
**all** parameters have moved. Embeddings, normalization gains, and the head
can rotate a body gradient or its activation distribution. The body-only
spectral interpretation has not established how much that channel contributes.

## Minimum diagnostic and premise checks

Use one saved model pair, with four corners: base, body advanced, all auxiliary
parameters advanced, and both advanced. Include every nonbody learned parameter
in the auxiliary group; inspect the actual parameter partition and checkpoint
step/config/token metadata. The combined corner must reconstruct the entire
saved next-state parameter dictionary exactly. Use one numerical model path,
the same loss reduction, and the same precision for all corners. Verify the
base and combined-corner outputs agree with directly loaded models.

The first comparison needs losses and body gradients, not a GN probe. On the
same held-out bank, retain each corner's loss, body-gradient norm, and gradient
vector differences. The exact finite-difference decomposition is

    g11 − g00 = (g10 − g00) + (g01 − g00)
                + (g11 − g10 − g01 + g00).

The last term is the finite-step interaction; do not drop it. Signed inner
products between these components distinguish additive effects from
cancellation. Report absolute quantities before normalized attribution ratios.

Common data across corners intentionally reduce variance in their difference,
but same-data cosines are empirical-bank measurements, not unbiased population
correlations. Repeat the four corners on a second disjoint bank. Cross-bank
products are preferable for population-gradient alignment, while within-bank
differences show the counterfactual sensitivity. Choose the bank sizes and CPU
wall-time limit before examining results; begin with a small implementation
qualification, then a fixed bounded measurement. If finite-sample uncertainty
dominates under that cap, retain an unresolved result rather than launching an
adaptive sequence of larger probes. Two CPU threads, sequential model calls,
no training, and no GPU allocation suffice for the initial instrument check.

PD 501→502 is a good next corroborating state if the first check establishes a
meaningful auxiliary contribution. It is a weaker primary target: no archived
body-only contrast is available there, and its positive lag-one tail does not
itself test the later reversal. Do not silently change the initial target to
whichever state yields the clearest effect.

## Why B remains valuable, but second

The V-map observation is coherent and large: learned mean-versus-centered gain
differs across all layers, with the measured post-attention mean kept distinct.
Its proposed functional displacement split asks a useful new question: is the
constant-value route productive under the actual update? A successful B could
justify architectural thinking even if A is unremarkable.

However, B requires additional intervention/JVP qualification, independent
definition and scoring of μ, and a full 2×2 GN quadratic with its cross term.
It does not explain the whole-model gradient rotation that produced the dense
correlation series. Its current evidence also admits a simple co-adaptation
account and a changing mean caused upstream of V. Resolving A first tests a
missing premise common to both interpretations at lower instrumentation cost.

Neither outcome authorizes a training branch or establishes a useful new
method. A large auxiliary effect would motivate investigating which auxiliary
components supply the moving geometry. A negligible effect at adequate
resolution would make the within-body V/attention diagnostic more compelling.

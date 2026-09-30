# Re-read the saved function-space coupling matrices

2026-09-28. Bounded CPU-only artifact analysis; no model, new gradients,
scheduler allocation, GPU, or training. Live queue was checked before starting:
main allocations 2567578 and 2618555 are live and untouched. All outputs stay
in this directory. Numerical threads are capped at two.

The observer's body/auxiliary counterfactual and value-route counterfactual
show positive finite loss interactions. Is the older matrix-level GN coherence
a broad positive overlap, or mostly a narrow identifiable channel? The saved
48-by-48 exact-GN Grams can distinguish broad versus concentrated coefficient
structure, but cannot identify a logit-space channel after logits were discarded.

Analyze all eight fresh-gradient Grams at the two original 1M states and all
eight paired momentum Grams at the four 1M/4M states. Retain slopes, raw GN
diagonals, signs, and correlations after diagonal normalization. Examine leading
eigenvalues and loadings, off-diagonal residuals, and repeatability across the
two existing score halves where available. Do not treat eigenvalues of a Gram
over 48 chosen directions as eigenvalues/rank of the full GN operator. Fit and
score on the same archive is descriptive, not a cross-validated intervention.

Competing explanations are broad positive overlap among descent directions,
raw-magnitude dominance by a few matrices, and a stable low-dimensional shared
response. A concentrated direction-level component would justify identifying
its token/logit responses, not immediately deleting a global GN eigenvector.
Inspect the existing global-deflation, block-GN, staging, and per-matrix scaling
failures before suggesting any follow-up. The smallest useful discriminator
should distinguish a named channel from generic shared descent, and preserve
the actual update scale and all cross terms.

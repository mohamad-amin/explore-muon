# Saved attention geometry audit

2026-09-28. CPU-only retrospective analysis, capped at two numerical threads.
No model calls, training, scheduler requests, or modifications outside this folder.

Missing premise: previous within/between fits identify the sequence-mean second
moment with one coherent component. That combines the global input mean with
between-sequence variation, although the two have different interpretations:
shared bias-like channels versus document-specific information. Also, matching
diagonal quadratic forms does not establish matching full matrix orientation.

Read all 32 existing marginal archives (four methods, eight checkpoints), retain
all eight layers and all kinds in descriptive summaries. Decompose attention's
C into within-sequence covariance, covariance of sequence means, and the global
mean outer product. Compare original two-component Frobenius fit with the
three-component fit and a C + rank-one-mean fit, retaining negative coefficients,
residuals, and conditioning. Report the difference between actual mean energy
||E[x]||²/tr(C) and the existing Rayleigh share uᵀCu/tr(C).

Competing explanation: attention aggregation can alter sequence coherence while
leaving global means uninformative; improved fits may simply overfit highly
collinear predictors. Curvature changes can also be trajectory consequences,
not causes. No training gain follows from fitting a marginal.

Decision: only propose a new discriminator if a reproducible cross-time and
cross-method separation survives these confounds. Do not revive per-kind alpha
scans or indiscriminate centering, which have prior negative evidence. Saved
sampled-label marginals lack split-sample uncertainty; retain this limitation.

Resource bound: 32 files ~1.3GB sequential reads; small matrix contractions,
<=2 CPU threads, expected under five minutes. Inputs hashed before and after.

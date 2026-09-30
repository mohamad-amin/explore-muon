# Two-sided data norm: controls (second-order audit, 2026-09-27)

These controls follow the independent review (`research/adamw_spectra/MUON_CASE.md`, 05:41 CDT). TS (α ¼, β ¼ with GN labels) beats PD α ¼ by 0.005 at 1M. The controls test whether that gain comes from the GN output eigenbasis, and whether the arms' different clipping rates (TS 58% of steps, PD 12%) matter.

- **Arms.** Seed 260925, batch 1M, LR 0.01, 4× RTX A4000:
  - `TSplacebo`: B's eigenvalues in a fixed random orthogonal basis per matrix (`data_norm_out_placebo`). If the gain is about the GN output structure, the placebo should lose it.
  - `TS ... ef`: data labels (empirical Fisher) instead of model-sampled labels.
  - TS and PD with clipping disabled (`grad_clip` 1e6): a clipping-matched pair.
- **Code.** Frozen from the live source, which adds `data_norm_out_placebo` (with a unit test; 83 tests pass) and allows `muon_momentum` and `grad_clip` as treatments.

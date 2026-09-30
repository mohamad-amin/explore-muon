# Two-sided data norm, follow-up (second-order audit, 2026-09-27)

The first paired test (`../soaudit_twosided_20260927`) gave TS − PD = −0.0052 at LR 0.01, seed 260925, on A4000.
Protocol entry: `research/adamw_spectra/MUON_CASE.md`, "Two-sided data norm: first training result and follow-up" (04:23 CDT).

- **Arms** (1M batch, 1.54B tokens, 4× RTX A4000 each, kept checkpoint 500). TS is α ¼ β ¼ with GN labels:
  - TS at LR 0.007 and 0.014, seed 260925: the LR bracket around 0.01;
  - TS β ½ at LR 0.01, seed 260925;
  - PD α ¼ and TS at LR 0.01, seed 260926: a second paired seed.
- **Code.** Frozen copy of the live source, identical to the first two-sided cohort's code (only MUON_CASE.md differs).

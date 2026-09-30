# Preconditioning strength with fresher averaging at 4M (second-order audit, 2026-09-27)

PD α ½ beat α ¼ at 4M once momentum was 0.9 (−0.004), after losing at 0.95 (+0.007). This cohort asks whether the same
holds for S∘PD α ½ and for both sides at ½ (TS α ½ β_out ½). Decision note and predictions:
`research/adamw_spectra/MUON_CASE.md`, 10:59 CDT.

- **Arms.** 4M-token batch, seed 260925, momentum 0.9, kept checkpoint 183.
  - priv-g14 (L40S): S∘PD α ½ at LR 0.02 and 0.01.
  - g20 (RTX 6000 Ada, after the TS β_out ½ arm of `soaudit_mom4m4_20260927`): TS α ½ β_out ½ (GN labels) at LR 0.01 and 0.02.
- **References** (same hardware):
  - S∘PD α ¼ @0.01: 3.8314 (L40S).
  - TS α ¼ β_out ¼ @0.01: 3.8500 (Ada).
  - PD α ½ @0.02: 3.8589 (Ada).
- **Code.** Frozen from the live source (84 unit tests pass).

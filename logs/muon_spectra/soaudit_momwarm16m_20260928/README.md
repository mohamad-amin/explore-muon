# Momentum warmup for S∘PD at 16M (second-order audit, 2026-09-28)

At 2× horizon, β 0.8 leads β 0.9 by ~0.09 mid-run and then loses most of it (final −0.010). At 1× the run ends near the
peak (−0.10). A window that grows with training should keep the early speed and the late filtering. Decision argument:
`research/adamw_spectra/MUON_CASE.md`, "Decision argument: a momentum warmup" (15:05 CDT).

- **Arms.** S∘PD α ½ @0.028, seed 260925, Ada, β linear in tokens from 0.8 to 0.9 over the first half of the budget:
  - 2× horizon (184 steps): reference β 0.9 3.9840, β 0.8 3.9740;
  - 1× (92 steps): reference β 0.8 4.4724, β 0.9 4.5742.
- **Code.** Frozen after adding `muon_momentum_start` and `muon_momentum_warmup` (87 tests pass). They run on g20 after
  the 4M PD β 0.8 arm.

## Result, 2× horizon (17:05 CDT)
β 0.8 → 0.9 warmup: **3.9628**, against constant β 0.9 3.9840 and β 0.8 3.9740 (and β 0.9 @0.02 3.9790). It beats both
constants. The lead over β 0.9 is −0.11 at step 46 and −0.021 at the end.

## Result, 1× (17:37 CDT)
β 0.8 → 0.9 over the first half: 4.5152, between constant β 0.8 (4.4724) and β 0.9 (4.5742). At 92 steps the whole run is
early phase, and a half-budget warmup rises too soon: define the warmup in steps.

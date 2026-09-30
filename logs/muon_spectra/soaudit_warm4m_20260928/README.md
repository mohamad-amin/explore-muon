# Momentum warmup at 4M, defined in steps (second-order audit, 2026-09-28)

Does "start the momentum short (β 0.8) and lengthen it to 0.9 over the first 120 steps" keep a head start at 4M, as it
did at 16M (2× horizon: −0.021, −0.024, −0.027 in three pairs)? Decision argument: `research/adamw_spectra/MUON_CASE.md`,
"Decision argument: the momentum warmup at 4M, defined in steps" (20:12 CDT).

- **Arms.** 4M, 368 steps, seed 260925, α ½ @0.02:
  - PD on g20 (Ada), reference β 0.9 3.8589;
  - S∘PD on priv-g14 (L40S), reference β 0.9 3.8260.
  Each runs after its node's PD-top arm(s).

## Results

- **PD (Ada), complete (2026-09-28 21:04 CDT).** Final 3.8462, vs β 0.9 3.8589 (−0.0127) and β 0.8 3.8629 (−0.0167). Ahead
  at every validation step: −0.055 / −0.081 / −0.036 / −0.028 / −0.015 / −0.015 / −0.011 / −0.013 at 50 / 100 / 150 /
  200 / 250 / 300 / 350 / 368 vs β 0.9.
- **S∘PD (L40S).** Running.
- **S∘PD (L40S), complete (2026-09-28 21:19 CDT).** Final 3.8166, vs β 0.9 3.8260 (−0.0094), β 0.8 3.8280 (−0.0114) and
  β 0.95 3.8608 (−0.044). Ahead at every validation step after 50.

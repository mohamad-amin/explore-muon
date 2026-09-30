# Two-tap momentum pre-filter at 16M (second-order audit, 2026-09-28)

At large batch the gradient is mostly a period-2 oscillation of the stiff directions (step profiles, 10:38 CDT). The
momentum's EMA filters it at the cost of ~β/(1−β) steps of lag. `muon_prefilter: two_tap` averages consecutive
gradients before the EMA, M ← βM + (g_t + g_{t−1})/2, which removes a period-2 component exactly with half a step of
lag. Decision argument and predictions: `research/adamw_spectra/MUON_CASE.md`, "Decision argument: a two-tap pre-filter
for the momentum at large batch" (10:38 CDT).

- **Arms** (seed 260925, 92 steps, hardware matched to the references in `../soaudit_batch16m_20260927`):
  - g20 (RTX 6000 Ada), S∘PD α ½ @0.028 (reference β 0.9: 4.5742): two-tap β 0.8, two-tap β 0.9, plain β 0.8.
  - priv-g14 (L40S): Muon @0.02 two-tap β 0.9 (reference 4.9104), then PD α ½ @0.028 two-tap β 0.9 (reference 4.6711).
- **Code.** Frozen from the live source after adding `muon_prefilter` (train.py validation, muon.py momentum update,
  run_arm.py treatments, a unit test; 85 tests pass). The option is off by default, so earlier configs are unchanged.

## Results (12:18 CDT)
| Arm | Final | Reference | Change |
|---|---|---|---|
| Muon two-tap β 0.9 | 5.5101 | 4.9104 | +0.600 |
| PD two-tap β 0.9 | 4.8657 | 4.6711 | +0.195 |
| S∘PD two-tap β 0.9 | 4.6675 | 4.5742 | +0.093 |
| S∘PD two-tap β 0.8 | 4.5655 | 4.5742 | −0.009 |
| S∘PD plain β 0.8 | **4.4724** | 4.5742 | **−0.102** |

- The filter removes the period-2 feedback that stabilizes the directions an optimizer steps in. It hurts in proportion to stiff stepping (Muon > PD > S∘PD).
- Plain β 0.8 is the best 16M result so far. The β sweep follows in `../soaudit_mom16m_20260928`.

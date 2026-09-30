# Momentum warmup replication at 2× horizon, 16M (second-order audit, 2026-09-28)

S∘PD α ½ with β 0.8 → 0.9 over the first half of the budget reached 3.9628 at 2× horizon on seed 260925. That beats
constant β 0.9 (3.9840) and β 0.8 (3.9740), but the margin (−0.016 to −0.021) is one seed and one method. Findings:
`research/adamw_spectra/MUON_CASE.md`, 17:05 CDT.

- **Arms.** 16M, 2× horizon (184 steps, 3.08B tokens):
  - g20 (Ada), S∘PD α ½ @0.028 on seed 260926: warmup, then its own β 0.9 reference;
  - priv-g14 (L40S), PD α ½ @0.028 on seed 260925: warmup, then its own β 0.9 reference.
- **Prediction.** The warmup beats β 0.9 by ≥ 0.01 for both.

## Results (19:42 CDT)
| Pair | β 0.9 | Warmup | Gain |
|---|---|---|---|
| S∘PD seed 260926 (Ada) | 3.9885 | 3.9646 | −0.024 |
| PD seed 260925 (L40S) | 4.0446 | 4.0179 | −0.027 |

With the seed-260925 S∘PD pair (−0.021), the warmup gain replicates across two methods and two seeds.

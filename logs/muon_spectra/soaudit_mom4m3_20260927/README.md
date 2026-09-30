# Momentum 0.9 at batch 4M for every optimizer (second-order audit, 2026-09-27)

At 4M, Muon @0.014 with β 0.9 gave 3.9218, against 3.9524 with β 0.95 (−0.031) and 3.9437 with β 0.81
(`../soaudit_mom4m_20260927`). The 4M preconditioner gains were all measured at β 0.95, so they must be re-measured
with momentum retuned for every optimizer.

- **priv-g14 (L40S)**, after the momentum sweep: S∘PD @0.01 with β 0.9, and Muon @0.02 with β 0.9 (Muon's LR bracket at β 0.9).
- **g20 (Ada)**, after PD with β 0.81: PD α ¼ @0.02 with β 0.9, and TS @0.01 with β 0.9. The normalized-gradient (clip 0.1) queue (`../soaudit_clip4m_20260927`) now runs after these.

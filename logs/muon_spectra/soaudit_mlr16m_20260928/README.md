# Muon's LR at β 0.8, 16M: a fairness check for the momentum result (second-order audit, 2026-09-28)

Muon gained −0.020 from β 0.8 at LR 0.02, against −0.10 to −0.12 for the whitening methods. Muon's LR curve at β 0.9 is
sharp (0.014 +0.016, 0.028 +0.053), and its β arm was not LR-re-tuned (review, MUON_CASE 13:50). If Muon at β 0.8
prefers another LR, part of the difference between methods is Muon's untuned LR.

- **Arms.** Muon at β 0.8, 16M, seed 260925, L40S, LR 0.014 and 0.028. Reference: β 0.8 @0.02, 4.8901. It runs on
  priv-g14 after the clip-0.1 arms of `../soaudit_clip16m_20260928`.

## Results (15:27 CDT)
Muon at β 0.8: LR 0.014 4.9042, 0.02 **4.8901** (reference), 0.028 4.9285. The LR optimum stays at 0.02.

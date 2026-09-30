# Gradients normalized before momentum (clip 0.1) at batch 4M (second-order audit, 2026-09-27)

At 4M, S∘PD clips on 94–96% of steps (norm 1.0). Without clipping it loses 0.020 (3.8657 vs 3.8454, L40S). So clipping,
which for polar updates only reweights the momentum by making it an average of unit-norm gradients, is part of S∘PD's
4M advantage. This cohort asks whether normalizing every gradient before momentum also helps the optimizers that
rarely clip at 1.0: Muon (7% of steps) and PD α ¼ (8%).

- **Arms** (g20, Ada; seed 260925; batch 4M; 368 steps):
  - PD α ¼ @0.02 with grad_clip 0.1; baseline 3.8847 on Ada.
  - Muon @0.014 with grad_clip 0.1; baseline 3.9524 on L40S, so this comparison is cross-hardware (~±0.002).
- **Launch.** `start_g20.sh`, after `../soaudit_mom4m2_20260927`'s g20 queue.

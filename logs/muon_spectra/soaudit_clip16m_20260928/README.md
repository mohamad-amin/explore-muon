# Muon with clipping 0.1 at 16M: the clipping confound of the momentum result (second-order audit, 2026-09-28)

PD and S∘PD clip on 87–90 of 92 steps at 16M (pre-clip norm 1.7–4.8 mid-run), while Muon clips only in its first 18–25
steps. The whitening methods' momentum therefore averages normalized gradients and Muon's does not. At 4M clipping acted
as a first-order lever that substitutes for β. The review (MUON_CASE, 13:5x) named this the main confound of the claim
that whitening methods gain ~0.1 from β 0.8 while Muon gains 0.02.

- **Arms.** Muon @0.02, 16M, seed 260925, L40S, grad_clip 0.1 at β 0.9 and at β 0.8. References: clip 1.0 at β 0.9,
  4.9104, and at β 0.8, 4.8901.
- **Decisive both ways.**
  - Muon with clip 0.1 gains ~0.1 from β 0.8: the difference between methods is the clipping, not stiff stepping.
  - It still gains ~0.02: the difference survives its main confound.
  - Clip 0.1 itself improving Muon would mean the 16M Muon baseline was under-tuned. The speedups would then need
    recomputing.

## Results (14:38 CDT)
| Muon @0.02 | β 0.9 | β 0.8 |
|---|---|---|
| clip 1.0 | 4.9104 | 4.8901 |
| clip 0.1 | 4.8882 | 4.8916 |

With every gradient normalized, Muon gains nothing from β 0.8 (+0.003). The whitening methods' ~0.1 gain from β 0.8 is
not a clipping effect.

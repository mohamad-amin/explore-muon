# Does the off-diagonal part of C matter? Structural check (2026-09-26)

`offdiag_probe.py` → `offdiag.json`. Read-only, on tuned Muon's final weights (LR 0.007, seed 260925, A6000, width 512). C is the uncentered input second moment from 128 validation sequences, scaled to unit mean eigenvalue as in PD. R is PD's root at α = ¼ and damping 1e-3. Means by layer kind (q, k and v share one input):

| input to | dim | off-diagonal share of ‖C‖_F² | ‖R_full − R_diag‖ / ‖R_full‖ | mean-input spread (coords) | top-eigenvector spread |
|---|---|---|---|---|---|
| q/k/v | 512 | 0.95 | 0.34 | 164 | 161 |
| o | 512 | 0.92 | 0.35 | 154 | 135 |
| up | 512 | 0.96 | 0.34 | 154 | 152 |
| down | 2048 | 0.89 | 0.26 | 357 | 336 |

**Reading.**
- About 90–96% of C's Frobenius mass is off the diagonal.
- A diagonal-only root differs from PD's full root by about a third.
- The dominant structure (the shared mean input and C's top eigenvector) is spread over roughly 150 of 512 coordinates (about 350 of 2048 for down). A per-coordinate scaling cannot represent it.
- This is structural evidence only. The training ablation (PD with diag(C)) has not been run.

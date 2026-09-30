# GN-PD as a training rate at 16M (second-order audit, 2026-09-28)

Is the exact Gauss-Newton geometry a per-step rate at large batch, or a one-time harvest of a co-adapted state?
Decision argument and predictions: `research/adamw_spectra/MUON_CASE.md`, 2026-09-28 09:40 CDT.

- **Start.** Every arm starts from the PD α ½ @0.028 16M state at step 9 (`soaudit_batch16m_20260927`, seed 260925).
  It trains the remaining 83 steps with `../newton_train.py` on 4 GPUs, with the run's own data, schedule and LR 0.028.
  Clipping is 1.0 as in the trainer. Body steps use per-matrix norms √min·√max(1, m/n) × LR, with the trainer's
  weight decay and AdamW for the aux parameters.
- **Maps.**
  - `kron`: L polar(L M R) R with L, R the −½ roots of the step's B (sampled labels) and C (positions ≥ 1), from 32
    curvature sequences per rank. It is TS ½/½ with instantaneous statistics.
  - `gnpd`: −(G + μ)^-½ polar((G + μ)^-½ M) per matrix, with G the exact GN of all 48 body matrices on the same
    sequences (Lanczos 64, μ = 1e-3 ρ).
- **Inputs.**
  - β 0.9: the momentum EMA, initialized from the checkpoint's buffer.
  - β 0: the fresh batch gradient.
  - transport: the EMA carried to the current weights, A + (1 − β) G D.
- **Logged per step.** The top GN eigenvalue (Lanczos 20 from a fixed random start) and the cosine of each matrix's
  step with the Kronecker map (`--log-kron`). Validation every 5 steps.
- **Smoke tests.** In `../gnrate_smoke/`, 2 steps each. The harness reproduces the real trainer's step-10 loss
  (7.17841) and gradient norm.
- **Analysis.** `../gnrate_compare.py` gives smoothed loss, step-equivalent speedups at matched loss (against Muon
  @0.02 and against the control), sharpness and cosines.

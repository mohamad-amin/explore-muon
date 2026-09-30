# Momentum transport: runs that record the displacement EMA (second-order audit, 2026-09-27)

Curvature corrections (GN inverse, sequential Gauss-Seidel coupling) help on fresh gradients and hurt on the momentum.
In the GN model the momentum's staleness is explicit: M_t ≈ g(W_t)/(1 − β) − G Q_t, with Q_t = Σ_k β^k (W_t − W_{t−k}).
So M_t + G Q_t is the averaged gradient moved to the current weights. These runs record Q with the measurement-only
option `muon_track_displacement` (Q ← β Q + β/(1−β) ΔW; the trajectory is unchanged, see the unit test). The step-500
checkpoint then allows a one-step test of transported momentum: plain, with GN scaling, and staged.

- **Arms.** Muon @0.007 and PD α ¼ @0.01, 1M batch, seed 260925, 4× A4000, kept checkpoint 500 plus next-step weights. The runs go to completion so the finals double as extra hardware replicates.
- **Code.** Frozen live source: adds `muon_track_displacement` (MEASUREMENT in `run_arm.py`; 84 unit tests pass).

# Dense-lag trajectories for gradient persistence (second-order audit)

Fresh reruns of the trajectory cohort's Muon@0.007 and PD α ¼@0.01 arms (same seed, data and frozen code:
`../soaudit_traj_20260926/frozen`). Each stops at step 532 and keeps checkpoints at steps 501, 503, 507, 515 and 531,
plus next-step weights, giving lags 1–4, 7, 8, 15, 16, 31 and 32 from step 500.
Measurement only: `measure_persistence.py` reads these to find how the expected gradient decorrelates between
1 and 32 steps, per curvature class. The runs are on 4× RTX A4000 (cluster), so they are not bitwise copies of
the L40S and RTX 6000 Ada trajectories.

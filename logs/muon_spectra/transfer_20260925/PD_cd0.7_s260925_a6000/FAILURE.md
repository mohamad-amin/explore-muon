# Failed in the tiny CUDA stage (2026-09-25 ~19:02 CDT); no training

`ValueError: Warmup overlaps cooldown`: the controller's 3-step tiny smoke run (1 warmup step of 128
tokens) cannot hold a 70% cooldown. The full run (warmup 50 of 1469 steps) is unaffected. The controller
now overrides the tiny stage's cooldown to 0.1 when the arm's is above 0.5. Rerun unchanged in
`../../transfer_cd_20260925/`.

# Candidate D: numerical pass, resource gate failure

All five 21-update methods completed on one 16 GB A4000 (Slurm 2627171_0, exit 0, 7m37s). Numerical qualification passed: finite trajectories and states; paired initialization, data and hardware; all 1,168,147 development targets; saved-loss reconstruction; 48 covariance modules where applicable; 2,688 collected training forwards; and the third input-root refresh at step 21. No optimizer ranking is claimed from this smoke.

The predeclared eight-GPU-hour screen cap fails. The 30 initial arms are forecast at 12.41 GPU-hours; the maximum 40 arms at 16.55. Training alone projects 14.47 GPU-hours for 40 arms, so this failure is not just conservative evaluation-overhead accounting. No scientific screen is launched. The small-surrogate goal remains unfulfilled.

| Method | Smoke training seconds | Full-run forecast minutes |
|---|---:|---:|
| adamw | 47.37 | 17.91 |
| muon | 47.50 | 16.77 |
| pd | 93.40 | 30.17 |
| spd | 93.52 | 30.15 |
| ts | 89.92 | 29.12 |

The forecast assumes 368 updates and full development evaluation every eight updates, plus steps 0/1/final and final reconstruction. It estimates runtime; no full run was performed. Frozen scientific criteria and all previous failed comparisons remain unchanged. No sealed test panel was model-scored, and all panels remain unbound.

Evidence: `results.json`, `RESOURCE_GATE_VERDICT.json`, `READOUT_SOURCE_BINDING.json`, and the cohort-level `SLURM_COMPLETION.json`. The original verdict JSON is retained because its two plan-hash labels needed correction; the result and decision were unchanged.

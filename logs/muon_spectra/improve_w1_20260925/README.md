# Muon improvement search, wave 1 (2026-09-25)

Protocol, decision argument and rules: `research/adamw_spectra/MUON_CASE.md`, section "Muon improvement search, wave 1". Frozen source: `frozen/`, with sha256 values in `frozen_manifest.json`. Each arm runs `frozen/adamw_spectra/run_arm.py`.

| arm | hardware | status | final val NLL |
|---|---|---|---|
| S2_baseline_s260925 | L40S (priv-g14) | complete | 3.71168 (seed noise vs 3.71310 at seed 260924) |
| C_track1_drop | RTX 6000 Ada (g20) | complete | 3.72302; paired Δ vs M_lr0.01_g20 = **+0.0094** (harmful) |
| A_paper_deflation | – | cancelled before start (plan revised after review) | – |
| B_track1_restore | – | cancelled before start; re-run in wave 2 as B_track1_restore_g20 | – |

C's time course:
- −0.07 at steps 100–200;
- crosses near step 450;
- +0.013 in the stable phase;
- +0.009 at the end.

The momentum's top-pair energy stays near 0.75.

Post-hoc probe: C's deficit is not a refittable implicit-bias offset (`../bias_lag_probe_20260925`).

Cross-wave summary: `../improve_w4_20260925/compare_all.py` → `summary_all.md`.

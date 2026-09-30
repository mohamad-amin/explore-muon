# Muon improvement search, wave 2: mean-whitened Muon and controls (2026-09-25)

Protocol: `research/adamw_spectra/MUON_CASE.md`, "Wave-2 amendment" sections. Amendments made during the wave:
- `AMENDMENT_tiny_stage.md`: the tiny stage kept the arm's architecture flags, and the failed E_auto_lr0.01 was re-run as `_r2`.
- `AMENDMENT_measurement_device.md`: the A6000 arms compute spectra on the GPU (`*_gpusvd`); the superseded arms are marked.

Queues:
- `node_queue.py` runs arms in sequence on an allocation;
- records are in `queue_*.json`, `queue_*.log` and `queue_launch.json`.

Results (final val NLL at step 1469; Δ paired by seed and hardware):

| arm | Δ | notes |
|---|---|---|
| M_lr0.01_g20 | – | 3.71366 (hardware twin of the L40S reference, 3.71310) |
| M_lr0.014, M_lr0.02 | +0.0059, +0.0070 vs M@0.01 | LR 0.01 is the best of the bracket |
| E_auto_lr0.01_g20 | −0.0042 | inconclusive |
| E_auto_lr0.01_s260925_a6000_gpusvd | −0.0022 | inconclusive |
| E_b0.25_lr0.01_s260925_a6000_gpusvd | −0.0061 | promising (single seed) |
| E_auto_lr0.014 | −0.0019 vs M@0.014; +0.0040 vs M@0.01 | |
| E_auto_lr0.02 | +0.0066 vs M@0.02 | harmful |
| E_zero_lr0.01_g20 | +0.0056 | harmful (like C) |
| B_track1_restore_g20, N_nesterov_lr0.01_g20, E_auto_lr0.01_r2 | see `compare_all.py` | |

The early gains (−0.04 to −0.07 at steps 100–200) mostly wash out. Their shape matches raising Muon's LR. The wave-4 exact-polar control tests the NS-conditioning explanation.

`compare_w2.py` is the wave-2 summary. The cross-wave summary is `../improve_w4_20260925/compare_all.py`.

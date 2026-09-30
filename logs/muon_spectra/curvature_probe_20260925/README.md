# Curvature along the mean-input direction (2026-09-25)

`curvature.py` → `curvature.json`. Read-only, on the frontier-norm Muon run's final weights (val NLL 3.71310). It uses 64 fresh sequences (offset 2.2e9), an FP32 forward, and central finite differences with step 0.02‖W‖_F/√d_in along unit rank-1 directions u zᵀ, with a fixed random output direction u. It ran on the dev-gpu partition (1× A6000).

| matrix | mean share of E‖x‖² | 2nd-moment ratio x̂ vs random z | loss-curvature ratio x̂ vs random z | momentum pair curvature: top / rank-8 / median | cos(v1, x̂) |
|---|---|---|---|---|---|
| block04.q | 0.36 | 281 | 25 | 0.029 / 0.001 / 0.000 | 0.99 |
| block04.v | 0.36 | 325 | 276 | 0.452 / 0.003 / 0.000 | 0.88 |
| block04.up | 0.35 | 300 | 98 | 0.023 / 0.001 / 0.000 | 1.00 |
| block04.o | 0.06 | 37 | 11 | 0.007 / 0.002 / 0.000 | 0.91 |
| block04.down | 0.05 | 113 | 18 | 0.019 / 0.002 / 0.001 | 0.95 |
| block08.q | 0.21 | 148 | 7 | 0.018 / 0.001 / 0.000 | 0.93 |
| block08.o | 0.44 | 427 | 57 | 0.050 / 0.001 / 0.000 | 0.99 |
| block08.down | 0.25 | 680 | 73 | 1.284 / 0.007 / 0.001 | 0.93 |

The mean-input column is 7–276× stiffer than a typical input column. The loss ratio is below the pure input-second-moment ratio because output-side sensitivity differs (smallest for Q).

The momentum's top pair (v1 ≈ x̂) is 20–180× stiffer than its rank-8 pair. Muon steps along it at unit size, so it sets the curvature-weighted step. This supports the premise of mean-whitened Muon: MUON_CASE.md prediction P1 and SAMuon's edge-of-stability finding.

# Muon improvement search, wave 3: SOAP-Muon core (2026-09-25)

Protocol: `research/adamw_spectra/MUON_CASE.md`, "Wave 3". The arm reproduces the modded-nanogpt Track-3 SOAP-Muon core, without the record's other tricks. Its denominator was checked against record #44's `soap_precondition_momentum`:
1. The momentum is projected onto the eigenbases of EMA(GGᵀ) and EMA(GᵀG), with β2 0.9 and one sorted-QR refresh per step.
2. It is divided by the square root of the EMA of the projected momentum's squares.
3. It is rotated back, its Frobenius norm is restored, and NS is applied.

| arm | final val NLL | paired Δ vs M (A6000, seed 260925) |
|---|---|---|
| S_soap_lr0.01_s260925_a6000_gpusvd | 3.68699 | **−0.0276** (promising, single seed) |

S is worse early: +0.23 at step 50 and +0.055 at 100, while its statistics warm up. It is better from step ~120: −0.077 at 200, −0.035 at 1000, −0.028 at the end.

The first attempt, `S_soap_lr0.01_s260925_a6000`, was cancelled after the spectral measurement moved to the GPU (`AMENDMENT_measurement_device.md`).

Replication and dissection: wave 4 (`../improve_w4_20260925`).

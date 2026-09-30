# The cube-root momentum law and the missing constant-β controls (second-order audit, 2026-09-28)

Review (21:48 CDT): every warmup gain so far is against β 0.9, not the best constant, and one law, 1 − β* ∝ (q/r)^⅓
(Cutkosky & Mehta's momentum-error bound for normalized updates, anchored at β 0.95 for 1M), matches the β ladder
across batch sizes. Its sharp prediction: at 16M the early β should be ≈ 0.45 (step 9) rising to ≈ 0.75–0.8 (steps
46–83). Decision argument: `research/adamw_spectra/MUON_CASE.md`, "Decision argument: test the cube-root momentum law
and the missing constant-β controls" (21:49 CDT).

- **g20 (Ada), after the 1M PD α ½ arm:**
  1. `SPD_a0.5_b16M_T2x_lr0.028_momwarm0.45to0.8in55_s260925_ada`: β 0.45 → 0.8 over the first 55 of 184 steps
     (references: β 0.9 3.9840, β 0.8 3.9740, warmup 0.8→0.9 3.9628);
  2. `SPD_a0.5_b16M_T2x_lr0.028_mom0.7_s260925_ada`: constant β 0.7;
  3. `PD_a0.5_b4M_lr0.02_mom0.85_s260925_ada`: constant β 0.85 (references: 0.9 3.8589, 0.8 3.8629, warmup 3.8462).
- **priv-g14 (L40S), after the 1M PD-top α ½ arm:**
  1. `PD_a0.5_b16M_T2x_lr0.028_momwarm0.45to0.8in55_s260925_l40s` (references: β 0.9 4.0446, warmup 4.0179);
  2. `PD_a0.5_b16M_T2x_lr0.028_mom0.7_s260925_l40s`;
  3. `SPD_a0.5_b4M_lr0.02_mom0.85_s260925_l40s` (references: 0.9 3.8260, 0.8 3.8280, warmup 3.8166).
- **Predictions.** At 16M 2× the law schedule beats the 0.8→0.9 warmup and constant 0.7 by ≥ 0.005 (both optimizers);
  at 4M the warmup beats constant 0.85 by ≥ 0.003 (both).
- **Source.** Frozen `research/adamw_spectra` (muon.py 5d816a1f…: adds the default-off split momentum; otherwise as
  soaudit_warm1m). Kept checkpoints: law arms 9 and 46; constant 0.7 arms 46; 4M none.

## Results

- **Law schedule, 16M 2× (23:10 CDT).**
  - S∘PD (Ada): **3.9610**, vs the warmup's 3.9628 (−0.002), β 0.8 3.9740 and β 0.9 3.9840.
  - PD (L40S): **4.0183**, vs the warmup's 4.0179 (+0.0005) and β 0.9 4.0446.
  - A larger early lead (−0.12 … −0.15 vs β 0.9 at step 50) that the final does not keep. The prediction of a ≥ 0.005 margin over the warmup fails.
- Constant 0.7 (16M 2×) and constant 0.85 (4M): running.
- **Constant β 0.7, 16M 2× (00:12 CDT).** PD 4.0535 (vs warmup 4.0179, β 0.9 4.0446); S∘PD 3.9973 (vs warmup 3.9628,
  β 0.8 3.9740, β 0.9 3.9840). The schedules beat every constant tried at this horizon.
- **Constant β 0.85, 4M (00:43 CDT).** PD 3.8559 and S∘PD 3.8258, the best constants. The warmup beats them by 0.010 and
  0.009. Prediction 2 holds for both; prediction 1 (law > warmup) fails for both.

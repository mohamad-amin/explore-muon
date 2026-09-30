# Momentum warmup at 1M, ending where the buffer turns noisy (second-order audit, 2026-09-28)

At 1M the momentum buffer is signal-dominated at step 50 (noise fraction 0.06) and noise-dominated by step 200 (0.60)
(`direction_drift_probe.py`, clip-corrected). Does a warmup that ends by step 150 gain at 1M, as β 0.8→0.9 did at 4M
and 16M (5/5 pairs)? Decision argument: `research/adamw_spectra/MUON_CASE.md`, "Decision argument: a momentum warmup at
1M that ends where the buffer turns noisy" (21:24 CDT).

- **Arms.** PD α ¼ @0.01, 1M, 1468 steps, seed 260925, 4× A6000 (sbatch, run in parallel):
  - `PD_a0.25_lr0.01_mom0.95_s260925_a6000`: β 0.95 constant (control; earlier A6000 run of this arm 3.6876);
  - `PD_a0.25_lr0.01_momwarm0.8to0.95in150_s260925_a6000`: β 0.8→0.95 linear over the first 150 steps.
- **Predictions.** (b) − (a) ≤ −0.03 at step 100; (b) ≥ 3 steps ahead (smoothed train loss) from step 300 to the end;
  final difference ≤ −0.002.
- **Source.** Frozen `research/adamw_spectra` (muon.py 3452e244…, train.py 31cc2b3f…). Kept checkpoints: 50, 200.

## Results (2026-09-28 22:25 CDT)

- Warmup 0.8→0.95 in 150: **3.6849**; control β 0.95: 3.6870 (−0.0021; earlier A6000 run of the control 3.6876).
- Lead: −0.097 @100, −0.101 @150, −0.036 @300, −0.017 @500, −0.008 @900, −0.004 @1300. Step lead 12–22 steps from 200
  on. All three predictions hold; the final gain is small.

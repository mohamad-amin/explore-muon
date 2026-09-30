# Do the SNR gate and the momentum warmup stack? PD α ½ at 4M (second-order audit, 2026-09-29)

The two robust changes of 2026-09-28/29, the momentum warmup (β 0.8→0.9 over the first 120 steps: 3.8462, vs β 0.9 3.8589)
and the SNR-gated tail amplification (3.8560), act on different things: time filtering and per-direction scaling. This
arm combines them. PD α ½ @0.02, 4M, 368 steps, seed 260925, Ada (g20).
- Prediction: they stack roughly additively: combined ≤ 3.845 (warmup − 0.001); a result above the warmup alone (> 3.8462)
  means they interact negatively.
- Source: frozen `research/adamw_spectra` (same gate code as soaudit_snrgate_*).

## Stopped (2026-09-29 01:19 CDT)

Stopped about 2 minutes into the scientific run (srun step 2618555.226 cancelled) after the user's steering: "Please don't
spend time trying to simply improve the numbers, this is not a benchmark eval. We should try to pursue the goal." A
stacking test of two recipe changes is a number-improvement check, not an understanding question. No result is recorded.

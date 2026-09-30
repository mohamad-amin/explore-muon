# Momentum, clipping and Nesterov at batch 4M (second-order audit, 2026-09-27)

Decision: `research/adamw_spectra/MUON_CASE.md`, "Independent review of the audit's GN claims" (05:41 CDT) and
"Momentum sweep at 4M, and the warm-CG test" (05:44 CDT).

- **Setup.** Batch 4,194,304, 368 steps, seed 260925, warmup 13, kept checkpoints 37/183/330. priv-g14, L40S, paired with the L40S Muon and S∘PD runs in `../soaudit_batch4m_20260927` and `../soaudit_alpha4m_20260927`.
- **Arms**, in queue order:
  1. Muon @0.014 with β 0.81, which keeps momentum's half-life constant in tokens (13.5M tokens, as β 0.95 at 1M). The 4M baseline at β 0.95 is 3.9524.
  2. S∘PD α ¼ @0.01 without clipping (grad_clip 1e6). The clipped L40S control is 3.8454; S∘PD clips on 94–96% of steps.
  3. Muon @0.014 with β 0.9.
  4. Muon @0.014 with Nesterov momentum.
  5. Muon @0.014 without clipping. Muon clips on 7% of steps.
- **Code.** Frozen from the live source. The only change from the earlier 4M cohorts is that `run_arm.py` now allows `muon_momentum` and `grad_clip` as treatments (82 unit tests pass).

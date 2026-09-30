# Two-sided data norm at batch 4M (second-order audit, 2026-09-27)

The two-sided rule (α ¼, β ¼ with GN labels; `research/adamw_spectra/MUON_CASE.md`, "Two-sided data norm") beat PD α ¼ by
0.0052 at 1M. The input-side gains grew ~3.7× from 1M to 4M. This cohort asks whether the output side's gain grows too.

- **Arms.** TS at LR 0.02 and 0.01. Batch 4,194,304, 368 steps, seed 260925, warmup 13, kept checkpoints 37/183/330. g20 (RTX 6000 Ada), paired with PD α ¼'s Ada bracket in `../soaudit_batch4m_20260927`: 3.8854 / 3.8847 / 3.8959 at LR 0.01 / 0.02 / 0.04.
- **Code.** Frozen copy of the live source, the same code as `../soaudit_twosided_20260927`.
- **Launch.** `start_g20.sh` runs `node_queue.py` inside an srun step on allocation 2618555.

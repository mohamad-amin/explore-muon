# Does the realized speedup keep growing with batch size? 16M-token batch (second-order audit, 2026-09-27)

At 1M the curvature-aware normalized maps are ~1.1× faster than Muon in step-equivalent terms, at 4M 1.2–1.34× and
growing through training. This cohort repeats the 4M β 0.9 comparison at 16,777,216 tokens per step, with the same
1.54e9-token budget (92 steps). Decision note and predictions: `research/adamw_spectra/MUON_CASE.md`, 16:50 CDT.

- **Changes from the 4M β 0.9 arms:**
  - batch 16M;
  - warmup 3 steps (tokens-constant);
  - PD/TS statistics refresh every 2 steps;
  - validation every 8 steps;
  - kept checkpoints 9, 46 and 83 (floor anneals);
  - checkpoint every 23 steps.
  Everything else is copied from `soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s`, `soaudit_strength4m_20260927/SPD_a0.5_..._lr0.02...` and `TS_a0.5_b0.5gn_..._lr0.02...`, and `soaudit_mom4m4_20260927/PD_a0.5_..._lr0.02...`.
- **Arms** (seed 260925, momentum 0.9):
  - g20 (RTX 6000 Ada): Muon @0.028, S∘PD α ½ @0.04, TS α ½ β_out ½ @0.04, Muon @0.04, S∘PD α ½ @0.028.
  - priv-g14 (L40S): Muon @0.02, TS α ½ β_out ½ @0.028, PD α ½ @0.04, PD α ½ @0.028.
- **Code.** Frozen from the live source, which is byte-identical to the `soaudit_strength4m_20260927` freeze (84 unit tests passed there). `make_cohort.py` is copied from `soaudit_batch4m_20260927`; `node_queue.py` from `soaudit_strength4m_20260927`.

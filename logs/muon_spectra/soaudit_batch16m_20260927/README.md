# Does the realized speedup keep growing with batch size? 16M-token batch (second-order audit, 2026-09-27)

At 1M the curvature-aware normalized maps are ~1.1× faster than Muon in step-equivalent terms, at 4M 1.2–1.34× and
growing through training. This cohort repeats the 4M β 0.9 comparison at 16,777,216 tokens per step, with the same
1.54e9-token budget (92 steps). Decision note and predictions: `research/adamw_spectra/MUON_CASE.md`, 16:50 CDT.

- **Changes from the 4M β 0.9 arms:**
  - batch 16M;
  - warmup 3 steps (tokens-constant);
  - PD/TS statistics refresh every 2 steps;
  - kept checkpoints 9, 46 and 83 (floor anneals).
  Validation stays at every 50 steps (the frozen `run_arm.py` rejects non-treatment changes). At 16M, each step's training loss on unseen data (16.8M tokens) serves as the loss curve.
- **First attempt**, kept as `../soaudit_batch16m_20260927_failed_config`. It also set `validation_every` 8 and `checkpoint_every` 23. `run_arm.py` refused every arm before training ("Arm changes non-treatment settings").
  Everything else is copied from `soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s`, `soaudit_strength4m_20260927/SPD_a0.5_..._lr0.02...` and `TS_a0.5_b0.5gn_..._lr0.02...`, and `soaudit_mom4m4_20260927/PD_a0.5_..._lr0.02...`.
- **Arms** (seed 260925, momentum 0.9):
  - g20 (RTX 6000 Ada): Muon @0.028, S∘PD α ½ @0.04, TS α ½ β_out ½ @0.04, Muon @0.04, S∘PD α ½ @0.028.
  - priv-g14 (L40S): Muon @0.02, TS α ½ β_out ½ @0.028, PD α ½ @0.04, PD α ½ @0.028.
- **Code.** Frozen from the live source, which is byte-identical to the `soaudit_strength4m_20260927` freeze (84 unit tests passed there). `make_cohort.py` is copied from `soaudit_batch4m_20260927`; `node_queue.py` from `soaudit_strength4m_20260927`.
- **Added 18:10 CDT:** Muon @0.014 (priv-g14, after the first queue). Muon @0.02 (4.9104) beat @0.028 (4.9635), so the bracket's bottom needed extending. The arm was created like `make_cohort.py` creates arms, into the same frozen cohort.

## Results (2026-09-27 20:10 CDT)
Best finals: Muon @0.02 4.9104; S∘PD α ½ @0.028 4.5742; TS α ½ β_out ½ @0.028 4.6677; PD α ½ @0.028 4.6711. Full table,
speedups and reading: `research/adamw_spectra/MUON_CASE.md`, 20:10 CDT entry; figure
`logs/muon_spectra/second_order_audit_20260926/figures_floor/speedup_by_batch.png`.

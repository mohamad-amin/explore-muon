# Newton-Muon and PMuon at LR 0.01 on the TTIC A6000 partition (2026-09-25)

User request (~21:45 UTC): run Newton-Muon and PMuon at LR 0.01 on the TTIC cluster. The g20 and
priv-g14 allocations were busy with wave 24, so both run as `gpu`-partition jobs on 4×RTX A6000.

- `frozen/` and `frozen_manifest.json`: the live `research/adamw_spectra` with the Newton-Muon and PMuon
  ports; code identical to `../h2h_h100_20260925/frozen`. `unit_tests.log`: 67 tests pass.
- Arms: `NM_lr0.01_s260925_a6000` (job 2622039) and `PM_lr0.01_s260925_a6000` (job 2622040), with the
  records' constants (see `MUON_CASE.md`, "Head-to-head with Newton-Muon and PMuon on H100").
- Seed 260925 on A6000 pairs with existing arms: tuned Muon@0.007 3.70630, Muon@0.01 3.71457 and
  PD@0.01 3.68909 (`improve_w*`).
- Single-seed screens: report the differences, make no claim. This cohort is deliberately not named
  `improve_w*`, because `compare_all.py` does not recognise these methods and would label them Muon.

## Results (2026-09-25 ~23:00 UTC; A6000, seed 260925, single-seed screens)

The first attempts (no `_gpusvd` suffix) were stopped by the controller's time gate: CPU SVD with 2 CPUs.
See their `FAILURE.md`. The `_gpusvd` reruns differ only in `svd_device`.

| method | LR | final val NLL | vs tuned Muon@0.007 (3.70630) | vs Muon@0.01 (3.71457) | PD@0.01 − method |
|---|---|---|---|---|---|
| Newton-Muon | 0.01 | 3.70884 | +0.0025 | −0.0057 | −0.0197 |
| PMuon | 0.01 | 3.71021 | +0.0039 | −0.0044 | −0.0211 |

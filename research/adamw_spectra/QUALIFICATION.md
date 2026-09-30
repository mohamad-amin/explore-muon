# Implementation qualification — 2026-09-24

This is instrument qualification, not evidence about 77M AdamW training.

Subsequent update: the user requested 25-step sampling and efficient execution.
See [EFFICIENCY.md](EFFICIENCY.md) for the current execution defaults, 11-test
suite, original-versus-optimized comparison, and successful tiny CUDA checks.
The results below preserve the initial qualification record; its statement
that CUDA was untested describes that earlier implementation milestone.

Full-size execution update: g20 qualification passed after correcting the
scalar Frobenius-norm reduction to FP64 (direct SVD and updates remain FP32).
The original 2e-4 normalized-energy tolerance was preserved. All 12 current
CPU tests pass. The 77M model takes about 3.94 seconds per steady update and
3.24 GiB peak allocated memory on RTX 6000 Ada. Three real-data LR pilots
are running; no selected-rate scientific conclusion is available yet. See
`logs/adamw_spectra/g20_20260924_212246_r2/QUALIFICATION_PASSED.json` and its
live pipeline. The failed normalization qualification is preserved in the
first attempt, `logs/adamw_spectra/g20_20260924_212246/`.

- Ten CPU tests passed in 18.755 seconds. Numerical checks cover multiple
  sequential AdamW writes including bias correction, epsilon and decoupled
  decay; no buffer mutation; known rectangular spectra and quantiles; zero
  and nonfinite inputs; token-shard boundaries; schedule/budget; LR selection;
  full model size/panels; bitwise CPU resume and measurement non-interference;
  orphaned partial-file recovery; and complete-versus-failed result reporting.
- A tiny synthetic six-update run exercised training, validation, full
  spectrum archives, checkpoints, and all PNG/PDF plot families. A separate
  three-rate, six-update synthetic sweep exercised paired initialization,
  selection, boundary reporting and selected-config export. Its chosen rate
  is not a recommendation for real-text training.
- Full-size CPU construction: **76,993,536 parameters**, **1,539,870,720
  tokens** at 20N, **1,469 updates**, final batch **561,152 tokens**.
  Exactly 24 matrices are measured at blocks 2/4/6/8. Available local data:
  3.2B training tokens and 100M held-out tokens, with disjoint shard identities.
- Direct SVD on random matrices with all 24 real measurement shapes completed
  in 1.322 seconds using four CPU threads. Normalized energy sums were
  checked in the saved record. This is an instrumentation timing, not measured
  training overhead or a forecast for ill-conditioned learned updates.
- Fresh-context independent design/code review supported the measurement and
  paired sweep. Fixes included pre-cooldown summaries, temporary-file recovery,
  device-local CUDA RNG handling, environment checks on resume, and explicit
  run status versus durable-checkpoint reporting.

CUDA execution and the real-data LR sweep have **not** been qualified. All
four L40S GPUs were observed at 99% utilization during the final occupancy
check; their jobs were left alone. The default LR 6e-4 remains provisional.
Before interpreting a scientific run, use the documented short device
qualification and fixed 300-update rate pilots. No full training was launched.

Evidence (repository-relative):

- `logs/adamw_spectra/qualification_20260924/tests.txt`
- `logs/adamw_spectra/qualification_20260924/full_size_cpu.json`
- `logs/adamw_spectra/qualification_20260924/tiny_run/`
- `logs/adamw_spectra/qualification_20260924/tiny_sweep/`

The earlier persistent smoke run predates the lightweight checkpoint sidecar;
the final test suite explicitly verifies that sidecar and failed-run reporting.

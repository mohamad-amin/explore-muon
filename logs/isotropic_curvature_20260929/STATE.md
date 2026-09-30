# Finite-curvature trajectory atlas: complete

2026-09-29. Completed under the user's "proceed" after the Su2511.00674
paper discussion. The previous observer goal remains cleared; no new goal
or training objective was created.

## Completed dataset

All18matrix panels and6joint states completed: Muon/PD, updates10/500/1300,
MLP-up blocks1/4/8, five fixed directions, four512-token score contexts,
two banks, eight separate calibration contexts, and15signed scales.
Full selected gradients, lagged momentum, actual writes, input/activation
geometry, per-token directional derivatives and loss curves, finite-span
Hessian/GN matrices, and source hashes are retained in run1/.

The measured Hessians are restricted to five directions per matrix and
three actual writes jointly. They are not full parameter Hessians. Four
contexts are not2048independent replications. The true-Hessian-minus-GN
term is second order; finite departures beyond the true quadratic are
reported separately. Directions and states both change with checkpoint.

## Execution and verification

- Producer unified exec session76815 exited0 after6024.69seconds, within
  the three-hour bound. PeakRSS about10.1GiB. Two CPU numerical threads.
- First completion watcher/session7100 exited0. It generated all main
  reductions/figures (181.41s) and geometry outputs (14.43s).
- Second watcher/session88113 exited0. Transmission analysis completed
  in17.96s; independent verification of every main CSV table passed in5.89s;
  denominator qualification, compact tables and overview figures completed.
- No producer/analysis restart, dropped panel or changed comparison grid.
- Qualification and final numerical/source checks passed. See
  qualification/result.json, run1/verification.json and analysis result files.
- All model measurements used the separate FP64 diagnostic source. No main
  training source, scheduler state, GPU job or sealed study was modified.

All three execution sessions are terminal; do not restart from historical
launcher commands. No continuation of this measurement panel remains queued.

## Main measured findings

- Actual/mean-left Hessian contrasts are4.16–20.14 across all18panels,
  despite exact per-token activation-norm preservation by the rotations.
- Along actual writes, H/GN changes from6.13–10.66early to0.857–0.996at
  midpoint and0.900–0.993late. Large tokenwise absolute differences remain;
  agreement in means conceals cancellation.
- At16times the actual write, even remainder / true quadratic is0.984–1.032;
  odd/even ranges−0.149to0.038. Directional anisotropy and nonquadratic
  departure are distinct measurements.
- Local GELU/down transmission reduces pooled GN orientation contrasts
  from4.93–11.54to1.07–3.69. The factor remaining after normalization grows
  strongly with training in the last selected MLP block.
- Joint cost for the three selected writes is1.40–1.71times the sum of their
  Hessian costs. At the actual step, the finite interaction is98.1–99.8%
  of the Hessian cross-term prediction.
- Input whitening changes displacement-moment predictions but does not
  establish spherical inputs or conditional radial loss cost.

## Final map and scope

REPORT.md is the complete interpretation and artifact index. README.md links
its compact plots and all18rows. RESULT_REVIEW.md records the independent
complete-result discussion. Earlier partial observations and protocol timing
amendment are preserved. MANIFEST.json records the final study artifact hashes.

This is a completed explanatory measurement task, not a claim of a better
optimizer or a general falsification of a distributional model. No optimizer
intervention, new radius range, state selection, GPU/training run, or change
to main/tiny-surrogate research follows automatically.

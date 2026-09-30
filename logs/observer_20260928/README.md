# Independent observer notebook

Started 2026-09-28 (CDT). This branch follows the user's observer goal: inspect
existing artifacts, find anomalies and connections, and test explanations before
proposing methods. It does not take over the main training program.

## Resource and evidence contract

- Initial work uses existing JSON/logs and CPU only, at most two BLAS threads per
  analysis. No training or GPU job is launched. The four-GPU training resources
  on g20 and priv-g14 remain available to the main program.
- Historical runs, failures, frozen sources, comparison criteria and source
  metadata are preserved. New output lives under this directory.
- Separate numerical facts, instrument scope, mechanistic interpretations, and
  intervention proposals. Counterexamples bound claims; they do not by
  themselves explain the real network.
- A robust optimizer/architecture improvement is a possible outcome, not a
  premise. An attractive local score is not a training-rate claim.

## First independent lenses

1. **Mean geometry:** connect mean-product momentum spikes, implicit bias routes,
   body centering and the head-whitening intervention. Notes in `mean_geometry/`.
2. **Spectral resolution:** audit whether saved Lanczos moments support exact
   spectral-band interpretations, and whether own-state relative bands compare
   equivalent directions. Notes in `spectrum_reliability/`.
3. **Momentum dynamics:** distinguish a temporal filter's ordinary closed-loop
   stability cost from a specific explanation of learned oscillations. Connect
   to clipping and beta sweeps. Notes in `momentum_dynamics/`.
4. **Body–auxiliary coupling:** compare body-only counterfactual steps with actual
   consecutive whole-model states. Notes/results in `body_aux/`.

## First concrete anomaly (13:24 CDT)

At the same 4M Muon checkpoint (beta .9, step 183), `step_profile/` reports
gradient cosine +0.095 while `river_hill/` reports -0.526. These are not yet a
contradiction: the probes change both data/sample size (128 vs 8192 sequences)
and the parameter update (body only vs full next state), plus numerical
precision. The former script's gradient-next path adds only hidden-matrix
displacements; the latter loads all next-state weights. The observer will
separate these factors instead of assigning the difference a mechanism.

Source code and raw artifacts:

- `../muon_spectra/second_order_audit_20260926/step_profile_probe.py`
- `../muon_spectra/second_order_audit_20260926/river_hill.py`
- `../muon_spectra/second_order_audit_20260926/step_profile/soaudit_mom4m_20260927__M_b4M_lr0.014_mom0.9_s260925_l40s__183.json`
- `../muon_spectra/second_order_audit_20260926/river_hill/M_b4M_lr0.014_mom0.9_s260925_l40s_183.json`

An independent peer supported a bounded four-corner check, with these limits:
report signed losses and absolute gradient norms, avoid interpreting ratios
near cancellation, include all auxiliary parameters, and do not turn a small
held-out sample into a claim about long-run dynamics.

## Status

Initial artifact/algebra checks and the bounded saved-checkpoint diagnostic are
in progress. No discovery or optimizer improvement is claimed.

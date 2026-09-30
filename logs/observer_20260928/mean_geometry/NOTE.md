# Observer evidence: the mean direction and head whitening

Read-only source/artifact audit on 2026-09-28. The only new computation was a tiny NumPy algebra check and JSON extraction, with at most two BLAS threads. No checkpoints, model forwards, GPU calls, or training were used. `algebra_probe.py` regenerates `evidence.json`. Shared protocols and research state were not edited.

## 1. The head's norm control is weaker than the protocol claims

**Established from code.** `research/adamw_spectra/MUON_CASE.md:3807` says the update is norm-matched to its unwhitened version, so only the direction changes. The actual head step in `research/adamw_spectra/muon.py:892–898` computes

`A_R = (m_hat R) / (sqrt(EMA[(G R)^2]) + eps)`, then `D = A_R R * ||A_R|| / ||A_R R||`.

Thus `||D|| = ||A_R||`, not the norm of ordinary-coordinate Adam `A_I`. The frozen scientific implementation confirms this at `logs/muon_spectra/soaudit_headwhite16m_20260928/frozen/adamw_spectra/muon.py:887–892`. It does not maintain a counterfactual ordinary-coordinate second moment. The qualification test reproduces this formula and checks only one gradient step (`test_whitening.py:123–141`), when both steps are nearly elementwise signs with the same norm.

**Algebraic counterexample, not an empirical run measurement.** A fixed, symmetric positive definite 2D root and two gradients give `||A_R|| / ||A_I|| = 1.0000` on step 1 but **1.32355 on step 2**. See `evidence.json`, `norm_matching_counterexample`. No changing covariance or numerical error is necessary. The implementation is internally consistent; the claim that its comparison with ordinary Adam changes only direction is unsupported.

**Consequence.** The observed head-whitening gain/loss combines coordinate geometry, the size of the adaptive step, and potentially a history mismatch when the root changes. This does not invalidate the measured training results.

**Cheapest falsification.** If future logged head gradients/moments become available, carry a shadow ordinary Adam denominator and report `||A_R||/||A_I||` and the response along the mean. Current scalar step logs do not contain these. A future measurement on one saved state can compare the current whitened step with a same-norm ordinary Adam step, but must not claim the shadow denominator reproduces the missing trajectory history. No training repair is recommended from this audit alone.

## 2. Centering improves the early head deficit, but does not preserve its mean step

**Saved measurements.** `logs/muon_spectra/second_order_audit_20260926/head_center_probe.log:1–3` reports at the PD reference:

| State | Mean energy / trace | cos(mean, leading eigenvector) | Uncentered top/mean | Centered top/mean |
|---|---:|---:|---:|---:|
| 16M step 9 | 0.599 | 0.999 | 314.4 | 118.8 |
| 16M step 46 | 0.174 | 0.991 | 97.2 | 38.6 |

These are 32-sequence probes, not the exact online EMA (`head_center_probe.py:22–41`). Early head anisotropy is much more extreme than the 80–120× mid/late-state summary used to motivate the intervention. A large mean route diminishes during training, while substantial centered anisotropy remains.

The raw `scientific/steps/step*.json` files, extracted into `evidence.json`, show train-NLL gaps versus PD:

| Step | Uncentered head alpha 1/2 | Centered head alpha 1/2 |
|---|---:|---:|
| 5 | +0.92684 | +0.37238 |
| 10 | +0.93302 | +0.31053 |
| 15 | +0.82130 | +0.36808 |

Centering removes approximately 55–67% of this early deficit, but does not eliminate it. These are an interim prefix from an independently running authorized job, one seed. No endpoint conclusion is made. Completed uncentered head results are 4.965640 (+0.294529) for alpha 1/2 and 4.630769 (−0.040343) for alpha 1/4, against reference 4.671111.

**Mathematical limitation.** Subtracting `mu mu^T` from C does not enforce `R mu = mu`, and even that equality alone would not make entrywise Adam in transformed coordinates preserve the ordinary Adam response along mu. The code's comment and notebook phrase “keeps its Adam step” are explanations to test, not properties of centering. A first-step, centered-covariance example in `evidence.json` gives only **0.63293×** the ordinary Adam mean response while keeping the full step norm equal.

**Discriminating hypothesis.** Centering helps because it weakens suppression of the early output-bias route, while the residual deficit comes from either residual mean suppression, adaptive magnitude changes (finding 1), or body/head co-adaptation. A spectral summary alone cannot separate these.

**Cheapest falsification.** Measure the actual directional responses `D mu`, `D_Adam mu`, and `D P_perp` at an early state; also evaluate the body-only, aux-only and full consecutive-step counterfactuals on one common sample. If the mean response is already preserved while the deficit remains, “suppressed unigram learning” cannot be the complete explanation. A mean-preserving comparison must explicitly replace the mean column, e.g. `D_white P_perp + D_Adam P_mu`; ordinary global norm rescaling would destroy that preservation. This is a diagnostic construction, not a recommended training change.

## 3. The old body-centering result has the opposite sign, and the comparison contains a time-scale difference

**Earlier empirical anchor.** `logs/muon_spectra/improve_w4_20260925/summary_all.md:20,30,55` gives same-seed/same-hardware final NLLs: tuned Muon **3.70630**, body PD **3.68909**, centered body PD **3.69736**. Centering lost **0.00827**, about half PD's benefit. At step 100 it also lost 0.0563 versus uncentered PD. Mean-only body whitening did not beat tuned Muon (same file:114,119). These do not support either “always suppress means” or “always protect means.”

The original spike diagnostic found a stable input mean but noisy/drifting mean error (`logs/muon_spectra/spike_diagnostics_20260925/README.md:60–77`), and architecture changes preserved mean-product spikes while modestly improving loss (`RESEARCH_STATE.md:593–621`). A mean-product spike can serve a useful output-bias route; its mere existence does not establish a defective update. The head/body sign difference is an empirical question about function and stage, rather than a contradiction in matrix algebra.

**Source-level competing explanation.** `StatLinear` updates its covariance EMA **on every microbatch forward** with decay 0.998 (`model.py:74–100`), whereas Adam's denominator uses a per-optimizer-step beta2 of 0.95. With four ranks, 16-sequence microbatches and length 512, this is 32 forward updates per 1M step versus 512 per 16M step. The effective covariance decays are **0.937945 and 0.358787**, corresponding to approximately **15.11 and 0.56 optimizer steps of stationary lag**. The alpha/centering/batch comparisons therefore also differ in the relative time scales of the input frame, first moment and second moment. Nonpersistent covariance state means this history is absent from ordinary checkpoints.

**Hypothesis, not a finding.** Fast-moving early head coordinates combined with a much slower denominator could account for some transient penalty. The stable body input mean measured at final checkpoints does not rule out this early effect. This explanation competes with, and may interact with, unigram learning rather than replacing it.

**Cheapest falsification.** Inspect any saved online roots/covariance telemetry first; none was found in the scalar logs examined here. At the next already-authorized instrument opportunity, record root change and mean-response ratios, rather than launch another arm. If the online root is nearly constant during the early deficit, this specific history-mismatch explanation weakens substantially. Batch-dependent covariance time constants should be disclosed when interpreting the mean-geometry comparison.

## Scope and source caveats

- Early body tests: 1M batch, 1469 steps, alpha 1/4, body momentum 0.95. Head tests: 16M batch, 92 steps, alpha 1/2 body, momentum 0.9, head Adam. Their different signs are not a controlled causal interaction.
- The training measurements distinguish body momentum and the head's Adam adaptive step. All intervention results remain ordinary next-token NLL outcomes.
- The head centering prefix is deliberately recorded as a dated partial observation. It is not a claim about the job's current state or final result.
- The algebra probes prove that stated invariances do not hold generally. They do not measure the size of these effects in the real network.

# Muon improvement search: findings (2026-09-25)

> **Correction (23:15 UTC).** The curvature-exponent probe was limited by float32 resolution (see `../gamma_probe_20260925/README.md`). The claim below that α = γ/2 ≈ ¼ is *derived* from measured curvature is withdrawn: where curvature is resolved it grows close to linearly with input variance. α = ¼ is an empirical choice. The exact Hessian-vector-product re-measurement gives γ ≈ 0.99 (curvature ∝ input variance), so the curvature-matched exponent would be ½, which fails; the best exponent ¼ is its square root (see the gamma-probe README). The mean-input energy shares quoted here mix architectures: in the frontier-norm model they are 5–44% under Muon. See MUON_CASE.md, "Correction: the curvature-exponent probe was resolution-limited".

Numbers regenerate from `compare_all.py` → `summary_all.md`. The protocol, decision arguments, predeclared predictions and claim rules are in `research/adamw_spectra/MUON_CASE.md`, from "Muon improvement search" onward.

**Setting.** Frontier-norm GPT on FineWeb: depth 8, width 512, 8 heads, no biases, RMSNorm and QK-norm, about 77M parameters.
- Horizon: 1469 steps × 1M tokens (20 tokens per parameter). Warmup 50, then constant LR, then linear cooldown over the last 10%.
- Optimizer: Muon on the body matrices (momentum 0.95, five Polar-Express-style NS steps) and AdamW on everything else.
- Every comparison is paired: same seed and same GPU type.
- Baseline: tuned Muon at LR 0.007, which beats 0.005 and 0.01 on the same seed and beats 0.01 on three seeds and two GPU types. The earlier default of 0.01 was 0.006–0.009 worse.

## The result: partial data-norm Muon

For each body matrix, with M its momentum and C = E[xxᵀ] the running second moment of its inputs (scaled to unit mean eigenvalue, plus 1e-3·I):

  ΔW = s · polar(M C^−¼) C^−¼,  s = rescale to Muon's Frobenius norm.

This is steepest descent under the partial data norm ‖ΔW C^¼‖_op. Muon is α = 0; α = ½ is the operator norm measured on the input distribution. The method needs only forward statistics. A minimal reference is in `research/adamw_spectra/data_norm_muon.py`.

**Vs tuned Muon** (PD at its bracketed best LR 0.01: 0.007 and 0.014 are worse; Muon at 0.007). α and LR were chosen on selection seeds 260925/260926.

| check | Δ val NLL (nats/token) |
|---|---|
| Width 512, fresh seeds on 3 GPU types | −0.0186 (L40S), −0.0198 (RTX 6000 Ada), −0.0189, −0.0200 (A6000); mean −0.0193, p < 0.001 |
| Width 512, all 6 seeds | mean −0.0188, SD 0.0010 |
| 2× token budget | −0.0164 (L40S), −0.0145 (RTX 6000 Ada) |
| Width 768 (134M), LRs re-bracketed | Carried-over LRs confirmed (Muon 0.007, PD 0.01). Fresh seeds −0.0118 (L40S), −0.0169 (RTX 6000 Ada), −0.0130 (A6000); mean −0.0139, p ≈ 0.006, ≈ 72% of the width-512 gain. Selection seed −0.0142 |
| Token efficiency (PD at 0.70× and 0.85× budgets) | PD reaches tuned Muon's final loss with ≈ 9% fewer tokens, ≈ 1.10× |

**Why α = ¼: equal curvature cost across input directions.** Loss curvature along an input eigendirection q_j grows as λ_j^γ, measured directly by finite differences over 24 matrices:
- γ ≈ 0.52 at width 512;
- γ ≈ 0.44 at width 768.

Muon's polar step gives every input direction a unit response ‖ΔW q_j‖ = 1 on square and tall matrices. Its curvature cost per direction is therefore ∝ λ_j^γ: the few high-variance directions, led by the shared token-mean input, are the stiffest and cap the LR.

The partial data norm gives ‖ΔW q_j‖ ∝ λ_j^−α, a cost ∝ λ_j^(γ−2α), which is equal across directions when **α = γ/2 ≈ 0.22–0.26**. This matches:
- the swept optimum: α ⅛ −0.017, ¼ −0.026, ⅜ −0.021, ½ −0.001 vs Muon@0.01;
- the higher best LR (0.01 vs 0.007);
- the failure of the full data norm (α = ½ assumes γ = 1).

Setting α = γ/2 per layer kind gives no further gain (−0.0171 vs −0.0177).

**Caveat on the derivation.** The same probe data are fit at least as well by curvature ∝ λ_j + b (a floor at b ≈ 0.1 × mean λ). That model implies the *damped full* data norm (α = ½, damping 0.1–0.3), which gains nothing vs tuned Muon (−0.0017, −0.0050). So the curvature measurement is consistent with α = ¼ but does not uniquely derive it. Gradient noise in low-variance directions, which the argument ignores, plausibly favours the milder exponent. Treat α = ¼ as chosen empirically and matched by the power-law exponent.

**Controls.**
- Muon's direction at PD's per-layer output energy equals tuned Muon (−0.0017), so the gain is geometric, not a hidden LR cut.
- Instrumentation alone: +0.0007.
- Exact polar (NS numerics): +0.0052.
- Centering C halves the gain (−0.0089), so the shared mean direction matters as part of the full anisotropy.
- Inside-only (−0.0147 at α = ½) and outside-only (−0.0158) versions each give part; the sandwich gives the most.

## The stacked method: SOAP-Muon in data-norm coordinates (S∘PD)

SOAP-Muon's core runs on M C^−¼, with statistics from G C^−¼, then post-multiplies by C^−¼ with the same RMS rule. Vs tuned Muon:
- n = 6 on 3 GPU types: −0.0284, −0.0327, −0.0302, −0.0325, −0.0302, −0.0336. Mean −0.0313; the 4 fresh seeds −0.0316. The LR is bracketed: 0.007 −0.0273, 0.01 −0.0284, 0.014 −0.0254.
- The SOAP-Muon core alone (Track-3 reference) is −0.0235 (n = 5). **The data norm improves SOAP-Muon by 0.006–0.009 at every seed.**
- At width 768: −0.0241 (L40S) and −0.0270 (RTX 6000 Ada) vs tuned Muon, mean −0.0256 (≈ 82% of width 512). PD on those seeds gives −0.0118 and −0.0169. The SOAP-Muon comparison at width 768 is running.
- The gain decomposes into two sides: output-side SOAP in whitened coordinates keeps 79% of the stacking gain (−0.0261).
  - **input side:** the data norm (forward statistics);
  - **output side:** SOAP's entrywise normalization in the left eigenbasis of the whitened gradient.

## How the search got here (the spike branch and SOAP dissection)

1. **The momentum spike is the extreme case of input anisotropy, not the lever itself.**
   - The top singular pair of Muon's momentum is a token-mean product ē x̄ᵀ.
   - The shared mean input x̄ holds 10–70% of E‖x‖², and its column is 7–276× stiffer than random input directions.
   - Deflation, whitening only x̄, dropping the spike step, and capping the head's step do not beat tuned Muon. Nesterov does not either.
   - The centered inputs are also strongly anisotropic: top/median eigenvalue 55–590, effective rank 50–240 of 512.
2. **SOAP-Muon's gain over Muon is mostly input-side.**
   - Input-side basis only keeps ~80% (n = 4).
   - The standard basis keeps 18%.
   - An activation basis works as well as the gradient Gram.
   - The momentum-vs-gradient second moment does not matter.
   - The gain sits on the square and tall matrices, not MLP-down.
3. **The data norm reproduces most of it from forward statistics, and the two stack.**
   - SOAP's gain leans toward rare target tokens, like Adam on heavy-tailed data; the data norm's is broad across frequencies (`../frequency_probe_20260925`).
   - On square and tall matrices an NS output cannot rebalance step size across input directions, but the data norm's post-multiplication can.

## Compute and limits

- **Step time vs Muon** on one L40S node (median / mean), unoptimized eager code:
  - PD +6% / +9%;
  - SOAP-Muon +9% / +8%;
  - S∘PD +15% / +19%.
- **Wall clock.** Token efficiency is ≈ 1.10× for PD, and by Muon's 1×→2× slope ≈ 1.16× for SOAP-Muon and ≈ 1.22× for S∘PD. Wall-clock gains are therefore small today, ≈ 1.0–1.07×. PD's overhead is the FP64 eigh every 10 steps and the covariance EMA, both easy to cut.
- **Scope.**
  - Depth 8, widths 512 and 768 (≤ 134M parameters);
  - one data set;
  - 20 and 40 tokens per parameter;
  - one batch size.
- **State as:** holds across 1.5× width (≈ 72% of the gain kept) and 2× horizon (≈ 82% kept) at ≤ 134M. It is not a scaling law.
- **Novelty.** The method is in the activation-sandwich family proposed at toy scale (GO-MUON: both sides, exponent ¼). The input-only full-matrix form, its curvature-matched exponent, its stacking with SOAP-Muon, and this evidence appear new (literature scan, MUON_CASE.md waves 5–6).

# Curvature exponent γ along input eigendirections (2026-09-25)

`gamma_probe.py` → `gamma.json`. Read-only, on the final weights of tuned Muon (LR 0.007, seed 260925, A6000, val 3.70630).

For each of 8 body matrices:
- take the eigenvectors q_j of its input second moment C at 12 ranks spread over the spectrum;
- measure the finite-difference loss curvature along u q_jᵀ (2 random unit outputs u, averaged; 64 fresh sequences, FP32);
- fit log curvature against log λ_j. The slope γ is the exponent in curvature ∝ (qᵀCq)^γ.

| matrix | γ | positive curvatures |
|---|---|---|
| block02.q | 0.456 | 10/12 |
| block02.up | 0.741 | 12/12 |
| block04.v | 0.747 | 12/12 |
| block04.o | 0.557 | 12/12 |
| block06.up | 0.721 | 11/12 |
| block06.down | 0.404 | 12/12 |
| block08.q | 0.356 | 11/12 |
| block08.o | 0.539 | 11/12 |

**Median γ = 0.548, so the curvature-matched data-norm exponent is α = γ/2 ≈ 0.27.** The α sweep found α = ¼ best, with ⅛ and ⅜ worse and ½ gaining nothing (MUON_CASE.md, waves 5 and 10).
- Loss curvature grows sublinearly with the input second moment (γ < 1). That is why the full data norm (α = ½, which assumes γ = 1) over-corrects, and why partial whitening wins.
- Across matrices γ ranges 0.36–0.75, so a per-layer α is a possible refinement. It is untested.

## All six kinds at depths 2, 4, 6, 8 (`--all-kinds`, 24 matrices)

Median γ by kind:

| kind | q | k | v | o | up | down |
|---|---|---|---|---|---|---|
| γ | 0.43 | 0.33 | 0.72 | 0.61 | 0.67 | 0.43 |
| α = γ/2 | 0.22 | 0.17 | 0.36 | 0.31 | 0.34 | 0.21 |

The overall median is 0.523, so α = 0.26; the uniform ¼ was chosen by sweep. The QK-normed query and key inputs have the flattest curvature scaling; values, the output projection and the MLP up-projection have the steepest.

## Width 768 (tuned Muon@0.007, seed 260925, 24 matrices)

**Median γ = 0.44, so α = γ/2 = 0.22.** At width 512 these were γ = 0.52 and α = 0.26.

| kind | q | k | v | o | up | down |
|---|---|---|---|---|---|---|
| γ | 0.40 | 0.28 | 0.68 | 0.41 | 0.65 | 0.39 |

The curvature exponent falls slightly with width, but the curvature-matched α stays near the ¼ carried over unchanged. At width 768 the selection-seed bracket confirmed the carried-over LRs, with PD −0.0142 vs tuned Muon.

## Correction (2026-09-25 ~23:15 UTC): resolution limit

The second differences above are quantized at float32 loss spacing (2^−22). Most low-variance directions sit within a few spacings of zero, and only the top ~8 eigendirections per matrix are resolved.
- **Resolved points only:** γ ≈ 0.69–1.00 (width 512) and 0.67–0.96 (width 768).
- **Earlier values** (0.52 and 0.44): artifacts of unresolved points.
- **Replacement:** exact Hessian-vector-product measurement in `gamma_probe_hvp.py` → `gamma_hvp.json`.

## Exact Hessian-vector-product measurement (2026-09-25 ~23:35 UTC)

`gamma_probe_hvp.py` → `gamma_hvp.json`. It uses tuned Muon@0.007 (seed 260925, width 512) and the same 24 matrices and 12 input eigendirections each. Curvature Dᵀ H D along D = u q_jᵀ comes from double backpropagation in FP32 with the math attention kernel, averaged over 3 random unit outputs u. All 288 values are resolved and positive.

| kind | q | k | v | o | up | down |
|---|---|---|---|---|---|---|
| γ (median over 4 depths) | 1.00 | 0.78 | 1.02 | 1.02 | 0.92 | 0.96 |

**Median γ = 0.99. Loss curvature along an input direction is essentially proportional to its second moment**, as a Gauss–Newton picture predicts. K is flatter, likely because of the QK-norm.
- The curvature-matched ("equal cost") exponent would be α = γ/2 ≈ ½, the full data norm. It gains nothing vs tuned Muon, while α = ¼ is best.
- So α = ¼ is **not** explained by curvature matching. It sits halfway, in log scale, between Muon (α = 0) and the curvature-matched norm (α = ½).
- **A plausible, untested reading:** with gradient noise that also scales with curvature, the square root of the curvature-matched preconditioner is preferable. That is the same square root used by Adagrad and Shampoo.

A width-768 HVP measurement is running (`gamma_hvp_run2_w768.log`).

**Width 768 (exact HVP, tuned Muon@0.007, seed 260925).** Median γ = 0.985: q 0.99, k 0.74, v 0.94, o 1.03, up 0.98, down 0.99. This is the same as at width 512, so curvature is proportional to input variance at both widths.

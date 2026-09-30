# Paper-figure and expert-claim check (read-only)

2026-09-25. This checks factual claims in `explain_paper_gist.md` and the attached
expert discussion. It checks them against the vector figures of arXiv 2606.04058v2
and against the saved local 8/12-layer Muon pre-NS spectra. No training was run
and no job was launched. No model tensor was re-decomposed; the analysis reads
archived singular values and figure geometry only.

## Sources

`sources/` holds the paper's figure SVGs, fetched from
`https://arxiv.org/html/2606.04058v2/<name>`:

- `svalue_hist_2.8B_1450_all_layers.svg` (Fig. 14)
- `scaling_laws_all_quantiles.svg` (Fig. 16, same data as Fig. 1)
- `svalue_hist_2.8B_1450_combined.svg` (Fig. 6)

Checksums are in `sources/SHA256SUMS`.

The extraction reads bar and line geometry. It uses tick spacings read from the rendered tick labels. Three checks validate it:

- every histogram panel sums to exactly 2559 or 2560 singular values, with integer error ≤ 0.22;
- the 16 attention panels of Fig. 14 agree with Fig. 16 at 2.8B to within 1–9%;
- the recovered exponents reproduce the printed ones.

## Findings

1. **The MLP labels are swapped between figure sets.** Take the MLP panels under their printed labels. Fig. 14 then disagrees with Fig. 16 by 1.2–4.4×. After swapping Up and Down, they agree to within 1–7% at all four depths (24% for the noisy final pair). The steep −0.96 matrix goes by three names:
   - "Final MLP Projection" in Fig. 5 and the text;
   - "MLP proj" in Fig. 6;
   - "MLP Down" in Fig. 14.

   It is therefore most likely the down projection (GPT-2 `c_proj`). In that case Figs. 1/7/12/13/15/16 mislabel Up and Down. The final-block exponents then read: down ≈ 0.96, O 0.66, V 0.58, up ≈ 0.52, Q 0.46, K 0.38. The authors' code would settle this.
2. **Fig. 6 is 2.8B data.** Each matrix has 2,560 singular values, and the bulk medians equal Fig. 14's L23/L31 values. Only its "Layer 20/27" panel titles are wrong.
3. **Outlier share at 2.8B, from Fig. 6.** σ₁²/‖M‖_F² ≈ 0.92 for final O, ≈ 0.985 for the steep MLP matrix, and ≈ 0.30 for mid-late Q. An exponential-bulk estimate from the median understates the bulk energy. The top decile of the bulk holds 78–87% of the non-outlier energy.
4. **Descent that 5-step NS keeps at 2.8B.** This is first-order descent along M, relative to exact polar, using the paper's NanoGPT polynomials:
   - steep MLP matrix: 0.77;
   - final V: 0.93;
   - final O and L7 V: 0.95;
   - everything else: ≥ 0.97.

   Rank-1 deflation restores ≈ 1.00 everywhere. Schatten-4 normalization alone does not help, because the spike dominates.
5. **The steep law is curved.** Its local log-log slopes run −0.31, −0.53, −1.51, −0.85, −1.42, −1.71 across the size steps. The small end is near the 1/√d width baseline and the large end is where each fit window covers the smallest fraction of training. The mid layers follow the width baseline at every step. In all 24 panels the q=0.9 slope is steeper than the q=0.1 slope.
6. **Local runs at width 512:**
   - The paper's 1300–1500 window overlaps our LR cooldown. It changes medians by −8% to +4% (mean −3% at 8 layers, −1% at 12). That is ≤ 0.01 of a fitted exponent at the 77M leverage point.
   - The MLP bulks are MP-like (q.9/q.5 ≈ 0.53), not piled at zero.
   - Going from 8 to 12 layers leaves final-block medians at ×0.98–1.38. The paper's 77M→160M step implies ×0.5–0.8.
   - V has the largest outlier share at every depth (ρ₁ 0.59–0.87).

## Limits

- Single seed and two local depths.
- The local model differs from the paper's: Linear biases, local initialization, LayerNorm. The 12-layer run also has the same token budget as the 8-layer one.
- Findings 1–5 read published figures. They are not the authors' raw data.
- The descent measure uses the momentum M as a proxy for the gradient. It says nothing about whether the bulk directions carry true-gradient signal.

## Reproduce

```bash
# figure extraction needs PyMuPDF (kept out of the shared .venv)
python paper_figures.py > paper_figures_output.txt          # from this directory
# local spectra, from the project root
.venv/bin/python logs/muon_spectra/paper_claims_check_20260925/local_spectra.py
```

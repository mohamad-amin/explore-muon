# Which step-spectrum claims survive the saved evidence?

**The saved artifacts support a real difference in how the methods distribute a step inside the measured Krylov space. They do not yet resolve an exact GN spectral projector onto the flattest eigenspace.** Global energy, slope and curvature checks are excellent; exact global checks do not qualify hard spectral-band boundaries. The distinction affects the proposed mechanism more than the observed training results.

This was a CPU-only audit of 37 saved states, including the three β=0.8 states added after the original 34-state report. No GPU, checkpoint, scientific source, historical audit, training job or project-wide note was changed. All new files are in this directory. Current allocations were checked with `squeue`: the user's `g20` and `priv-g14` allocations were running. The audit used at most two BLAS/OpenMP threads and finished in seconds. The first invocation with system Python lacked NumPy; rerunning with the project's existing `.venv/bin/python` succeeded without any installation.

## What was computed

`audit.py` imports the existing read-only `logs/observer_20260928/spectrum_reliability/check_quadrature.py`, reconstructs prefixes of length 8,16,24,32,40,48 from every saved 48-node actual-step measure, and extends its comparisons. The normalization λmax is the maximum saved Ritz value over the actual step, held-out gradient and momentum; it is held fixed across prefixes. This isolates changes to the quadrature from changes to the cutoff normalization. Common absolute thresholds and a shared Muon-state threshold are also compared.

Re-Lanczos of the saved discrete measure reconstructs the Jacobi prefix problem, including its signed gradient-step cross measure. It does not supply any new GN products or information beyond the saved 48-dimensional approximation. The largest recovered full48 node error is 2.3e-12. Every prefix preserves the saved total energy, slope and curvature within 4e-15 relative error. Saved totals agree with independently evaluated direct slope to 1.1e-6 and direct curvature to 1.6e-7 relative error, across all 37 states. This is strong evidence that the algebra and aggregation are functioning.

The instrument measures the **actual adjacent-checkpoint parameter change of hidden matrices**, including the effects of weight decay and parameter precision. It is not a pure momentum or pre-decay update measurement, and it excludes the full model's embedding/head parameter change. Its GN uses128 held-out validation sequences, and its gradient uses128 separate held-out training sequences.

Raw computed outputs: `results.json`, `bands.csv`, `summary.json`. `TABLES.md` is the compact evidence table. `source_hashes.json` fixes the exact inputs inspected, including the original audit and the paired profiles.

## What “exact” does and does not mean here

Let Q contain the Lanczos basis columns and T=QᵀGQ, with D in its span. For an indicator I selecting some eigenvalues of T, the vector D_low=Q I(T) QᵀD is a perfectly legitimate parameter direction. In exact arithmetic:

- its gᵀD_low and D_lowᵀG D_low are the reported band slope and curvature;
- components from disjoint Ritz bands are G-orthogonal within the Krylov space;
- c*=−gᵀD_low/(D_lowᵀG D_low) is its GN quadratic optimum, holding the other Ritz components fixed.

Therefore c*>1 can establish an available **within-Krylov quadratic-model improvement**. The audit should not erase that information. However, Q I(T) Qᵀ is generally different from I(G). A direction's small Rayleigh quotient does not imply that it has no high-eigenvalue components. Exact totals and a valid reduced quadratic do not make each band an exact GN eigenspace, nor establish that its slope is persistent across training. FP32 orthogonality and projected-operator errors were not independently measured here; the relevant basis/residual vectors were not saved in these JSON artifacts.

The source notes currently call the band split “exact” and attach a true-spectrum interpretation to it. The supported wording is **“a decomposition in the 48-step Krylov approximation to the sampled GN, with exact totals and valid projected quadratic comparisons up to numerical error.”**

## Which observations are robust, and which are not?

| Claim | Evidence and judgment |
|---|---|
| Most step energy is in the lowest observed curvature component | Robust descriptively across prefixes 24–48: 91.6–99.995% in the own-relative <1e-4 band. It is a broad concentration, not a resolved fine spectral density. |
| Every step has 95–100% flat energy | Literally false even in the saved full48 results: Muon has 92.66% at 4M@183, 91.60% at 4M@330, 94.41% at 1M@500, and 94.96% at 16M@83. The approximation issue is separate from this numerical correction. |
| Flat c*>1 | Holds in every saved full48 state (1.315–29.470), and every tested prefix32/40/48 (minimum 1.052). This supports a within-Krylov model opportunity. At prefix24 some states are below1, so the conclusion is resolution-dependent near its weak end. |
| The specific flat multiplier is established | No. At 16M@46, Muon c* rises 17.71→23.36→29.36 and SPD 3.21→3.85→4.55 at prefixes32→40→48. Neither has plateaued. Prefix disagreement does **not** prove full48 wrong; it leaves its true-spectrum interpretation unqualified. |
| Better geometries have greater flat slope share than Muon | Within the saved approximation, broadly robust. In five common mid/late groups (1M@500,@1300;4M@183;16M@46,@83), SPD's flat share exceeds Muon's at every prefix24/32/40/48 for own-relative1e-4, absolute1e-3 and absolute1e-2. At full48 the ratio is 2.17–5.48. This is a descriptive comparison at different optimizer-adapted states. |
| The relative cutoff alone creates that optimizer ordering | Not supported by this check. A common absolute1e-3 cutoff selects exactly the same full48 band in36/37 states; the common matched Muon-state cutoff selects the same full48 band in37/37. The lone abs1e-3 difference is Muon16M@9. Wide gaps between the lowest Ritz nodes make many threshold choices identical. This invariance is not independent confirmation of spectral resolution. |
| The ordering is threshold-invariant | No. Absolute1e-4 frequently contains no Ritz nodes. At16M@46, for example, SPD's first node is1.098e-4 and its estimated slope share below1e-4 is zero, while Muon's is7.32%. This is **not** evidence that SPD has no true flat slope: it shows the current rule cannot resolve that tighter threshold. |
| Flat slope share ranks training quality | Refuted as a general ranking rule by the existing observations. At16M@46/@83, TS shares21.22%/20.12%, exceeding SPD17.71%/17.54%, while recorded final losses favor SPD (4.5742 vs4.6677). SPD β0.8 improves final loss to4.4724 while its corresponding flat shares **fall** to15.87%/15.90%. Its absolute flat descent rises because its total descent rises. A fraction alone confounds the band numerator with the rest of the step. |
| Every band above1e-3 λmax has c*=0.3–0.9 after the early states | Too strong even at full48. Muon16M@46/@83 has1.679/1.322 in1e-3–1e-2; SPD16M@46 has0.019 and0.163 in the next two bands; TS has0.143 in1e-2–1e-1. Most late bands are near the whole-step edge, but the universal statement obscures meaningful exceptions. |
| The flat slope is the persistent training signal | Hypothesis, not established by this static decomposition. It requires temporal cross-correlation after accounting for changing projectors/states, plus an out-of-sample check. The existing period-2 observations motivate it but do not directly identify persistence in this exact band. |

Only one or two full48 Ritz nodes lie in the nominal flat band: one in36 states, two in Muon16M@9. Energy estimates can look stable while tiny curvature denominators and signed cross measures move substantially. Mixed slope is signed, unlike energy; a “share” is not a probability or a measure of useful optimization rate.

The source's under-stepping range is also not uniformly “largest early”: Muon16M@46 has29.36 versus9.33 at@9. In that particular early state the nominal low band gains its second Ritz node between prefixes40 and48, reducing c*. A moving number of nodes can interrupt otherwise smooth prefix trends.

## A surviving connection: momentum quality is stronger evidence than a universal flat-share ranking

The β0.8 comparison contains evidence that survives the band-resolution problem. At16M@46, total GN model quality a²/(2q) rises from0.03024 to0.04504; at@83 it rises from0.02795 to0.04042. These are total quantities, unaffected by how the spectrum is binned. They accompany the observed training improvement and are measured at the methods' different own states, so they are descriptive, not an isolated causal estimate.

There is also a robust coarse band contrast at@46: summing both bands from1e-3 to1e-1 λmax gives only about2.1–5.5% of β0.9's slope over prefixes24–48, versus19.1–23.7% of β0.8's. The precise smaller band's c* is unstable (β0.9 changes0.339→0.178→−0.059→0.019), but the broader contrast survives. This supports “fresher momentum improves alignment in the middle of the measured spectrum” more strongly than the exact quoted value0.02 or a claim that flat-share itself ranks rate. The cutoffs remain state-relative, and no conclusion about a transported common eigenspace follows.

## An independent lower bound on energy concentration

For PSD G and threshold τ, DᵀGD ≥ τ ||P_{λ≥τ}D||². Thus the true sampled-GN spectral energy share belowτ is at least max(0,1−q/(τ||D||²)), independent of the Ritz band approximation. The table reports this conservative bound.

At16M@46 it guarantees at least91.8% low-band energy for SPD and97.8% for TS, versus75.5% for PD and no nonzero guarantee for Muon. At4M@183 it guarantees95.4% for SPD and96.7% for TS. Thus some of the “large flat energy” conclusion has direct support beyond the plotted quadrature, but the95–100% universal claim does not. A weak bound does not disprove concentration. It says only that the first moment cannot certify it.

The paired profile files give other-sample/same-sample *total* curvature ratios0.937–1.058. Good total sample agreement does not certify the tiny flat component's curvature or a low eigenspace's sample stability.

## A two-dimensional example explains the limitation

`synthetic.py` uses G=diag(1e-5,1), D=(1,0.1), g=(0,−1), and cutoff0.02. The exact low eigenspace holds99.01% of D's energy, **zero** slope, and curvature1e-5. One-step Lanczos has a single Ritz node0.0099109, below the cutoff, and exactly reproduces all total energy/slope/curvature. Calling this node the low band reports100% of slope and c*=9.99.

There is no contradiction: rescaling that Ritz component really improves this quadratic, but it also rescales the original stiff coordinate. Its improvement is misattributed if called an exact-low-eigenspace intervention. This example proves only that exact totals are insufficient; it does not estimate the error of the project's48-step probes. The full reproducible numbers are in `synthetic.json`.

## Smallest decisive next measurement, not launched

1. **Numerical qualification:** on just the saved Muon and SPD16M@46 states, extend the same actual-step Lanczos computation to96 and192 products on exactly the same curvature sample. Save the tridiagonal entries, terminal residual, and projected held-out gradient; check basis orthogonality and low-Ritz residuals. Evaluate both the existing relative threshold and a common absolute1e-3 threshold, plus a smooth nearby cutoff. Compare energy, signed slope and curvature separately before inspecting their ratio. Agreement of96→192 would be evidence of stabilization, not proof; persistent movement would justify either longer probes or abandoning the hard-band attribution.
2. **Qualify the actionable interpretation:** form the resulting low-Ritz component, evaluate gᵀD_low and D_lowᵀG D_low directly, then repeat those scalars on one disjoint curvature/gradient sample and evaluate true loss for a small predeclared rescaling grid. This would distinguish a stable opportunity from sample-specific quadratic gain without training an optimizer. If only a one-step check is affordable, step1 and direct scalar validation answer the immediate numerical question; they do not establish a training-rate claim.
3. **Persistence is a separate question:** only after a stable direction is found, measure its next-step gradient correlation with transported directions or a fixed reference subspace. A positive one-step gain still does not guarantee an improvement along its own training trajectory.

No GPU work was launched. A future probe must use CPU or genuinely noncompeting resources under the user's restriction. The present evidence justifies refining the measurement claim now, without dismissing the successful training interventions or declaring an optimizer discovery.

## Reproduce

From the project root, run each script with the existing virtual environment and capped CPU threads:

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python logs/observer_connections_20260928/spectral_claims/audit.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python logs/observer_connections_20260928/spectral_claims/report_tables.py
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python logs/observer_connections_20260928/spectral_claims/synthetic.py
```

The source interpretations audited are the2026-09-28 12:13 and12:37 entries in `research/adamw_spectra/MUON_CASE.md`, the synthesis in `RESEARCH_STATE.md`, and `logs/muon_spectra/second_order_audit_20260926/step_spectrum_probe.py`. No external literature was needed for the direct artifact checks, elementary PSD bound, or analytic counterexample.

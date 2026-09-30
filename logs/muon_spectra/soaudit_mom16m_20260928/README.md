# Momentum at large batch: β sweep at 16M (second-order audit, 2026-09-28)

Plain S∘PD at β 0.8 ran −0.05 to −0.11 ahead of its β 0.9 reference mid-run (`../soaudit_prefilter16m_20260928`).
β 0.9 came from the 4M sweep and was never tuned at 16M. Mechanism: at large batch the EMA's noise averaging matters
little, its lag of ~β/(1−β) steps is a large share of a 92-step run, and it must pass the period-2 stabilizing
feedback, attenuated by (1−β)/(1+β) (two-tap results). Decision argument and predictions:
`research/adamw_spectra/MUON_CASE.md`, "Decision argument: the momentum at large batch" (12:10 CDT).

- **Arms.** 16M, seed 260925, 92 steps, hardware matched to `../soaudit_batch16m_20260927`:
  - g20 (Ada): S∘PD α ½ @0.028 at β 0.7, then β 0.6 (β 0.9: 4.5742; β 0.8 in the prefilter cohort).
  - priv-g14 (L40S): PD α ½ @0.028 at β 0.8 (β 0.9: 4.6711), then Muon @0.02 at β 0.8 (β 0.9: 4.9104).
  - Each queue waits for the node's running arm.
- **Code.** Frozen from the live source, which includes the prefilter and head-whitening options; both are off here.

## Results (13:23 CDT)
| Method | β 0.9 | β 0.8 | β 0.7 | β 0.6 |
|---|---|---|---|---|
| S∘PD α ½ @0.028 (Ada) | 4.5742 | 4.4724 | **4.4691** | 4.4891 |
| PD α ½ @0.028 (L40S) | 4.6711 | **4.5475** | — | — |
| Muon @0.02 (L40S) | 4.9104 | **4.8901** | — | — |

The whitening methods gain 0.10–0.12 from a fresher momentum, while Muon gains 0.02. S∘PD's step-equivalent speedup over
the best-tuned Muon is now 1.30–1.41×. Interpretation: `research/adamw_spectra/MUON_CASE.md`, 13:23 CDT.

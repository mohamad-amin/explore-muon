# Adam in input-whitened coordinates for the unembedding (second-order audit, 2026-09-28)

The head's input (the final normalized hidden state) is as anisotropic as the body inputs PD whitens: top eigenvalue
80–120× the mean, participation ratio 0.03–0.06 (`../second_order_audit_20260926/head_input_probe.json`). The head is
trained by AdamW everywhere, which cannot decorrelate inputs. `head_whitening_alpha` gives the head the PD treatment:
Adam in coordinates where its input is whitened. The step is dW = −Adam(G R) R, with R = (C_h/mean + 1e−3)^−α from the
head's StatLinear (`model.track_head_cov`). The first moment stays in weight space, the second moment is of G R, and
the update is norm-matched, with decoupled decay; rank 0 computes it and broadcasts the head. Decision argument:
`research/adamw_spectra/MUON_CASE.md`, "Decision argument: PD's principle for the unembedding" (11:16 CDT).

- **Arms.** 16M, seed 260925, 92 steps, hardware matched to `../soaudit_batch16m_20260927`:
  - priv-g14 (L40S): PD α ½ @0.028 with a whitened head at α ½, then α ¼ (reference 4.6711).
  - g20 (Ada): S∘PD α ½ @0.028 with a whitened head at α ½, then α ¼ (reference 4.5742).
  - Each queue waits for the node's last two-tap arm (`../soaudit_prefilter16m_20260928`).
- **Code.** Frozen from the live source after adding `track_head_cov` (model) and `head_whitening_alpha` (optimizer),
  both off by default: weights and the initial-model hash are unchanged. A unit test checks the step against the formula,
  and all 86 tests pass.

- **Re-created 11:37 CDT** from the fixed source after the replica failure kept in `../soaudit_headwhite16m_20260928_failed_replica`.
- **g20 queue stopped 12:10 CDT before it started** (S∘PD α ½ and α ¼ never ran): PD α ½ with the whitened head ended +0.295 (4.9656 vs 4.6711), with a +0.93 deficit in steps 5–10. The S∘PD arms would re-test a failed design; g20 went to the momentum follow-up instead.

# B_track1_restore_g20: failed in qualification (2026-09-25 ~10:40 UTC)

`qualification.log`: at the first measured update (step 1), `measure.spectrum` raised
"Non-finite update matrix". The other ranks then hit an NCCL all-gather timeout.

**Likely cause** (verified in part, 2026-09-25 ~10:55 UTC):
- The tracked-restore entry is `rest/‖rest‖_F + (1/1.01)·û v̂ᵀ`. Its spectral norm is not bounded by 1.
  - With the exact head it is about 1.01; a synthetic spike case gives 1.012.
  - If the one-step, randomly warm-started power iteration has not found the true top pair, the normalized remainder still carries much of the spike near the restored head. The top singular value can then reach up to about 2.
- The paper's five-polynomial NS map diverges for singular values ≳ 1.33. Iterating the scalar map gives:
  - σ = 1.30 → 0.98;
  - σ = 1.35 → 3e143.
- The drop variant (C, head weight 0) has no restored head. Its entry is bounded by 1, and it ran to completion.
- The failing matrix was not identified.
- Unit tests used the exact head vector, so they did not exercise this.

**Not re-run.** B was the control for NS conditioning. The wave-4 exact-polar arm (`../../improve_w4_20260925/X_svd_lr0.01_s260925_a6000`) answers that question directly (MUON_CASE.md, wave 4). A fixed restore mode would need a bound on the entry's spectral norm, such as dividing by an estimate of its top singular value.

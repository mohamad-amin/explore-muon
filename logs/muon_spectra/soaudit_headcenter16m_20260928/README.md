# Centered head whitening at 16M (second-order audit, 2026-09-28)

Whitening the head's uncentered input second moment suppresses the head input's dominant direction, its mean (the
output-bias channel). This cost PD +0.93 early at α ½ and +0.72 at α ¼. After that phase α ¼ gained, ending −0.040
(`../soaudit_headwhite16m_20260928`). Here `head_whitening_center` whitens the covariance about the EMA mean instead,
as `data_norm_center` does for the body, so the mean direction is not singled out. Note and prediction:
`research/adamw_spectra/MUON_CASE.md`, "Whitened head, PD at 16M" (12:25 CDT): centered α ½ ≤ 4.64.

- **Arms.** PD α ½ @0.028 β 0.9, 16M, seed 260925, 92 steps, with the head in centered whitened coordinates at α ½ and
  α ¼. They run on the cluster's L40S (g21) through `job.sbatch`, hardware matched to the PD reference 4.6711 (L40S).
- **Code.** Frozen from the live source after adding `head_whitening_center`, with a unit test; 86 tests pass.
- **12:37 CDT:** the cluster jobs (2626491, 2626492) stayed pending (the L40S node was reserved) and were cancelled before starting. Both arms run on priv-g14 (L40S) after the momentum sweep's Muon arm.
- **13:24 CDT, α ¼ arm replaced before it started.** The step was norm-matched in weight space (|dW| set to the norm of
  the whitened-coordinate Adam step). Whitening amplifies low-variance input directions by up to (1e-3)^−α, so the
  matching shrinks the head's function-space step. That confounds the test with a smaller effective head LR, which is
  likely part of the early deficits. The principled design is pure Adam in the whitened coordinates (no rescaling,
  `head_whitening_norm: none`), in `../soaudit_headcenter2_16m_20260928`. The queue stops after the α ½ arm: its α ¼
  arm's controller.log already exists.

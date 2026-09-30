# PD-top vs PD α ½ at 1M: is the tail amplification a large-batch lever? (second-order audit, 2026-09-28)

At 16M, PD's amplification of the weak input directions adds 11–23% of its gain over Muon (PD-top 4.7267 vs PD 4.6711
at β 0.9). At 1M, stronger whitening loses (A6000 bracket: α ½ 3.7125 vs α ⅛ 3.6955). Does the tail amplification,
which multiplies the step in noise-dominated weak directions by up to ~32×, cause the 1M loss? Decision argument:
`research/adamw_spectra/MUON_CASE.md`, "Decision argument: does PD's tail amplification pay only at large batch?"
(20:54 CDT).

- **Arms.** 1M, 1468 steps, seed 260925, β 0.95, LR 0.01, α ½, damping 1e-3. PD-top (`data_norm_top_only`) vs PD, the
  same pair on both nodes:
  - g20 (Ada): PD-top first, then PD. Context: PD α ¼ @0.01 3.6869 (Ada, soaudit_traj_20260926).
  - priv-g14 (L40S): PD first, then PD-top. Context: Muon @0.007 3.7047 (L40S, soaudit_traj_20260926).
  - Each queue waits for its node's 4M momentum-warmup arm (soaudit_warm4m_20260928).
- **Predictions.**
  1. PD-top α ½ beats PD α ½ by ≥ 0.005 on both nodes.
  2. PD-top α ½ is within 0.01 of PD α ¼ at 1M.
- **Source.** Frozen copy of `research/adamw_spectra` (muon.py sha256 3452e244…, identical to soaudit_pdtop_20260928).
  Checkpoints kept at 200 and 900 only (shared disk at 96%).

## Results (2026-09-28 22:08 CDT)

- Ada: PD-top α ½ 3.6992, PD α ½ 3.7104 (−0.0112; PD-top ahead from step 300 on).
- L40S: PD α ½ 3.7100; PD-top α ½ running.
- Value of tail amplification (PD − PD-top): 16M −0.056, 4M −0.007, 1M +0.011. The sign flips with batch size.
- L40S (22:25 CDT): PD-top α ½ 3.7002, PD α ½ 3.7100 (−0.0098). Replicates the Ada pair.

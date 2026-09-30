# Two-sided data norm, paired 1M-batch test (second-order audit, 2026-09-27)

Rule and motivation: `research/adamw_spectra/MUON_CASE.md`, "Two-sided data norm implemented" (02:13 CDT).
D = L polar(L M R) R, with R = C^-α and L = (B / mean eig + 1e-3 I)^-β.

- **Arms.** Seed 260925, batch 1M, 1.54B tokens, LR 0.01, 4× RTX A4000 each (cluster jobs 2624897 and 2624898):
  - `PD_a0.25_lr0.01_s260925_a4000`: the control, PD α ¼. Same config as the trajectory cohort's PD arm, except for the kept checkpoints.
  - `TS_a0.25_b0.25gn_lr0.01_s260925_a4000`: plus the output factor, β ¼, with GN (model-sampled) labels. Refresh every 10 steps on 8 local sequences, EMA 0.8.
- **Why β ¼ with GN labels.** In the one-step test (`../second_order_audit_20260926/one_step_twosided_*.json`), the output factor adds +3% on PD's step-500 state and +6–10% on Muon's. GN labels are slightly ahead of data labels.
- **Reading.** Final validation loss, TS − PD on the same hardware and seed. One LR only; an LR bracket follows only if the difference is at least ~0.005.

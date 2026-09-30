# SNR-gated tail amplification for PD: 1M pilot on A6000 (second-order audit, 2026-09-28)

Decision argument: `research/adamw_spectra/MUON_CASE.md`, "Decision argument: SNR-gated tail amplification for PD, a
Wiener gate per input direction" (22:26 CDT). PD's root factors f = (u + δ)^-α become min(1, f) + max(0, f − 1)·w per
input direction, w = SNR/(1+SNR) of the direction's log-u bin; SNR from the across-rank noise of the gradient projected
on the (rank-averaged) input eigenbasis, EMA 0.9. w = 0 is PD-top, w = 1 is PD.

- **Arms** (1M, 1468 steps, LR 0.01, β 0.95, seed 260925, 4× A6000):
  - `PDgate_a0.5_lr0.01_s260925_a6000`: gate at α ½;
  - `PDtop_a0.5_lr0.01_s260925_a6000`: PD-top at α ½ (completes the A6000 trio; PD α ½ on A6000 3.7131,
    `soaudit_alpha_20260927`-era run of the same configuration);
  - `PDgate_a0.25_lr0.01_s260925_a6000`: gate at α ¼ (vs PD α ¼ 3.6870, `soaudit_warm1m_20260928` control, and PD-top α ¼,
    `soaudit_pdtop1m_a025_20260928`).
- **Predictions** (22:26): at 1M the gate ≈ PD-top (within 0.003) and beats PD at the same α; if the α ½ gate lands near PD
  α ½ (> 3.707 on A6000-equivalent), the SNR explanation of the sign flip fails.
- **Source.** Frozen `research/adamw_spectra` (muon.py f266c69e…, distributed.py ab812539…); the gate is new code, first
  exercised by the full-size qualification (5 steps, replica audit) of these arms.

## Results

- **Gate α ½ (23:37 CDT): 3.6955**, vs PD α ½ A6000 3.7131 (−0.0176). Ahead at every validation step from 100. Tail w
  falls 0.98 → 0.31 as the tail SNR falls from 79 to 0.37 over the first ~700 steps.
- PD-top α ½ and gate α ¼: queued.
- **PD-top α ½ (00:31 CDT): 3.7008** (−0.0123 vs PD α ½). Gate − PD-top = −0.0053; ahead of PD-top at every validation
  step from 400 on. The gate beats both fixed choices on the same hardware.
- **Gate α ¼ (00:50 CDT): 3.6907**, vs PD α ¼ 3.6870 (+0.0037) and PD-top α ¼ 3.7008 (−0.0101). The prediction fails at
  α ¼: the Wiener weight cuts α ¼'s useful amplification.

# PD-top α ¼ at 1M: does removing the tail amplification help at the best 1M power too? (second-order audit, 2026-09-28)

At 1M, PD-top α ½ beat PD α ½ by 0.011 (Ada; `soaudit_pdtop1m_20260928`), while at 4M and 16M the tail amplification
helps (+0.007, +0.056). The best 1M configuration is PD α ¼ @0.01 β 0.95. MUON_CASE.md, "PD-top vs PD at 1M: the tail
amplification's value flips sign with batch size" (22:08 CDT), "Next, cheap".

- Arm: `PDtop_a0.25_lr0.01_mom0.95_s260925_a6000` (A6000, seed 260925, 1468 steps).
- Control: `soaudit_warm1m_20260928/PD_a0.25_lr0.01_mom0.95_s260925_a6000` (same config without top_only, same GPU type,
  run this evening); earlier A6000 run of that config 3.6876.
- Prediction: PD-top α ¼ ≤ PD α ¼ − 0.003 (the tail is noise-limited at 1M for α ¼ too). If PD-top α ¼ is worse, the 1M
  tail at α ¼ (max amplification (1e-3)^-¼ ≈ 5.6×) still carries net signal and the α ½ result is about the strength of
  the amplification (up to 32×).
- Source: frozen `research/adamw_spectra` at 22:09 CDT (muon.py 5d816a1f…; defaults unchanged).

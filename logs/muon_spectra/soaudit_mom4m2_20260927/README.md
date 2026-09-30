# Momentum 0.81 combined with the preconditioners at batch 4M (second-order audit, 2026-09-27)

Muon with β 0.81 (the same momentum half-life in tokens as β 0.95 at 1M) beat β 0.95 by 0.0087 at 4M
(`../soaudit_mom4m_20260927`). Two-sided beat PD by 0.016 at 4M (`../soaudit_ts4m_20260927`). This cohort tests whether
the shorter momentum stacks with the input and output preconditioners. Each arm changes one factor against an
existing 4M baseline on the same GPU type.

- **g20 (Ada)**, after the TS 4M bracket:
  - TS @0.02 with β 0.81; baseline TS @0.02, β 0.95: 3.8688.
  - PD @0.02 with β 0.81; baseline 3.8847.
- **priv-g14 (L40S)**, after the momentum sweep:
  - Muon @0.02 with β 0.81, the LR bracket for β 0.81; @0.014 gave 3.9437.
  - S∘PD @0.01 with β 0.81; baseline on L40S 3.8454.
- **Protocol.** `research/adamw_spectra/MUON_CASE.md`, "4M results: shorter momentum helps Muon, and the two-sided gain grows with batch" (06:18 CDT).

# SNR-gated PD at 4M and 16M: does the gate match PD where PD's tail amplification pays? (second-order audit, 2026-09-28)

At 1M the gate beat PD α ½ by 0.018 (`soaudit_snrgate1m_20260928`). At large batch the tail's SNR is high, so the gate
should keep w ≈ 1 and reproduce PD (which beats PD-top by 0.056 at 16M and 0.007 at 4M). Decision argument: MUON_CASE.md
(22:26 CDT); predictions: gate ≈ PD within 0.003 at 16M, and at 4M it beats both PD and PD-top by ≥ 0.003.

- **priv-g14 (L40S), after `soaudit_mlaw_20260928`'s queue:**
  - `PDgate_a0.5_b16M_lr0.028_mom0.9_s260925_l40s` (references: PD β 0.9 4.6711, PD-top β 0.9 4.7267);
  - `PDgate_a0.5_b16M_lr0.028_mom0.8_s260925_l40s` (references: PD β 0.8 4.5475, PD-top β 0.8 4.6003).
- **g20 (Ada), after `soaudit_mlaw_20260928`'s queue:**
  - `PDgate_a0.5_b4M_lr0.02_mom0.9_s260925_ada` (references: PD 3.8589, PD-top 3.8662).
- **Source.** Frozen `research/adamw_spectra` (muon.py f266c69e…, distributed.py ab812539…), the same gate code as the 1M pilot.

## Results

- **16M β 0.9 (00:51 CDT): 4.6717**, vs PD 4.6711 and PD-top 4.7267. The gate keeps w ≥ 0.95 (tail SNR 5000 → 23), so it
  matches PD as predicted.
- **4M β 0.9 (01:16 CDT): 3.8560**, vs PD 3.8589 (−0.0029) and PD-top 3.8662 (−0.0102). Tail w falls to 0.7 mid-run
  and 0.4 in the cooldown.
- **16M β 0.8 (01:42 CDT): 4.5505**, vs PD 4.5475 (+0.0030) and PD-top 4.6003 (−0.050). The gate tracks PD at large
  batch, within the predicted 0.003. The pre-registered gate checks are complete. Per the user's 01:19 steering, no
  further recipe variants follow.

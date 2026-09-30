# Top-only whitening, PD-top (second-order audit, 2026-09-28)

PD-top clamps PD's input root at 1: it suppresses only the above-mean input directions and keeps the bulk isotropic, as
in Muon (no tail amplification). It asks which part of PD's geometry carries its gain and its α×β interaction with the
momentum. Decision argument and predictions: `research/adamw_spectra/MUON_CASE.md`, "Decision argument: top-only
whitening" (19:58 CDT).

- **Arms.**
  - priv-g14 (L40S): PD-top α ½ @0.028, 16M, seed 260925, β 0.8, then β 0.9. They start after the no-training
    pre-flight in `../second_order_audit_20260926/step_profile_pdtop/` passes.
  - g20 (Ada): PD-top α ½ @0.02, 4M, β 0.9, against PD α ½ @0.02 3.8589 (Ada).
- **Code.** `data_norm_top_only` (muon.py `data_norm_root`), with a unit test; 88 tests pass.

## Results (20:53 CDT)
| Arm | Final |
|---|---|
| PD-top 16M β 0.9 | 4.7267 |
| PD-top 16M β 0.8 | 4.6003 (−0.126; lead 6.5 steps at step 75, = PD's) |
| PD-top 4M β 0.9 | 3.8662 (PD 3.8589) |

The top suppression carries the momentum interaction in full and 77–89% of PD's gain. Details: MUON_CASE, 20:53 CDT.

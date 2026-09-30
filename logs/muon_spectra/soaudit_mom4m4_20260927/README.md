# Stronger preconditioning with fresher momentum at batch 4M (second-order audit, 2026-09-27)

In the one-step test with fresh-plus-stale inputs g + cM (`../second_order_audit_20260926/gnmix/`), the best input power
rose from ¼ (c = 0.95, the momentum) to ½–¾ (c ≤ 0.25). In training at β 0.95, α ½ lost to α ¼ at both 1M and 4M.
Principle under test: stronger curvature preconditioning pays only when the averaging is fresh enough.

- **Arms** (g20 Ada, after the clip tests): PD α ½ @0.02 with β 0.9, paired with PD α ¼ @0.02 with β 0.9 in `../soaudit_mom4m3_20260927`; TS with β_out ½ @0.01 and momentum 0.9, paired with TS β_out ¼ @0.01 and momentum 0.9 in the same cohort.
- **Prediction.** If the principle holds, α ½ (or β_out ½) does not lose to the ¼ settings at momentum 0.9, as it did at momentum 0.95 (by +0.007 for PD at 4M).

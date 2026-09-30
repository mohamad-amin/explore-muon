# Does the best input power grow with batch? (second-order audit, 2026-09-27)

Decision argument and prediction: `research/adamw_spectra/MUON_CASE.md`, "Batch-size result, the exact-GN reference,
and a batch-dependent power test" (03:44 CDT).

- **Setup.** Batch 4,194,304 tokens, 1.54B tokens (368 steps), seed 260925, warmup 13, cooldown 0.1, kept checkpoints 37/183/330. Same as `../soaudit_batch4m_20260927` except the arm changes below.
- **Arms.**
  - g20, RTX 6000 Ada: PD α ½ at LR 0.02 and 0.04. Paired with PD α ¼ in `../soaudit_batch4m_20260927`: 3.8854 / 3.8847 / 3.8959 at LR 0.01 / 0.02 / 0.04.
  - priv-g14, L40S: S∘PD α ½ at LR 0.01 and 0.02, plus S∘PD α ¼ @0.01 as a hardware control. The α ¼ bracket in `../soaudit_batch4m_20260927` ran on A6000.
- **Code.** Frozen from the live source (`frozen/`). The only change against the batch-4M cohort's frozen code is the two-sided output factor, which is inactive at `data_norm_out_beta` = 0.
- **Launch.** `wait_gn_then_queue.sh` waits for the audit's `one_step_gn.py` measurements on each node to finish, then runs `node_queue.py` (queues `a4m_g20`, `a4m_privg14`).
- **Prediction.** At 4M, best-LR α ½ beats best-LR α ¼ by ≥ 0.005 for both PD and S∘PD.

Note (04:30 CDT): the first priv-g14 launcher (wait_gn_then_queue.sh) never started its queue: its pgrep matched the
launching shell's own command line (priv-g14 is also the interactive node). It was stopped and replaced by
start_privg14.sh, which runs the same queue directly after the GN measurements had finished.

Aborted baseline attempts (2026-09-25 ~22:15 UTC), kept for the record; not results.
- Job 2622198 (4xL40S, g21): cancelled at 22:14:18 UTC by mistake, seconds after it had started. It was
  believed to be still pending, because the node had been reported as held for a higher-priority partition.
- Job 2622200 (4xA6000, g19, Track 3 log 39614e64-...): started as the fallback and cancelled after ~40 s,
  once the resubmitted L40S job (2622202) was running.
The baseline control is job 2622202 on 4xL40S (g21).

## #36 at 1M batch (`train_gpt_simple_bt1M.py`), priv-g14, killed externally (2026-09-26 00:46:49 CDT)

- Slurm step 2567578.402 (started 00:12:15 CDT) was cancelled at 00:46:49 (signal 9) at about step 1000 of 1625. Last validation: step:1000/1625 val_loss:3.45931.
- 16 s earlier, interactive `zsh` steps holding all 4 GPUs had started on both allocations: 2567578.403 on priv-g14 and 2618555.72 on g20.
- Not caused by this session. The tmux session running `run_on.sh` died too, so `queue.log` has no end line.
- Files: `9a427aa8-eae7-4cd0-82f8-9337c717cca3.txt` and its console log, moved here.
- Pair: `train_gpt_pd_bt1M.py` completed on g20 (3.30041).
- The val losses up to step 1000 are kept in this log for the mid-run comparison.

## α ⅛ + geometry decay, 3250 steps (`train_gpt_pd_pdwd_a0.125.py`), g20: stopped by user direction (2026-09-26 20:14 CDT)

- The user switched the step-count effort back to the compressed 3150-step schedule ("add one more compressed 3150 run on priv-g14; stop the 3250 step runs").
- Slurm step 2618555.88 was cancelled at about step 500 (last validation: step:500/3250 val_loss:3.80203).
- In the same instruction, the three queued 3250 cluster jobs (2624732–2624734) were cancelled before starting.
- Files: `3506f7ea-f7ff-4d1e-8d99-f58341028aac.txt` and its console log, moved here.

## Chained second 3250-step run on priv-g14 (2026-09-26 20:16 CDT): cancelled at startup by user direction

- It was chained after the priv-g14 3250-step run (`62a8c871`, finished normally).
- Slurm step 2567578.421 was cancelled one second after it started, before training or a Track 3 logfile.
- The compressed 3150-step run then started on priv-g14 (step 2567578.422).
- Only its console log exists; it is moved here.

## 2026-09-27 23:49 CDT: two legal-TS runs stopped on user request

- **User direction:** "Sorry don't use g20 and priv-g14". I had started one run of each legal-TS arm on idle GPUs of those allocations at 23:38 CDT, because the cluster queue projected starts ~18 h later.
- **Stopped** by cancelling only my own steps, 2567578.511 and 2618555.183. The other study's steps on both allocations were untouched.
  - `train_gpt_pd_pdwd_a0.125_s3150_tsef_b0.125_geo2.py`: priv-g14, 2× L40S, training log `../944d2527-5e40-4f6e-b85c-4db94a031a5a.txt`, stopped at step 166.
  - `train_gpt_pd_pdwd_a0.125_s3150_tsef_b0.125.py`: g20, 2× RTX 6000 Ada, training log `../fc46980a-f949-4f88-a5be-2927006f1d10.txt`, stopped at step 129.
- **Not counted.** The analysis scripts pool only finished runs. The replacements are gpu-partition jobs 2626222 (geo2) and 2626223, alongside the queued 2626218 and 2626219.
- Moved into this folder on 2026-09-28 05:20 CDT, with their console logs, so the pooling scripts no longer list them. The `../` paths above now resolve to this folder.

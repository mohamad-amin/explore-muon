# Original experiment compute accounting

The tracked original training job intervals account for **946.55 GPU-hours**: 942.87 from Slurm and 3.68 from four cloud pipelines. Qualification, failures, cancellations, timeouts and within-job overhead are included. This is scoped accounting, not a complete project invoice.

The second-order/batch/momentum training cohorts (`soaudit_*`) account for **351.41 GPU-hours**. Among 233 completed standard 76,948,992-parameter / 1,539,870,720-token pipelines, the median is **3.15 GPU-hours per run**. All 233 use four GPUs; hardware and optimizer variants differ.

Representative matched 4M-token-batch runs, seed 260926, on four L40S GPUs. Wall time includes pipeline qualification and overhead:

| Method | Wall minutes | GPUs | GPU-hours |
|---|---:|---:|---:|
| M | 23.63 | 4 | 1.575 |
| PD | 25.55 | 4 | 1.703 |
| SPD | 48.37 | 4 | 3.225 |
| TS | 25.79 | 4 | 1.719 |

The preliminary raw step sum was 1,028.77 GPU-hours. Some selected steps overlapped on the same four-GPU allocation; summing them counted 82.22 hours twice. The corrected result merges their time intervals before multiplying by four. Allocation parents are never added to child steps. Cloud overlap was checked by physical GPU UUID.

Separate diagnostic/probe jobs and idle allocation time outside the selected intervals are excluded. The wider campaign also includes approximately 90M and 134M variants. GPU-hours are not normalized for hardware speed.

Raw evidence: `pipeline_inventory.json`, `scheduler_records.psv`, `scheduler_query.json`, `scheduler_summary_rows.json`, `allocation_interval_union.json`, `cloud_identity_check.json` and `SUMMARY.json`. The preliminary calculation remains recorded separately. All audit files are inside `tiny_surrogate/`; original experiments were read only.

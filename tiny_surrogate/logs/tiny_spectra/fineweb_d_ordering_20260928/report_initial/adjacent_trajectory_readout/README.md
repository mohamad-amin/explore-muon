# Initial-grid adjacent gaps over the full recorded trajectory

This descriptive readout uses the endpoint-selected joint rates from the complete initial grid and retains every recorded development point. No model inference or test scoring is performed. Qualification continues to use the fixed endpoint at step368.

For SOAP-PD versus TS, the mean difference is−0.0129813 at step328, the last scheduled observation before cooldown starts after step331. At the fixed endpoint, step368, it is−0.00292184. The two seed differences are−0.0106154/−0.0153471 before cooldown and−0.00374212/−0.00210156 at the endpoint. The advantage changes through training; this observation alone does not identify the cause.

`all_recorded_adjacent_gaps.csv` contains all48recorded points for each of the four adjacent method comparisons, with both seeds and their mean. `README.json` binds the source report and records the schedule-defined before/after summaries. The initial learning-rate brackets remain provisional pending the six declared edge checks.

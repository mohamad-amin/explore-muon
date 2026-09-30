# Single-run multi-GPU execution qualification

Qualification passed: **False**. All checks use16GB A4000s on one node.

| Method | Native/world1 exact | 2GPU training speedup | 4GPU training speedup | 2GPU wall speedup | 4GPU wall speedup |
|---|---|---:|---:|---:|---:|
| adamw | False | 2.97x | 5.81x | 2.05x | 2.76x |
| muon | False | 1.88x | 3.65x | 1.46x | 1.91x |
| pd | False | 1.62x | 3.07x | 1.43x | 2.12x |
| ts | False | 1.66x | 2.90x | 1.44x | 2.04x |
| soap | False | 1.68x | 2.96x | 1.39x | 1.78x |
| spd | False | 1.58x | 2.81x | 1.37x | 1.98x |
| sts | False | 1.59x | 2.88x | 1.42x | 2.05x |

Same-state checks and short trajectories only. FP32 association changes; use the same GPU count within scientific comparisons. Root optimizer/evaluation remain serial.

Complete per-tensor same-state metrics, accumulated parameter/loss drift, GN hashes, full-state inventories, partial-batch checks and active-worker GPU-hours are retained in results.json. Allocation cost for the benchmark includes all four reserved GPUs even during serial reference runs. No independent test panel was scored.

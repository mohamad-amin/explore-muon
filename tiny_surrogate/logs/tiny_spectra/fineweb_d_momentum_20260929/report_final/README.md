# Momentum and horizon-control results

All 44 new runs completed successfully; 12 completed base runs were reused. Both methods pass the prospective fixed-LR momentum-interaction criterion. This is development evidence; the full surrogate remains unqualified.

| Method | Small-batch gain from beta 0.8 | Large-batch gain | Interaction |
|---|---:|---:|---:|
| muon | 0.000950 | 0.041436 | 0.040486 |
| spd | 0.009200 | 0.039146 | 0.029946 |

Gains are NLL(beta 0.9) minus NLL(beta 0.8), averaged over both fixed rates and both seeds. Positive favors shorter momentum. The large-batch benefit and its increase over small batch have the required signs at every rate and seed, and each seed’s mean effects exceed 0.005. A consistent reversal of momentum preference at small versus large batch is not established; two beta values do not identify a global optimum.

Matched128-update interactions change sign across seeds for both methods. They do not reproduce a consistent batch interaction at matched update count. At the common control LRs, SoapMuon+PD’s large-batch benefit falls from 0.049389 at the base horizon to 0.021542 at twice the horizon; Muon changes from 0.031365 to 0.042197, with mixed seedwise attenuation. These are fixed-recipe phase controls, not isolated batch causality or independently retuned horizon optima.

Allocation cost: 31.6628 GPU-hours for the 44 new runs, all on 16 GB A4000s. Both automatic CPU stages completed successfully. All three independent panels remain sealed.

The prior robust five-method ordering failure and unresolved common-loss batch-growth criterion are unchanged. No further experiment follows automatically. Every fixed-rate contrast, joint-LR secondary comparison, document group and trajectory is preserved in the base and final results.json files.

# adamw: old architecture vs frontier-norm variant (depth 8, one seed each)

Final validation NLL: old 3.87925, variant 3.86532 (difference -0.01393 nats/token).

AdamW update spectra, mean over steps 1300–1469; readings from the spike diagnostic.

| depth | kind | ρ1 old → new | median old → new | reading old → new | mean share old → new | pos0 share old → new |
|---|---|---|---|---|---|---|
| early | q | 0.11 → 0.09 | 8.30e-03 → 8.15e-03 | distributed → distributed | 0.21 → 0.01 | 0.00 → 0.00 |
| early | k | 0.12 → 0.07 | 8.90e-03 → 8.98e-03 | distributed → not reproduced | -0.00 → 0.20 | 0.26 → 0.76 |
| early | v | 0.15 → 0.19 | 7.61e-03 → 7.55e-03 | mean → mean | 0.74 → 0.88 | 0.16 → 0.11 |
| early | o | 0.08 → 0.08 | 1.23e-02 → 1.34e-02 | distributed → mean | 0.35 → 0.85 | 0.07 → 0.03 |
| early | up | 0.08 → 0.07 | 1.66e-02 → 1.72e-02 | mean → mean | 0.99 → 1.00 | 0.04 → 0.02 |
| early | down | 0.08 → 0.07 | 3.19e-02 → 3.03e-02 | mean → mean | 0.94 → 0.53 | 0.01 → 0.46 |
| mid | q | 0.10 → 0.07 | 1.36e-02 → 1.49e-02 | distributed → not reproduced | 0.01 → -0.87 | -0.00 → -0.00 |
| mid | k | 0.10 → 0.06 | 1.14e-02 → 1.43e-02 | distributed → distributed | 0.00 → -0.09 | 0.19 → 0.06 |
| mid | v | 0.11 → 0.14 | 1.02e-02 → 1.14e-02 | mean → mean | 0.96 → 0.81 | 0.03 → 0.16 |
| mid | o | 0.08 → 0.07 | 1.39e-02 → 1.64e-02 | distributed → mean | 0.21 → 0.74 | 0.08 → 0.05 |
| mid | up | 0.07 → 0.06 | 2.53e-02 → 2.65e-02 | mean → mean | 0.97 → 0.97 | 0.01 → 0.01 |
| mid | down | 0.10 → 0.05 | 3.02e-02 → 3.14e-02 | distributed → sink | 0.00 → 0.31 | 0.00 → 0.61 |
| late | q | 0.06 → 0.06 | 1.73e-02 → 1.94e-02 | mean → not reproduced | 0.94 → 0.84 | -0.00 → 0.00 |
| late | k | 0.06 → 0.05 | 1.40e-02 → 1.77e-02 | distributed → not reproduced | 0.00 → 0.01 | -0.00 → -0.12 |
| late | v | 0.07 → 0.12 | 1.57e-02 → 1.44e-02 | mean → mean | 0.90 → 0.72 | 0.03 → 0.29 |
| late | o | 0.05 → 0.06 | 1.82e-02 → 1.87e-02 | distributed → not reproduced | -0.07 → 0.30 | 0.07 → 0.04 |
| late | up | 0.05 → 0.05 | 2.82e-02 → 3.03e-02 | mean → mean | 0.93 → 0.95 | 0.03 → 0.01 |
| late | down | 0.06 → 0.03 | 2.90e-02 → 3.10e-02 | sink → distributed | 0.02 → 0.48 | 0.78 → 0.03 |
| final | q | 0.05 → 0.06 | 1.74e-02 → 2.04e-02 | mean → not reproduced | 0.90 → 0.76 | 0.00 → -0.00 |
| final | k | 0.06 → 0.05 | 1.71e-02 → 2.05e-02 | not reproduced → not reproduced | -0.00 → -0.01 | -0.12 → -0.01 |
| final | v | 0.10 → 0.13 | 1.37e-02 → 1.44e-02 | mean → mean | 0.70 → 0.72 | 0.26 → 0.28 |
| final | o | 0.07 → 0.08 | 1.59e-02 → 1.69e-02 | not reproduced → mean | 0.59 → 0.82 | 0.00 → 0.00 |
| final | up | 0.05 → 0.06 | 2.96e-02 → 2.96e-02 | mean → mean | 0.86 → 0.83 | 0.03 → 0.00 |
| final | down | 0.18 → 0.18 | 2.37e-02 → 2.45e-02 | distributed → distributed | -0.01 → -0.05 | 0.30 → 0.06 |

old: readings {'not reproduced': 6, 'distributed': 17, 'mean': 22, 'sink': 3}; cross-fitted/in-sample medians mode1 0.79, modes2-5 0.62, to_q0.1 0.38, q0.1-q0.5 0.22, q0.5-q0.9 0.15, below_q0.9 0.14

variant: readings {'distributed': 7, 'mean': 27, 'not reproduced': 9, 'sink': 5}; cross-fitted/in-sample medians mode1 0.92, modes2-5 0.58, to_q0.1 0.37, q0.1-q0.5 0.20, q0.5-q0.9 0.15, below_q0.9 0.13

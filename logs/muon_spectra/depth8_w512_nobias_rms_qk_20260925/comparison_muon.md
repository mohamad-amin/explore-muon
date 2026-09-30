# muon: old architecture vs frontier-norm variant (depth 8, one seed each)

Final validation NLL: old 3.72099, variant 3.71310 (difference -0.00789 nats/token).

momentum spectra, mean over steps 1300–1469; readings from the spike diagnostic.

| depth | kind | ρ1 old → new | median old → new | reading old → new | mean share old → new | pos0 share old → new |
|---|---|---|---|---|---|---|
| early | q | 0.44 → 0.42 | 5.18e-03 → 8.41e-03 | distributed → mean | 0.30 → 1.01 | -0.00 → -0.00 |
| early | k | 0.36 → 0.29 | 5.79e-03 → 8.23e-03 | sink → not reproduced | 0.00 → 2.65 | 0.62 → -1.50 |
| early | v | 0.87 → 0.74 | 2.80e-03 → 4.48e-03 | mean → mean | 0.98 → 0.94 | 0.04 → 0.22 |
| early | o | 0.40 → 0.31 | 9.68e-03 → 1.25e-02 | distributed → mean | 0.41 → 0.98 | -0.01 → 0.02 |
| early | up | 0.72 → 0.53 | 9.94e-03 → 1.39e-02 | mean → mean | 0.99 → 1.01 | -0.01 → 0.01 |
| early | down | 0.28 → 0.20 | 2.20e-02 → 2.49e-02 | sink → sink | 0.00 → 0.05 | 0.85 → 0.75 |
| mid | q | 0.63 → 0.44 | 1.05e-02 → 1.20e-02 | mean → mean | 0.99 → 0.97 | -0.00 → 0.00 |
| mid | k | 0.18 → 0.23 | 1.16e-02 → 1.08e-02 | distributed → sink | 0.00 → 0.00 | 0.06 → 0.75 |
| mid | v | 0.84 → 0.73 | 3.86e-03 → 4.98e-03 | mean → mean | 0.95 → 0.90 | 0.06 → 0.24 |
| mid | o | 0.23 → 0.13 | 1.54e-02 → 1.71e-02 | mean → mean | 0.92 → 0.95 | 0.05 → 0.03 |
| mid | up | 0.68 → 0.46 | 1.37e-02 → 1.90e-02 | mean → mean | 0.99 → 1.00 | 0.01 → 0.00 |
| mid | down | 0.14 → 0.13 | 2.55e-02 → 2.67e-02 | sink → mean | -0.01 → 0.89 | 0.75 → -0.01 |
| late | q | 0.53 → 0.35 | 1.23e-02 → 1.40e-02 | mean → mean | 0.97 → 0.96 | -0.00 → -0.00 |
| late | k | 0.13 → 0.21 | 1.51e-02 → 1.31e-02 | distributed → sink | -0.00 → -0.03 | 0.07 → 1.00 |
| late | v | 0.83 → 0.69 | 4.89e-03 → 6.15e-03 | mean → mean | 0.89 → 0.70 | 0.15 → 0.40 |
| late | o | 0.15 → 0.14 | 1.79e-02 → 1.70e-02 | mean → mean | 0.94 → 0.95 | 0.10 → 0.08 |
| late | up | 0.50 → 0.29 | 1.90e-02 → 2.31e-02 | mean → mean | 0.98 → 1.02 | 0.07 → 0.02 |
| late | down | 0.15 → 0.16 | 2.59e-02 → 2.60e-02 | mean → mean | 1.08 → 0.99 | -0.02 → -0.01 |
| final | q | 0.38 → 0.31 | 1.43e-02 → 1.36e-02 | not reproduced → mean | 1.04 → 0.86 | 0.00 → 0.00 |
| final | k | 0.16 → 0.16 | 1.51e-02 → 1.37e-02 | not reproduced → distributed | -0.00 → 0.16 | -0.05 → -0.03 |
| final | v | 0.74 → 0.55 | 7.13e-03 → 8.46e-03 | mean → mean | 0.75 → 0.74 | 0.28 → 0.35 |
| final | o | 0.66 → 0.57 | 9.82e-03 → 1.15e-02 | mean → mean | 0.92 → 0.96 | 0.10 → 0.07 |
| final | up | 0.32 → 0.29 | 2.25e-02 → 2.36e-02 | mean → mean | 0.98 → 0.69 | 0.04 → -0.00 |
| final | down | 0.47 → 0.40 | 1.58e-02 → 1.69e-02 | mean → mean | 0.95 → 0.88 | 0.09 → 0.08 |

old: readings {'distributed': 7, 'mean': 33, 'sink': 5, 'not reproduced': 3}; cross-fitted/in-sample medians mode1 1.12, modes2-5 0.80, to_q0.1 0.52, q0.1-q0.5 0.31, q0.5-q0.9 0.21, below_q0.9 0.18

variant: readings {'mean': 35, 'distributed': 5, 'not reproduced': 1, 'sink': 7}; cross-fitted/in-sample medians mode1 1.16, modes2-5 0.87, to_q0.1 0.53, q0.1-q0.5 0.30, q0.5-q0.9 0.21, below_q0.9 0.19

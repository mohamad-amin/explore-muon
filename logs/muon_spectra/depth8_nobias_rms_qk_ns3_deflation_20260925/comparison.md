# NS-budget deflation replication: results

One seed per arm; readings as predeclared in research/adamw_spectra/MUON_CASE.md.

| run | final val NLL | median step seconds |
|---|---|---|
| 5-step reference | 3.71310 | 0.898 |
| 3-step plain | 3.71941 | 0.894 |
| 3-step deflated | 3.71144 | 1.150 |

Δ = deflated − plain = -0.00797 nats/token → **consistent but small**.
Plain 3-step − 5-step reference = +0.00631; deflated 3-step − 5-step reference = -0.00166.

Validation NLL along training:

| step | 5-step reference | 3-step plain | 3-step deflated | deflated − plain |
|---|---|---|---|---|
| 50 | 6.2518 | 6.3506 | 6.2729 | -0.0777 |
| 100 | 5.4469 | 5.6378 | 5.5614 | -0.0764 |
| 200 | 4.6524 | 4.8782 | 4.8307 | -0.0475 |
| 300 | 4.3235 | 4.4472 | 4.4137 | -0.0335 |
| 500 | 4.0765 | 4.1208 | 4.1042 | -0.0166 |
| 750 | 3.9389 | 3.9580 | 3.9449 | -0.0131 |
| 1000 | 3.8633 | 3.8763 | 3.8654 | -0.0110 |
| 1250 | 3.8137 | 3.8235 | 3.8137 | -0.0098 |
| 1469 | 3.7131 | 3.7194 | 3.7114 | -0.0080 |

Deflation gate (all 48 body matrices per step):
- steps 1–100: 0.38 of matrices fired, mean 5.7 pairs
- steps 101–700: 0.08 of matrices fired, mean 9.2 pairs
- steps 701–1469: 0.17 of matrices fired, mean 7.4 pairs

Momentum ρ1 and median (Frobenius-normalized), mean over steps 1300–1469:

| depth | kind | 5-step reference ρ1 / median | 3-step plain ρ1 / median | 3-step deflated ρ1 / median |
|---|---|---|---|---|
| early | q | 0.42 / 8.41e-03 | 0.34 / 8.48e-03 | 0.38 / 7.92e-03 |
| early | k | 0.29 / 8.23e-03 | 0.25 / 8.60e-03 | 0.23 / 7.75e-03 |
| early | v | 0.74 / 4.48e-03 | 0.65 / 5.62e-03 | 0.70 / 4.44e-03 |
| early | o | 0.31 / 1.25e-02 | 0.36 / 1.09e-02 | 0.29 / 1.20e-02 |
| early | up | 0.53 / 1.39e-02 | 0.45 / 1.53e-02 | 0.47 / 1.40e-02 |
| early | down | 0.20 / 2.49e-02 | 0.48 / 2.03e-02 | 0.58 / 1.83e-02 |
| mid | q | 0.44 / 1.20e-02 | 0.42 / 1.30e-02 | 0.45 / 1.17e-02 |
| mid | k | 0.23 / 1.08e-02 | 0.17 / 1.22e-02 | 0.17 / 1.14e-02 |
| mid | v | 0.73 / 4.98e-03 | 0.74 / 5.44e-03 | 0.75 / 4.91e-03 |
| mid | o | 0.13 / 1.71e-02 | 0.30 / 1.40e-02 | 0.26 / 1.53e-02 |
| mid | up | 0.46 / 1.90e-02 | 0.48 / 1.93e-02 | 0.49 / 1.86e-02 |
| mid | down | 0.13 / 2.67e-02 | 0.06 / 2.82e-02 | 0.05 / 2.84e-02 |
| late | q | 0.35 / 1.40e-02 | 0.42 / 1.31e-02 | 0.43 / 1.35e-02 |
| late | k | 0.21 / 1.31e-02 | 0.23 / 1.29e-02 | 0.16 / 1.35e-02 |
| late | v | 0.69 / 6.15e-03 | 0.72 / 5.77e-03 | 0.74 / 5.47e-03 |
| late | o | 0.14 / 1.70e-02 | 0.15 / 1.52e-02 | 0.14 / 1.58e-02 |
| late | up | 0.29 / 2.31e-02 | 0.36 / 2.30e-02 | 0.36 / 2.25e-02 |
| late | down | 0.16 / 2.60e-02 | 0.10 / 2.75e-02 | 0.11 / 2.72e-02 |
| final | q | 0.31 / 1.36e-02 | 0.31 / 1.35e-02 | 0.34 / 1.41e-02 |
| final | k | 0.16 / 1.37e-02 | 0.18 / 1.34e-02 | 0.20 / 1.43e-02 |
| final | v | 0.55 / 8.46e-03 | 0.57 / 7.75e-03 | 0.57 / 8.04e-03 |
| final | o | 0.57 / 1.15e-02 | 0.55 / 8.45e-03 | 0.49 / 9.79e-03 |
| final | up | 0.29 / 2.36e-02 | 0.32 / 2.32e-02 | 0.32 / 2.30e-02 |
| final | down | 0.40 / 1.69e-02 | 0.38 / 1.82e-02 | 0.38 / 1.81e-02 |

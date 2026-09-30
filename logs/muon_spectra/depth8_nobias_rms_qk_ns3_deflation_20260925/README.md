# NS-budget deflation replication, frontier-norm depth 8 (2026-09-25)

The user authorized this on 2026-09-25 ("run it to confirm … the way they do it", on priv-g14 and g20). The decision argument, recorded prediction and predeclared reading are in [MUON_CASE.md](../../../research/adamw_spectra/MUON_CASE.md#ns-budget-deflation-replication-2026-09-25).

## Arms

Both arms are identical to the frontier-norm Muon run (`../depth8_w512_nobias_rms_qk_20260925/muon`) except for the NS map. That run used the paper's five polynomials and reached val NLL 3.71310; it is the 5-step reference.

| arm | allocation | NS map |
|---|---|---|
| `plain/` | 2567578, priv-g14 (4× L40S) | classic Muon quintic (3.4445, −4.7750, 2.0315) × 3 |
| `deflated/` | 2618555, g20 (4× RTX 6000 Ada) | the same × 3, behind the Sun, Kang & Yang deflation entry |

The deflation entry is ported from their public code (github.com/ComputationalRobotics/Spectral-Deflation @ 1eb15ce, `defmuon/optim/deflation_batched.py`). It matches their `batched_rmfro_clip` to 1e-7 in `test_deflation.py`. Constants are their operating point: window 0.025, oversampling 0.025, one subspace iteration, threshold 0.1, pad 1.01, and a sketch seeded per step/bank/chunk.

## Files

- `expectation.py` → `expectation.json`: the prediction recorded before launch, from the reference run's final momentum.
- `frozen/` and `frozen_manifest.json`: the frozen sources (33 files). `unit_tests.log` has 33 tests passing (one skip, which needs the live log tree).
- `plain/` and `deflated/`: each holds `config.json`, `launch.json`, `pipeline.json` and its stages (`tiny_cuda`, `qualification`, `scientific`).
- `compare_deflation.py` → `comparison.md`.

## Results (complete 2026-09-25 07:50 UTC)

Full tables: [comparison.md](comparison.md). One seed per arm.

| run | final val NLL |
|---|---|
| 5-step reference (paper polynomials) | 3.71310 |
| 3-step plain | 3.71941 |
| 3-step deflated | 3.71144 |

- **Predeclared reading.** Δ = deflated − plain = −0.0080 nats/token: "consistent but small". The deflated 3-step run finishes 0.0017 below the 5-step reference, like the paper's Polar Express result; the difference is within single-seed noise.
- **The effect is mostly early.** Deflated − plain is −0.078 at step 50, −0.076 at 100, −0.048 at 200, −0.017 at 500, −0.011 at 1000 and −0.008 at the end. The gate fires on 38% of matrices in steps 1–100 (mean 5.7 pairs), 8% in steps 101–700 and 17% in steps 701–1469, matching the paper's finding that most of the gain comes early.
- **Cutting NS steps mostly costs early at this scale.** Plain 3-step versus 5 steps is +0.099 at step 50, +0.23 at 200, +0.044 at 500 and only +0.006 at the end.
- **Step time is mostly hardware.** The medians are 1.150 s (deflated, g20) versus 0.894 s (plain, priv-g14). The old-architecture 5-step Muon on g20 ran at 1.138 s, so most of the gap is hardware; deflation's own overhead is not separable here.
- **The momentum spectra are similar** across the three runs over steps 1300–1469.

At 77M, deflation does what the paper says: it recovers what a reduced NS budget loses, mostly early. This does not test the late-layer, large-scale squeeze the scaling paper predicts.

# Is C's late deficit a lagging implicit bias? (2026-09-25)

Read-only probe on final checkpoints. In a bias-free network, the component of W along the
mean input x̂ = x̄/‖x̄‖ acts as a bias. All weights are frozen, and one output vector δ is fitted per body matrix
along a single input direction d (48 matrices; y += (x·d / RMS(x·d)) δ):
- the mean input direction x̂;
- control: a fixed random unit direction z ⊥ x̂.

The fit uses Adam on 4096 fresh training sequences (offset 2.4e9), batch 32, 400 steps, constant then linear decay, FP32, one A6000 on dev-gpu. Loss is measured on the first 256 validation sequences.

`probe.py` (v1) is kept for provenance, but it is invalid. It left the mean-direction δ unnormalized, so in output units it stepped about 17× harder than the control, and it made both models worse (M −0.0062, C −0.0063). `probe_v2.py` normalizes each direction by RMS(x·d) and runs two learning rates.

| checkpoint | held-out NLL | mean-dir gain (lr 1e-4 / 3e-4) | random-dir gain (1e-4 / 3e-4) | in-sample gains |
|---|---|---|---|---|
| M_lr0.01_g20 (seed 260924) | 3.66417 | −0.00006 / −0.00020 | +0.00010 / +0.00000 | +0.0013 to +0.0032 |
| C_track1_drop (seed 260924) | 3.67802 | +0.00004 / −0.00001 | +0.00014 / +0.00011 | +0.0012 to +0.0034 |

**Reading.** Neither model gains anything on held-out data from refitting its implicit biases (|gain| ≤ 0.0002). In-sample gains are noise fitting, of equal size along random directions. On these sequences C's deficit is +0.0139 (+0.0094 on the full validation set), and none of it is a recoverable bias offset.

At δ = 0, C's single-batch gradient along x̂ is 1.2–1.5× M's for q, o and up. That is consistent with C's 2× larger training gradient norm, but it is batch noise, not a systematic misfit. Dropping the spike step changed where the rest of the network went, not just its biases.

The bias-split variant (`mean_bias_adam`, implemented and unit-tested) therefore loses its main motivation. It is not launched.

## Input anisotropy beyond the mean (`input_covariance.py` → `input_covariance.json`)

Final checkpoints at seed 260925 on A6000: Muon (M, val 3.71457) and the SOAP-Muon core (S, val 3.68699). The statistics use 128 validation sequences, position 0 excluded, FP64. q, k and v share one input.

| matrix | mean share M → S | centered top/median eigenvalue M → S | effective rank M → S |
|---|---|---|---|
| block02 q/k/v | 0.31 → 0.67 | 325 → 405 | 100 → 73 (of 512) |
| block02 up | 0.41 → 0.74 | 394 → 422 | 81 → 63 |
| block02 o | 0.19 → 0.50 | 306 → 591 | 105 → 48 |
| block05 q/k/v | 0.31 → 0.61 | 57 → 73 | 230 → 183 |
| block05 o | 0.09 → 0.32 | 73 → 198 | 204 → 114 |
| block08 q/k/v | 0.17 → 0.39 | 59 → 66 | 238 → 211 |
| block08 o | 0.41 → 0.71 | 187 → 144 | 155 → 150 |
| block05 down | 0.07 → 0.06 | 116 → 213 | 810 → 622 (of 2048) |

**Readings.**
- Beyond the mean, body-layer inputs are strongly anisotropic. The centered top eigenvalue is 55–590× the median, the 90th percentile is 3.5–8× the median, and the effective rank is 50–240 of 512. So Muon's implicit isotropic-input assumption fails well beyond x̄.
- The SOAP-trained network has a much larger common mode: the mean share is often twice Muon's. It also has lower-rank inputs, yet its loss is 0.028 better. So growth of the mean input (seen in C and E) is not itself harmful. This weakens the "common-mode growth" explanation of C's late deficit.

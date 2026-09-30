# Retrospective decision argument: what does the SNR statistic identify?

2026-09-28, before the numerical null audit.

## Observable and missing premise

`frame_snr_probe.py` estimates squared entrywise signal by
`max(mean(g)^2 - sample_var(g)/K, 0)` from K=64 microbatches of 16384
tokens, a total of 1M tokens. It extrapolates the denominator variance to
1M/4M/16M and reports estimated signal energy above SNR=1. The protocol
interprets the resulting 95–100% as evidence that normalization is largely
signal-dominated. The estimator's null behavior has not been reported.

## Mechanism and competing explanation

Positive clipping makes the estimated squared signal positively biased. For
fixed-frame iid zero-mean Gaussian entries, the high-SNR condition is
`t_(K-1)^2 > 1 + K*m/B`, a modest threshold even without signal. Weighting by
the same positively clipped quantity may make nearly all its mass clear the
threshold when B exceeds the measurement budget. Genuine large signal is the
simplest competing explanation; a null example cannot exclude it in a network.

## Bounded discriminator

On CPU only, compute the analytic Gaussian-null result and check it against
Monte Carlo sufficient statistics; extract the existing three states' raw
and SOAP aggregate results beside that baseline. Also construct a small
heterogeneous-signal example to show whether signal-energy weighting and
adaptive-direction weighting answer different questions. Preserve all
scripts, source hashes, raw numbers and seed here. No model forwards or
training; two CPU threads and under two minutes expected.

## Decision rule

If pure noise produces the reported qualitative high-energy-fraction pattern,
that statistic alone cannot select the proposed SOAP mechanism. Existing
training improvements remain valid. If the null energy fraction is small,
this particular objection weakens. Either way, identify what raw sufficient
statistics or cross-fitted measurement would actually distinguish signal
equalization from noise control before recommending an intervention.

## Independent discussion

The signal_noise_review peer independently derived the same Student-t
threshold before seeing this derivation. It also identified basis selection
on the same gradients, time-varying frames, and pre-normalization weighting as
additional scope limits. Its full note is stored in `signal_noise_peer/`.

## Follow-up on saved tensors (after null calibration)

The finite-K null gives 86.11%, 98.69%, and 99.91% estimated signal energy
above SNR=1 at 1M/4M/16M even with zero true signal. This establishes that the
high weighted fraction does not identify a mechanism. It does not establish
that the real gradients lack signal.

The older `frame/` artifacts offer a complementary measurement: eigenframes
learned on separate sequences, per-sequence mean/variance from 8192 sequences,
and the actual independent training displacement projected into those frames.
Read four methods' saved step-500 tensors on CPU, sequentially, and compare
raw signal-energy fractions to actual-displacement-energy fractions under
the same estimated SNR masks. Report optimizer-only displacement and full
body displacement separately, retain signed aggregate signal estimates, and
report the saved block-variance diagnostic. No quadratic loss model or GN
eigenband interpretation is needed. Estimated masks are descriptive; their
selection bias remains and no population class-membership claim is made.
Expected cost: four ~672 MiB reads, <=2 CPU threads, <=3 minutes. Results
guide which missing measurement matters; they do not license an intervention.

Step-500 result: >99% of positive signal energy clears the estimated SNR
threshold for all four optimizers, but only 13–23% of actual update energy does.
The complementary weak group has positive estimated first-order descent for
all six matrix kinds and all four methods. This does not establish that every
weak entry is useful; it makes a raw-energy rationale for dismissing them
untenable. Extend the same fixed analysis to steps 200 and 900 (eight additional
saved files, about 5.4 GiB) to check whether this is a transient or persistent
pattern. No new masks or tuned cutoffs; expected CPU runtime under one minute.

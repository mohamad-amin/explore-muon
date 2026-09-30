# Calibrate the retained factor differences before new model work

After the completed fixed-state run, before this saved-tensor calculation.
Independent review recommended this order (`../ts_repeatability_peer/RESULT_REVIEW.md`).
Fresh eight-sequence directions have cosine~.70; same-input new-label draws
have~.74, despite median B cosine~.95. Pooling helps, and the EMA-refresh proxy
is much steadier. It remains unclear whether small raw covariance differences
are magnified in weak factor directions or whether most instability is already
present before polar normalization.

Read retained A0/A1/B0 factors and use independent CD as the reference. Compare
trace-normalized factor differences both directly and in CD inverse-root
coordinates. CD is a finite estimate, not population truth. Retain eigenvalue
condition summaries and contributions from the lower three quarters of CD's
eigenbasis, including group dimensions; do not label these exact population
signal/noise bands or change damping.

For the two fixed comparisons A0/A1 and A0/B0, inspect all48 matrices at
LMR, NS(LMR), and L NS(LMR) R. Normalize each comparison stage to the same
per-matrix target norm to avoid changing aggregate matrix weights. Also hold
pre- or post-L fixed to A0 in two hybrid maps, retaining the interaction rather
than assigning additive noise percentages. Validate final A0/B0 maps against
their saved tensors.

No models, forwards, new labels, loss scores or tuning. Two CPU threads;
maximum180s and expected under120s. New files stay in `run2/calibration/`.
This identifies sensitivity in the implemented map, not optimization harm or
historical EMA variance. It precedes any proposed held-out functional scoring.

The first calibration stopped at its retained-direction check before any
summary. It had reassociated FP32 products relative to the frozen source.
The second attempt restores right-factor-first operation order and keeps
the1e−5 tolerance. Original source/failure/log remain under `run2/calibration/`
and `run2/calibration.log`; successful retry outputs belong to `calibration2/`.

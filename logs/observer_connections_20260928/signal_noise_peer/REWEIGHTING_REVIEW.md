# Independent review of the observer's numerical audits

2026-09-28. Reviewed `signal_noise_audit.py`, `signal_noise_audit.json`,
`reweight_saved_frames.py`, `saved_frame_reweighting.json`, and
`saved_frame_reweighting_temporal.json` against their original producers,
`measure_frame.py` and `research/adamw_spectra/gn_probe.py`. No model calls,
GPU work, or original-source changes.

## Findings

The null calibration is correct. Its finite-K Beta-tail expression matches
the independent derivation in PEER_REVIEW.md; the Monte Carlo draws the exact
sufficient-statistic distribution. The entrywise-normalization counterexample
also computes what its labels say and clearly excludes the subsequent polar
map. It shows non-identification, not an empirical claim about real SOAP.

The saved-frame reweighting reads the statistics correctly:

- `mean` is a per-sequence loss-gradient mean; `noise` is an unbiased
  per-sequence sample variance. Both definitions are in `Frame.summary()`.
- The mean-estimation count is 8192 sequences, and the target training batch
  is 2048 sequences (1M tokens). Subtracting `noise / 8192` and comparing
  positive signal times 2048 with `noise` gives the stated estimated SNR mask.
- Frames are learned on 512 separate sequences. Unlike the newer frame-SNR
  probe, this avoids selecting bases using the scored gradients themselves.
- `update` is literally the projected adjacent-checkpoint difference
  W[t+1]-W[t]. The sign in `-mean * update` is therefore correct for linear
  descent. This is the actual hidden-body displacement, excluding the
  unembedding and other non-matrix parameters.
- All four run configurations specify ordinary decoupled decay. Subtracting
  the saved `-lr * wd * W[t]` term is appropriate. **Call the residual
  "decay-subtracted displacement," not the exact pre-write optimizer update**:
  floating-point parameter-write rounding remains in this residual.

At step 500 the estimates are:

| Method | Positive estimated signal energy in high mask | Actual displacement energy in high mask | Complementary weak group's linear-descent share |
|---|---:|---:|---:|
| Muon | 99.195% | 22.938% | 5.997% |
| PD | 99.591% | 15.603% | 10.567% |
| SOAP-Muon | 99.293% | 17.229% | 13.199% |
| S∘PD | 99.711% | 13.178% | 16.250% |

The weak group's estimated slope is positive in every one of the 24
method-by-matrix-kind combinations. The step-200 and step-900 extensions
preserve that property in all 48 additional combinations. At step 200 the
high mask holds 17.1–30.9% of displacement energy; at step 900 it holds
13.4–22.2%. The contrast is consequently not restricted to the single
step-500 snapshot.

## Claim limits

This is strong descriptive evidence that the raw-energy weighting discards
information about where normalized optimizers actually move. It is not
evidence that weak coordinates individually carry useful population signal,
nor that their large update energy is well allocated. The same held-out mean
both defines each mask and scores its slope; the full displacement is
independent of that mean, but the masked displacement is not. Signed group
slopes are therefore **in-sample with respect to mask selection**, even though
the gradient data are held out from training.

The weak group holds most displacement energy but only 5–16% of the linear
descent at step 500. Larger energy is not itself a benefit, and omitting its
curvature and cross terms prevents a net loss-decrease interpretation.
These are Kronecker-coordinate masks, not exact-GN spectral bands. Methods
are examined at their own different states. There is no isolated causal
comparison or established weak-subspace improvement.

The saved block-variance medians range roughly 1.04–1.35 over all these
checkpoints. This modest excess supports reporting the independence
assumption explicitly; eight-sequence block diagnostics do not validate a
1/B noise law all the way to a 2048-sequence batch. None of these caveats
undermines the direct observation about measured energy allocation.

## Most useful next connection from saved data

Connect **where update energy goes** with **whether the mean gradient persists**.
The existing persistence files internally use one fixed frame and independent
sequence sets for t and later checkpoints. Aggregate their signed cross
moments in bins determined only by the separate basis eigenvalues and matrix
kind; retain odd/even lags where available. Associate those bins with actual
displacement-energy allocations. This can distinguish immediate large,
oscillating gradient components from small persistent ones without new model
calls or pretending that static sampling SNR measures temporal usefulness.

An enticing alternative is to reconstruct a same-state independent split:
the frame file averages 8192 sequences, while a persistence file uses the
first 4096 at t. Under an identical frame, the second-half mean equals
2*full_mean - first_mean, and the second moment follows the same subtraction
before converting back to an unbiased variance. However, **the bases were not
saved**. Equal basis eigenvalues, seeds, and source are useful consistency
evidence but cannot by themselves rule out eigenvector sign changes or
degenerate rotations. A failure of variance positivity or strong-mean
consistency would refute the reconstruction; passing cannot certify it.
Keep such reconstruction exploratory unless provenance independently
establishes identical coordinate bases. Do not silently infer sign alignment
from the gradient samples themselves, because that would bias the proposed
cross-fit.

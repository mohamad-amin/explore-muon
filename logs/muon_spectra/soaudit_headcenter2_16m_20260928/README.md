# Centered head whitening without norm matching at 16M (second-order audit, 2026-09-28)

This is the principled version of the head test. Adam acts in coordinates where the head's input is whitened, about its
EMA mean, and the step is mapped back unscaled: dW = −lr Adam(G R) R, with R = (Σ_h/mean + 1e−3)^−α and Σ_h the centered
covariance. This is plain reparametrized Adam, whose function-space step is comparable to plain Adam's.

The earlier arms (`../soaudit_headwhite16m_20260928`, `../soaudit_headcenter16m_20260928`) norm-matched |dW|. That
shrinks the function-space step by |u|/|u R|, and the uncentered arms also suppressed the input's mean, the output-bias
channel. Both confound the early phase.

- **Arms.** PD α ½ @0.028 β 0.9, 16M, seed 260925, on priv-g14 (L40S), with the head in centered, unmatched whitened
  coordinates at α ½ and α ¼. Reference: PD 4.6711 (L40S).
- **Prediction.** Early deficits under +0.1 at step 10; final ≤ 4.64 at α ½.
- **Code.** Frozen after adding `head_whitening_norm` (default "match", as the earlier arms), with a unit test.

## Stopped (13:49 CDT) after an independent review; kept as a record
The α ½ arm had passed qualification and taken a few scientific steps; the α ¼ arm never started. The review
(MUON_CASE, 13:5x) judged the head-whitening line a repair cascade: three designs in two hours, each queued before the
previous was read. It also found the principled control missing: plain Adam on the head at a head-only LR of ×⅓ and ×3,
since an early deficit that recovers slowly is what a smaller effective head LR produces. And it noted that PD itself is
norm-matched in weight space, with a re-tuned LR. The line pauses until that control exists.

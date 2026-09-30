# Coordinate covariance is a qualification, not a new optimization objective

2026-09-28. The prior V/O gauge observation prompted a possible connection
between PD's activation geometry, normalization and geometry decay. A fresh
independent review recommends dropping a broad saved-matrix stress test:
its main failures follow from known algebra and would not establish a
training limitation. Existing magnitude-matched and hyperball controls also
constrain a normalization-only remedy.

Qualify only these exact distinctions on one declared2×2full-rank example:
the model product is unchanged by V'=SV,O'=OS^-1; ideal undamped/raw alpha.5
PD is covariant on O's input side, but not on V's output side; the full ideal
two-sided half-power oracle is covariant under this internal gauge; per-leg
parameter-Frobenius matching generally breaks it; ordinary scalar weight
decay commutes with the gauge but the code's shaped decay need not.

Matrices and transform are fixed in`check.py`, not searched for large errors.
Use exact FP64SVD and SPD roots. Require product and positive-control
covariance errors<1e−10. Report all other errors rather than assigning a
performance score. Include raw versus mean-normalized C, .001damping, and
norm graft as distinct qualifications; no conclusion about online momentum
transport, clipping or cached statistics follows.

CPU milliseconds, no checkpoint/model/GPU/training. Primary sources and
existing negative controls will accompany the algebra. This is not a novelty
claim, new optimizer, gauge-balancing proposal or request for a training arm.
Keep all files here. Review:`../gauge_optimizer_peer/REVIEW.md`.

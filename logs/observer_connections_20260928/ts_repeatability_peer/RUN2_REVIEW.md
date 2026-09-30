# Runtime revision and instrument review

2026-09-28; reviewed while run 2 was still constructing its fixed input
roots, before any scientific pair metrics were available. No model loaded
or evaluated by this reviewer. This supplements, rather than replaces,
the prospective `NOTE.md`.

**Proceed.** Run 1's preserved result verifies zero relative error against
the frozen B function and against reconstruction of all 48 weight gradients
from their hidden input/error tensors. It stopped before bank measurement
because its runtime model priced all 48 matrices at the largest dimension.
Using actual shape counts with a 30% margin is a legitimate runtime
qualification correction. It leaves the scientific banks, factors, 48-matrix
scope, numerical rules, and 900-second limit unchanged.

The corrected script prices 8 large and 40 small roots, 32 square maps,
8 MLP-up maps, and 8 MLP-down maps. Its map count includes the PD control.
The qualification forward timing includes a second gradient calculation,
which the bank computations do not perform. Preserving the original stop,
timing inputs, and frozen executed sources makes the correction auditable.

The revised use of saved rank-0 input statistics is acceptable for a common
conditional R. It is after-step-46 input geometry, not the actual cached
step-45 root or the other owners' roots. The protocol now states that
distinction explicitly; the diagnostic isolates B variation conditional on
this proxy rather than replaying training.

Read-only inspection of `ts_repeatability/probe.py` and its frozen helpers
confirms the following:

- The model is eager FP32, in evaluation mode, with statistic tracking
  disabled. The MLP and attention weight tensors alone require gradients.
- The factor error is differentiated from summed cross entropy at all
  positions and averaged over all token rows. This includes future-token
  attention paths. No unintended token-mean scaling is introduced.
- All 48 body matrices are used in their recorded parameter order. Input
  covariance is divided by its saved weight, and eigenvalue normalization
  and damping agree with the frozen root formula.
- Bank A and B have two independent label draws, C and D one. All six raw
  factors are retained. Pools use first draws only and average raw B.
- The leave-one-bank-out anchor and .8/.2 refresh mixing are implemented
  before root normalization. The two-sided map and per-matrix shape/norm
  convention are correct. The BF16 control explicitly changes only the
  NS arithmetic, with FP32 products outside NS, and is not called CUDA
  replay.
- Whole-body, per-kind, and individual-matrix distances remain available.
  Trace normalization of L distances is accompanied by trace ratios and
  pre-normalization direction magnitudes, so magnitude changes are not
  wholly hidden.

One interpretation detail was sent to the parent before bank outcomes:
the executable's material-sensitivity flag compares independent-16 cosine
with the *minimum* of six single-8 cosines. The review's suggested pooling
comparison was their *mean*. Beating the worst pair is an order-selected,
weak demonstration of pooling improvement. Preserve the executed flag,
but report the independent-16 value against the full single-8 range and
mean, and do not infer a general pooling benefit merely from the minimum
comparison. If a stricter interpretation is adopted before outcomes, keep
the original flag and that prospective amendment visibly separate.

There is no scientific reason to restart or alter the ongoing instrument
over this reporting issue. No additional model computation or broader
experiment is requested by this review.

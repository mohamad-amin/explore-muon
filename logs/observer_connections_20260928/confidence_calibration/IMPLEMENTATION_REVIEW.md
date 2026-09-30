# Independent confidence-calibration implementation review

2026-09-28, before execution/outcome inspection. Reviewed `probe.py` against
the fixed protocol and completed pairing review. No model or scientific
score was run or read. This is the only new file.

**Verdict:** no critical scientific, temperature-math or state-loading
issue found. The fixed six-endpoint scoring can proceed after the routine
terminal-status handling improvement noted below. No scientific criterion
needs changing.

## Temperature mathematics and units

The parameter a multiplies logits, so a<1 softens and a>1 sharpens. It is
inverse temperature, not temperature itself. The code computes every fixed
grid loss as `logsumexp(a*z)-a*z_y` and preserves the scale-one reference.
The output shape `(6,32,5,512)` correctly indexes model, sequence, scale
and target position.

At a=1 the implemented derivative is `E_p[z]-z_y`, exactly equal to
`NLL-H(p)`. The implemented curvature is the centered second moment
`Var_p(z)`, which is the second derivative with respect to **a**. It is
not mislabeled as the log(a) second derivative. The latter would also
contain the first derivative, so the analysis should retain the present
scale convention.

The fixed synthetic qualification checks the derivative by central
difference and verifies invariance to per-token common logit shifts at
all grid values. Every scientific sequence checks the derivative identity,
finite/nonnegative curvature and CE, and FP64 versus ordinary FP32 CE at
scale one. FP64 calculations follow a frozen FP32 forward with no
probability clipping, gradients or fitted response transformations.

## Endpoint, source and data gates

The script explicitly verifies completion, final steps 92/184, total
tokens, sidecar consistency and final scalar-score records. It checks the
intended alpha/LR/batch/seed and each constant/warmup momentum configuration.
After removing intended schedule/horizon/checkpoint-list differences and
normalizing disabled defaults, the remaining configurations must match.
Execution-source hashes are checked against each run's metadata, and
the required core implementation files must be present.

The actual copied inference helpers must be byte-identical to the already
qualified dose-function source. Binary checkpoint config, step and tokens
must match its metadata. Strict parameter loading is followed by an exact
model-state tensor hash comparison, plus 84-tensor and parameter-count
checks. Reusing the same architecture for all six weight sets is therefore
properly qualified. Input-statistic buffers are disabled by the common
builder and are not used for inference.

The corrected data banks are inside the verified 3.2B-token stream and
beyond all six training prefixes. The runtime manifest must match the
archived reference, and offset bounds include the extra target-shift token.
Both banks are disjoint from each other and from the explicitly listed
prior observer intervals. Saved input/target hashes specify the exact
common score panel. No wrapping, additional data staging or sealed-panel
access is present.

## Resources and score preservation

Numerical environment variables are now assigned to two threads before
NumPy/Torch imports, and Torch is explicitly limited to two intra-op and
one inter-op threads. CUDA is disabled and every model is placed on CPU.
Only one model and one sequence's vocabulary tensors are processed at a
time; the six-endpoint score arrays are small. The model's parameters are
never passed to an optimizer.

After the first complete sequence, the forecast correctly includes 191
remaining scientific sequences with all five vocabulary reductions, five
additional load/hash costs, 25% margin and overhead. The repeated first
forward has already occurred and is included in elapsed time. There are
192 scientific forwards plus that one qualification forward. Wall time
is checked after each sequence, so the checked boundary can overshoot by
one sequence/load without external interruption, as in earlier probes.

**Routine supervision improvement:** in the reviewed version, a forecast
stop saves `result.json` as `cost_gate_stop` and raises into a generic
handler that writes only `failure.json`. No terminal `status.json` is
written there; a later wall failure can likewise leave a stale running
status. Write an explicit terminal status distinguishing a cost stop from
an unexpected failure, and preserve current partial score arrays when
available. This is an execution-record fix, not a gate relaxation or a
new scientific decision. The original frozen protocol/selected family
should remain unchanged.

## Analysis boundary

This script scores the fixed family; it does not yet perform cross-fitted
selection. The downstream analysis must fit a model's scale on one bank
and score it on the opposite bank, retain all scale-one comparisons, and
apply the declared tie and boundary rules. Primary comparators remain
warmup versus beta .8 within each horizon, regardless of fresh raw ranking.

No inference about training-rate mediation or a causal step clock follows
from the stored derivative alone. The meaningful discriminator is the
held-out change in the **paired schedule gap** after equal calibration
opportunity, with both folds, raw-order checks and boundary selections
retained. No score-guided grid or sample expansion is authorized by this
implementation review.

## Pre-launch supervision fix verified

The parent adopted the terminal-record fix before execution. Final reviewed
`probe.py` SHA256:
`b48af6367d42690a89506afcd68914c9b386ae59faac672ce4ea5d68decfaf06`.
The partial-array writer and start time are registered for the outer failure
handler; timeout paths write `cost_gate_stop`, other exceptions write
`failed`, and both preserve traceback, terminal status and available partial
arrays. The same 600-second forecast/wall limits and scientific comparison
remain unchanged. No outcome was inspected in this follow-up verification.

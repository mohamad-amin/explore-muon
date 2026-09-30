# Independent implementation review before dose-function interpretation

2026-09-28. Reviewed `dose_function/PROTOCOL.md`, `probe.py`, the frozen
helper bytes and the old-half versus new-quarter execution-source differences.
No fresh scientific scores, result/status outputs, checkpoint tensors or
model calls were inspected or executed. This is the only new file.

**Verdict:** no critical flaw found. The implementation answers the declared
own-state predictive-amplitude question with actual adjacent saved weights,
not a lagged-momentum proxy or a reconstructed optimizer step. Interpretation
must retain the own-state, finite-panel scope.

## Factorial pairing and source handling

The four paths select the intended alpha {.25,.5} by beta {.9,.8} PD family.
The source checks compare each available scientific execution file with its
recorded metadata hash and require the five central implementation files.
Configuration comparison removes only the two intended factors and fills
specified newer disabled defaults. Initialization, data manifests, parameter
count, budget, GPU type and world size must match.

I inspected the old-half to new-quarter differences independently. The new
head-whitening and prefilter code is disabled; the additional output-factor
SOAP path is inactive for ordinary PD; the new momentum schedule function
returns the original configured beta when warmup is disabled. The default
head remains Linear. These differences do not reveal an unintended active
scientific change. Matched active equations are the claim, not byte-identical
compiled executions across all historical revisions.

All four imported evaluation helpers are byte-identical to the qualified
`body_aux/source` copies. The helper constructs plain Linear body layers
with input-statistic collection disabled, CPU FP32 weights and eval mode.
Strict checkpoint loads, disabled gradients and inference mode are suitable
for these common architectures. No optimizer state is advanced.

## Actual states and four corners

The declared states are steps 9 and 46 for all four arms. Both the full base
checkpoint and the next-weights record are checked for their expected step
numbers. The data offsets are beyond their training exposure. Parameters
must agree exactly by key between the model and both saved endpoints.

The partition is exhaustive: 48 block weight matrices and the 36 remaining
parameter tensors, totaling 84. Each mask reloads **every parameter** from
base or next weights. In particular, switching from body-only to auxiliary-
only resets the body to base, avoiding an accumulated-corner error.
Masks 0/1/2/3 implement base/body/aux/full as intended. Exact tensor equality
is checked for every corner in the initial qualification, and the restored
base receives a repeated-logit check.

Actual body and auxiliary squared displacements are computed from FP64
differences of the stored FP32 endpoints. Their disjoint sum gives the full
norm; these quantities correctly include decay and parameter-write effects.
The ideal reference `192*lr` uses `traces[step+1]`, the rate of the measured
next update, rather than the preceding step's rate.

## Prediction quantities and indexing

The same two four-sequence banks and targets are reused at all eight states.
Tokens are saved and hashed. Every corner performs a complete forward of
that corner's parameter state, so body/auxiliary interaction is measured
rather than approximated by adding predictions.

- Logits come from CPU FP32 inference; log-softmax and probability
  reductions are FP64.
- `sum p_base*(logp_base-logp_corner)` is the declared forward KL.
- `sum p_corner*(logp_corner-logp_base)` is reverse KL.
- Both are retained separately at every token; their symmetric mean is a
  later fixed scalar combination, not an alternative selected after scoring.
- CE gathers the correct target from each corner's log probabilities.
  Base entropy is computed from the same base distribution used in KL.
- KL finiteness and nonnegativity tolerance are checked for every scored
  token. The restored-base logit tolerance, null divergence and FP64 versus
  FP32 CE checks match the stated qualification.

No probability clipping, fitted logit gauge or response rescaling is present.
The per-token arrays preserve the same denominators for NLL and KL, with
four CE corners and three nonbase divergence corners.

The signed finite label-linear term recommended in the design review is
recoverable as `Delta_CE-KL_forward`. If reported, describe it as the
identity-based decomposition of retained measurements. The probe does not
independently compute/store that term from raw logit displacements, so it
should not be advertised as a separate numerical identity gate. This is
not a flaw in the declared primary amplitude comparison.

## Cost and preservation

The first timed four-corner input includes restoration qualification, so
using it to forecast 63 remaining four-corner inputs is conservative. The
seven remaining paired-load costs, 25% margin and overhead are correctly
included. The forecast concerns 256 score-forwards; the extra initial
restoration forward is qualification and already included in elapsed time.

Torch thread counts are explicitly set to 2/1; the launcher's numerical
environment supplies the two-thread bound where BLAS variables otherwise
use `setdefault`. The 600-second wall boundary is checked after every
four-corner sequence, so it may overshoot by one sequence/load without
external interruption. The protocol appropriately calls this a checked
boundary rather than an exact asynchronous deadline.

Large vocabularies make temporary FP64 log-probability arrays material,
but only one score sequence's corner logits are processed at a time.
The probe keeps one model, two memory-mapped endpoints, base probabilities
and the current corner; scalar token arrays are small. It never stores
all states' vocabulary logits or performs a backward pass. Partial arrays,
source/checkpoint identifiers and explicit failures are preserved on the
declared execution paths.

## Interpretation gates after completion

The scalar analysis should retain absolute body/aux/full KLs, both banks,
both stages and all four factorial cells before reporting ratios. Keep
signed NLL changes and the exact finite-loss interaction alongside amplitude;
smaller KL alone is not useful progress. Reverse/symmetric KL are fixed
asymmetry qualifications and must not replace forward KL when their result
is more convenient.

The data do not share a common base state or base predictive distribution
across arms. Consequently these ratios describe actual learned states,
not an isolated effect of applying alpha at fixed momentum/weights. A mixed
result can reject a uniform own-state global-shrink explanation, but neither
a mixed nor consistent result identifies the training mediator. Preserve
the positive main alpha-by-beta training interaction independently of this
local descriptive reading.

# Read-only implementation review

2026-09-28. Reviewed `../value_split/probe.py`, its frozen-probe API and
`../value_split/PROTOCOL.md` while the root monitored execution. No model calls,
new computations, or mutations of probe files were made by this reviewer.

**No critical implementation issue found.** The implemented intervention,
derivatives, scoring units and declared gates agree with the bounded protocol.

## Intervention and AD

`Family` captures each matching checkpoint's FP32 displacement, frozen mu and
Dmu in a separate hook closure. The hook reads `inputs[0]` on every forward;
it does not detach, cache, or replace the live input. Its returned value is
therefore exactly the intended W x + a Dmu + b D(x−mu), modulo floating-point
ordering. The coefficient tensor is unbound directly; neither coefficient
passes through `float`, `item`, a detached tensor, or a parameter-copy operation
in the derivative path. Forward-mode coefficient tangents can reach all later
layers even though model parameters have reverse-mode requires_grad disabled.

The mutable coefficient reference is replaced at the beginning of every
`logits` call. No prior JVP dual is reused for a later derivative or finite
loss. `direct` disables hooks, assigns W+cD to every V, evaluates the model,
then restores all V weights before re-enabling hooks. Other weights remain
at their checkpoint values. On an exception during `direct`, restoration is
not in the finally block, but the exception fails qualification and terminates
the run; there is no scientific continuation in a partially changed model.
The in-memory mutation never writes a checkpoint.

Explicit dropout-free math attention is used for both coefficient and direct
parameter evaluations, avoiding fused-kernel/forward-AD differences.

## Loss and GN units

For each sequence, `terms` uses the base logits' softmax and computes
mean_t [(p−onehot(y)) dot J_i] for the signed CE slopes. It computes
mean_t Cov_p(J_i,J_j) for all entries of the predictive GN matrix. All these
inner products use FP64; losses use the probe's ordinary FP32 token CE.
Both reduce over the same sequence/token axes, and with one sequence per call
their units are mean loss per token. There is no missing T factor. The small
precision difference in softmax is explicitly checked by reverse-mode
coefficient gradients under the predeclared tolerance.

The signed mixed GN entry is retained and the symmetric 2x2 matrix is checked
for PSD within an absolute 1e−9 tolerance. Logit RMS is an additional
descriptive scale and must not be interpreted as predictive curvature.

## Qualification and resource gates

Every state is checked before that state's banks: JVP additivity <=1e−5,
diagonal direct-parameter equivalence at c=0.5 and 1 <=1e−5, central differences
at the declared h=0.02 <=0.02, and signed slopes checked against reverse-mode
coefficient differentiation. These are the criteria written in the protocol;
there is no hidden threshold relaxation in the implementation.

Direct equivalence is relative to total logits, not the usually smaller
perturbation. JVP additivity compares three hook-family tangents rather than
an independent direct-weight JVP. These are nonblocking limits given the exact
algebra, direct finite equivalence and resolved finite-difference checks;
retain their actual errors so a reviewer can judge their numerical strength.

The qualification timing multiplied by 17 is a conservative projection
because qualification contains more forward/derivative work than one science
sequence. The code applies this projection before the first Muon bank; it
does not implement an independent hard wall-time interrupt. The root explicitly
monitors the live execution handle and resource time. CPU numerical threads
are capped at two, interop at one, and CUDA_VISIBLE_DEVICES is emptied before
torch import. No optimizer or scheduler submission occurs.

## Provenance and analysis boundaries

The script verifies source snapshot hashes, records its own hash and exact
token hashes, saves tokens, hashes means and both endpoints of every accessed
V tensor, and records/checks checkpoint size and mtime. The mean archive's
hash, state step and method path are recorded/verified. Unchanged non-V weights
are loaded from the checkpoint but not separately tensor-hashed; do not claim
that every checkpoint byte was hashed.

The token stream path and length 512 are fixed in source rather than derived
from each config. They should be reported as the explicit shared scoring
choice and checked against the stored configs; the selected original frontier
models use this length. Banks and coefficient pairs are fixed before scores.
The first qualification sequence is reused within its bank, which is harmless
for an instrument check with no label-based scientific selection. The four
sequences per bank remain four, not five independent score samples.

This review does not enlarge the scientific claim: only actual V displacements
at each method's own step-500 state are measured. Means define an operational
split, and cross-layer/nonlinear finite loss effects remain part of the result.

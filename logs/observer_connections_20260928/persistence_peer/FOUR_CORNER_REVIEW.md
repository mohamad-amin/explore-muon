# Review of the CPU four-corner probe

2026-09-28. Read-only implementation review while the predeclared probe was
running; no repeated model calls. Reviewed `../body_aux/probe.py`, its frozen
model/data/probe sources and manifest, the protocol, direct-load qualification,
and first-bank outputs.

No critical implementation issue found:

* The filesystem root and source snapshot imports resolve as intended.
* Body parameters are the weights of all 48 hidden linear layers, selected by
  identity. The remaining learned parameters are auxiliary. The equality of
  state-dictionary keys and partition keys excludes omitted buffers/parameters
  for this model configuration.
* Every corner overwrites all parameters. Freezing auxiliary parameter
  gradients does not freeze their forward influence on the body gradient.
* Base and full corners reproduce directly loaded model logits exactly on the
  qualification sequence. Eval mode has no stochastic dropout or moving
  statistics, and no optimizer is constructed.
* Per-sequence mean losses and the 1/8 gradient accumulation produce the bank
  mean body gradient. Gradient list order is consistent across all corners.
* The finite-difference interaction and symmetric cross-bank inner-product
  calculation have the intended signs and normalizations. Cross-bank cosine
  estimates should remain unbounded and undefined when self-products are
  nonpositive, as the implementation permits.

The two eight-sequence banks are tiny empirical samples. Same-bank cosines
include shared sampling noise; their strongly positive first-bank values
(base/body .780, base/full .738) cannot reproduce or refute the prior 128- and
8192-sequence measurements. Cross-bank ratios can be highly uncertain even
when both denominator estimates are positive. Their variation is evidence
about resolution, not permission to enlarge this predeclared probe after
seeing the effect.

The saved cross-bank corner Gram supports a useful additional calculation
without forward/backward calls: apply the contrasts

    body = body_corner − base
    aux = aux_corner − base
    full = full_corner − base
    interaction = full_corner − body_corner − aux_corner + base

on both sides to produce a cross-bank **change** Gram. Matched-sequence noise
can cancel strongly within each difference. Retain negative cross-bank
diagonals and signed cross terms; this is still a two-bank estimate without
a validated confidence interval.

Report the loss interaction separately from the gradient-change interaction.
The first-bank loss interaction is +.01858 NLL, whereas the gradient-change
interaction norm is relatively small. These are compatible: finite changes
in the gradient as a vector and a scalar loss's finite mixed difference are
different observables. Neither justifies replacing the complete finite-step
response with a sum of independently scored component steps.

Any eventual attribution is specific to this checkpoint, parameter partition,
numerical path, and held-out banks. A finding that auxiliary scope is modest
would narrow the old discrepancy toward data amount/precision; it would not
show that auxiliary optimization is generally unimportant or that its loss
contribution is modest.

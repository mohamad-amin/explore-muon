# A sign disagreement is mainly sample-dependent; a loss interaction remains

2026-09-28. One saved Muon4M step183→184, two fixed disjoint banks of eight
never-trained sequences each, CPU FP32 eager. The full protocol and independent
review preceded execution. Direct checkpoint and corner reconstructions have
bitwise-identical logits. Runtime80.04 seconds, two threads, no GPU or training.

## The hypothesis that did not survive this check

Earlier probes reported a positive body-gradient cosine for a body-only
counterfactual and a negative one for the full next model. Auxiliary updates
were a plausible explanation. Holding sample and precision fixed shows that
parameter scope changes the angle modestly here; sample pairing changes its
sign much more strongly.

| Base gradient versus | Same bank A | Same bank B | Symmetric cross-bank estimate |
|---|---:|---:|---:|
| Body matrices advanced | .7804 | .7859 | −.6810 |
| Auxiliary parameters advanced | .9948 | .9947 | .9795 |
| Full model advanced | .7381 | .7426 | −.7192 |

The body-only and full-gradient cross-bank alignment is .9921. Both bank
orders give negative base/body and base/full inner products, not merely their
symmetrization. Forty-seven of48 hidden matrices have negative symmetrized
products for both, while all48 base/aux products are positive. These matrices
share data and are not independent replications.

The sample-specific contribution can be separated algebraically, without
assuming a noise law. The mean same-bank base/body inner product is

    1.72065 = −0.12462 + 1.84527,

where the first term is the symmetric cross-bank product and the second is
half the inner product of the bank-difference vectors at the two states.
Those difference vectors have cosine .916; their persistence overwhelms the
opposing cross-bank alignment. Under exchangeable independent sampling this
is shared-sample gradient noise. With two contiguous banks it can also include
correlated or distribution-specific variation. It is therefore most precise
to call it a sample-specific component here.

This is not a numerical reproduction of the earlier128/8192-sequence probes.
It establishes a matched example in which even the body-only gradient changes
sign after removing the same-bank contribution. The old sign discrepancy
cannot be assigned to auxiliary scope alone. Eight-sequence cross-bank ratios
remain noisy estimates; no population confidence interval is claimed.

## Retaining the finite-step gradient interaction

Write g00,g10,g01,g11 for the four body gradients. The exact decomposition is

    g11−g00 = (g10−g00) + (g01−g00) + (g11−g10−g01+g00).

Transforming the cross-bank Gram by these contrasts gives squared-norm
estimates .6153 for the body contribution, .01602 for the auxiliary
contribution, .004245 for the interaction, and .7661 for the total. The body
change and total change have cosine .9979. Their signed cross terms remain
in `analysis.json`; the components cannot be assigned additive norm shares.
This agrees with body-driven gradient rotation at this state without ruling
out important auxiliary effects elsewhere.

## A different, reproducible finite-step interaction affects loss

Let Delta_body=L10−L00, Delta_aux=L01−L00, and
I=L11−L10−L01+L00. Then the full loss change is Delta_body+Delta_aux+I.

| Loss change, nats/token | Bank A | Bank B |
|---|---:|---:|
| Body only | +.000804 | −.021135 |
| Auxiliary only | −.009668 | −.012252 |
| Interaction I | +.018581 | +.017600 |
| Full model | +.009718 | −.015787 |
| Full minus body-only | +.008914 | +.005349 |

Auxiliary-only loss improves on all16 sequences. Yet the interaction is
positive on all16, and the full step is worse than body-only on all16. The
pooled interaction is+.018091. This is a finite-step interaction on a small
held-out sample, not an isolated Hessian block or proof that changing the
auxiliary optimizer would help training. It does show that the body and
auxiliary loss changes cannot be assessed independently and then added.

This connects to the distinction between a body-only GN model and the whole
learning process. The existing GN operator deliberately holds auxiliaries
fixed. Its body-direction diagnostics can be valid within that scope while
missing a material part of the actual joint step. The present check does
not turn its one-step interaction into a rate or a new optimizer result.

## Next decision

Do not expand this sample or optimize its outcome. The fixed experiment has
answered its premise check: sample pairing explains the sign better than the
missing auxiliary update in this case, while a separate positive joint-step
loss interaction is worth retaining. The next architecture question remains
the local usefulness of the shared value route identified in
`../attention_geometry/NOTE.md`. Score actual value displacements in constant
and centered functional components with their cross term before recommending
any change to centering, bias parameters or momentum. Attribution of the
auxiliary loss interaction to head/embedding/norm groups is a distinct future
question, not silently included in this experiment.

## Artifacts and qualifications

- `probe.py`, `source/`: executed probe and frozen model/data/probe helpers.
- `PROTOCOL.md`: fixed samples, resource gate and decisions before the run.
- `result.json`, `analysis.json`, `probe.log`, `status.json`: complete scalar
  measurements, signed Grams, per-sequence losses and terminal status.
- `bank0_body_gradient_means.pt`, `bank1_body_gradient_means.pt`: all four
  body-gradient means, saved for follow-up without another backward pass.
- `probe_tokens.npz`, `PROVENANCE.json`: exact token arrays and source hashes.
- Independent review: `../persistence_peer/FOUR_CORNER_REVIEW.md` and its
  follow-up numerical interpretation.

All48 body matrices and every remaining learned parameter were accounted for.
There are no mutated checkpoints or partial training steps. Tensor entries,
matrices and tokens are not treated as independent experimental replicates.
The run is complete; no observer process is left running.

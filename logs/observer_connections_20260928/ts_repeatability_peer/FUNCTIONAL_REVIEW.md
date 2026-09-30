# Functional check: close the immediate estimator branch, preserve qualifications

2026-09-28. Read-only review of the functional instrument and its saved eight
per-sequence Grams. Independent NumPy aggregation reproduces the pooled
matrix and all three primary ratios to absolute error below 1e-12. Evidence
is in `functional_validation.json`; no model or GPU calls were made here.

The instrument uses the complete body Jacobian and the exact predictive
logit Hessian, conditional on the eight held-out inputs. Centering each
logit tangent under p and taking its sqrt(p)-weighted Gram implements that
quadratic form correctly. The separate diagonal and difference checks
support the instrument's numerical accuracy. Averaging four equal-length
sequences in each of two banks gives the same weighting as averaging all
eight directly.

| Pair | Predictive-GN cosine | Variation / TS–PD difference energy |
|---|---:|---:|
| A0/B0, disjoint eight-sequence factors | .988432 | .005828 |
| A0/A1, same sequences and new labels | .993133 | .003313 |
| AB/CD, independent pooled factors | .993387 | .003759 |

The direction of this result agrees across both score banks and all eight
individual score sequences. For A0/B0 the per-sequence ratio spans
.00476–.00781. This materially weakens the argument that the conspicuous
parameter-space variability is, by itself, a major bottleneck in the
implemented output factor. It justifies **closing this immediate premise
check without launching a new factor estimator or a larger-sample run**.
Retain the candidate/prior-art note as a proposal whose priority has fallen.

Two qualifications are necessary.

First, the functional variation is smaller, not zero. The A0/B0 difference
has GN norm .1549 times A0's own response norm; label-only is .1174. No
cross-entropy slope, true finite loss, online EMA trajectory, rare-target
effect, or future optimization consequence was measured. “Harmless noise”
and “the estimator cannot limit training” would exceed the evidence.

Second, the denominator includes a substantial functional amplitude change.
TS–PD GN cosine is .979–.981, while PD's response norm is 2.27–2.41 times
TS's. The selected eight-direction Gram has leading trace share .9888;
after diagonal normalization it is .9919. This concerns the chosen response
span, not the rank of the full model GN.

As a **post-hoc qualification only**, matching all directions to unit GN
norm changes the variation-to-TS–PD ratio to .286 for A0/B0, .165 for A0/A1,
and .170 for AB/CD. These do not replace the original endpoint or justify
reopening the probe. They show why the correct conclusion is small
variation relative to the *actual, fixed-parameter-norm response change*,
rather than elimination of all uncertainty about geometric shape. The
small original ratio partly reflects the large amplitude effect of TS.

## Broad next artifact question

Move away from repeated B sampling. The unresolved **body–auxiliary
interaction** is a better independent next question: which auxiliary
parameter group accounts for the already measured antagonism between a
helpful auxiliary step and the simultaneous body step?

The existing body/auxiliary counterfactual has a positive finite interaction
on all 16 scored sequences, while auxiliary-only helps on all 16. It has
not attributed that interaction to the head, embeddings, or normalization
gains. The main notebook separately reports that a head-whitening attempt
was confounded by functional step magnitude and missing head-LR controls.
The present result adds another example in which parameter-norm matching
conceals a large functional amplitude change. These observations motivate
checking actual joint-step allocation and parameter scope, without claiming
that an architectural defect or a better auxiliary optimizer is already
known.

Start with an inventory of saved actual auxiliary displacements and their
configuration/scale history, and connect it to existing body/auxiliary
measurements and head intervention records. Avoid assuming that positive
interaction itself is waste: the coupling archive already shows that common
response can carry useful descent. Any later attribution probe should
separate first-order benefit, shared predictive response, and finite-step
interaction. This is a proposed new question requiring its own short
decision argument, not authorization to expand the present probe or tune
head learning rates.

The newer main-thread clipping control is already complete and does not
explain the whitening methods' momentum advantage. Repeating that premise
would duplicate existing work; its current notes should be incorporated
before planning anything involving clipping.

# A preserved shared-value route is not an independently additive gain

2026-09-28. Fixed CPU premise check, complete in98.80s on two threads, no
optimizer or GPU. Eight held-out sequences (two banks of four) are shared
across the Muon and PD own states atstep500. Every other parameter is fixed;
the displacement is the actual saved eight-V-matrix step to501.

## What was measured and qualified

For each layer, mu is frozen from a disjoint2048-sequence validation archive.
With the live current input x, inject `a Dmu + b D(x−mu)` at the V output.
The diagonal a=b=c reproduces direct V-weight replacement W→W+cD, including
changes to the input of downstream layers. The separate component paths
temporarily introduce affine offsets; they are functional counterfactuals,
not standalone bias-free parameter steps. Actual displacement magnitudes
are kept throughout, with no norm matching or fitted step sizes.

Both methods passed all predeclared gates. Relative diagonal logit errors
were2.4–3.3e−7, JVP additivity errors4.4–5.6e−7, and central finite-difference
errors0.09–0.26%. Reverse-mode and JVP loss slopes agreed within1.5e−7.
The source snapshot, means, accessed weights, exact score tokens and per-sequence
results are retained. GN below means the predictive softmax Gauss–Newton,
not the full Hessian; finite interactions are scored on the true model loss.

## Pure, conditional and joint effects

Pooled true loss changes in millinats/token; negative improves loss:

| Actual functional component | Muon | PD |
|---|---:|---:|
| Constant alone | −.759 | −.305 |
| Centered alone | −1.987 | −1.071 |
| Both, exact actual V parameter step | −2.116 | −1.134 |
| Constant added after centered | −.129 | −.062 |
| Finite constant–centered interaction | +.630 | +.243 |

The conditional effect is not consistent across the two banks:

| Constant added after centered | Bank A | Bank B |
|---|---:|---:|
| Muon | +.198 | −.455 |
| PD | +.407 | −.531 |

PD's constant-alone effect also changes sign (+.158/−.769 millinats). The
centered-alone effect is negative in both banks for both states. These are
small fixed samples; pooled negative effects do not establish population
significance. A larger learned mean route therefore did not translate into
a demonstrated repeatable extra local gain for PD at this checkpoint.
This closes the stated pointwise premise check without expanding its sample.
It does not establish route uselessness or absence of a longer-term benefit.

The Muon V-only step has the larger pooled local improvement here despite
PD's better training result. Own-state one-step contributions should not be
used to rank the optimizers or explain their trajectory speed.

## The robust observation is predictive overlap

Write Jc and Jz for the two logit tangents at the base, and Qij for their
predictive GN inner products. The complete quadratic includes2Qcz.

| Quantity | Muon | PD |
|---|---:|---:|
| Loss slope, constant | −.001290 | −.000432 |
| Loss slope, centered | −.002564 | −.001435 |
| Qcc | .001000 | .000249 |
| Qzz | .001110 | .000716 |
| Qcz | .000593 | .000239 |
| Qcz / sqrt(Qcc Qzz) | .563 | .566 |
| 2Qcz / Qcombined | 36.0% | 33.1% |
| Observed finite interaction / Qcz | 1.063 | 1.018 |

Qcz and the true finite interaction are positive on all eight sequences at
each state. These are the same eight sequences at two states, not16
independent replications. Predictive GN explains the observed interaction
closely in this small neighborhood; it need not do so for arbitrary steps.
The GN prediction for the whole V step is −.002206/−.001147 versus observed
−.002116/−.001134. Independent result review reproduces all aggregations.

The overlap is nearly the same under both optimizers despite their different
absolute scales. About80–83% of the pooled constant-alone benefit is offset
by the positive interaction; those cancellation ratios are descriptive and
do not override the bankwise sign changes. The directly useful conclusion is
that improving these two functions separately does not make their benefits
additive, even when their input-space decomposition is called centered.

## Connection to the learned geometry

**Subsequent invariance check:** the learned pre-O mean amplitudes in this
section are coordinate-dependent. `../value_gauge/REPORT.md` finds no universal
larger constant residual-branch component after O, although quieter actual
joint V/O perturbations persist. The logit-space/GN measurements in this
report remain qualified and are invariant to a consistent fixed V/O gauge.

The independent archive contraction (`../value_step_geometry/REPORT.md`)
shows a larger learned mean channel under PD but approximately9.7× smaller
current Dmu energy than Muon atstep500. Its constant-route GN cost here is
only4.0× smaller, reflecting different downstream sensitivities at the two
own states. Neither activation energy nor parameter norm alone measures loss
sensitivity. SOAP-PD also has smaller current shared-route movement in the
archive, but it was not part of this functional probe.

A plausible picture is that whitening permits a shared value route to grow
while making its current perturbations quieter. The present losses do not
show that this is what causes the training improvement. They especially do
not support deleting the shared component: removing the constant step helps
on one bank and hurts on the other, and earlier training interventions that
removed/whitened only the mean were already unsuccessful.

Centering by itself does not remove this coupling. The mean is fixed from
an independent reference distribution, and attention's data-dependent
selection changes the centered component's output mean. The loss metric
adds further weighting and cross-layer effects. Classical affine centering
frameworks include a compensating offset and qualified curvature assumptions;
freeing that offset in a bias-free V is an architectural change, not just
renaming the existing parameterization. See `../centering_literature/NOTE.md`.

## Decision and limits

Do not launch a centering/bias or component-learning-rate intervention from
this result. The local shared-route benefit is unresolved and not uniquely
PD-specific, while coupling is substantial and similar in both methods.
Keep the full displacement and predictive cross terms when investigating
other geometry claims. The next broad lens is the existing direction-Gram
archive: does this interference reflect a few common prediction responses,
or many unrelated alignments? It is a retrospective question, not a new
optimizer proposal.

One seed, one own-state pair, eight shared scoring sequences, fixed empirical
means and only V parameters are involved. Finite counterfactuals and GN
responses are qualified locally; no sustained rate, invariant decomposition,
novelty or causal optimizer comparison is claimed.

## Artifacts

`PROTOCOL.md`, `source/`, `probe.py`, `probe.log`, `result.json`, `status.json`,
`tokens.npz`, `analysis.json`, and `effects.png/.pdf` preserve the full run.
Independent design, implementation and result review is in
`../value_split_peer/`. The probe is terminal and no observer process remains.

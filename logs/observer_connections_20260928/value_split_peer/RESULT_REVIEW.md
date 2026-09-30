# Completed result: keep the coupling observation, close the local route-gain premise

2026-09-28. Independent review of the completed 98.8-second CPU probe,
`result.json`, `analysis.json` and `analyze.py`. No additional model execution.
`check_results.py` independently recomputes finite effects and predictive-GN
errors using scalar JSON only; `RESULT_CHECKS.json` records these checks and
the two source-artifact hashes.

The aggregation is correct. The claim that PD's more strongly preserved
learned mean route earns a repeatable extra gain through this actual V step
is not supported. Muon's constant-only effect improves both bank means, while
PD's improves only one. The conditional constant effect reverses sign between
banks for both methods. Own-state effects cannot isolate an optimizer defect,
and this small experiment does not test sustained learning or the usefulness
of the route under a different step. Close this fixed premise check without
resampling, coefficient fitting, or an architecture recommendation.

## A more stable observation survives

Both the predictive GN cross entry and the true finite interaction are positive
on every scored sequence for both methods. There are eight shared score
sequences and two model states, not sixteen independent document replications.

| Pooled descriptive quantity | Muon | PD |
|---|---:|---:|
| GN cosine of constant and centered logit tangents | 0.563 | 0.566 |
| Cross contribution to total GN quadratic, 2G12 / sum(G) | 0.360 | 0.331 |
| Full GN quadratic / block-diagonal approximation | 1.562 | 1.495 |
| True finite interaction / predictive G12 | 1.063 | 1.018 |
| True constant-only loss change | −0.000759 | −0.000305 |
| True constant effect given centered step | −0.000129 | −0.000062 |

The finite interaction is close to predictive G12 even sequence by sequence:
their ratio ranges 0.920–1.128 for Muon and 0.844–1.146 for PD. This supports
the specific reading that overlapping downstream predictive effects account
for most of this interaction at this state and scale. It does not assert an
exact equality between GN and the coefficient-path Hessian.

The GN model's pooled combined-loss error is +0.000090 nats/token for Muon and
+0.000013 for PD; individual errors are larger (at most 0.000289 and 0.000101).
Keep those numerical limits alongside the agreement. Agreement was not used
to choose a new step.

The pooled interaction removes 83% and 80% of the constant-alone improvement,
respectively. This is simple bookkeeping, not a robust population fraction:
the constant-alone and conditional effects have large sequence variation, and
conditional bank signs disagree. Report the bank values ahead of such ratios.

## Connection to the existing evidence

The input split is algebraically constant/centered; it does not imply
orthogonality after data-dependent attention and the downstream network, in
particular in the predictive GN metric. The almost equal normalized overlap
under the two methods is more defensible here than an explanation based on
the absolute magnitude of the learned mean route. Learned activation-route
strength, actual displacement scale, and independent useful loss contribution
are three different quantities, and this probe directly separates them.

The earlier body/auxiliary counterfactual also found positive finite loss
interactions. Both examples warn against adding standalone gains as though
components acted independently. Only this value-split probe has measured GN
overlap, so it cannot establish that the earlier body/auxiliary interaction
has the same curvature explanation.

The data permit locally useful constant components in some samples and the
pooled scores; they do not justify saying that the component is uniformly
useless, that deleting it helps, or that its preservation explains PD's gains.
Retain this distinction in the observer synthesis and proceed to a different
unresolved connection using the existing evidence.

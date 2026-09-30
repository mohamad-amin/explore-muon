# Positive coupling is broad and mostly aligned with descent

2026-09-28. Reanalysis of saved small matrices only, CPU threads capped at two.
No GPU, model execution, or training. `analyze.py` ran in under one second.

**The archive contains a large, repeatable shared response among chosen update
pieces, but does not identify that response as the value mean route or as a
small eigenspace of the full GN operator.** Raw first-layer dominance mostly
reflects magnitude. After removing that magnitude, the common component is
spread over many layers and kinds and aligns closely with the descent slopes.
It is not evidence that a disposable interfering component has been located.

## What was actually measured

`second_order_audit_20260926/gn/*_gram.json` contains eight Grams from fresh
4M-token gradient directions at the Muon and PD original 1M-batch states at
step 500. The directions are Muon, PD alpha 1/4, K-FAC, and damped exact GN.
The four main `gn/*.json` files contain eight more direction Grams: Muon and
damped GN applied to each state's next momentum, at Muon/PD 1M step 500 and
4M step 183. Those Grams retain two separate 64-sequence score halves.

Each piece is one of 48 hidden matrices. The operator is predictive GN:

    Q_ij = mean_tokens [(J D_i)^T (diag(p) − p p^T) (J D_j)],
    a_i = <g, D_i>.

All these directions hold auxiliary parameters fixed. They are generated
counterfactual directions, using exact SVD polar maps for the Muon/PD entries,
not the actual checkpoint displacements. A scalar optimizer-step convention
therefore must not be inferred from their raw magnitudes. The fresh-direction
Grams use 128 score sequences; the momentum half-Grams use 64 each. These all
come from the validation stream at the archived `EVAL` offset, not the new
observer score banks. No independent seeds were added.

For normalized comparisons use R_ij=Q_ij/sqrt(Q_ii Q_jj). A leading
eigenvector of R is a coefficient pattern over normalized selected responses;
it is not a model parameter or activation direction. Every analyzed matrix
passes finite-value, symmetry and PSD checks. `analysis.json` retains exact
file hashes, all raw/normalized matrices, loadings and pair summaries.

## Concentration persists after normalization, but it is broad

| Selected family and states | q(sum)/sum q(piece) | Leading raw trace share | Leading normalized trace share |
|---|---:|---:|---:|
| Fresh Muon, two 1M states | 16.44–18.18 | .679–.710 | .470–.494 |
| Fresh PD alpha 1/4, two 1M states | 16.93–18.85 | .655–.658 | .456–.474 |
| Fresh K-FAC, two 1M states | 8.22–12.50 | .611–.617 | .437–.449 |
| Fresh damped GN, two 1M states | 2.05–3.22 | .208–.311 | .070–.137 |
| Momentum Muon, four states | 11.57–15.82 | .573–.732 | .343–.461 |
| Momentum damped GN, four states | 2.23–2.40 | .379–.515 | .131–.194 |

- Every one of 1128 off-diagonal entries is positive in every Muon/PD/K-FAC
  Gram. PSD alone would not imply this. Positive coupling occurs across kinds
  and across layers, not only within attention or one special group.
- For Muon/PD directions the top normalized loading has one sign on all 48
  matrices and participation 32.7–38.5 out of 48. V carries 18.8–22.2% of its
  squared loading mass; up and down often carry more. There is no isolated V
  or head channel in these data (the head was not included).
- First-layer share of the leading *raw* loading ranges 31–82% for these
  directions. After diagonal normalization it is 10–13%, close to one-eighth.
  Thus the conspicuous raw first-layer channel is primarily magnitude, not
  a unique correlation pattern.
- With fresh Muon/PD input, average correlations remain .48–.51 for the same
  kind across layers, .52–.59 across kinds within a layer, and .38–.41 across
  both different layers and different kinds. Most pairs belong to the last
  group; the phenomenon is broad.
- One leading component approximates much, but not all, of the off-diagonal
  pattern: residual Frobenius norm is 20–31% of the original for normalized
  Muon/PD Grams. Leading normalized trace share remains well below 1.

The leading raw component accounts for 91–98% of the selected combined
direction's GN curvature. This does **not** mean that almost all response
energy lies in one mode: the components add along its common sign when the
48 pieces are summed. It also does not mean that the full GN has low rank.

## The common component is also where the measured descent lies

For Muon/PD directions, squared alignment between the leading normalized
loading and the normalized slope vector a_i/sqrt(Q_ii) is .968–.993. The raw
leading coefficient mode contributes 84–96% of the chosen combined direction's
first-order slope. Consequently, removing a top component because it carries
curvature also removes most measured descent in this selected-direction span.
That algebraic projection is descriptive; no modified step was selected or
evaluated on true loss.

A one-parameter descriptive fit of normalized off-diagonals to the outer
product of normalized slopes leaves 25–34% relative Frobenius residual for
Muon/PD. This is consistent with broad shared descent overlap. It does not
identify the corresponding logit-space response, because only inner products
and slopes survive in the archive. In particular, matching the coefficient
pattern to slopes does not prove that each token's tangent aligns with its
own loss gradient.

The pattern is not a sampling accident at the coarse coefficient level:
momentum-Muon leading normalized loading cosines between the two existing
score halves are .99979–.99991, and off-diagonal correlations are .9959–.9989.
These are repeatability measurements at fixed weights and directions, not
independent trajectory replications. Fresh-direction Grams lack saved halves.

## An analytical reason coherence need not determine a new allocation

Consider the limiting coefficient-space model

    R_model = D + alpha u u^T,   b = c u,
    R_model^(-1) b = c D^(-1)u / (1 + alpha u^T D^(-1)u).

Here b is the slope vector in normalized-piece coordinates, u is a shared
response loading, and **D is the residual diagonal after the shared term is
removed**, not diag(R), which equals the identity. In this exact limit the
shared response changes only the overall scale of the solution relative to
D; it does not change its relative coefficient allocation. Large coherent
curvature can therefore coexist with little gain from coordinating matrix
scales. This supplies a possible connection to the archived scaling failure
and subsequent finding that most of the GN gap was within matrices.

As a bounded check, fit u to the leading normalized mode on one 64-sequence
half, fit alpha only against that half's off-diagonal entries, and set
D_ii=1−alpha u_i^2. Predict the other half's normalized off-diagonals without
refitting u or alpha. All four momentum-Muon cases, in both bank directions,
leave 20–31% relative off-diagonal Frobenius residual, almost the same as the
fit-half error. Fitted residual diagonals are positive, with entries spanning
.17–.99. The score-half slope's relative residual off the fixed u is 8–19%
when its amplitude is fitted for this orientation-only summary, or 9–22% when
both the orientation and amplitude are carried from the fit half. Each half
uses its own measured diagonal to define normalized coordinates; this checks
the correlation pattern, not an unnormalized curvature estimator.

**This is not an exact rank-one-plus-diagonal fit or an optimized update.**
The retained 20–31% off-diagonal error and 8–19% slope residual can matter
substantially after inversion. In particular, suppressing a large common mode
can amplify the relative importance of a small orthogonal slope component.
High coefficient alignment alone therefore does not prove that the actual
optimal allocations agree. No 48-parameter coefficient optimization is used
to turn this limiting identity into a scientific result.

## Existing interventions limit the interpretation

The report numbers below were checked against saved JSON, not copied only
from the historical prose.

1. **Full GN decorrelates pieces, but low coherence is not an optimization
   criterion.** GN on stale momentum has lower coherence than Muon and still
   loses the archived one-step comparison. K-FAC is also less coherent than
   Muon while worse in the original scoring. PSD concentration and correlation
   cannot establish that removing shared response will help.
2. **Global stiff-mode deletion was already tested.** On PD 1M step 500 with
   a fresh 4M gradient, removing the top 32 global Ritz directions and adding
   Newton inside that space moves Muon from .378 to .479 of the damped-GN
   reference, but PD alpha 1/2 from .739 to .411. Top 16 reaches .785 for PD;
   the top 32 Newton step alone gets .102. These outcomes do not contradict
   the direction-Gram concentration here: a coherent combination of 48
   selected responses is not a top eigenvector of the full parameter GN.
3. **Most of the one-step GN gap was within matrices.** At Muon 1M step 500,
   independent exact-matrix blocks score .020086 versus full .023251; per
   kind .022596 and per layer .021881. The historical claim that cross-matrix
   redundancy was the core limitation was subsequently withdrawn.
4. **Sequential correction depends on freshness.** One complete forward or
   reverse GN sweep gets .023097/.023484; restricting the sweep to within
   layers gets .020552. Staged Muon with fresh 4M input reaches .024814 versus
   .014483 plain; with next momentum it gets .003183 versus .004455 plain.
   Staged PD likewise falls from .004464 to .002811 with momentum. These are
   cross-fitted local GN-model decreases, not training rates. They argue
   against treating generic positive interaction as permission for a new
   layerwise staging recipe.

Sources for the interventions are recorded under `prior_interventions` in
`analysis.json`: `blockgn/M1M_g4M_*.json`, `blockgn/gsmap_M1M*.json`, and
`gn3/PD_a0.25_lr0.01_s260925_ada_step000500.json`. The per-matrix scaling
failure is recorded in the original `OBSERVATIONS.md`03:40 entry; it was not
recomputed here.

## Smallest next discriminator

The archive supports a **generic-overlap alternative** to interpreting the
observer's positive body/aux or constant/centered finite interaction as a
special architectural defect. It cannot establish that alternative, because
these are different states, different direction families, and different
parameter scopes. Finite true-loss interaction also contains effects absent
from base-point predictive GN.

The next useful measurement should identify a function-space response, not
produce another coefficient eigenspectrum. If the observer continues the
already-defined value split, the smallest direct follow-up is to split its
eight V pieces by layer and retain their constant/centered logit tangents
*or sufficient cross-inner-products with the chosen constant-route tangent*.
Then ask, on a separate score bank: does projection onto the fixed shared-value
response remove most positive cross-layer curvature **while leaving substantial
first-order descent**, or does it remove descent in nearly the same proportion?
The second outcome supports generic descent overlap and ends a channel-removal
proposal; the first would locate a more specific inefficiency worth checking
with finite loss. The response used for projection must be fixed independently
of the score bank; it cannot be the bank-fitted leading mode. Retain per-layer
diagonals, mixed terms and finite loss checks at the existing scale. This is a
proposal, not an authorized launch from this report, and it still would not
establish a training improvement.

An even smaller bookkeeping check is available first: compare the already
measured 2×2 GN cross term of the value split to its finite loss interaction.
Agreement would show that base-point functional overlap explains the local
effect; disagreement would direct attention to path curvature or nonlinear
propagation before any geometry prescription. This report does not recompute
those separate observer measurements.

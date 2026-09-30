# Independent review of auxiliary-partition attribution

2026-09-28. Read-only scientific review before the proposed model probe. Sources:
`RESEARCH_GUIDE.md`, current `RESEARCH_STATE.md`, observer `STATE.md`, and
`body_aux/{PROTOCOL.md,REPORT.md,result.json,PROVENANCE.json,probe.py}` plus its
frozen model definitions. No model calls, GPU use, scheduler changes, training,
or changes outside this peer directory were made.

The proposed bounded experiment is scientifically useful. It can identify
which blocks of one **actual saved displacement** participate in the existing
positive body–auxiliary loss interaction. It cannot establish that auxiliary
learning is wasteful, that the head learning rate is wrong, or that suppressing
a block improves training. The relevant competing explanations are a largely
pairwise body–head effect, effects involving embeddings or norm gains, and a
conditional multigroup effect that resists a single-block attribution.

## Minimum sufficient design

Use all sixteen corners of Body/Head/Embeddings/Norm at the same saved
Muon4M183→184 step. Full factorial measurement is justified here because a
four-corner decomposition of each block alone would discard exactly the
auxiliary–auxiliary and higher-order conditions that can change the answer.
No learning-rate fit, extra coefficient grid, or repeated gradient-angle probe
is necessary for the attribution question.

First reproduce the four original corners on the stored sixteen token inputs
with the existing frozen FP32 eager forward and scoring convention. Verify
base and full directly against checkpoint loading and confirm the exact
parameter partition. If reproduction fails beyond the declared numerical
tolerance, stop scientific interpretation and preserve the failure.

The old sixteen sequences are discovery/reproduction data. Two fixed new
four-sequence banks are a useful check of fresh-input consistency for this
**selected state and selected interaction**. Freeze their offsets, token
arrays, source identity, corner order, contrasts, and stop rules before scoring.
Audit token-range disjointness from training exposure and the original banks;
do not say globally never-scored unless that stronger statement was checked.
The separate character-surrogate confirmation data must remain untouched.

Do not pool the old and new samples into a nominal confirmation result. Report
all four bank means and paired sequence values; the new eight have no credible
population precision and are not new optimizer trajectories. Contiguous
sequences may also share documents. No automatic sample expansion after a
reversal or near-zero result is justified.

## Exact contrasts to preserve

Let L(S) be the loss after advancing only groups in S and define the anchored
finite contrast

    m(T) = sum_{U subset T} (-1)^(|T|-|U|) L(U).

Keep the four single-group, six pair, four triple and one quadruple contrasts
for every sequence and bank, including their signs. The previously observed
body–auxiliary interaction is exactly

    I(B, HEN) = m(BH) + m(BE) + m(BN)
             + m(BHE) + m(BHN) + m(BEN) + m(BHEN).

Reconstruct that sum against the direct four-corner result. Also reconstruct
L(BHEN)-L(B) and L(HEN)-L(empty). These target quantities distinguish the
observed harmful conditional auxiliary change from the favorable auxiliary-only
change. A large m(BH) is insufficient by itself if the higher terms cancel it.

For a readable summary, give each auxiliary group's marginal loss effect both
alone and conditional on the full remaining step. These are exact comparisons
already contained in the factorial. Retain their context dependence. There is
no unique additive ownership of mixed terms; assigning half of a pair or one
third of a triple is an attribution convention, not a mechanistic discovery.
Signed absolute effects are preferable to percentages when terms cancel.

## Parameter and scope checks

The recorded model is bias-free, has untied input/output embeddings, and uses
RMSNorm plus Q/K normalization. Its actual head key is `head.weight`, not
`lm_head`. The complete partition should be:

- Body: the existing 48 Q/K/V/O/up/down matrix weights.
- Head: `head.weight`.
- Embeddings: `embed.weight` and `position.weight`.
- Norm: 33 gain tensors, comprising 16 residual-stream gains, 16 Q/K gains,
  and the final `norm.weight`.

Assert exact disjoint coverage by names and actual tensor storage; do not
silently interpret an unexpected residual tensor as a norm. Freeze all buffers
and model mode. Report group parameter counts and actual displacement norms,
but do not compare raw norms as functional amplitudes. These saved parameter
changes include each optimizer's full realized write, including decay and
rounding; they are neither raw momentum nor adaptive directions in isolation.

The norm group is intentionally heterogeneous. A result implicating it does
not identify Q/K gains or final normalization separately. Attribution also
depends on the chosen coordinate partition, especially for coupled final-norm
and head transformations; it is not an architecture-invariant statement.
Further subdivision would be a new question, not an automatic repair cascade.

## Optional coefficient slopes and predictive GN

The factorial alone answers the attribution question. If the predeclared total
cost gate leaves room, fixed-base coefficient slopes and the 4×4 predictive GN
on the new eight inputs add a useful local functional interpretation. With
actual block displacements d_i, define s_i = grad(L) dot d_i and
K_ij = (J d_i)^T (diag(p)-p p^T) (J d_j), averaged over scored tokens. Use
the exact predictive categorical curvature, not sampled-label curvature.

For L(c) ≈ L(0)+s^T c+0.5 c^T K c, a finite pair contrast is predicted by
**K_ij**, not 2 K_ij. The total body–auxiliary interaction is predicted by
K_BH+K_BE+K_BN. Preserve diagonal amplitudes, signed cross terms, slopes and
correlations; high correlation alone is not redundant or expendable learning.
The directions may share a favorable response and jointly overshoot locally.

Qualify autodiff/JVP numerics separately from the scientific result, check
symmetry/positive-semidefiniteness to numerical tolerance, and report any
endpoint-rounding difference if the coefficient path uses before+c*(after-before)
while corners load saved tensors exactly. No fitted step is needed.

A mismatch between finite loss contrasts and the predictive GN does **not**
uniquely identify higher-order nonlinear interaction. It combines the true
Hessian's logit-second-derivative residual with curvature variation over the
finite displacement (and numerical error). Higher-order factorial terms are
finite mixed contrasts, not pure high-order derivatives. Without measuring the
true coefficient Hessian, retain the combined interpretation. Adding that
Hessian is not required for this branch and should not expand the cost limit.

## Budget and decision gates

CPU only, at most two numerical threads, a ten-minute total ceiling including
qualification and optional derivatives is plausible but must be measured. The
prior four-corner run's 80 seconds included 64 forward/backward evaluations;
the new factorial has 384 forward evaluations, so previous timing alone does
not certify the new budget. Forecast from a bounded first-sequence check and
write partial results incrementally. If the derivative projection would exceed
the remaining ceiling, omit that optional stage and retain the full factorial.
Do not solve a cost failure by lowering sample counts after inspecting effects
or launching GPU work.

- If the old four corners fail reproduction, stop attribution pending a
  numerical/execution explanation; preserve the original and failed outputs.
- If one body–auxiliary pair consistently dominates in the old and both new
  banks and the mixed remainder is small in absolute terms, report a local,
  chosen-partition attribution and its exact remainder.
- If ranking/signs change, or higher terms are material, report conditional
  multigroup or sample-sensitive interaction without selecting a winner.
- Agreement with predictive GN supports shared local predictive response as a
  description; disagreement leaves residual/nonlocal curvature unresolved.
- Either outcome completes this point study. No head-LR recommendation,
  architecture intervention, sample-size escalation, or training-rate claim
  follows automatically. A proposed optimization change would need a distinct
  decision argument and appropriate trajectory evidence.

Recommendation: proceed with the fixed factorial after recording these
comparison and resource gates. The optional derivative stage is secondary;
its absence would not invalidate a complete finite-step attribution.

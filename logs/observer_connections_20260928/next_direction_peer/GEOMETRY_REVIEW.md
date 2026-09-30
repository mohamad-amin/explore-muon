# Independent choice: momentum may change the model's implicit angular learning rate

2026-09-28. Independent high-level discussion after the early-split result.
Read the current guide/main state, observer state/synthesis, early-split report
and independent review, momentum×LR report, momentum-map report and prior
gauge/normalization review. The only file written is this review. No model
construction, forward, gradient, GPU or training job was used.

**Recommendation: do one bounded saved-weight test of normalization-induced
angular step adaptation.** Do not reopen useful early mixed nonlinearity,
mean-channel attribution, a static stiff-energy explanation, or a generic
normalizer search. The early-split finite-benefit premise failed; preserve the
relative improvement it did show without inventing a subgroup mechanism.

## The connection worth separating

The existing beta×LR analysis shows the short-momentum benefit survives a
43% increase in nominal body arc. This excludes "more prescribed body norm"
as its simple explanation. It does not tell us whether changing LR actually
changes the rate at which scale-invariant features rotate. Q and K are followed
by per-head RMSNorm, so increasing a head's weight radius alone does not
change its normalized feature map, apart from epsilon effects. A fixed-norm
update applied to a growing weight radius has a declining relative angle.

Longer momentum can create outward radial motion: an average of gradients
formed at previous points need not remain tangent to the current normalized
weight sphere. The nonlinear polar/whitening map can add its own radial
component, so the linear heavy-ball calculation is not a proof for this
optimizer. The relevant observable is available exactly from saved writes.

Consequently, the same nominal update norm can mean different actual feature
turns under different beta values. Also, if a higher LR creates proportionally
larger radii, changing LR may leave late relative turns similar. That would
reconcile an apparent step-indexed phase with the prior LR insensitivity
without making optimizer step count a causal clock. It could connect three
existing observations: stronger whitening likes a shorter window, short beta
wins early but is caught later, and a later momentum increase can help.

This is a candidate endogenous learning-rate effect, not a new theorem or an
identified mediator. In particular the 1× warmup failure and 2× success show
schedule/horizon sensitivity; they do not establish a universal 100–150-step
switch or prove that radial adaptation causes that sensitivity.

## A small necessary-condition screen already gives a reason to check

For orientation only, I mmap-read the three already-completed 2× SOAP-PD
trajectories at common kept steps 37/92/165 and their saved next weights.
Only Q/K weights and their RMS gains were touched. For each of 8 layers ×
2 projections × 8 heads, I treated its 64×512 row block as one vector and
computed the exact written-step chord between normalized weights:

    radius = ||W_h||_F
    chord  = ||W_h,next / ||W_h,next||_F − W_h / ||W_h||_F||_F.

The per-head split matters: normalizing a whole Q/K matrix would still count
changes of relative head scales, each of which has its own RMS symmetry.
Medians across 128 heads (not independent replicates):

| Step | beta .9 radius / chord | beta .8 radius / chord | warmup radius / chord |
|---|---:|---:|---:|
| 37 | 5.02631 / .043765 | 4.52018 / .048813 | 4.58100 / .048103 |
| 92 | 6.40414 / .034633 | 5.59690 / .040020 | 5.89455 / .037404 |
| 165 | 7.58052 / .029513 | 6.39561 / .034689 | 7.23232 / .030844 |

Thus beta .8 turns these normalized feature weights about 12–18% farther per
saved step than beta .9 despite their identical configured body norm budget.
The warmup is intermediate and approaches beta .9 late. That pattern is
compatible with endogenous angular-step annealing. It does not show that Q/K
carry the winning loss effect, that a larger turn is better, or that the
unchanged-architecture network has a matching prediction-space distance.

The written-update radial cosine is also positive: medians at step 37 are
.149/.089/.101 (constant .9/.8/warmup), and at 165 .041/.033/.053. Therefore
**do not assume pure random-walk radius growth** r≈eta sqrt(t). The radial
cross term can be several times the squared-step term. The exact accounting
below must decide whether that diffusion approximation is useful here.

A second small check did not motivate a separate temperature branch: median
RMS of the learned QK gain product changes approximately 1.01→1.09→1.21
through these states and differs little across the three trajectories. This
coefficient proxy is not measured attention entropy or logit temperature.

This prescreen was exploratory and has no new inferential threshold. Its
purpose is to establish that the proposed quantities are available and that
the effect is not obviously absent. It is not the full LR discriminator.

Sources, under `logs/muon_spectra/`, each `scientific/kept/step{t:06d}.pt` and
`step{t+1:06d}_weights.pt`:

- `soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada`
- `soaudit_horizon16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.8_s260925_ada`
- `soaudit_momwarm16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada`

Tensor keys are `model['blocks.{l}.attn.{q,k}.weight']`, reshaped `(8,-1)`;
float64 vector arithmetic, CPU two threads. The displayed radial cosines use
the total saved write, including decay and rounding. The recommended full
analysis separates those pieces.

## Smallest discriminator that changes the decision

Use **the existing four-arm beta×LR factorial**, already paired and qualified
in `../momentum_lr/`: beta {.8,.9}, body LR {.028,.04}, same seed/hardware,
with retained steps 9/46/83 and their next weights. Do not add an LR, seed,
model replay or hand-selected point. Reuse its provenance checks and frozen
sources; check actual head shapes and matching next-step numbers.

For Q/K heads, retain all layers/heads and compute:

1. Radius, exact normalized chord, and per-head written-step norm.
2. The exact identity

       ||W+Delta||² − ||W||² = 2<W,Delta> + ||Delta||².

   Also separate adaptive displacement from the recorded scalar decay using
   the actual next-step LR and decay convention. Keep signed terms rather
   than dividing by a small net radius change.
3. Radial and tangential displacement relative to the incoming weight,
   together with the next-step direction norm allocated to each head.
   Per-matrix norm matching does not imply equal norm per head.
4. Matched LR and beta contrasts, reporting per-head distributions and
   layer summaries. Keep Q and K separate as a diagnostic; do not average
   over all model matrices and call the result scale invariant.

No fitted power law, time warp, best collapse or multiple clock sweep is
needed. The concrete comparisons are:

- Does a 1.4286× LR increase produce a substantially smaller late increase
  in normalized Q/K turns because the radii rise with LR? Compare both
  beta values and both later checkpoints. If angle ratios remain near
  1.4286 rather than near one, the proposed radius compensation does not
  explain the prior LR insensitivity in this bracket.
- Does shorter beta consistently have smaller radii and larger turns at
  the same LR? If its relative turns are indistinguishable or the result
  is confined to a few heads, this mechanism is a poor explanation of
  the broad momentum result.
- Is the radius change primarily accumulated step-squared energy or
  coherent radial drift? These imply different stories: the former a
  geometry-induced annealing scale, the latter a momentum/map-dependent
  radial feedback. Do not label both "diffusion."

Interpret these as falsifiable structural predictions, not evidence that a
loss improvement was caused by Q/K angles. Before executing, the root should
fix a small effect threshold/range from the measurement's precision and
scientific relevance, or report only the numerical contrasts without a
pass/fail label. Do not retroactively choose a favorable definition of
"collapse." A <=2-minute, <=2-thread weight-only pass is sufficient; no
model calls or scheduler resources are justified for this screen.

## Consequences and stop rules

**If radius compensation is absent:** close this candidate as an explanation
of the LR-insensitive phase. Do not respond with a norm-floor, renormalization,
head-temperature or body-normalizer intervention. A changed direction or an
auxiliary/statistical transient remains possible.

**If the predicted contrast holds:** revise the conclusion from "the phase
follows optimizer steps" to "configured steps and LR do not identify the
actual relative feature-motion clock; normalization creates an endogenous
rate." That is a substantive correction and a reason to measure state motion
in later comparisons. It still does not justify changing normalization or
beta. A future causal discriminator would need to hold measured angular
motion fixed while changing memory, and that would require a separately
reviewed design; it is not proposed for execution here.

**Prior negative evidence still applies.** Track 3's hyperball/norm-control
experiments and magnitude-matched Muon do not support a generic scalar
normalizer fix. Their decay-equilibrium regime differs from the current
short, weak-decay trajectory, so they do not prove this phase connection
false either. V/O gauge tests ruled out the broad covariance explanation
for shaped decay; Q/K's architectural scale symmetry is narrower and does
not restore that abandoned story. The early mixed-response failure stays
closed. No tiny-surrogate confirmation data were read.

## Agreed bounded criteria after availability check

The root confirmed all four factorial arms retain 9/46/83 and next 10/47/84.
I agree with the following deliberately demanding necessary-condition screen,
fixed before reading those twelve states:

- **P1:** at steps 46 and 83, median paired high-LR/low-LR head-chord ratio
  <1.20 for each beta, reporting Q and K separately (nominal LR ratio 1.4286).
- **P2:** at steps 46 and 83, median paired beta-.8/beta-.9 head-chord ratio
  >1.10 at each LR, again Q and K separately.

Use medians of the matched head ratios, not a silently substituted ratio of
pooled norms or medians. Preserve every head and layer, including counter-signs.
These are scientific-effect cutoffs, not sampling-error confidence bounds;
heads and times do not supply independent seed replication. The early step 9
and all 48 matrix-kind summaries are context, not alternative acceptance tests.

A failed conjunction closes the broad proposed compensation account at this
declared scope, without a fitted time warp or more states. It does not prove
that normalization never changes a learning rate. A passing conjunction makes
a relative-motion explanation plausible, not causal. This review is complete;
the root can write the bounded protocol without another approval cycle.

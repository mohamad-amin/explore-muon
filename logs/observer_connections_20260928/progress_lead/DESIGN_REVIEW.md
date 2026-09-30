# Design review: distinguish a secant lead from resolved trajectory catch-up

2026-09-28. Read the three completed warmup-pair descriptions, their prior
qualification, and the proposed readout. No new trajectory reduction, model,
evaluation, checkpoint or training call. Only this review is written.

**Verdict: go as a bounded descriptive cadence/sensitivity audit. Do not
promise that the fixed-bank records can decide genuine catch-up versus a
constant head start.** The proposed data discipline is appropriate, but the
available evaluation cadence imposes a stronger identification limit than
the number of displayed target levels suggests.

## Define the quantity and expose its resolution

For each of the three pairs and fixed loss level ell, define

    lead(ell) = crossing_control(ell) − crossing_warmup(ell).

Positive means warmup reaches that recorded loss earlier. Use precisely
{4.25,4.30,4.35,4.40}, only where both models bracket the level. No alternative
levels, extrapolation or clipping of negative leads. Compare lower versus
higher loss levels to describe whether the lead erodes as progress advances;
do not infer catch-up merely from a shrinking vertical NLL gap.

However, **100 and 150 are the only fixed-validation observations inside
the common-beta/common-LR phase**. For targets bracketed there,

    crossing_m(ell) = 100 + 50*(L_m(100)−ell)/(L_m(100)−L_m(150))

is a secant convention. The paired lead is therefore an affine function of
ell derived from the same four endpoint scores. Four target levels are not
four independent observations of learning dynamics, nor evidence that the
intervening loss curve is linear.

Retain each model's original crossing bracket. Under a stated monotone-
crossing assumption, [l_c,u_c] and [l_w,u_w] imply a lead interval
[l_c−u_w, u_c−l_w]. If both brackets are [100,150], that interval is
[−50,+50]. Monotonicity alone cannot establish a precise few-step lead or
its trend. Across two levels within the same interval, joint monotonicity
bounds the lead change by approximately [−50,+50], not an independently
estimated error bar. These are conditional crossing brackets, not verified
first-passage times over the entire unseen trajectory.

The secants remain useful summaries, but the primary export must identify
them as such. A monotonicity violation or flat segment at a target has to be
retained explicitly; do not force an inverse by isotonic regression or choose
the most favorable crossing. Missing coverage remains missing.

## Cadence controls are worthwhile but cannot become certified bounds

The proposed omit-100 and omit-150 controls use the same target levels and
are good checks of how much the inferred lead depends on evaluation cadence.
Reconstructing the recorded 100 anchor from 50/150 and the 150 anchor from
100/184 also makes interpolation curvature visible.

Report errors for each arm and the paired contrast, in NLL and the clearly
specified secant-step conversion. Common interpolation error may partly
cancel between paired arms; differing slopes can prevent that cancellation.
Never divide both arms' errors by whichever slope gives a smaller uncertainty.

Neither control estimates a guaranteed error distribution inside 100–150:
one spans changing momentum and the other includes cooldown. Large changes
or sign flips demonstrate that a precise lead reading is fragile. Small
changes do not establish that unobserved curvature is small. Consequently,
the envelope of omitted-anchor estimates is a **sensitivity envelope**, not
a confidence interval, worst-case bound or tolerance that licenses a rate
claim. Preserve the original protocol/readout before extraction.

## Dense training curves: keep the useful evidence and its nuisance

Centered means of exactly 11/21/31 rows are reasonable fixed secondary
summaries. Require every contributing row to lie in 93–166. Their valid
center ranges are respectively 98–161, 103–156 and 108–151. Conclusions about
agreement across all three smoothers should use common coverage, rather
than compare one smoother's early endpoint with another's late endpoint.
Export the complete valid ranges and explicitly missing target crossings.

Training row t measures the incoming batch at W_(t−1); validation at t is
after update t. Show both conventions or shift training coordinates by one;
do not silently treat them as the same model state. This matters when leads
are only a few updates. The first common-coefficient update is 93; an
incoming-row convention includes W92 at that boundary, which must remain
explicit. All compared window rows should have the same full-batch token
counts; use the recorded counts and do not include the shortened endpoint.

The main nuisance is conceptual. Paired *vertical* training-loss differences
at a common step evaluate both arms on the same batch, so a shared data-
difficulty component largely cancels. Inverse loss crossings compare
**different** steps and therefore different batches. For example,

    observed_loss_m(t) = learning_curve_m(t) + data_difficulty(t)

can make an exactly constant latent head start appear to change when the
curves are inverted. Smoothing reduces rapid variation but not broad corpus
drift. The two SOAP seeds and the PD pair use the same deterministic stream,
so agreement across them does not independently eliminate this nuisance.
State-dependent responses to data also prevent assuming exact additive
cancellation.

For each smoother, retain all downward/upward crossings and flat target
segments. If there is more than one admissible crossing, flag ambiguity and
export its bracket set; do not silently choose a convenient first crossing.
A unique downward crossing within the displayed window still is not proof
of global first passage. Centered means also use future rows; they cannot
justify an online scheduling rule.

## Conservative scientific reading

I recommend separating three outputs rather than forcing a binary verdict:

1. **Fixed-bank resolution:** do the conditional validation brackets actually
   distinguish a shrinking lead from a constant one? With shared 100/150
   brackets, they normally cannot. Mark that primary identification unresolved.
2. **Secant estimate and cadence sensitivity:** report the inferred level-
   dependent lead and its changes under each predetermined omission, without
   presenting the extra curves as new measurements.
3. **Secondary training-pattern description:** call the pattern consistent
   with preserved head start, apparent erosion, or mixed/ambiguous only if
   that wording accurately reflects *all* three fixed smoothers and available
   pairwise coverage. A disagreement is not permission to select a smoother.
   Keep pair-specific differences; the two methods are not a crossed
   two-seed replication design.

A “genuine catch-up” or continuing-rate claim would require the fixed-bank
crossing uncertainty to exclude the competing lead behavior, or additional
independently justified assumptions about the intervening curves. This
protocol supplies neither automatically. Training-curve consistency can
strengthen a descriptive reading, but must not be promoted above its
changing-batch limitation. Likewise, nearly constant secant leads do not
prove both optimizers inhabit the same latent state trajectory.

The audit can still change the research direction: it can determine whether
the current precise 4–7-step narrative is well resolved by its artifacts,
and prevent a vertical-gap contraction from being mistaken for lost progress.
The original warmup endpoint gains remain established regardless. If cadence
and data nuisance leave the rate distinction unresolved, close this scalar
question at that limit—no fitted warp, preferred smoother, new evaluation
point or new training run follows from the outcome.

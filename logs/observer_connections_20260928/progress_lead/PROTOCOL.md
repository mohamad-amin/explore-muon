# Warmup progress lead: a bounded archive and resolution audit

2026-09-28. Read the root research guide/current state and verified live main
steps 2567578.551 and 2618555.221. This observer does not use those GPUs.
The preceding orientation turn completed a historical survey and verified the
new PD-top execution; it did not advance the open progress-lead reduction.
The present turn completes that outstanding analysis rather than repeating
the status. The broad observer goal remains active.

## Decision argument

The completed warmup comparisons retain smaller validation loss after their
momentum coefficients become equal. Their vertical loss gaps shrink. A shrinking
gap can mean catch-up, or the same horizontal lead on a flatter learning curve.
Main now describes a 4–7-step head start and supplies a training-curve crossing
helper, but no located three-pair readout restricts both smoothing and crossings
to common beta and constant LR. MAIN_COVERAGE_REVIEW.md records that coverage.

Compare all three completed 184-update, 16M pairs: PD seed260925 and SOAP-PD
seeds260925/260926, warmup .8→.9 versus each own constant .9. No comparator
selection by final loss; no extra seed or family. Initial/final scores were
already inspected in earlier work; this is a retrospective analysis with
readout fixed before reducing inverse curves, not a new prospective experiment.

DESIGN_REVIEW.md independently supports a descriptive cadence audit but warns
that only validation100 and150 fall in the common-beta/common-LR interval.
SOURCE_CLOCK_REVIEW.md independently checks indexing/source identity. Main's
endpoint criterion and all earlier outputs remain unchanged.

Primary interpretation is whether the recorded fixed-bank anchors can distinguish
constant, eroding or increasing progress lead. Secondary changing-batch curves
may describe an apparent pattern; they cannot by themselves resolve this
identification problem or establish a causal mechanism. If unresolved, close
this scalar archive route without finer fitting, new evaluations or training.

## Qualification and provenance

Read JSON/scientific source text only, hash every input; verify all six complete
184-step statuses/checkpoint sidecars, exact full step coverage, paired initial
hash/data/model/hardware/precision metadata, and per-step token/LR clocks.
Check recorded source hashes. The new PD and SOAP replication pairs have
identical sources. The original SOAP pair uses distinct revisions with inspected
inactive defaults; preserve that qualification from confidence_calibration's
PAIRING_REVIEW.md rather than claiming byte-identical training across revisions.

Training record t scores W_(t−1) on the incoming batch; validation t scores W_t
on the common fixed bank. Beta matches starting update93. LR first cools at167.
Conservatively use training rows94..166, thus states93..165, for all secondary
windows. The final partial batch is excluded. Retain complete original scalar
rows in CSV, with separate record-step and state-step columns.

## Fixed validation readout

1. At exactly loss levels {4.25,4.30,4.35,4.40}, calculate all bracketing linear
   inverse secants. Lead = control crossing minus warmup crossing. Primary
   crossings must have both bracketing endpoints in93..166. No extrapolation,
   isotonic regression or selected crossing. Export upward/flat/ambiguous
   crossings and missing coverage explicitly. A unique secant is descriptive.
2. Export monotonicity-conditional crossing brackets, and their lead difference
   interval. These are not sampling confidence bounds or certified first-passage
   times. The four targets sharing one anchor interval are not independent.
3. Predetermined cadence sensitivities: omit100, and separately omit150, from
   the original validation series; recalculate at the same target levels. These
   controls may span beta changes or cooldown and are sensitivity envelopes,
   not guaranteed error bounds. Reconstruct each omitted anchor from its adjacent
   remaining anchors: report NLL error and inverse-time error for each arm, plus
   the paired contrast. No favorite interpolation convention is selected.
4. Endpoint warmup score below all recorded control scores is right-censored:
   do not manufacture a time-to-match beyond184.

## Constructive identification check

The independent design review motivates a stronger mathematical resolution
check. At the already known100/150anchors, check the ordering
R100 > W100 > R150 > W150 for each pair. Construct three possible monotone
reference curves from knots (100,R100), (100+delta100,W100), (150,R150),
(150+delta150,W150). Define warmup L_W(t)=L_R(t+delta(t)), t in[100,150].
Use precisely delta(t)=5, 10−.16(t−100), and 1+.16(t−100): constant,
eroding10→2, and increasing1→9 updates. Check monotonicity, all four anchor
identities and exact level-crossing relations. All reference knots lie≤159,
inside the common LR phase. These are alternative mathematical completions
of unobserved curves, not model scores, fits judged by predictive performance,
or a proposed training law. They establish nonidentification if they pass.
They do not claim each completion matches unmeasured training-batch curves.

## Secondary training readout

Use centered arithmetic means of exactly11/21/31 consecutive training rows,
each entirely inside94..166. Export every valid center as model-state index
t−1, its full support, paired vertical gap, and local monotonicity violations.
At warmup state centers110,120,130,140, invert each reference mean over its
complete valid center range. Keep all crossing sets, censoring and missing
centers. These anchors lie in all three warmup center ranges; reference
crossings may still be absent and remain so. Also export common fixed loss
levels above when covered. All3smoothers/all3pairs are shown; overlapping
windows are not independent replicates and no p-values are computed.

For a concise pattern description, compare lead at140 with110 only when both
are uniquely bracketed, for every pair/width. Report the signed changes and
the full intermediate anchors, with no claim of equivalence from small changes.
Same-step paired gaps share training data; inverse crossings compare different
corpus offsets. All pairs share the ordered stream. Smoothing does not remove
unknown broad data-difficulty trends, state-specific responses or historical
optimizer-state differences. Centered means cannot specify an online schedule.

## Resources and stop

One CPU scalar reduction, at most2 numerical threads, no model/tensor/archive
loads, no GPU, no optimizer calls or scheduler mutation. Expected <30seconds;
hard bound two minutes for reduction/plotting. Outputs stay here. Meaningful
synthetic qualifications cover linear translation, nonmonotone crossing sets,
censoring and the explicit alternative-curve construction. Preserve failures.
Independent result review follows; then update observer STATE/SYNTHESIS only.

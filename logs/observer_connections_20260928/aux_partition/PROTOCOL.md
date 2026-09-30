# Attribute the measured joint body–auxiliary interaction

2026-09-28, before new model evaluations. Previous goal turn was progress:
the completed conditional TS experiment and its functional qualification
changed the interpretation and closed the immediate sampling branch. Main
Slurm state was checked at15:09; observer work remains CPU only.

## Missing premise and alternatives

At the saved Muon4M183→184 step, the actual auxiliary step improves loss by
itself but worsens the simultaneous body step on all16 measured inputs.
Which group accounts for that observation: head, embeddings or normalization
gains? A second distinction matters: ordinary overlap between useful
prediction changes versus a mixed change to the logits when both groups move.
Neither positive interaction alone nor parameter norms identify a defect.

## Fixed factorial

Same original checkpoint pair and actual endpoint displacements, no fitted
scale or LR. Four disjoint groups cover every parameter:

- B: all48 Q/K/V/O/up/down matrices.
- H: `head.weight` (untied readout).
- E: `embed.weight` and `position.weight`.
- N: all33 RMSNorm gain tensors, including final and Q/K norms.

Evaluate all16 binary corners. Reuse the old two8-sequence banks for exact
reproduction and attribution; they are discovery inputs. Add exactly two new
four-sequence banks at2,550,131,072 and2,650,131,072 in the same never-trained
stream. These are fresh-input consistency at a selected state, not independent
mechanism confirmation. No new sample or corner is chosen after results.

Preserve all Möbius contrasts. In particular the old interaction equals

    m(BH)+m(BE)+m(BN)+m(BHE)+m(BHN)+m(BEN)+m(BHEN).

Retain all seven rather than forcing three pairwise effects to sum to it.
Report conditional marginal effects as the other groups change.

## Exact logit/loss split

For B versus H/E/N at each of the four settings of the other auxiliary bits,
and for B versus all auxiliaries, form the four corner logits z00,z10,z01,z11.
Let zadd=z10+z01−z00. The observed finite interaction separates exactly into

    I_overlap = LSE(zadd)−LSE(z10)−LSE(z01)+LSE(z00),
    I_mixed = CE(z11,y)−CE(zadd,y).

Means are over token positions. I_overlap is label independent; I_mixed is
the loss effect of the nonadditive logit response. Their sum is the original
finite CE interaction. This is a finite identity, not a GN approximation or
an attribution of optimizer-state transport. For head/body with E/N fixed,
the mixed logit response is exactly D_head times the changed head input.
Retain sufficient scalar contrasts and per-sequence values, not enormous
copies of all logits. Original checkpoints/source/tokens permit reproduction.
Also retain the base-probability-weighted Gram of the four singleton finite
logit changes and their exact predictive KLs. These are finite-response
descriptors, not an infinitesimal GN or Hessian calculation.

## Qualification and resource boundary

CPU FP32 eager, two threads, no CUDA, optimizer, or training. Frozen helpers
from the earlier qualified body_aux probe. Corner weights must reproduce
base/full exactly. Recompute the old four-corner FP32 losses with error<=2e−6;
analyze logits/CE/LSE in FP64 to reduce cancellation, retaining both precisions.
Check all factorial reconstruction identities to1e−10 and logit/loss split
identities to1e−10. Qualify the head's bilinear identity at the first input;
report absolute/RMS error and its signal ratio, with numerical tolerance
max(2e−6 RMS, .02 times mixed-logit RMS). Do not relax these gates silently.
Because the head is untied and linear, its two settings may reuse the same
captured head input: eight hidden-model forwards plus eight alternate readout
matmuls evaluate all16 corners. Qualify the shortcut against direct head-only
and full-model loads with max logit error<=1e−5 before interpreting anything.

Time one full16-corner sequence and all contrasts before committing to the
remaining inputs. If projected total exceeds600s, stop and retain qualification;
also enforce a600s wall-clock cap. All files remain here and no main job or
source changes. This replaces the optional GN idea with a more direct finite
decomposition; it is not a reduced attribution scope.

## Decisions

If one group explains most interaction consistently, its label-independent
versus mixed-logit terms identify what needs a subsequent mechanistic test.
If higher-order terms dominate, preserve that dependence rather than naming
a single culprit. If the effect is mostly ordinary useful prediction overlap,
do not launch a staged/head-LR remedy. Any cross-state follow-up needs a fixed
prediction before its data are read; this state alone cannot establish a rate
or robust optimizer/architecture improvement.

Fresh reviews: `../aux_partition_peer/REVIEW.md` and
`../priority_review/REVIEW.md`; inventory: `../aux_inventory/REPORT.md`.

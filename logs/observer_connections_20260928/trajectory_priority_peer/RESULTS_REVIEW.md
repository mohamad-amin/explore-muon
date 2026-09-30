# Independent result review: momentum × body LR

2026-09-28. Read `momentum_lr/PROTOCOL.md`, `analyze.py`, `result.json`,
`source_differences.patch`, the preserved auxiliary-clock failure and the
rendered figure. Independently reduced original step JSONs and checked
source hashes using standard-library CPU arithmetic only. No model,
checkpoint tensor, GPU or scheduler operation; this is the only new file.

## Verdict

The extraction is correct and the bounded discriminator is complete.
The beta advantage remains close to .10 NLL at both body rates despite
42.86% more nominal body movement at LR .04. Later constant-LR interaction
and both measured validation interactions are small relative to that main
effect. The initial interaction is not zero: preserve the -.0211 contrast
over steps 24–43 instead of describing complete LR independence.

The plotted curves do not support a simple invariant beta-benefit curve
indexed only by cumulative body LR. The archive is adequate to stop this
specific body-dose-clock line. It does not establish an intrinsic step
clock, isolate auxiliary adaptation, or settle differential feedback. No
new run, fitted time warp or more elaborate clock search is warranted by
these results.

## Implementation and arithmetic checks

- Delta is consistently `loss(.8)-loss(.9)`: negative favors the shorter
  momentum. The interaction is `Delta(.04)-Delta(.028)`: negative means
  the beta .8 advantage is larger at the higher LR. Independent reduction
  reproduced every validation contrast and all five token-weighted window
  interactions exactly.
- All four arms have matching per-step `tokens` and `batch_tokens`, including
  the shortened final batch. Same-step training comparisons therefore pair
  the same training exposure. All configured/model/data/hardware identity
  checks are appropriate for the stated recipe comparison.
- Cumulative-LR indexing is correct: the training record uses the sum before
  the current update, and the validation clock includes it. The first
  training row starts at zero arc. I recomputed all retained pre-update
  cumulative values with maximum error zero. The weighted cooldown mean
  correctly treats the final shortened batch.
- The fixed constant-LR windows are complete and nonoverlapping. The
  nine-step centered smoother is display-only. Near step 83 it mixes some
  cooldown values into the display; the scientific window summaries do not.
  It must not be used for a precise transition-time claim.
- The norm-192 derivation is correct: per block, q/k/v/o/down each contribute
  squared norm 512; the tall up projection contributes 2048 after aspect
  scaling. Eight blocks give `8*(5*512+2048)=36864=192^2`. Archived sampled
  direction norms agree with the per-matrix formula to maximum relative
  error 1.41e-7. This is the normalized adaptive direction, not actual
  weight displacement including decay/rounding, net displacement, or
  function-space movement.
- The full nominal arcs independently reproduce as 466.04812767825115
  and 665.7830395403587, exactly matching between betas at each LR. The
  logged decay exposures are .02427334/.03467620; scalar shrink products
  are .97601565/.96591158. Body movement and decay exposure remain coupled,
  but this is far from the prior large-decay-equilibrium regime.

## Source and auxiliary-clock qualification

I independently hashed both the cohort frozen source and the per-run
`scientific/source` copies against the metadata's expected hashes:
**200 checks, zero mismatches**. This supplements `analyze.py`, which
records metadata hashes and source diffs but does not itself assert every
source-file hash.

The differences inspected are inactive in these arms:

- the new gradient prefilter is `none`, so the aliased `grad` has the same
  role as the old direct `parameter.grad`;
- added two-sided SOAP-statistics multiplication requires output roots,
  absent when the configured output power is zero;
- head whitening is disabled by alpha zero and head covariance tracking
  defaults false; the head remains ordinary Linear with auxiliary AdamW;
- related validation and metadata additions do not change the active rule.

This supports matched active behavior, not byte-identical source or
bit-identical full trajectories.

The auxiliary-clock gate repair is appropriate. Eight cross-rate log values
have maximum absolute difference 4.34e-19 from equivalent FP64 evaluation
orders; they pass the new four-ulp bound and have identical FP32 values.
Within-beta-pair clocks/config checks remain exact. The original failed
assertion and values are preserved, and no result, window or loss criterion
was changed. Describe this as a numerical logging qualification, not proof
that realized auxiliary updates are identical: their gradients differ with
the body trajectory.

## Supported reading, including the non-null portion

| Readout | Delta at .028 | Delta at .04 | Interaction |
|---|---:|---:|---:|
| Validation 50 | -.107564 | -.107913 | -.000349 |
| Validation 92 | -.101797 | -.105437 | -.003640 |
| Train 4–23 | -.028584 | -.021149 | +.007436 |
| Train 24–43 | -.057131 | -.078186 | -.021055 |
| Train 44–63 | -.096811 | -.092789 | +.004022 |
| Train 64–83 | -.080335 | -.079447 | +.000889 |
| Train 84–92 | -.095144 | -.095590 | -.000445 |

Higher LR changes the early path of the momentum advantage, then the curves
come close again. The validation checks corroborate the small interaction
near step 50 and at completion. This is not an equivalence test, and the
single seed supplies no confidence interval for the interaction.

The raw-step figure preserves similar broad timing of the main trough and
later recovery despite different cumulative-rate locations. Replotting by
cumulative LR visibly separates those broad features instead of producing
a common curve. That weighs against the simple claim that the reported
100–150-step phase is merely reaching a fixed nominal body arc. It does
not reject all movement-based explanations: changing LR changes directions,
curvature, weights, clipping history and realized auxiliary responses, so
equal nominal arc would not imply equal function-space progress.

The 92-step archive ends before the later catch-up observed in the 184-step
run. It cannot directly test whether the *end* of the advantage obeys a
different clock. Preserve that boundary and the .021 early interaction.
The proper stopping conclusion is that this cheap, declared comparison
found no strong late beta-by-LR dependence or simple cumulative-rate collapse
worth expanding into a new clock-fitting branch. The phase-dependent
training phenotype remains; its causal clock and feedback mechanism remain
unidentified. The already-authorized momentum-warmup comparison retains its
original practical criteria.

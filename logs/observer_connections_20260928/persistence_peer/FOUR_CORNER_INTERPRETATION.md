# Independent reading of the completed four-corner result

2026-09-28. CPU algebra and saved-gradient reads only; no additional model
calls or sample-size extension. Detailed calculations are retained in
`FOUR_CORNER_INTERPRETATION.json`.

## The sign reversal has good internal support

The symmetric cross-bank base/body cosine is −.681 and base/full is −.719,
versus same-bank values around +.78 and +.74. The sign is not an artifact of
symmetrizing one large negative and one positive dot product:

| Pair | Bank A base against bank B target | Bank B base against bank A target |
|---|---:|---:|
| Base/body | −.08246 | −.16679 |
| Base/full | −.11022 | −.20760 |

At the matrix level, 47/48 symmetric products are negative for both pairs;
both bank orders are negative in 41/48 matrices for base/body and 43/48 for
base/full. Base/aux products are positive in all 48, in both orders. This is
broad layer/kind support within these data, not independent replication.

The symmetric cross-bank four-corner Gram is positive definite, with
eigenvalues .000886, .002128, .1161, and .6183. Thus the ratios do not have an
obvious negative-norm or gross indefinite-Gram pathology. Two banks still
cannot supply a credible population confidence interval.

## A useful exact sample decomposition

Let W_ab be the average same-bank dot product, and S_ab the symmetric
cross-bank dot product. Then exactly

    W_ab − S_ab = ½ (g_A,a − g_B,a)ᵀ(g_A,b − g_B,b).

This is a bank-difference Gram, with no distributional assumption. For
base/body:

    +1.72065 = −.12462 + 1.84527.

The bank-difference vectors themselves have cosine .916 between base/body,
and .900 between base/full. Thus strongly persistent sample-specific
components can reverse the observed alignment sign when the same small bank
is reused at both states. Under exchangeable independent bank sampling, this
estimates shared-sample noise covariance. With these fixed corpus slices,
"sample-specific component" is the assumption-free description.

This demonstrates a plausible mechanism for the old probe disagreement on
the actual checkpoint pair. It does not quantitatively reconstruct the
different 128/8192-sequence estimates or exclude numerical-precision effects.

## Auxiliary scope changes magnitude, but does not supply the reversal

Applying corner contrasts on both sides of the cross-bank Gram gives:

| Body-gradient change | Cross-bank squared norm |
|---|---:|
| Body advanced | .61533 |
| Auxiliary advanced | .01602 |
| Full actual update | .76610 |
| Finite-step interaction | .004245 |

The body-only change and full change have cross-bank cosine .9979. Auxiliary
change mostly points in the same direction as body change (cosine .904),
amplifying rather than independently creating the reversal. The signs and
absolute Gram entries support this reading more directly than assigning a
percentage to an interacting decomposition.

Do not turn that result into "auxiliaries are unimportant." Auxiliary-only
loss improves by .00967 and .01225 NLL in the two banks, but adding auxiliaries
to the body update *increases* loss by .00891 and .00535. The loss interaction
is +.01858 and +.01760. All sixteen measured sequences have positive loss
interaction, and all have loss(full) > loss(body). That finite-step coupling
is consistent across the measured sequences even though auxiliaries are a
small source of body-gradient rotation.

## Next action

No extra model calls are needed to support the bounded finding: parameter
scope does not explain the sign contrast in these matched data, while shared
sample-specific gradient components are sufficient to reverse it. Preserve
the fixed two-bank result and its uncertainty. Do not tune sample count until
it reproduces an old estimate.

The most useful remaining check here is interpretive: ensure reporting
distinguishes empirical-bank alignment, cross-bank alignment, change norms,
and the loss interaction. The planned value-route diagnostic is a separate
mechanistic question and can now be considered on its own evidence, without
assuming that auxiliary updates dominate the observed gradient rotation.

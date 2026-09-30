# Independent dose-function results and mixed-logit identity review

2026-09-28. Read the completed probe and scalar-analysis implementations,
results and retained token arrays. Independently recomputed the KL ratios
and the proposed post-hoc interaction identity. No new model, sample,
rescaling or optimizer experiment. This is the only new file.

## Primary result

The probe completed all 256 declared score-forwards in 139.32 seconds.
Restored-base logits reproduce exactly, the maximum initial FP64/FP32 CE
discrepancy is 2.32e-6, and the null KL is zero. Source-helper hashes still
match. Independent recomputation of every reported forward/reverse/symmetric
ratio agrees within 4.45e-16.

The uniform predictive-shrink account fails in its declared strong form.
Half/quarter-power body forward-KL ratios at `(step9,beta.9)`, `(9,.8)`,
`(46,.9)`, `(46,.8)` are approximately .845, **1.085**, .716 and .684,
while the corresponding actual body parameter norms differ by only
.009–.036%. The early beta-.8 reversal occurs in both banks (about 1.020
and 1.154). Stronger whitening does not simply make every actual body
predictive move smaller at these own states.

Auxiliary forward-KL ratios are also heterogeneous: .461, .400, 1.586 and
.793. Thus identical auxiliary settings do not imply matched auxiliary
prediction changes. These are state-conditioned amplitudes with different
base predictive distributions, not common-state map or mediation effects.
Retain the reverse/symmetric divergence checks without choosing them over
the declared forward KL. Neither a larger nor a smaller local KL identifies
the training-rate cause.

The positive alpha-by-beta endpoint interaction remains a separate
training result. The local amplitude findings neither erase it nor identify
stiff filtering as its unique mediator.

## Exact post-hoc identity

The proposed algebra is sound. Write z0,zB,zA,zF for base/body/aux/full
logits at one token, p0=softmax(z0), and y for its one-hot target. For any
corner c,

    CE(zc,y)-CE(z0,y)
       = KL(p0 || pc) + (p0-y) dot (zc-z0).

Consequently, with the saved finite-loss interaction

    I = CE(zF)-CE(zB)-CE(zA)+CE(z0)

and divergence contrast `K = KL_F-KL_B-KL_A`,

    J = I-K = (p0-y) dot (zF-zB-zA+z0).

Averaging over the fixed tokens preserves the identity. J is the
base-residual-weighted mixed-logit response. It is invariant to arbitrary
per-token constant shifts of each corner's logits, because p0-y sums to
zero. A nonzero J therefore establishes label-relevant nonadditivity
beyond the irrelevant softmax logit gauge.

The scalar arrays suffice to derive J without another forward. This is an
exact identity-based post-hoc interpretation, not an independent direct
measurement of the full mixed-logit vector or a preregistered endpoint.
Its magnitude here is far above the retained forward/CE numerical errors.

## Signs and banks

| State | Mean I | Mean K | Mean J | J: bank 0 / bank 1 | Positive J sequences |
|---|---:|---:|---:|---:|---:|
| Quarter, beta .9, step 9 | +.08715 | +.07152 | +.01563 | +.01644 / +.01482 | 7/8 |
| Quarter, beta .8, step 9 | +.06136 | +.02983 | +.03153 | +.03534 / +.02773 | 8/8 |
| Half, beta .9, step 9 | -.04674 | -.01399 | -.03275 | -.03274 / -.03276 | 0/8 |
| Half, beta .8, step 9 | -.03529 | -.00559 | -.02970 | -.03019 / -.02922 | 0/8 |
| Quarter, beta .9, step 46 | +.02247 | +.02198 | +.00049 | -.00093 / +.00192 | 5/8 |
| Quarter, beta .8, step 46 | +.03109 | +.02816 | +.00293 | +.00267 / +.00318 | 7/8 |
| Half, beta .9, step 46 | +.02959 | +.02267 | +.00692 | +.00575 / +.00809 | 8/8 |
| Half, beta .8, step 46 | +.02530 | +.02334 | +.00195 | +.00150 / +.00240 | 7/8 |

At step 9, I itself is positive on all eight inputs for both quarter-power
arms and negative on all eight for both half-power arms. At step 46 it is
positive on all eight inputs for all four arms. The half-power early
mixed-logit term is negative on every input at both betas; its sign is
positive on average later. The quarter-power beta-.9 early exception in
one sequence, and its late bank reversal, must remain visible.

These are consistent signs on the fixed panel, not eight independent
training replications or a significance test across states.

## Crucial distinction from the earlier overlap decomposition

The earlier `aux_partition` study evaluated the additive-logit counterfactual
`zadd=zB+zA-z0` and measured

    I_mixed = CE(zF,y)-CE(zadd,y).

That is **not J**. Their relationship is

    I_mixed = J + KL(p0 || pF) - KL(p0 || p_add).

The last KL is not retained in the new scalar arrays. Likewise K is not
the earlier pure additive-prediction-overlap term: it combines that overlap
with changes involving the mixed response. Neither the old 96–98% overlap
fraction nor its complement can be recomputed or reversed from J and K
alone. A large base-linear mixed term can be offset by the additional
nonlinear KL difference when that response is applied after zadd.

Therefore the new result shows **substantial early label-relevant
nonadditivity**, but does not establish the exact finite-loss usefulness
or fraction attributable to that mixed response along the previous path.
Do not label J as "the nonlinear transport loss" or K as "pure overlap."

## Scientific reading and decision

This is a legitimate stage-dependent qualification of any extrapolation
from the late Muon4M ordinary-overlap result. The old conclusion was
already scoped to a selected state and remains intact. The new half-power
PD early states show a different local interaction sign and a material
mixed-logit response at the base residual. Later states in the same arms
have positive interaction again. This is new function-level evidence worth
preserving, without declaring a universal phase law.

It does not explain the shorter-momentum training advantage: at step 9,
beta .8 actually makes J less favorable in both alpha families, despite
its better training endpoint. That alone cautions against making this
one-step term a rate mediator. The source is also not localized to head,
embeddings or norms; the new partition deliberately keeps all auxiliaries
together.

Report the post-hoc identity and signs as a distinct observation. Do not
invent a common-token explanation, transplant the earlier late-state
closure universally, or conclude that freezing/staging an auxiliary group
would help. The primary amplitude discriminator is complete. Any future
mechanistic experiment would need its own narrowly stated premise; no
additional samples, layer decomposition or model call follows this review.

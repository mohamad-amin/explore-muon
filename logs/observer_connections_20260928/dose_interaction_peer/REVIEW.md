# Review: whitening-dose interaction and a bounded predictive-step discriminator

2026-09-28. Read current guide/state, the 16:45 dose entry, the observer's
momentum-by-LR and TS functional qualifications. No model, GPU, job or
checkpoint-tensor call. Root is checking source/trajectory pairing and
availability of actual next weights. This is the only new file.

## Preserve the positive result, narrow the mediator claim

The new result is stronger than a correlation across existing optimizer
states. At a common PD body LR .028, changing alpha changes the benefit
of beta .8 versus .9:

- alpha .25: 4.655709 -> 4.569439, approximately -.08627;
- alpha .5: 4.6711 -> 4.5475, approximately -.1236.

The difference-in-differences is about **-.0373 NLL**, and the better alpha
reverses between momentum choices. Subject to root's pairing checks, this
is positive evidence for the predeclared dose prediction and a real
whitening-strength-by-momentum interaction in this one-seed recipe. It
supports tuning these choices jointly.

It does not identify *stiff oscillation filtering* as the mediator. Alpha
also changes the functional magnitude/orientation of each step, the state
the optimizer reaches, clipping history, auxiliary responses and subsequent
curvature. The intervention changes that whole map, not only one measured
oscillatory component. The Muon alpha-zero point also uses a different body
LR and different clipping history, so the common-LR two-PD factorial is
the cleanest primary interaction; the three-point monotone story is broader
recipe-level evidence.

The correct distinction is: **the predicted interaction is confirmed at the
tested operating point; spatial filtering of stiff oscillations remains a
supported candidate explanation, not an identified causal mediator.**
This should not be diluted into "nothing was learned" or upgraded to a
complete mechanism claim.

## Endorse one function-level probe, with a precise question

The proposed CPU forward probe is worth doing if it asks:

**At actual saved steps from the dose factorial, does equal nominal body
parameter movement conceal a systematic change in predictive amplitude,
and is that change carried by the body or by its interaction with auxiliary
updates?**

This is a concrete bridge from the new training intervention to quantities
the model actually changes. It can expose a simpler functional-step-scale
competitor before another stiff-mode story. It will not by itself settle
mediation, differential feedback or a training-rate explanation.

The TS qualification shows why the bridge matters: matched parameter
norms concealed a 2.27–2.41x difference in local prediction-response norm.
The momentum-by-LR result excludes a generic explanation in terms of more
nominal body arc, but says nothing equivalent about predictive arc. The
new dose factorial gives a distinct matched recipe in which to examine
that missing premise.

## Smallest complete design

Keep the parent's fixed states: both steps 9 and 46 for each of
`alpha{.25,.5} x beta{.9,.8}`. These are two declared constant-LR stages,
not a post-hoc search over checkpoints. All comparisons use the actual
saved `W_s` and `W_(s+1)` and the same fixed fresh inputs.

I recommend **four** corners per state rather than three:

- base;
- body next, auxiliaries base;
- auxiliaries next, body base;
- full actual next weights.

With two banks of four sequences each, this is 8 states x 4 corners x 8
inputs = **256 sequence-forwards**. The recent prediction-profile probe
took 136 seconds for 320 forwards, so this is feasible in principle on
two CPU threads. Exact probability/FP64 reductions add cost; qualify the
actual four-corner sequence before committing, retain the declared panel
and a 600-second cap, and stop if the forecast fails. This review starts
no computation.

The auxiliary-only corner is worth its modest cost. Three corners can
observe a body/full difference but cannot tell whether it comes from the
auxiliary step's own magnitude or joint interaction. The earlier one-state
Muon experiment showed that interaction can be substantial, so omitting
the fourth corner would leave an avoidable ambiguity in this new factorial.
Do not split auxiliaries into further groups here.

## Fixed readout, with no new direction fitting

For each state, bank and corner, retain actual parameter-displacement norms,
signed NLL change and exact `KL(p_base || p_corner)`. Accumulate scalar
reductions in FP64 from FP32 forward logits. Parameter differences include
decay/write rounding; separately report the common ideal body norm
`192*eta_(s+1)` as a reference, not as the measured parameter displacement.

Also retain the exact finite-logit identity, which costs no new forward:

    L(z0+dz,y)-L(z0,y)
      = KL(p0 || p_corner) + [sum_v p0_v dz_v - dz_y].

The bracketed term is signed label-linear improvement. This prevents
mistaking a quieter predictive move for better progress: small KL may
accompany little useful response. Report both terms in their original
units. Do not create a scale-selected quality score or normalize all
responses to equal KL and call the result a trained improvement.

The fourth corner supplies the exact finite-loss interaction

    I = L_full - L_body - L_aux + L_base.

Keep the body, auxiliary and full changes alongside I. No sum of separate
KLs is presumed to equal the full KL, and positive I is not automatically
waste; the completed auxiliary study showed that useful predictive overlap
can generate it.

Across each stage/bank, report the full alpha-by-beta table for these
primitives and fixed differences/ratios. In particular, compare how beta
shortening changes body KL at alpha .25 versus .5, while retaining each
absolute KL and signed loss term. A ratio without its denominator can
repeat the misleading TS amplitude comparison.

## Discriminating readings and stopping rules

The simplest **functional damping** alternative predicts that stronger
alpha produces a smaller actual predictive body move, especially when
momentum is shortened. If that pattern is consistent at both stages/banks,
equal parameter norm did not isolate amplitude, and the dose result cannot
be attributed to a specially identified stiff filter without controlling
this simpler change. It is still not proof that the amplitude difference
mediates the training gain.

If body predictive amplitudes are comparable, or stronger whitening gives
larger actual functional motion, the simple own-state global-shrink account
weakens. That would leave selective geometry, response direction, altered
state dynamics and feedback as alternatives, without automatically proving
any one of them.

If the dose/beta difference is predominantly auxiliary or joint interaction,
the body-only explanation is incomplete at those actual steps. If it changes
sign across banks or stages, report that and stop; do not expand the sample
or select a favorable corner.

Every state is co-adapted to its training recipe. These are not common-state
counterfactual alpha maps, and the fixed fresh input panel does not remove
that limitation. No one-step NLL/KL ratio should rank training efficiency:
the main audit already showed that own-step quadratic quality does not
rank the optimizers' accumulated gains. The probe tests an amplitude premise,
not the complete training mechanism.

Required numerical gates are exact base/full endpoint reconstruction from
the saved states, exhaustive/disjoint body/auxiliary partition, finite CE/KL,
the finite-logit identity and consistent score inputs. Keep both banks and
all four factorial cells. The tiny sample gives paired descriptive evidence,
not independent training-seed significance.

## Decision

Proceed with this single bounded function-level premise check after source
pairing and next-weight availability pass. It adds useful information beyond
the existing static maps because it crosses the newly intervened dose with
momentum and includes the actual full update. Do not append a new estimator,
alpha schedule, stiff-mode projection or derivative replay to it. The
positive training interaction stands regardless of the local result; the
probe determines which simpler interpretation must remain in the account,
not whether the original result may be discarded.

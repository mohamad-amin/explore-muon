# Independent early-split result review

2026-09-28. Read the completed implementation, analysis and retained scalar
arrays. Independently recomputed the split identities, original-input
reproduction, state/group summaries and declared decision signs. No new
model, input, subgroup or alternative endpoint. This is the only new file.

## Verdict

**The negative early half-power interaction reproduces, but the proposed
helpful finite-nonlinearity explanation fails its declared test.** On the
fresh panel, its mean benefit comes from complementary additive prediction
changes. The mixed-output response is mildly adverse on average and changes
sign between fresh banks.

P1 fails, P2 passes as a relative comparison, and P3 fails. This is the
specified stopping outcome. Do not expand to the 16-corner subgroup
factorial, add samples or relabel "less harmful than quarter-power" as
"helpful in absolute terms."

## Numerical and formula checks

The probe completed all 256 score-forwards in 207.47 seconds. The original
eight inputs reproduce every old per-token CE corner and forward KL
exactly. Frozen helper hashes remain unchanged.

Independent per-token maxima for the three identities are approximately
4.89e-15, 5.14e-15 and 1.84e-15, well inside the declared 1e-10 tolerance.
Every inspected mean/sequence-dispersion summary reproduces exactly from
the retained arrays. State order, fresh-bank slices and paired contrast
signs agree with the protocol.

The key quantities remain distinct:

    I_total = I_overlap + I_mixed,
    I_mixed = J + KL_full - KL_add.

The log-probability construction implements the intended additive logits
up to an irrelevant tokenwise constant. There is no evidence that the
result comes from an indexing, label, normalization or precision error.

## Fresh-input finite split

| State | Total interaction | Additive overlap | Finite mixed loss |
|---|---:|---:|---:|
| Quarter, step 9 | +.091008 | +.058449 | +.032559 |
| Half, step 9 | -.047696 | -.053006 | +.005310 |
| Quarter, step 46 | +.017875 | +.016257 | +.001617 |
| Half, step 46 | +.026797 | +.024265 | +.002532 |

The early half-power total interaction is negative in both fresh banks.
Its finite mixed term is **+.018310** in the first bank and **-.007689**
in the second; pooled +.005310. Thus the negative additive term accounts
for the mean interaction benefit, while the mixed term offsets some of it.
The individual prediction changes are complementary in the measured
finite-CE interaction sense. This does not assert a particular GN angle
or an optimizer mechanism.

The early quarter-power mixed term is adverse on all eight fresh inputs.
The half-minus-quarter mixed contrast is negative on all eight, pooled
**-.027249**, so P2 genuinely passes. Its meaning is a smaller adverse
mean mixed effect, with bank-dependent absolute helpfulness for half-power.
It does not rescue P1's absolute helpfulness prediction.

The half-power late-minus-early mixed contrast is **-.015690** in fresh
bank 0 and **+.010133** in bank 1. The first bank becomes less adverse
later; the second loses its early helpful term. Hence P3's requirement of
uniformly less favorable mixed contribution later fails. A pooled stage
mean must not replace that predeclared bank condition.

## Why J's sign was not enough

On fresh half-power step-9 inputs,

    J = -.038475,
    KL_full - KL_add = +.043785,
    I_mixed = +.005310.

The unmeasured finite KL correction reverses the tempting reading of the
negative base-linear term. The original discovery inputs make this even
clearer: J was **-.032749** with negative sign on every input, yet the
newly measured finite mixed loss is **+.014743**, positive on every one
of those same inputs. This is not merely a new-panel sign fluctuation.

The original observation of label-relevant nonadditivity remains true.
What fails is treating its pairing with the base loss residual as the
finite usefulness of adding it after the body/auxiliary additive response.
The measured example is a concrete instance of why a base-state linear
score does not determine a finite step's effect.

The early dose contrast in total interaction separates into approximately
**-.111455 additive** and **-.027249 mixed** terms. These are absolute
differences along the defined logit path, not shares of the eventual
training gain or an intervention on a causal mediator.

## Scientific consequence

The successful part of the new observation is the stage/state dependence
of the **additive prediction interaction**: it is negative for early
half-power PD and positive later, whereas early quarter-power is positive.
This extends the earlier late-Muon ordinary-overlap picture by showing
that such interactions can be beneficial as well as costly. It does not
require a new transport defect or a specially helpful nonlinear channel.

The previous late-state finding is preserved, and the separate positive
alpha-by-momentum training interaction and warmup result are untouched.
Neither this local split nor its failures establishes their mediator.
All states are learned under different trajectories, the additive logits
are synthetic, and the fresh panel contains eight sequences rather than
independent training replications.

Close the helpful-mixed-response premise at the declared scope. Preserve
the positive relative P2 contrast, both fresh-bank signs, all old-input
results and the exact KL correction. There is no basis here for a
head/embedding/norm factorial, new response rescaling, additional sampling
or a staging/freezing remedy.

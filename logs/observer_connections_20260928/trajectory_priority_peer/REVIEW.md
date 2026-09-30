# Independent priority review: the completed momentum-by-LR factorial

2026-09-28, after the main protocol's 15:52 CDT 4M result. Read current
RESEARCH_GUIDE/RESEARCH_STATE, the latest main decisions, observer synthesis,
clipping/horizon addendum, estimator-clock note, auxiliary factorial and
embedding inventory, then inspected four-arm configurations, metadata and
artifact availability. No model, checkpoint-tensor, GPU or scheduler call.
Only this review file is written. This is a retrospective recommendation:
the main notebook's endpoints and broad trajectories were already visible.
No new thresholds are presented as predeclared scientific criteria.

## Priority and the one connection

**Use the completed 16M SOAP-PD beta-by-body-LR factorial before attempting
feedback-derivative replay.** It changes the body movement scale while
holding the explicit auxiliary and estimator clocks fixed. This is a more
direct opportunity to test whether the observed momentum benefit is tied
to movement/stability exposure, or behaves like a common update-count
transient, than the new 4M/16M timing comparison alone.

It is one bounded use of existing training trajectories, not another clock
survey or optimizer search. The parent inventory finds the derivative branch
would require additional incoming-gradient/frame reconstruction and numerical
controls, potentially model work. The already-qualified analytic distinction
between static suppression and differential feedback should stand; completing
a local derivative surrogate now would risk measuring another difficult
proxy before fully using the actual trajectories.

## Exact four arms

All paths below are relative to `logs/muon_spectra/`:

| Body LR | Momentum | Arm |
|---|---:|---|
| .028 | .9 | `soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada` |
| .028 | .8 | `soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada` |
| .04 | .9 | `soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.04_mom0.9_s260925_ada` |
| .04 | .8 | `soaudit_strength16m_20260928/SPD_a0.5_b16M_lr0.04_mom0.8_s260925_ada` |

Each has a completed 92-step summary and 93 step records including step 0.
The initial hash is identical:
`f8efae98b3d9e0dd5e9cd2318ea2b56a8b01358742af2d91e8c86ea9caf41d19`.
Configurations match apart from the intended LR/momentum and newly recorded
disabled defaults (`muon_prefilter=none`, zero head-whitening alpha/false
centering). They use the same seed, 16M batch, 1.54B token exposure, alpha .5,
three-step LR warmup, final 10% cooldown, and Ada hardware recipe. The word
`prefilter` in a cohort name does not mean the selected arm uses two-tap
filtering: its configuration explicitly says `none`.

Before interpretation, independently verify the data manifests, hardware,
actual per-step rates, scientific worker completion and execution-source
differences/hashes. Configuration similarity alone does not certify identical
active source behavior. My inspection is an inventory, not that full audit.

## Fixed contrasts worth extracting

Define, at each matched score location,

    Delta_eta = loss(beta .8, eta) - loss(beta .9, eta)
    I = Delta_.04 - Delta_.028.

1. **Ordinary validation remains the endpoint.** Report every matched
   scheduled validation score, expected at 0, 50 and 92, and both Delta
   values and I. Retain all four losses. Do not silently compare beta .8's
   best rate with a different beta .9 rate, or select a checkpoint because
   the interaction looks largest there. There is one scientific seed.

2. **Use paired training differences to locate the interaction.** The
   same update number consumes the same data within and across these arms.
   Compute Delta and I per step, then show means in the four fixed
   constant-LR windows 4–23, 24–43, 44–63 and 64–83. Keep warmup 1–3 and
   cooldown 84–92 separately. These windows partition the entire available
   constant-LR segment; they do not select the known favorable region.
   The records are correlated observations of one trajectory, not dozens
   of independent replicates. Training is a paired trajectory diagnostic,
   not an additional independently held-out outcome.

3. **Display step and movement clocks without fitting a winner.** Show
   the two beta-gap curves against raw step and cumulative body LR. Row t's
   training loss is evaluated before update t, so its cumulative rate is
   `sum_{j<t} eta_j`. Validation at t is after update t and uses
   `sum_{j<=t} eta_j`. Avoid an off-by-one phase artifact at startup.
   Use the existing sparse validation points to qualify the dense training
   view. Do not optimize a time warp, smoothness bandwidth, target threshold,
   peak window or amplitude normalization to make either clock collapse.

An informative outcome is a beta-by-LR interaction throughout the constant-LR
segment: its sign and temporal development constrain a purely step-timed
account. If the higher body LR systematically advances a visible feature
under the cumulative-rate coordinate, that is evidence for a movement-linked
candidate; similar raw-step timing despite different movement budgets favors
a common-clock candidate. But magnitude changes alone do not establish a
timing shift, and this 92-step experiment may end before either rate's
catch-up. In that case the honest result is that the existing factorial
identifies an early interaction and leaves the later phase clock unresolved.

Neither outcome authorizes a new broad sweep. In particular, a near-zero I
does not prove an auxiliary clock, and a nonzero I does not prove stiff-mode
feedback or rule out secondary changes to the auxiliaries.

## A useful exact movement fact

Online PD/SOAP-PD rescales each body direction to `sqrt(min(rows,cols))`
and applies Muon's aspect multiplier. For this eight-layer width-512 model,
each block has five matrices with squared norm 512 and an up projection
with squared norm 2048. Across eight blocks the squared norm is 36,864;
the ideal concatenated direction norm is **192**.

Consequently the ideal adaptive body-step arc is `192 sum eta`, and the
sum of squared step norms is `36864 sum eta^2`. Both are identical between
the two momentum values at a fixed LR. A beta benefit there cannot be
explained by receiving a larger scalar body-step norm budget. These exclude
decay and parameter-write rounding and are not net displacement: direction
correlations decide how much movement accumulates. Sampled archived
`adaptive_step_norm` fields can verify the expected normalization.

This is a useful separation from a conventional heavy-ball effective-LR
argument based on the raw momentum norm. Degree-zero polar normalization
removes that global amplitude, while direction and SOAP's historical
coordinatewise normalization still change.

## What the new 4M result does and does not distinguish

The 4M beta .8 SOAP-PD run finishes 3.8280 versus 3.8260 at beta .9, after
a substantial earlier lead. Together with the 184-step 16M run, this
supports transient benefit under these recipes and rejects the simplest
claim that the large 92-step endpoint gap is a durable large-batch law.
The existing horizon audit already establishes attenuation before cooldown.

It does **not** yet identify an intrinsic 100–150-step landscape phase.
The 4M and 16M comparisons use body LR .02 versus .028, LR warmup 13 versus
3, and C-root refresh 10 versus 2. They share SOAP beta2 .9 and auxiliary
Adam betas (.9,.95), whose clocks are in optimizer steps; input C's clock
is largely token-based. Aux LR remains .002. Similar step timing is
therefore compatible with body progress, normalized-step feedback, estimator
adaptation or auxiliary/representation co-adaptation. The useful prediction
overlap found by the auxiliary factorial makes the latter credible without
identifying it as a defect.

The completed four-arm factorial removes several of these differences:
all explicit clocks, warmup and token counts agree, while body LR changes.
Realized auxiliary gradients and steps still need not match, because the
body trajectory changes their inputs. Thus it is a discriminator of simple
clock explanations, not an isolation of a body-only causal mechanism.

## Decay and startup constraints

Cumulative LR and decay exposure are nearly collinear in this factorial;
report `-sum log(1-eta*wd)` rather than pretending the two axes are separate
interventions. However, even the crude upper bound for the high-LR run is
`92*.04*.01=.0368` in the first-order exposure, only about 3.6% shrink.
This is far below the earlier Track 3 decay-equilibrium regime around
exposure 2–3. That known equilibrium explanation is implausible for this
short-run effect, while ordinary decay/weight-scale interaction is not
mathematically excluded.

The same-beta short/long horizon runs use the same three-step warmup and
retain almost the same first constant-LR trajectory; attenuation before
the longer cooldown is therefore not a simple cooldown artifact. It remains
possible that an early advantage is repaid through trajectory co-adaptation,
rather than that changing beta at a particular clock time yields both
advantages. The already-authorized momentum-warmup result addresses that
practical question. It should retain its original comparison criteria,
regardless of how the retrospective clock views look.

**Decision:** prioritize this small actual-trajectory factorial, defer
derivative replay, and retain "early benefit followed by catch-up under
these recipes" as the current conclusion. A unique step, token, path-length,
auxiliary or estimator phase clock has not been established.

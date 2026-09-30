# Three dynamical accounts that remain compatible with the trajectory evidence

2026-09-28. Read the current observer STATE/SYNTHESIS, PD warmup transfer,
Q/K functional result, conditional-input provenance gate, persistence audit,
failed instantaneous-edge audit and GN-PD floor history. No model/probe,
checkpoint read, tensor arithmetic, training or GPU work. Only this note
was written. These are competing dominant accounts; mixtures are possible.

## What is established, and what should stop acting as a gate

The matched PD warmup pair gains .02667 at 184 updates, after both schedules
use beta .9 from update 93. This extends the result beyond SOAP-specific
statistics. SOAP-PD's short-horizon warmup loses to constant .8, while the
long-horizon warmup beats both tested constants. These are real trajectory
facts, but neither a universal step clock nor a mediator is identified.

The Q/K experiment establishes larger finite predictive movement under
shorter beta, while its last-added usefulness criterion fails. That failure
closes the chosen **local** premise, not the broader radius/history account.
Q/K helps alone and has positive symmetric credit; full steps improve loss.
The earlier greedy-Muon control already showed that insisting on immediate
held-out improvement can reject a successful training dynamic. We should
stop using another component's one-step helpfulness as a necessary condition
for its long-run role. Conversely, a favorable local margin would not have
proved that role. No more four-corner localization is justified by these
outcomes alone.

## 1. Early tracking advantage, followed by an ordinary head start

**Conjecture:** a short window follows the rapidly changing early optimization
target more promptly. Once the trajectory evolves more slowly, longer
averaging becomes adequate; warming beta preserves an early advance along
roughly the same later learning path. This does not require the early gain
to remain a superior per-step rate indefinitely, or require iid gradient
noise to be its cause.

**Compatibility:** GN-PD's superior floor direction produced a front-loaded
release whose cumulative advantage shrank, rather than a lasting rate.
The momentum lead also peaks early. A smaller vertical NLL gap late does
not by itself demonstrate that the other run catches up: a constant lead
in updates produces a shrinking NLL gap on a flattening learning curve.
The PD lead after beta matches is compatible with retained progress rather
than continuing distinct-beta dynamics.

**Distinct prediction:** after schedules coincide, a strong common-progress
version predicts an approximately bounded/constant *horizontal* lead over
a common pre-cooldown loss range, even while the vertical gap narrows. A
systematically growing lead would contradict the simple one-time-head-start
account; genuine shrinking horizontal lead would support actual catch-up.
Neither outcome alone identifies a biological notion of "the same features."

**Smallest qualified artifact check:** the complete, source-paired scalar
curves in `../warmup_transfer/trajectory.csv` and `result.json`, plus the
completed original SOAP-PD two-horizon records, already suffice for this
limited distinction. Separate the common-beta, common-LR interval 93–166
from cooldown. Use fixed anchors/windows and the whole common loss range;
retain raw curves and an interpolation/smoothing uncertainty check. Do not
fit an arbitrary time warp or choose only favorable crossings. This is the
best next cheap discriminator, although too-coarse or noisy crossings can
make it unresolved. It is not a new causal clock test.

## 2. Momentum changes an endogenous relative learning rate

**Conjecture:** momentum/map history produces coherent radial growth in
scale-normalized weights. Radius then changes relative feature turns even
at the same prescribed parameter norm. Short beta supplies larger relative
steps early; raising beta later changes that endogenous annealing process.
This is a specific state variable, distinct from nominal LR or buffer age.

**Known support:** the fixed factorial gives 15–25% larger later Q/K turns
under beta .8 at almost identical head-step norms, and partial absorption
of a 43% LR increase into radii. Both effects reach finite predictive KL.
Coherent radial cross terms dominate sampled squared-step contributions,
so a pure eta*sqrt(t) diffusion explanation is not established. No causal
role or Q/K share of the training gain follows.

**Distinct prediction:** a dominant continuing-radius account predicts that
post-switch relative-motion differences track residual radius differences,
not merely the now-common beta coefficient. Equal nominal memory and LR
need not erase them. If radii/relative turns have converged while a continuing
rate advantage remains, this simple continuing mechanism is inadequate.
A remaining fixed head start would still be compatible with an earlier
radius-mediated effect, so endpoint radii cannot decide its origin.

**Artifact limit:** the original long-horizon SOAP-PD constants/warmup retain
92/165 and adjacent writes; they could support a separately declared
post-switch descriptive comparison without model calls. That would be a
new history question, not an extension rescuing the closed twelve-state
gate. Current records contain no intervention matching radius while changing
memory. Thus no qualified existing artifact isolates radial causality.
The local Q/K-last failure is not that intervention.

## 3. A moving geometry changes the feedback that memory must filter

**Conjecture:** whitening, learned representations, gains and other updates
co-adapt into a time-varying coupled system. The useful memory window depends
on that evolving response and its delayed restoring forces, beyond a scalar
angular rate. Early feature/metric rearrangement can favor a fresh input;
later a longer filter can stabilize or average a different trajectory.

**Known constraints:** preconditioned states sharpen strongly where their
steps move little, yet train better. Two-tap cancellation of alternating
gradients hurts. Dense 1M lag data reject a clean universal period-two or
steady persistent-tail picture. The earlier differential edge test also
failed its universal scalar stability prediction. These facts permit a
moving coupled-feedback account; they do not verify it. Finite NS may have
a nonsymmetric/indefinite derivative, so a frozen positive-real eigenvalue
or static stiff-energy fraction is not the full dynamical object.

**Distinct prediction:** even after relative step sizes and nominal beta
match, orientation/coupling history can affect subsequent rates. Stability
would depend on products of changing joint parameter/optimizer-state
responses, rather than one instantaneous eta*lambda condition. A radius-only
account would not predict this irreducible orientation dependence.

**Artifact limit:** there is no qualified saved sequence of those joint
responses for the warmup family. Old scalar EOS values cannot be repaired
into it; the unqualified g1M/g4M de-mixing cannot supply the missing input
repeatability. Do not launch another selected-state derivative probe merely
to recover the preferred stability story.

## Input-curvature marginals deserve a better role, not another alpha scan

We risk overlooking the marginals by treating them only as encouragement
for an inverse power or as isolated K/V anomalies. Their useful role is to
constrain how the model's metric evolves: full input moments, mean versus
centered structure, norms, and exact-marginal orientation errors across a
trajectory. Agreement along C's eigenvectors does not establish equality of
full operators. Conversely, a changed marginal is not automatically an
unexploited optimization opportunity; per-kind powers and token-weighted C
already had disappointing controls.

The original `second_order_audit_20260926/marginals/` panel is principally
the four older 1M trajectories, not a matched beta/warmup comparison. It can
constrain general co-adaptation claims but cannot adjudicate these three
16M-history accounts directly. Newer rank-0 covariance sidecars are useful
proxies only within their recorded scope: they are not all owners' cached
roots or SOAP state. The root's inventory should establish that distinction
before any cross-time metric comparison is proposed.

My priority is therefore: first distinguish preserved head start from
continued rate change using the already qualified post-switch trajectories;
then use available state-history evidence to constrain radius versus richer
geometry accounts. None of these steps requires another local component-
helpfulness gate. No strong new optimizer intervention follows yet.

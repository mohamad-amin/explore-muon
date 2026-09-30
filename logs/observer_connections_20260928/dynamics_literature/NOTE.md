# Three literature connections for the observer anomalies

2026-09-28. Primary papers checked online; local conclusions below distinguish
published results from our inferences. No model calls, GPU work, or training.
This is a bounded interpretation note, not a novelty claim or a proposal to
launch experiments. Local starting evidence: `../SYNTHESIS.md` and
`../SIGNAL_NOISE_FINDINGS.md`.

## 1. Noise is relevant in the geometry of a step

**Published framework.** McCandlish et al., *An Empirical Model of Large-Batch
Training*, §2.2, equations 2.5–2.9, distinguishes curvature-weighted noise
`tr(H Sigma)/(g^T H g)` from the simplified raw-gradient statistic
`tr(Sigma)/||g||^2`. The latter follows by replacing the Hessian with a scalar
identity. Section 2.4 explicitly identifies short-horizon optimization bias,
oscillation in stiff directions, differing component noise scales, and the
quadratic approximation as limitations. The paper does not provide a theorem
that a raw signal-energy fraction measures the usefulness of Muon/SOAP steps.
[Paper, §2.2–2.4](https://arxiv.org/html/1812.06162#S2.SS2)

**Application derived here.** For any random update direction `d`, a quadratic
loss gives the identity

```
E[L(theta - eta d)] - L(theta)
 = -eta g^T E[d]
   + eta^2/2 * (E[d]^T H E[d] + tr(H Cov(d))).
```

For a fixed linear map `d=A(g+noise)` the covariance term becomes
`tr(H A Sigma A^T)/B`. For a nonlinear polar map, both `E[d]` and `Cov(d)`
must instead be measured after that map; in general `E[d] != d(E[g])`.
This elementary identity explains why the archive's >99% raw-signal-energy
summary does not settle noise relevance in coordinates holding most movement.
It also cautions that Euclidean displacement energy alone is insufficient:
noise in nearly flat directions can be cheap in the local loss.

**Archive discriminator.** Preserve raw signal, displacement energy, and
signed slope as separate summaries. If weak-coordinate movement is productive,
independent scoring should retain positive slope; a large energy share alone
does not predict it. Existing masks selected using the scoring gradients
cannot supply that independent test. The archive lacks repeated actual SOAP
directions and their bases, so it cannot recover `Cov(d)` for SOAP exactly.
Replacing H by GN supplies a GN-model diagnostic, not this true-Hessian identity.

## 2. Preconditioning changes the stability statistic

**Published framework.** Cohen et al., *Adaptive Gradient Methods at the Edge
of Stability*, §2 and §4, studies `lambda_max(P^-1/2 H P^-1/2)` for an update
using `P^-1`. Appendix A/Table 1 gives different thresholds for heavy-ball,
EMA heavy-ball, and Nesterov conventions. For EMA heavy-ball, the quadratic
threshold is `2(1+beta)/(eta(1-beta))`. Their adaptive-optimizer experiments
show raw curvature increasing while preconditioned curvature stays near an
edge; they explicitly use a frozen-preconditioner approximation to interpret
the adaptive dynamics. [Paper, §2/§4 and Appendix A/Table 1](https://arxiv.org/pdf/2207.14484)

**Local implication.** The project observation that whitening methods inhabit
sharper states is compatible with established adaptive dynamics. It is not
evidence by itself that sharpness causes their gain or failure. Here the saved
operator is often GN, whereas the paper's stability threshold concerns the
true Hessian. Moreover, polar normalization is nonlinear, and the optimizer
state changes. A raw-GN eigenvalue or a step's Rayleigh quotient cannot simply
be multiplied by an Adam threshold to certify Muon's stability. A complete
local stability calculation would differentiate the joint parameter/state
update, including normalization and preconditioner updates.

**Archive discriminator.** Compare signed lag structure in separately defined
groups and retain actual update weighting. If large raw-gradient components
mainly alternate while the high-movement groups carry persistent slope, then
raw sharpness/gradient norm are poor summaries of the effective dynamics.
Failure to find this separation would weaken that explanatory connection.
This comparison tests a mechanism premise, not the paper's threshold theorem.

## 3. Alternation can be deterministic feedback rather than sampling noise

**Published framework.** Damian, Nichani and Lee, *Self-Stabilization: The
Implicit Bias of Gradient Descent at the Edge of Stability*, §4.1 equation 3,
describes an unstable-mode displacement x and sharpness excess y:

```
x[t+1] approximately = -(1 + eta*y[t]) x[t]
y[t+1] approximately = y[t] + eta*(a - b*x[t]^2/2).
```

The squared oscillation supplies a curvature-reducing feedback. The rigorous
coupling result is §5.3/Theorem 1, subject to the paper's smoothness,
progressive-sharpening, eigengap and other assumptions. It concerns ordinary
deterministic GD, not stochastic polar momentum. [Paper, §4.1/§5.3](https://arxiv.org/html/2209.15594#S4.SS1)

**Local implication.** A period-two component can have excellent fixed-state
SNR and negligible net displacement over two steps while still affecting
stability. Thus failure of the project's two-tap cancellation intervention is
compatible with feedback removal. It does not demonstrate this cubic mechanism:
the same intervention also changes phase/lag and all other temporal frequencies.

**Archive discriminator.** In a fixed saved frame, use independently sampled
base and target gradients to estimate signed group products
`C(k)=sum_i mu_i(t) mu_i(t+k)` without entrywise positive clipping.
Persistent signal predicts positive products across multiple lags. Dominant
period-two structure predicts negative odd and positive even products, with
an envelope that can decay. Pure independent measurement noise predicts zero
cross-products. Neither one negative lag nor positive static energy establishes
any of these pictures. Only a full sign pattern would support alternation.

## An exact momentum calculation that makes these connections testable

This paragraph is elementary algebra, not a newly discovered optimizer result.
The frozen mom16M implementation has `M[t]=beta*M[t-1]+g[t]` and
`muon_nesterov=false`. Define `m=(1-beta)M` to express its temporal filter in
unit-DC-gain units. Before any nonlinear normalization, the steady-state
frequency response is

```
F_beta(omega) = (1-beta)/(1-beta*exp(-i*omega)).
```

| beta | Mean sample age | White-noise variance factor | Period-two amplitude factor |
|---|---:|---:|---:|
| 0.9 | 9 | 0.05263 | 0.05263 |
| 0.8 | 4 | 0.11111 | 0.11111 |
| 0.7 | 2.3333 | 0.17647 | 0.17647 |

The formulas are `beta/(1-beta)` and `(1-beta)/(1+beta)`; the period-two
**power** factor is the square of the last column. The noise formula assumes
temporally independent, equal-variance noise in an externally supplied
gradient stream. Lower beta reduces lag while allowing more noise and more
alternation through. It cannot be described as universally improving noise
suppression. These exact pre-normalization quantities do not predict the final
polar direction or training performance on their own.

For an exogenous sequence of population gradients, alignment of this filter
with the present gradient is a geometrically weighted sum of past signed
cross-products. The saved `t`-versus-`t+k` products test persistence locally;
they do not constitute that complete past-time series. Stationarity or
reversing time would be additional assumptions. Nor may we identify probe
noise with actual training noise: the current state depends on past training
noise, so its correlation with the current gradient need not vanish.

**Important dependence in the archive.** `measure_persistence.py` uses one
sequence set A at the base state and a disjoint set B at every later state.
Base–target noise is independent for each product, but errors at different
lags are dependent because A and B are reused. Treat lags as a structured
curve, not independent replications. The independent-basis rank bins are
Kronecker groups, not exact GN eigenspaces. Do not combine signed coordinates
from separately fitted frames whose eigenvectors were not saved.

## What this changes now

The most discriminating available analysis is the signed persistence curve
of groups that carry the actual movement, with static signal/noise summaries
kept beside it. A positive, sustained weak-group curve would justify examining
how normalization retains weak persistent signal. A near-zero curve would
weaken that premise; an alternating curve would point toward dynamics and
feedback. None alone demonstrates a new training improvement. The published
frameworks constrain interpretation and suggest what to measure; they do not
settle the archive's mechanism.

Local source checks: `logs/muon_spectra/soaudit_mom16m_20260928/frozen/adamw_spectra/muon.py`
lines 657–664; that cohort's scientific configurations (`muon_nesterov=false`);
`logs/muon_spectra/second_order_audit_20260926/measure_persistence.py` lines 80–96.
All new files from this task are confined to this directory.

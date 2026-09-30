# Clipping and momentum: bounded retrospective check

Recorded 2026-09-28 before computing the coefficient summaries.

The main notebook's 13:50 review identifies frequent clipping as a competing
explanation for the large beta=.9 -> .8 benefit in whitening methods at16M.
The old observer `logs/observer_20260928/momentum_dynamics/audit.py` already
checked clip frequency, matched recipes, and two-tap scalar stability. This
extension asks whether the clipping sequence materially changes the history
weights, rather than merely whether clipping fires.

For the plain body momentum, m_t=beta*m_(t-1)+c_t*g_t,
c_t=min(1,clip/(global_gradient_norm+1e-6)). The log norms reconstruct its
scalar coefficients, not the saved gradient vectors or their directions.
Report the age distribution on raw gradients, and separately on whole-model
unit gradients (coefficients additionally multiplied by the global norm).
The latter is not a per-matrix direction age. For two-tap arms use the exact
first-step exception in the frozen source. Preserve the no-clipping counterfactual
kernel on the same logged trajectory only as an algebraic comparison, not a
counterfactual training trajectory.

Main competitor: nominal beta and the short92-step horizon already account
for the memory change; clipping can still affect direction through norm/angle
correlations and auxiliary Adam state, even if scalar ages barely change.
Compute effective normalized raw-gradient history decay, mean ages in steps
and tokens, clipping dispersion, and pre-cooldown loss differences. Preserve
4M and2x-horizon controls without equating different geometries/trajectories.
Check current completion separately for newly arriving clip arms. A finding
that frequent clipping does not inflate memory weakens only that specific
mechanism, not all clipping explanations.

Cost is stdlib JSON/scalar arithmetic on this CPU, at most two numerical
threads, no torch imports, forwards, training, GPU requests or job changes.
All new outputs stay in this directory. Main protocols, state, old failures
and notes remain unchanged. This is a retrospective extension of an existing
observer check, not a new scientific training commitment.

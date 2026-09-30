# The momentum result at twice the horizon, 16M (second-order audit, 2026-09-28)

At 1× (92 steps) S∘PD at β 0.8 gained −0.102 over β 0.9. The review (MUON_CASE, 13:5x) noted that a ~10-step memory is
a large share of a 92-step run with a 9-step cooldown. At 4M, PD α ¼ at β 0.81 was +0.012 worse than 0.9. So the gain may
be specific to short runs.

- **Arm.** S∘PD α ½ @0.028 at β 0.8, 16M, twice the horizon (3.08B tokens, 184 steps), seed 260925, Ada. Reference: β 0.9
  at twice the horizon, 3.9840 (`../soaudit_b16mlong_20260928`).
- **Readings.**
  - A gain near 0.1 that holds at 2×: the lag cost is not a short-run artifact.
  - A gain that shrinks roughly in proportion to 1/horizon: it is the lag's share of the run, and the cooldown.
- **Queue.** After it, the LR 0.04 β 0.8 arm of `../soaudit_strength16m_20260928`.

# Saved-profile audit: the map, its state, and the time index

2026-09-28, before aggregate extraction. CPU arithmetic on saved JSON only.

The main study's new 15:43 profiles propose that whitening spatially suppresses
stiff oscillations, allowing shorter momentum. The reported actual-step
energy contrast is across each optimizer's own state. Those states have very
different curvature; the table also juxtaposes checkpoint M_s with the actual
next displacement driven by M_{s+1}. Existing JSON includes an unused cleaner
comparison: ideal Muon and PD maps of the identical M_s at identical W_s,
scored against the same sampled GN.

Question: how much of the observed stiff-energy separation is already a
frozen-map property, and what cannot be inferred about temporal feedback?
The competitor is a separation that appears only after optimizer/state
co-adaptation. A further qualification is that Muon's polar map itself can
redistribute momentum energy; it need not be a transparent spatial pass.

Read all nine new beta-.8 profiles (M, PD, SPD at 9/46/83), and exactly the
nine matching original beta-.9 16M profiles. Retain original scalar fields,
file hashes and sources. Compute same-state PD/Muon ratios of normalized
energy in top 1/4/16 Ritz vectors, directional Rayleigh quotient, signed
held-out slope per norm, and curvature per norm squared. Keep raw norms and
the ratio between curvature/held-out samples. Report nonnegative-step model
quality max(-a,0)^2/(2q) separately from the archived sign-erasing a^2/(2q).
No optimizer ranking will be inferred from either local score.

For all three maps (momentum, Muon, PD), quantify the spatial redistribution
on the same saved momentum. Display the actual next displacement separately;
do not treat it as the output of the logged M_s. Source audit must retain
that PD uses a fresh root estimated from the curvature bank and no per-matrix
norm matching, while actual training uses cached/EMA roots and the NS map.
The next-gradient measurement is body-only; it is not the full W_{s+1} state.

An effect consistent within states supports a conditional map effect, but
neither identifies online Jacobian response to an oscillatory perturbation
nor explains training rate. Only an actual perturbation comparison with the
same state, incoming momentum and fresh gradient could isolate that response.
If existing scalars lack it, state that limitation; do not automatically run
another model probe. Broad input/auxiliary prediction overlap is a separate
finding unless the artifacts show a concrete link.

Bound: at most seconds of CPU work, no checkpoint loading, model/GPU call,
new training, external test data, or main-file edits. New outputs stay here.
Independent source/concept review is running in `../momentum_map_peer/`.

## Analytic qualification addendum, before computation

The completed peer discussion identifies a precise missing object: the
derivative of the map, rather than its output's spectral projection. Qualify
one exact 3x3 example with unit-Frobenius positive diagonal momentum and a
skew transverse perturbation. Verify the polar and fixed-root norm-matched
PD derivatives by central differences, plus the joint weight/momentum block
Jacobian for the code's unnormalized momentum convention. Use epsilon=.02,
one fixed R=diag(1,.2,.2), and declared finite-difference step sizes
1e-4/1e-5/1e-6 for convergence. This is an arithmetic check of an analytic
counterexample, not a toy optimizer search or evidence about network rates.
Require relative error below1e-5 at the finest step and below1e-5 for the
full joint Jacobian; retain all errors. No network or checkpoint is used.

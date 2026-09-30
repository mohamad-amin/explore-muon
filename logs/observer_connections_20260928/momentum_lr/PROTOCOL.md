# Retrospective trajectory contrast: momentum × body learning rate

2026-09-28. This comparison uses an already completed 2×2 identified by an
independent peer while root inventoried feedback tensors. It is retrospective,
not a preregistered training experiment or independent confirmation. The
arithmetic/readout below is fixed before root extracts its loss contrasts.

## Question and competing explanations

The newer4M result weakens a batch-only reading of the short-momentum benefit:
SOAP-PD beta.8 leads early and ties beta.9 at completion. The proposed fixed-step
phase clock remains confounded across batches by body LR, warmup, root refresh
and data exposure. An existing same-batch2×2 can test a narrower claim: does
changing body step size change the momentum contrast when the auxiliary
schedule, statistical clocks, data and number of steps are held fixed?

A separable fixed-clock explanation predicts a similar beta contrast at both
body rates. A material beta×LR interaction would instead require dependence
on body step size, state evolution or relative body/auxiliary scale. These
possibilities cannot be uniquely separated by this2×2. A collapse against
cumulative LR might support a movement-clock description, but will not prove
one: loss phase, curvature, decay and realized auxiliary updates co-evolve.

## Frozen comparison

All four are SOAP-PD alpha.5,16M tokens/update,92updates,seed260925,Ada, with
body LR{.028,.04} and beta{.9,.8}. Exact arm paths are embedded in `analyze.py`.
Require equal initial-weight hash, data manifests, budget, hardware, auxiliary
LR/betas, clipping, model and active method options. Retain configuration and
frozen-source differences, including newer disabled prefilter/head defaults.
Stop interpretation on an unexplained active difference; no rerun follows.

Read every per-step record. Define Delta_beta(eta,t)=L(beta.8,eta,t)−
L(beta.9,eta,t), and I(t)=Delta_beta(.04,t)−Delta_beta(.028,t). Report all common
validation points separately from same-batch training losses. Training row t
is evaluated before update t; validation at t is after update t. No significance
test treating steps or windows as seeds. The2×2 has one paired seed.

For training, report four complete20-step constant-LR windows4–23,24–43,
44–63,64–83; keep the84–92cooldown separate and preserve all raw observations.
Window means use actual batch-token weights, including the shortened final
batch. No fitted peak, favorable loss threshold, endpoint choice or window
extension. Plot raw contrasts and a declared9-step centered moving mean;
the smoother is display-only. Compare step and cumulative-body-LR axes
without a fitted time warp or amplitude normalization.

For training row t the path proxy uses sum_{j<t}eta_j; for validation at t
use sum_{j<=t}eta_j. In this architecture, online PD/SPD rescales each matrix
to sqrt(min(rows,cols)) and applies sqrt(max(1,rows/cols)); their combined
direction norm is192. Nominal body arc is therefore192*sum(eta), before
decay/write rounding, and identical within a beta pair at fixed LR. This is
not net displacement, function-space path length or an empirical norm trace.
Also retain decay exposure and exact scalar shrink product, and the common
auxiliary LR clock. Actual auxiliary directions can differ despite matched
settings. Body LR and decay exposure are proportional here and cannot be
identified separately merely by replotting.

## Decision and cost

The interaction constrains a separable fixed-step/auxiliary-clock story;
it cannot by itself prove spatial filtering, identify an optimal schedule,
or justify another training arm. A weak interaction closes this discriminator
without opening a sweep. Preserve all negatives and pairing limits.

CPU only, a few hundred JSON records, at most two numerical threads and
seconds of arithmetic/plotting. No checkpoints, models, GPU, scheduler call
or sealed-test data. All new files live here. Scientific review is in
`../trajectory_priority_peer/REVIEW.md`.

Execution qualification: the first extraction stopped on exact equality of
the auxiliary LR clock across body rates. Eight logged values differ by at
most4.34e−19 (relative2.94e−16), from evaluating aux_peak*(eta/eta_peak).
Every pair rounds to identicalFP32values. Preserve the failed script and
differences in`failed_exact_aux_clock.py`/`failure_exact_aux_clock.json`;
the corrected execution requires agreement within4FP64ulps and identical
FP32representations. No loss threshold, arm, window or scientific criterion
changes. Config/within-beta-pair checks remain exact.

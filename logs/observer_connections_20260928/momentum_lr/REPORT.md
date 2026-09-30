# The short-momentum advantage survives a43% increase in body step budget

2026-09-28. Retrospective analysis of four completed16M-token SOAP-PD runs;
one seed, common Ada hardware,92updates. CPU JSON arithmetic and static
plotting only. The full comparison and limits were fixed in`PROTOCOL.md`
before root extracted loss contrasts, after an independent peer identified
the existing factorial. This is not a prospective confirmation experiment.

The new4Mtrajectory suggests that the beta.8 advantage is early-phase rather
than an effect unique to large batches. However, comparisons across batches
also change warmup, body LR, statistic refresh and data exposure. The four-arm
same-batch comparison holds these fixed and crosses beta{.9,.8} with body
LR{.028,.04}. It tests whether the momentum advantage depends strongly on
body step size inside this existing bracket.

## Qualification and pairing

All four runs have identical initial-weight hash
`f8efae98b3d9e0dd5e9cd2318ea2b56a8b01358742af2d91e8c86ea9caf41d19`,
training/validation manifests, architecture, horizon, world size, hardware,
clipping, auxiliary Adam settings, active whitening/SOAP options and statistical
clocks. Initial validation NLL is identical. Each has all92training records
and completed status. Configs differ only in the two factors and newer
disabled prefilter/head-whitening defaults.

Frozen source differences are retained in`source_differences.patch`. The
changes add disabled options, metadata and a correction inside the inactive
output-factor path. The active equations agree; bitwise identity of all
compiled executions across source revisions is not asserted.
The independent results reviewer checked200frozen/scientific source hashes
against metadata with zero mismatches, and recomputed the scalar contrasts.

The first extraction stopped because an exact comparison of auxiliary LR
values failed: eight FP64values differ by at most4.34e−19. This is rounding
in aux_peak*(body_lr/body_peak); every rate rounds to identicalFP32. The
failed script and discrepancies are preserved. The revised execution requires
agreement within4FP64ulps and identicalFP32rates, with no arm/window/loss-rule
change. Within-beta-pair clocks and effective configs remain exact matches.

## Fixed validation contrasts

Let Delta_beta(eta)=L(beta.8,eta)−L(beta.9,eta). Negative values favor shorter
momentum. The interaction is Delta_beta(.04)−Delta_beta(.028).

| Validation point | Delta at LR.028 | Delta at LR.04 | Interaction |
|---|---:|---:|---:|
| Initialization | 0 | 0 | 0 |
| Step50 | −0.107564 | −0.107913 | −0.000349 |
| Final, step92 | −0.101797 | −0.105437 | −0.003640 |

Final NLLs are4.574190/4.472393 at LR.028 (beta.9/.8), and4.584631/4.479195
at LR.04. The higher LR slightly worsens both endpoints, while the beta
advantage remains about0.10. The small interactions are descriptive; a
single paired seed does not establish statistical equivalence.

## Full trajectory, including its transient interaction

Training row t measures the incoming batch before update t; validation at t
is after that update. The four arms consume the same batch at each training
row. Declared windows are token weighted, with the shortened final batch
accounted for. Windows and individual steps are not independent replications.

| Training steps | Delta at LR.028 | Delta at LR.04 | Interaction |
|---|---:|---:|---:|
| 4–23 | −0.028584 | −0.021149 | +0.007436 |
| 24–43 | −0.057131 | −0.078186 | −0.021055 |
| 44–63 | −0.096811 | −0.092789 | +0.004022 |
| 64–83 | −0.080335 | −0.079447 | +0.000889 |
| 84–92, cooldown | −0.095144 | −0.095590 | −0.000445 |

The early/middle interaction is not zero: the high-LR short-momentum arm
gains an extra0.021 in steps24–43. It does not persist through the later
windows or endpoint. Keep this transient rather than describing the curves
as identical or inferring an exact fixed clock.

`contrasts.png`/`.pdf` show every raw difference and a declared9-step centered
display smoother, against step and cumulative body LR. The broad trough
remains near the same step in both arms; plotting against the movement proxy
separates it. No fitted peak, time warp, favorable crossing or amplitude
normalization was used. This is suggestive timing evidence within a short
run, not identification of a universal phase clock or behavior after92steps.

## An exact nominal movement control

Online PD/SOAP-PD restores each matrix direction's norm to sqrt(min(rows,cols))
and multiplies by sqrt(max(1,rows/cols)). For each width512block, the four
attention matrices and MLP-down contribute5×512 squared norm; MLP-up contributes
2048. Across8blocks, the squared direction norm is36864, so the norm is192.

Thus the nominal sum of body-step lengths is192*sum(eta), independently of
beta, before weight decay and parameter-write rounding. It is neither net
weight displacement nor function-space distance. Its endpoint values are
**466.05 at LR.028 and665.78 at LR.04**, a42.86% increase. Within each beta
pair it is exactly the same nominal budget. A generic “shorter momentum
takes more total body-step norm” account is therefore excluded here.

Decay exposure is0.02427 and0.03468, with scalar retention products0.97602
and0.96591. These are far below the earlier Track3decay-equilibrium regime
at exposures near2–3. This scale comparison argues against importing that
specific explanation. It does not prove decay has no effect. Body LR and
decay exposure remain proportional, and actual parameter/auxiliary trajectories
can differ despite identical clocks.

## What changes the scientific decision

There is no large sustained beta×LRinteraction in this bracket: the~0.10
short-window advantage survives43%more nominal movement and a changed
body/auxiliary LR ratio. A simple body-distance threshold does not explain
the observed timing as well as the step-indexed view. An auxiliary or
estimator transient, optimization phase, evolving response geometry, and
directional feedback remain possible; this comparison does not choose among
them. In particular, equal configured auxiliary clocks do not imply equal
auxiliary prediction changes.

Close this bounded factorial discriminator without an extra LR sweep or a
claim that momentum should follow a universal step schedule. It strengthens
the directional/temporal interpretation of the momentum gain and constrains
an amplitude explanation, while the missing differential-feedback measurement
from`../momentum_maps/` remains unresolved. The main program's warmup and
whitening-strength tests are separate evidence; this observer does not
duplicate or manage them.

Evidence: `result.json` includes all original metadata, input hashes, clocks,
contrasts and points; `training.csv`, `contrasts.png`/`.pdf`, and`analyze.py`
are reproducible exports. Independent design and results reviews are in
`../trajectory_priority_peer/`. No model/GPU work was launched.

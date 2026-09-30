# Clipping changes the gradient mixture, but does not make its scalar history older

Snapshot: 2026-09-28 14:17:33 CDT. This is a retrospective scalar-log audit,
not a new training result or a claim about matrix directions. All42 discovered
arms are listed in `audit.json`; 40 have positive scientific-step records,
38 of those are complete, and two are partial. The records include7329 steps.
Exact counts are also obtained from the audit file below; no GPU or model
forward was used. Initial14:16:46 results are preserved separately.

## Conclusion and useful constraint

**Frequent clipping did not create an older raw-gradient momentum in PD or
SOAP-PD.** Their late pre-cooldown coefficient histories are already slightly
younger than Muon's. Reducing beta from.9 to.8 shortens Muon's measured history
at least as much, while its loss benefit remains much smaller. This rejects
the narrow explanation “whitening only needs fresher momentum because clipping
made its history longer.” It does not reject norm/direction correlations,
layer-relative reweighting, or clipping's effect on auxiliary Adam as possible
parts of the mechanism.

Clipping is a real treatment difference: across the full92-step seed260925
runs it fires18 times for Muon,87 for PD,90 for SOAP-PD, and91 for Muon with
clip.1. But clipping incidence by itself is a weak mechanism variable.

## What can be reconstructed

The frozen distributed runner clips the norm over **all model parameters**
after reduction and before either optimizer. For body matrices with the plain
momentum used here,

```
c_t = min(1, C / (n_t + 1e-6))
m_t = beta m_(t-1) + c_t g_t
    = sum_i beta^(t-i) c_i g_i .
```

The epsilon is verified in this environment's PyTorch `clip_grad.py`.
The reconstruction uses Python double arithmetic on the logged FP32 norms,
so coefficient summaries should not be described as bitwise training replay.
All retained runs start at step1 with zero momentum; contiguous steps are
checked. Pre-momentum mean whitening and PMuon are absent. Nesterov controls
are explicitly labelled as buffer-only histories and excluded from the main
table. Geometry and SOAP act **after** this momentum recurrence.

There are two distinct coefficient distributions:

1. **Raw-gradient coefficients:** normalize `beta^(t-i) c_i` by their sum.
   Their mean age is a property of the scalar kernel on the raw gradient
   vectors. It is not the age of useful descent or a vector-energy fraction.
   Writing S_t=beta S_(t-1)+c_t, the normalized raw-gradient average obeys
   `m_t/S_t = b_eff,t m_(t-1)/S_(t-1) + (1-b_eff,t) g_t`, where
   `b_eff,t = beta S_(t-1)/S_t`. This identity is for plain momentum only.
2. **Whole-gradient unit-vector coefficients:** write g_i=n_i u_i and
   normalize `beta^(t-i) c_i n_i`. The recorded n_i is the global norm; u_i's
   body-matrix restriction is **not** a unit matrix direction. When all steps
   clip, c_i n_i is nearly constant and this is almost exactly the nominal EMA.

The same-trajectory “unclipped” columns set c_i=1, leaving the observed norms
and states fixed. They are algebraic comparisons, not a no-clipping training
experiment. Norms do not identify gradient angles, cancellations, signal,
per-matrix norms, functional movement, or the nonlinear polar/SOAP output.

## Ages before cooldown

Mean across steps47–83, seed260925, same rates as the main momentum comparison.
These37 steps have the full body and auxiliary learning rates; cooldown begins
at84. Ages are in optimizer steps.

| Run | Raw-gradient age | Whole-gradient unit-vector age | Mean effective beta | Final NLL |
|---|---:|---:|---:|---:|
| Muon beta.9, clip1 | 8.716 | 9.860 | .8993 | 4.91042 |
| Muon beta.8, clip1 | 3.995 | 4.079 | .8000 | 4.89013 |
| PD beta.9, clip1 | 7.944 | 8.896 | .8911 | 4.67111 |
| PD beta.8, clip1 | 3.983 | 4.000 | .8003 | 4.54751 |
| SOAP-PD beta.9, clip1 | 8.321 | 8.896 | .8967 | 4.57419 |
| SOAP-PD beta.8, clip1 | 4.053 | 4.000 | .8025 | 4.47239 |
| Muon beta.9, clip.1 | 8.107 | 8.896 | .8959 | 4.88818 |

The nominal finite-history beta.9 age over this window is8.896, approaching
the infinite-history9 steps. Clipping generally **shortens** the raw-gradient
coefficient age in these trajectories because the scalar c rises as norms
fall. In steps10–46, beta.9 raw ages are5.586/5.735/6.265 for Muon/PD/SOAP-PD;
the finite unweighted reference is6.997. Thus the late conclusion is not
concealing an enormous early whitening-specific age inflation.

Under the whole-gradient normalization, Muon's age is longer than the nominal
EMA because its older large gradients retain higher weights after clipping
ceases. PD and SOAP-PD clip throughout steps10–83, so their unit-vector kernel
is the nominal one. Reducing beta shortens this age by5.781 steps for Muon and
4.896 for PD/SOAP-PD. A larger scalar age reduction does not predict a larger
training benefit.

## The improvement exists before cooldown

All listed beta.8/.9 pairs have matching initial-model hash, training/validation
manifest fingerprints, device and per-step body/auxiliary LR. Original versus
later16M cohorts have frozen-source revisions and disabled-option defaults;
their exact config/source changes are retained in `audit.json`. These are
recipe comparisons, not newly established isolated mechanism experiments.

| beta.8 minus beta.9 | Validation at50 | Mean train NLL47–83 | Final validation92 |
|---|---:|---:|---:|
| Muon seed260925 | −.01390 | −.00627 | −.02030 |
| PD seed260925 | −.12166 | −.10616 | −.12360 |
| SOAP-PD seed260925 | −.10756 | −.08722 | −.10180 |
| SOAP-PD seed260926 | −.10594 | −.09567 | −.10701 |

Thus the large whitening gain is established before cooldown. Muon's small
gain is not literally confined to cooldown: the step50 validation already
improves by.0139. Its step83 *training* difference is+.00908 and its late-window
mean is only−.00627, so the careful wording is “little consistent gain before
cooldown,” not “entirely in cooldown.” Training losses are on the same ordered
batches within pairs but are not an independent validation uncertainty estimate.

At beta.9, clip.1 Muon versus clip1 has validation differences+.00529 at50 and
−.02224 at92; mean train difference47–83 is−.00604. Making Muon nearly always
clipped does not by itself reproduce the large whitening gain. The beta.8
clip.1 arm is not yet complete and no outcome is inferred from its first steps.

## Token clock and the short-horizon alternative

One full16M step is16,777,216 tokens. The asymptotic raw EMA ages9 and4 steps
correspond to151.0M and67.1M tokens, or9.74% and4.33% of the1.540B-token horizon.
At4M the same betas would span37.75M and16.78M tokens: a fourfold different
data-history clock. The CSV uses actual logged cumulative tokens, including
the short final batch, rather than multiplying all ages by a nominal batch.

The old4M PD alpha1/4 beta.81 versus.9 comparison is particularly informative:
same source hashes, device, data, initial weights and LR. Beta.81 improves
validation by.0563/.0589 at steps50/100, but loses by.01177 at368; its mean
train loss over the late constant-LR window is.01717 worse. Late raw ages are
4.263 versus9.000, with negligible clipping. This is a real **early benefit
that reverses**, not simply a negative final scalar. It keeps a short-run
or transient-state explanation alive for the16M effect. The4M alpha differs
from16M alpha1/2, so it does not identify a pure batch effect.

The completed2x-horizon beta.9 runs are184 steps. Late raw/unit ages are
8.998/9.009 for Muon and8.939/8.999 for SOAP-PD; memory is now about half the
fraction of the horizon. At the audit snapshot, the new beta.8 SOAP-PD long
run has56 steps, not a final comparison. Its step50 validation difference
is−.09654 versus beta.9, which only confirms the early advantage. Later
outcomes must be read after actual completion, not extrapolated.

## A remaining connection: the clip control also changes auxiliary dynamics

The same c_t multiplies embeddings, unembedding, norm gains and body gradients.
Auxiliary Adam keeps first-moment weights proportional to beta1^k c_i and
second-moment weights proportional to beta2^k c_i^2. A time-independent scalar
largely cancels between Adam's moments (up to epsilon); a changing scalar does
not. Therefore a global clip intervention is **not a body-momentum-only
normalization control**.

For illustration, during steps47–83 the auxiliary second-moment raw-squared-
gradient coefficient ages are14.750 for Muon clip1,11.916 for Muon clip.1,
11.835 for PD and13.359 for SOAP-PD (all beta_body.9; auxiliary beta2=.95).
These are coefficient ages on squared gradients, not estimates of Adam's
actual denominator or coordinate-specific memory. They are an exact reason
to preserve the auxiliary-coupling alternative from `../body_aux/REPORT.md`.
That report's positive local body–aux interaction does not establish that
clipping is its cause.

## Two-tap history and preserved failures

The implemented filter acts after clipping and uses the previous **clipped**
gradient. Its first step is g'_1, not g'_1/2. The reconstruction includes both
facts. For an established nearly constant-magnitude input it adds only half
a step of mean age: SOAP-PD beta.9 unit age9.396 versus8.896 plain; beta.8
4.500 versus4.000. Nevertheless the filter's same-beta final penalties are
.09336 and.09313. Mean age alone is therefore a poor descriptor of this
intervention too. It changes the frequency response and delays restoring
feedback. The older observer's heavy-ball stability result is retained as
a linear-model explanation, not an exact stability law for normalized Muon.

Preserved constraints, not rewritten conclusions:

- The two-tap proposal to remove period2 oscillation failed in all three
  methods; original losses and frozen code remain untouched.
- The main notebook's13:50 withdrawal of narrow Krylov “flat-band” claims and
 14:13 withdrawal of one-step-quality explaining momentum benefit remain in
  force. Scalar coefficient ages do not rescue either claim.
- The older beta.81 PD4M endpoint failure remains a failure, despite its
  early favorable phase.
- The first clip.1 Muon arm is complete; the second clip.1 arm and long
  beta.8 SOAP-PD arm are partial at this snapshot. No jobs were launched,
  canceled, restarted or altered. Slurm steps on the existing allocations
  were verified live at14:17 CDT; a stale `status: running` alone was not
  used as completion or process evidence.

The next discriminating evidence is the existing clip.1 beta pair and the
existing longer-horizon pair after completion. If they fail to reproduce the
whitening beta benefit, clipping frequency/history length becomes a weaker
explanation. If they reproduce it, the body versus auxiliary locus and
norm/angle weighting still remain unresolved. This audit does not authorize
a training sweep or imply a new optimizer.

## Reproduction and scope

Run from the project root:

```
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python logs/observer_connections_20260928/clipping_connection/audit.py
```

The script writes only here. `audit.json` retains exact configs, source hashes,
input JSON hashes, completion checks, all pair differences and window summaries;
`step_weights.csv` retains all per-step scalar measurements. The computation
took about4 seconds on the local CPU using the Python standard library only.
`verification.json` confirms80 frozen optimizer/runner source hashes against
their execution metadata and the installed clipping epsilon formula.
Analytic finite-EMA age and direct scalar recurrences independently qualify the
kernel expansion, including the two-tap first-step exception. No norm-only
result is promoted to a direction, curvature, or training-rate mechanism.

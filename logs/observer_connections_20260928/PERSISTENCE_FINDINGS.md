# Locally aligned weak input directions do not form a steady slow gradient

2026-09-28, second observer pass. Existing saved tensors only; CPU, two
numerical threads; no forward/backward, optimizer, GPU or scheduler calls.

## The independent-group check

The first pass found that weak estimated SNR coordinates contain most body
displacement. Because those masks were selected on the scoring gradient,
their positive slope was not an independent test. Here the groups are fixed
input/output eigenvalue ranks from separate basis sequences. Gradient means
at the base and later state use disjoint sequence sets A and B. No entry is
selected or clipped according to its measured signal.

For each group, retain the signed product C=sum(g_A*g_B), raw squared norms,
per-sequence variance traces and debiased norms S_A=sum(g_A²−var_A/N). The
reported cosine is C/sqrt(S_A*S_B) only if both corrected norms are positive.
Raw cosine and corrections are also saved. These are estimates, not confidence
intervals; shared data and sequence dependence remain explicit below.

## Six saved states: two geometrically distinct input components

Input rank1 is the leading eigenvector of each matrix's input second moment.
Input ranks65+ pool every smaller input mode. They are not the top/flat
eigenspaces of the full GN operator, nor exactly global-mean/centered splits.

| Method / base step | Next-step cosine, input rank1 | Next-step cosine, ranks65+ | Ranks65+ group trace SNR at training batch | Ranks65+ body-update energy |
|---|---:|---:|---:|---:|
| Muon200 | −.427 | .252 | 2.166 | 87.2% |
| Muon500 | −.342 | .171 | 1.423 | 87.5% |
| Muon900 | −.313 | .021 | 1.444 | 87.5% |
| PD200 | −.474 | .267 | 1.877 | 97.3% |
| PD500 | −.291 | .292 | 1.036 | 97.1% |
| PD900 | −.004 | .341 | .923 | 97.0% |

The tail contains90.6% of matrix coordinates. Its energy enrichment relative
to dimension is only .965 for Muon and1.071 for PD at step500. Much of its
large energy share is dimensional, although the optimizer difference is real.
Meanwhile it contains only3.9–4.4% of raw gradient signal under Muon and
0.49–1.01% under PD. This reinforces the first pass's weighting distinction.

The dominant input mode, although tiny in coordinate count and update energy,
holds47–67% of raw gradient signal under Muon and81–88% under PD. Its opposed
next-step product dominates whole-body gradient summaries at the early and
middle states. High fixed-state SNR in this group does not imply that its
gradient is a persistent target for momentum.

Noise correction matters more in the tail than globally. Multiplying the
subtracted variance by1.25 changes the tail correlations to .268/.188/.023
for Muon and .287/.330/.390 for PD; signs persist. This is sensitivity analysis,
not a measured correction. Archived eight-sequence block-variance medians
suggest typical~10% departures from IID scaling, reaching~25% in some matrices.

At step500, next-step tail products are positive in five of six Muon matrix
kinds (V is negative) and all six PD kinds. A same-coordinate signed-vector
association with the separate frame-displacement probe is not attempted:
the saved bases are not identical and eigenvectors were not retained. Only
coarse rank-group squared energies are compared, with boundary ambiguity.

[Figure](persistence_groups.png), `persistence_compact.json`, full data in
`persistence_groups.json` and `persistence_groups_temporal.json`.

## The denser lag check rejects a simple long-lived tail

The archive also contains fresh A4000 reruns with a base at step501 and
later gradients at501+k. Apply the exact same four input-rank groups, but do
not import displacement energy from the preceding trajectories. The weaker
input group's noise-corrected cosines are:

| Lag | Muon ranks65+ | PD ranks65+ | PD matrices with positive tail product /48 |
|---|---:|---:|---:|
| 1 | .053 | .248 | 46 |
| 2 | .024 | −.161 | 11 |
| 3 | .028 | .242 | 46 |
| 6 | −.030 | −.183 | 0 |
| 7 | −.010 | −.055 | 12 |
| 14 | .038 | −.040 | 9 |
| 15 | .009 | −.020 | 19 |
| 30 | .162 | .091 | 43 |
| 31 | −.058 | −.050 | 13 |

The positive one-step product is therefore not evidence of a steady gradient
lasting anything like momentum beta .95's19-step mean age. PD's sign changes
occur broadly across matrices. Muon's one-step value is also much smaller in
this rerun than in the previous step500 state. Local phase, data forcing,
cross-layer effects and model-wide changes remain alternatives to a stable
population of long-lived useful coordinates.

There is no clean global period-two law here: for example, PD's leading
input mode is negative at lags1 and2, then positive at3. These are 1M-batch
runs; they do not refute the separate large-batch stiff-mode observations.
Nor does the rebound at30 establish30-step memory. It is one anchor and a
dependent set of lag measurements.

An independent provenance audit found no cooldown, resume, clipping,
measurement-accumulator or full-versus-weights-only loading explanation for
the late rebound. The training horizon is1469; `stop-after532` does not
compress the schedule. The actual saved base is501. Source hashes and kept
checkpoint metadata match. Batch529 is easier for both runs, and later
gradient norms vary substantially; data forcing is a competing explanation,
not an established cause. The measurement executables themselves were not
source-hashed in their output, so current-code consistency is not equivalent
to complete frozen measurement provenance.

Full signed products, corrected/raw norms and per-matrix results are retained
in `dense_input_persistence.json`; do not analyze only the cosines.

## What this supports

There is a useful separation between raw gradient magnitude, spatial update
allocation and temporal predictability. Input whitening shifts update energy
toward groups that have small raw signal and positive short-lag alignment,
but these groups are not a steady independent gradient process. Treating them
as such would skip the model's changing function and coupled dynamics.

The fixed-state SNR argument from the first pass is consequently incomplete
in two ways: it uses the wrong weights for the actual update, and it says
nothing about the temporal target. Fresher momentum can reduce lag without
making any of these signals less noisy. Published adaptive-stability and
gradient-noise frameworks are consistent with this distinction, but do not
establish a Muon-specific mechanism; see `dynamics_literature/NOTE.md`.

## Limits and next discriminator

- One scientific seed and each optimizer's own state: no causal comparison.
- All later states reuse sample B; lag errors are correlated. A and B are
  disjoint contiguous token ranges, not certified independent documents.
- Signed noise-corrected norms can be uncertain; no group/lag significance
  test treats tensor entries as independent samples.
- Persistence probes load the full next model while measuring body gradients.
  The newer step-profile probe changes only body matrices. The head and
  normalization parameters can therefore contribute to the observed gradient
  changes, an unresolved link to the attention shared-mean route.
- Per-kind energy shares in the JSON have whole-body denominators. They must
  not be mislabeled as within-kind fractions.

The next useful measurement separates full-model from body-only change on
one fixed sample and checkpoint pair. It should precede a proposal to adjust
momentum specifically in the input tail. The architecture observation in
`attention_geometry/NOTE.md` provides another interpretation to distinguish:
the model may regulate a shared bias-like value route jointly with contextual
features and auxiliary parameters. No intervention is recommended from the
present lag curves alone.

## Reproduction and review

Run `persistence_groups.py` (default step500), then with `--steps200900`
(as separate arguments: `--steps 200 900 --out persistence_groups_temporal.json`),
then `plot_persistence_groups.py`. Run `dense_input_persistence.py` for the
dense-lag tables. Set OMP/MKL/OPENBLAS thread counts to2.
The independent slice recomputation matches every fixed-rank aggregate across
all six states to <3.8e−13 relative error. Reviews and source checks are in
`persistence_peer/NOTE.md` and `FOLLOWUP.md`. Large input files have SHA256s in
the output manifests. All computations completed; no process or job remains.

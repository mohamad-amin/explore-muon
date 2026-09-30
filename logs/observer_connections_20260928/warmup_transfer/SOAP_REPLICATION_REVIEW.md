# Completed SOAP-PD fresh-seed momentum-warmup replication

2026-09-28. Read-only review of final statuses, saved step scalars,
configurations, source manifests, and frozen source bytes. No model call,
checkpoint-tensor read, GPU use, training, or additional arm. Only this
review is written; the earlier PD review remains a historical snapshot.

**Pass.** SOAP-PD on seed 260926 ends at 3.9645820762962103 with warmup and
3.9885257240384817 with constant beta .9: warmup minus reference is
**−0.023943647742271423**. This exceeds the original .01 gain criterion.
Together with PD's previously reviewed −0.02667088247835636, **both planned
cohort pairs now meet the original criterion**.

## Exact pair and integrity checks

Cohort: `logs/muon_spectra/soaudit_warmrep_20260928/`.

- `SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260926_ada/scientific/`
- `SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260926_ada/scientific/`

Both status files report complete at step 184 and 3,079,741,440 tokens.
Each `steps/` directory contains exactly steps 0–184. Both runs use four
NVIDIA RTX 6000 Ada ranks, BF16, compiled Muon NS5 plus fused auxiliary
AdamW, SOAP-PD alpha .5, body LR .028, auxiliary LR .002, batch 16,777,216
tokens, and the same final partial batch of 9,510,912 tokens.

The only configuration differences are momentum start (.8 versus disabled
−1) and momentum-warmup fraction (.5 versus disabled 0). The reference's
target and constant beta are .9. Initialization, training/validation
manifests, gradient-reduction metadata, and parameter optimizer assignments
match. Initialization SHA256:
`01fd7f349a7c93d508445860d69991628de3fc01960d8854d974139c25e2806c`.

All **25 frozen source files per arm** match their recorded SHA256 values;
the two source manifests are identical. They also match the reviewed PD
pair's source manifest. Source anchors:

- `train.py`: `a73307e7dfb51e5f17a65cf5114fcedbda25b52df639e98460baf5fda1fc5025`
- `distributed.py`: `7312fc88c8654f26b67a87046d4a530dbb46db657e65b72402a2a686212d522c`
- `muon.py`: `9178cd3b3d0b132067282952a9fc4faf151d8f382bd95c170248b2014f459f1d`
- `model.py`: `034c0f37a986b994bb78078d4f9002e21b9c5737566f8fb05d9e56ce84168b4f`

Every recorded step has identical tokens consumed, batch tokens, body LR,
auxiliary LR, world size, and dummy-sequence count between arms. The common
code defines the beta ramp in tokens consumed before each update, over half
the budget: it first reaches .9 at update 93. LR cooldown begins at update
167. Those conventions were checked in the PD review and the source bytes
here are identical.

The cohort README now includes a results section, while retaining the
original “beats beta .9 by >= .01 for both” prediction. Current README hash:
`526ce3f8aca2f8593f39b2259be9f4668d8ec8c3152b24052b765562eb9fb7d2`.
The earlier PD review preserves the pre-results README hash
`7a74bc23a49c551d8257ab198619af00b15ccee24be125ee40bde4a3b66aa94e`
and its criterion. The added outcome section does not change the decision.

## Retained trajectory, not just the endpoint

All saved validation evaluations:

| Step | Warmup | Constant .9 | Warmup minus reference |
|---:|---:|---:|---:|
| 0 | 10.937214151 | 10.937214151 | 0 |
| 50 | 5.282120556 | 5.384198703 | −.102078147 |
| 100 | 4.433242723 | 4.513542000 | −.080299277 |
| 150 | 4.118943669 | 4.151087740 | −.032144072 |
| 184 | 3.964582076 | 3.988525724 | −.023943648 |

Descriptive training-loss means use the same retrospective horizon/cooldown
windows as the PD review; they are not additional decision criteria:

| Updates | Mean paired difference | Negative differences |
|---|---:|---:|
| 1–46 | −.052559868 | 42 / 46 |
| 47–92 | −.091267718 | 46 / 46 |
| 93–138 | −.050754818 | 46 / 46 |
| 139–166 | −.030599336 | 28 / 28 |
| 167–184 | −.022779884 | 18 / 18 |

The last window's token-weighted mean is −.022749056 because its final
batch is partial. All earlier windows have equal-sized batches. Training
loss is measured before each update on the current training batch;
validation loss is a checkpoint outcome. Both show an early lead that
narrows substantially and remains at the end. They do not identify the
precise validation-gain peak or an optimal transition time.

Both arms clip gradients on steps 1–177 and are unclipped thereafter.
Equal clipping flags do not imply equal clip factors. The inference is
about the implemented schedule within this complete recipe, not an isolated
momentum-filtering mechanism.

## Relation to the existing pairs

| Method | Seed | Hardware | Warmup minus constant .9 |
|---|---:|---|---:|
| SOAP-PD, original | 260925 | Ada | −.021193206310 |
| SOAP-PD, replication | 260926 | Ada | −.023943647742 |
| PD, method transfer | 260925 | L40S | −.026670882478 |

The original pair's values and active-source qualification are retained in
`../confidence_calibration/PAIRING_REVIEW.md`; PD's qualification is in
`REVIEW.md` beside this file. These are **two seeds for SOAP-PD and one seed
for PD**, not a crossed two-method/two-seed design. Method and GPU family
also differ in the transfer comparison.

The result strengthens the doubled-horizon schedule finding and meets the
declared cohort prediction. It does not establish PD seed robustness, a
causal step-versus-token clock, or a whitening-mediated explanation. Neither
new pair includes its own constant-beta-.8 control, so their gains cannot
be described as beating both constants. The earlier 1x SOAP-PD warmup
failure and the original 2x beta-.8 comparison remain separate evidence.

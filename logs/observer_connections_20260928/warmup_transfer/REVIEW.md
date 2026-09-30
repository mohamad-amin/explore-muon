# Completed PD momentum-warmup transfer check

2026-09-28. Read-only review of saved scalar trajectories, configs, statuses,
and frozen sources. No model calls, checkpoint-tensor reads, new training,
or new probe. Only this review is written.

**The PD pair passes its original criterion:** warmup ends at
4.017883177846670 versus constant beta .9 at 4.0445540603250265, a validation
NLL difference of **−0.02667088247835636**. The cohort README predicted
at least .01 improvement for each of two planned pairs. This completes the
PD method-transfer arm; the fresh-seed SOAP-PD pair is still incomplete.

This is one seed, one learning rate, one batch size, one doubled horizon,
and matched L40S hardware. There is no PD beta-.8 doubled-horizon reference
in this comparison, so this result does **not** show that PD warmup beats
both constant momenta.

## Exact comparison and qualification

Cohort root: `logs/muon_spectra/soaudit_warmrep_20260928/`.

- Warmup: `PD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_l40s/`.
- Reference: `PD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_l40s/`.
- Decision criterion: cohort `README.md`, “The warmup beats beta .9 by
  >= .01 for both.” README filesystem timestamp precedes both completions;
  inspected SHA256 is
  `7a74bc23a49c551d8257ab198619af00b15ccee24be125ee40bde4a3b66aa94e`.

Both `scientific/status.json` files report complete at step 184 and
3,079,741,440 tokens. Each arm has exactly the scalar step records 0–184.
Both use seed 260925, PD alpha .5, nominal body LR .028, auxiliary LR .002,
16,777,216-token batches, context 512, eight width-512 layers, BF16,
compiled Muon NS5 plus fused AdamW, and four NVIDIA L40S ranks. The final
partial batch is 9,510,912 tokens in both.

The **only config differences** are:

| Field | Warmup | Reference |
|---|---:|---:|
| `muon_momentum_start` | .8 | −1 (disabled) |
| `muon_momentum_warmup` | .5 | 0 (disabled) |

Initial model hash, training and validation manifests, gradient-reduction
metadata, parameter optimizer assignments, and all 25 recorded source
hashes match between arms. Every one of those 25 source hashes matches
the corresponding frozen file in each arm. Per-step tokens, batch tokens,
body LR, auxiliary LR, world size, and dummy-sequence fields match exactly.
The shared initial hash is
`f8efae98b3d9e0dd5e9cd2318ea2b56a8b01358742af2d91e8c86ea9caf41d19`.

Source anchors within each arm's `scientific/source/`:

- `train.py:217–224` defines a momentum ramp linear in **tokens consumed
  before the update**, over half the full token budget.
- `distributed.py:298–304` sets this momentum before taking that update.
- `train.py:227–232` defines the shared LR warmup/cooldown.

Thus the implemented ramp starts at .8 on update 1, is .8991464176940777
on update 92, and first reaches .9 on update 93. The shared LR first falls
below .028 on update 167. This implementation is token-budget based;
calling it an experiment that identifies a uniquely causal step clock
would misstate what was varied.

Selected shared source hashes:

- `train.py`: `a73307e7dfb51e5f17a65cf5114fcedbda25b52df639e98460baf5fda1fc5025`
- `distributed.py`: `7312fc88c8654f26b67a87046d4a530dbb46db657e65b72402a2a686212d522c`
- `muon.py`: `9178cd3b3d0b132067282952a9fc4faf151d8f382bd95c170248b2014f459f1d`
- `model.py`: `034c0f37a986b994bb78078d4f9002e21b9c5737566f8fb05d9e56ce84168b4f`

## Full-trajectory readout

All stored validation evaluations, from `scientific/steps/stepNNNNNN.json`:

| Step | Warmup NLL | Reference NLL | Warmup minus reference |
|---:|---:|---:|---:|
| 0 | 10.919180125 | 10.919180125 | 0 |
| 50 | 5.344818577 | 5.461515300 | −.116696723 |
| 100 | 4.494535614 | 4.565436509 | −.070900895 |
| 150 | 4.176683431 | 4.209168755 | −.032485323 |
| 184 | 4.017883178 | 4.044554060 | −.026670882 |

The early lead narrows substantially but remains after the two schedules
have used the same beta .9 for 92 updates. That is a retained benefit of
their different histories, not evidence for a continuing lower-beta rate
advantage at the endpoint. Sparse validation evaluations cannot localize
the exact time of the largest validation gain.

For context, descriptive paired means of the **training-loss** difference
over all retained updates are below. These retrospective windows follow
the horizon quarters and actual cooldown boundary; they are not additional
pass criteria or independent statistical replicates.

| Updates | Mean warmup minus reference | Negative differences |
|---|---:|---:|
| 1–46 | −.042625632 | 38 / 46 |
| 47–92 | −.086130291 | 46 / 46 |
| 93–138 | −.049851115 | 46 / 46 |
| 139–166 | −.028603052 | 28 / 28 |
| 167–184 (cooldown) | −.024918162 | 18 / 18 |

The last row's token-weighted difference is −.024883900 because the final
batch is partial; preceding windows contain equal-sized batches. Training
NLL is evaluated on each current training batch before its update, while
validation NLL is a checkpoint outcome. The paired training curve supports
the same broad pattern without replacing the declared endpoint readout.

Both runs use global clipping at 1. Warmup logs clipping on steps 1–172,
reference on 1–173. Same clipping flags do not mean identical clip factors.
This is a matched intervention on the momentum schedule within the full
training recipe; the trajectory does not isolate whitening's stiff-mode
filtering, clipping, or another particular mediator.

## Scope of the completed result

The result provides positive transfer from the original SOAP-PD warmup
case to PD, at the original doubled-horizon setting and selected seed.
It does not establish a universal schedule, a causal phase clock, optimal
warmup duration, seed robustness for PD, or a spatial-versus-temporal
filtering mechanism. The earlier 1x SOAP-PD warmup failure remains relevant
and is not superseded by this successful 2x transfer.

At this review's read, fresh-seed SOAP-PD warmup was complete, but its own
constant-.9 reference remained running (latest saved step 59; the coarse
status file still reported its startup step). No endpoint comparison was
made for that incomplete pair. Therefore the README's joint prediction
for **both** pairs is not yet resolved.

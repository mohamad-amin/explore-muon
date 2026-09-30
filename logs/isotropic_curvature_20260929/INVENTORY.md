# Inventory for the isotropic-curvature trajectory probe

2026-09-29. Read-only filesystem/source inventory. No checkpoint tensors,
models, forwards, backward passes, reductions, GPUs, or jobs were run. No tiny
surrogate data or panels were read. All paths below are relative to
`/share/data/dl-theory/amin/projects/explore_muon`.

## Exact saved trajectories

- Muon arm: `logs/muon_spectra/soaudit_traj_20260926/M_lr0.007_s260925_l40s`
- PD arm: `logs/muon_spectra/soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada`

The following files exist under **each** arm's `scientific/kept/`:

| Base state | Complete checkpoint | Actual next weights | Base JSON sidecar | Base tokens |
|---|---|---|---|---:|
| 10 | `step000010.pt` | `step000011_weights.pt` | `step000010.json` | 10,485,760 |
| 500 | `step000500.pt` | `step000501_weights.pt` | `step000500.json` | 524,288,000 |
| 1300 | `step001300.pt` | `step001301_weights.pt` | `step001300.json` | 1,363,148,800 |

All three Muon complete checkpoints are 822,883,705 bytes; all three PD
checkpoints are 822,884,153 bytes. Every next-weight file is 307,828,101 bytes.
The `.json` sidecars record `step`, `tokens`, and `world_size: 4`.
Frozen `distributed.py:80` saves complete payloads with `model`, `optimizer`,
`step`, `tokens`, `config`, `metadata`, `rank_rng`. Lines 417–431 save the kept
checkpoint after its completed update and the **next update's** weights with
only `model`, `step`, `tokens`. Hence `next['model']−base['model']` is the
actual saved parameter change, including weight decay and write rounding.
It is not the checkpoint's momentum buffer or a unit polar direction.

Both arms' `scientific/status.json` say complete at step 1469 and
1,539,870,720 tokens. Both `scientific/metadata.json` have matching:

- initial model SHA256 `f8efae98b3d9e0dd5e9cd2318ea2b56a8b01358742af2d91e8c86ea9caf41d19`;
- full `source_sha256` dictionaries, train/validation manifests, model size
  76,948,992, token budget, 1469 steps, and world size 4;
- seed 260925, context 512, 8 layers, width 512, 8 heads, vocabulary 50,304,
  bias-free RMSNorm with learned gains and Q/K RMSNorm;
- batch 1,048,576 tokens, warmup 50 updates, cooldown fraction .1,
  momentum .95 without Nesterov, clipping 1, weight decay .01, auxiliary
  AdamW LR .002, betas (.9,.95), epsilon 1e−8;
- BF16 compiled training, FP32 parameters/momentum, five BF16 NS polynomials,
  and fused auxiliary AdamW under torch 2.11.0+cu130.

Differences are the intended body recipe and hardware: Muon LR .007 on L40S;
PD LR .01, α .25, uncentered two-sided input map on RTX 6000 Ada. PD enables
`track_input_stats`/`track_input_cov`, which add nonpersistent buffers. These
are matched exposures on separately evolving trajectories, not identical
states or a hardware-isolated causal comparison. At base step 10, the next
update 11 is still in LR warmup; steps 500/1300 are on the plateau.

The M/PD metadata files themselves have SHA256:

- M: `a70d3d73d2e2174d3973ce99ab2c29b551f4233491c747bb8af21384bac38223`
- PD: `eeb12d587315d0881a32c4aef2ba1ef6bb44842b80108c800349b75c57d5adc6`

## Frozen execution source and observer helper source

Training source root:
`logs/muon_spectra/soaudit_traj_20260926/frozen/adamw_spectra/`.
Each hash below was recomputed from the small source file and matches **both**
the cohort's `frozen_manifest.json` and both arms' metadata:

| File | SHA256 |
|---|---|
| `model.py` | `7bb63819666a4917ac7919a6ac16540293a9ef074a6015e1729e21ea4f6fb25a` |
| `data.py` | `99134dc89336636a65ce9eb6774c14e73843a6f9e4be4cf523bccfc0fd612124` |
| `muon.py` | `7f00a1214dd6d2fa16a5772de945a4d66528c58e629bb383a5f2df16763fe4e0` |
| `distributed.py` | `db0e1de57dbb17a2fc35febb5ef67df1bb80748b0fd9992e0a9c60c6ba5617a7` |
| `train.py` | `482a107ca762dcc490fefbbde9f44a38e5fb0678af37c02b98efd0ee242b3d2c` |

Qualified observer helper root:
`logs/observer_connections_20260928/body_aux/source/adamw_spectra/`.
The manifest is `logs/observer_connections_20260928/body_aux/source/manifest.json`:

- `model.py`: `034c0f37a986b994bb78078d4f9002e21b9c5737566f8fb05d9e56ce84168b4f`
- `gn_probe.py`: `60ddcf53466b59cbb9cd23988de6ce6da0125f4fb295af39c7dad973a9fdd391`
- `data.py`: same data hash as the training source above.

The observer model differs from the trajectory model only by the later,
default-false `track_head_cov` option and corresponding conditional head
constructor/validation. It does not alter the forward for these configs.
`P.build_model` disables input-stat buffers, loads the saved parameters,
converts explicitly to FP32, and calls `.eval()`. The old observer qualified
FP32 corner reconstruction; this is **not** prior qualification of a new
FP64 diagnostic or of its curvature implementation. The trajectory frozen
source does not contain `gn_probe.py`; use the separately frozen observer
helper rather than assuming it belonged to training.

## Requested MLP-up parameters and optimizer mapping

| Zero-based block | Exact module | Parameter key | Probe label | Weight shape | Body-group position |
|---|---|---|---|---|---:|
| 0 | `blocks.0.mlp.up` | `blocks.0.mlp.up.weight` | `block01.up` | [2048,512] | 4 |
| 3 | `blocks.3.mlp.up` | `blocks.3.mlp.up.weight` | `block04.up` | [2048,512] | 22 |
| 7 | `blocks.7.mlp.up` | `blocks.7.mlp.up.weight` | `block08.up` | [2048,512] | 46 |

`MuonAdamW.__init__` gathers body parameters from `model.named_parameters()`
using `name.startswith('blocks.') and parameter.ndim == 2`. Within each block
their order is Q,K,V,O,up,down. To map safely, zip this filtered key list with
`saved['optimizer']['param_groups'][0]['params']`, then read
`saved['optimizer']['state'][id]['momentum_buffer']`. Do not assume that the
raw optimizer integer IDs are the unfiltered `named_parameters()` positions.
The table gives positions within the body group, not a newly inspected tensor
ID. Existing `step_profile_probe.py:69–73` uses precisely this mapping.

The recurrence is `M <- .95*M + clipped_gradient`, without a factor `(1−β)`
(`muon.py:599`). The buffer stored at state s was used for completed update s;
the actual s→s+1 step uses the **next** gradient/momentum. Plain
`polar(M_s)` is therefore a lagged probe direction, not reconstruction of the
next actual parameter change. Training NS direction, exact polar direction,
and saved displacement must remain distinct. For MLP-up the training shape
factor is sqrt(2048/512)=2, applied after the NS output.

## Hook point and FP64 constraints

The MLP is `down(gelu(up(x), approximate='tanh'))`.
`up` receives `ln2(residual_after_attention)` and returns [B,512,2048] before
GELU. A module forward hook on `blocks[i].mlp.up` is therefore the clean
preactivation/output perturbation location; its first input is the matching
[B,512,512] calibration feature. A returned additive perturbation changes
that output before GELU; removing the handle restores the normal path.
Use the chosen token/sequence indexing consistently. Existing `P.Recorder`
only records detached inputs and detached gradient errors; it does not retain
the differentiable output needed for an output Hessian or inject perturbations.

Important casts in the observer source:

1. `RMSNorm.forward`, lines 52–54, calls `F.rms_norm(x.float(), ..., eps=1e-6)`
   and multiplies `self.weight.float()` before casting the result back to x's
   dtype. `model.double()` alone thus retains FP32 normalization internally.
2. `GPT.forward(tokens, targets)` casts logits to FP32 for CE. Call the logits
   path and a dtype-preserving external loss for an FP64 diagnostic.
3. `P.token_losses`, `P.sample_labels`, `P.directional_terms`, `P.gn_quadratic`,
   and frame/marginal helpers also cast floating inputs to FP32. Reusing them
   unchanged does not provide FP64 loss/curvature.
4. `P.math_attention` has no FP32 cast: matmul, scaling and softmax preserve
   the supplied floating dtype. `P.explicit_attention()` swaps SDPA for this
   causal, dropout-free expression and restores it in `finally`. This avoids
   fused-SDPA higher-order/forward-AD restrictions.
5. The network has no dropout. With `eval()` and plain Linear layers there
   are no input-stat updates. A diagnostic override must preserve epsilon,
   gains, GELU approximation, causal masking, shape and loss normalization;
   finite-radius/derivative identities must be checked on that exact function.

The ordinary model/FP32 checkpoint interpretation should remain available
for agreement checks; an FP64 diagnostic with changed cast behavior is a
separate explicitly defined numerical function.

## Which C is archived, and whether it replaces calibration

PD has `step000010_input_stats_rank0.pt`, `step000500_input_stats_rank0.pt`,
and `step001300_input_stats_rank0.pt` in `scientific/kept/`; Muon has none.
These contain rank 0's nonpersistent per-module buffers, including
`input_cov`, `input_cov_mean`, `input_cov_weight`. Training C is an EMA with
decay .998 per training forward, sampled at positions 1,33,…,481, per rank;
the optimizer's eigensystem cache is owner-local and refreshed every 10
steps. These sidecars are historical, sampled training statistics, not the
same eight new calibration sequences and not necessarily the factors used
by the owner of each matrix.

Both arms have matching-step marginal files under
`logs/muon_spectra/second_order_audit_20260926/marginals/`:
`<arm_basename>_step000010.{json,pt}`,
`<arm_basename>_step000500.{json,pt}`,
`<arm_basename>_step001300.{json,pt}`.
Their JSON confirms 2048 validation sequences beginning at token offset
2,097,152, context 512, shared across methods and states. Source
`measure_marginals.py:66–79` saves `lam_C`, `lam_B`, projected curvature
diagonals and `x_mean` for every hidden matrix; **full C is saved only for
each block's q matrix**, not for MLP-up. Thus the up entries cannot supply
the required calibrated eigenvectors/full C.

Some `gn/*_directions.pt`, `gn2/*_directions.pt`, and related archives contain
all-matrix C eigenvectors, but `one_step_gn.py:636–639` stores those in FP16,
with FP32 eigenvalues, on a different curvature bank. They do not provide an
exact C from the intended same eight calibration inputs. `step_profile_probe`
forms C freshly on its own 128-sequence bank and does not save the full C.

**Use freshly measured C on the predeclared same eight input sequences** for
the new matched diagnostic, with an explicit all-position or position≥1
convention. The archived C may contextualize the scale but cannot silently
substitute for that calibration. `TokenStream.batch(offset,8,512,cpu)` reads
4097 consecutive tokens, forms eight adjacent input windows and shifted
targets; a sequence-mean then mean over sequences equals the mean over the
4096 token losses. Record exact input/target arrays or their hashes.

## Existing probes: related evidence, not this exact question

- `logs/muon_spectra/second_order_audit_20260926/step_profile_probe.py`
  scores the **joint 48-matrix** actual step, momentum, gradient and mapped
  directions by infinitesimal GN/slope and top Ritz projections. Original1M
  M/PD JSON results exist in `step_profile/soaudit_traj_20260926__<arm>__500.json`
  and `...__1300.json` (also 100); no step10 result was found there. These
  records do not compare controlled output orientations and finite radii
  at the three requested single MLP-up modules.
- `one_step_gn.py` evaluates true loss at .5/1/2 times each direction's
  GN-optimal scalar. `normgn_probe.py` also constructs a global trust-region
  radius. These are finite-step weight-space direction/scale tests over the
  body, not the matched per-layer output-orientation/radius/time question.
- `measure_marginals.py` estimates sampled-label GN marginals across all
  original1M states. Its eigenvalue profiles are local quadratic statistics;
  they do not establish finite-radius isotropy. They are GN, not the entire
  possibly indefinite exact loss Hessian downstream of a layer output.
- `step_coupling_probe.py` and the observer's `coupling_archive/` examine
  cross-matrix response Grams. Their chosen-direction Grams do not establish
  isotropy over all output directions or replace a radius-dependent test.

A bounded source/filename search in the main audit and observer trees found
no completed probe with this exact orientation + finite-radius + selected
trajectory-state scope. This is a search result, not a claim that every
historical file was exhaustively read.

## Execution state

An ordinary sandbox `squeue` attempt could not resolve controller `beehive`
and exited 1; sandbox-local `ps` cannot establish global process state. Root
independently verified live Slurm via an authorized read-only escalated query
at 07:59 UTC: allocations 2567578/2618555 running plus tiny array 2627494.
No scheduler action is inferred from archived launchers or idle resources.
This inventory changed only this document.

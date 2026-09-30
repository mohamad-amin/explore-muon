# Saved post-switch geometry: availability and source contract

2026-09-28. Inventory only: read directory entries, file sizes, JSON metadata
and checkpoint sidecars, prior reviews, and frozen producer source. No tensor
load, geometry calculation, model, GPU, or job. Only this review is written.

**Feasible:** all seven requested arms retain the actual adjacent weight
pairs **92→93** and **165→166**, plus full model/standard optimizer state at
92 and 165. They support a future saved-weight comparison of Q/K radii and
normalized turns, and a stored-momentum carry decomposition. They do **not**
retain SOAP's full adaptive state or every owner's PD statistic/root cache.

## Exact arm inventory

All paths below are relative to `logs/muon_spectra/` and use `scientific/`.

| Label | Arm directory |
|---|---|
| PD25 warm | `soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_l40s` |
| PD25 .9 | `soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_l40s` |
| SOAP25 warm | `soaudit_momwarm16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada` |
| SOAP25 .9 | `soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada` |
| SOAP26 warm | `soaudit_warmrep_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260926_ada` |
| SOAP26 .9 | `soaudit_warmrep_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260926_ada` |
| SOAP25 .8 | `soaudit_horizon16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.8_s260925_ada` |

Every metadata config has `keep_checkpoints=[37,92,165]`. Every arm has:

| Base state | Full checkpoint and JSON sidecar | Rank0 input statistics | Actual next weights |
|---:|---|---|---|
|37|`kept/step000037.pt`, `.json`|`step000037_input_stats_rank0.pt`|`step000038_weights.pt`|
|92|`kept/step000092.pt`, `.json`|`step000092_input_stats_rank0.pt`|`step000093_weights.pt`|
|165|`kept/step000165.pt`, `.json`|`step000165_input_stats_rank0.pt`|`step000166_weights.pt`|

All sidecars agree on world size 4 and tokens 620,756,992 / 1,543,503,872 /
2,768,240,640 at 37/92/165. Full files are approximately 823 MB; next-weight
files are 307,828,101 bytes and input-stat files 176,538,717 bytes. Final
`checkpoint.json` identifies state 184 at 3,079,741,440 tokens in all arms.
No next-weight 185 file exists, and there is no dense saved model-state history
between the kept states. This inventory verifies presence and producer
semantics; a later numerical reader must verify binary metadata, keys,
shapes, dtypes, and state IDs when it opens the files.

## What persists

The newer frozen `distributed.py:80–92` saves `model.state_dict()`,
`optimizer.state_dict()`, step, tokens, config, metadata, and per-rank RNG
state. Lines 439–453 save the kept checkpoint, a separate rank0 statistic
file, then the next step's model-only weight file. The older SOAP25 reference
uses the same retention semantics.

- **Weights and learned gains:** all model parameters persist, including
  Q/K matrices and their RMSNorm gain vectors. No separate radius variable
  exists; radius is derived from the saved weights. Keys are
  `blocks.{layer}.attn.q.weight`, `.k.weight`, `.q_norm.weight`, and
  `.k_norm.weight`. For 8 heads, width 512, a Q/K head corresponds to 64 output
  rows, a 64×512 block. The head_dim gain vector is shared across heads by
  broadcasting in this architecture; it is not 8 independent gain vectors.
- **Muon buffers:** all 48 body momentum buffers and their step counters
  live in the shared standard optimizer state. These are unnormalized
  momentum sums, not applied directions or weight displacements.
- **Auxiliary Adam state:** first/second moments and step counters for its
  parameters persist through the shared `self.state` dictionary
  (`muon.py:311–321`).
- **Input statistics:** `StatLinear` buffers are explicitly nonpersistent
  in `model.py:78–86`; the separate rank0 file saves input_mean,input_sq,
  input_weight, input_cov, input_cov_mean, input_cov_weight for those modules.
  These are rank0's activation EMAs, with position 0 excluded. They are not
  a capture of each matrix owner's actual cached root.
- **Not saved:** SOAP's row/column Grams, bases, projected second moment,
  and readiness state live in `self.soap['states']`, outside optimizer state
  (`muon.py:356–365,736–752`). PD roots/cached covariances likewise live in
  `self.data_norm['roots']` (`muon.py:403–421,485–494`). No custom state_dict
  override serializes these fields. Owner-local caches and their refresh
  timing cannot be reconstructed exactly from the rank0 input-stat file.
- **Not implied by next weights:** no gradient, clipping factor, momentum,
  adaptive direction, or SOAP state for states 93/166 is saved by the
  next-weight producer. Those files contain model, step, tokens only.

Actual weight subtraction remains available without reconstructing the
optimizer. It includes scalar decay and parameter-write rounding. An
adaptive-displacement estimate would require explicit use of the next
update's recorded LR and frozen decay rule; it must remain distinct from
the total write. The model used mixed-precision forwards, but this pass
does not substitute that fact for checking stored tensor dtypes later.

## Momentum ownership and the proposed carry decomposition

`MuonAdamW.__init__:303–311` builds the body group by model named-parameter
order: every 2D `blocks.*` parameter, exactly six matrices per block. The
auxiliary group follows. `step:649–674` iterates **every** body parameter,
updates its replicated momentum and step counter, and only subsequently
partitions map computation among owners. The all-reduce communicates applied
directions; it does not leave holes in rank0's momentum state.

The trainer clips the full gradient before `optimizer.step()`
(`distributed.py:336–347`). The current and older frozen Muon sources both
implement

    M_t = beta_t M_(t−1) + g_t,clipped.

There is no `(1−beta)` factor. Prefilter, Nesterov, Newton-Muon, and PMuon are
disabled here. SOAP consumes the resulting momentum later; it does not
replace this recurrence. Source semantics therefore support all 48 buffers;
the eventual reader should still assert full group/name/shape coverage rather
than infer it from the 24-matrix spectral panel or assume arbitrary ID order.

For each warm/.9 pair, updates 93…165 use beta=.9, exactly 73 updates. In real
arithmetic,

    M_165 = .9^73 M_92 + sum_{j=93}^{165} .9^(165−j) g_j,clipped.

The paired difference has the same decomposition. Stored FP32 multiply/add
operations introduce arithmetic residuals. Defining the newer contribution
as `M165−.9^73*M92` gives an exact saved-vector decomposition, but that
remainder includes accumulated arithmetic effects unless bounded separately.
It is not a recovered sequence of the 73 individual gradients. The constant
.8 comparator uses .8^73 for its own decomposition; it is outside the
equal-beta contrast.

Small direct old-buffer carry would only exclude direct persistence of that
particular buffer component. The early schedule can still affect current
weights, radii, adaptive statistics, and newer gradients. This distinction is
the reason to pair the buffer check with saved geometry rather than declare
all treatment memory erased by .9^73.

## Timing, prior coverage, and useful scope

`progress_lead/SOURCE_CLOCK_REVIEW.md` qualifies the exact clock: update 93 is
the first common-beta update, and cooldown begins at 167. Thus 92→93 measures
the first matched-beta write, while 165→166 measures a late matched-beta,
constant-LR write. W165 contains 73 common-beta updates after W92. These two
snapshots cannot locate the time at which radii or turns converged or estimate
a continuous relaxation rate.

**Prior exploratory exposure must be disclosed.**
`next_direction_peer/GEOMETRY_REVIEW.md`, section “A small necessary-condition
screen,” already inspected Q/K weights, gains, and next writes at 37/92/165 for
the original SOAP25 warm/.9/.8 triple and reported aggregate radius/chord
and radial-cosine summaries. This is not an unseen dataset for that triple.
The PD25 and SOAP26 pairs supply the new method/seed replication extension;
the stored-buffer carry question is also distinct from the earlier screen.
Do not describe them as multiple PD seeds or treat heads as seed replication.

The completed `angular_clock/REPORT.md` instead covered the 92-update
LR×beta factorial at 9→10, 46→47, 83→84. It did not test post-switch equality.
The bounded main-notebook/source search found its citation of the angular
result, but no completed main analysis of this exact post-switch geometry.
No new optimizer replay is needed to inventory or eventually measure these
saved written-step quantities; full SOAP/PD state reconstruction is not
supported by the retained files.

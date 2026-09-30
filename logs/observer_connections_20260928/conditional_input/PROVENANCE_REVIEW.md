# Historical provenance gate: two 1M step-500 GN2 artifacts

2026-09-28. Bounded read-only source/metadata review. No model execution,
gradient-tensor loading, numerical gradient comparison, predictor computation,
or new score. Only this review is written. Guide/state read first.

**Decision: the strict historical sample-membership gate does not pass.**
The retained evidence strongly supports the intended nested-gradient design,
but does not establish the executed producer version and its sample arguments.
Do not yet describe `(4*g4M-g1M)/3` as a historically qualified disjoint 3M
gradient or execute the proposed conditional-input comparison on that basis.
This is missing provenance, not evidence that the batches were nonnested.
Stop this bounded pass rather than grow a source-reconstruction repair.

## Exact artifacts and strongest execution records

All paths below are relative to the project root. Let
`A = logs/muon_spectra/second_order_audit_20260926` and
`T = logs/muon_spectra/soaudit_traj_20260926`.

- `A/gn2/M_lr0.007_s260925_l40s_step000500_directions.pt`
  (1,082,758,069 bytes; filesystem mtime 2026-09-27 04:18:51 CDT).
- `A/gn2/PD_a0.25_lr0.01_s260925_ada_step000500_directions.pt`
  (1,082,763,379 bytes; mtime 04:18:50 CDT).
- Matching `.json` files identify respectively
  `T/M_lr0.007_s260925_l40s` and `T/PD_a0.25_lr0.01_s260925_ada`, step 500,
  with input labels `g1M`, `g4M`, `momentum`.
- `A/gn2/node_privg14_gpu0.log` and `node_privg14_gpu1.log` bind those
  respective checkpoint paths and input labels and record `done: true`.
  They contain results/progress, not the launch command or sample contract.
- `A/gn2/launcher_privg14.log` is zero bytes (mtime 03:41:29 CDT).
- `A/gn_node.sh:9` invokes **`A/one_step_gn.py`**, not a file under
  `research/adamw_spectra/`. The wrapper passes its output directory and
  caller-supplied task/option strings; it does not record those strings.
  Its mtime is 2026-09-27 02:52:24 CDT.
- `A/OBSERVATIONS.md:419` documents the second round and the selected
  Muon/PD step-500 states. It does not give sample offsets or a source hash.

The result JSON top-level keys are only `arm`, `step`, `eval_loss`, `inputs`,
and `seconds`. Each input entry contains geometry/scoring results, not input
sequence identifiers, token offsets, gradient sample counts, executable hash,
dependency hashes, or full CLI arguments. The producer's retained save code
stores only `arm` and `step` in the tensor archive's `meta` field.

## Gate-by-gate verdict

| Gate | Verdict | Evidence and limit |
|---|---|---|
| Artifact labels, intended training arm, and step | PASS at retained metadata level | Matching JSONs and GPU0/GPU1 logs identify each exact arm at step 500; no checkpoint-content hash is stored by the diagnostic. |
| Producer path | PASS | `A/gn_node.sh:9` and the notebook identify `A/one_step_gn.py`. |
| Exact executed producer bytes/options | FAIL / unresolved | The only located producer has mtime 04:55:59 CDT, after both outputs. The cached bytecode also corresponds to that later source. No execution snapshot, source hash, or complete launch command was found in this bounded search. |
| Historical g1M/g4M sizes, offset, and exact nesting | FAIL / unresolved | Current code has the desired 2,048/8,192-sequence same-offset calls; historical logs/JSON do not preserve those arguments independently. Labels alone do not certify nesting. |
| Current data-reader semantics | PASS for the retained reader | `data.py` is byte-identical to both training-source snapshots and their recorded hash; deterministic nonwrapping token slices are explicit. This does not certify that historical diagnostic call offsets were unchanged. |
| Intended sign and loss units | CONDITIONAL, well supported | `gd=-b`; gradient helper sums per-sequence mean-token-loss gradients and producer divides by sequence count. The helper predates the run, but the complete executed producer is not frozen. |
| Diagnostic model/source identity | PARTIAL | Both arm metadata files retain the intended 8-layer, width-512, context-512 model config and original model-source hash. The diagnostic imports a shared helper/model, not the arm's frozen source. Current shared `model.py` was changed later. |
| Clipping / actual next training input | NOT established by these records | The retained producer computes raw body gradients without full-model clipping and constructs a held-out next-momentum input. Missing auxiliary-gradient norms preclude qualifying an actual clipped next training input from body-only `gd`. |

“FAIL” here means the proposed interpretation has not met its declared gate;
it does not assert that the archived values themselves are wrong.

## What the current source would mean, conditionally

The following describes the inspected producer, **not a recovered execution
contract**.

`A/one_step_gn.py:57–60` sets

    BASE = 2 * 1,048,576
    GRAD = BASE + 80,000 * 512 = 43,057,152

At lines 436–453 it loads the selected kept checkpoint once, constructs an
FP32 evaluation model through `gn_probe.load_checkpoint`, selects all 48
body matrices, and leaves embeddings, head, and gains outside the gradient
dictionary. At lines 455–457 it computes g4 from 8,192 sequences and g1 from
2,048 sequences at the same GRAD offset. The `fresh_gradient` calls use their
own default microbatch size 8; the CLI's curvature/scoring `--micro` option
is not forwarded to these calls.

Both frozen arm configurations have `seq_len=512`. Under these calls,
zero-based half-open **input** token intervals would be:

| Input | Sequence count | Token count | Input interval |
|---|---:|---:|---|
| g1M | 2,048 | 1,048,576 | `[43,057,152, 44,105,728)` |
| g4M | 8,192 | 4,194,304 | `[43,057,152, 47,251,456)` |
| Complement within g4M | 6,144 | 3,145,728 | `[44,105,728, 47,251,456)` |

Targets are shifted by one token. `TokenStream.batch` reads one extra token,
so adjacent input/target blocks share the expected boundary token; their
scored target positions do not overlap. All sequences use the same length
and chunk boundaries. These are contiguous corpus blocks, not independent
random draws.

`research/adamw_spectra/gn_probe.py:126–139` differentiates
`losses.mean(1).sum()` with respect to body weights. The producer's
`fresh_gradient:257–266` sums these microbatch gradients and divides by the
number of sequences. With constant sequence length this is the ordinary
mean-token cross-entropy gradient. `closed_form_directions:279` assigns
`gd=-g`; no learning rate, momentum factor, or per-matrix norm belongs in
that sign reversal. The retained save code at line 628 converts the chosen
directions to BF16. Consequently the disjoint-complement identity would
hold before numerical accumulation/storage error if historical nesting
were established; it does not become exact for the rounded saved values.

The current producer's constructed `momentum` uses `.95*M+fresh` at
lines 463–468; `--momentum-gradient` defaults to g1M but is not recorded in
these result files. Both selected training metadata configs do record
`muon_momentum=0.95`. Buffer ownership and exact recurrence are being
reviewed separately by the root agent. No `(1-beta)` EMA factor should be
introduced merely to make the buffer appear normalized.

## Source preservation evidence and its limits

Current inspected SHA256 values (identifiers for this review, **not execution
hashes recovered from GN2**):

- `A/one_step_gn.py`:
  `620d043a5dddacb235cfcbcfcfee44d66c3b30acdd2dad59ad9c57aaeeab729b`
- `A/gn_node.sh`:
  `c03f2a4f507e70f9feee4e7ad15aa2d2f9b1610b8440c204f72c2a78cc1d96ff`
- `research/adamw_spectra/gn_probe.py`:
  `60ddcf53466b59cbb9cd23988de6ce6da0125f4fb295af39c7dad973a9fdd391`
- `research/adamw_spectra/data.py`:
  `99134dc89336636a65ce9eb6774c14e73843a6f9e4be4cf523bccfc0fd612124`

The producer's `.pyc` timestamp header is 1790502959 with source size 36,901,
matching the later 04:55:59 producer; the cache file itself is from 05:55:10.
It therefore does not recover the 03:41 execution source. The notebook
explicitly describes subsequent producer additions (`OBSERVATIONS.md:482`),
so treating this live file as an unchanged frozen snapshot would be unsafe.

The shared `gn_probe.py` has mtime 2026-09-27 01:49:20 and matching cached
source timestamp/size, predating the run. `data.py` has mtime September 24
and matches the hash and bytes in both selected arms' `scientific/source/`.
Those are useful corroborations for sign/units/reader semantics. Neither
arm's 23-file training source manifest includes `gn_probe.py` or the
diagnostic producer. Both preserve model hash
`7bb63819666a4917ac7919a6ac16540293a9ef074a6015e1729e21ea4f6fb25a`;
current shared `model.py` differs and has mtime September 28 11:17:06.

The two training validation manifests identify the same one-shard,
100,000,000-token FineWeb validation stream, size 200,001,024 bytes and
mtime_ns 1787181688651489935. They predate the diagnostics and corroborate
the intended data family; the diagnostics themselves do not record a data
manifest or input hashes. The historical path in those manifests was read
as metadata only; no out-of-scope study was accessed or changed.

The bounded search covered the audit directory's producer, source-backup
inventory, launch wrappers, GN2 logs/results, the matching arm metadata and
frozen source files, and the helper caches. No earlier producer copy or
explicit GN2 CLI record was found. The workspace is not a Git repository,
so no version history was available there. This is a scoped negative search,
not proof that no external execution transcript exists.

## Consequence for the next decision

Do not spend compute on cancellation statistics while the key disjointness
claim remains conditional. Existing results can still be described by their
recorded g1M/g4M labels, with current-source nesting explicitly identified
as an assumption. That weaker statement does not meet the proposed
disjoint-complement discriminator's gate. No new model replay, broad
archive search, source-history excavation, or alternative endpoint is
recommended in this pass. A later externally supplied execution snapshot
or equivalent contemporaneous record could reopen the gate.

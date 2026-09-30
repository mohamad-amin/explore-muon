# Tiny surrogate qualification

This branch asks whether a small language model reproduces the larger study's
optimizer ordering, increasing geometry benefit with batch size, and shorter
useful momentum at larger batch. A successful LR screen alone is insufficient.
The decision, criteria, and bounded expansion rule are in the latest tiny
surrogate entry of [PROTOCOL.md](../../PROTOCOL.md).

The first candidate is a four-layer, width-128, context-128 character GPT on
commit-pinned Tiny Shakespeare. It uses the existing pure model and optimizer
kernels, with a standalone data/schedule/training loop. No original FineWeb
trainer, DDP runner, or full-pipeline controller is invoked.

- `model.py`: small GPT and covariance statistics pooled once per update.
- `data.py`: explicit pinned download, source checksums, disjoint corpus splits,
  deterministic paired character windows. Training windows intentionally repeat.
- `optim.py`: common auxiliary AdamW; AdamW/Muon/PD/TS/SOAP variants on body
  matrices; predictive-label GN statistics; complete analysis snapshots and
  unique tensor-storage memory accounting.
- `train.py`: fresh runs only, fixed validation bank, full final validation,
  distinct direction and parameter-change norms, runtime/memory/hardware logs.
- `cohort.py`: immutable source copies, arm configs, manifest checking, Slurm
  array submission, and refusal to resubmit an existing cohort.
- `analyze.py`: all-arm discovery report and curves (no single-seed claim).
- `stories.py`: train-only BPE preparation, nonrecycling training blocks, and
  document-respecting evaluation including short stories and tails.
- `confirm.py`: pre-training family locks, complete-family verification before
  inference, sealed scoring, and release only after every declared run finishes.
- `clock_diagnostic_report.py` and `scale_diagnostic_report.py`: readouts of the
  separately declared diagnostics, retaining all arms and comparison gates.

Only cluster GPUs with **strictly less than 48 GB VRAM** are permitted. Execution
has been qualified on 16 GB RTX A4000 and 11 GB RTX 2080 Ti; paired scientific
comparisons stay on one hardware type. The runner rejects devices
with >=45 GiB physical memory, also excluding nominal 48 GB cards with slightly
less usable capacity. CPU is allowed for unit and integration qualification.

Run the scientific-contract checks from this study's root:

```bash
./run -m unittest research.tiny_spectra.test_model_data research.tiny_spectra.test_optim -v
```

The frozen first GPU smoke and discovery cohorts live under
`logs/tiny_spectra/`. Inspect their `submission.json`, live Slurm handles and
`runs/*/status.json` before any continuation. Do not infer authorization from
old launchers or README status lines.

The local recipe deliberately pools covariance over each optimizer batch,
then applies EMA 0.9 once per update. Covariance GEMMs are FP32, forward/backward
autocast is BF16, parameters and momentum are FP32. This prevents accumulation
count from silently changing the statistic clock. TS uses predictive-label GN
statistics; a data-label empirical-Fisher arm would have a different name.

`final.pt` contains all supported external SOAP/PD/TS state and input buffers
for analysis. This is **not** a qualified resumable-run format. Failed or
interrupted attempts remain separate from any replacement run.

## Independent confirmation

Both real confirmation populations remain unscored. Existing character and
TinyStories development candidates are unqualified; the existence of a scorer
does not authorize opening either panel. Freeze the complete scientific family
and analysis in the protocol before creating a lock or training that family.

For TinyStories, a plan must explicitly name panel kind
`tiny_stories_byte_bpe_v1` and primary metric `token_weighted_nll`. The scorer
requires the pinned all-target v2 test and training manifests, matching tokenizer,
vocabulary and membership identities, and frozen model/data source hashes.
It pools document-respecting windows into batches, masks only padding, retains
real EOS targets, and reports token-weighted and equal-document means. Every
saved checkpoint uses the full BPE test population; the character panel retains
its separately declared curve/full-window policy.

Synthetic contracts can run without reading or scoring either real panel:

```bash
./run -m unittest research.tiny_spectra.test_confirm research.tiny_spectra.test_confirm_stories -v
./run -m unittest research.tiny_spectra.test_confirm_binding -v
```

The independent adapter review is in
`logs/tiny_spectra/confirmation_adapter_20260928/independent_stories_data_review.json`.
GPU scoring throughput is unbenchmarked. A hard-killed attempt needs audited
recovery of its existing active guard; do not bypass the identical-family retry
rule. Expanding training data requires a new explicit compatibility audit before
changing the pinned training identity. The code verifies the declared family;
scientific adequacy and avoiding reuse of a consumed panel remain protocol
requirements.

The scorer also binds each canonical panel to one lock and complete weight
family before its first inference. It publishes an atomic, non-overwriting
receipt in the panel's `.confirmation/family.json`; run-attempt records share
that panel-local directory. A new lock, changed weights, or a copied lock cannot
reset that history. Failed scoring retains the binding and permits only the
same family's retry. Collection verifies the receipt without recreating it.
Real panels must use their canonical resolved paths; compatibility symlinks are
accepted. No binding has been created for either real panel.

The combined 38 synthetic tests, including concurrent claims and malformed
receipts, are recorded in
`logs/tiny_spectra/confirmation_adapter_20260928/panel_binding_review.json`.
These controls prevent accidental panel reuse; they do not establish scientific
adequacy or protect against deliberate deletion of audit metadata.

# Train one experiment on 1, 2 or 4 GPUs

The new runner uses one node and 16 GB A4000 GPUs. Each run keeps the configuration's **global** batch, token budget, learning rate and four-sequence microbatches. Adding GPUs does not multiply the batch or learning rate.

The scripts are implemented. The public two- and four-GPU submitters completed real Slurm jobs and passed first-update numerical checks against the deterministic native reference. CPU contracts also pass.

An extended numerical qualification campaign was stopped after the user clarified the requested scope. Its completed and partial artifacts are preserved. Full all-method or long-trajectory numerical qualification is **not claimed**. The default CUDA profile showed native repeat variability; the supplied launchers default to the deterministic profile used by the successful two-/four-GPU checks.

Run these commands from `tiny_surrogate/`. Use a new output directory for every run.

```bash
# Submit one run using two GPUs.
./scripts/submit_2gpu.sh configs/multigpu/spd.json logs/tiny_spectra/my_spd_2gpu

# Submit one run using four GPUs.
./scripts/submit_4gpu.sh configs/multigpu/spd.json logs/tiny_spectra/my_spd_4gpu

# Inspect the frozen job and Slurm command before submission.
./scripts/submit_4gpu.sh configs/multigpu/spd.json logs/tiny_spectra/preview_4gpu --dry-run

# Submit that already-prepared job without changing its files.
./run -m research.tiny_spectra.multi_gpu submit-prepared --out logs/tiny_spectra/preview_4gpu

# Run directly inside an allocation with at least the requested visible GPUs.
./scripts/run_multi_gpu.sh 2 configs/multigpu/spd.json logs/tiny_spectra/local_spd_2gpu
```

The generic submitter also accepts `1`, `--time HH:MM:SS`, `--node NODE` and `--profile deterministic|native`:

```bash
./scripts/submit_multi_gpu.sh 4 configs/multigpu/spd.json logs/tiny_spectra/custom_run --time 02:00:00
```

The default is `deterministic`: the launcher sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization and every worker enables strict deterministic algorithms and deterministic cuDNN behavior. The setting is recorded in the frozen configuration and runtime metadata. See [PyTorch’s deterministic-algorithm contract](https://docs.pytorch.org/docs/2.11/generated/torch.use_deterministic_algorithms.html). `--profile native` is retained explicitly for diagnosis/compatibility; it did not pass the declared fidelity checks. Determinism does not make different GPU-count reductions bitwise identical.

`spd.json` is the current 4.86M-parameter, 96M-target FineWeb working recipe at the large batch, with body momentum 0.8. `sts.json` changes the method to SoapMuon+TS; it is a starting configuration, not a tuned or independently confirmed winner. The original training/development data identities are pinned in both files.

## What each submitted job does

The submitter freezes the source, configuration and generated `job.sh` before calling Slurm. It requests `--nodes=1 --ntasks=1 --gres=gpu:nvidia_rtx_a4000:N`, two CPU cores and 8 GiB RAM per GPU. That one task uses [PyTorch torchrun](https://docs.pytorch.org/docs/2.11/elastic/run.html) to launch N local workers. Every rank checks that its GPU is strictly below nominal 48 GB. There are no automatic retries or resumes.

Forward/backward work is split across complete, contiguous original microbatches. Weighted gradients are summed before a single global clipping operation. Rank zero owns the complete optimizer, including SOAP moments, PD roots and TS output statistics; it updates parameters and broadcasts them to the workers. This avoids the original optimizer's implicit owner sharding. Evaluation and saving remain on rank zero.

Input statistics preserve the global microforward clock and order in real arithmetic through affine EMA composition. FP32 summation/composition order can change numerical results. Worker statistic buffers are temporary local accumulators; the saved global statistics and all optimizer state belong to rank zero. The model's forward-relevant position buffer is deterministic and unchanged during training.

Keep GPU count fixed within a scientific comparison. Check measured wall-clock speedup **and GPU-hours**: the optimizer and evaluation are serial, so four GPUs need not provide fourfold speedup or lower total compute cost. Short execution checks do not prove full-trajectory identity or optimizer-ranking generalization.

## Outputs and failures

- `RUN_PLAN.json`, `config.json`, `source_manifest.json`, `frozen/`: immutable launch identities.
- `submission.json`: the Slurm handle and exact request.
- `console/slurm_JOBID.log`, `console/training.log`: launcher and worker output.
- `run/metadata.json`, `run/metrics.jsonl`, `run/summary.json`: hardware, timing and training records.
- `run/final.pt`: complete rank-zero model, optimizer and global statistics.
- `run/final_validation.pt`: saved development-window losses and target identities.
- `EXECUTION_COMPLETE.json` or `EXECUTION_FAILURE.json`: authoritative launcher completion. A failed worker can leave an earlier `run/status.json`; use the launcher record and Slurm state to determine termination.

An existing output directory is never overwritten. A failed run remains available for diagnosis. Independent test panels are not available through this training interface.

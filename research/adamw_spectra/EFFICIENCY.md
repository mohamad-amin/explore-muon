# Efficient execution — 2026-09-24

The user requested an efficient implementation of the same AdamW study.
No scientific training run or LR pilot was launched by this work.

## Changes

- CUDA defaults to `torch.compile(model, dynamic=False)` for training and
  validation, with the normal Inductor mode, and standard PyTorch fused
  AdamW. CPU stays eager and uses foreach AdamW. The original eager/single
  implementations remain selectable. Hyperparameters and model equations
  are unchanged; fused kernels need not preserve every floating-point bit.
- Read/cast each accumulated token batch directly into one allocation and
  transfer it once using pinned memory. Microbatches are views on the device.
  With the default 16-sequence microbatch, this replaces 128 host-to-device
  transfers per optimizer step with one. The held-out token bank is cached
  once (8 MiB), eliminating 128 repeated transfers per validation evaluation.
- Cache position indices. Read the 24 pre-step weight norms together and
  read optimizer counters together (fused AdamW stores them on CUDA), avoiding
  a scalar device synchronization for each matrix. Update reconstruction
  reuses private temporaries; optimizer buffers remain unmodified.
- Keep exact direct SVD, normalized quantile definitions, first/final samples,
  and the requested 25-update interval. No Gram-matrix approximation,
  randomized SVD, change of sample selection or change to the adaptive update.
- Log effective optimizer/compiler choices and compiler paths; check them on
  resume. Use the native C/C++ toolchain recorded by this Python interpreter.
  The inherited Conda compiler failed to find this system Python's headers;
  selecting the matching compiler fixes that local compiled-execution issue.
- Include setup in total invocation time. Compilation costs are not silently
  omitted, and remain distinct from steady-state step measurements.

## Verification

- **11 CPU tests passed**, including numerical writes for single, foreach
  and fused AdamW; batched measurement equivalence/non-mutation; shard-crossing
  cached microbatch views; exact CPU resume; and analysis behavior.
- **53-update comparison against the saved original implementation:** same
  initial model hash, bit-identical final CPU parameters, identical training
  and final validation NLL, and spectra at steps 1/25/50/53.
- **Tiny CUDA comparison on L40S:** compiled versus eager FP32 gradients had
  maximum global relative L2 difference `3.17e-7` across three updates; each
  fused AdamW write matched reconstruction within the declared numerical
  tolerance. BF16 compiled/eager loss difference was zero in the check, with
  global gradient relative L2 difference `0.001412` (0.1412%). Post-fused-step
  spectra were finite and normalized. Peak allocated CUDA memory was 115 MiB
  rounded up. These small-model checks do not certify a long-run endpoint.
- The initial CUDA test used a pure parameterwise relative-gradient criterion
  that was inappropriate for nearly zero key-bias gradients: their norms
  were about `3e-11` to `5e-11`. Raw failure logs are retained. Qualification
  instead checks each gradient with absolute plus relative tolerance and the
  global gradient relative norm; all three FP32 steps passed. The separate
  compiler failure and successful rerun are also preserved.
- **Tiny CUDA end-to-end runner:** BF16, compilation, fused AdamW, pinned batch
  upload, cached validation, exact spectra, and checkpoint writing completed
  three updates, including the shortened last batch, at the intended 160
  synthetic tokens. First/last spectral sampling worked.
- A warmed CPU data-read microbenchmark over 40 alternating repetitions
  decreased median 1,048,577-token read/cast time from 0.327 ms to 0.254 ms.
  This measures only a small data-loading operation, not total training.

All evidence lives under `logs/adamw_spectra/efficiency_20260924/`:
`tests.txt`, `reference_comparison.json`, `data_benchmark.json`,
`cuda_check.json`, `cuda_check_initial.log`, `cuda_check_compiler_failure.log`,
`check_cuda.py`, `tiny_cuda_run/`, and `final_resume_check.txt`.
The `before/` directory preserves the earlier implementation for comparison.

Other GPU work was active during CUDA qualification. A clean 77M throughput
benchmark is outstanding; no end-to-end speedup factor is claimed. The tiny
CUDA check qualifies the execution path, not full-size memory/throughput or
the still-provisional learning rate.

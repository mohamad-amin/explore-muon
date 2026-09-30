# TS16M frozen implementation and checkpoint audit

Read-only source/checkpoint audit for the observer's proposed repeatability
measurement. No model was instantiated or evaluated, no GPU was used, and no
job was submitted. Python checkpoint reads used CPU mmap, with numerical
threads capped at two and bytecode disabled. Live Slurm check showed the main
allocations 2567578/priv-g14 and 2618555/g20 plus tiny-surrogate job2626637_3/g9;
none was touched.

Reference run:
`logs/muon_spectra/soaudit_batch16m_20260927/TS_a0.5_b0.5gn_b16M_lr0.028_mom0.9_s260925_l40s/scientific/`.
Source references below are relative to the same cohort's
`frozen/adamw_spectra/`. Configuration is the arm's `scientific_config.json`.

## Mapping from momentum to the direction

- All48 body matrices use plain momentum `M <- .9 M + clipped_global_gradient`,
  no Nesterov (`muon.py:619–643`). Global clipping threshold is1 after DDP mean
  reduction (`distributed.py:321–325`). Momentum is FP32, not normalized EMA;
  its global scalar convention cancels in the normalized NS direction.
- Input moment C is the bias-corrected uncentered EMA, normalized by mean
  eigenvalue; `R = (C/mean_eigenvalue(C)+.001 I)^(-.5)`.
  Symmetrize and eigendecompose in FP64, clamp negative eigenvalues to0,
  reconstruct and cast R toFP32 (`muon.py:243–262`).
- Output factor root `L = (B/mean_eigenvalue(B)+.001 I)^(-.5)` follows the
  same FP64 eigen / FP32 reconstruction convention (`muon.py:482–498`).
- Construct `X=L M R` in FP32; Frobenius normalize X with minimum denominator
  1e-7, cast toBF16 **on CUDA only**, then apply the five ordered quintics below.
  The recurrence is `A=X X^T; X=a X+(b A+c A^2) X`. Transpose before NS when
  rows>columns, then undo that transpose and return FP32 (`muon.py:14–63`).

| Polynomial | a | b | c |
|---|---:|---:|---:|
|1|4.0848|-6.8946|2.9270|
|2|3.9505|-6.3029|2.6377|
|3|3.7418|-5.5913|2.3037|
|4|2.8769|-3.1427|1.2046|
|5|2.8366|-3.0525|1.2012|

- Postmultiply NS result by R and premultiply by L, both FP32. Per matrix,
  rescale to Frobenius norm `sqrt(min(rows,cols))`, then apply shape factor
  `sqrt(max(1,rows/cols))` (`muon.py:690–741`). Up matrices2048×512 therefore
  get2; all512×512 and down512×2048 get1.
- No SOAP, row normalization, centering, deflation, magnitude matching, or
  geometry decay applies in this arm. The parameter update is
  `W <- (1-lr*.01) W - lr*D` (`muon.py:814–823`); lr=.028 at retained step46.
  D, momentum, and rounded parameter displacement remain different objects.
- Matrices are grouped by shape in body parameter order, assigned owners
  round-robin **within each shape**, and computed in owner batches of at most8.
  Owners share disjoint FP32 direction slices by all-reduce. Thus owner0 has
  q at every layer and up/down at layers0,4. k/v/o belong to ranks1/2/3;
  MLP ownership is layer modulo4 (`muon.py:643–687,794–805`).

The native frozen `newton_schulz` on CPU uses FP32 arithmetic even when the
scientific config saysBF16. Explicit BF16 CPU recurrence must be requested to
approximate training operation boundaries; it still does not establish CUDA
bitwise replay. A harmless64×96, two-thread CPU BF16 five-quintic check was
finite and completed in0.4602s. No model compute occurred.

## Exactly what the output statistic measures

`distributed.py:301–310` takes the first8 sequences of **each rank's** local
current training batch, before ordinary gradient accumulation. At16M this is
8 of8192 local sequences, not8 globally or32 pooled. Refreshes are steps1,3,…
and use seed `260925*1000003 + 97*step + rank` on a device generator.

`muon.py:832–870` switches the model to eval (restoring mode afterward),
hooks every body linear output, computes logits using the trainer's AMP
context, samples one label independently for each position from current
model probabilities, and computes summed cross entropy. Autograd is requested
only with respect to the hooked outputs, leaving parameter `.grad` untouched.
For each body module,

`B_new = e.reshape(-1,dout).T @ e.reshape(-1,dout) / (8*512)`.

All512 positions, **including0**, enter this statistic. e at one hidden token
is the derivative of the **sum of all position losses** with respect to that
activation, hence includes downstream attention pathways from other positions.
It is not the derivative of only that token's CE. The factor is a second
moment about zero, with no mean subtraction. CUDA autocast gives BF16 linear
outputs/hidden errors, then e is castFP32 for its Gram. OnCPU the frozen AMP
context is null (`train.py:231–234`), so an ordinary CPU probe is FP32.

An independent sequence cannot attend to another sequence; therefore, summed
CE gradients per sequence and pooling `e.T e/512` across8 sequences reproduce
the same mathematical B. Independent per-sequence label draws permit label
Monte Carlo noise to be separated from sequence variability. Fixed label
sampling seeds need not reproduce the CUDA categorical RNG stream onCPU.

B's first observation is copied directly, then subsequent calls use
`.8 B_old + .2 B_new`; roots are invalidated after every refresh
(`muon.py:472–480`). No B all-reduce is present. Only the matrix owner's B is
used in its final direction. This is raw-B averaging **before** normalization
and inverse power, not averaging individually normalized roots or directions.

Input C has a different estimator: samples positions1,33,…,481; decay.998 per
training forward; bias correction by EMA weight. C is not updated during the
B eval pass (`model.py:60–100`). C and B roots both refresh every2 updates,
but C includes all local microbatch forwards preceding the optimizer step.

## Available saved evidence at46

CPU mmap inspected `kept/step000046.pt` and
`kept/step000046_input_stats_rank0.pt` directly. The model is after46 updates,
at771751936 training tokens.

- Checkpoint keys: model, optimizer, step, tokens, config, metadata, rank_rng.
-48 body optimizer states each contain exactly momentum_buffer and step.
-36 auxiliary optimizer states contain step, exp_avg, exp_avg_sq.
- No C/B/R/L/SOAP buffers exist in model or optimizer state.
- Separate rank0 statistics file contains all48 modules, each with input_mean,
  input_sq, input_weight, input_cov, input_cov_mean, input_cov_weight, FP32.
- This rank0 C is from the actual owner only for q and layers0,4 up/down.
  Rank0 C for other matrices is an independently sampled proxy.
- Even for owner0, C saved after step46 is not the R cached at step45 and used
  at46. The next R refresh is47 and would include the next training batch.
- Historical B/L, other owners' C, and cached R are not saved. Actual online
  direction/history and lag cannot be reconstructed from these files.

## Recommended validation boundary

1. Call the proposed measurement fixed-state estimator repeatability, with
   fixed saved M and explicitly specified reconstructed/proxy R. Do not call
   it a replay of the actual step or online-EMA stability.
2. Specify whether observed B is fresh8-sequence B or a synthetic EMA. Fresh
   B variability is not measured variability of the already-smoothed online B
   (nominal stationary weight ESS9 refreshes). It is not automatically an
   upper bound on direction variability under changing states and nonlinear
   inverse-power/polar maps.
3. Freeze sequence banks, label seeds, all-position reduction, root damping,
   per-layer norm scaling, and PD reference before scoring. Preserve B raw
   magnitude; pool raw factors before normalization/inversion.
4. Check positive finite traces, symmetry, PSD to eigensolver tolerance,
   output shapes, intended Frobenius norms, and no parameter or C writes.
   Report FP32 versus explicit BF16 NS direction error on a bounded sample.
5. Fixed M/R isolates B's estimator sensitivity. Direction disagreement alone
   does not prove poor generalization; independent score-bank slope/curvature
   is needed to judge improvement. Training benefit still requires a separate
   sustained trajectory comparison and is not authorized by this audit.

Reproducible schema/source evidence: `inspect_schema.py` writes `schema.json`.
All four relevant frozen source hashes match the run's original metadata.
Run with the existing `.venv/bin/python`, PYTHONDONTWRITEBYTECODE=1, and
OMP/MKL/OPENBLAS_NUM_THREADS=2. The retained checkpoint bytes were not fully
hashed; mmap schema inspection does not touch most tensor payloads.

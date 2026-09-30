# Eight-GPU 12- and 20-layer AdamW spectrum runs

User request, 2026-09-24: submit two cluster jobs using eight GPUs each;
prefer 96 GB RTX Pro GPUs for 20 layers, and allow 48 GB or RTX Pro for 12.

## Decision argument

The user requests extending the existing descriptive AdamW update-spectrum
study to the paper's next two model configurations: 12 layers/width 768/12
heads and 20 layers/width 1024/16 heads, both at context 512. The existing
77M run is executing from frozen sources and must continue independently.
The larger runs can reveal whether update quantile trajectories and depth
differences persist with scale; comparing them with Muon's pre-NS momentum
still does not isolate an optimizer effect. Ordinary dimension changes and
adaptation to different learning rates are competing explanations for any
spectral trend. Do not claim scaling exponents from an isolated visual fit.

Keep the reconstruction's architecture family, untied vocabulary, next-token
NLL, global 1,048,576-token batch, 20N training budget, AdamW parameters and
25-update sampling cadence. Each size gets the same bounded three-rate
300-update pilot protocol, followed by one selected-rate scientific seed.
Use eight replicated DDP ranks, accumulating locally with one synchronized
gradient update. Weight gradients by actual token count, including uneven
last batches, to reproduce the global mean loss before clipping. Measure
the same 24 relative-depth matrices; distribute independent direct SVDs
across ranks and gather complete spectra without changing the updates.
Qualify partitioning and resume on CPU and perform bounded CUDA qualification
inside each allocation before scientific execution. The budgets are about
3.254B and 7.109B tokens; stage sufficient data in a separate directory so
the running 77M job's file manifest is unchanged. Each submitted allocation
is bounded to the cluster's eight-hour limit, with checkpoints for recovery.
No extra seed, candidate expansion, all-parameter panel, or Muon arm is added.

## Frozen choices

- 12-layer model: 162,716,160 parameters, 3,254,323,200 tokens, 3,104 updates.
  Sample blocks 3/6/9/12, Q/K/V/O/up/down.
- 20-layer model: 355,473,408 parameters, 7,109,468,160 tokens, 6,781 updates.
  Sample blocks 5/10/15/20, Q/K/V/O/up/down.
- Main seed 260924; pilot seed 260923. LR candidates 3e-4/6e-4/1.2e-3;
  selection mean NLL at updates 200/250/300; exact tie chooses lower rate.
- Two separate single-node, eight-GPU jobs in the `gpu` partition, eight-hour
  limits. Slurm 22.05 lacks `--prefer`: request `96g` for the 20-layer job
  and `48g` for the 12-layer job (the cluster's 96g nodes also advertise 48g).
- DDP arithmetic can change reduction rounding; CPU equivalence is checked
  numerically. Global batch, token order, budget and objective remain fixed.
- Keep per-size source/config snapshots and distinct output/checkpoint paths.

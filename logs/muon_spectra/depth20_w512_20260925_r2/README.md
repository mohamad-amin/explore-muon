# Muon depth 20, width 512

Authorized on 2026-09-25 UTC. 4 GPUs; global batch 1,048,576 tokens;
1,539,870,720 total tokens; 1,469 updates. Momentum and captured post-NS update
spectra for the same 24 body matrices every 25 steps plus first/final (60 each).
Muon LR .01, auxiliary AdamW LR .002, constant momentum .95, five paper NS
polynomials. Width/context 512, heads8, seed260924, scientific warmup50.

Frozen protocol: [MUON_CASE.md](frozen/adamw_spectra/MUON_CASE.md).
Independent discussion: [MUON_REVIEW.md](frozen/adamw_spectra/MUON_REVIEW.md).
Controller performs tiny distributed and five-step peak-rate qualification,
then a fresh scientific run and both analyses, within 7h45m.
`pipeline.json` records exact job/step/PIDs/stages. The five-step qualification
has warmup1 as an engineering stress check; scientific warmup remains50.

Updates: `scientific/spectra`, `scientific/plots`, `scientific/summary.json`.
Momentum: `scientific/spectra_momentum`, `scientific/plots_momentum`,
`scientific/summary_momentum.json`. Raw per-step metrics are in `scientific/steps`.

Final revision uses one owner per matrix for NS and shares the direction.
NS is eager and batched; the model is compiled. Prior8-layer tiny CUDA
failure and diagnostic compiler disagreement are retained in the original
run root and qualification directory; neither reached scientific training.

Original eight-GPU submission **2620619** was canceled before launch.
Replacement four-GPU job **2620623** uses `job_4gpu.sbatch`; see
`submission_4gpu.json`, `gpu_count_change.json` and `pipeline.json`.

GPU-count amendment, 2026-09-25 01:55 UTC: the user requested one depth20 job
on four GPUs. This Muon cell requests four96 GB GPUs and eight CPUs;
AdamW20 job2620278 remains at eight GPUs. The frozen training sources and
scientific config are unchanged. Global batch stays1,048,576 tokens and
total horizon1,539,870,720; only rank count and the per-rank token share change.

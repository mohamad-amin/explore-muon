# Muon depth 8, width 512

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

Active Slurm step **2618555.9**, detached launcher3372751.
Tiny and full-size peak-rate qualification passed; science is running.
[Qualification report](QUALIFICATION_PASSED.json):1.02s/update, .74s for both
spectral panels, 3.44GiB peak per rank; rough31.5-minute main forecast.

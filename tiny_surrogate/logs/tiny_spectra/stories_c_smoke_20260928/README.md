# Tiny surrogate: smoke

See the tiny-surrogate decision entry in research/adamw_spectra/MUON_CASE.md.
5 fresh runs; sources frozen here. RTX2080 Ti only (11GB), plus runtime VRAM gate.
Submitted as Slurm array 2626880; inspect its live state. tasks.json lists all arms; configs and failures are retained.
This is a standalone small subword-model harness, not the original FineWeb pipeline.

CandidateC: four layers,width128,4096byteBPE, full-development all-target evaluation with masked tails. Test documents are not read. This is a12-step numerical qualification only.

# Tiny surrogate: custom

See the tiny-surrogate decision entry in research/adamw_spectra/MUON_CASE.md.
15 fresh runs; sources frozen here. RTX2080 Ti only (11GB), plus runtime VRAM gate.
Submitted as Slurm array 2626881; inspect its live state. tasks.json lists all arms; configs and failures are retained.
This is a standalone small subword-model harness, not the original FineWeb pipeline.

CandidateC: exactly15 predeclared development arms. LR selection uses full development NLL; the test panel is never read. Same seed,8,388,608 unique targets,batch16384,commonmomentum0.8. Full data/candidate decision in MUON_CASE.md.

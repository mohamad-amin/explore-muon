# Tiny surrogate: smoke

See the tiny-surrogate decision entry in research/adamw_spectra/MUON_CASE.md.
5 fresh runs; sources frozen here. RTX2080 Ti qualification only (11GB), plus runtime VRAM gate.
Submitted as Slurm array 2626757; inspect its live state. tasks.json lists all arms; configs and failures are retained.
This is a standalone character-model harness, not the original FineWeb pipeline.

This tests actual BF16 forward/backward and NS, then all five methods. Stop on unsupported operations; no dtype or kernel changes. Existing scientific cohorts stay on A4000.

# Tiny surrogate: custom

See this study's PROTOCOL.md for the decision and comparison criteria.
2 fresh runs; sources frozen here. RTX2080Ti only (11GB), plus runtime VRAM gate.
Submitted as Slurm array 2626932; inspect its live state. tasks.json lists all arms; configs and failures are retained.
This is a standalone small subword-model harness, not the original FineWeb pipeline.

Two-arm input-covariance clock diagnostic, not surrogate qualification. Only cov_ema changes against preserved paired controls. Test sets remain sealed.

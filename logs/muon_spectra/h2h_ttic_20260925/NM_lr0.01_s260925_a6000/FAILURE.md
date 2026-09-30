# Stopped by the controller's time gate (2026-09-25, job in slurm log)

Tiny CUDA and full-size qualification passed, but the step-1 spectral measurement took ~400 s: the
config inherited `svd_device: "cpu"` and gpu-partition jobs get 2 CPUs. The forecast (60 spectra)
exceeded the 7 h budget, so the scientific run never started. Earlier A6000 arms used GPU SVD
(`_gpusvd`). Rerun unchanged except `svd_device: "cuda"` (measurement-only) in `../NM_lr0.01_s260925_a6000_gpusvd`.

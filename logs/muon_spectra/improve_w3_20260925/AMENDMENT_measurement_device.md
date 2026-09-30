# Controller amendment (2026-09-25 ~08:55 UTC)

The cluster A6000 arms ran with the only CPUs free there (2 per job). The per-sample spectral SVDs run on CPU, so qualification measured 426 s for one spectral panel and forecast more than the 7 h bound. `M_lr0.01_s260925_a6000` failed at qualification. The other A6000 arms were cancelled before science (E_auto_r2, E_b0.25, SOAP). Their directories and logs are kept.

`run_arm.py` now treats `svd_device` as a measurement-only setting. The A6000 arms are re-run as `*_gpusvd` with `svd_device: cuda` (FP32 gesvd on GPU; FP64 Frobenius norm unchanged). Training code and treatments are unchanged.

# Controller amendment (2026-09-25 ~08:45 UTC)

The first whitened arms (`E_auto_lr0.01`, `E_auto_lr0.01_s260925_a6000`) failed in the tiny CUDA stage. The tiny config replaced the model dict with bare tiny dimensions and dropped the architecture flags, so `mean_whitening` was rejected: it requires `track_input_stats`. Earlier arms' tiny stages therefore exercised the old default architecture. That was harmless, but did not test the frontier-norm flags.

`run_arm.py` now keeps the arm's architecture flags in the tiny model. Only the controller changed; training code is unchanged. Arms started after this amendment use the patched controller. `frozen_manifest_before_tiny_fix.json` keeps the original hashes. The failed arms' logs are preserved, and the arms are re-run as `*_r2` directories.

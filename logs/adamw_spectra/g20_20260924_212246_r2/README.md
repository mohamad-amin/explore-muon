# Completed g20 AdamW spectrum pipeline

Completed **2026-09-24 23:45:21 UTC** (18:45 Central). Slurm accounting
reports step 2618555.3 **COMPLETED**, exit 0:0; all workers exited 0.
The main run completed 1,469 updates / 1,539,870,720 tokens at selected peak
LR 0.0012. Final validation NLL is **3.879250004887581**. All 60 spectral
snapshots of the 24 selected matrices and all PNG/PDF figures are present.
See `77m_seed260924/summary.json` and `77m_seed260924/plots/median.png`.
Main training took about 1h49m; the pipeline including parallel pilots and
plotting took 2h13m. The allocation itself remains user-owned; completion of
this step does not end it. The launch details below are retained as history.

User-authorized run on **g20**, allocation **2618555**, Slurm step
**2618555.3**. Detached srun client **3256377** on priv-g14; supervisor
**2698287** on g20. Verify live handles and `pipeline.json` before acting.

The pipeline runs five full-size qualification updates, three simultaneous
300-update rate pilots on GPUs 0/1/2, then one full 1,469-update scientific
run on GPU 0 at the selected rate, followed by plotting. The planned first
seed is 260924, paired pilot seed 260923, rates 3e-4/6e-4/1.2e-3, selection
mean held-out NLL at updates 200/250/300. Spectra are saved every 25 updates
plus first/final (60 samples of 24 matrices). No extra condition is added.

`pipeline.json` contains worker PIDs and phase; `launch.json` contains the
exact command. `QUALIFICATION_PASSED.json` records the measured throughput,
memory and rough forecast. Per-worker logs, metrics and checkpoints are
under this directory. Frozen sources/configs are hashed in
`frozen_manifest.json`. `selection.json` and `selected_config.json` will
appear after the pilots; final plots appear under `77m_seed260924/plots/`.

The original attempt, `../g20_20260924_212246`, completed five training
updates successfully but stopped before any pilots because the normalized
singular-value squared-energy error reached 4.77e-4, exceeding the original
2e-4 tolerance. This was traced to FP32 accumulation of the scalar Frobenius
norm for nearly sign-valued Adam directions. Only that scalar accumulation
was changed to FP64; the training update and direct FP32 SVD are unchanged.
A real checkpoint remeasurement gives worst energy error 9.31e-7. The
original tolerance, artifacts and failure logs are preserved. Five targeted
measurement tests passed. The provisional prelaunch test/manifest in
`prelaunch_test_revision/` was replaced before any worker ran; the launched
snapshot is now fixed. See `REPAIR.json` and the original normalization finding.

The supervisor is independent of the chat execution session. Wall-time
safeguards remain 15 minutes for qualification, two hours per pilot and six
hours for the scientific run. No full allocation cancellation is part of
this study. If explicitly asked to stop it, the study's Slurm step is the
appropriate scope; allocation 2618555 remains user-owned.

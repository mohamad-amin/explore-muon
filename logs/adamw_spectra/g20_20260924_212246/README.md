# 77M AdamW spectrum run on g20

User-authorized execution on allocation **2618555**, Slurm step **2618555.2**,
node **g20**, four 48 GiB RTX 6000 Ada GPUs. Training uses one GPU per worker.
Three independent rate pilots run in parallel; the chosen-rate scientific
run follows on GPU 0. No DDP, extra rate, second scientific seed or Muon run.

The detached `srun` client is PID **3253712** on priv-g14; supervisor PID
**2681990** runs on g20. These are recorded launch identifiers, not proof a
process is still alive. Check live Slurm and `pipeline.json` before acting.
All outputs and the frozen code are on the shared filesystem.

Execution order:

1. Five full-size qualification updates at LR 6e-4, seed 260924.
2. Three 300-update pilots at LR 3e-4 / 6e-4 / 1.2e-3, common seed 260923,
   on GPUs 0/1/2. No spectra in pilots. Select minimum mean held-out NLL at
   steps 200/250/300; exact tie picks lower LR. No boundary expansion.
3. One full 1,539,870,720-token run, seed 260924, selected LR, spectra every
   25 updates plus first/final (60 samples of 24 matrices).
4. Render PNG/PDF figures and descriptive summaries automatically.

`pipeline.json` is the supervisor state; each worker also has its own
`status.json`, checkpoint, metadata, and `steps/` records. Worker console
logs and supervisor output are files alongside this note. `launch.json`
contains the exact detached launch command; `frozen_manifest.json` hashes
the immutable code/config snapshot. The supervisor never deletes completed
outputs or automatically extends the sweep. A failed stage is recorded and
does not silently become a successful scientific result.

Short qualification supplies measured runtime/memory and a rough forecast
in `QUALIFICATION_PASSED.json`. Wall-time safeguards are 15 minutes for
qualification, two hours per pilot, six hours for the scientific worker.
Worker runtime counts active assigned GPU time; it is different from the
lease time of this already-existing four-GPU allocation.

To inspect live Slurm state: `scontrol show step 2618555.2`.
Do not cancel allocation 2618555 merely to stop this study; it is user-owned.
The study's Slurm step is the appropriate scope for any explicitly requested
stop. Source/package changes elsewhere in the repository do not alter this
frozen run. No sealed initialization-study data were accessed.

# Track 3 benchmark: partial data-norm Muon branched from result #36

The modded-nanogpt optimization benchmark (Track 3): a 124M GPT on FineWeb, 524,288 tokens per step,
scored by the steps needed to reach 3.28 validation loss. Upstream:
`KellerJordan/modded-nanogpt/records/track_3_optimization`, commit in `UPSTREAM_COMMIT`.
Protocol entry: `research/adamw_spectra/MUON_CASE.md`, "Track 3 benchmark port (branch of #36)".

- `baseline/train_gpt_simple.py`: result #36 (tuned Muon + aux AdamW, 3250 steps), byte-identical to upstream.
- `train_gpt_pd.py`: the PD branch. The diff is 65 lines, all in the optimizer section and the optimizer
  construction; model, data, batch, aux Adam, schedule and training loop are unchanged.
  - Update: `polar(M R) R`, with `R = (C / mean eigenvalue + 1e-3 I)^(-1/4)`, rescaled to Muon's
    Frobenius norm, then Muon's shape factor. Their Nesterov momentum and 12-step NS are kept.
  - `C`: EMA of `E[x x^T]` of each hidden matrix's input. Forward hooks sample every 16th position after
    position 0, in training forwards only; the EMA decays 0.938 per step. q, k and v share the attention
    input. The hooks are registered before the first compiled call, so they are traced into the graph.
    Hooks added after compilation can silently not run.
  - `R`: recomputed every 10 steps (FP32 `eigh`) by the rank that owns the matrix, from its local statistics.
- `run_queue.sh`: runs scripts one after another on the g20 allocation (4×RTX 6000 Ada), as a Slurm
  step with the system compilers. The conda environment's `CC`/`CXX` break `torch.compile`.
- `compare.py`: our validation curves against the mean of the 10 published #36 H100 logs.
- `logs/`: Track 3 logfiles (`<uuid>.txt`, code included), console output and `queue.log`.
- `smoke/`: 40-step copies used once to check the port.

## Smoke test (40 steps, g20, 2026-09-25)

- Both runs start from identical weights (validation 10.82584 at step 0; Track 3 does not reseed).
- Step 40: baseline 5.66977, PD 5.63737.
- The hook statistics' EMA weight was 0.9227, as expected after 80 training forwards, so the hooks ran
  inside the compiled model.
- Steady step time: 1.34 s (baseline) and 1.45 s (PD). A full run takes about 80–85 minutes.

## Runs

- PD (`train_gpt_pd.py`, Muon LR 0.025 and WD 0.05 as in #36): g20 (4×RTX 6000 Ada), started 22:06 UTC
  via `run_queue.sh`.
- #36 baseline control (`baseline/train_gpt_simple.py`, unchanged): 4×L40S on g21, job 2622202, started
  22:16 UTC via `baseline/job_l40s.sbatch`. L40S is the same Ada chip family as g20 (no RTX 6000 Ada
  node was free).
  - It was moved off g20 at the user's request. The upstream copy moved from `train_gpt_simple.py` to
    `baseline/`, so the g20 queue's second item (`train_gpt_simple.py`) fails at once instead of running.
  - Two aborted attempts are in `logs/aborted/` (see its README).
- PD learning-rate probes (the user chose 0.03 and 0.035; 0.035 is 1.4× Muon's, the PD-to-Muon ratio
  that was best in our setup). Each script differs from `train_gpt_pd.py` only in the Muon LR:
  - `train_gpt_pd_lr0.035.py`: 4×L40S, job 2622371 (`job_pd_lr0.035_l40s.sbatch`), starts when the control
    frees g21.
  - `train_gpt_pd_lr0.03.py`: g20, chained in tmux `track3b` after the first g20 queue ends.

## Follow-up runs (user goal: improve PD over the Muon baseline; 2026-09-25 evening, CDT)

Protocol entries: `research/adamw_spectra/MUON_CASE.md`, "Why PD's gain shrinks in Track 3" and the decisions after it.
`screen.py` groups runs by step count and family (#36 decoupled decay vs #37 hyperball) and shows each family's
published H100 mean. `run_on.sh JOBID SCRIPT...` replaced `run_queue.sh` (which logged the exit code of `date`).
`submit_l40s.sh SCRIPT [GRES]` submits to the gpu partition.

Scripts (each differs from its parent only as stated):
- #36 family (from `train_gpt_pd.py`):
  - LR: `train_gpt_pd_lr0.02.py`, `train_gpt_pd_lr0.03.py`, `train_gpt_pd_lr0.035.py`.
  - WD: `train_gpt_pd_wd0.025.py`, `train_gpt_pd_wd0.0125.py`.
  - PD details: `train_gpt_pd_a0.125.py` (α = ⅛), `train_gpt_pd_damp0.01.py`, `train_gpt_pd_mlponly.py`, `train_gpt_pd_attnonly.py`.
  - PD-geometry weight decay: `train_gpt_pd_pdwd.py` (W ← W − ηλ W R²/mean eig R²) and `train_gpt_pd_pdwd1.py` (first power, W R/mean eig R).
  - Controls: `train_gpt_simple_save.py`, `train_gpt_simple_wd0.025_save.py`.
- `h1625_*.py`: the same at 1625 steps (half length, same schedule shape), for a cheaper A6000 screen.
- #37 family (hyperball): `baseline/train_gpt_simple_muonh.py` is #37 (sha f612c7c3, identical in all 10 published
  logs); `train_gpt_simple_muonh_save.py` is the control; `train_gpt_pdh.py` (PD-H) swaps MuonH's direction for PD's.
- Scripts ending in `_save`, and the newer PD variants, append an analysis-only save of the final weights and
  PD input statistics (`add_final_save.py`; after the last validation, so training is unchanged).
  They are read by `weights_report.py` and `gamma_probe_t3.py`.

## Geometry decay, the 2×2 control and the deferred S∘PD port (2026-09-25, 21:00–22:30 CDT)

- PD-geometry decay: `train_gpt_pd_pdwd.py` (p = 2) and `train_gpt_pd_pdwd1.py` (p = 1).
  - LR and WD variants: `train_gpt_pd_pdwd_lr0.02.py`, `train_gpt_pd_pdwd_lr0.03.py`, `train_gpt_pd_pdwd_lr0.035.py`, `train_gpt_pd_pdwd_wd0.1.py`, `train_gpt_pd_pdwd_wd0.025.py`.
- `train_gpt_muon_geowd.py` ("MG"): #36's Muon update unchanged, with PD's root R used only for the decay. It is the missing {Muon, PD} × {decoupled, geometry} cell (MUON_CASE.md, 22:30 CDT).
- `train_gpt_pdh.py`: PD on #37 (hyperball); control `train_gpt_simple_muonh_save.py`.
- Deferred after the independent review; written and smoke-tested, not run at full length:
  - `train_gpt_spd_pdwd.py`: S∘PD, i.e. SOAP-Muon in PD's whitened coordinates, plus geometry decay;
  - `train_gpt_soap.py`: SOAP-Muon control (α = 0, so R = I).
  - Smoke test (`smoke/smoke_spd_pdwd.py`, `smoke/smoke_soap.py`; 40 steps, 1 A6000, world size 1): validation at step 40 is 5.66302 (S∘PD + geometry decay) and 5.61793 (SOAP-Muon). Both passed the PD-statistics and SOAP-state checks. The earlier 4-GPU smoke test gave 5.66977 (#36) and 5.63737 (PD).
- Analysis tools:
  - `weight_share_t3.py`: share of ‖W‖² along the top input eigendirections, with C measured on the model itself.
  - `gamma_probe_t3.py`: exact-HVP curvature exponent.
  - `screen.py --steps`: lead in steps.

## Result: PD + geometry decay reaches 3.28 at 3150 steps (2026-09-26, 21:16 CDT)

- **Script:** `train_gpt_pd_pdwd_a0.125_s3150.py`. #36 with partial data-norm Muon (α ⅛) and PD-geometry weight decay (p = 2). #36's LR 0.025, WD 0.05 and schedule shape (70% linear cooldown) are unchanged, compressed to 3150 steps.
- **Runs:** six, all reported: 3.27783, 3.27989, 3.27886, 3.27719, 3.27730, 3.27787 (L40S, RTX 6000 Ada and A6000).
- **Criterion:** mean 3.27816, (3.28 − mean)·√6 = 0.0045 ≥ 0.004, passes. It is also pairwise-significant vs #36 (3250 steps).
- **Disclosure:** the sixth run was added after five scored 0.00399. No run is excluded. Nothing was run on H100.
- **Checking:** `python track3/earliest_step.py train_gpt_pd_pdwd_a0.125_s3150.py` recomputes this from the logs. It pools only runs with identical logged code.
- **Details and the full history:** MUON_CASE.md.

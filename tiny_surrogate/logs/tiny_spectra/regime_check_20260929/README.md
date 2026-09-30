# Regime check for second-order-structure ideas (2026-09-29)

Claude session, at the user's request. Decision argument: `../../../PROTOCOL.md`, entry "Regime check for
second-order-structure ideas". This branch changes no earlier verdict, selects no recipe and never reads
the sealed test panels. Fresh probes are training-stream blocks beyond the 96M-target horizon.

## Question

Does Candidate D reproduce three structures of the reference study that the proposed ideas rely on?

- **(a) Edge of stability at large batch.** LR × top GN eigenvalue in a stable band, gradients in the
  stiffest directions flipping sign step to step, about half the steps raising a fixed held-out loss,
  and much lower step energy in the stiff directions for PD than for Muon.
- **(b) Momentum error dominated by staleness in the stiff directions.**
- **(c) RMSNorm gains uneven enough to make Muon's step depend on the gauge.**

## Runs (GPU array 2628728, one 16 GB A4000 each, snapshot every step)

| Run | Source config | Reference final NLL |
|---|---|---|
| `muon_b262144_lr0.01_s20261001` | ordering `muon_D_lr0.01_m0.9_s20261001` | 4.55501 |
| `pd_b262144_lr0.01_s20261001` | ordering `pd_D_lr0.01_m0.9_s20261001` | 4.52002 (first listed as 4.47917, SoapMuon+PD's value) |
| `muon_b1048576_lr0.01_s20261001` | batch `muon_Dbatch_b1048576_lr0.01_s20261001`, eval every 4 | 5.65535 |
| `pd_b1048576_lr0.02_s20261001` | batch `spd_Dbatch_b1048576_lr0.02_s20261001` with method `pd`, eval every 4 | new |

Configs: `make_configs.py`, `configs/`. Launch: `job_train.sh`. Sources hashed in `source_manifest.json`.

## Analysis (jobs in `analysis_jobs.txt`, each after its run)

`regime_probe.py all <run>` writes `analysis/<run>/`:
- `momentum.json`, `momentum_state_*.pt`: momentum rebuilt from replayed training gradients, stale-free
  momentum (same batches at the current weights), staleness and noise sizes per direction class.
- `gains.json`: gain spread through training; fold check (Muon, PD α ¼, PD α ½ with and without the
  gains folded into q/k/v/up).
- `edge.json`: top GN eigenvalue through training, fixed-probe loss at every snapshot, step-to-step
  gradient behaviour in and outside the top-16 GN subspace, stiff energy of the steps.

`regime_report.py` makes `analysis/figures/` and `analysis/verdict.json`. The CPU code check in `smoke/`
uses tiny sizes and is not a result.

## Review and decision thresholds

An independent read-only review (17:00 CDT) found the plan sound but the edge and momentum criteria
miscalibrated for D. The analysis was held, revised (c* edge measure, cross-probe cosines, basis every 5
steps, chance-normalized stiff energy, D-matched momentum states, Newton-Schulz and norm checks, gauge
fold with the optimizer's Newton-Schulz and a data metric) and released at 17:08 CDT. The fixed
thresholds are in PROTOCOL.md ("Review addendum"), written before any analysis result; `regime_report.py`
applies them and writes `analysis/verdict.json`. Source hashes: `source_manifest.json` ("analysis").

## Status

- 2026-09-29 16:50 CDT: runs submitted; two running (262K), two queued.
- 17:08 CDT: three runs running (Muon 262K, PD 262K, Muon 1M); PD 1M waits for an A4000. Analyses
  2628742–45 follow their runs (8 h limit each).
- 2026-09-30 00:50 CDT: all runs and analyses complete; the PD 1M exact rebuild (consistency check) finished at 01:42 CDT and passes all three states (see PROTOCOL.md). Earlier note: it was then
  2 of 3 states done, both passing). Results below; verdict in `analysis/verdict.json`.

## Results

| Run | Final NLL | Reference | Difference |
|---|---|---|---|
| Muon 262K | 4.55428 | 4.55501 | −0.0007 |
| PD 262K | 4.51631 | 4.52002 | −0.0037 |
| Muon 1M | 5.65297 | 5.65535 | −0.0024 |
| PD 1M | 5.51089 | — | — |

- **(a) Edge: present at 262K and 1M** (E1, E2 hold; E3 fails). Muon mid window c* 0.59 / 0.64; top-16
  cross-probe consecutive-gradient cosine −0.69 / −0.76 against +0.08 / +0.00 elsewhere; body steps raise
  the probe loss 0% / 7% of the time. PD alike. D sits slightly below the edge (overshoot about 1.6×, not
  2×). PD's stiff step energy is only 0.20× Muon's at 262K (reference 40–160× lower): supporting criterion
  fails. PD's states are 4–7× sharper than Muon's.
- **(b) Staleness: the signature holds at every pre-cooldown state of every run.** Top-16 GN directions:
  cos(M, ḡ) −0.83 to −0.95, stale-free +0.998 to +1.000, staleness 1.06–1.15× the signal, noise 1–3%. The
  GN complement is less anti-aligned (−0.30 to −0.43). Gone after the cooldown. Gates: PD 1M passes all
  three states as planned; Muon 1M fails the Newton-Schulz check with K = 40 (0.92–0.996) and passes all
  three when rebuilt from step 1 with the trainer's replay microbatch (0.9995–0.9997; metrics within
  0.005); 262K states are flagged by the numerical Newton-Schulz margin (Muon) or a 2.2× replay-mean gap
  (PD, 13–16% of the signal).
- **(c) Gauge: deprioritized.** Gains stay within 1.07× of their median (reference models ≤ 1.26×), Muon's
  fold data-metric cosine is 0.9997, the algebra check 1.000.
- **Decision:** D is usable for the momentum-transport and oracle studies. The gain-gauge line is
  deprioritized; the per-head value/output gauge is untested.

Figures: `analysis/figures/edge.png`, `momentum_error.png`, `momentum_cos.png`, `gains.png`.

## Correction (2026-09-30 02:15 CDT)

The momentum cosines above compare M_t, the momentum that produced the step to W_t, with the gradient at
W_t: the post-step view, which is the overshoot signature. At the time of use the optimizer applies
M_{t+1} = β M_t + κ g_{t+1}(W_t), and cos(M_{t+1}, ḡ(W_t)) is +0.58 to +0.79 at all eight 262K and 1M
states. The stale-free momentum is 11–14× the norm of the time-of-use momentum, 73–93% of it in the
top-16 GN directions: in the stiff directions, "staleness" is mostly heavy ball's damping of the period-2
oscillation. The (b) verdict stands as a statement about post-step structure, not as "the momentum the
optimizer uses points the wrong way". See PROTOCOL.md (review of the oracle plan).

# Spike-origin diagnostics at saved final checkpoints (2026-09-25)

Authorized by the user on 2026-09-25 ("go ahead", four GPUs). The decision argument, predeclared readings, gates and amendments are in [MUON_CASE.md](../../../research/adamw_spectra/MUON_CASE.md#spike-origin-diagnostic-at-saved-checkpoints-2026-09-25). This is read-only with respect to all runs: no parameter is updated, and no checkpoint or earlier output is modified.

## What was run

Final checkpoints, step 1,469 at the end of LR cooldown:

| label | run | state used as reference |
|---|---|---|
| muon_d8 | `logs/muon_spectra/depth8_w512_20260925_r2/scientific` | momentum × (1 − 0.95) |
| muon_d12 | `logs/muon_spectra/depth12_w512_20260925_r2/scientific` | momentum × (1 − 0.95) |
| adamw_d8 | `logs/adamw_spectra/g20_20260924_212246_r2/77m_seed260924` | AdamW first moment (β1 = 0.9) |
| adamw_d12 | `logs/adamw_spectra/depth12_w512_20260924/scientific` | AdamW first moment (β1 = 0.9) |

- **Data.** Fresh gradients use 64 contiguous batches of 524,288 never-trained training-stream tokens, starting at offset 2,000,000,000 (training used tokens [0, 1,539,870,721)). All four checkpoints see the same tokens.
- **Precision.** BF16 autocast with TF32 disabled, as in training, run eagerly with hooks. Microbatches are 32 sequences of 512 tokens.
- **Hardware.** One L40S GPU per checkpoint, inside allocation 2567578 (priv-g14). The g20 allocation was busy with another project's job and was not used.
- **Launch.** `launch.sh gates|probe|science [tag]`, run via `srun --jobid=2567578 --overlap --gres=gpu:4`.

Source: `research/adamw_spectra/spike_diagnostics.py`, CPU tests in `test_spike.py`. Each output directory holds a copy of the package source, and `metadata.json` records its hashes.

## Qualification record

- `gates/`: first gate attempt. Muon passed; both AdamW checkpoints failed at load because their pre-Muon configs lack the `optimizer` key. Kept unchanged.
- `gates_r2/`: all four pass after loading through `load_config`:
  - validation NLL at the loaded weights within 3e-5 of the recorded final values (gate 0.01);
  - hook reconstruction of every body weight gradient ≤ 3.5e-3 relative (gate 5e-3; BF16 rounding).
- `probe/`: 2-batch probe. The AdamW runs tripped a badly normalized implementation check (hook vs gradient divided by u1ᵀḠv1 ≈ 0). Kept unchanged.
- `probe_r2/`: all four complete after renormalizing that check. The probe also motivated the bulk noise edge and the applicability rule recorded in MUON_CASE.md before the science run.
- `science/`: the 64-batch runs. `report.py` → `report.md`, `matrices.csv`.

## Results

Details: [report.md](report.md) and [matrices.csv](matrices.csv). Readings apply the predeclared rules. One seed; one end-of-cooldown checkpoint per run; width 512.

1. **Most momentum spikes are the token-mean product.** The fresh gradient reproduces the saved top singular pair (u1, v1) for 45/48 Muon matrices at 8 layers and 66/72 at 12 layers. Where it does, the product of the mean output-gradient and the mean input usually explains ≥ 0.5 of u1ᵀḠv1:
   - Readings at 8 layers: 33 mean, 5 sink, 7 distributed.
   - Readings at 12 layers: 47 mean, 2 mean+sink, 7 sink, 10 distributed.
   - Final block: O 0.92/0.97 and down 0.95/0.98 (8/12 layers); V 0.75/0.81, with a position-0 share of 0.28/0.24.
   - Across whole fresh gradients, the mean-product term's energy share tracks ρ1 (e.g. final O: 0.91 vs 0.92).
2. **Shared factors.**
   - Input side: Q, V and up share v1 (|cos| 0.85–0.99 from block 3 on), and v1 is the mean-input direction.
   - Output side: late-block O and down share u1 (0.80–0.92), which also aligns with the final block's.
   - K's v1 is unrelated to these. K spikes are position-0 (sink) dominated where reproduced.
   - The early/mid MLP-down spikes of the 8-layer model are also position-0 dominated (0.75–0.85).
3. **Outlier channels.** In mid blocks the mean input is dense (participation ≈ 130–200 of 512). At the final block it concentrates on a few residual outlier channels (participation 6–19; channels 345/168 at 8 layers, 224/279 at 12 layers). There are attention sinks: 12–32% average mass on key 0 in upper blocks (uniform ≈ 1.2%), and some heads above 60%.
4. **AdamW control.**
   - AdamW's first moment is reproduced less often (42/48 at 8 layers, 44/72 at 12 layers).
   - Its mean input sits on one or two outlier channels at every depth (participation 3–5).
   - Its residual outlier channels are stronger: channel max/mean-abs 38–96, versus ≤ 28 under Muon.
5. **The bulk is mostly noise, with a real minority signal.** Cross-fitted/in-sample descent (Muon medians):
   - mode 1: 0.93–1.12; modes 2–5: ≈ 0.8; rest of the top decile: ≈ 0.5;
   - q0.1–q0.9: 0.20–0.31, statistically clear ("mixed");
   - bottom decile: 0.18, mostly "noise".

   Only a median 2–3 modes per matrix exceed the bulk noise edge (≈ 0.2 of the state's Frobenius norm). A hard threshold at that edge would therefore keep a rank-≈3 update.
6. **At this scale, NS keeps all first-order descent** (in-sample and cross-fitted ≈ 1.00).
7. **The fresh-gradient spike grows with token count** (e.g. O: σ1/median 40 → 160 from 0.5M to 32M tokens). The saved momentum is about as concentrated as one 1M-token gradient, not the 32M-token average that iid EMA filtering would imply. That filtering argument therefore overstates the effect at this checkpoint (end-of-cooldown caveat).
8. **Follow-up check (from the saved arrays).**
   - The fresh gradient's top pair is (ē, x̄) at |cos| 0.95–1.00 for Q (mid on), V, O, up and late down.
   - The saved momentum's v1 matches x̄ (0.86–1.00), but its u1 matches the current ē only at 0.1–0.45. The input side is stable; the mean error's direction drifts, so the momentum holds a smoothed version of it.
   - The spike is not the only outlier. The momentum's σ1/σ2 is 3–5 for V, early/mid up, final O and late Q, and σ2 is still 8–140× the median. K, early/mid down and some early Q/O have no single dominant pair (σ1/σ2 1.1–1.5).
9. **Follow-up: controls and compounding.**
   - **K is a built-in control.** K reads the same LN1 input as Q and V, but softmax ignores a shift of every key. The K bias therefore gets zero gradient and ē_K = 0 exactly. Measured K mean-product shares are ≈ 1e-4 (rounding), and K's v1 is unrelated to x̄ (|cos| 0.0–0.3), whereas Q's and V's v1 is x̄.
   - **No step-to-step coherence.** With μ = 0.95, a direction that stayed fixed would be as large in (1−μ)M as in one gradient. A direction random at each step would be √((1−μ)/(1+μ)) ≈ 0.16 as large. Measured: σ1 of (1−μ)M is 0.90–1.36× the random-each-step prediction across all 120 Muon matrices (median ≈ 1.1); full persistence would give 6.2×.
   - **Per-step pieces share x̄.** They share the input vector x̄ but not the output vector, so they add into one rank-one spike of reduced size.
   - **Much of each step's spike is batch-specific.** At fixed weights, one 1M-token batch's σ1 is 1.5–2.5× the 32M-token average.
   - These are end-of-cooldown measurements; the constant-LR phase is untested.
10. **Follow-up: why the input vector repeats although batches are never reused.** [`batch_means.py`](batch_means.py) → `batch_means/`. It uses 16 fresh 1,048,576-token batches (offset 2,100,000,000) at the fixed final Muon 8/12-layer weights. Medians by kind:
    - x̄ from two different batches agrees at |cos| 0.998–0.9999.
    - The shared mean is about half of each token's squared input norm for the LayerNorm-fed Q/K/V/up (0.46–0.54). It is 0.11–0.12 for O's input and 0.05–0.06 for down's input.
    - ē from two different batches agrees only at 0.12–0.50, even with the weights frozen.
    - ē is only 1e-6–2e-5 of a token's squared error norm (K ≈ 1e-11, the exact zero).
    - The momentum's v1 matches each batch's x̄ at 0.93–1.00; its u1 matches each batch's ē at 0.12–0.24.
    
    Every token carries the shared input component, so it survives averaging. The mean error is a tiny residual of large, mostly cancelling per-token errors, so the particular batch sets its direction.

Predeclared decision map: the mean route holds for final O/down/V, so the next candidate is a centering/bias-path training test. That needs separate authorization. The bulk is "mixed", not noise-dominated, so a noise-edge threshold is not favoured.

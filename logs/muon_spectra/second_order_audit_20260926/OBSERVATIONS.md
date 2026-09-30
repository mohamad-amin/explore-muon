# Second-order audit: lab notebook

Working notes: what the figures show, guesses, and what to look at next. These are not conclusions. Entries are dated in CDT, newest last. Figures are in `figures/`, data in `marginals/` (and later folders).

## 2026-09-27 00:30: one-sided GN marginals along the Muon and PD trajectories

Figures: `ratio_along_C.png`, `curvature_vs_variance_block4.png`, `over_training.png`, `spectra_block4.png`. The runs are Muon and PD, seed 260925, with 8 checkpoints each (steps 10 to 1469) and 2048 held-out sequences.

**Seen**
- **q, o, up, down.**
  - The exact GN input marginal equals K-FAC's tr(B)·C along every eigenvector of C.
  - This holds over five decades of input variance, at every checkpoint.
  - For these matrices, C is the right input statistic.
- **k.**
  - The top eigenvector (≈ the mean input direction) is 5–10× flatter than K-FAC from step ~200 on, and about 20× flatter (0.05) at step 10.
  - At step 50, the whole key spectrum is briefly 2–5× stiffer than K-FAC. Why?
- **v.**
  - Stiffer than K-FAC across the whole spectrum early: 5–10× at steps 10–50.
  - Relaxes toward 1 in the bulk by the end.
  - The top direction stays 3× (Muon) and 7× (PD) stiffer. Block 4's top eigenvector is about 30× at step 10.
- **Output factor B.**
  - Keys: a cliff after about 8 eigenvalues, i.e. one dominant direction per head (8 heads), at every step.
  - Up: a rank-512 cliff at step 10, which is W_down's rank.
- **The mean input direction's share of tr C.**
  - It falls from 0.6–0.9 at step 10 to about 0.15 at steps 100–200, then grows again: to about 0.3 under Muon and 0.55–0.6 under PD.
  - PD's C also has a steeper low-variance tail late. The optimizer reshapes the curvature it later sees.
- **Gradient noise scale per matrix** (trace ratio, in sequences): about 5 at step 10, 100–600 mid-training (below the 2048-sequence batch), then a jump at the end of cooldown.

**Guesses to check**
- **Values early.** Attention is diffuse, so values are averaged coherently across positions; as attention sharpens, the bulk decorrelates. Look at attention entropy per head vs the values' excess curvature over training.
- **Keys' cliff.** Each head's key gradient may be dominated by that head's mean query direction. Look at the alignment of B_k's top-8 eigenvectors with the per-head mean queries.
- **Keys' step-50 bump.** Maybe the QK-norm gains, or the attention sink forming near the end of warmup. Look at per-head attention to position 0 over training.
- **PD's growing mean direction.** PD steps less along the mean input, so it can grow. Does it matter for the loss, and does it relate to the attention sink?

**Ideas**
- Input statistics per kind: C_within for keys, and for values something that adds the sequence-coherent part.
- Keys' output side: per-head rank-1 plus isotropic might capture most of B_k cheaply.

## 2026-09-27 01:05: gap map launched (measure_frame.py, analyze_frame.py)

- **The four reruns reproduce the original runs within 0.002.** Finals: Muon 3.70465, PD 3.68692, SOAP-Muon 3.68339, S∘PD 3.67587. So the kept checkpoints are representative.
- **Gap map.** In each matrix's Kronecker frame, measure the exact curvature, signal and noise of every direction, plus each optimizer's applied update and momentum projected into the same frame.
  - 8192 sequences per checkpoint, with the frame from 512 other sequences.
  - 28 checkpoints on 8 GPUs (g20 and priv-g14).
- **Smoke-test hints** (PD step 500, block 4 v, only 64 sequences):
  - Contiguous 8-sequence blocks behave like independent sequences (block-variance ratio 1.01), so the 1/b noise law holds at this granularity.
  - The momentum after step t is anti-aligned with the gradient at W[t] (cos −0.11).
  - Guess: oscillation along the stiffest directions (edge of stability), with consistent descent in flat ones. The per-curvature-bin cos(M, g) will show whether it is concentrated in stiff bins.

## 2026-09-27 01:12: marginals for all four optimizers (figures re-plotted with SOAP-Muon and S∘PD)

- **The attention pattern holds on all four trajectories.**
  - Keys' mean direction is about 0.2× K-FAC after the first ~100 steps.
  - Values are 3–7× stiffer than K-FAC at the end and ~40× early. Their weight on the sequence-mean part of C falls from ~40 to 3–6.
  - q, o, up and down stay ≈ 1.
  - So this is a property of the architecture and loss, not of the optimizer's path.
- **The keys' step-50 bump is Muon-only** (≈ 1 for PD, ≈ 0.3 for SOAP-Muon and S∘PD). It is a trajectory transient, not structure.
- **The mean input direction's share of tr C.**
  - Every optimizer shows the U shape: a drop over the first ~100–200 steps, then a rise.
  - The late rise is much larger for the three better optimizers: PD ~0.55–0.6, SOAP-Muon ~0.4–0.55, S∘PD ~0.6–0.75, against Muon ~0.25–0.3. For the o input (attention output), S∘PD reaches 0.73.
  - Why do the better optimizers let the mean direction grow? Is it related to attention sinks or massive activations, and is it cause or effect of their lower loss?
- **Gradient noise scale** follows the same path for all four: ~1–10 sequences early, 50–1000 mid-training (below the 2048-sequence batch), and a jump at the end of cooldown.

## 2026-09-27 01:22: gap map, first look (step 500 for all four optimizers; steps 200 and 900 for Muon and PD)

Figures: `figures_gap/gap_profiles_step500.png`, `gap_2d_step500.png`, `gap_over_training.png`. Directions are frame pairs binned by their exact GN curvature h, with all depths pooled per kind.

**Seen**
- **The Newton decrement lives in flat directions.** It peaks at h ≈ 1e-5 to 1e-4 and falls off steeply toward stiff directions. In the frame, it sits in the bulk-bulk corner (low-variance inputs × low-curvature outputs), not at the top eigenvectors.
- **Those directions are noise-dominated at our batch.** Their critical batch b* is 5k–20k sequences (2.5–10M tokens), against the 2048-sequence batch. b* falls with curvature and crosses 2048 around h ≈ 1e-3.
  - This matches the GN paper, where GN pulls ahead of SOAP only at 4M+ token batches.
  - The share of the decrement reachable at b = 2048 is ~10–25% for q, 20–60% for k, 80–98% for v, and 30–65% for o, up and down.
  - It is lower for the better optimizers. Guess: they have already used up the reachable part.
- **Stiff directions oscillate.** cos(momentum after t, gradient at W[t]) is −0.4 to −0.8 in the stiffest bins, for every optimizer. That is a period-2, edge-of-stability signature. The overall cos is negative too (−0.15 to −0.7), because the oscillating stiff directions dominate the gradient norm.
- **Step profiles.** Each optimizer's effective step per unit gradient, relative to the noise-aware Newton ideal, rises steeply with curvature: too little in flat directions and too much in stiff ones, in shape (10–100× across the range).
  - The order of flatness is S∘PD (flattest) > PD ≈ SOAP-Muon > Muon, which is the same order as their final losses.
- **Where the update energy goes (2D view).**
  - Muon spreads it evenly over input directions.
  - PD moves it to the low-variance input directions, where the decrement is.
  - SOAP-Muon moves it to the low-curvature output directions.
  - S∘PD does both.
  - A picture of why PD and SOAP stack: they correct different sides.
- **Steps are small relative to curvature.** In the bins that carry the decrement, the step is ~1–5% of what the quadratic model would allow along its own direction (overshoot ratio 0.01–0.05).

**Emerging hypothesis**
- The global learning rate is capped by the stiff, oscillating directions, while the gain lives in flat, noise-dominated directions where steps are 20–100× below their local optimum.
- Better optimizers partly decouple the two by preconditioning, which is what PD and SOAP each do on one side.

**What would close more of the gap** (ideas, not conclusions)
- Step less where it's stiff and more where it's flat, on both sides (S∘PD's direction, pushed further).
- Average longer where it's flat: flat directions are noise-dominated per step but their signal changes slowly, so per-direction averaging timescales could help (long for flat, short or none for oscillating stiff directions).
- Damp or cap the oscillating stiff directions so the bulk can take a larger step. Momentum works against the gradient there.

**Caveats**
- This is a one-step, quadratic, frame-diagonal view: a map for locating structure, not a success metric.
- ḡ comes from held-out data, and the frame's off-diagonal terms are ignored in these sums.

## 2026-09-27 01:32: gap map across training (all 28 checkpoints)

- **The picture at step 500 holds at step 1300 and for all four optimizers.**
  - The decrement peaks in flat directions (h ~1e-5 to 1e-4), where the critical batch b* ~1e4 sequences is above our batch.
  - Momentum is anti-aligned with the gradient in stiff directions.
  - Effective-step profiles are too steep, with S∘PD the flattest.
- **There is a regime change around steps 50–100, for every optimizer.**
  - At step 10, ~100% of the decrement is reachable at batch 2048, and cos(momentum, gradient) is +0.6 to +0.8 everywhere: plain descent.
  - By step 50–100, the stiff directions oscillate (cos −0.3 to −0.8).
  - From steps 100–200 on, the reachable share falls: q ~10%, k ~20%, o, up and down ~20–40%, v 60–95%.
  - After the first ~100 steps, most of the potential second-order gain is noise-dominated at our batch.
- **Overshoot in the decrement-carrying bins grows from 1e-4 (step 10) to ~0.03–0.07 (step 1300).** Updates stay far below the quadratic optimum along their own direction where the gain lives.
  - One anomaly: SOAP-Muon's o matrices overshoot at step 100 (ratio ~2.5). This matches its known early deficit (worse than Muon for the first ~120 steps).
- **Next.**
  - Test whether the oscillating stiff directions set the LR ceiling: top GN eigenvalues of a few matrices by Lanczos, and the step along them vs the stability threshold.
  - Test whether the signal in flat directions is persistent enough over steps to justify longer averaging there (gradient autocorrelation across nearby steps, per bin).
  - Try the ideas on the noisy-quadratic model with the measured spectra before touching training code.

## 2026-09-27 01:57: progress-rate model, and a surprise in gradient persistence

**Progress-rate model** (`rate_model.py`, Muon's step-500 landscape).
- The model assumes a persistent gradient and compares each preconditioner shape's best rate with the noise-aware Newton rate. At 1M tokens, no averaging:

| Shape | Share of ideal rate |
|---|---|
| Exact curvature^-¾ | 0.69 |
| Exact curvature^-1 | 0.68 |
| Kronecker (λ_B λ_C)^-1 | 0.63 |
| Measured S∘PD | 0.57 |
| Measured PD | 0.44 |
| Measured SOAP-Muon | 0.36 |
| Input-only^-1 | 0.34 |
| Measured Muon | 0.21 |
| Isotropic | 0.07 |

- The ranking is broadly right: S∘PD first, Muon last among the measured optimizers. PD and SOAP-Muon swap places relative to training.
- At 4M and 16M the ideal rate rises and exact curvature^-1 reaches 0.88 and 0.96, while the measured optimizers stay flat. The absolute gap grows with batch.
- **Discrepancy:** the model favors strong input power, up to λ_C^-1 (α ½), while training preferred α ¼ at a single learning rate. That comparison has a known output-energy confound.
  - Test 1: the LR-bracketed α sweep (cohort `soaudit_alpha_20260927`).
  - Test 2: `one_step_alpha.py`, the same state with α varied and each direction at its own best scale.
- The fixed local quadratic model (`nqm_measured.py`) saturated: every rule removes ~99% of the local excess within 900 steps. Progress in real training comes from the landscape continuing to slope down, not from settling into one bowl.

**Gradient persistence** (`measure_persistence.py`; Muon and PD at t = 200; true gradients from independent sequence sets; `figures_gap/persistence_partial.png`).
- **One step later:**
  - flat directions: correlation only +0.2 to +0.3;
  - the bulk: ~0.1;
  - stiff directions: −0.2 to −0.6, a sign flip, i.e. oscillation.
- **300 steps later:** about 0 everywhere.
- The expected gradient along flat directions is mostly new after every step. A per-direction quadratic picture says flat directions barely change in one step, so the change must come through couplings (off-diagonal and cross-layer GN terms, or the non-GN Hessian), driven by the large, oscillating steps elsewhere.
- **Hypothesis:** the stiff-direction oscillation injects self-generated noise into the flat-direction gradients. Momentum's averaging mostly cancels this oscillating component. Damping the stiff directions could make the flat-direction signal persistent and allow bigger flat steps.
- **Test:** split each actual update into its stiff and flat parts, compute H·ΔW (Hessian-vector products) for each part, and see which part moves the flat-direction gradients. Also measure intermediate lags (2–50 steps) with densely kept checkpoints.

## 2026-09-27 02:06: persistence at t = 200, 500 and 900 (Muon and PD); the atlas page

- **Persistence** (`figures_gap/persistence.png`).
  - One step apart, the correlation of the true gradient is +0.2 to +0.4 in flat and middle directions, at every checkpoint.
  - Stiff directions go negative under Muon.
  - PD's stiff directions stay positive at t = 900 (+0.1 to +0.3 for v, o, up, down and k): PD's input scaling reduces the stiff-direction oscillation, as the gap map suggested.
  - 300–400 steps apart, correlations are ≈ 0, with a small sign split in stiff key directions (Muon −0.5, PD +0.5).
  - Open: how correlation decays between lag 1 and lag 300. Is it a period-2 fast part plus a slow persistent part? This decides what momentum and averaging can extract.
  - Plan: dense lags (steps 501–532) from a fresh Muon and PD run with kept checkpoints at 501, 503, 507, 515 and 531 (next-step weights give lags 1–4, 7, 8, 15, 16, 31, 32), stopped at step 532.
- **Atlas.** The interactive page, built by `make_atlas.py` with `atlas_template.html`: https://claude.ai/artifact/8tZX3HjvacAn7ZhDJSxC8x. Views: curvature vs K-FAC, over training, gap map, spectra, batch size.

## 2026-09-27 02:12: correction to the reading of cos(momentum, gradient), and to the one-step design

- **First version of `one_step_alpha.py`.** It built each direction from the momentum saved after step t alone. It came out uphill (⟨g, D⟩ > 0) for every α at step 200.
  - That was an artifact. The real next step uses M' = 0.95 M + g, with g the fresh gradient at W[t].
  - In a period-2 oscillating direction the momentum's oscillating part is only ~1/(1 + β) ≈ 0.5 of one gradient, so the fresh gradient flips its sign and the actual step goes downhill.
  - In slow, flat directions the momentum dominates (~20 gradients).
- **Fix.** Both one-step scripts now add a fresh 1M-token gradient (2048 held-out sequences) before forming directions. The first job was cancelled and its log kept in `aborted/`.
- **Rereading the gap map.** Negative cos(M after t, g at W[t]) in stiff bins is the period-2 oscillation signature, not evidence that momentum drives the step uphill. The gap map's effective-step and overshoot panels used the actual applied update W[t+1] − W[t], which includes the fresh gradient, so they stand.
- The earlier phrase "momentum works against the gradient there" should read "stiff directions oscillate period-2; the momentum carries the last half-cycle".

## 2026-09-27 02:08: the stiff part of the update drives the gradient change everywhere (measure_step_coupling.py)

Figure: `figures_gap/coupling_step500.png`. Muon and PD at step 500. Each actual update W[t+1] − W[t] is split in each matrix's frame by K-FAC curvature rank (stiff = top 1% of pairs, middle = next 9%, flat = bottom 90%); aux = embeddings, head and norm gains. The gradient change each part causes (H·ΔW and G·ΔW) is estimated unbiasedly with two independent sequence sets.

**Update energy (Frobenius²):**

| Run | Stiff | Middle | Flat | Aux |
|---|---|---|---|---|
| Muon | 0.039 | 0.22 | 1.50 | 9.3 |
| PD | 0.012 | 0.17 | 3.49 | 9.3 |

PD moves 3.3× less along stiff directions and 2.3× more along flat ones.

**Share of the squared gradient change caused, in every curvature class of the direction whose gradient moves** (both runs):

| Part | Share |
|---|---|
| Stiff | 70–90% |
| Middle | 10–20% |
| Flat | 2–10% |
| Aux | ~5% |

- GN-only products ≈ full Hessian: it is Gauss-Newton coupling, not the non-GN term.
- So the ~2% of update energy in the stiffest directions sets most of how every other direction's gradient changes. A rough estimate agrees: |G_fs ΔW_s| ~ √(h_f h_s)|ΔW_s| far exceeds h_f|ΔW_f| when h_s/h_f ~ 1e4.
- This explains the low one-step persistence in flat directions (0.2–0.4): their gradients are slaved to the stiff coordinates, which oscillate period-2.

**Mechanism** (picture, not yet a result)
- The stiff subspace is tiny, but it (1) caps the learning rate through its oscillation and (2) injects that oscillation into every flat direction's gradient through GN coupling.
- Momentum filters most of the period-2 part (gain ~1/(1+β) vs 1/(1−β)).
- The exact Newton step accounts for the coupling through the Schur complement: flat steps should follow g_f − G_fs G_ss⁻¹ g_s, the flat gradient after the stiff coordinates relax, with curvature G_ff − G_fs G_ss⁻¹ G_sf.
- PD and S∘PD help partly because they step less along stiff directions, so there is less oscillation, cleaner flat gradients and room for a larger LR (PD's best LR is 1.4× Muon's).

**Design idea** (principled and cheap): second-order where it is stiff, first-order where it is flat.
- Treat the small stiff subspace (top ~1% of Kronecker pairs, or top eigenvectors of C and B) with a (damped) Newton step, so it relaxes each step instead of oscillating.
- Let the data-norm update handle the flat bulk, with a learning rate no longer capped by stiff stability.
- Before building it: check with the dense-lag persistence (does the flat-direction correlation recover at even lags, i.e. period-2?), and measure how much of the stiff subspace is shared across layers.

## 2026-09-27 02:08: batch-size study, first results (one LR each; brackets running)

- **Muon.** 4M batch, LR 0.014: final 3.95243. At 1M batch the same seed reaches 3.70465. Four times the batch at the same tokens costs Muon 0.25 nats.
- **PD α ¼.** 4M batch, LR 0.02: **3.88473**, so PD − Muon = **−0.068**, against −0.018 at 1M: about 3.8× larger.
- **Step 200 at 4M:** Muon 4.360, PD 4.221, S∘PD (LR 0.02) 4.170.
- This is consistent with the gap-map prediction: more of the Newton decrement becomes reachable at larger batch, so curvature-aware preconditioning gains more. It matches the direction of the Gauss-Newton paper's batch-size result.
- Wait for the LR brackets: Muon {0.007, 0.028} and PD {0.01, 0.04} are queued on the allocations, S∘PD {0.01, 0.02, 0.04} runs on A6000.
- Also from the one-step test (corrected): along PD's actual next direction at step 200, the one-step optimal scale is only ~0.16–0.22× the LR in use. At the real LR, the quadratic model predicts a loss increase, driven by the stiff directions: the edge-of-stability compromise.

## 2026-09-27 02:09: one-step α (corrected), PD state at step 200

- **Best one-step decrease, and its optimal scale ÷ the LR in use:**
  - α 0: 0.00509 at 0.16×;
  - α 1/16: 0.00536 at 0.22×;
  - α 1/8: 0.00566 at 0.31×;
  - α 1/4: 0.00636 at 0.61×;
  - α 3/8: 0.00714 at 1.06×.
- At the LR in use, the quadratic model predicts a rise for α ≤ 1/8 and a fall for α ≥ 1/4.
- **Reading.** Stronger input preconditioning gives a better one-step direction, and moves its optimal scale up to the LR in use. This agrees with the progress-rate model.
- **Training's old preference for α ¼ at LR 0.01** may reflect that training likes running ~1.6× above the one-step optimum (α ¼ at LR 0.01 sits at 1/0.61).
- **Prediction for the LR-bracketed α sweep:** α 3/8 does best near LR ≈ 0.01 × 1.06/0.61 ≈ 0.017, and α ½ higher still.

## 2026-09-27 02:15: one-step α at step 500; two-sided optimizer ready

- **One-step α at step 500 (PD state).** Best decrease 0.00225 (α 0), 0.00245 (1/16), 0.00265 (1/8), 0.00303 (1/4), 0.00328 (3/8), 0.00336 (1/2). Optimal scale ÷ LR in use: 0.12, 0.18, 0.26, 0.52, 0.90, 1.31.
- At step 200 the trend continues to α ¾ (0.00815 vs 0.00779 at ½). Stronger input preconditioning gives better directions, saturating around 3/8–1/2 by mid-training.
- **Prediction for the LR-bracketed training sweep:** α 3/8 or ½ at LR ≈ 0.014–0.02 matches or beats α ¼ at 0.01.
- **Two-sided data norm built and smoke-tested:**
  - 2-rank CPU run through the distributed trainer, both label sources, replica audits equal at steps 1 and 10;
  - the trajectory differs from PD's (body weight difference 4.4 after 10 steps in the tiny model).

## 2026-09-27 02:21: gap map on the 4M-batch Muon trajectory (figures_gap4m/)

- **Reachable share of the decrement at the run's own batch.**
  - 4M trajectory (steps 183 and 330), at b = 8192 sequences: 0.85–1.0 for every kind (q 0.85–0.95).
  - 1M trajectory (steps 500 and 900), at b = 2048: q 0.12, k 0.2–0.4, o 0.35–0.5, up 0.4, down 0.33–0.39, v 0.95.
  - At 4M almost all of the second-order gain is signal-dominated per step.
- **Muon's effective-step profile at 4M is as skewed as at 1M:** about 0.03× to 50× vs the noise-aware ideal across curvature.
- **Reading.** At 4M the ideal is near plain Newton, so Muon's skew toward stiff directions costs much more, and preconditioning that corrects it gains more. This is consistent with PD − Muon = −0.068 at 4M vs −0.018 at 1M, one LR each so far.
- The stiff-direction oscillation (negative cos) is present at 4M too.
- Plot caveat: the gap-profile b* reference line is drawn at 2048 sequences. For the 4M trajectory the relevant batch is 8192.

## 2026-09-27 02:45: the Kronecker-frame parts of an update are strongly coupled; the frame-diagonal gap map overstates flat-direction under-stepping by 20–37×

**One-step split** (`one_step_split.py`, `one_step_split.json`). Each optimizer's actual update W[t+1] − W[t] is split in each matrix's frame by K-FAC rank: stiff = top 1% of pairs, middle = next 9%, flat = the rest. On 256 held-out sequences, the exact GN quadratic model is scored at the step taken, at the best single scale, and at the best separate scale per part.

| Run, step | Taken | Best single (scale) | Best separate (scales stiff / middle / flat) |
|---|---|---|---|
| Muon 200 | −6.24e-3 | −8.70e-3 (0.65) | −14.8e-3 (0.44 / −0.80 / 6.99) |
| PD 200 | −4.21e-3 | −6.74e-3 (0.62) | −9.23e-3 (0.53 / −0.50 / 3.24) |
| Muon 500 | −1.42e-3 | −4.75e-3 (0.54) | −4.85e-3 (0.51 / 0.32 / 1.25) |
| PD 500 | −1.11e-3 | −3.43e-3 (0.55) | −3.47e-3 (0.53 / 0.36 / 0.93) |
| Muon 900 | −0.69e-3 | −4.01e-3 (0.52) | −4.08e-3 (0.63 / 0.09 / 0.80) |
| PD 900 | −1.37e-3 | −2.69e-3 (0.59) | −2.70e-3 (0.65 / 0.39 / 0.72) |
| SOAP-Muon 500 | −2.36e-3 | −3.55e-3 (0.63) | −3.84e-3 (0.48 / 0.35 / 1.66) |
| S∘PD 500 | −1.74e-3 | −2.89e-3 (0.61) | −2.95e-3 (0.62 / 0.33 / 1.01) |

- **Every optimizer steps at about 1.6–1.9× the one-step optimum along its own update** (best single scale 0.52–0.65), at every checkpoint from 200 on. This looks like an edge-of-stability operating point.
- **From step 500 on, separate scales per part gain almost nothing**: 1–2% for Muon, PD and S∘PD, 8% for SOAP-Muon. At step 200 they gain 40–70%, but with a negative middle scale, so that estimate is fragile.
- **The three parts are strongly coupled** under the exact GN: correlations of their output-space images are 0.45–0.73.

**Frame-diagonal vs exact** (same updates; frame-diagonal = Σ h_ij D_ij² with the gap map's exact per-pair diagonal).

| Run, step | Stiff | Middle | Flat | Whole update |
|---|---|---|---|---|
| Muon 200 / 500 / 900 | 23 / 21 / 18× | 9 / 10 / 11× | 6 / 7 / 8× | 30 / 31 / 27× |
| PD 200 / 500 / 900 | 32 / 31 / 27× | 12 / 12 / 11× | 7 / 8 / 8× | 37 / 36 / 33× |
| SOAP-Muon 500 | 12× | 8× | 6× | 20× |
| S∘PD 500 | 20× | 11× | 7× | 27× |

(Exact q ÷ frame-diagonal q.)

- The first-order terms agree within about 5%.
- The frame-diagonal model predicts a best scale 12–24× the step actually taken, where the truth is about 0.55×.
- So the gap map's "flat directions are 20–100× under-stepped" came from its diagonal approximation.
  - Along the actual update, the flat part's own exact optimum is only 1.9–2.7×, and about 1× once coupling to the other parts is included.
- The per-pair diagonal numbers themselves (h, s, n per pair) stand. What fails is summing them as if the pairs were independent.
- **Where the coupling comes from, partly:** for Muon's direction, q(body) ÷ Σ q(kind) = 3–4 (`one_step_alpha_*.json`), so updates of different matrix kinds add up coherently in output space. The rest (up to ~30×) is cross-layer or within-matrix off-diagonal structure, not yet separated.
- SOAP-Muon's update is the least coherent (20×).

**Reading.**
- The per-direction (frame-diagonal) picture of the gap is quantitatively wrong for the updates the optimizers actually take.
- Along the update, the stiff/flat compromise costs little in one step. The real one-step budget is set by how coherently all the pieces move the output.
- This pushes the question to the exact GN step itself: how much better is the exact one-step GN direction, and what does it change? `one_step_gn.py` computes it by Lanczos on the exact GN matrix. It is compared on held-out data with Muon, PD, two-sided, K-FAC, EKFAC with the exact diagonal, and swaps of per-matrix norms between Muon and GN (shape vs allocation), at 1M and 4M gradient batches and with the momentum input. It also gives the 48×48 per-matrix GN Gram of Muon's and GN's direction, which separates cross-layer from within-matrix coupling.

## 2026-09-27 02:45: two-sided one-step test, and α on Muon's own trajectory

- **Output side, one step** (PD step 500, `one_step_twosided_PD.json`). Adding L = B^−β to PD α ¼:

| Setting | Best decrease | vs α ¼ alone |
|---|---|---|
| α ¼ alone | 3.364e-3 | — |
| + β ⅛, GN labels | 3.419e-3 | +1.6% |
| + β ¼, GN labels | 3.453e-3 | +2.6% |
| + β ½, GN labels | 3.465e-3 | +3.0% |
| + β ¼, data labels (EF) | 3.401e-3 | +1.1% |

  - On Muon's direction alone (α 0), β ½ adds 5%, while α ¼ adds 24%.
  - In this metric the output factor is a small effect next to the input factor. GN labels are slightly better than data labels.
- **α on Muon's own state** (step 500, `one_step_alpha_M.json`).

| α | 0 | ⅛ | ¼ | ⅜ | ½ |
|---|---|---|---|---|---|
| Best decrease | 4.18e-3 | 4.41e-3 | 4.55e-3 | 4.54e-3 | 4.37e-3 |

  - Optimal scale ÷ LR at α 0 is 0.51.
  - On the PD state the best α falls over training: ¾ at step 200, ½ at step 500, ¼ at step 900.
- **Noise note.** α 0 on the PD state scored 2.25e-3 with one set of 256 held-out sequences and 2.72e-3 with another. Absolute one-step numbers carry ~10–20% sampling noise; comparisons within one script (same sequences) are paired and much tighter.

## 2026-09-27 02:45: batch-size study, brackets nearly closed

Final validation loss at 1.54B tokens, batch 4M (4,194,304 tokens, 368 steps), seed 260925:

| Optimizer | LR 0.007 | 0.01 | 0.014 | 0.02 | 0.028 | 0.04 |
|---|---|---|---|---|---|---|
| Muon | 3.9567 | | **3.9524** | | running | |
| PD α ¼ | | 3.8854 | | **3.8847** | | running |
| S∘PD | | **3.8478** | | 3.8495 | | 3.8675 |

- **Best-LR gaps to Muon:**

| Batch | S∘PD − Muon | PD − Muon |
|---|---|---|
| 4M | −0.105 | −0.068 |
| 1M (same seed, one LR each) | −0.029 | −0.018 |

- The gains from curvature-aware preconditioning grow about 3.6–3.8× with the 4× larger batch. This is in line with the reachable-decrement result.

## 2026-09-27 03:25: dense-lag gradient persistence (Muon vs PD, steps 501–532)

Figure: `figures_gap/persistence_lags.png`; data: `persistence_lags/`. Setup:
- Frame and gradient at W[501] on sequence set A. Gradients at W[501 + lag] on a disjoint set B, for lags 1, 2, 3, 6, 7, 14, 15, 30 and 31.
- Classes are K-FAC curvature ranks within each matrix: top 0.1%, 0.1–1%, 1–10%, 10–50%, 50–100%.
- The runs are fresh A4000 reruns of the trajectory arms (`soaudit_denselag_20260927`). Each correlation is the cosine between two gradient vectors from one trajectory, not an average over time.

**Seen**
- **Muon.**
  - Stiff classes flip at lag 1 (−0.2 to −0.5). From lag 2 to lag 15 they sit near 0.
  - Flat classes are +0.1 to +0.24 at lag 1 and +0.05 by lag 3–6.
  - At lags 30 and 31, every class, flat ones included, comes back to +0.1..+0.5 and then −0.0..−0.4. This coincides with a gradient-norm burst at steps 531–532 in this run (0.415 and 0.472, against a typical 0.33–0.37).
  - So there is a period-2 oscillation that keeps its phase over 30 steps, with an amplitude that comes and goes, and it drags every class along.
- **PD.** A coherent pattern over the first ~7 lags in every kind and class, strongest in stiff classes:

| Lag | 1 | 2 | 3 | 6 | 7 |
|---|---|---|---|---|---|
| Correlation, stiff classes | −0.1 to −0.4 | **−0.65 to −0.8** | +0.3 to +0.56 | −0.25 to −0.45 | −0.2 to −0.4 |

  - By lags 14–15 it is ≈ 0.
  - Flat classes follow the same pattern at a third to half the amplitude (v: −0.33 at lag 2).
  - This is not period-2. It looks more like a slower, underdamped oscillation.
  - In a linear heavy-ball picture (β = 0.95) the oscillation period gives ηλ along a mode: period 2 means ηλ ≈ 3.9 (the edge), period 3 about 2.9, period 4 about 1.95. So PD's stiff modes would sit below the edge and Muon's at it. This is only a reading; one realization cannot pin down the frequencies.
- **Neither optimizer has a large persistent component.** Beyond ~10 steps the correlation in flat classes is ≤ 0.05, except for the burst.

**Reading.**
- The expected gradient at any step is mostly transient: oscillation in stiff modes, carried into every class by the GN coupling measured earlier.
- Momentum (β = 0.95, a ~20-step window) mostly averages these transients away and keeps a small persistent part.
- Per-step "signal" (ḡ²) therefore overstates the drive available for sustained descent. This is another reason the frame-diagonal Newton decrement was a poor guide.
- Muon and PD differ qualitatively in their stiff-mode dynamics: at the edge (period 2) versus below it (slower rotation).
- PD's state is also much sharper: the top GN Ritz value at step 500 is 134 on PD's state against 15 on Muon's (next entry). That fits sharpness adapting to the optimizer's effective step along stiff directions.

## 2026-09-27 03:40: the exact one-step Gauss-Newton direction (`one_step_gn.py`, plain Lanczos/CG, `gn/`, `figures_gn/`)

**Setup.**
- States: Muon and PD at step 500 of the 1M-batch runs, and Muon (LR 0.014) and PD (LR 0.02) at step 183 of the 4M-batch runs.
- Inputs:
  - a fresh gradient over 2048 held-out sequences (1M tokens) or 8192 (4M);
  - the optimizer's next momentum M' = 0.95 M + g, where g is the run's own batch gradient.
- Exact GN over all 48 hidden matrices, on 256 curvature sequences.
- Every direction is scored on 512 held-out sequences at its own best scale.
- Krylov depth 48 and 96 give the same answer for damping ≥ 1e-3 ρ, so plain CG is converged there.
- The EKFAC-preconditioned variant (CG on G + δP) was worse than plain CG on every state tested. Frame-diagonal preconditioning misses the coupling, as EKFAC's direction does. It was stopped after its first input (`gnp/`, partial).

**Share of the exact-GN one-step decrease** (fresh gradient; 1M → 4M gradient):

| State | Muon | PD α ¼ | PD α ½ | Two-sided | K-FAC | EKFAC | GN shape, Muon's norms |
|---|---|---|---|---|---|---|---|
| Muon, 1M run @500 | 0.68 → 0.55 | 0.78 → 0.64 | 0.79 → 0.68 | 0.77 → 0.65 | 0.42 → 0.32 | 0.51 → 0.39 | 0.95 → 0.92 |
| PD, 1M run @500 | 0.49 → 0.36 | 0.71 → 0.54 | 0.87 → 0.70 | 0.71 → 0.56 | 0.48 → 0.35 | 0.56 → 0.41 | 0.91 → 0.87 |
| Muon, 4M run @183 | 0.93 → 0.81 | 0.98 → 0.87 | 0.97 → 0.89 | 0.97 → 0.86 | 0.57 → 0.50 | 0.67 → 0.58 | 0.93 → 0.89 |
| PD, 4M run @183 | 0.83 → 0.71 | 0.95 → 0.83 | 1.02 → 0.90 | 0.94 → 0.82 | 0.65 → 0.55 | 0.75 → 0.63 | 0.96 → 0.90 |

**Seen**
- **The gap to GN grows with gradient accuracy.**
  - GN's own decrease rises 21–58% from a 1M to a 4M gradient, while Muon's direction rises only ~10–15%.
  - The best GN damping falls from ~1e-2..1e-3 ρ to 1e-3..1e-4 ρ; the less noisy gradient lets it trust the flat directions.
  - This is the one-step version of the batch-size result.
- **Where the gap is not.**
  - Per-matrix step allocation: GN's shape with Muon's per-matrix norms keeps 87–96%, and Muon's shape with GN's norms gains nothing.
  - Cross-validated per-matrix scales fitted on the exact 48×48 Gram do worse than one scale.
  - The output-side Kronecker factor: two-sided ≈ PD ¼.
- **Frame-diagonal curvature models lose to Muon.** K-FAC (best damping 1e-3) and EKFAC with the exact per-pair diagonal both do. They move energy to the bulk-bulk corner of the frame, where the coupling makes it expensive. Orthogonalization (polar) is a robust curvature proxy: gradient → Muon doubles the one-step decrease.
- **The within-matrix shape is where the gap lives.**
  - The exact GN step is far from orthogonal. Its singular values decay 1–2 decades across each matrix (`gn_singular_*.png`).
  - Its output change per whitened input direction (D C^½) is more concentrated than Muon's or PD's (participation ratio 0.00–0.07, against Muon 0.04–0.15 and PD ¼ 0.53–0.56; `gn_shape_Muon_1M_500_g4M.png`).
  - In the Kronecker frame, GN keeps relatively more energy on high-variance input directions and less in the bulk-bulk corner (`gn_frame_energy_*_g4M.png`).
- **The momentum as input reverses the ranking.**
  - For M', the best GN direction is heavily damped (0.1 ρ) and worse than every optimizer direction: Muon 1.4–1.9×, PD 1.5–2.1×, even the raw momentum 0.9–1.35× GN.
  - M' is a sum of ~20 past gradients whose true-gradient correlation has decayed (dense lags). Newton scaling amplifies exactly those stale flat components.
  - In one step, the fresh 1M gradient beats the momentum for every direction family: Muon 12.9e-3 vs 4.5e-3 on Muon's 1M state.
- **Coupling in numbers** (48×48 Gram of the momentum-input directions):
  - Muon's per-matrix pieces are strongly coherent: q(sum) ÷ Σ q(matrix) = 11.6–15.8, and ÷ Σ q(kind) = 3.7–3.9.
  - The GN direction's pieces: 2.2–2.4 and 1.8–1.9.
- **Beyond the optimal scale, the true loss curves up much faster along the GN direction than the GN model says.** At 2× its optimal scale the loss rises by +4 to +28e-3 where the quadratic says 0. Muon and PD directions stay within ±2e-3 of the quadratic. The GN step has a small trust region.
- **Sharpness.** The top GN Ritz value (Krylov space of the input) is 134 on PD's 1M state against 15 on Muon's, and 16 vs 7.7 on the 4M states. PD's states are much sharper in their top directions.

**Reading (principles, provisional).**
1. **The value of curvature information grows with gradient accuracy.** In one step, noise is what separates Muon from GN at 1M. With less noise, GN pulls further ahead, and so do the preconditioners that move toward it (PD α ½ over α ¼). This matches training: PD and S∘PD gain 3.7× more at 4M.
2. **Orthogonalization is a robust curvature proxy; frame-diagonal curvature is not.** Muon's polar divides by the momentum's own spectrum, which tracks the coherent top structure. K-FAC and EKFAC treat frame pairs as independent, and the coupling (20–37×) breaks that.
3. **The remaining gap is the within-matrix shape.** GN's step is not flat: it concentrates output changes on fewer directions, with a decaying spectrum. Muon's flat spectrum is an assumption the exact GN step does not share. Next: a partial-orthogonalization power U S^p V^T (p = ¼, ½), and PD α ¾ and 1, in one step at both gradient batches (running: `gn2/`, eight states on priv-g14 and g20).
4. **Newton scaling needs fresh gradients.** Applied to the momentum it amplifies stale components. A principled design has to separate noise averaging (momentum) from curvature scaling, for example by preconditioning the fresh gradient and averaging afterwards, with averaging strength set by the noise level.

## 2026-09-27 04:26: second GN round (eight states, `gn2/`), the GN step's decorrelated pieces, and training checks

**What the second round adds.** States:
- 1M runs: Muon @500 and @1300, PD, SOAP-Muon and S∘PD @500;
- 4M runs: Muon, PD and S∘PD @183.

Plain CG, 48–64 steps. New directions: spectral powers U S^p V^T, PD α ¾ and 1, the fractional GN power, and per-kind shape swaps between PD α ½ and GN. Table: `scratchpad gn2_table.py`; the atlas "Exact Gauss-Newton" view shows all of it.

**Full orthogonalization is right within its family.**
- U S^p V^T with p = ¼ or ½ loses to p = 0 (Muon, polar) on every state and every input. For example, Muon @500 with a 4M gradient: 0.55 / 0.43 / 0.34 of GN for p = 0 / ¼ / ½.
- The same holds inside PD. Keeping part of the momentum's own spectrum never helps.

**Input power α.**
- Fresh gradient: the optimum is α ½ on Muon's states and ¾ on the PD-family states.
- Momentum input M':

| States | Best α |
|---|---|
| 1M | ¼ ≈ ½, with ¾ worse |
| 4M | ½ ≥ ¼ (S∘PD 4M: ¾ 1.34, ½ 1.30, ¼ 1.09 × GN) |

- The momentum-input ranking is the one that matched training at 1M:

| α (1M training, best LR) | ¼ | ⅜ | ½ | ⅛ |
|---|---|---|---|---|
| Final loss | 3.6891 | 3.6939 | 3.7125 | 3.6955 |

  So it predicts a larger best α at 4M. The 4M α ½ training test is now running.

**Two-sided, momentum input.** Best of all closed-form directions on 6 of 8 states: 1.44–2.36 × GN, against PD ¼ 1.39–2.20. Training agrees:

| 1M, LR 0.01, A4000 | Final loss |
|---|---|
| TS (β ¼, GN labels) | 3.68303 |
| PD | 3.68824 |
| Difference | −0.0052 |

A bracket and a second seed are running (`soaudit_twosided2_20260927`).

**Where the within-matrix GN shape matters** (fresh gradients; swap ratios, both inputs similar):
- Replacing one kind of the GN direction with PD α ½'s shape (at GN's norms) costs the most for **down** (0.71–0.94 of GN), then o (0.86–1.0) and v (0.91–1.0). For q and k, PD's shape is as good as GN's.
- Inserting GN's shape for all eight down matrices into PD α ½ gains +3% to +41%. The gain is largest late in training (Muon @1300: +41%) and on Muon's states.

**Energy profiles** (`figures_gn/gn_profile_Muon_1M_500_g4M.png`).
- Per unit norm, the GN step puts its energy on the TOP eigenvectors of both C (input) and B (output), falling about 100× toward the tail.
- Muon is flat over the input eigenbasis (by construction).
- PD and K-FAC rise toward the tail, the opposite direction.
- So GN does not "step more where the Kronecker curvature is low". It keeps the step on high-variance inputs and outputs and gets its advantage elsewhere.

**How: the GN step's per-matrix pieces are decorrelated** (`gram_saved.py`; Muon 1M @500, 4M gradient; correlation of per-matrix output changes, exact GN metric):

| Direction | q(sum) ÷ Σ q(matrix) | Mean correlation of pieces | One-step decrease |
|---|---|---|---|
| Muon | 16.4 | 0.39–0.54 | 16.2e-3 |
| PD ¼ | 16.9 | 0.38–0.52 | 18.8e-3 |
| K-FAC | 8.2 | 0.35–0.48 | 9.9e-3 |
| Exact GN | 3.2 | 0.08–0.15 | 27.2e-3 |

- Every per-matrix method pushes all 48 matrices' output changes in nearly the same function-space direction (the loss gradient), so the curvature of the combined step is 8–17× the block-diagonal sum.
- The GN step makes the pieces complementary.
- A global learning rate can absorb a uniform coherence factor, but not the redundancy: the per-matrix methods spend their step budget moving the function along one direction many times over.

**Next.**
- Does a handful of global GN eigen-directions carry this coherence? `one_step_gn.py` now projects the top-k (1, 4, 16, 32) global Ritz vectors out of Muon's and PD's directions and adds a Newton step inside that subspace (running on PD @500).
- A per-kind α scan and fresh + stale mixed inputs (g + ½ M) are running on dev-gpu (`gn3/`).

## 2026-09-27 04:45: two follow-ups that did not pan out (per-kind α, global deflation), and what they rule out

**Per-kind input power** (`gn3/`, Muon @500, 4M gradient). One kind at a time moved from α ½ to −¼ / 0 / ¼ / ¾, all others at ½:

| Kind | α −¼ | α 0 | α ¼ | α ¾ |
|---|---|---|---|---|
| q | 0.81 | 0.95 | 0.99 | 1.00 |
| k | 0.85 | 0.95 | 0.99 | 1.01 |
| v | 0.67 | 0.88 | 0.98 | 1.00 |
| o | 0.87 | 0.99 | 1.01 | 0.99 |
| up | 0.75 | 0.89 | 0.97 | 1.01 |
| down | 0.81 | 0.93 | 0.99 | 0.99 |

(Relative to all-α ½.)

- The optimum is ¼–¾, flat, for every kind; no kind wants α ≤ 0.
- So GN's top-heavy energy profile along C (previous entry) is not a negative input power. Its advantage in down is orientation, not input weighting.

**Down swaps, one layer at a time.** GN's shape in a single layer's down (blocks 1, 4, 8), inside PD α ½, gains +5.8%, +2.6% and +3.5%. All eight together gained +24%. The down gain is roughly additive over layers, so it is within-matrix, not coordination among the down matrices.

**Global top-k GN deflation** (PD @500, 4M gradient).
- Project the top 1, 4, 16 or 32 global Ritz vectors of the exact GN out of Muon's or PD's direction, and add the Newton step inside that subspace:

| Direction | Alone | With k = 16 | With k = 32 |
|---|---|---|---|
| Muon | 0.378 | 0.453 | 0.479 |
| PD α ½ | 0.739 | 0.785 | 0.41 (the k = 32 projection hurts) |

- The Newton step alone in the top-32 subspace gets 0.10 of GN.
- The top Ritz values form a broad cluster (135, 120, 115, 112, 108, 104, 100, 96, …), not a few isolated stiff modes.
- **So the coherence that separates the optimizers from GN is not carried by a handful of global stiff directions.** It is high-dimensional: at every token, every matrix pushes the logits along that token's own loss gradient. Cheap global deflation (a few tracked eigenvectors plus Newton in that subspace) is ruled out as the main fix.
- The input-whitened CG (gni) was also worse than plain CG on Muon's state (22.2 vs 26.4e-3), so plain CG remains the reference.

## 2026-09-27 05:29: how much stale averaging each direction tolerates (`gnmix/`: input g + c M), and the 4M α result

**Setup.** Input g + c·M: a fresh 1M gradient plus a fraction c of the saved momentum (c = 0.95 is the optimizer's M'). Share of the exact-GN decrease, with GN's own decrease in parentheses (×1e-3):

| State | c | GN | Muon | PD ¼ | PD ½ | PD ¾ | Two-sided |
|---|---|---|---|---|---|---|---|
| Muon @500 | 0 | (20.0) | 0.71 | 0.81 | 0.82 | 0.77 | 0.81 |
| Muon @500 | 0.25 | (15.1) | 0.84 | 0.94 | 0.95 | 0.87 | 0.95 |
| Muon @500 | 0.5 | (9.6) | 1.04 | 1.14 | 1.11 | 0.99 | 1.16 |
| PD @500 | 0 | (14.7) | 0.47 | 0.69 | 0.85 | 0.87 | 0.70 |
| PD @500 | 0.25 | (10.7) | 0.57 | 0.82 | 0.99 | 1.00 | 0.83 |
| PD @500 | 0.5 | (5.8) | 0.84 | 1.15 | 1.33 | 1.26 | 1.18 |
| PD @500 | 0.75 | (3.1) | 1.16 | 1.52 | 1.62 | 1.44 | 1.56 |

- In one step, every direction family loses decrease as stale momentum is mixed in. GN loses the most: −52% on Muon's state by c = ½, against −30% for Muon and −32% for PD ¼.
- GN's best damping rises with staleness, from 1e-3 to 0.1 of ρ.
- The optimizers' directions overtake GN between c ≈ ¼ and ½.
- **Reading.** Curvature scaling needs a fresh gradient. Stale averaging adds components that GN amplifies (its flat directions), while orthogonalization bounds them.
- The one-step metric only charges momentum for staleness; it credits none of momentum's multi-step benefit (noise averaging across steps). It says nothing on its own about the best β in training. Given the 4M batch's lower noise, it motivates asking whether the best β falls with batch.

**4M α ½: closed for PD.**
- PD α ½: 3.8917 (LR 0.02), 3.9094 (LR 0.04). Its best is +0.007 vs α ¼'s best.
- S∘PD α ½ @0.01 is +0.023 vs the same-hardware α ¼ control (L40S 3.8454). S∘PD α ½ @0.02 is running.
- The batch-dependent-power prediction fails (MUON_CASE 05:17).

## 2026-09-27 05:53: warm-started CG does not reliably close the gap; the review's corrections, and what is running

**Warm-started CG** (`gnwarm/`, fresh 1M gradient, damping 1e-3 ρ). k GN products of CG on (G + λ)x = −b, starting from the optimizer's direction at its GN-optimal scale:

| State, start | Start | k = 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|---|
| Muon @500, from Muon | 0.71 | 0.83 | 0.86 | 0.77 | 0.82 | 0.87 |
| Muon @500, from PD ½ | 0.82 | 0.90 | 0.83 | 0.82 | 0.84 | 0.87 |
| PD @500, from Muon | 0.47 | 0.61 | 0.64 | 0.59 | 0.66 | 0.75 |
| PD @500, from PD ½ | 0.85 | **0.35** | 0.37 | 0.43 | 0.60 | 0.74 |

- CG optimizes the curvature-sample quadratic, and its early iterates can be much worse on held-out data. On PD's sharp state, one step from a good start is a large loss.
- A "Muon + a few GN products" optimizer is not supported. It stays shelved (MUON_CASE 05:44).

**The review's corrections** (MUON_CASE 05:41) change the reading of the 03:40–04:45 entries:
- The GN direction's one-step advantage exists only near its own optimal scale. At the 1.6–1.9× scale where the optimizers run, it loses on the 4M states.
- The coherence ratio does not rank directions (GN on the momentum has low coherence and still loses). The cross-matrix vs within-matrix question stays open until the block-GN test.
- Clipping rates differ a lot between compared arms. The GN references at 4M-gradient dampings are not fully converged, and selection happens on the scored sequences.

**Running.**
- `one_step_blockgn.py`: exact block-diagonal GN per kind, layer and matrix, plus block Gauss-Seidel, against full GN. Cross-fitted scoring, one global scale, the same Krylov budget per block; smoke test pending.
- `eos_invariant.py`: LR × s × λ_max(R G R) in each optimizer's own coordinates.
- Two-sided controls: placebo basis, data labels, a clipping-matched pair (`soaudit_tsctl_20260927`). The TS LR bracket at 1M is done: 3.6827 @0.007 and 3.6830 @0.01, against PD @0.01 3.6882 on the same A4000; TS @0.014 is about 3.691.
- The 4M momentum sweep (`soaudit_mom4m_20260927`) and TS at 4M (`soaudit_ts4m_20260927`).

## 2026-09-27 06:20: an edge-of-stability invariant for Muon, not for PD (`eos_invariant.py`, `eos_invariant.json`)

**Setup.** LR × s × λ_max(R G R) in each optimizer's own coordinates: R = (C / mean + 1e-3)^−α, with R = I for Muon. s is PD's norm-matching factor from the run's log. λ_max is from 30 Lanczos steps on the exact GN.

| State | α | LR | s | λ_max | Invariant | Logged grad. norm |
|---|---|---|---|---|---|---|
| Muon 1M @200 | 0 | 0.007 | 1 | 19.3 | 0.135 | 0.44 |
| Muon 1M @500 | 0 | 0.007 | 1 | 14.8 | **0.104** | 0.33 |
| Muon 1M @900 | 0 | 0.007 | 1 | 15.0 | **0.105** | 0.30 |
| Muon 4M @183 | 0 | 0.014 | 1 | 7.7 | **0.108** | 0.38 |
| PD α ¼ 1M @500 | ¼ | 0.01 | 0.60 | 10.8 | 0.065 | 0.64 |
| PD α ¼ 1M @900 | ¼ | 0.01 | 0.58 | 11.0 | 0.064 | 0.62 |
| PD α ¼ 4M @183 | ¼ | 0.02 | 0.64 | 2.1 | 0.026 | 0.42 |
| PD α ½ 4M @183 | ½ | 0.02 | 0.26 | 11.0 | 0.057 | 1.20 |

- **Muon has an edge-of-stability invariant.** LR × λ_max is 0.104–0.108 across batch sizes and training steps once past the early phase (0.135 at step 200). The sharpness self-tunes to Muon's step.
- **PD's raw "9× sharper" state is mostly a coordinate effect.** In its own coordinates λ_max(R G R) is 11 against a raw 134.
- **PD runs below Muon's edge.** Its invariant is 0.064–0.065 at 1M and only 0.026 at 4M (λ_max 2.1). PD's stiff modes are sub-critical, which matches its slower, non-period-2 oscillation in the dense-lag data.
- Yet PD's LR bracket at 4M rejects a larger LR (0.04 is worse than 0.02). So something other than the stability edge in PD's geometry sets PD's best LR: gradient noise, the auxiliary Adam parameters, or momentum.
- α ½ at 4M sits closer to the edge (0.057) and loses in training.

## 2026-09-27 06:53: the two-sided gain needs the GN output basis (placebo), a second seed agrees, and it grows at 4M

**Two-sided minus PD α ¼**, validation loss, LR 0.01, A4000 (the controls are still running):

| Step | Seed 260925 | Placebo basis (same seed) | Seed 260926 |
|---|---|---|---|
| 100 | −0.029 | +0.012 | −0.005 |
| 200 | −0.022 | +0.012 | −0.008 |
| 300 | −0.016 | +0.008 | −0.016 |
| 500 | −0.008 | | −0.015 |
| 1000 | −0.005 | | −0.013 |
| Final | **−0.0052** | | |

- **Placebo** (B's eigenvalues in a fixed random orthogonal basis per matrix): worse than PD while the real TS leads. So the gain comes from the GN output eigenbasis, not from any output-side reweighting.
- **Second seed:** the TS lead is similar or larger through step 1000; the final is pending.
- **β ½ with GN labels** tracks β ¼ (−0.003 to −0.008 through step 900).

**Bracket.**

| Batch | TS by LR | TS best | vs PD α ¼'s best |
|---|---|---|---|
| 1M | 3.6827 / 3.6830 / 3.6864 at 0.007 / 0.01 / 0.014 | 3.6827 | −0.0055 |
| 4M | 3.8658 / 3.8688 at 0.01 / 0.02 | 3.8658 | −0.019 (vs 3.8847) |

**Reading.** The GN output factor is a real preconditioner. Like the input side, it gains ~3–4× more at the larger batch. It is placebo-controlled, one seed-pair is complete, and a second is well ahead. The data-label and clipping-matched controls are still to come.

## 2026-09-27 07:05: exact block GN, per matrix / layer / kind (`one_step_blockgn.py`, `blockgn/`)

**Setup.** Fresh 4M gradient, 128 curvature sequences. Each block is solved by Lanczos on its own exact GN block: 16 steps per block, 96 for full. Damping and the global scale are chosen on one half of 512 held-out sequences and scored on the other half, and vice versa.

**Share of the full-GN one-step decrease** (full GN in parentheses, ×1e-3):

| State | Muon | PD α ½ | GN per matrix (48 blocks) | per layer (8) | per kind (6) | full |
|---|---|---|---|---|---|---|
| Muon 1M @500 | 0.62 | 0.78 | 0.86 | 0.94 | 0.97 | 1 (23.3) |
| Muon 4M @183 | 0.90 | 0.99 | 0.91 | 0.94 | 1.00 | 1 (32.2) |

- **Most of GN's advantage is available within single matrices.** Exact per-matrix GN gets 0.86–0.91. On Muon's 1M state that closes 63% of the Muon→full gap.
- The rest (0.09–0.14) comes from coupling across layers within one matrix kind: the per-kind blocks, which hold one kind across all 8 layers, reach 0.97–1.0.
- Coupling across kinds within a layer matters less: per layer is 0.94.
- This largely agrees with Abreu et al.'s "layerwise GN ≈ full GN", and it refutes the "cross-matrix redundancy is the core" reading of 04:26. The coherence of Muon's pieces is a symptom of per-matrix directions, not what limits them.
- **PD α ½ is between Muon and per-matrix GN on the 1M state (0.78), and above it on the 4M state (0.99 vs 0.91).** Kronecker input whitening with polar recovers a large part of what per-matrix GN offers. On the 4M state the 16-step per-block solves may be under-converged, so per-matrix GN is a lower bound there.
- Block Gauss-Seidel (per matrix, one sweep) and the PD 1M state are running.

**What this says about where to look.** The per-matrix GN block, which is non-Kronecker (cross-position terms, error–input dependence), is the right per-matrix target. PD and the two-sided rule are its Kronecker-with-polar approximations; per the swap tests of 04:26, the remaining per-matrix gap concentrates in the MLP down projection. A smaller, second target is coupling across layers within a kind, which only a cross-layer method could capture.

## 2026-09-27 07:30: momentum and clipping at 4M; block GN on PD's state

**Muon @0.014 at 4M, L40S, one seed:**

| Momentum setting | Final loss |
|---|---|
| β 0.95 (standard) | 3.9524 |
| β 0.9 | **3.9218** |
| β 0.81 (half-life matched in tokens) | 3.9437 |
| Nesterov (β 0.95) | 3.9933 |

- There is an interior optimum near 0.9.
- Nesterov, which puts weight ~2 on the fresh gradient and ~0.9 on the old momentum, is much worse (+0.041). "Fresher is better" is not monotone: the averaging has an optimum, and Nesterov's extra weight on the current gradient hurts under edge-of-stability dynamics.
- The 1M check of β 0.9 vs 0.95 is queued on A4000.

**Clipping acts early.** Muon at 4M without clipping: gradient norms spike to 9.6–10.6 at steps 11–13, and the loss stalls there (7.63 vs 7.25 at step 15). It is 0.3 behind by step 20 and still +0.36 at step 100 (final pending).
- With clipping (norm 1.0), the momentum is built from normalized gradients during the spiky early phase (norms 2–10 in steps 1–30). Polar normalizes the step size but not the momentum's composition.
- Over the whole run only 7% of Muon's steps clip, but those early steps set the trajectory.
- For S∘PD (clipped on 94–96% of steps) the effect runs all through training.

**Block GN on PD's 1M state (16 Lanczos steps per block).**

| Direction | Share of full GN |
|---|---|
| Muon | 0.42 |
| PD α ½ | 0.79 |
| GN per kind | 0.77 |
| GN per layer | 0.78 |

- Full GN is 20.3e-3.
- Unlike on Muon's states, block GN does not reach full GN here. The convergence check with 48 steps per block is queued: PD's state is sharper and harder to solve.

## 2026-09-27 07:37: one Gauss-Seidel sweep of per-matrix GN solves reaches full GN

**Setup.** Muon 1M @500, fresh 4M gradient, cross-fitted scoring. One forward sweep over the 48 matrices in model order (block01.q, k, v, o, up, down, block02.q, …). Each matrix solves its own exact GN block (16 Lanczos steps, damping 1e-3 ρ) against the gradient after the matrices already moved. The update uses one exact cross-block GN product per matrix.

| Direction | Decrease (×1e-3) | Share of full GN |
|---|---|---|
| Per-matrix Jacobi (independent blocks) | 20.1 | 0.86 |
| Per-matrix Gauss-Seidel (one sweep) | **23.1** | **0.99** |
| Full GN | 23.3 | 1.00 |

- Per-matrix curvature plus a sequential account of how each update changes the later matrices' gradients recovers essentially all of GN's one-step advantage. The symmetric cross-block coupling of the full GN adds nothing beyond that.
- **So what separates a per-matrix method from GN is small and sequential.** Each matrix should step against the gradient after the others have moved, not against the stale joint gradient.
- It matches the block pattern: per kind 0.97 (cross-layer within a kind), per layer 0.94, per matrix 0.86.
- **Open.**
  - Which part of the sweep carries the gain: forward (input-shift) coupling from earlier to later layers, or coupling within a layer?
  - Whether a cheap surrogate exists: propagate the earlier layers' input shift into the later layers' updates without extra GN products.
  - Gauss-Seidel on Muon's 4M state and PD's 1M state is queued.

## 2026-09-27 07:45: clipping matters for Muon even more than for S∘PD at 4M

**4M, L40S** (clipped means norm 1.0):

| Optimizer | Clipped | Without clipping | Difference |
|---|---|---|---|
| Muon @0.014 | 3.9524 | 4.0129 | **+0.060** |
| S∘PD @0.01 | 3.8454 | 3.8657 | +0.020 |

- Without clipping on either side, S∘PD − Muon = −0.147. So clipping does not inflate S∘PD's advantage; it helps Muon more.
- Muon's loss comes from the spiky early phase: gradient norms of 9.6–10.6 around steps 11–13, a loss stall, and a lag that never fully closes (+0.36 at step 100, +0.08 at 300, +0.06 at the end).
- At 4M the warmup is only 13 steps and the early phase is a large share of the run. Clipping makes the early momentum an average of normalized gradients.
- **1M no-clip pair (A4000, in progress).**
  - At step 500, PD without clipping is +0.021 behind clipped PD.
  - At step 600, TS without clipping is +0.006 behind clipped TS.
  - Clipping helps both; so far it helps PD more, which would make the 1M TS gain partly independent of clipping.
- **Placebo, step 800.** 3.8937, against PD 3.8860 and TS 3.8816 (A4000). The placebo is still behind PD.
- **Next.** Whether normalizing every gradient (clip 0.1) helps beyond clip 1.0, for Muon and PD at 4M (queued on g20).

## 2026-09-27 08:10: Gauss-Seidel on Muon's 4M state; per-matrix GN on PD's state

| State | Muon | PD α ½ | Per-matrix Jacobi | Per-matrix Gauss-Seidel | Per kind | Full GN |
|---|---|---|---|---|---|---|
| Muon 1M @500 | 0.62 | 0.78 | 0.86 | **0.99** | 0.97 | 1 |
| Muon 4M @183 | 0.90 | 0.99 | 0.91 | **0.97** | 1.00 | 1 |
| PD 1M @500 (16 steps per block) | 0.42 | 0.79 | 0.65 | pending | 0.77 | 1 |

- On both Muon states, one sequential sweep over the 48 per-matrix solves recovers 97–99% of full GN.
- On PD's sharper state, the 16-step block solves fall below PD α ½ itself. Convergence is the likely reason; the 48-step check and Gauss-Seidel are queued.
- Also queued: the sweep in reverse order, and Gauss-Seidel within layers only (layers independent). These ask which coupling the sweep captures.

## 2026-09-27 08:22: Gauss-Seidel on PD's state

- PD 1M @500, fresh 4M gradient, 16 Lanczos steps per block. Share of full GN (20.3e-3):

| Direction | Share |
|---|---|
| Muon | 0.42 |
| Per-matrix Jacobi | 0.65 |
| PD α ½ | 0.79 |
| **Per-matrix Gauss-Seidel** | **0.90** |

- The sweep's gain over independent per-matrix solves holds on all three states:

| State | Per-matrix Jacobi | Per-matrix Gauss-Seidel |
|---|---|---|
| Muon 1M | 0.86 | 0.99 |
| Muon 4M | 0.91 | 0.97 |
| PD 1M | 0.65 | 0.90 |

- Running next:
  - Staged versions of Muon and PD α ½: the same per-matrix maps, but each matrix sees the gradient after the earlier ones stepped. Inputs are a fresh 4M gradient or the next momentum.
  - The sweep in reverse order, and within layers only.
  - The 48-step block solves.

## 2026-09-27 08:32: block-GN convergence check (48 Lanczos steps per block)

- **Per-kind block GN:**

| State | 16 steps per block | 48 steps per block |
|---|---|---|
| Muon 1M @500 | 22.60 (0.97) | 22.59 (0.97) |
| PD 1M @500 | 15.72 (0.77) | **18.93 (0.93)** |

  (×1e-3, share of full GN in parentheses.) Muon's state was already converged at 16 steps.
- So on PD's sharper state the 16-step block solves were under-converged. With enough steps, block GN reaches 0.93 of full GN there too.
- The per-matrix Jacobi (0.65) and Gauss-Seidel (0.90) numbers on PD's state are lower bounds.
- The within-matrix plus sequential picture holds on all three states.

## 2026-09-27 09:08: staged ("Gauss-Seidel") Muon and PD: a large gain with fresh gradients, a loss with the momentum

**Setup.** Muon 1M @500. Staged means Muon's or PD α ½'s own per-matrix map applied in model order, with each matrix seeing the gradient after the earlier matrices' steps (exact GN products). "×k" is the step the earlier matrices take, relative to the plain direction's GN-optimal scale. Scores are cross-fitted, ×1e-3; full GN on the fresh 4M gradient is 23.3.

| Input | Muon | PD α ½ | Staged Muon (×0.5 / ×1 / ×2) | Staged PD (×0.5 / ×1 / ×2) |
|---|---|---|---|---|
| Fresh 4M gradient | 14.5 | 18.1 | 18.1 / 21.2 / **24.8** | 26.6 at ×2 |
| Next momentum M' = 0.95 M + g | 4.46 | 4.46 | **3.18** / 2.75 / 2.07 | **2.84** / 2.71 / 2.56 |

- **With a fresh gradient, staging is large.** Staged Muon beats plain Muon by 71% and staged PD beats plain PD by 47%. Both exceed the damped full-GN reference (1.07 and 1.14 of it), and both still rise at the largest step tested (×2).
- **With the optimizer's momentum as input, staging hurts** (−29% for Muon, −37% for PD α ½). This is the same pattern as Newton scaling: curvature-based corrections (the GN inverse, sequential GN coupling) help on fresh gradients and hurt on stale momentum.
- **Reading.**
  - Staging is where GN's advantage over per-matrix methods comes from on this state.
  - It can only be used where gradients are fresh (large batch, short or no averaging). In the momentum regime today's optimizers run in, a within-step sequential correction adds error. The next step's gradient already carries the coupling, one step late.
  - Queued: partly fresh inputs (c = 0.25 and 0.5), to find where the crossover lies.

**Placebo two-sided run: timed out at step 1320** (3-hour limit on a slower A4000 node; the failure record is kept). Through step 1300 it trails PD by +0.004 to +0.008 at every validation step, while the real TS leads by −0.003 to −0.008. That comparison is clear without the final.

## 2026-09-27 09:45: Gauss-Seidel order and scope; clip 0.1 for Muon at 4M; 4M second seed

**Which coupling does the Gauss-Seidel sweep capture?** (Muon 1M @500, fresh 4M gradient, cross-fitted, ×1e-3; full GN 23.3)

| Direction | Decrease | Share of full GN |
|---|---|---|
| Per-matrix Jacobi | 20.0 | 0.86 |
| Sweep within each layer only (layers independent) | 20.55 | 0.88 |
| Per-layer exact blocks | 21.9 | 0.94 |
| Per-kind exact blocks (one kind across all layers) | 22.6 | 0.97 |
| Sweep over all 48 matrices, model order | 23.10 | 0.99 |
| Sweep over all 48 matrices, reverse order | 23.48 | 1.01 |

- **Order does not matter; crossing layers does.** A sequential pass through all layers, either way round, recovers full GN. The same pass restricted to within layers barely beats independent solves.
- Coupling of the same kind across layers (per kind 0.97) matters more than coupling between kinds within a layer (per layer 0.94).
- A cheap surrogate for GN's advantage would therefore have to carry the effect of other layers' updates, for example through the residual stream, not a within-layer correction.

**Clip 0.1 (normalized gradients before momentum) at 4M, β 0.95:**

| Optimizer | Clip 0.1 | Clip 1.0 | Change |
|---|---|---|---|
| Muon @0.014 (Ada; clip 1.0 on L40S) | 3.9338 | 3.9524 | −0.019 |
| PD α ¼ @0.02 (Ada) | 3.8746 | 3.8847 | −0.010 |

- Both gain, about 60% as much as from β 0.9 (−0.031, −0.022).
- One reading: with raw gradients, the sum-momentum is dominated by the early large-norm gradients (the median norm falls from ~4 to ~0.6 within 50 steps at 4M). Normalizing makes it fresher. Both changes reduce staleness.

**4M, β 0.9, seed 260926 (L40S):** Muon @0.014 3.9239, TS @0.01 3.8545 (−0.069). Seed 260925 gave −0.072 (TS on Ada). PD and S∘PD for this seed are running on priv-g14.

## 2026-09-27 09:55: where staging stops paying; the edge does not depend on momentum

**Staged Muon vs plain Muon as the input gets staler** (Muon 1M @500; input g + c M, M the saved momentum; cross-fitted, ×1e-3):

| c | Input | Muon | Staged Muon (best ×) | Ratio | PD α ½ |
|---|---|---|---|---|---|
| 0 | fresh 4M gradient | 14.48 | 24.81 (×2) | 1.71 | 18.13 |
| 0.25 | fresh 1M + 0.25 M | 11.32 | 17.06 (×2) | 1.51 | 13.24 |
| 0.5 | fresh 1M + 0.5 M | 8.90 | 10.70 (×0.5–1) | 1.20 | 9.85 |
| 0.95 | next momentum M' | 4.46 | 3.18 (×0.5) | 0.71 | 4.46 |

- Staging stops paying near c ≈ 0.7. The best step shrinks with it: ×2 → ×0.5.
- Both the plain directions and the staged gain decay with the stale share. The one-step value of Muon's own input is three times below a fresh gradient's, so the momentum's staleness is the largest one-step gap measured so far.

**Edge-of-stability number LR × λ_max(G) at 4M, step 183 (Muon, R = I):**

| Muon variant | LR | λ_max | LR × λ_max |
|---|---|---|---|
| β 0.95 | 0.014 | 7.7 | 0.108 |
| β 0.9 | 0.014 | 7.02 | 0.098 |
| β 0.81 | 0.014 | 7.46 | 0.104 |
| Nesterov, β 0.95 | 0.014 | 7.81 | 0.109 |

- Muon 1M at steps 500 and 900 gives 0.104 and 0.105.
- **The edge does not depend on momentum or batch size.** The prediction LR λ_max ≈ 2(1 − β) is refuted: at β 0.81 it would be 0.38.
- Heavy ball's linear threshold 2(1 + β) (Nesterov: 2(1 + β)/(1 + 2β)) does not organize these either. Nesterov's threshold is 2.9× lower, yet its number is the same.
- A normalized update sets the edge. Muon moves at most LR per step along any rank-one direction, so a stiff mode sits in a limit cycle whose amplitude scales with LR.
- PD α ¼ at 4M in its own coordinates: 0.026 (β 0.95), 0.0315 (β 0.9), 0.0306 (β 0.81). It is also independent of β, and 3–4× below Muon's.
- Checks still running:
  - LR 0.02 at β 0.9: an edge predicts λ_max ≈ 5.
  - The step-330 states.
- **Linearized check (smoke test only).** Newton-Schulz's Jacobian at the next momentum (`eos_linearized.py`) puts LR μ_max(J G) at ≈ 15 for both the Muon and the PD state. That is 3.9× above the linear heavy-ball bound. It comes from directions where the momentum is small and Newton-Schulz's gain is large (up to 492/|M|); the map saturates there.
  - So the stiff modes run past linear stability and are held by saturation: a nonlinear limit cycle, not a linear edge.
  - Converged runs are queued on the cluster.

**Two-sided control with data labels (empirical Fisher), 2026-09-27 09:56.** TS α ¼ β ¼ with B from data labels: 3.68516. With model-sampled (GN) labels: 3.68303. PD α ¼: 3.68824. All A4000, seed 260925, LR 0.01.
- Data labels keep 60% of the gain (−0.0031 against −0.0052).
- The ordering matches theory, since the GN output factor is the curvature and the empirical Fisher is a biased stand-in. But the 0.002 difference is inside the seed-to-seed spread.
- With the placebo result (a random basis with the same spectrum trails PD), the gain needs an output basis aligned with the curvature; data labels give most of that basis.

**4M, β 0.9, second seed complete for Muon, PD and TS (seed 260926, L40S), 2026-09-27 10:05:** Muon 3.9239, PD α ¼ 3.8617 (−0.062), TS 3.8545 (−0.069); S∘PD is running.
- Seed 260925 (Muon on L40S, PD and TS on Ada) gave −0.059 and −0.072.
- On one hardware type, TS − PD = −0.0072. With seed 260925, TS − PD = −0.013.

**Edge checks, completed (Muon at 4M):**
- **LR 0.02 at β 0.9** (step 183): λ_max 4.1 and LR × λ_max 0.082, against 7.02 and 0.098 at LR 0.014. The state is flatter at the higher LR, by more than 1/LR.
- **Step 330** (just before cooldown, same peak LR): 0.069 (β 0.95), 0.071 (β 0.9), 0.075 (β 0.81). Late in training the state sits below its step-183 value, still independent of β.

## 2026-09-27 10:16: momentum at 1M; the linearized edge

**Muon β 0.9 vs β 0.95 at 1M** (A4000, seed 260925, LR 0.007): 3.70503 vs 3.70564, **−0.0006**. At 4M the same change gains −0.031.
- β 0.9 led by 0.06–0.10 at steps 100–200 and by 0.001 at step 1000.
- **The best averaging window depends on batch size.** At 1M, longer averaging costs nothing in the end. At 4M, where each gradient is 4× less noisy, the shorter window wins clearly. Stale averaging trades noise for staleness, and its best balance moves toward fresh as the batch grows.

**Linearized edge of stability (`eos_linearized.py`).** A perturbation of the weights follows heavy ball with preconditioner J = df/dM at the next momentum. J is symmetric; f is the run's own map (Newton-Schulz; for PD, whitening and norm matching included). Heavy ball's bound is LR μ_max(J G) < 2(1 + β).

| State | LR μ_max(J G) | Bound | Ratio |
|---|---|---|---|
| Muon 1M @500 | 3.78 | 3.90 | 0.97 |
| PD α ¼ 1M @500 | 3.28 | 3.90 | 0.84 |

- Muon sits at the edge of its own linearized dynamics, with several modes near it (the top three are 0.97, 0.92 and 0.87 of the bound). PD sits somewhat below.
- The raw number LR λ_max(G) ≈ 0.1 is this edge seen through J, whose gain on the stiff modes is ~36 (μ/λ = 540/14.8).
- The momentum and LR variants at 4M, including Nesterov with its 2.9× lower bound, are running on g20.

## 2026-09-27 11:03: every step lands on the far side of the valley (midpoint losses)

`midpoint_loss.py`: the held-out loss (1024 sequences) at W_t + c (W_{t+1} − W_t) for c from −0.5 to 1.5, all parameters interpolated, from each kept state and its next-step weights. A parabola fit gives the loss minimum along the step ("argmin", in step units). The midpoint drop is the average of the endpoint losses minus the loss at c = ½.

| State | Step change | Midpoint drop | argmin |
|---|---|---|---|
| Muon 1M @50 / 200 / 500 / 900 / 1300 | −0.011 / −0.006 / −0.000 / −0.001 / +0.001 | 0.017 / 0.007 / 0.0053 / 0.0046 / 0.0030 | 0.59 / 0.60 / 0.51 / 0.54 / 0.45 |
| PD α ¼ 1M, same steps | −0.029 / −0.005 / −0.001 / −0.002 / 0.000 | 0.011 / 0.006 / 0.0039 / 0.0029 / 0.0022 | 0.87 / 0.60 / 0.52 / 0.57 / 0.50 |
| SOAP-Muon 1M | −0.032 / −0.008 / −0.001 / −0.000 / −0.000 | 0.006 / 0.006 / 0.0031 / 0.0034 / 0.0024 | 1.15 / 0.67 / 0.55 / 0.51 / 0.51 |
| S∘PD 1M | −0.030 / −0.008 / +0.000 / 0.000 / +0.001 | 0.008 / 0.005 / 0.0030 / 0.0022 / 0.0017 | 0.98 / 0.71 / 0.49 / 0.50 / 0.44 |
| Muon 4M @183: β 0.95 / 0.9 / 0.81 / Nesterov | −0.006 / −0.008 / −0.003 / −0.006 | 0.017 / 0.026 / 0.034 / 0.031 | 0.56 / 0.56 / 0.54 / 0.55 |
| PD α ¼ 4M @183: β 0.95 / 0.9 | −0.006 / −0.009 | 0.012 / 0.022 | 0.58 / 0.57 |
| S∘PD 4M @183 (β 0.95) | −0.003 | 0.009 | 0.55 |
| TS 4M @183 (β 0.95) | −0.000 | 0.011 | 0.52 |

- **Past the first ~200 steps, every optimizer's step overshoots about 2× along its own direction.** The loss minimum sits at 0.44–0.58 of the step and the loss barely changes per step: each step crosses the valley. Early (step 50) the preconditioned optimizers still descend (argmin 0.9–1.15) while Muon already overshoots (0.59).
- **Stored oscillation loss at the midpoint:**
  - At 1M it is ordered like the final losses: Muon > PD > SOAP-Muon ≈ S∘PD.
  - At 4M it is 3–5× larger and grows as β falls (Muon 0.017 → 0.026 → 0.034 for β 0.95 → 0.81).
  - Yet β 0.9 has the best final loss at 4M, so stored loss (released by the cooldown) is not what decides the outcome.
- **Two-point river/hill split (`river_hill.py`) is inconclusive for separating the persistent part.** Consecutive full-batch gradients have cos −0.18 (Muon 1M), +0.01 (PD 1M), −0.53 to −0.57 (4M): the oscillation is not period-2 everywhere, so the half-sum still contains oscillation. The dense-lag persistence data remain the reference for its time structure.

**Consecutive full-batch gradients (`river_hill.py`, 4M held-out tokens at W_t and W_{t+1}, same data), 2026-09-27 11:20.**

| State | cos(g_t, g_{t+1}) | cos(M_t, g_t) | cos(ΔW, −g_t) |
|---|---|---|---|
| Muon 1M @500 / @900 | −0.18 / −0.24 | −0.40 / −0.40 | +0.04 / +0.04 |
| PD α ¼ 1M @500 | +0.01 | −0.52 | +0.01 |
| Muon 4M @183, β 0.95 / 0.9 / 0.81 | −0.53 / −0.53 / −0.58 | −0.65 / −0.70 / −0.77 | +0.06 / +0.08 / +0.10 |
| Muon 4M @183, Nesterov | −0.82 | −0.73 | +0.09 |
| PD α ¼ 4M @183, β 0.95 / 0.9 | −0.57 / −0.56 | −0.67 / −0.76 | +0.03 / +0.04 |

- The momentum after step t is anti-aligned with the full-batch gradient at W_t in every state; it carries the oscillation's previous half-cycle. (Review, 11:24: at 1M the top-16 GN modes hold 56% (Muon) and 44% (PD) of the gradient's energy, so "dominated" holds only for those few modes.)
- The step taken has only a small descent component along −g_t.
- The flip between consecutive gradients is strongest for Nesterov (−0.82), which is also the worst 4M Muon variant (+0.041 against heavy ball). PD at 1M has a quarter-turn per step (cos ≈ 0), matching its slower oscillation in the dense-lag data.
- This is descriptive: it fixes the phase structure, not how much of each gradient is persistent.

## 2026-09-27 11:31: a 16-step anneal releases 7× the one-step stored loss (valley-floor premise check, part 1)

`anneal_branch.py` continues a run's own optimizer and state on its own next training batches, with the LR taken linearly from its scheduled value to ~0 over 16 steps (a cooldown in miniature). This follows the review at 11:24.

**Muon 1M from step 500.** Held-out loss (256 EVAL sequences) 3.9654 after the first step, then 3.9585 (k = 5), 3.9494 (k = 8), 3.9365 (k = 12) and 3.9291 (k = 16). The total drop from the branch point is 0.037.
- Normal training at this point lowers validation loss by about 0.011 per 16 steps (4.065 → 4.029 over steps 500–550). The anneal releases about 0.026 beyond that, against a one-step midpoint drop of 0.005.
- The run's real cooldown releases about 0.06 beyond the trend at the end (steps 1300 → 1469).
- So at step 500 a loss excess of ≈ 0.03 is maintained by the LR itself. It is the same size as GN's one-step gain there (0.020 on a fresh 1M gradient, 0.026 on the stale-free momentum). Whether GN keeps its advantage over PD/TS once this excess is released is the question the re-score at the annealed state answers (running). The PD 1M and Muon/PD 4M anneals are also running.

## 2026-09-27 12:08: transport one-step tables (complete), the valley-floor re-scores, and the GN trainer

**One-step value of each input under each map** (`transport_test.py`, tracked runs at step 500, oscillating states; cross-fitted, ×1e-3):

| Input | Muon run: GD / Muon / PD α ¼ / PD α ½ / GN | PD run: GD / Muon / PD α ¼ / PD α ½ / GN |
|---|---|---|
| Fresh 1M gradient | 6.3 / 12.8 / 14.9 / 15.2 / 19.6 | 1.8 / 6.5 / 10.0 / 12.4 / 14.5 |
| M' (next momentum) | 3.2 / 4.6 / 4.8 / 4.4 / 2.3 | 0.7 / 1.9 / 2.6 / 2.6 / 1.6 |
| M*' (stale-free) | 5.4 / 13.6 / 16.6 / 18.2 / 26.3 | 1.6 / 6.5 / 10.9 / 14.6 / 19.5 |
| M' + G Q | 3.6 / 7.4 / 8.5 / 8.7 / 9.4 | 1.2 / 3.3 / 5.0 / 6.3 / 7.0 |
| M' + G Q / 2 | 3.7 / 7.4 / 8.6 / 8.8 / 9.7 | 1.2 / 3.3 / 5.0 / 6.4 / 7.2 |
| M' + H Q | 3.9 / 7.4 / 8.2 / 7.9 / 9.3 | 1.4 / 3.4 / 4.7 / 5.7 / 6.2 |
| M' + per-matrix G Q | 3.2 / 10.0 / 11.7 / 11.7 / 6.4 | 1.2 / 4.9 / 7.3 / 8.9 / 1.1 |
| M' + K-FAC Q | 1.0 / 8.6 / 9.2 / 8.0 / 2.9 | 0.7 / 4.0 / 5.6 / 6.0 / 0.6 |
| Replay mean (100M-token current gradient) | 5.6 / 14.1 / 17.4 / 19.3 / 28.1 | 1.7 / 7.0 / 11.8 / 16.1 / 21.8 |

- Exact transport roughly doubles every map's one-step value on the momentum. Half the transport is as good as the full term, consistent with its 2× size error.
- Per-matrix and K-FAC transports help the polar maps but ruin the GN map.
- These are scores at oscillating states, so per the 11:24 review they include the loss an anneal would release. Transport is not pursued further until a training-level reason appears.

**Valley-floor re-scores** (details and table: `MUON_CASE.md`, 12:08):
- At the annealed states GN's one-step advantage over the optimizers grows. Muon reaches 0.27–0.40 of GN, PD α ½ 0.62–0.73.
- With 16M-token gradients it grows further: Muon 0.30, PD α ½ 0.50, GN +65%.
- The anneals themselves released 0.035–0.038 at 1M and 0.078–0.095 at 4M in 16 steps, without lowering the top GN eigenvalue.

**GN trainer (`gn_train.py`, the paper's recipe), first runs.** From Muon's 1M step-100 state, 2 GPUs, 16 inner Muon steps (64k tokens each) per 1M-token outer step, line search on never-seen training sequences, ~12.8 s per outer step.
- **Inner LR 0.007** (the baseline's LR): each inner loop overshoots its quadratic (GN-model change +0.22), and the line search sits at its smallest α (0.25).
- **Inner LR sweep:**

| Inner LR | α | Validation at step 108 |
|---|---|---|
| 0.002 | 0.5 | 5.304 |
| 0.001 | 0.71 | 5.341 |
| 0.0005 | 1.0 | 5.377 |

  Baseline Muon: 5.490 at step 100.
- A GN arm against a no-linearization control (same inner steps on true gradients at the inner iterate, same line search) is running for 80 outer steps. The control separates curvature feedback from taking more, smaller steps per batch.

**Provenance note (2026-09-27 12:35).** The GN-trainer runs so far (`gntrain_smoke`, `gntrain_sweep`, `gntrain_pair`) were launched with `.venv/bin/torchrun`. That entry script's shebang still points at `../last_layer/.venv/bin/python`, left over from the project's relocation, so those runs executed that environment's interpreter and site-packages.
- The use was read-only; nothing under `../last_layer` was written.
- Both environments have torch 2.11.0+cu130, so the numbers are computationally equivalent.
- From now on the trainers are launched with this project's `.venv/bin/python -m torch.distributed.run` (interpreter `/usr/bin/python3.11`, prefix `explore_muon/.venv`).
- Other entry scripts in `.venv/bin` (hf, httpx, …) carry the same stale shebang. The venv is left unchanged.

**The Newton trainer's first smoke test failed** (`job_newton_smoke_2625099.log`, kept): the GN product returned flat parts instead of matrix-shaped ones. This is fixed and the test re-run.

**Newton trainer v1 (CG + Levenberg-Marquardt) failed, 2026-09-27 12:53; its outputs are kept in `newton_ref/M1M_from500` and `newton_ref/PD1M_from500`.**
- The reduction ratio was measured at α = 1, not at the step taken, and on held-out data, where the batch-noise part of the predicted decrease does not transfer. So it was negative almost every step and λ grew ×1.5 per step: 0.06 → 330 by step 525, 3754 by step 575 (PD arm). The body steps shrank to ~1e-5.
- Validation still fell (PD arm 4.028 → 3.966 by step 575, against the PD baseline's 3.995 at 550). This came from the first steps (the excess release) and from the AdamW-updated embeddings and head.
- **v2** (same file): one k-step Lanczos run per step gives x(d) = −(G + d ρ I)^-1 g for d ∈ {1e-2, 1e-3, 1e-4}. The held-out line search picks damping and step together over a 3 × 5 grid, with a skip option. There is no damping heuristic, the same selection the cross-fitted one-step analyses use.
- Runs restart in new directories (`*_v2`).

**Newton v2 from Muon's 1M state, stopped at step 600 by choice (2026-09-27 13:26).**
- Validation against Muon's own run: 525 4.0035 (−0.044), 550 3.9912 (−0.038), 575 3.9831 (−0.031), 600 3.9761 (−0.023; baseline 3.9988). After the initial release Newton gains ~0.3–0.5e-3 per step, against Muon's ~0.6e-3.
- The PD-state arm shows the same trend: −0.034 at 525, then −0.017 at 600 and −0.002 at 650 (baseline 3.9469). It continues on the dev GPUs to the end.
- priv-g14 was freed for the 4M test (Newton from Muon's 4M state at step 183). The run's checkpoint at 600 and its step logs are kept.

**Provenance clarification (2026-09-27 13:58).** Runs launched with `.venv/bin/python -m torch.distributed.run` import torch from this project's `.venv` (`torch.__file__` = `explore_muon/.venv/.../torch/__init__.py`). Their logs still show `last_layer/.venv/...` paths in some warnings. Those are `co_filename` strings compiled into `.pyc` files that were copied at the relocation, not a second environment. Only the three earlier `torchrun`-launched runs (`gntrain_smoke`, `gntrain_sweep`, `gntrain_pair`) executed the other interpreter.

**Each optimizer's state is neutral only along its own step (normgn_probe.py, 2026-09-27 15:57).** 1M states at step 500. Held-out loss change when only the hidden matrices move by c × a step (512 held-out sequences, EVAL region):

| State | Step | c = ¼ | ½ | 1 | 2 |
|---|---|---|---|---|---|
| Muon | the run's actual step W501 − W500 | −0.0034 | −0.0048 | −0.0016 | +0.028 |
| Muon | Muon's step rebuilt from M' = 0.95 M + g501 | −0.0034 | −0.0047 | −0.0016 | +0.028 |
| Muon | M' at Muon's per-matrix norms | +0.018 | +0.119 | +0.51 | +1.63 |
| PD α ¼ | the run's actual step | −0.0025 | −0.0036 | −0.0015 | +0.019 |
| PD α ¼ | Muon's direction at PD's step norms | −0.0010 | +0.017 | +0.108 | +0.50 |
| PD α ¼ | M' at PD's per-matrix norms | +1.16 | +3.2 | +4.9 | +5.6 |

- The rebuilt Muon step matches the actual one to 1e-5, so the probe's momentum, batch and LR are the run's own.
- Both optimizers' own steps are best near c = ½ and about neutral at c = 1: the self-regulated edge along the step.
- Another direction at the same per-matrix norms meets curvature the dynamics never regulated. The PD state is ~30× sharper along Muon's direction than along its own.
- GN's direction normalized to the same per-matrix norms, at c = 1. The curvature sets are the trainer's 64 in-batch sequences and 64 or 256 fresh ones:

  | Lanczos steps | Muon state | PD state |
  |---|---|---|
  | 16 | +0.002 to +0.003 | +0.14 to +0.15 |
  | 64 (damping ≤ 1e-3 ρ) | −0.0006 to −0.0012 | +0.014 to +0.022 |

  - The direction saturates below damping 1e-3 ρ (tested down to 1e-5).
  - Held-out curvature along the step, over the set's own, is 1.5–3 for 64 sequences and 1.2–1.7 for 256 at the Muon state. At the PD state it is 1.1–1.8.
  - More Lanczos steps help much more than more curvature sequences. Krylov 256 and a trust-region solve at the trainer's radius are being measured.

**Normalized GN in training sharpens the network (17:10, interim at step 549).** `newton_train.py --normalize 0.001 --cg 64 --momentum 0.95` from Muon's 1M step-500 state (2× L40S). GN's direction on the momentum, each matrix at Muon's own step norm and LR.
- **Steps 501–507:** the training loss (on the run's own batches) is 0.002–0.005 below the baseline's.
- **Steps 508–545:** it falls behind steadily, to about +0.075, where it has stayed since ~541. Validation at 525: 4.0911, against 4.0443 for the in-harness Muon control.
- **Curvature along the way.** On the step's 64 curvature sequences, the top GN Ritz value rises from 15 to 84 (at 530), then to 52–56. The curvature along the momentum, ρ, rises from 2.6 to 11–23. The cosine with Muon's step falls from 0.5 to ≈ 0.05.
- **Candidate reading.** Muon's edge-of-stability oscillation, with energy in the stiff directions, is what keeps sharpness in check. GN's damped inverse keeps the step out of those directions, so nothing stops progressive sharpening, and the fixed-norm steps then meet the new curvature.
- The rate over 550–650 decides whether this settles into a new equilibrium or keeps losing.

**The edge invariant does not explain the stored excess at 4M (17:39).** `eos_invariant.py` gives LR × s × λ_max in each optimizer's own coordinates (30 Lanczos steps, exact GN, 256 curvature sequences). 4M β 0.9 runs; stored excess from the 16-step anneals:

| Step | Muon @0.014: invariant (excess) | PD α ½ @0.02: invariant (excess) |
|---|---|---|
| 37 | 0.175 (0.244) | 0.018 (0.287) |
| 183 | 0.098 (0.106) | 0.061 (0.085) |
| 330 | 0.071 (0.066) | 0.102 (0.047) |

- Muon's invariant falls through training at constant LR. PD's starts ~10× below Muon's and rises to Muon's level by step 330.
- The ordering of stored excess does not follow edge proximity:
  - at 37, PD is far below its edge yet stores more excess;
  - at 330, it is closer to its edge than Muon yet stores less.
- Some other property, such as the curvature along the actual step or how many directions oscillate, must set the stored loss.

## 2026-09-28 (CDT): exact-GN rate test replaced by a pre-flight; step profiles across optimizers

**GN-PD as a rate at 16M: the pre-flight stopped the training arms (09:40–10:13).** `newton_train.py` gained a Kronecker control (`--kron-control`, TS ½/½ from the step's own B and C), `--log-kron`, `--clip`, `--sharpness-every`, `--keep` and `--transport`. Every run stores a copy of its source.
- **Smoke tests** (`gnrate_smoke/`).
  - The harness reproduces the trainer's step-10 loss and gradient norm.
  - GN-PD's first step beats kron's (−0.085 vs −0.065 held-out), and its second goes uphill.
  - The transport correction is inconsistent with clipping (26× the momentum).
  - With a fresh gradient (β 0), both maps oscillate and sharpen (λ_max 614 → 2000–2700).
- **Pre-flight** (`gnrate_preflight.py`, figure `figures_gnrate/preflight.png`).
  - The exact-GN PD direction depends on the 128-sequence curvature sample: cosine 0.15 (@9) and 0.40 (@46) between disjoint samples.
  - Its damping is 100× the GN's mean eigenvalue.
  - It reaches 0.5–0.8× of the Kronecker map's best one-step decrease, and the Kronecker map is the best direction at both states.
  - After Kronecker whitening, the GN keeps a stiff band of eigenvalues 10³–10⁴× its mean.
  - Muon's step at PD's state meets 100–150× PD's curvature and raises the loss by 0.9–2.0 in one step.

**Step profiles (`step_profile_probe.py`, 34 states; figure `figures_gnrate/step_profile.png`, table `step_profile_summary.json`).** Each optimizer's own step against its own state's exact GN.
- **Universal edge.** c* of the own step is 0.45–0.94, converging to 0.55–0.65 late, at every batch size and for every optimizer.
- **Sharpness bands.** λ_max at own states: Muon 5–25, PD and SOAP∘Muon 100–600, S∘PD and TS 500–2250. The steps' energy in the top-16 GN vectors mirrors this: ~1e-3, ~1e-5, ~1e-6.
- **Period-2 oscillation at large batch.**
  - At 16M and 4M the gradient flips sign along the top GN vectors between consecutive states. At 16M, 53–86% of its energy sits there at the preconditioned states, and consecutive gradients are anti-correlated.
  - Late at 1M there is no flip, and consecutive gradients align (+0.8).
- **No per-step ranking.** The own step's one-step GN-model decrease and quality at own states do not rank the optimizers. The trajectory speedups are not visible in one-step measurements at own states.
- **Next.** A two-tap momentum pre-filter at 16M (`soaudit_prefilter16m_20260928`), testing whether oscillation leakage is what forbids a fresher momentum.

**Displacement efficiency and where the 16M lead forms (10:45).** `displacement_efficiency.py` (net |W_s2 − W_s1| over the steps' path length, from kept states; `displacement_efficiency.json`).
- **Better optimizers are less path-efficient.**
  - 16M 46→83: Muon 0.65, PD 0.59, TS 0.58, S∘PD 0.54.
  - 4M 183→330: Muon 0.33, PD 0.25.
  - 1M 500→900: Muon 0.23, SOAP∘Muon 0.21, PD 0.21, S∘PD 0.19.
  - Efficiency falls through training at every batch size, and it is much higher at 16M than at 1M: fewer, larger steps point the same way more often.
- **Over 46→83 at 16M all four drop the loss by the same amount** (0.72–0.77). S∘PD does it from a lower loss, which is its speedup at matched loss.
- **Where the lead forms.**
  - Muon leads in steps 2–5.
  - S∘PD passes PD in steps 7–15, during the spiky phase (pre-clip gradient norms 5–40, every preconditioned step clipped): −0.155 at step 15.
  - The lead eases to −0.075 by step 30, then settles at −0.09 to −0.10 to the end.
  - The preconditioned runs keep pre-clip gradient norms of 2–10 through step 30, against Muon's 0.5. That fits their sharpness bands.

**Head input: the dominant direction is the mean (13:02).** `head_center_probe.py`, final normalized hidden state, 16K tokens:

| State | Mean's share of E[hh^T] | cos(mean, top eigvec) | Uncentered top/mean, PR | Centered top/mean, PR |
|---|---|---|---|---|
| PD @9 | 0.60 | 0.999 | 314, 0.005 | 119, 0.023 |
| PD @46 | 0.17 | 0.991 | 97, 0.039 | 39, 0.086 |
| S∘PD @46 | 0.16 | 0.988 | 88, 0.046 | 37, 0.098 |

- Early, the head's input is nearly rank-one along its mean, the output-bias channel through which token frequencies are learned. Whitening the uncentered moment suppresses exactly that channel, hence the whitened heads' early deficits (+0.7 to +0.9).
- After centering, the input is still as anisotropic as the body's inputs: the centered-whitening arms (`soaudit_headcenter16m_20260928`) have structure to fix.

**Fresher momentum: less net displacement, more loss per unit displacement; Muon's step also shrinks (14:39).** `displacement_efficiency.py` on the β 0.8 16M runs (`displacement_efficiency_mom.json`), against β 0.9:

| Method | Efficiency 9→46 / 46→83, β 0.9 | Same, β 0.8 | Path 9→46, β 0.9 → β 0.8 | Net displacement 46→83, β 0.9 → β 0.8 |
|---|---|---|---|---|
| Muon | 0.70 / 0.65 | 0.53 / 0.54 | 104 → 92 | 81 → 59 |
| PD | 0.58 / 0.59 | 0.44 / 0.45 | 199 → 199 | 114 → 88 |
| S∘PD | 0.52 / 0.54 | 0.40 / 0.42 | 199 → 199 | 105 → 81 |

- With the fresher momentum every method turns more: net displacement falls by 23–35% at the same or shorter path, and loss drop per unit net displacement rises by 40–60%. The lagged momentum keeps stepping along an older direction.
- PD's and S∘PD's steps have fixed norms (√min(m,n) × shape factor), so their path is unchanged. Muon's step norm is |NS(M)|, which shrinks by 12–13% with β 0.8: the fresher momentum is more concentrated.
  - Part of Muon's non-gain may therefore be a smaller effective step. The queued Muon β 0.8 LR arms (0.014, 0.028) test this.

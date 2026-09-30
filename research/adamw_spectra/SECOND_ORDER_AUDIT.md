# Second-order structure audit: what to measure on the way to Gauss-Newton

**Status.** Design draft, 2026-09-26 ~23:30 CDT, written at the user's direction. Nothing has been run for it.
The protocol entry is in `MUON_CASE.md` ("Second-order audit: principles behind the progress and a measurement plan").

**Scope.** The audit uses our own setup and varies batch size, horizon, training phase and width. Track 3 is at most a late external check: its fixed batch, step budget and strong decay hide optimization gains that appear elsewhere, for example at larger batch.

**Goal.** Measure the structure of the per-layer Gauss-Newton (GN) matrix that practical optimizers approximate:
- which of their assumptions hold;
- which cheap estimates would move an optimizer toward GN;
- how much of the available gain can be reached at a given batch size.

## 1. The object

Take one linear map y_s = W x_s at positions s = 1..T of a sequence. Let z_t be the logits, H_t = diag(p_t) − p_t p_tᵀ the softmax Hessian of the cross-entropy loss, and J_t = ∂z_t/∂θ.

- **GN quadratic form** along a weight change D:
  - q(D) = E_seq (1/T) Σ_t Var_{p_t}(J_t D).
  - One forward-mode (or double-backward) pass gives J_t D.
  - It is exact for any D and any set of layers.
  - Computing it as Var_p avoids cancellation.
- **Sampled-label identity.**
  - Draw ŷ_t ~ p_t, backpropagate the sampled-label loss, and let e_s be the error at y_s.
  - The per-sequence gradient is ĝ = Σ_s e_s x_sᵀ.
  - E[vec ĝ vec ĝᵀ] = E Σ_{s,s'} (x_s x_s'ᵀ) ⊗ (e_s e_s'ᵀ) is the exact GN block, up to the loss normalization.
  - So the GN block is a four-index object built from activations x and backprop errors e. Every practical method keeps part of it and assumes the rest.
- **The same expression with true labels is the empirical Fisher (EF).** It is not GN plus the signal:
  - per context, E_y[r rᵀ] = H_q + (p − q)(p − q)ᵀ, where q is the data's next-token distribution;
  - EF weights tokens by ‖p − e_y‖², which favours confidently wrong and rare targets, while GN weights them by 1 − ‖p‖²;
  - true-label residuals are correlated across positions, whereas sampled ones are not. *(Corrected after review.)*
- **Minibatch-gradient Grams (Shampoo):**
  - E[G_b G_bᵀ] = Ḡ Ḡᵀ + (1/b) · Tr_in Σ, with Σ the centered per-unit covariance.
  - b must count independent units. Training batches are contiguous token slices, so neighbouring sequences can share documents; check the 1/b law with block means.
- **Two oracles**, both cheap at 77M parameters:
  - exact GN-vector products (forward-mode pass, multiply by H_t, backward pass);
  - per-sequence sampled-label gradients, from forward/backward hooks on one batched pass: ĝ_seq = Σ_s e_s x_sᵀ.

## 2. What each method assumes

| # | Assumption | What it discards | Made by | Test | If it fails, estimate and incorporate |
|---|---|---|---|---|---|
| A1 | Layers independent | cross-layer GN blocks | every practical method | coupling of each method's full update across layer pairs (q–k, v–o, up–down, adjacent residual writers/readers) | pairwise corrections or gauge balancing where the coupling is large |
| A2 | Positions independent | s ≠ s' terms | K-FAC, PD, Newton-Muon (per-token statistics); not Shampoo/SOAP (minibatch gradients keep them) | per-token vs per-sequence sampled second moments in the Kronecker frame (§3) | per-sequence statistics |
| A3 | Activations independent of errors (Kronecker) | x–e dependence | K-FAC; Shampoo/SOAP bases; PD (input side) | exact vs Kronecker curvature; error-weighted input factor E[‖e‖² x xᵀ]; token-group mixtures (sink position, high-norm tokens, frequent tokens) | error-weighted C, mixture of Kronecker terms, per-head factors |
| A4 | Separable eigenvalues in the Kronecker basis | per-pair curvature | K-FAC, Shampoo | exact diagonal vs λ_B,i λ_C,j for all pairs; off-diagonal mass | per-pair curvature (EKFAC-style) |
| A5 | Statistic source | — | GN (sampled labels); K-FAC-emp (true labels); Shampoo/SOAP (minibatch Grams); Muon (current momentum) | alignment and eigenvalue agreement of each with the sampled-label factors, vs batch and training phase | a sampled-label backward on a data subsample |
| A6 | Noise handling | per-direction signal and noise | fixed powers (PD, Shampoo ¼); Adam normalization (SOAP); polar (Muon) | curvature, signal and noise per pair (§3): noise-vs-curvature exponent κ, per-direction critical batch size, Newton-decrement density | a shrinkage rule derived from the measured noise |
| A7 | One side isotropic | output-side factor | PD, Newton-Muon | anisotropy of B; share of the reachable decrement that needs the output side | two-sided factors |
| A8 | Both sides isotropic, spectral trust region | all curvature | Muon | alignment of the momentum's singular subspaces with the GN basis; overshoot ratio η·DᵀGD/(−gᵀD) along stiff directions | — |

Cross-cutting assumptions:
- **S1 Stationarity.** Statistics stay valid for many steps. Test: drift of eigenbases and eigenvalues over 10–200 steps.
- **S2 GN ≈ Hessian along steps.** Test: Hessian-vector vs GN-vector products along actual updates.
- **S3 Quadratic model valid over a step.** Test: predicted vs actual loss change at 1×, 2× and 4× the step.
- **S4 No symmetry directions.** Test: curvature along V/O gauge and radial (scale-invariant) directions, which should be ≈ 0; this is also a tooling check. Also measure the share of each update that lies in them.
- **S5 Scale.** For normalized weights, curvature ∝ 1/‖W‖², so the step should scale with the weight norm. Test: radial and angular curvature vs weight norm over training.

## 3. The measurement frame: one coordinate system per layer

For each layer, take the eigenvectors U of B (sampled labels) and V of C. Every weight direction is a combination of pairs u_i v_jᵀ.

**One pass of hooks over N sequences gives, for all pairs (i, j):**
- the K-FAC curvature λ_B,i λ_C,j;
- the per-token (EKFAC) curvature E[(u_iᵀ e_s)² (v_jᵀ x_s)²], one matrix product of squared projections;
- the exact GN diagonal E[(u_iᵀ ĝ v_j)²], from per-sequence sampled-label gradients (includes cross-position terms and x–e dependence);
- the signal ḡ_ij and per-sequence noise σ_ij², from true-label per-sequence gradients, which give the SNR at any batch b without extra runs;
- SOAP/Adam's second moment at batch b (ḡ_ij² + σ_ij²/b), and Shampoo's Gram matrices at batch b (full matrices, compared with U and V);
- each method's actual update, computed from the same minibatch and projected into the frame.

**Checks on the frame itself:**
- top-k Lanczos eigenvectors of the exact G_W, via GN-vector products restricted to the layer: how much each concentrates on few pairs;
- a random sample of off-diagonal covariances.

**Derived quantities** (per layer kind × depth × phase × batch size):
- **Newton-decrement density** ḡ_ij²/h_ij: where the second-order gain lives.
- **Per-direction critical batch size** b*_ij = σ_ij²/ḡ_ij²: whether that gain is reachable at batch b. Momentum raises the effective batch by roughly 1/(1−β) while the gradient is stationary.
- **Reachable decrement at batch b:** the part carried by pairs with b*_ij < b. Its growth with b is the per-direction version of the GN paper's batch-size result.
- **The noise exponent κ** (σ² ∝ h^κ), the ideal per-pair step under noise, and each method's distance from it.

## 4. Spectra and interactions to report

**Spectra** (per kind × depth × phase × width):
- C, error-weighted C, B (sampled and true labels);
- exact G_W (top-k);
- per-sequence gradient covariance;
- Shampoo Grams at several b;
- momentum and weight singular values.

For each, report: power-law slope, effective rank, top/median ratio, outliers, share of the mean direction.

**Interactions:**
- activation × error: the Kronecker residual, and which tokens carry it;
- position × position: share of the cross-position terms;
- layer × layer:
  - within-block pairs and adjacent depths;
  - writers into the residual stream share an output factor, and readers share an input factor, so measure how similar they are (a possible cost saving);
- head × head: block structure of B for attention;
- signal × curvature: decrement density;
- noise × curvature: κ and b*;
- weights × data:
  - singular vectors of W vs eigenvectors of C and B;
  - W C Wᵀ vs the next layer's C;
  - B pulled back through W;
- update × curvature: overshoot ratio, and the share of each method's update in stiff directions;
- symmetry × update: share of each update in gauge and radial directions;
- time: drift of all of the above between checkpoints.

## 5. Consequences: does a structural difference matter?

1. **Step level.**
   - Build preconditioners of increasing fidelity from the measured structure: Muon, PD, two-sided Kronecker, EKFAC, error-weighted variants, and per-layer GN with a Lanczos-truncated inverse.
   - Apply them to minibatch gradients at several b.
   - Measure the line-searched one-step decrease on held-out sequences, and the GN-quadratic decrease.
   - The line search sets each method's global step, which removes the learning-rate confound.
   - **Calibration gate:** before this metric is trusted, it must reproduce the known training orderings: α ¼ > ⅜ > ⅛ > ½, and S∘PD > SOAP-Muon > PD > Muon.
2. **Short branches are not a success criterion** (user direction). Short-term gains don't imply a better optimizer, and their absence doesn't imply a worse one: the LR schedule, momentum and other state, and the trajectory all confound them. At most they are a diagnostic of local dynamics.
3. **Full runs** only for components that the measurements justify on principle. These include a critical-batch-size study of Muon, PD and S∘PD in our setup: a real optimization gain that Track 3 cannot show.

The calibration gate in item 1 may only use orderings whose learning rates were bracketed. The α ordering was not bracketed per α, so it is excluded.

## 6. Known results the measurements must explain

- **Input-side curvature.** Exact curvature is ∝ input variance along input eigendirections (exponent ≈ 1, both widths and Track 3's model).
- **Centering.** Centering C halves PD's gain. This is consistent with GN's uncentered input factor, but not for keys: see the review revision below.
- **The power α.** Among {⅛, ¼, ⅜, ½} at one LR, α ¼ is best, but output energy relative to Muon is 0.64 / 0.46 / 0.37 / 0.32 there.
- **Post-multiplication** carries about half of PD's gain.
- **Gradient vs activation basis.** SOAP's gradient-Gram basis beats the activation basis (−0.016 vs −0.012). Candidates: A3 (x–e dependence), A5 (source), or the Gram's batch-dependent effective power.
- **Output side.** S∘PD stacks on PD, and its output-side basis carries 79% of the stacking. A7 fails somewhere.
- **Batch dependence.** PD keeps 56% of its gain at half batch (weak decay). In Track 3's model its gain is larger at 1M than at 0.5M (confounded with decay exposure).

## 7. Where, tooling and cost

- **Runs.** Our 77M setup (width 512; 768 for trends), weak decay. Trajectories: Muon@0.007, PD α ¼@0.01, SOAP-Muon, S∘PD. Save checkpoints with optimizer state at about steps 50, 200, 500, 900, 1300 and the end. Existing runs keep only rolling checkpoints. Cost: about 4 runs × 50 min on 4 GPUs.
- **Probe library** (one GPU):
  - forward/backward hooks for x and e, with true and sampled labels;
  - per-sequence gradients and their projections into the frame;
  - GN-vector products in FP32, using the math attention kernel or double-backward, with the logit quadratic computed as Var_p;
  - Lanczos, Hessian-vector products, gauge and radial directions.
  - Cost: about 10–30 min per checkpoint, with N up to 8192 sequences (4M tokens) for the noise statistics.
- **Tooling checks first**, on an existing final checkpoint:
  - GN curvature ≈ 0 along exact V/O gauge directions;
  - the sampled-label second moment matches exact GN-vector products on random directions;
  - per-sequence gradients sum to the batch gradient.

## 8. Readings that change what we build (draft, to be fixed before the measurements)

- **A2:** if per-token statistics capture ≥ 90% of the exact diagonal, per-token statistics suffice.
- **A3/A4:** if EKFAC or error-weighted C cuts the log-error to the exact diagonal by ≥ 2×, incorporate it.
- **A7:** if B's top/median ratio is ≥ 100 and the output side carries ≥ 30% of the reachable decrement, go two-sided.
- **A6:**
  - κ ≈ 1: noise does not explain α ¼; test stiff-direction capping (PD on the top-k input directions only).
  - κ ≤ ½: build the noise-derived shrinkage.
- **b\* spectrum:** if the reachable decrement at 1M is small compared with 4M+, second-order gains are a large-batch phenomenon here, and batch-scaling experiments come first.
- **A1:** if the block-diagonal part is ≥ 90% of each update's quadratic form, drop cross-layer terms.
- **S4:** if ≥ 10% of an update lies in gauge or radial directions, test gauge handling.

## 9. Revision after the independent design review (2026-09-26 ~23:55 CDT)

The review confirmed the §1 identity numerically (error 4e-16 in FP64; Monte Carlo within 2%). Its corrections are applied above, and these additions are now part of the plan.

**Attention's cross-position terms (the main gap).**
- Softmax is invariant to shifting all of a query's scores, so without QK-norm a head's key errors sum to zero over positions (1e-15 in a toy model). W_k then sees only within-sequence-centered inputs, and the shared-mean input direction is exactly flat for keys. QK-norm breaks this partly (14% residual in the toy).
- In the toy, the ratio of exact to per-token curvature along a dominant shared direction was:
  - keys: 0.03–0.5;
  - values: 1.3–3.4;
  - queries: 1.05–1.4;
  - bulk directions: ≈ 1.
- So per-token statistics (PD's C included) misjudge the curvature of the very direction PD reshapes most, differently per kind. This fits the lower key exponent we measured (0.74–0.78).
- It is also a fourth candidate for why the gradient-Gram basis beats the activation basis.
- The A2 reading becomes two-sided, per kind, and weighted by the decrement.

**Exact symmetries (S4).**
- Per-head V/O gauge (GL(64));
- per-head q/k row scale (QK-norm);
- norm gain against the reading matrices' columns (ln1 with q, k and v jointly; ln2 with up);
- the joint residual-stream scale (embeddings and every o and down).

A single v/o/up/down matrix's scale is not a symmetry. In the toy, K-FAC assigns the exactly flat q/k scale directions 20–90% of a random direction's curvature.

**The frame reports; it does not compute.**
- Each method's q(D), overshoot and decrease come from forward-mode passes on its actual update D.
- Report bins in (λ_B, λ_C), not single pairs, because bulk eigenvectors are near-degenerate.
- Estimate U and V on held-out sequences, include position 0, and debias the signal (ḡ² − σ²/N).
- Check the frame with a Hutchinson estimate of ‖G‖_F² vs Σ h², targeted probes, and a frame-free gᵀ(G + λI)⁻¹g by conjugate gradient with λ swept. The Newton decrement sits in the bottom bins, where errors are largest.

**Sample sizes.**
- Curvature per pair has relative error ≈ √(2/N).
- The signal is the bottleneck: the unbiased ḡ² has relative error ≈ √(4r + 2r²) with r = b*/N. Keeping it ≤ 0.5 needs N ≈ 16 b*, about 33k sequences to resolve b* at a 1M-token batch. Bins of ~10³ pairs reach b* ≈ 10N.
- EMA momentum raises the SNR by (1 + β)/(1 − β) ≈ 39, and only where the signal is stationary over about 20 steps.

**Added measurements.**
- **Embeddings and head.** They hold 51.5M of the 77M parameters. The head's GN is exactly per-token, E[x xᵀ ⊗ H_t], with a Zipf-shaped output factor: the cleanest test of A3/A5, and it gives the body's share of the decrement.
- **Trajectory effects.**
  - Curvature settles into equilibrium with the optimizer (edge of stability), so measure preconditioned sharpness against the stability threshold.
  - Evaluate every method's update on every method's checkpoints (update rule × trajectory).
- **Damping vs power.** Fit both families to the Wiener step profile from (h, ḡ, σ).
- **Heavy tails.** Per-bin kurtosis, and the shares of position 0 and high-norm tokens.

**Feasibility.**
- Forward-mode AD needs the math attention kernel (explicit attention in `gn_probe.py`).
- Per-sequence gradients are never stored; FP64 projected moments for all 48 matrices take ~0.6 GB.
- Capture is FP32 with TF32 off.
- Lanczos costs about 1500 GN-vector products per matrix, so limit it to ~12 matrices.

**Priority order.**
1. Exact one-sided GN marginals (Tr_out G = T·E[ĝᵀĝ] and Tr_in G) vs C, error-weighted C, EF and minibatch Grams, per kind, on existing final checkpoints (minutes).
   - If they differ for k or v: PD with per-kind GN-marginal factors.
   - If not: C is the right input factor.
2. Binned κ and b* with N ≥ 16–32k sequences, sampled in contiguous blocks.
3. Exact update quadratics and preconditioned sharpness on the trajectory checkpoints.
4. Per-layer damped-GN headroom on ~12 matrices at b ∈ {0.25, 1, 4}M tokens.

**Tooling status (2026-09-26 ~23:55 CDT).**
- `gn_probe.py` provides: hooks for x and e; true-label and model-sampled backward passes; per-sequence gradients; exact q(D) by forward mode through explicit attention; all exact symmetry directions; the per-layer frame; one-sided marginals with a report.
- `test_gn_probe.py`: 7 CPU tests pass on a tiny frontier-norm model:
  - explicit attention = SDPA;
  - per-sequence gradients sum to the batch gradient;
  - the sampled second moment matches exact q(D) within 12% with 1200 samples;
  - symmetry curvature < 1e-6 of random, while a single writer's scale is curved;
  - frame and marginal trace identities.
- Full-size checks and measurement 1 on the final Muon and PD checkpoints: `logs/muon_spectra/second_order_audit_20260926/check_tools.py`.

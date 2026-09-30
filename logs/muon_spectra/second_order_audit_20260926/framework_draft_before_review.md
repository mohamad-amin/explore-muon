# Second-order framework: principles behind the progress, and an audit (DRAFT for independent review)

Project: `/share/data/dl-theory/amin/projects/explore_muon` (protocol `research/adamw_spectra/MUON_CASE.md`,
Track 3 notes `track3/README.md`, summary `logs/muon_spectra/improve_w4_20260925/FINDINGS.md`).
Gauss-Newton paper text: scratchpad `gn_paper.txt` (Abreu, Vyas, Kakade, Morwani, ICLR 2026).

## Evidence already in hand (77M frontier-norm GPT, width 512, 8 layers, 1M-token batch, 1.5B tokens, unless noted)
- PD (partial data-norm Muon): polar(M C^-a) C^-a, C = uncentered E[x x^T] of the layer input, RMS-matched to Muon.
  vs tuned Muon: -0.019 (n=6), width 768 -0.014, 2x horizon -0.015. alpha sweep: 1/8 -0.009, 1/4 -0.017, 3/8 -0.012, 1/2 +0.007.
  Damped alpha=1/2 (damping 0.1-0.3) ~ 0. Inside-only (alpha 1/2 inside polar) -0.006; outside-only -0.008.
  Centered C halves the gain. Magnitude-matched Muon ~ 0 (gain is geometric).
- SOAP-Muon core -0.024; S∘PD (SOAP-Muon core run in PD-whitened coordinates) -0.031 (stacks; output-side SOAP basis
  carries ~79% of the stacking). Input-side-only SOAP ~ -0.015; SOAP in the activation eigenbasis -0.012 (worse than
  the gradient-Gram basis); SOAP in the standard basis +0.003.
- Exact-HVP curvature along input eigendirections is proportional to input variance (median exponent ~1.0) in our model
  (widths 512, 768) and in Track 3's model. C is extremely anisotropic (Track 3: effective rank 18-36 of 768).
- Track 3 (124M, 0.5M batch, 3250 steps, strong decoupled decay, 70% cooldown): plain PD -0.002 (~50 steps).
  Mechanism: with total decay exposure E = integral(lr*wd) >~ 2 the weights equilibrate per input direction and the
  relative step along each direction becomes sqrt(2*lr*wd) independent of the update's per-direction size, undoing
  PD's reallocation (weights 2-3x depleted along the top input directions). Decay in PD's geometry
  (W <- W - lr*wd*W R^2/mean eig R^2) restores it; Muon + that decay is worse (+0.010). alpha 1/8 + geometry decay
  reaches 3.28 at 3150 steps vs #36's 3250 (n=6). Track 3's own model at 1M batch: plain PD -0.011.
- GN paper: full GN (inner Muon on the linearized model, N = batch/131k inner steps per outer step, line search)
  >> SOAP only at batch >= 4-12M tokens for 150M; ~ SOAP at 1.2M. But GN is capped by its inner solver (Muon at 131k),
  and SOAP at 1.2M is already within ~0.035 of that cap, so small-batch curvature headroom is not measured there.
  Layerwise GN 78 vs full 54 outer steps at huge batch; prox-linear ~ GN.

## Claimed big picture
Every hidden-matrix update rule on the Track 3 leaderboard can be read as a partial approximation to one object:
a stochastic, regularized, per-layer Gauss-Newton step. It has four separable parts:
1. Basis / rotation: which directions. Per-layer GN block for y = W x is
   G_W = E_t[ (sum_s x_s (x) A_ts)^T H_t (sum_s' x_s' (x) A_ts') ]; Kronecker approx C (x) B,
   C = E[x x^T] (input), B = E[e e^T] with e = backprop error at the layer output from model-sampled labels (GN factor);
   true-label e gives the empirical-Fisher factor.
2. Magnitude per direction: curvature^-1 shrunk by per-direction signal-to-noise. One-step quadratic model with
   gradient noise gives eta_i = (1/h_i) * s_i / (s_i + n_i/b) (Wiener-Newton). Fixed powers (Shampoo 1/4 per side,
   PD alpha) and Adam-type normalization (SOAP, NorMuon) are crude surrogates; full GN is the b -> infinity limit.
3. Scale / regularizer: weight decay acts as damping; with (partial) scale invariance the weight norm sets the
   effective curvature/LR. Under a preconditioner P, decoupled decay implies a penalty tr(W P^-1 W^T) that is not
   isotropic; the Newton-consistent regularizer is decay through the same geometry (our geometry decay).
   Refinement: an update ending in a polar has isotropic per-input-direction magnitudes (tall/square W), so its gain
   is pure rotation and is untouched by isotropic decay equilibria; rules that reallocate magnitude after the polar
   (PD's post-multiplication, any GN-like 1/h scaling) are undone at equilibrium unless the regularizer matches.
   This would explain why polar-terminated gradient-Gram methods (SOAP-Muon) transferred to Track 3, while
   activation methods looked weak there (Newton-Muon is inside-only = rotation only; plain PD lost its magnitude part).
4. Cross-layer, higher-order loss terms, symmetries (gauge, scale invariance): second-order effects at moderate batch
   (GN paper: layerwise 1.4x at huge batch; prox-linear ~ GN).
Tentative leaderboard mapping (to be checked against the records):
- curvature via minibatch-gradient Grams: Shampoo, SOAP(-Muon), KL-SOAP, SinkSOAP, PSGD, one-sided Shampoo, PMuon;
  via activations: Newton-Muon, PD.
- magnitude / trust: NorMuon (row/col variance after polar), Adam-in-eigenbasis, spectral-power shaping over training
  (DynMuon, Contra-/Soft-/Tempered-polar), momentum schedules, EMA/lookahead/extrapolation/tail averaging.
- scale: hyperball (MuonH/AdamH/NorMuonH), u/w floors, radial brake, Muown, row floors, cautious WD, WD tuning,
  our geometry decay.
- symmetry / cross-layer: Circuit-Muon (V/O gauge), MuLoCo outer loop.

## Hypotheses
H1 statistic source: per-token statistics (x_t, e_t, and their joint weighting) estimate the GN factors at any batch
   size; minibatch-gradient Grams G^T G = sum_{t,t'} (e_t . e_t') x_t x_t'^T mix per-token curvature (the t = t'
   "noise" part, relative weight ~ 1/b) with the mean-gradient outer product (signal). Above the gradient noise scale
   the Grams become signal-dominated and lose curvature content (consistent with Morwani et al. 2024 needing
   batch-size-1 gradients for Shampoo ~ Kronecker GN, and with SOAP plateauing at large batch in the GN paper).
   Prediction: activation-based preconditioning keeps or grows its gain with batch; gradient-Gram eigenbases
   decorrelate from the GN factors as b grows.
H2 weighting: the GN input marginal is E[b_t x_t x_t^T] (b_t = per-token output curvature). Plain C ignores the
   weighting; gradient Grams include it through ||e_t||^2 (may explain the activation-basis SOAP deficit).
   Prediction: error-weighted C_e = E[||e_t||^2 x_t x_t^T] predicts exact curvature better than C and improves PD.
H3 Kronecker: exact GN curvature along u v^T ~ (u^T B u)(v^T C v); attention q/k may deviate (cross-position terms).
H4 noise sets the power: the best fixed alpha rises with batch toward 1/2 (full input factor); alpha 1/4 at 1M is a
   noise effect; the Wiener-Newton profile from measured (h, s, n) predicts alpha*(b) (and beta*(b) for the output side).
   Existing hint: alpha* 1/4 at 1M vs 1/8 in the 0.5M Track 3 recipe (confounded with decay).
H5 rotation vs magnitude under decay: see part 3 above.
H6 not necessary at our batch: cross-layer curvature, higher-order loss terms, exact inverses.

## Audit (read-only diagnostics on checkpoints; early/mid/late; our 77M Muon and PD runs; Track 3 finals)
A1 curvature map: exact GN quadratic form via a JVP through the model (then the CE logit Hessian) along rank-1
   directions u_i v_j^T, with u from eigvecs of B (sampled labels) / B_emp (true labels), v from eigvecs of C / C_e;
   fit log h = beta log lambda_B(u) + alpha log lambda_C(v) + c, report exponents and R^2 per kind, depth, phase.
A2 signal / noise: project per-minibatch gradients at b in {0.06, 0.25, 0.5, 1, 2, 4}M tokens on the same
   directions: s_ij, n_ij, SNR(b).
A3 predicted magnitude profile and alpha*(b), beta*(b) from (h, s, n), with momentum's effective averaging; compare
   with the measured alpha* (1/4 at 1M).
A4 ladder: directions from Muon, PD(alpha), PD with C_e, two-sided (alpha, beta), K-FAC damped inverse, per-layer
   exact GN (CG on GN-vector products), full GN (CG). Report noiseless quadratic efficiency
   rho = <g,D>^2 / (D^T G D * g^T G^-1 g) and cross-fitted one-step held-out decrease at batch b (line search).
A5 eigenbasis drift of C, B and gradient Grams over 10-100 steps (refresh cost).
A6 per-input-direction weight profile of SOAP-Muon under decoupled decay (checks H5).

## Training tests (only after A1-A4 point somewhere)
T1 alpha x batch sweep at weak decay vs the A3 prediction. T2 PD with C_e. T3 two-sided (beta from A1-A3).
T4 noise-aware per-direction magnitudes in the Kronecker eigenbasis, with geometry-consistent decay.

## Draft decision rules
- If A4's cross-fitted per-layer-GN advantage over PD / S∘PD at b <= 1M is small (< 20% of PD's gain over Muon),
  curvature headroom at our batch is exhausted: shift effort to noise / scale.
- A1 output exponent beta ~ 0: output side unnecessary at our batch; ~ 1: run T3.
- C_e better than C in A1: run T2.
- A3 alpha*(1M) ~ 1/4: adopt the SNR-derived rule; T1 checks the batch trend.

## Competing explanations
- Track 3 progress is mostly noise / schedule handling, not curvature (A2 SNR spectra quantify).
- PD's gain is outlier / feature-learning (mean direction, spikes) rather than curvature (A1, A4 distinguish).
- alpha 1/4 reflects estimation error in C (damping, eigenvalue noise), not gradient noise; that would not depend on
  batch (T1 distinguishes).

## Cost
A1, A2, A5: minutes per checkpoint on one GPU. A4 with CG: ~1 GPU-hour per checkpoint. T1: ~16-20 runs x 30-60 min
on 4 GPUs. Code: JVP probe on the uncompiled model (torch.func), backward hooks for per-token errors, a sampled-label
backward.

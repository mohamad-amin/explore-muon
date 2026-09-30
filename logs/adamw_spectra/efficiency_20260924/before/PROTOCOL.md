# AdamW update spectra at 77M

Status: implementation and qualification; no scientific training result yet.

Cadence amendment, 2026-09-24: before any scientific run, the user requested
spectra every approximately 25 steps instead of every step. Use every 25
optimizer updates plus the first and final updates. This supersedes the
initial every-update measurement plan; existing qualification records remain
unchanged. Training-loss logging remains every update.

## Decision argument

The user asks whether the layer-dependent spectral dynamics reported in
*Spectral Scaling Laws of Muon* also appear in ordinary AdamW's adaptive
updates. The observable is the update matrix, not just Adam's first moment.
This is an exploratory description of learning dynamics; it does not assume
that spectral concentration is a training limitation. Adam's coordinatewise
second-moment normalization can change singular vectors as well as values.
Ordinary dynamics could produce stabilization and depth differences under
both optimizers, or could produce a substantially different update spectrum.
Either outcome informs whether the paper's observations transfer to AdamW;
neither establishes an optimizer speedup or a head bottleneck.

Use one approximately 77M GPT-style model on FineWeb, ordinary next-token
cross-entropy, and standard PyTorch AdamW throughout. Match the paper's stated
depth, width, heads, context, token-to-parameter ratio, depth sampling and
descending-rank quantiles. Record exact singular values of the normalized
adaptive update every 25 optimizer steps, plus first and final. Preserve update norms and validation
loss so normalization cannot hide scale or a failed training run. Select a
learning rate using a fixed small validation-loss sweep, without selecting on
spectral appearance. A second fresh seed can assess reproducibility after the
rate is fixed. A causal optimizer comparison would require a matched Muon
run and its post-NS update measurements; the paper's pre-NS momentum plots
are descriptive references, not that control.

Implementation qualification is bounded to CPU tests and a short smoke run;
do not launch a full training cohort as part of an implementation check.
The prepared scientific budget is three 300-update rate pilots followed by
one 20-tokens/parameter run, with a second seed explicitly separate. No
automatic boundary expansion or additional hyperparameter rescue is allowed.

## Source and reconstruction limits

- Paper: https://arxiv.org/pdf/2606.04058v2 (Table 1, sections 3–4, appendix A.1).
- Paper specifies 8 layers, width 512, 8 heads, context 512, FineWeb,
  approximately 77M parameters, 20 training tokens/parameter, Adam auxiliary
  betas (0.9, 0.95), weight decay 0.01, and a constant rate followed by
  linear decay over the last 10%.
- Its citation points to modded-nanogpt, without an experiment-specific
  revision. Batch size, precise architecture/tokenizer/initialization and
  Adam epsilon are not fully specified. An exact replication is not claimed.
- Reconstruction: conventional pre-LayerNorm GPT, learned positions, GELU,
  4x MLP, separate Q/K/V projections, untied input/output embeddings,
  vocabulary 50,304 (padded GPT-2 vocabulary), no dropout or logit cap.
  Normal initialization std=0.02; attention-output and MLP-down std scaled
  by 1/sqrt(2*depth). These are explicit implementation choices, not claims
  about the authors' model. The paper's abbreviated initialization-scaling
  statement does not specify a complete initialization recipe.
- FineWeb data are the repository's GPT-2-tokenized `fineweb10B` shards.
  Training and held-out validation shards must be disjoint.

## Optimizer and budget

- FP32 parameters and optimizer states; BF16 autocast on CUDA, FP32 on CPU.
- Unmodified `torch.optim.AdamW`, one rate for all parameters, betas
  (0.9, 0.95), epsilon 1e-8, decoupled weight decay 0.01 on all parameters.
- Provisional peak LR 6e-4; candidates 3e-4, 6e-4, 1.2e-3.
  The default is a conventional GPT starting point, not a measured optimum.
- 50 updates of linear warmup (an explicit AdamW adaptation), then constant
  LR, then linear cooldown during the final 10% of the complete token budget.
  The last update uses a positive LR; zero is reached after the budget.
- Global batch 1,048,576 tokens, accumulated from configurable microbatches.
  This is a reconstruction inference: Figure 3's full-budget 77M run ends
  around 1,500 updates, so 20 x 77M / 1,500 is about 1.03M tokens/update,
  consistent with 2^20. It is not an explicitly reported or uniquely recovered
  value. Section 3's plots of the *first* 1,500 steps alone would not establish
  total training length. Untied vocabulary matrices similarly explain the
  approximately 77M count; tying them would give approximately 51M.
- Budget = 20 x the actual parameter count, rounded up to a whole sequence;
  shorten the final batch rather than silently adding a full batch.
- Clip global gradient norm at 1.0; record unclipped norms/clipping incidence.
  This is an explicit standard AdamW stabilization choice.
- Pilots use the same initialization seed, data order, batch size, and the
  first 300 updates of the full-run schedule (no accelerated pilot cooldown).
  Choose lowest mean validation NLL across the last three scheduled pilot
  evaluations. Report if the best candidate is a boundary; do not pretend the
  rate is established as optimal. Freeze selection before spectral analysis.
  Pilot seed is 260923; the first scientific seed is 260924. Evaluate a fixed
  1,048,576-token held-out bank every 50 updates and select by the mean at
  updates 200, 250, 300. An exact tie chooses the lower rate. Failed/nonfinite
  pilots are ineligible; if none qualify, no selected configuration is written.
  Pilot spectra are disabled so selection cannot be based on their appearance.
  This qualifies early learning, not final-budget LR optimality. Report all
  three loss curves and clipping incidence. The pilot tokens are extra tuning
  compute and are not part of the scientific run's 20N training-token count.

## Measurement contract

After step t, use the moments stored by PyTorch to reconstruct

`U_t = (m_t / (1-beta1**t)) / (sqrt(v_t / (1-beta2**t)) + eps)`.

This is the direction applied on that same step, after any gradient clipping,
before multiplying by -LR. It excludes weight decay. FP32 parameter-delta
rounding is not folded into this mathematical adaptive direction.

- Measure blocks 2, 4, 6, 8 (one-based) and Q, K, V, O, MLP up/down: 24
  matrices. The vocabulary head is outside this paper-matched panel.
- Exact SVD, no randomized/rank-truncated approximation; normalize by the
  original Frobenius norm. Keep all singular values so histograms can be
  replotted without saving the much larger update matrices.
- Descending-rank quantile q means `sigma[ceil(q*r)-1]`, for q in
  {0.1, 0.25, 0.5, 0.75, 0.9}. This is not numpy's ascending quantile.
- Record step, tokens, LR, raw update norm, adaptive step norm, weight-decay
  step norm, largest singular value, top-two ratio, top-one squared-energy
  share, and stable rank. Zero matrices are explicitly flagged, with
  undefined normalized diagnostics represented as null, not fabricated data.
- Removing the top singular value for the bulk histogram does not renormalize
  the remaining values. Use common bins for comparisons.
- Record training NLL, fixed held-out validation NLL, timings and checkpoint
  metadata. Default spectral sampling is every 25 updates plus first/final;
  explicitly record any override. This is intentionally sparser than the
  paper's cadence and may miss very brief transients.
- Plot all quantiles over steps/tokens, 24-panel median trajectories, full and
  top-removed histograms, and loss. Summarize steps 1300–1500 only over
  observed steps and report count/coverage and cooldown overlap. Do not infer
  stabilization from averaging alone, or a scaling law from one model size.
  Also summarize the predeclared 1,100–1,300 window, which precedes cooldown
  at the default operating point, and mark warmup/cooldown boundaries.

## Independent design review

Fresh-context peer review supported this bounded descriptive implementation.
Accepted recommendations: a separate constant-rate summary window, fixed
pilot selection evaluations/tie/failure rules, and multi-step checks against
actual parameter deltas using pre-step weights for the decay term. The peer
questioned batch inference from first-1,500-step plots; the actual inference
uses Figure 3's full-budget endpoint, and remains explicitly conditional.
Ordinary Adam dynamics may already produce approximate stabilization; its
first update is approximately elementwise gradient sign when epsilon is
negligible. A plateau alone therefore identifies neither an optimizer-specific
mechanism nor a training limitation. No new training result is implied.

## Qualification

Tests must verify the reconstructed update against PyTorch's applied delta
after subtracting weight decay, quantile indexing and normalization on known
spectra, non-mutation of optimizer state, data boundaries, fixed validation,
budget/schedule semantics, and checkpoint resume against uninterrupted
training. A tiny end-to-end run must produce readable metrics, spectra,
checkpoint and plots. Full-sized construction verifies the parameter count
and 24 selected matrices. No sealed initialization-study data are used.

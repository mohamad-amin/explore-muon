# Fixed-width Muon depth comparison

User authorization, 2026-09-25 UTC: stop the unlaunched 16-layer AdamW cell;
run 8-layer Muon on four allocated g20 GPUs, submit 12-layer Muon on four
48 GB-or-larger GPUs and 20-layer Muon on eight 96 GB GPUs. The completed
AdamW baselines are preserved. The previously requested 20-layer AdamW job
remains the comparator; this instruction does not cancel that queued run.

## Decision argument

The existing AdamW runs describe adaptive-update geometry, whereas Figure 5
of *Spectral Scaling Laws of Muon* describes momentum before orthogonalization.
An apparent difference could come from measuring different mathematical
objects, ordinary depth-dependent learning, or the optimizer. No spectral
limitation or loss improvement is assumed. The authorized Muon cohort supplies
both its pre-NS momentum and actual post-NS update, allowing a like-object
update comparison with AdamW and a momentum comparison with the paper.

Keep the existing GPT architecture, initialization recipe, seed, data order,
width, batch and token horizon. Use the paper's spectral-study Muon LR 0.01
and auxiliary AdamW LR 0.002 at all depths. Auxiliary rate and optimizer are
part of this recipe comparison: this is not an isolated causal test of NS.
Ordinary dynamics may produce concentrated momentum but flatter updates
without a corresponding loss advantage. That result would discourage treating
momentum concentration alone as evidence of a training bottleneck. Different
post-NS trajectories or validation NLL motivate inspection, not a scaling law
or a speedup claim from one seed. No per-depth tuning or additional seeds.

Each job has tiny distributed and five-update full-size qualification before
one 1,469-update run, bounded by 7h45m; public allocations request 8h. Stop
on nonfinite loss, inconsistent replicas, failed spectral energy checks or
insufficient forecast time. There is no automatic hyperparameter rescue.
The five-step engineering qualification sets warmup to one step to test the
actual peak rates. The scientific run starts fresh with the unchanged 50-step
warmup. This is a numerical stress check, not a selected scientific pilot.

## Frozen recipe and source limits

- Paper: https://arxiv.org/pdf/2606.04058v2, Algorithm 1, section 3, A.1, A.2.
- Muon on six body weight matrices per block. AdamW on token/position
  embeddings, LM head, normalization parameters and biases; betas (0.9,0.95),
  epsilon 1e-8. Both use weight decay 0.01.
- Plain momentum M <- 0.95 M + gradient, no Nesterov, following Algorithm 1.
  The numerical momentum coefficient is an explicit reconstruction choice.
- Five distinct NS polynomials from equation 2, FP32 input normalization
  (norm clamped at 1e-7), BF16 CUDA products, FP32 output and state.
  Standard aspect multiplier sqrt(max(1, rows/columns)); this convention is
  explicit and not claimed to recover the paper's unspecified implementation.
- Preserve the existing baseline initialization; do not apply a new
  paper-specific initialization scaling. This is a matched local recipe,
  not an exact reproduction of the paper's unavailable experiment revision.
- Width 512, heads 8, context 512; depths 8/12/20; seed 260924; global batch
  1,048,576 tokens; total 1,539,870,720; final batch 561,152. All 1,469 steps.
- Preserve warmup 50 and final-10%-token linear cooldown, global clipping 1.
  These are the existing baseline adaptations, not fully paper-stated choices.
- At steps 1,25,50,...,1450,1469, direct FP32 SVD with FP64 Frobenius norm of
  24 matrices (Q/K/V/O/up/down at four relative depths), for BOTH momentum
  and the captured FP32 post-NS direction actually passed to the parameter
  update. Exclude LR/decay from normalized spectra; log their norms separately.
  The latter direction includes shape scaling and BF16 NS rounding. It is not
  reconstructed by rerunning NS or by substituting exact polar singular values.
- Momentum and update artifacts have separate names and figure labels. This
  does not expand to all parameters or measure the head/embeddings.

Independent scientific discussion: [MUON_REVIEW.md](MUON_REVIEW.md).

## Distributed NS qualification amendment

The first tiny CUDA check stopped before science: rank0's last MLP-down
weight differed by 1.96e-5 from peers, despite identical gradients and all
optimizer states. This localizes the problem to NS direction computation or
application; the particular compiler/backend rounding cause is unproven.
The failed run and per-rank tensors are preserved under
`logs/muon_spectra/depth8_w512_20260925` and
`logs/muon_spectra/qualification_20260925/cuda_diagnostic`.

Compute each matrix's NS result on one owner, round-robin within each shape
bank, and sum disjoint FP32 slices with one all-reduce before all replicas
apply the direction. Nonowner slices start at zero every step. This removes
redundant NS products and enforces common applied directions. The added
communication is part of measured runtime. Momentum remains replicated and
measurements capture directions after communication. Qualification checks
that communicated owner slices equal their computed originals, in addition
to the unchanged strict replica audit. Scientific settings and NS equations
are unchanged. Fresh peer discussion supported this bounded execution repair.

An additional archived-input comparison found the compiled NS direction
differed from the literal eager BF16 polynomial by 3.407% in Frobenius norm
for the 32x32 tiny matrices, exceeding the prechosen 2% check. Preserve this
failed diagnostic in `qualification_20260925/ns_reference.log`. Use the
literal eager batched NS operation sequence for the final implementation;
model compilation remains enabled. This fixes the numerical convention
without loosening that check, changing coefficients or tuning an LR. Shared
directions and batched products retain the principal NS execution savings.

## Four-GPU depth20 scheduling amendment

2026-09-25 01:55 UTC: user requested one depth20 job on four GPUs. Replace the
unstarted Muon job2620619 with2620623 requesting four96 GB GPUs. Leave the
AdamW20 comparator on eight GPUs. Frozen training source/config are unchanged;
the generic controller receives `--world-size 4`, with the same global batch,
token horizon, schedule, measurements and qualification gates. Rank-dependent
floating-point rounding may differ; mathematical global gradient weighting and
optimizer rules remain the same. No new scientific cell or tuning is added.

## Spike-origin diagnostic at saved checkpoints (2026-09-25)

User authorization, 2026-09-25 UTC: run the proposed spike diagnostic on four
available GPUs. The g20 allocation was busy with another project's job, so the
diagnostic uses the four idle L40S GPUs of allocation 2567578 (priv-g14).
No training, tuning or checkpoint modification is involved.

Decision argument. Observation: every tracked pre-NS momentum matrix has one
dominant singular pair (local top-mode energy 0.10–0.87; V is largest at every
depth). The paper's steep final-layer exponents are equivalent to that pair
absorbing Frobenius mass. Proposed mechanisms: the token-mean product (the
coherent error ē times the input mean x̄; the gradient a bias would receive),
or a position-0 attention sink. Simplest competing explanation: a generic
persistent population-gradient direction that is neither. A second question
is whether the momentum's bulk directions carry true-gradient signal at all.
If they do not, flattening them (or lifting them with better NS) adds little.

Comparison. At each final checkpoint, compute fresh gradients on never-trained
tokens from the training stream (offset 2,000,000,000; training used
[0, 1,539,870,721)). The same tokens are used for all four checkpoints
(Muon/AdamW × 8/12 layers). The reference pair for each body matrix comes
from its saved optimizer state: Muon momentum, or AdamW first moment. Record:
- whether the fresh mean gradient reproduces that pair;
- the share of u1ᵀḠv1 from the token-mean product (ē x̄ᵀ), from position-0
  tokens, and from the top 1% of tokens by |contribution|;
- sign consistency of the per-token contributions;
- coordinate concentration of u1/v1;
- Q/K/V input-pair and O/down output-pair alignment;
- attention mass on position 0, and residual-stream channel outliers.
Also record the cross-fitted per-band descent ⟨Ḡ, Σ_band u_i v_iᵀ⟩ against
its in-sample value (1−μ)Σσ_i, with standard errors across batches. Finally,
record the gradient-noise edge at the 1,048,576-token training batch
(half-difference of paired 524,288-token gradients), rescaled to momentum
noise by sqrt((1−μ)/(1+μ)) for Muon, and ρ₁ and median versus token count.

Predeclared readings, per matrix:
- mean route if the token-mean product explains ≥ 0.5 of u1ᵀḠv1;
- sink route if position-0 tokens explain ≥ 0.5;
- otherwise "distributed".

A band is signal-bearing if its cross-fitted descent is ≥ 0.5 of in-sample
and z ≥ 3; noise-dominated if < 0.2 or z < 2.

Instrument gates:
- loaded-weight validation NLL within 0.01 of the recorded final value;
- per-matrix hook reconstruction of the weight gradient within 5e-3 relative
  Frobenius error.

Next decisions:
- mean route in final O/down/V → a later centering/bias-path training test;
- sink route in V/K → a sink-bias variant;
- distributed → keep deflation as the pragmatic fix;
- noise-dominated bulk → favours a noise-edge threshold;
- signal-bearing bulk → favours lifting the bulk.
Any training follow-up needs separate authorization.

Limits. One seed. One end-of-cooldown checkpoint per run. Small scale, where
the paper's final-layer steepening has not appeared. AdamW's first moment
uses β1 = 0.9 rather than 0.95. Results speak to the origin of the spike that
exists at every depth, not to the scale trend.

Cost. At most about 34M fresh tokens per checkpoint in eager BF16, one GPU
per checkpoint, well under an hour including CPU and tiny-GPU qualification.
Outputs go in a new `logs/muon_spectra/spike_diagnostics_20260925/` with
frozen source and hashes.

Qualification amendment (before any science run). The instrument gates
passed for all four checkpoints: validation NLL within 3e-5 of the recorded
values, hook reconstruction ≤ 3.5e-3. The AdamW checkpoints first failed at
load because their configs predate the `optimizer` key. They now load through
`load_config`, which fills defaults; the failed logs are kept in `gates/`.

A 2-batch probe also tripped an implementation-only consistency check (hook
versus gradient projection of u1ᵀḠv1). That check was not one of the protocol
gates. It divided by u1ᵀḠv1, which is near zero wherever the fresh gradient
does not contain the saved state's top pair. The check is now normalized by
the absolute per-token contribution scale. The probe outputs are kept.

The probe also showed that, for V, the largest noise singular value lies along
the spike's own input direction. The noise-edge summary therefore adds a
second-singular-value ("bulk") edge.

Applicability rule, fixed before science: the mean/sink/distributed readings
apply only to matrices whose fresh gradient reproduces the reference pair,
i.e. cross-fitted u1ᵀḠv1 ≥ 0.5 of the in-sample σ1 with z ≥ 3. Other
matrices are reported as "spike not reproduced by the fresh gradient". The
predeclared thresholds are unchanged.

## Frontier-norm architecture variant (2026-09-25)

User authorization, 2026-09-25 UTC: adopt no biases, RMSNorm with learnable
gains, and Q/K norms with gains. Run on the four L40S GPUs of allocation
2567578 (priv-g14).

**Decision argument.**
- **Observation.** The local GPT (Linear biases, LayerNorm with bias, no Q/K
  norm) differs from current open frontier checkpoints. In the 15-lab scan,
  14 have no linear biases, all norms carry gains and no biases, and about
  half normalize queries and keys with gains
  ([scan](../../logs/muon_spectra/spike_diagnostics_20260925/frontier_biases/README.md)).
- **Why it matters.** The spike diagnostic found the dominant momentum spike
  is the token-mean product, i.e. the bias-gradient route. Biases give that
  mean error its own parameter, so spike and loss conclusions may not
  transfer to bias-free models.
- **Change.** Three `ModelConfig` flags:
  - `bias=False` on q/k/v/o/up/down;
  - `norm="rmsnorm"` for ln1/ln2/final (gain 1 at init, ε 1e-6, computed in
    FP32);
  - `qk_norm=True`: per-head RMSNorm with a shared (head_dim,) gain on q and k
    after projection.
- **Held fixed.** Learned absolute positions, GELU 4× MLP, 8×64 multi-head
  attention, the untied bias-free head, and GPT-style matrix initialization.
  Also seed 260924, data order, batch, tokens, schedule, clipping, the Muon
  recipe and the 24-matrix spectral cadence.
- **Defaults unchanged.** They reproduce the recorded 8/12-layer initial-weight
  hashes exactly (`test_arch.py`). The variant's initial weights differ from
  the old cohort because biases no longer consume RNG draws; they are
  identical between the variant's Muon and AdamW arms.
- **Simplest competing explanation.** Any difference from the old cohort
  reflects the three changes jointly, plus the different initial draw. No
  single-factor attribution is attempted.

**Comparisons.**
- Variant Muon depth 8 against the existing Muon depth 8: validation NLL, and
  momentum/update spectral trajectories.
- The same spike diagnostic on the variant's final checkpoint (same fresh
  tokens and instrument): mean/sink/distributed readings, cross-fitted bands,
  and Q/K spikes.
- Variant AdamW against variant Muon, with paired initialization and data.

**AdamW learning rate.** Same pilot procedure as the original: 300 updates,
pilot seed 260923, spectra off, selection by mean validation NLL at updates
200/250/300, exact tie to the lower rate. The grid is predeclared as
{0.0006, 0.0012, 0.0024}, centred on the previous selection of 0.0012, which
sat on the upper boundary of the original grid. The main AdamW run uses
four-rank DDP; the original 8-layer AdamW run was single-device.

**Outcomes that change the next decision.**
- If final O/down/V spikes stay mean-route and ρ1 rises without biases, the
  mean route is architecture-robust. That favours an optimizer-side
  mean-direction deflation or centering test.
- If Q/K norms remove Q's mean-route spike or K's sink spike, the paper's Q/K
  behaviour is architecture-specific.
- Loss differences are single-seed.

**Gates and cost.** Stop on a failed tiny CUDA replica audit, a failed
five-update full-size qualification or forecast, nonfinite loss, failed
spectral energy checks, or a pairing hash mismatch. There is no
hyperparameter rescue. Estimated cost is about 2 hours of the 4-GPU
allocation. Outputs go to
`logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925/`.

## NS-budget deflation replication (2026-09-25)

User authorization, 2026-09-25 ~07:15 UTC: "Let's run it to confirm! Use priv-g14 and g20
for two experiments. Let's first do it the way they do it."

Decision argument.
- **Question.** Does spectral deflation (Sun, Kang & Yang 2026, arXiv 2609.21102) recover what a reduced NS budget loses in our setup? The paper's Table 1 at GPT-2 Large reports classic NS with 3 steps at 3.520 versus 3.440 deflated.
- **Mechanism tested.** The spike (a token-mean product here) dominates the Frobenius divisor, so fixed-step NS under-lifts the rest. Removing the head from the divisor should restore it.
- **Simplest competing outcome.** At 77M the late-training spectra are only moderately head-dominated, so the paper's gate may rarely fire and the effect may be undetectable. This does not refute the mechanism at scale.

Comparison. Two arms, identical except the NS entry:
- **plain** (priv-g14): Muon with three classic quintic steps (3.4445, −4.7750, 2.0315).
- **deflated** (g20): the same, behind the paper's deflation entry, ported from their public code (commit 1eb15ce). The operating point is w = 0.025, δ = 0.025, J = 1, τ = 0.1, pad = 1.01. The port matches their `batched_rmfro_clip` to 1e-7 in `test_deflation.py`.

Everything else equals the frontier-norm Muon run: architecture, initial weights (hash checked), data, seed, schedule and our Muon recipe (plain momentum .95, LR .01, aux .002, decay .01, shape scaling). The paper's other recipe choices (Nesterov, zero decay, 40% constant LR) are deliberately not adopted: only the NS map changes. That run (paper five polynomials, 5 steps; val NLL 3.71310) is the 5-step reference.

Prediction, recorded before launch (`expectation.py`, final momentum of that run):
- **Three plain steps keep less.** First-order descent kept falls from ≈ 1.00 to a median of 0.72 (V) to 0.92 (down).
- **The gate fires rarely.** It fires on 11 of 48 matrices (V 7/8, up 3/8, O 1/8; Q, K, down 0), removing 5–12 pairs where it fires.
- **Only V gains.** Its kept descent rises 0.72 → 0.82.
- **Expected outcome.** A small late-training effect, with the paper's gain concentrated early.

Predeclared reading of Δ = val NLL(deflated) − val NLL(plain) at step 1469 (one seed each, no significance claim):
- Δ ≤ −0.02: clearly confirms at our scale.
- −0.02 < Δ < −0.005: consistent but small.
- |Δ| ≤ 0.005: no detectable effect.
- Δ > 0.005: against.

Also reported: validation curves; per-step gate firing and mean pairs; step time; plain-3 versus the 5-step reference. Only after this "their way" result may we consider variants (mean-input-seeded head, dropped head, other gates). Any variant needs its own authorization.

Gates.
- Tiny CUDA run with replica audit. Its deflation window is 0.2 only, so the gate fires on 32-wide matrices.
- Five-update full-size qualification with replica equality, the reference initial-weight hash, and a forecast within the controller's 7 h bound.

Hardware and cost. Both GPU types are Ada AD102 (L40S versus RTX 6000 Ada); cross-hardware rounding is part of the single-seed uncertainty. About 35 minutes per arm, run concurrently.

Outcome (2026-09-25 07:50 UTC). Val NLL: plain 3-step 3.71941; deflated 3-step 3.71144; 5-step reference 3.71310. Δ = −0.0080 reads "consistent but small". Most of the effect is early (−0.078 at step 50), where the gate fires most (38% of matrices in steps 1–100). Details: `logs/muon_spectra/depth8_nobias_rms_qk_ns3_deflation_20260925/comparison.md`.

## Muon improvement search, wave 1 (2026-09-25)

User goal, 2026-09-25 ~08:00 UTC: proceed with the two deflation variants and search for a principled, solid improvement over Muon. Explore this branch first, then broaden. Use the two allocations and the GPU cluster.

Decision argument. Established so far:
- the momentum spike is a bias-like token-mean product ē x̄ᵀ;
- the bulk is mostly noise (cross-fitted descent about 0.2–0.3);
- at a reduced NS budget, deflation recovers loss, mostly early.

Two questions are open. Does handling the spike help at the standard 5-step budget? Does Muon's unit-weight step along the spike help or hurt? The simplest competing explanation for any difference is seed noise, which is not yet measured.

Wave 1 arms. All use the frontier-norm depth-8 architecture and our Muon recipe with the paper's five NS polynomials; only the listed setting changes.
- **S2:** baseline, seed 260925. Seed noise, on the same L40S type as the seed-260924 baseline (3.71310).
- **A:** the paper's deflation (gated rSVD head, restore at 1/1.01) at 5 steps.
- **B:** cheap rank-1 deflation (`deflation_mode` track1; head tracked by one warm-started power iteration; restore at 1/1.01).
- **C:** the same tracked head dropped from the update (head weight 0). Directly tests whether the unit-weight spike step helps. If it hurts, the bias-like direction carries needed updates, motivating centering with a separate offset over dropping. If it helps, Muon over-steps that direction.

Predeclared decision rules (final val NLL at step 1469; Δ = arm − baseline, paired by seed):
- σ̂ is the baseline seed SD, estimated from S2 and the seed-260924 run, and refined as more baseline seeds arrive.
- **Promising:** a single-seed Δ ≤ −max(0.005, 2σ̂).
- **Improvement claim:** needs at least 2 seeds per arm, every paired Δ < 0, and mean Δ ≤ −max(0.005, 2σ̂/√n).
- **Harmful:** Δ ≥ +max(0.005, 2σ̂).
- Other outcomes are inconclusive.

Early-training differences are reported but do not decide. Hardware differs across arms (L40S, RTX 6000 Ada, A6000); claims need seeds on matched hardware or a demonstrated hardware effect below σ̂.

Gates and cost: `run_arm.py` (tiny CUDA replica audit, five-update full-size qualification, reference initial-weight hash for seed 260924, analysis). 36 unit tests pass, and the tracked variant's replica consistency is tested. Each arm takes about 30 minutes on 4 GPUs.

### Wave-2 amendment after peer review and literature scan (2026-09-25 ~08:40 UTC)

**Independent review.** The reviewer argues that the lever is the input mean, not the gradient spike. Muon's polar update treats every input direction alike. But about half of each LayerNorm-fed token's squared norm lies along the shared mean direction x̄, so each step shifts every token's output by a common, noisy amount (the implicit bias Wx̄), whatever the spectrum. Deflation at 5 steps should therefore be null (only B is kept, as control and tracker). Queued arms A and B were cancelled before starting (`CANCELLED_BEFORE_START.txt`).

**Literature scan.** SAMuon (arXiv 2608.25990) finds that the rank-1 head sits at the edge of stability and limits Muon's learning rate. DynMuon and Contra also favour down-weighting the head. SOAP-preconditioned momentum has the strongest Track-3 evidence. No paper centres Muon's inputs in pretraining.

**Hypothesis H_E: data-norm steepest descent.**
- For y = Wx with input second moment Σ ≈ c0 I + x̄ x̄ᵀ, steepest descent under ‖ΔW Σ^{1/2}‖_op is ΔW ∝ polar(M P) P, with P = I − (1 − β) v vᵀ, v = x̄/‖x̄‖ and β* = sqrt(c0/(c0 + ‖x̄‖²)).
- β* ≈ 1/√d_in for LayerNorm-fed inputs, where ‖x̄‖² ≈ ½E‖x‖².
- **Mean-whitened Muon (E).** Muon's update along x̄ is scaled by β*. The Frobenius divisor is no longer dominated by the mean-product spike, and the forward pass is unchanged. β = 1 recovers Muon exactly (tested). β = 0 removes the x̄ column's update, similar to C but with v taken from the data.
- **Implementation.** Per-layer EMAs of the token-mean input and its mean squared norm, position 0 excluded (`StatLinear`, non-persistent buffers, so initial-weight hashes are unchanged).

Predictions, recorded before results:
- **P2.** Muon at LR 0.02 is not better than at 0.01 (near its stability limit). E* holds or improves at higher LR. If both improve at higher LR, the baseline was under-tuned and the comparison is best-versus-best.
- **P3.** E0 ≈ E* means the x̄-column step is irrelevant; E0 worse means a small x̄ step is needed.
- **P4.** In tracked arms, the head's output vector u shows low or negative step-to-step autocorrelation (implicit-bias oscillation).

**Arms.** All use seed 260924 unless noted.
- **priv-g14 (L40S), in order.** E*@0.01, M@0.02, E*@0.02, M@0.014, E*@0.014.
- **g20 (RTX 6000 Ada).**
  - M@0.01, a same-seed hardware twin of S1 that also pairs with C;
  - E*@0.01;
  - E0@0.01;
  - B (tracked restore, logging u/v autocorrelation);
  - Nesterov Muon@0.01.
- **Cluster (A6000, pinned).** Seed-260925 pair M@0.01 and E*@0.01, as a third-hardware replicate.

**Revised decision rule for an improvement claim.** Best LR of each method, at least 3 paired seeds on matched hardware, mean paired Δ ≤ −0.005, and a one-sided paired t-test with p < 0.05. Screening-stage single-seed results are labeled as such.

Wave-2 amendment 2 (2026-09-25 ~08:50 UTC):
- **Arm C result.** Drop-spike, g20, seed 260924: final 3.72302, versus the L40S seeds' 3.71310 and 3.71168.
  - It is −0.054 at steps 100–200, crosses over near step 400, and is +0.010 to +0.015 from step 750.
  - The momentum's top-direction share rises to about 0.75.
  - Reading: "harmful" as a single seed; the same-hardware twin M_lr0.01_g20 is pending.
  - A small late-stage update along x̄ appears necessary.
- **Curvature probe** (`curvature_probe_20260925`). Along x̄, loss curvature is 7–276× that of random input directions, below the 37–680× input second-moment ratio. This suggests β ≈ √(1/ratio) ≈ 0.06–0.4, above the automatic β* ≈ 0.06.
- **New arm.** E with fixed β = 0.25 (A6000, seed 260925), paired with M and E* at the same seed and hardware, to trace loss against β. No other rule changes.

### Wave 3: SOAP-preconditioned Muon (2026-09-25 ~08:55 UTC)

The literature scan's strongest small-scale evidence is the SOAP-Muon core of the Track-3 records:
- momentum projected onto the eigenbases of EMA(GGᵀ) and EMA(GᵀG) (β2 = 0.9, refreshed every step);
- divided by the square root of an EMA of the projected squares;
- rotated back, restored to the input's Frobenius norm, then NS.

Hypothesis linking it to our findings: because G ≈ ē x̄ᵀ + covariance, the top right eigenvector of EMA(GᵀG) is the mean-input direction. The unit test confirms > 0.99 alignment on synthetic mean-product gradients. So SOAP damps the x̄ column automatically, as a gradient-based, full-rank generalization of mean whitening.

Arm S: SOAP-Muon core at LR 0.01 (A6000, seed 260925), paired with M and E at the same seed. Only the SOAP core is adopted, not the record's other tricks (u/w-floor, EMA-Nesterov, radius control, aux β2). SOAP statistics are owner-local and not checkpointed. If S is promising, an ablation restricted to the top right-eigendirection tests how much of SOAP's gain the mean direction carries.

### Waves 2–3 results (2026-09-25 ~10:20 UTC)

Final validation NLL at step 1469. Δ is the arm minus Muon at the same seed, hardware and learning rate.

| arm | seed / hardware | final | Δ | reading (wave-1 rules) |
|---|---|---|---|---|
| C: drop tracked head | 260924 / RTX 6000 Ada | 3.72302 | +0.0094 | harmful |
| E*@0.01 (β* ≈ 0.04–0.1) | 260924 / RTX 6000 Ada | – | −0.0042 | inconclusive |
| E*@0.01 | 260925 / A6000 | 3.71235 | −0.0022 | inconclusive |
| E0.25@0.01 | 260925 / A6000 | 3.70846 | −0.0061 | promising (single seed) |
| E*@0.02 vs M@0.02 | 260924 / L40S | 3.72670 | +0.0066 | harmful |
| **S: SOAP-Muon core@0.01** | 260925 / A6000 | **3.68699** | **−0.0276** | **promising (single seed)** |

**Muon LR bracket** (L40S, seed 260924): 0.01 → 3.71310, 0.014 → 3.71900, 0.02 → 3.72007. 0.01 is the best of the three; 0.007 is untested.

**Shape of the E/C difference curves.**
- M@0.02 − M@0.01 has the same shape as C and E: better at steps 50–150 (−0.06 at 100), crossing near 200, worse afterwards.
- So E/C's early gains are consistent with a larger effective early step. One cause is that removing the spike from the Frobenius divisor lets 5-step NS converge the bulk further.
- S has the opposite shape. It is +0.23 at step 50 while its statistics warm up. Then it is −0.05 to −0.08 at steps 150–300, −0.03 in the stable phase and −0.028 at the end.

**Gradient norm and clipping.**
- Every arm that departs from Muon along the head has a larger total gradient norm, monotone in suppression strength: E* 3.3×, C 2.3×, E0.25 2×, S 1.8×.
- Clip rates: M clips 49% of steps 1–100, never after. S clips 100% then 17% (steps 101–300), never after step 300. E* still clips 22% in steps 301–1000.

**Bias-lag probe** (`logs/muon_spectra/bias_lag_probe_20260925`).
- Refitting the 48 implicit-bias columns (along x̂) on final checkpoints gains ≤ 0.0002 on held-out data for both C and M. A random-direction control gains the same.
- So C's deficit is not a lagging implicit bias. The bias-split variant (`mean_bias_adam`, implemented and tested) is not launched.

**Reference check.** The Track-3 reference (record #44, `soap_precondition_momentum`) also uses the EMA of the projected *update's* squares (β2 0.9, power ½). Our S core matches it. On a μ = 0.95 momentum, that denominator makes the normalization nearly an entrywise sign in the Kronecker eigenbasis, not SOAP's signal-to-noise weighting.

### Wave 4: controls and a SOAP dissection (2026-09-25 ~10:25 UTC)

**Decision argument.**
- S is the only arm with a large, lasting gain: about 20× the baseline seed SD, on one seed. It is a known method (the Track-3 record backbone), so replicating it is not a discovery. The open questions are:
  - where its gain comes from;
  - whether the mean-input story explains it (x̄ is EMA(GᵀG)'s top right eigenvector);
  - whether a principled or cheaper method captures it.
- The E/C effects are small, and a step-size confound (NS conditioning) is plausible. So wave 4 pairs the SOAP dissection with controls.
- Independent review (10:10 UTC). Its main recommendations:
  - an exact-polar control;
  - a Muon LR bracket from below;
  - a sharp test of "the head sets the LR ceiling";
  - replicating S across hardware;
  - dissecting S with S_right and S_none;
  - checking which second moment the reference uses (done, above);
  - no stacking of extra x̄ shrinkage on SOAP.

**Arms.** A6000 (cluster), seed 260925, LR 0.01 unless noted; paired with M/E*/E0.25/S at that seed.
- **X_svd.** Muon with the exact polar factor (FP32 SVD). Controls for NS conditioning and gives a norm-matched baseline.
- **S_right.** SOAP with only the input-side (right) eigenbasis.
- **S_none.** Entrywise normalization in the standard basis.
- **S_gradsq.** Second moment of the projected *gradient* (SOAP's signal-to-noise weighting) instead of the projected momentum.
- **E0.5@0.02.** Head at Muon@0.01's step, bulk at 0.02 (SAMuon-style).
- **M@0.007.**
- **Replication.**
  - M and S at seed 260926 on A6000.
  - S at seed 260924 on RTX 6000 Ada (g20, after its queue), paired with M_lr0.01_g20.
  - S at seed 260924 on L40S (priv-g14, after its queue), paired with the reference run. It runs after E_auto_lr0.01_r2, which had already started (10:19 UTC) when this plan was written, so it was kept.

**Predictions, recorded before results.**
- **P5.** If E/C's early gains are NS conditioning, X_svd reproduces most of E*'s early gain (≤ −0.03 at steps 100–200) and ends within ±0.005 of M.
- **P6.** If S's gain lives on the input side, S_right keeps ≥ ⅔ of S's final gain. If the eigenbasis matters, S_none keeps ≤ ⅓.
- **P7.** S_gradsq differs from S by ≥ 0.005. Better means signal-to-noise weighting beats the near-sign normalization.
- **P8.** If the head sets Muon's LR ceiling, E0.5@0.02 beats M@0.01 by ≥ 0.005. If it tracks M@0.02 (+0.007), the head branch closes.
- **P9.** M@0.007 is within ±0.004 of M@0.01.

**Decision rules.** Unchanged: 3 paired seeds, mean Δ ≤ −0.005, one-sided paired t, p < 0.05. Added before results:
- The seeds behind an S-family claim span at least two GPU types.
- If 0.05 < p < 0.2, extend to 5 seeds.
- A dissection arm counts as "retaining" S's gain relative to the S–M gap at its own seed.

### Wave 5: partial data-norm Muon (2026-09-25 ~10:45 UTC)

**Decision argument.**
- This is the review's main proposal and the principled generalization of E. For y = Wx with input second moment C = E[xxᵀ], steepest descent under ‖ΔW C^α‖_op is ΔW = polar(M C^{−α}) C^{−α}.
  - α = ½ is the full data norm, the operator norm measured on the actual input distribution.
  - Smaller α is milder.
  - A rank-1-plus-isotropic C recovers E with β = (c0/(c0+‖x̄‖²))^α. So E0.25, the best E arm, corresponds to α ≈ ¼ on the mean direction alone.
- Literature status (subagent scan, 10:00 UTC):
  - An activation-covariance sandwich exists only at toy scale (GO-MUON).
  - A diagonal, clamped version is IsoMuon: −0.004 at 124M, with early gains.
  - The input-only, full-matrix sandwich has not been tested at ≥ 100M.
- It uses forward statistics only and is independent of SOAP's gradient statistics. That separates two questions:
  - whether input anisotropy beyond x̄ matters;
  - whether it accounts for part of S's gain.
- Implementation:
  - `StatLinear(cov=True)` keeps an EMA of E[xxᵀ] from every 32nd position (8192 tokens per rank per step, decay 0.998 per forward).
  - C is scaled to unit mean eigenvalue plus 1e-3·I. A float64 eigh runs every 10 steps and is cached owner-locally.
  - The update is rescaled to Muon's Frobenius norm, so only its shape changes.
- Unit tests check:
  - the covariance statistic;
  - the root;
  - the steepest-descent property: polar(GR)R beats 50 random feasible directions;
  - RMS matching;
  - that α → 0 recovers Muon.

**Arms.** A6000, seed 260925, LR 0.01, paired with M, E and S at that seed:
- PD α = ¼;
- PD α = ⅛, milder whitening, which Mousse preferred.

**Prediction, recorded before results.**
- **P10.** At least one PD arm reaches Δ ≤ −0.005 at the end.
- If PD α = ¼ ≈ E0.25 (−0.006), the input side is just x̄.
- If PD gets within 0.01 of S (≤ −0.018), input anisotropy explains much of SOAP-Muon's gain.

### Wave 6: what carries the input-side SOAP gain (2026-09-25 ~10:40 UTC)

**Evidence at launch** (wave-4 arms at steps 100–300, seed 260925, vs M):
- S_right (input-side eigenbasis only): −0.079, −0.103, −0.077. It has none of S's early deficit.
- S_gradsq: −0.045, −0.106.
- S_none (standard basis): +0.091, +0.079, +0.023. So the basis matters.
- Exact-polar X_svd: −0.026, −0.018 at steps 100–200. Part of the early E/C gain is NS conditioning.

Finals are pending. These arms are launched early because they answer the next questions whatever the finals show, as long as S_right keeps most of S's gain. That is not yet known. If it does not, these arms are read only as mechanism probes.

**Decision argument.**
- S_right does three things:
  - rotates the momentum's input side into the eigenbasis of EMA(GᵀG);
  - normalizes each (output coordinate, input eigendirection) entry by its own running RMS;
  - rotates back before the polar.
- For a linear layer, EMA(GᵀG) is roughly an output-gradient-weighted input second moment, with x̄ on top. So two questions separate the parts.
  - Does the activation second moment (forward statistics only, `right_act`) give the same basis benefit?
  - Is the entrywise normalization needed, or does a one-sided Kronecker version suffice, one denominator per input eigendirection (`soap_norm column`)?
    - If the Kronecker version suffices, the method reduces to "whiten the momentum's input side before orthogonalizing".
    - Its activation-statistics analogue is polar(M C^{−½}), partial data-norm without post-multiplication (`data_norm_post false`, α = ½).

**Arms.** A6000, LR 0.01:
- S_right at seed 260926, a replication paired with M_lr0.01_s260926;
- S_rightact at seed 260925;
- S_rightcol at seed 260925;
- PDin α = ½ at seed 260925.

**Predictions.**
- **P11.** S_right replicates, with Δ ≤ −0.015 at seed 260926.
- **P12.** If input anisotropy measured on activations is enough, S_rightact keeps ≥ ⅔ of S_right's gain.
- **P13.** If the entrywise part matters, S_rightcol keeps ≤ ⅓ of S_right's gain. PDin ≈ S_rightcol would mean the basis source is secondary to the Kronecker/entrywise distinction.

**B_track1_restore_g20 failed** in qualification (10:40 UTC), with a non-finite update at step 1.
- Likely cause: the restored-head entry is not spectrally bounded when the one-step head estimate is off, and the paper's NS polynomials diverge above σ ≈ 1.33.
- Details: `logs/muon_spectra/improve_w2_20260925/B_track1_restore_g20/FAILURE.md`.
- Not re-run, because X_svd covers the NS-conditioning control. The g20 queue continued with N.

### Wave-4 results so far, and wave 7 (2026-09-25 ~11:15 UTC)

Final paired Δ vs Muon@0.01 (same seed and hardware):

| arm | Δ | reading |
|---|---|---|
| S (SOAP-Muon core), seed 260925, A6000 | −0.0276 | |
| S, seed 260924, L40S (priv-g14) | **−0.0320** | replicates on a second GPU type |
| S_right (input-side basis only), seed 260925 | −0.0214 | keeps 0.78 of S's gain, and has no early deficit (−0.08 to −0.10 at steps 100–200) |
| S_none (standard basis) | −0.0050 | keeps 0.18; after a +0.09 early deficit |
| Nesterov Muon (g20) | +0.0001 | no effect here |
| E(β = 0.5)@0.02, head at Muon@0.01's step | +0.0099 at step 1250 | tracks Muon@0.02 |

**Predictions.**
- P6 holds: the input side keeps ≥ ⅔ of S's gain, and the eigenbasis matters.
- P8 fails: the head does not set Muon's LR ceiling, which closes the head-LR branch.
- The output-side basis adds about 0.006 late, but causes S's early deficit.

**Wave 7 (priv-g14, L40S, seed 260924; decision argument).**
- The claim rule needs each method at its best LR. Muon's best on this hardware and seed is 0.01 among {0.01, 0.014, 0.02}. S has been run only at 0.01.
- S_right also needs a second hardware type.
- Arms, in order:
  - S_right@0.01;
  - S@0.014, paired with M_lr0.014;
  - M@0.007;
  - S@0.007.
- On g20 after S_s260924_g20: S_right@0.01 at seed 260924.

**S replication complete (11:22 UTC).** SOAP-Muon core@0.01 minus Muon@0.01, paired:
- −0.0276 (A6000, seed 260925);
- −0.0320 (L40S, seed 260924);
- −0.0270 (A6000, seed 260926).

Mean −0.0289, SD 0.0028, one-sided paired t = −17.9, df 2, p ≈ 0.002. Every paired Δ is negative, and two GPU types are covered.
- This meets the claim rule except for the best-LR clause. Muon's bracket has been run at {0.01, 0.014, 0.02}, with 0.01 best; 0.007 is pending. S has been run only at 0.01; 0.007 and 0.014 are queued (wave 7). A fourth seed on RTX 6000 Ada is running.
- Finals of the other arms:
  - exact-polar Muon: −0.0028 at step 1250;
  - S_gradsq: −0.0255 at 1250;
  - E(β = 0.5)@0.02: +0.0093 vs Muon@0.01.
- NS conditioning explains little of the late differences.

**Muon's LR bracket moves down (11:25 UTC).** On A6000 at seed 260925:
- M@0.007: 3.70630;
- M@0.01: 3.71457.

So 0.007 is 0.0083 better, and P9 fails. The review's advice to bracket from below was right, and 0.01 was not Muon's best LR.
- Best-vs-best comparison, same seed and hardware: S@0.01 3.68699 vs M@0.007 3.70630 gives Δ = −0.0193. This is still well beyond the rule's −0.005.
- S_gradsq final is 3.68727, the same as S. P7 fails: the choice of second moment does not matter here.

**Wave 8 (A6000, cluster).** The best-LR clause of the rule requires these arms:
- M@0.005 at seed 260925, bracketing Muon further down;
- S@0.007 at seeds 260925 and 260926;
- M@0.007 at seed 260926.

Together with the L40S arms of wave 7, they locate each method's best LR on two GPU types before any claim.

### Wave 9: replicate partial data-norm Muon at its best LR (2026-09-25 ~11:55 UTC)

**Result that triggers it.** Partial data-norm Muon (PD, α = ¼) is polar(M C^{−¼}) C^{−¼}, with C the EMA activation second moment and the update RMS-matched.
- Final 3.68908 vs Muon@0.01 3.71457 at seed 260925 on A6000: Δ = **−0.0255**.
- That keeps **0.92** of the SOAP-Muon core's gain at the same seed, using only forward statistics.
- The literature scan found no test of this input-only, full-matrix sandwich at ≥ 100M.
- Vs Muon at its better LR 0.007 (3.70630), same seed: Δ = −0.0172.

**Also final.**
- S at seed 260924 on RTX 6000 Ada: −0.0317. S now has 4 seeds on 3 GPU types: mean −0.0296.
- S_right at seed 260924 on L40S: −0.0248.

**Decision argument.**
- PD is the principled candidate: steepest descent under a partial data norm, where Muon is the α = 0 special case. It rests on measured input anisotropy: effective rank 50–240 of 512, top eigenvalue 55–590× the median.
- The claim rule needs PD and Muon at their best LRs with ≥ 3 paired seeds on ≥ 2 GPU types.
- Muon's bracket: 0.005 (running), 0.007, 0.01, 0.014, 0.02.
- PD needs 0.007 and a replication at 0.01.
- α = ⅛ and the other input-side variants finish shortly. They may change which α is carried forward, but they do not change the need for this bracket.

**Arms.**
- A6000: PD α = ¼ @0.01 at seed 260926; @0.007 at seeds 260925 and 260926.
- L40S (priv-g14, after wave 7): PD α = ¼ @0.01 and @0.007 at seed 260924.
- RTX 6000 Ada (g20, after S_right): PD α = ¼ @0.01 and Muon@0.007 at seed 260924.

**Prediction (P14).** PD α = ¼ at its best LR beats Muon at its best LR by ≤ −0.010 on every seed.

### Waves 5–6 results: what carries SOAP-Muon's gain (2026-09-25 ~12:25 UTC)

Final Δ vs Muon@0.01, same seed and hardware. "Share of S" is relative to S@0.01 at the same seed and hardware.

| arm | Δ | share of S | reading |
|---|---|---|---|
| S_right, input-side eigenbasis only (4 seeds, 3 GPU types) | −0.0214, −0.0224, −0.0248, −0.0250 (mean −0.0234) | 0.77–0.83 | the input side carries ~80%; no early deficit |
| S_rightact, activation second-moment basis, entrywise | −0.0205 | 0.74 | forward statistics suffice (P12 holds) |
| S_rightcol, one denominator per input eigendirection | −0.0145 | 0.53 | entrywise beats one-sided Kronecker (P13's ≤ ⅓ fails; partial) |
| PDin, polar(M C^−½), no post-multiplication | −0.0147 | 0.53 | ≈ S_rightcol: basis source secondary |
| **PD α = ¼, polar(M C^−¼) C^−¼, RMS-matched** | **−0.0255** | **0.92** | the full sandwich recovers what entrywise normalization gives |
| PD α = ⅛ | −0.0171 | 0.62 | α = ¼ > ⅛ |
| S_gradsq, projected-gradient second moment | −0.0273 | 0.99 | sign-like vs SNR: no difference (P7 fails) |
| S_none, standard basis | −0.0050 | 0.18 | the basis is essential |
| X_svd, exact polar | −0.0031 | 0.11 | NS conditioning is small |

**Reading.**
- SOAP-Muon's advantage over Muon is mostly input-side whitening of the momentum before orthogonalization.
- Whitening with the activation second moment works as well as with the gradient Gram matrix.
- With only a Kronecker (one-sided) whitening inside the polar, about half the gain is lost. Adding the post-multiplication, which is exact steepest descent under the partial data norm ‖ΔW C^¼‖_op, recovers it.
- The data norm and SOAP's entrywise normalization reach the same place by different routes. The data norm needs only forward statistics.
- The curvature probe measured loss curvature along input directions as roughly (second moment)^γ with γ ≈ 0.4–1.0, median ≈ 0.65. A curvature-matched norm would use α ≈ γ/2 ≈ ¼–⅓, consistent with α = ¼ beating ⅛. α = ¼ was picked from two values, so the replication in wave 9 is the test.

### Second review of the data-norm claim, and wave 10 (2026-09-25 ~12:45 UTC)

**Review, summarized.**
- **A structural point.** Every NS-ended method outputs P with PᵀP ≈ I on square and tall matrices (q, k, v, o, up). So ‖ΔW x‖ is the same for every token. There, SOAP (S) cannot take smaller steps along high-variance inputs; it only re-routes inputs to outputs. Only the wide MLP-down, and PD's post-multiplication on every layer, can reallocate step size across input directions. "PD keeps 92% of S" may therefore be two mechanisms reaching a similar loss.
- **Key test.** Run SOAP in PD's coordinates (S∘PD): SOAP on M C^−α with statistics from G C^−α, then post-multiply by C^−α, with the same RMS rule.
  - S∘PD − PD ≥ −0.005: S's gain is mostly input anisotropy, and PD is its cheap form.
  - S∘PD − PD ≤ −0.012: the claim is falsified; the two stack.
- **Controls.**
  - Magnitude-matched Muon: Muon's direction, rescaled per layer each step to PD's E‖ΔW x‖. RMS matching may act as a hidden per-layer LR cut.
  - PDout = polar(M) C^−α, outside only. With PDin it completes an inside/outside × α factorial.
  - α = ⅜.
  - One centered-C arm.
  - Localization: S only on MLP-down vs S on all other matrices.
  - Damping ×10 and ÷10 once.
- **Bar for "solid".**
  - Freeze α and LR on selection seeds, then confirm on fresh seeds on ≥ 2 GPU types.
  - Bracket PD's LR at {0.007, 0.01, 0.014}, since Frobenius-matched PD moves outputs less than Muon on square layers.
  - Beat magnitude-matched Muon.
  - Report tokens-to-target and wall-clock.
  - Check a longer horizon.

**Frozen now (before the new arms).**
- α = ¼ is the PD candidate.
- Selection seeds are 260925 and 260926. Confirmation seeds are 260924 (L40S and RTX 6000 Ada, already running at LR 0.01) plus fresh 260927 and 260928 on A6000.
- **P15.** The PD optimum over α lies in [¼, ⅜], and ½ (full data norm, with post-multiplication) is worse than ¼. This follows from the measured sublinear curvature, γ ≈ 0.65 ⇒ α ≈ 0.33.
- **P16.** Magnitude-matched Muon keeps < ⅓ of PD's gain. Otherwise the geometry claim fails.
- **P17.** S∘PD − PD ≥ −0.005, meaning S's gain is mostly input anisotropy.

The review's free cooldown check: PD α = ¼ is −0.0240 at step 1250 and −0.0255 at 1469, so its gap does not close in cooldown. S goes from −0.0295 to −0.0276.

**Wave 10 arms** (A6000, selection seed 260925, LR 0.01 unless noted; paired with the existing seed-260925 arms). Implementation, all with unit tests:
- `data_norm_pre` (outside-only);
- `data_norm_mode magnitude_matched`;
- SOAP in whitened coordinates (data norm together with SOAP);
- `soap_layers`;
- `data_norm_center`.

The arms:
- PD α = ¼ @0.014 (LR bracket);
- MM α = ¼, magnitude-matched Muon (P16);
- S∘PD α = ¼ (P17);
- PDout α = ¼;
- PD α = ⅜ and α = ½ (P15);
- PD α = ¼ with centered C;
- S only on MLP-down;
- S on all matrices except MLP-down.

Fresh-seed confirmation (260927, 260928) waits for PD's LR bracket.

### Wave 11: fresh-seed confirmation at the best LRs (2026-09-25 ~12:50 UTC)

**LR evidence on the selection seeds and early confirmation hardware.**
- Muon@0.007 beats Muon@0.01 on all three seeds and both GPU types: −0.0088 (L40S, seed 260924), −0.0083 (A6000, 260925), −0.0056 (A6000, 260926). Muon@0.005 is worse than 0.007 (260925). **Muon's best LR is 0.007**, bracketed on both sides.
- PD α = ¼ at 0.007 trails PD at 0.01 at step 500: −0.025 vs −0.042 at seed 260925, and −0.034 vs −0.043 at 260926, both vs Muon@0.01. So PD's best is ≥ 0.01; 0.014 is running.
- S's best on seed 260925 is 0.007 (3.68490 vs 3.68699 at 0.01).

**Frozen confirmation design** (before any confirmation result):
- PD α = ¼ @0.01 vs Muon@0.007 on fresh seeds 260927 and 260928 (A6000), plus seed 260924 on L40S and on RTX 6000 Ada.
- SOAP-Muon core @0.007 on 260927 and 260928, for a best-vs-best comparison with PD.
- If PD@0.014 beats PD@0.01 on the selection seeds, PD@0.01 is kept as the confirmation LR and reported as a conservative (lower-bound) setting.
- Claim rule: every paired Δ(PD@0.01 − Muon@0.007) < 0, mean ≤ −0.005, one-sided paired t p < 0.05 over ≥ 3 confirmation pairs on ≥ 2 GPU types.

**Best-vs-best so far** (vs Muon@0.007, same seed and hardware; `compare_all.py`):
- S@0.01: n = 3 on 2 GPU types, −0.0193, −0.0232, −0.0214 (mean −0.0213, t = −18.9). The SOAP-Muon core beats tuned Muon; this replicates a known method.
- S_right@0.01: n = 3, mean −0.0153.
- PD α = ¼ @0.01: −0.0172 (selection seed; confirmation running).
- The mean-input branch does **not** beat tuned Muon: E*@0.01 is +0.006 (n = 2) and E0.25 is +0.002. Its small gains over Muon@0.01 were a baseline-LR effect.
- Exact polar is +0.005.

**Wave 12 (one arm, seed 260925, A6000).** PD α = ¼ + output-row normalization (`data_norm_rows`). This is a cheap diagonal proxy for S's output-side basis, which added about 0.006 over S_right. The review ranked it low (it can add at most ~0.006), but it costs one arm and tests whether a one-sided data norm leaves output-side gain behind.

**Scheduling note (13:05 UTC).** The cluster was saturated. The four lowest-priority pending mechanism arms (PDc, S_down, S_notdown, PDrows) were put on `scontrol hold` so the wave-11 confirmation arms start first. They will be released afterwards. The depth-20 jobs were not touched.

**Selection-seed status** (vs Muon@0.007):
- PD α = ¼ @0.01: −0.0172 (260925), −0.0181 (260926).
- S@0.007: −0.0216, −0.0214, −0.0246.
- S@0.01: mean −0.0213.
- PD at 0.007 gains less than at 0.01 on both selection seeds, so PD's best LR is ≥ 0.01; 0.014 is running.

### Wave 13: twice the horizon (2026-09-25 ~13:20 UTC)

The review's bar includes holding up at a longer horizon. `run_arm.py` now takes its horizon from the arm's `total_tokens`:
- steps = ⌈tokens / batch⌉;
- spectral samples = 1 + ⌊steps/25⌋ + 1;
- a proportional time budget.

The default horizon is unchanged: 1469 steps and 60 samples.

**Arms.** 3,079,741,440 tokens (2938 steps), each method at its 1×-horizon best LR:
- Muon@0.007 and PD α = ¼ @0.01, paired on L40S (priv-g14, seed 260924);
- the same pair on RTX 6000 Ada (g20, seed 260925).

Both allocations run these after their current queues.

**Prediction (P18).** PD's advantage over Muon persists at 2×: Δ ≤ −0.010 on both pairs. The same LRs are used at 2×, so a shift in either method's optimal LR with horizon is a stated limitation.

### Wave-10 results and wave 14 (2026-09-25 ~14:00 UTC)

Seed 260925, A6000, LR 0.01 unless noted. Final Δ vs Muon@0.01 / vs tuned Muon@0.007:

| arm | vs M@0.01 | vs M@0.007 | reading |
|---|---|---|---|
| **S∘PD α = ¼** (SOAP-Muon core on M C^−¼ with statistics from G C^−¼, post C^−¼, RMS-matched) | **−0.0367** | **−0.0284** | best so far; beats S (−0.0214 vs M@0.007) and PD (−0.0172) |
| MM α = ¼ (Muon's direction at PD's per-layer E‖ΔW x‖) | −0.0099 | −0.0017 | ≈ tuned Muon: PD's magnitude change is an LR effect that tuned Muon already captures |
| PD α = ¼ @0.014 | −0.0189 | −0.0106 | PD's best LR is 0.01 (0.007 and 0.014 both worse) |
| PDout α = ¼ (outside only) | (−0.0188 at 1000) | | |
| PD α = ⅜ / ½ | (−0.034 / −0.012 at 500; α = ¼ was −0.042) | | α = ¼ best so far |

**Predictions.**
- **P16.** MM keeps 39% of PD's gain over Muon@0.01, not < ⅓, so it fails literally. But Muon@0.007 alone keeps a third. Against tuned Muon, MM keeps ~10% of PD's −0.0172. **PD's gain over tuned Muon is geometric, not a per-layer LR cut.**
- **P17 fails.** S∘PD − PD = −0.0112, just short of the review's "stacked" threshold (−0.012). SOAP's gain is **not** mostly input anisotropy. SOAP's entrywise basis normalization and the data norm's step reallocation across input directions are largely complementary. That fits the review's structural point: on square and tall matrices, SOAP alone cannot change step size across input directions.
- P15 is consistent so far: α = ¼ > ⅜ > ½.

**Wave 14 (A6000, LR 0.01, α = ¼): confirm S∘PD.**
- Seeds 260926 (selection), 260927 and 260928 (confirmation).
- S∘PD@0.007 at 260925 (LR check).
- More GPU types later, on the allocations after the 2× horizon arms.
- Claim rule as for PD, vs Muon@0.007.

**Cost check (same L40S node, priv-g14, seed 260924).** Median training step time:
- Muon: 0.894–0.898 s.
- PD α = ¼: 0.946 s (+6%).
- SOAP-Muon core: 0.971 s (+8%).
- Input-side SOAP: 0.930 s (+4%).

All use unoptimized eager implementations. PD's cost is the covariance EMA plus an FP64 eigh every 10 steps.

### Wave 15: token efficiency (2026-09-25 ~14:20 UTC)

A tokens-to-target estimate needs runs with their own cooldown. Arms: PD α = ¼ @0.01 at 0.70× and 0.85× of the token budget (seed 260925, A6000), compared with Muon@0.007 at 1× (3.70630).

**Reading, fixed now.** If PD at f× reaches ≤ 3.70630, PD needs ≤ f of Muon's tokens. The efficiency multiplier is interpolated between f = 0.70, 0.85 and 1.0.

### Claim: partial data-norm Muon beats tuned Muon (2026-09-25 14:15 UTC)

Partial data-norm Muon: ΔW = s·polar(M C^−¼) C^−¼, with C the activation second moment and s matching Muon's Frobenius norm. It is compared with Muon, each at its bracketed best LR: PD 0.01 (0.007 and 0.014 worse), Muon 0.007 (0.005 and 0.01 worse). α and LR were chosen on seeds 260925 and 260926.

Confirmation pairs, fresh with respect to that selection, all Δ = PD − Muon:

| seed | GPU | Δ |
|---|---|---|
| 260924 | L40S | −0.0186 |
| 260924 | RTX 6000 Ada | −0.0198 |
| 260927 | A6000 | −0.0189 |

Mean −0.0191, SD ≈ 0.0006, one-sided paired t ≈ −55 (df 2), p < 0.001. Every Δ is negative, and the pairs span three GPU types. With the selection seeds included, n = 5: mean −0.0185, SD 0.0010.

**This meets every clause of the predeclared claim rule.** The fourth confirmation pair (260928) is still running.

Scope: one model size (77M, depth 8, width 512), one data set, one token budget (20 tokens per parameter). The 2× horizon (wave 13) and token-efficiency (wave 15) checks are running.

**α curve and inside/outside factorial** (seed 260925, A6000, LR 0.01; Δ vs Muon@0.01):
- α sweep: ⅛ −0.0171, **¼ −0.0255**, ⅜ −0.0207, ½ −0.0014. **P15 holds**: the optimum is at ¼ within [¼, ⅜], and the full data norm (½) gains nothing.
- Inside/outside:
  - outside only, polar(M) C^−¼: −0.0158;
  - inside only, polar(M C^−½): −0.0147;
  - sandwich, α = ¼: −0.0255.
- Both halves matter, and the sandwich is best. Inside-only ½ and the ¼ sandwich have the same total exponent, so their gap is the sandwich effect, confounded with the per-side α.

### Wave 16: which SOAP side stacks with PD (2026-09-25 ~14:20 UTC)

S∘PD stacks (−0.0284 vs tuned Muon on the selection seed; confirmations running). PD already whitens the input side. The hypothesis is that the stacking gain comes from SOAP's **output** side, the left eigenbasis of the whitened gradient Gram G C^−½ Gᵀ.

Arm: S_left∘PD α = ¼ @0.01 (seed 260925, A6000): the left basis only, in whitened coordinates.

**Reading (P19).** If S_left∘PD keeps ≥ ⅔ of S∘PD's gain over PD, the recipe decomposes cleanly:
- data norm (forward statistics) on the input side;
- SOAP-style entrywise normalization (gradient statistics) on the output side.

### Wave 17: instrumentation control (2026-09-25 ~14:30 UTC)

PD runs use `StatLinear` layers that record input statistics during training forwards (`track_input_stats` and `track_input_cov`). Muon runs use plain `nn.Linear`. Initial weights and hashes are identical, and the forward math is the same, but the compiled graphs differ.

Control: Muon@0.007 with both tracking flags on (seed 260925, A6000), vs Muon@0.007 without them (3.70630). Expected |Δ| ≲ 0.0015 (hardware and seed-noise scale). A larger difference would mean the instrumentation itself changes training.

### PD confirmation complete; S∘PD on more GPU types (wave 18, 2026-09-25 ~15:05 UTC)

**PD α = ¼ @0.01 vs tuned Muon@0.007.** Fresh confirmation pairs:

| seed | GPU | Δ |
|---|---|---|
| 260924 | L40S | −0.0186 |
| 260924 | RTX 6000 Ada | −0.0198 |
| 260927 | A6000 | −0.0189 |
| 260928 | A6000 | −0.0200 |

Mean −0.0193, SD 0.0007. All six pairs: mean −0.0188, SD 0.0010.

**S∘PD α = ¼ @0.01 vs tuned Muon@0.007, all on A6000.**

| seed | role | Δ |
|---|---|---|
| 260925 | selection | −0.0284 |
| 260926 | selection | −0.0327 |
| 260927 | fresh | −0.0302 |
| 260928 | fresh | −0.0325 |

Mean −0.0309, SD 0.0020. For comparison, the SOAP-Muon core@0.007 is −0.0235 (n = 5) and PD is −0.0188.

The claim rule needs ≥ 2 GPU types. Wave 18 runs S∘PD@0.01 at seed 260924 on L40S (priv-g14) and on RTX 6000 Ada (g20), after the 2× PD arms. Both pair with the existing Muon@0.007 runs.

**Wave-13/10/14 readings (15:30 UTC).**
- **2× horizon, RTX 6000 Ada, seed 260925.** PD@0.01 − Muon@0.007 = **−0.0145** at step 2938. The gap narrowed to −0.010 by step 2500, then widened again in cooldown. P18 (Δ ≤ −0.010) holds on this pair; the L40S pair is running. Muon@0.007 itself improves by 0.108 from 1× to 2×. Using that per-doubling slope, PD's 1× gain is worth about 2^(0.019/0.108) ≈ 1.13× tokens, and S∘PD's about 1.22×. Wave 15 measures this directly.
- **Centered C (PDc).** −0.0089 vs Muon@0.007, compared with −0.0172 for uncentered PD at the same seed. Whitening the shared mean direction inside the full C is about half of PD's gain. Mean-only whitening was not enough, and neither is the centered anisotropy alone.
- **S∘PD@0.007** (seed 260925): −0.0273, compared with −0.0284 at 0.01. 0.01 is kept.

**2× horizon complete (15:40 UTC).** PD@0.01 − Muon@0.007 at 3.08B tokens (2938 steps): **−0.0164 (L40S, seed 260924) and −0.0145 (RTX 6000 Ada, seed 260925)**, mean −0.0154. **P18 holds on both pairs.** The advantage persists at twice the horizon, somewhat smaller in nats: −0.019 at 1× vs −0.015 at 2×.

Mechanism arms so far (seed 260925, vs Muon@0.007):
- SOAP only on MLP-down: +0.0003 at step 1250 (nothing).
- SOAP on everything except MLP-down: −0.0202 at step 1000.
  - So SOAP's gain lives on the square and tall matrices, where it can only re-route inputs to outputs. It is not in step-size reallocation on MLP-down.
- PD + output-row normalization ≈ PD (−0.0151 at 1000).
- The instrumentation-only Muon control tracks Muon@0.007 (+0.0013 at step 500).

### Wave 19: width-768 scale check (2026-09-25 ~15:55 UTC)

**Decision argument and review (15:50 UTC).** An independent review recommended width 768 over depth 12, a larger batch, or the old architecture.
- Width is the usual way optimizer gains shrink.
- Width is also the axis that changes the object PD acts on: the size and spectrum of C.
- LRs must be re-bracketed for both methods, since PD's RMS match and Muon's shape scaling need not transfer alike.

**Design.** Depth 8, width 768, 12 heads (head dim 64), 20 tokens per parameter; batch, warmup, cooldown, WD, momentum, NS and the aux AdamW LR unchanged.
- **Fixed, not re-tuned:** α = ¼, the covariance EMA, the eigh cadence, damping 1e-3, and the RMS match. That α carries over is part of the claim.
- **Selection bracket** (seed 260925, A6000): Muon {0.005, 0.007, 0.01}, PD {0.007, 0.01, 0.014}. If a best LR sits at the edge of the grid, extend that arm's grid.
- **Fresh pairs at the carried-over LRs** (Muon 0.007, PD 0.01): seed 260924 on L40S (priv-g14) and seed 260926 on RTX 6000 Ada (g20). They count only if the bracket confirms those LRs; otherwise they are re-run at the bracketed ones.

**Criteria, fixed now.**
- **Supports** "holds at width 768": every fresh-seed Δ ≤ −0.012, i.e. about ⅔ of the width-512 gain.
- **Weakens:** mean Δ above −0.009, or any seed ≥ 0.
- In between: ambiguous; add a seed.
- Also weakening:
  - PD's advantage depends on an edge LR;
  - its token-equivalent speedup falls under half the width-512 value;
  - its step-time overhead eats the speedup.
- Even if the check passes, the claim will be stated as "holds across 1.5× width and 2× horizon at ≤ 134M params", not as a scaling law.

**Token efficiency (wave 15, seed 260925, A6000, each run with its own cooldown).** PD α = ¼ @0.01:

| token budget | PD val NLL | Δ vs Muon@0.007 at 1× (3.70630) |
|---|---|---|
| 0.70× | 3.76158 | +0.0553 |
| 0.85× | 3.71833 | +0.0120 |
| 1.00× | 3.68909 | −0.0172 |

Linear interpolation puts the crossing at ≈ 0.91× tokens, so **PD needs about 9% fewer tokens than tuned Muon: a speedup of ≈ 1.10×**.
- Muon's own 1× → 2× slope (≈ 0.108 nats per doubling) gives a similar 1.13×.
- By the same slope, SOAP-Muon's −0.0235 is ≈ 1.16× and S∘PD's −0.031 is ≈ 1.22×.
- Against the unoptimized step-time overheads (PD +6%, SOAP +8%), the wall-clock gains are smaller: PD ≈ 1.04×.
- The loss gaps are highly significant, but they are modest compute multipliers. Cheaper statistics (fewer covariance tokens, rarer eigh) would bring the wall-clock gain toward the token gain.

**Curvature exponent measured directly (16:10 UTC; `logs/muon_spectra/gamma_probe_20260925`).** On tuned Muon's final weights, loss curvature along input eigendirections q_j scales as (qᵀCq)^γ.
- 8 matrices, 12 ranks each: γ = 0.36–0.75, **median 0.548**.
- **The curvature-matched exponent α = γ/2 ≈ 0.27, matching the sweep's optimum α = ¼.**
- The full data norm (α = ½) assumes γ = 1, which is why it over-corrects and gains nothing.
- This ties the method's one hyperparameter to a measured property of the loss.

### S∘PD confirmed across GPU types; mechanism arms final (16:12 UTC)

**S∘PD α = ¼ @0.01 vs tuned Muon@0.007, n = 6 on 3 GPU types.**

| seed | GPU | role | Δ |
|---|---|---|---|
| 260925 | A6000 | selection | −0.0284 |
| 260926 | A6000 | selection | −0.0327 |
| 260927 | A6000 | fresh | −0.0302 |
| 260928 | A6000 | fresh | −0.0325 |
| 260924 | L40S | fresh | −0.0302 |
| 260924 | RTX 6000 Ada | fresh | −0.0336 |

Mean −0.0313, SD 0.0020 (fresh only: mean −0.0316).
- S∘PD took α and LR from PD. Its LR is checked below (0.007 is worse: −0.0273) but not above; S∘PD@0.014 is queued (wave 20) to complete the bracket.
- Paired against the SOAP-Muon core at its best LR on the same seeds, S∘PD is better by ≈ 0.007–0.009 every time. The data norm improves SOAP-Muon as well as Muon.

**Mechanism arms (seed 260925, vs Muon@0.007).**
- Instrumentation-only Muon: +0.0007. The tracking layers do not change training.
- PD + output-row normalization: −0.0161, the same as PD (−0.0172).
- SOAP only on MLP-down: +0.0001. SOAP on all other matrices: −0.0173. SOAP's gain lives on the square and tall matrices.
- **S_left∘PD** (output-side SOAP basis only, in whitened coordinates): −0.0261. It keeps 79% of S∘PD's gain over PD, so **P19 holds**. The stacked method decomposes:
  - **input side:** partial data norm (forward statistics, α = γ/2);
  - **output side:** SOAP-style entrywise normalization in the eigenbasis of the whitened gradient's left Gram.

**Where the gains land (16:35 UTC; `logs/muon_spectra/frequency_probe_20260925`).** Per-token NLL on final checkpoints (seed 260925), binned by target unigram frequency:
- SOAP's gain leans toward rarer targets (−0.032 to −0.038 for frequencies 1e-6–1e-4 vs −0.011 above 1e-2), the Adam-on-heavy-tails pattern.
- PD's gain is broad (−0.016 to −0.020 on frequent targets as well).
- This is a further sign that the two mechanisms are distinct and complementary.

### Wave 21: curvature-matched per-kind α (2026-09-25 ~16:55 UTC)

**Decision argument.** Across all 24 probed matrices, the median curvature exponent is γ = 0.52 (α = 0.26, matching the swept ¼). γ varies systematically by kind:
- q 0.43, k 0.33;
- v 0.72, o 0.61, up 0.67;
- down 0.43.

If the data norm's exponent should track curvature, α per kind = γ/2 should match or beat the uniform ¼. That would be a principled refinement with no new free parameter, since α comes from a measurement rather than a sweep.

**Arms.** PD with α = {q 0.215, k 0.167, v 0.362, o 0.305, up 0.335, down 0.214} at LR 0.01, seeds 260925 and 260926 on A6000. These are paired with uniform PD (−0.0172, −0.0181 vs tuned Muon) and with Muon@0.007.

**Prediction (P20).** The per-kind α improves on uniform ¼ by ≥ 0.002 on both seeds. If the difference is within ±0.002, the uniform ¼ is adequate and the curvature link holds only at the level of the median.

**S∘PD LR bracket complete (17:16 UTC, seed 260925, vs tuned Muon).**

| S∘PD LR | Δ vs tuned Muon |
|---|---|
| 0.007 | −0.0273 |
| **0.01** | **−0.0284** |
| 0.014 | −0.0254 |

0.01 is S∘PD's best LR. With the four fresh confirmation pairs on three GPU types (−0.0302, −0.0325, −0.0302, −0.0336; mean −0.0316), **S∘PD also meets every clause of the claim rule against tuned Muon.**

**Per-kind α, first seed (17:28 UTC).** At seed 260925, PDk (α = γ/2 per kind) is −0.0179 vs tuned Muon, against −0.0172 for the uniform ¼. The difference of −0.0007 is inside P20's ±0.002 band. On this seed, per-kind α adds nothing beyond the median-level curvature match, so the uniform α = ¼ is adequate. Seed 260926 is running.

### Width-768 bracket results (18:30 UTC, seed 260925, A6000, 2562 steps)

| method | LR | final val NLL |
|---|---|---|
| Muon | 0.005 | 3.47205 |
| Muon | **0.007** | **3.46974** |
| Muon | 0.01 | 3.47610 |
| PD α = ¼ | 0.007 | 3.45742 |
| PD α = ¼ | **0.01** | **3.45549** |
| PD α = ¼ | 0.014 | 3.45894 |

Both best LRs are interior and equal to the carried-over ones (Muon 0.007, PD 0.01), so the fresh pairs count as designed.

**Selection-seed best-vs-best: −0.0142.** At width 512 the same seed gave −0.0172, so about 83% of the gain is kept at width 768 with α = ¼ unchanged. The verdict waits on the fresh pairs (seed 260924 on L40S, seed 260926 on RTX 6000 Ada) under the criteria fixed at launch.

**Per-kind α final (18:41 UTC).** PDk vs tuned Muon: −0.0179 (seed 260925) and −0.0162 (seed 260926), mean −0.0171. Uniform ¼ on the same seeds gave −0.0172 and −0.0181. **P20 fails.** Setting α = γ/2 per layer kind is no better than the uniform ¼, so the curvature link holds at the level of the median exponent, and the uniform α = ¼ stays the method.

**Width-768 fresh pair 1 (18:44 UTC).** L40S, seed 260924: PD@0.01 − Muon@0.007 = **−0.0118**. This is just short of the "supports" bar (every fresh Δ ≤ −0.012) and well inside the "not weakened" region (mean better than −0.009, no seed ≥ 0). Pair 2 (RTX 6000 Ada, seed 260926) finishes around 19:25.

Because pair 1 alone already sits in the ambiguous band, the rule's "add a seed" step is started now rather than after pair 2: Muon@0.007 and PD@0.01 at seed 260927 on A6000 (wave 22). The verdict will use all fresh pairs under the unchanged criteria.

**γ at width 768 (19:05 UTC).** Median 0.44, so α = γ/2 = 0.22; at width 512 it was 0.26. Curvature-matched α stays close to the ¼ carried over, which meets the review's condition that α from γ should not drift far from ¼ (`logs/muon_spectra/gamma_probe_20260925/README.md`).

**Width-768 fresh pair 2 (19:18 UTC).** RTX 6000 Ada, seed 260926: PD@0.01 − Muon@0.007 = **−0.0169**.

| seed | GPU | role | Δ |
|---|---|---|---|
| 260924 | L40S | fresh | −0.0118 |
| 260926 | RTX 6000 Ada | fresh | −0.0169 |
| 260925 | A6000 | selection | −0.0142 |

- Fresh mean −0.0144, about 75% of the width-512 fresh mean (−0.0193).
- All pairs: n = 3, mean −0.0143, SD 0.0025.

**Reading under the launch criteria.**
- No sign of weakening: the mean is well below −0.009 and every Δ < 0.
- The strict "supports" bar (every fresh Δ ≤ −0.012) is missed by one seed, by 0.0002, so the letter of the rule gives "ambiguous; add a seed".
- That seed (260927, A6000) is running and finishes around 21:00 UTC.

### Wave 23: the damped full data norm (2026-09-25 ~19:45 UTC)

**Decision argument.** Plotted log-log (FINDINGS figure, panel c), the per-direction curvature is closer to **linear in λ with a floor** than to a pure power law. The model curvature ∝ λ_j + b fits better than λ_j^γ on 19 of 24 matrices (log error), with median b ≈ 0.10 × mean λ.

Under that model, the curvature-matched norm is the full data norm with Levenberg–Marquardt-style damping: α = ½ with R = (C/mean λ + d·I)^(−½), d ≈ 0.1. The response ‖w_j‖ ∝ (λ_j + d)^(−½) makes the curvature cost equal across directions exactly.

The α = ¼ power-law form is the geometric compromise the sweep found. The damped form is a second principled parametrization with its constant taken from a measurement. (The earlier α = ½ arm used d = 1e-3, which is essentially undamped, and gained nothing.)

**Arms** (A6000, LR 0.01, paired with PD α = ¼ and tuned Muon):
- α = ½, d = 0.1 at seeds 260925 and 260926;
- α = ½, d = 0.3 at seed 260925.

**Prediction (P21).** The damped full data norm with d = 0.1 is within ±0.003 of PD α = ¼ (−0.0172 and −0.0181) or better. If it is clearly worse, the power-law form is the better description of what helps.

### Wave 24: the stacked method and SOAP-Muon at width 768 (2026-09-25 ~20:10 UTC)

The user pointed out that both allocations (priv-g14 and g20, 8 GPUs) were idle. Their most useful work is to extend the best method's evidence to width 768, where PD holds (−0.0142, −0.0118, −0.0169; a third fresh seed is running on A6000).

**Arms.** Each pairs with the tuned-Muon width-768 run already done on the same node and seed:
- S∘PD α = ¼ @0.01;
- the SOAP-Muon core @0.007.

Both use their width-512 best LRs: seed 260924 on priv-g14 (L40S) and seed 260926 on g20 (RTX 6000 Ada).

This is a first look, not a claim. A claim for these methods at width 768 would need their own LR brackets (the μP concern raised in the review).

**Prediction (P22).** S∘PD beats tuned Muon by more than PD does at width 768 (PD −0.0118 / −0.0169 on these seeds), and beats the SOAP-Muon core on both seeds.

**Wave-23 results (20:23 UTC): P21 fails.** Damped full data norm vs tuned Muon:
- α = ½, d = 0.1: −0.0017 (seed 260925) and +0.0071 at step 1250 (seed 260926; final pending).
- α = ½, d = 0.3: −0.0050 (seed 260925).

PD α = ¼ on the same seeds gave −0.0172 and −0.0181. The linear-plus-floor curvature model fits the probe data better in-sample, but the norm it implies does not work. The α = ¼ power form is essential.

**Consequence for the "principled" claim.**
- α = γ/2 from a power-law fit of curvature vs input second moment matches the swept optimum at both widths (0.26 and 0.22).
- A competing curvature model fits the same probe data at least as well, yet implies a norm that fails.
- So the curvature measurement is consistent with α = ¼ but does not uniquely derive it. Noise, which the curvature argument ignores (whitening amplifies low-variance, noise-dominated input directions), plausibly favours the milder exponent.
- The defensible statement: PD is steepest descent under the partial data norm with an empirically chosen α = ¼ that matches the measured power-law curvature exponent. It is not derived from curvature alone.
- Damped full data norm, final: d = 0.1 gives −0.0017 and −0.0066 (mean −0.0042, n = 2); d = 0.3 gives −0.0050. This confirms P21's failure.
- S∘PD at width 768, step 1000: −0.0375 (L40S) and −0.0350 (RTX 6000 Ada) vs tuned Muon. PD at the same step was ≈ −0.015.

### Head-to-head with Newton-Muon and PMuon on H100 (2026-09-25 ~21:05 UTC)

User request: replay PD on Google Cloud, port PMuon and Newton-Muon, and run them on the cloud once the code is complete. Separate cohort `logs/muon_spectra/h2h_h100_20260925` (Spot 4×H100, `cloud/`); it does not use the TTIC allocations.

**Decision argument.**
- **Missing premise.** PD has been compared only with tuned Muon. Two published relatives in modded-nanogpt Track 3 also precondition Muon's input side, and neither post-multiplies:
  - Newton-Muon (record #15; Du and Su, arXiv 2604.01472): msgn(G (ZZᵀ)⁻¹), from the activation second moment with a damped full inverse, applied to the gradient before momentum.
  - PMuon (record #18): polar(A^−γ M C^−γ), with streaming second moments on both sides, γ = 0.3.
  - Our closest before-NS-only analogues kept about a third of PD's gain over tuned Muon (seed 260925: PDin −0.0064, S_rightcol −0.0062, PD −0.0172).
- **Simplest competing explanation.** Their exact recipes (damping 0.2, full inverse, two-sided powers) recover what our analogues missed. PD's advantage over published methods would then be small.
- **Port** (`muon.py` options `newton_muon` and `pmuon`; only the preconditioners, with the records' constants):
  - Newton-Muon: damping 0.2 × mean eigenvalue; refresh every 64 steps; EMA weight 0.05 starting from 0.001 I; MLP down as 4 diagonal blocks; the gradient is preconditioned before momentum; statistics are averaged over ranks and rank 0's inverses are shared.
  - Newton-Muon statistics come from our `StatLinear` EMA (every 32nd position, decay 0.998 per forward) rather than the reference's full-token accumulation on the refresh step. The 20-refresh EMA dominates either way.
  - PMuon: γ = 0.3, β = 0.95, one QR iteration per step. Its statistics come from the NS input, as in the record's code (its README says gradient), at EMA-momentum scale.
  - Shared by every arm: our Muon (plain momentum 0.95, paper5 NS, shape scaling, decay 0.01, auxiliary AdamW 0.002). Not adopted: their 12-step cubic NS, Nesterov, per-group LR and decay, and their auxiliary settings.
- **Qualification before any GPU use.**
  - Six new unit tests (`test_baselines.py`) against reference code copied verbatim from the record logs: damped inverse, block statistics, preconditioned momentum at the first refresh, the PMuon power and first-step update, determinism and validation. All 67 tests pass.
  - A 2-rank CPU run with replica audits passes for both methods, with Newton-Muon refreshing every step.
  - `run_arm.py` engineering overrides: Newton-Muon refreshes every step in the tiny stage and every 2 steps in the 5-step qualification, so the audited gates exercise the preconditioner.
- **Design** (H100 only, width 512, 1× horizon, paired by seed):
  - LR brackets on seed 260924: Newton-Muon and PMuon at {0.005, 0.007, 0.01}. If the best LR is at an edge, extend one step.
  - Seeds 260925 and 260926: Newton-Muon and PMuon at their bracketed LR; PD@0.01 and Muon@0.007 at their carried-over LRs.
  - Seed-260924 Muon and PD on H100 come from `gcp_validation_20260925` (wave-9 code; the Muon and PD paths are unchanged).
- **Predeclared reading**, per baseline X over the 3 paired seeds: "PD beats X" if every Δ = PD − X < 0, the mean is ≤ −0.005 and the one-sided paired t has p < 0.05. Otherwise "not shown". Each baseline is also reported against tuned Muon.
  - Limits: one GPU type. X's LR is chosen on seed 260924, which favours X on that seed.
- **Prediction P22.** Mean PD − Newton-Muon ≤ −0.008 and mean PD − PMuon ≤ −0.005, because neither post-multiplies. If either baseline comes within 0.003 of PD, PD's edge over published relatives is not established at this scale.
- **Cost.** 14 runs of about 13 minutes on Spot 4×H100 (about $26/h), roughly $80–100. VMs are deleted when their queue ends.

**Amendment (21:20 UTC, user direction): single runs.** No LR brackets and no extra seeds for now.
- One run each on seed 260924 (H100): Newton-Muon at LR 0.005 (already running; 0.71× Muon's LR, within its record's own tuned ratio) and PMuon at 0.007 (Muon's LR, as its record did).
- The predeclared 3-seed reading is suspended. These are single-seed screens, reported as differences without a claim. The other arms were cancelled before start (`h2h_h100_20260925/cancelled_before_start/`).
- **PD replay on H100** (seed 260924, wave-9 code): 3.68530 vs tuned Muon 3.70386, Δ = −0.0186 (TTIC pairs on the same seed: −0.0198 RTX 6000 Ada, −0.0186 L40S).
- Newton-Muon's first GPU run passed the tiny and full-size replica audits, with the preconditioner refreshing every 1 and 2 steps respectively.

### Width-768 verdict (21:26 UTC)

PD α = ¼ @0.01 vs tuned Muon@0.007 at width 768 (134M params, 2562 steps). Both LRs are re-bracketed and interior, and match the carried-over values.

| seed | GPU | role | Δ |
|---|---|---|---|
| 260924 | L40S | fresh | −0.0118 |
| 260926 | RTX 6000 Ada | fresh | −0.0169 |
| 260927 | A6000 | fresh | −0.0130 |
| 260925 | A6000 | selection | −0.0142 |

- **Fresh:** mean −0.0139, SD 0.0027, one-sided paired t ≈ −9.0 (df 2), p ≈ 0.006.
- **All pairs:** n = 4, mean −0.0140.

**Reading under the launch criteria.**
- Not weakened: every Δ < 0 and the mean is well below −0.009.
- The strict per-seed "supports" bar (every fresh Δ ≤ −0.012) is met by 2 of 3 fresh seeds; the third misses it by 0.0002.
- The rule's "add a seed" step was taken (seed 260927: −0.0130).

**Conclusion.** PD holds at 1.5× width with about 72% of its width-512 fresh-seed gain (−0.0139 vs −0.0193), with α and both LRs unchanged. The small erosion is comparable to the 2× horizon's (−0.0154 vs −0.0188).

**S∘PD at width 768** (seed 260924, L40S): −0.0241 vs tuned Muon, against −0.0302 at width 512 on the same seed and GPU (80% kept). PD on this seed at width 768 gave −0.0118. The RTX 6000 Ada seed and the SOAP-Muon comparisons are running.

**S∘PD at width 768, both seeds (21:36 UTC).** Vs tuned Muon:
- −0.0241 (seed 260924, L40S) and −0.0270 (seed 260926, RTX 6000 Ada); mean −0.0256, about 82% of the width-512 mean (−0.0313).
- PD on the same seeds: −0.0118 and −0.0169. The stacked method adds 0.010–0.012 on top of PD at width 768.
- The SOAP-Muon core at width 768 on these seeds is running (P22).

**Single-seed screen result (21:40 UTC; H100, seed 260924).** Final val NLL: tuned Muon 3.70386, PD 3.68530, Newton-Muon@0.005 3.70296, PMuon@0.007 3.70167.
- Vs tuned Muon: Newton-Muon −0.0009, PMuon −0.0022, PD −0.0186.
- PD beats Newton-Muon by 0.0177 and PMuon by 0.0164 on this seed.
- One seed, with untuned baseline LRs, so no claim. The direction matches P22, and both baselines sit near tuned Muon, as their Track-3 step counts did.

**TTIC runs at LR 0.01 (21:50 UTC, user request).** Newton-Muon@0.01 and PMuon@0.01, seed 260925, on the 4×A6000 `gpu` partition (`logs/muon_spectra/h2h_ttic_20260925`, jobs 2622039 and 2622040), because the g20 and priv-g14 allocations are busy with wave 24.
- They pair with A6000 tuned Muon@0.007 (3.70630), Muon@0.01 (3.71457) and PD@0.01 (3.68909).
- Single-seed screens.

**User stop (21:48 UTC).** At the user's request, the width-768 SOAP-Muon core run at seed 260926 on g20 was stopped at about step 230 of 2562, by cancelling only its job step (2618555.62). It is recorded in `CANCELLED_BY_USER.txt` in that arm's directory.

P22's SOAP-Muon comparison therefore rests on the priv-g14 seed (260924, L40S) alone, which finishes around 22:35 UTC.

### Track 3 benchmark port (branch of #36) (2026-09-25 ~22:05 UTC)

User request: run PD in the exact Track 3 speedrun setup on g20, branching from result #36. The workspace is `track3/` (see its README).

**Decision argument.**
- **Missing premise.** All PD evidence so far is from one local setup: bias-free, learned positions, GELU, 1M-token batch. Track 3 has biases, RoPE, ReLU², a 0.5M-token batch and a heavily tuned baseline, and it scores steps to 3.28 rather than loss at fixed tokens. Gains for similar methods shrank sharply there: Newton-Muon was 6% in its paper and ~1.5% in Track 3.
- **Simplest competing explanation.** PD's gain is specific to our setup: batch size, architecture, or a baseline tuned only in LR.
- **Comparison.** Change only the hidden-matrix update of #36's script:
  - polar(M R) R, R = (C/mean λ + 1e-3 I)^−¼, rescaled to Muon's Frobenius norm;
  - C from traced forward hooks: an EMA of every 16th position with decay 0.938 per step (our StatLinear's per-step decay), refreshed every 10 steps by the owner rank;
  - #36's Nesterov momentum, 12-step NS, LR 0.025 and WD 0.05 are kept for the first run.
  - Control: #36 unchanged on the same GPUs (4×RTX 6000 Ada).
  - Reference: the mean of #36's 10 published H100 logs (3.27866 at step 3250).
- **Outcomes that change the next decision.**
  - PD reaches 3.28 ≥ 100 steps earlier than the same-hardware control: tune LR and WD at shorter horizons, then run multi-seed at the reduced step count for a per-optimizer entry. PMuon (#18, 3225) and Newton-Muon (#15, 3275) are the relatives to beat.
  - Within ±50 steps: PD's advantage does not transfer at #36's hyperparameters; try PD's LR ratio (×1.4, as in our setup) before concluding.
- **Qualification.** A 40-step smoke test on g20: identical initial weights (validation 10.82584), step-40 validation 5.63737 vs 5.66977, and hook EMA weight 0.9227 (the expected value) inside the compiled model.
- **Cost.** About 85 minutes per run on g20. Track 3 runs are not reseeded; run-to-run variation (σ ≈ 0.0013) comes from GPU nondeterminism.

**Width-768 SOAP-Muon core (22:34 UTC; seed 260924, L40S, LR 0.007).** Final 3.45386, **−0.0183** vs tuned Muon (3.47212). Same seed and GPU:
- PD −0.0118;
- **S∘PD −0.0241**, which beats the SOAP-Muon core by 0.0058.

At width 512 this seed gave S −0.0216 and S∘PD −0.0302. **P22 holds on the available seed:** S∘PD beats both tuned Muon and the SOAP-Muon core at width 768. The second width-768 SOAP-Muon seed was stopped at the user's request.

**TTIC LR-0.01 screens, final (23:00 UTC; A6000, seed 260925).** Newton-Muon@0.01 3.70884, PMuon@0.01 3.71021.
- Vs tuned Muon@0.007: +0.0025 and +0.0039. Vs Muon@0.01: −0.0057 and −0.0044. PD@0.01 beats them by 0.0197 and 0.0211.
- Across all four baseline screens (H100 seed 260924 at LR 0.005/0.007; A6000 seed 260925 at 0.01), both baselines land within ±0.004 of tuned Muon, while PD is −0.017 to −0.019.
- The first A6000 attempts stopped at the controller's time gate: CPU SVD with the 2 CPUs of gpu-partition jobs (`FAILURE.md`). The reruns used `svd_device: cuda`.

### Correction: the curvature-exponent probe was resolution-limited (2026-09-25 ~23:15 UTC)

While writing the report, the paper subagent found that the finite-difference second differences in `gamma_probe.py` are integer and half-integer multiples of one float32 spacing of the loss (2^−22 ≈ 2.4e-7). I verified this on four matrices: 35 of 48 second differences lie within 4 spacings of zero.

Only the top ~8 input eigendirections per matrix are resolved.
- **Resolved points only:** median slope γ ≈ 0.69–1.00 at width 512 and 0.67–0.96 at width 768.
- **Earlier reported values:** 0.52 and 0.44.
- **Consequences:**
  - The earlier statement that "α = γ/2 matches the swept ¼" is withdrawn. It relied on unresolved low-variance points.
  - The "floor" in the linear-plus-floor model was the same artifact. Wave 23's damped full data norm stands as an empirical null result, but its motivation from a measured floor does not.
  - The curvature probe's "7–276× stiffer" x̄ ratios share this weakness. They are order-of-magnitude estimates.
- **What remains:** α = ¼ was chosen empirically (sweep ⅛/¼/⅜/½) and carried unchanged to width 768. Where curvature is resolved, it grows close to linearly with input variance.

An exact Hessian-vector-product re-measurement (`gamma_probe_hvp.py`, no cancellation) is running (dev-gpu, 23:10 UTC). Its result will be recorded here. The original `gamma_probe.py` outputs are kept unchanged.

**Other number corrections from the report pass:**
- The stacking margin over the SOAP-Muon core is 0.006–0.009, not 0.007–0.009.
- The mean-input share of E‖x‖² is 5–44% under Muon in the paper's (frontier-norm) architecture (17–41% for RMSNorm-fed inputs) and up to 74% under SOAP-Muon. The "35–50%" and "10–70%" figures came from the earlier LayerNorm architecture or mixed both.
- Outlier attribution, O 0.92–0.98, V 0.75–0.81 and sink 0.24–0.28, comes from the earlier architecture. The frontier-norm values are O 0.96, down 0.88, V 0.74 and sink 0.35.
- Token saving: 9% fewer tokens.

**Exact curvature exponent (23:35 UTC).** The Hessian-vector-product probe (`gamma_probe_hvp.py`, all 288 values resolved) gives **median γ = 0.99** at width 512: q 1.00, k 0.78, v 1.02, o 1.02, up 0.92, down 0.96.
- Curvature along input directions is proportional to their second moment.
- Curvature matching would therefore pick α ≈ ½, which fails in training (+0.0068 vs tuned Muon). The swept optimum is α = ¼: the square root of the curvature-matched preconditioner.
- The previous "α = γ/2 ≈ ¼" account is replaced by this statement. The noise-based rationale for the square root is a hypothesis, not tested here.
- **Width 768, exact HVP:** median γ = 0.985 (q 0.99, k 0.74, v 0.94, o 1.03, up 0.98, down 0.99). Curvature ∝ input variance at both widths, and the chosen α = ¼ is the square root of the curvature-matched ½ at both.

### Why PD's gain shrinks in Track 3, and improving it (user goal, 2026-09-25 ~19:00 CDT)

User goal: use g20, priv-g14 and the gpu partition (no VMs) to improve PD over the Muon baseline as much as possible, working out how PD works and why Track 3 differs from our setup. Use few seeds until the explanations run out.

**Observations so far (Track 3, single runs).**
- PD at #36's settings (LR 0.025, WD 0.05): 3.27710 at step 3250, vs 3.27996 for the same-code #36 control on L40S and 3.27866 for the published 10-run H100 mean. PD first drops below 3.28 at step 3200, ~50 steps early.
- The lead is −0.051 at step 250 and −0.019 at 500. It is −0.006 by step 1000, before #36's decay starts at 975, then ~0 from 1250 until the last 10%.
- At matched tokens (0.52B) our setup has −0.038, so the collapse happens during the constant-LR phase.
- PD@0.035 trails by ~0.01–0.02 throughout; PD@0.03 sits between.

**Regime differences (per-step shrink lr×wd; weight norms).**
- Ours: 7e-5; weight norms grow throughout training (q 10 → 30–41) and never reach equilibrium.
- #36: 1.25e-3; norms equilibrate within ~800 steps, so the relative step size is set by lr and wd together.
- PD's weights grow 12–15% less than Muon's at the same LR in our runs, so at equilibrium PD's relative steps are larger. This is consistent with PD wanting a *lower* LR in Track 3.
- With a preconditioned update, decoupled decay implies a penalty tr(W R^−2 W^T), heavier on high-variance input directions. It is negligible in our weak-decay setup; under #36's decay it may distort the weights on the most-used directions.

**Hypotheses and tests.**
- H-WD: strong decay removes PD's gain.
  - Transfer test: our setup at wd 0.2 (~#36's per-step shrink).
  - Track 3: PD wd 0.025, and PD-geometry decay W ← W − ηλ W R²/mean(eig R²).
- H-LR: PD's optimal LR is lower in the equilibrium regime. Track 3: PD@0.02.
- H-batch, H-bias, H-schedule: one-factor transfer tests in our setup (batch 524,288; Linear biases; cooldown 0.7), each Muon@0.007 vs PD@0.01, seed 260925 on A6000, against the reference pair (−0.0172).
- H-curvature: in Track 3's model, loss curvature may not grow with input variance (γ ≈ 0), which would leave PD nothing to equalize. `track3/gamma_probe_t3.py` measures γ and the input anisotropy on final checkpoints; new Track 3 runs save final weights after training.
- H-noise: at half the batch, amplifying near-empty input directions amplifies noise. Track 3: PD with damping 1e-2 (vs 1e-3), and α = ⅛.

**Readings.**
- A transfer test "kills" the gain if PD − Muon ≥ −0.005 (vs −0.0172).
- A Track 3 variant "helps" if it beats PD's 3.27710 at step 3250 by ≥ 0.002 (~45 steps), about 1.5× the per-run σ of 0.0013. Such a variant then gets a replicate and a matched Muon control before any claim.

### Decision: does decay geometry cause the wd-0.2 flip? (2026-09-25 ~19:50 CDT)

**Evidence.** At wd 0.2, PD@0.01 − Muon@0.007 is +0.0153 at step 1100, vs −0.0172 for the reference pair. The flip is on the same seed and GPU type. The other transfer tests keep most of the gain: bias −0.0192 (step 850), batch 524,288 −0.0093 (step 1450, shrinking).

**Mechanism under test.** PD's update is steepest descent under ‖D R⁻¹‖_op. Decoupled decay then makes the update Frank–Wolfe on {‖W C^¼‖_op ≤ 1/λ}, which is tighter along high-variance inputs. Muon's constraint set is {‖W‖_op ≤ 1/λ}. Preconditioning the decay the same way as the update undoes this: W ← W − ηλ W R^p / mean eig(R^p).
- With aligned updates, p = 1 gives the equilibrium W ∝ polar(G R), so ‖W‖_op ≤ c/λ, as for Muon.
- With random-walk updates, p = 2 equalizes the equilibrium norm per input direction.
- The mean shrink per step equals decoupled decay's in both cases.

**Confound.** PD's LR is 0.01, so its shrink per step at wd 0.2 is 2e-3, vs 1.4e-3 for Muon.

**Arms.** Cohort `wdgeo_20260925`, our setup, seed 260925, A6000, frozen code with the `data_norm_decay` option:
- PD@0.01, wd 0.2, geometry decay p = 2;
- the same with p = 1;
- PD@0.007, wd 0.2, decoupled decay (LR-matched, as in Track 3, where both share LR 0.025).

**Readings** (final step, against M_wd0.2 from `transfer_20260925`):
- A geometry arm "restores" the gain if PD − Muon ≤ −0.010, is "partial" between −0.010 and −0.003, and "fails" if ≥ −0.003.
- If the LR-matched arm alone also reaches ≤ −0.010, the flip is mostly the extra shrink from PD's higher LR, not the geometry.
- The Track 3 counterpart is the p = 2 `pdwd` arm in the half-length A6000 screen. A p = 1 Track 3 arm follows only if p = 1 wins here.

### Decision: PD on the hyperball baseline (#37 MuonH) (2026-09-25 ~19:55 CDT)

**Evidence that the decay equilibrium removes PD's gain.**
- *Our setup at wd 0.2:* PD − Muon went from +0.0159 at step 1000 to +0.0264 at step 1350. The reference pair is −0.0172. The flip grows rather than recovering.
- *Track 3 at #36's settings:* PD's lead is 46 steps at step 500 (~9%, matching our setup's token saving). It is 26 steps at 1000 and ~0 from step 1250, after about one decay time 1/(ηλ) = 800 steps.
- *Track 3, lower decay:* PD with wd 0.025 or LR 0.02 keeps a mid-run lead. Part of that lead is the lower noise floor at a smaller ηλ, and it shrinks in the cooldown.
- *Track 3 history:* Muon needs its strong decay (wd 0.0125 → 0.025 → 0.05 took 3500 → 3375 → 3250 steps, #3/#6/#36). So PD cannot simply use weak decay against #36.

**Alternative regime.** #37 (MuonH, 3250 steps, H100 mean 3.27865 over 10 logs, the same as #36) removes weight decay from the hidden matrices.
- Each step moves W by lr·‖W‖ along the normalized direction, then projects back to the initial Frobenius radius.
- Radial motion is removed, and the only uniform shrink is ~lr²/2 = 1.6e-4 per step. That is 8× weaker than #36's decay, with a time constant (~6000 steps) longer than the run.
- So PD's direction can be tested in a tuned Track 3 baseline without a decay equilibrium.

**Comparison.**
- `track3/train_gpt_pdh.py` is #37's script (byte-identical in all 10 published logs, sha f612c7c3) with MuonH's direction polar(M) replaced by PD's polar(M R) R. The PD block (hooks, root, α = ¼, damping 1e-3, refresh 10) is copied verbatim from `train_gpt_pd.py`.
- LR 0.018, init, aux Adam and schedules are #37's. The hyperball step uses only the direction, so PD's norm matching does not matter.
- Control: #37 unchanged (plus an analysis-only final save) on the same GPU type (L40S).

**Readings (single runs; per-run σ ≈ 0.0013).**
- PD-H − MuonH ≤ −0.004 at step 3250 (≳ 90 steps): PD keeps its gain without a decay equilibrium. Next: PD-H at shorter step counts and a replicate.
- Within ±0.002: the equilibrium is not the whole story. The alternatives are the 0.5M batch (our transfer test keeps only 34% so far) and the long cooldown (transfer test pending).
- The geometry-decay run (`train_gpt_pd_pdwd.py`, #36 settings, g20) tests the same hypothesis inside #36.

**Transfer test wd 0.2, final (19:52 CDT; seed 260925, A6000).** Muon@0.007 3.70855 and PD@0.01 3.71204: **PD − Muon = +0.0035**, vs −0.0172 at wd 0.01. By the predeclared reading this kills the gain.
- Relative to wd 0.01, Muon moves by only +0.0022, but PD moves by +0.0230. Strong decoupled decay hurts PD, not Muon.
- The cooldown shrank the gap from +0.0264 at step 1350 to the final +0.0035, so mid-run gaps under strong decay are partly noise-floor differences.
- The `wdgeo_20260925` arms separate decay geometry from the extra shrink per step that PD's higher LR brings.

**Track 3 LR/WD results for PD, final at step 3250 (20:01 CDT; single runs):**

| Run | GPU | Final | vs L40S control 3.27996 | vs #36 H100 mean 3.27866 |
|---|---|---|---|---|
| LR 0.025, WD 0.05 | RTX 6000 Ada | 3.27710 | −0.0029 | −0.0016 |
| LR 0.03, WD 0.05 | RTX 6000 Ada | 3.27778 | −0.0022 | −0.0009 |
| LR 0.035, WD 0.05 | L40S | 3.27850 | −0.0015 | −0.0002 |
| LR 0.025, **WD 0.025** | L40S | 3.27800 | −0.0020 | −0.0007 |

- The WD-0.025 run led by 0.028–0.030 from step 750 to 2000, and by 0.022 at 2500, 0.009 at 3000 and 0.002 at the end.
- **Lowering PD's decay does not improve its final.** The mid-run lead at a smaller ηλ is the lower noise floor, which the cooldown removes, so "PD just wants a smaller ηλ" is refuted for the score that matters.
- PD's final is flat to within ~0.0015 across LR 0.025–0.035 and WD 0.025–0.05, about one per-run σ.
- Still open: decay geometry (`train_gpt_pd_pdwd.py`, g20, started 20:00) and PD-H (priv-g14, started 20:01).
- A same-GPU #36 control (g5, RTX 6000 Ada, with final save) started 19:51 for the g20 runs. The #37 control is queued on L40S.
- PD LR 0.02, WD 0.05 (L40S, 20:14 CDT): **3.27663**, −0.0033 vs the L40S control and −0.0020 vs the #36 H100 mean; first below 3.28 at step 3200. Final weights saved.
  - PD's LR curve at WD 0.05 is monotone and shallow: 0.02 → 3.27663, 0.025 → 3.27710, 0.03 → 3.27778, 0.035 → 3.27850.
  - Neighbouring points differ by less than the per-run σ; only the trend across all four is informative.
- **Transfer test, bias (final):** −0.0171 (reference −0.0172). Linear biases do not affect PD's gain.

**Transfer tests complete (20:36 CDT; seed 260925, A6000; PD@0.01 − Muon@0.007, one Track 3 feature at a time; reference −0.0172):**

| Feature | Muon | PD | PD − Muon | Share of gain kept |
|---|---|---|---|---|
| WD 0.2 (Track 3's shrink per step) | 3.70855 | 3.71204 | +0.0035 | −0.20 (killed) |
| Batch 524,288 (same tokens, 2938 steps) | 3.67603 | 3.66638 | −0.0096 | 0.56 |
| Linear biases | 3.70744 | 3.69036 | −0.0171 | 0.99 |
| Cooldown 0.7 | 3.71922 | 3.68982 | −0.0294 | 1.71 |

- **The long cooldown helps PD.** Muon loses 0.013 with it and PD nothing, so the schedule is not why the gain shrinks in Track 3.
- **The half batch keeps about half the gain.** As on the 2× horizon, the gap narrows in the stable phase (−0.0045 at step 2500) and widens again in the cooldown.
- **Strong decoupled decay is the one feature that removes the gain.** H-WD is supported, H-schedule rejected, H-bias rejected, and H-batch is partial.

### Mechanism hypothesis: decay equilibrium equalizes relative steps per input direction (20:45 CDT, before the finals)

**The argument.** Split W along input eigendirections q_j, and let the update have size u_j along q_j (PD: u_j ∝ r_j = (λ_j/mean λ + d)^−¼, Muon: u_j uniform). Under decoupled decay the equilibrium size w_j of W q_j follows from balancing growth and shrink:
- random-walk updates: w_j² ≈ η u_j² / (2λ);
- aligned updates: w_j ≈ u_j / λ.

In both regimes the relative step u_j η / w_j is independent of u_j: √(2ηλ) for random walks and ηλ for aligned updates. So once the weights equilibrate (1/(ηλ) = 800 steps in #36), any input-side reallocation of step size is undone. The weights shrink exactly where the preconditioner steps less. This is the per-direction version of the rotational-equilibrium result (per-neuron angular updates √(2ηλ) regardless of gradient size).

It explains four observations:
- PD ≈ Muon mid-run in #36 after ~1000 steps;
- the wd-0.2 flip in our setup;
- the absence of the effect under our weak decay (norms never equilibrate);
- small Track 3 gains for the other input-side preconditioners (Newton-Muon, PMuon).

**Geometry decay** multiplies the shrink by R̂^p. Relative steps along q_j then become ∝ r_j^{p/2} with random-walk updates (p = 2 restores PD's r_j exactly) and ∝ r_j^p with aligned updates.

**Competing explanation (noise floor).** Geometry decay also shrinks the most-used directions less, which lowers their relative step and so their mid-run noise floor, as lower WD did (PD wd 0.025: +185 steps mid-run, +45 at the end). Under this explanation the lead vanishes in the cooldown.

**Predictions, fixed now:**
- Mechanism: Track 3 `pdwd` final ≤ 3.2751, i.e. ≥ 0.002 better than decoupled PD's 3.27710 on the same GPU. Our-setup geo2 at wd 0.2: PD − Muon ≤ −0.010 at the final step.
- Noise floor only: `pdwd` final within ±0.0015 of 3.27710, and geo2 within ±0.004 of decoupled PD's +0.0035.

**H-curvature rejected (20:56 CDT).** Exact-HVP probe (`track3/gamma_probe_t3.py`, converted from finite differences before any result) on the final Track 3 model of PD LR 0.02 (`track3/gamma_t3.json`):
- **Curvature exponent:** all 240 curvatures resolved and positive. Median γ = **1.02** (q 1.03, v 1.21, o 1.05, up 0.99, down 0.96), the same as our model's 0.99. Curvature along input directions ∝ input variance holds in Track 3's architecture too.
- **Input anisotropy:**
  - top/median eigenvalue 470–2700 for most inputs, and 45,000 for block-4 down;
  - effective rank 18–36 of 768 for most inputs (2 of 3072 for block-4 down);
  - mean-direction share 1–41%.
- PD therefore has as much to act on as in our setup. The lost gain is not a property of the architecture.
- Dry run on random weights (median γ 1.04) is kept in the session scratchpad only.

**Mechanism test in our setup: geometry decay restores the gain (21:04 CDT; seed 260925, A6000, wd 0.2).**
- PD@0.01 with geometry decay p = 2: **3.68973**, so **PD − Muon@0.007 (wd 0.2) = −0.0188**. Decoupled-decay PD at the same wd 0.2 gave +0.0035; the wd-0.01 reference is −0.0172.
- Trajectory of the gap (geo2 / decoupled): −0.049 / −0.024 at step 500, −0.030 / +0.016 at 1000, −0.032 / +0.023 at 1250, −0.019 / +0.004 at the end.
- The cooldown narrows the gap but keeps all of the reference gain. The noise-floor-only prediction (within ±0.004 of +0.0035) is rejected, and **the mechanism prediction (≤ −0.010) holds**.
- With geometry decay, PD at wd 0.2 (3.68973) equals PD at wd 0.01 (3.68909). PD becomes as insensitive to the decay strength as Muon (3.70855 vs 3.70630).
- The p = 1 and LR-matched arms are running.

**Track 3 side (so far).**
- **Half length** (1625 steps, A6000; constant-LR phase 488 steps, shorter than one decay time): Muon 3.40488, PD 3.39544 (−0.0094), PD + geometry decay 3.39475 (−0.0101).
  - Geometry decay's mid-run lead (−0.027 at step 750) mostly vanished in the cooldown.
  - This regime barely equilibrates, which is also why plain PD keeps 3× its full-length gain there. So it does not test the mechanism.
- **PD-H** (#37 MuonH, no weight decay; L40S): **3.27816**, −0.0005 vs the #37 H100 mean.
  - Its lead peaked at +42 steps at step 1000, then faded to +12 at 3000. The same-GPU control finishes ~21:20.
  - Without weight decay the gain also fades, so something besides the decay equilibrium limits PD in Track 3. The batch is the leading candidate (our half-batch test keeps 56%).
- Full-length geometry decay (#36, g20) finishes ~21:25. Geometry decay at LR 0.02 started on priv-g14 at 21:04; it pairs with PD LR 0.02 on L40S, 3.27663.

**PD-H final vs its same-GPU control (21:12 CDT; L40S).** #37 control 3.27930 (published H100 mean 3.27864); PD-H 3.27816, **−0.0011 (~25 steps)**. The PD-H decision's "within ±0.002" branch applies.
- PD-H's lead in steps: +5–7 up to step 750 (plain PD over #36: +35–46), +42 at 1000, then +20–25 to the end.
- On the hyperball, PD gains less than in #36. So removing weight decay does not by itself restore the gain in Track 3.
- One difference stands out. #36 zero-initializes the output projections and its weights grow early; MuonH holds every matrix at its initial Frobenius norm from step 0.
- **Hypothesis (growth/constraint):** PD's per-direction shaping pays off while weights grow freely (our weak-decay setup, #36's first ~500 steps). In a norm-constrained regime, the constraint set decides where the weights settle:
  - under decoupled decay, PD's set {‖W C^¼‖_op ≤ 1/λ} caps the high-variance directions;
  - the hyperball sets the same Frobenius sphere for both methods.
  - Geometry decay (p = 1 aligned-regime form) gives PD Muon's constraint set, and in our setup it restored the full gain.
- **Checked and rejected: PD wastes its Frobenius budget on near-null inputs in Track 3.** From the PD LR 0.02 final statistics, for isotropic momentum, where the share along direction j is r_j²/Σr²:
  - 99% of the input variance spans 67–90% of the dimensions (median 85%), despite participation ratios of ~20;
  - PD puts a median 73% of its update energy there (Muon 85%), i.e. a per-direction step of 0.93× Muon's.
  - Only the few top directions get the intended ~0.3× step. Budget spent on unused inputs is not what limits PD-H.

### Track 3: PD-geometry decay beats PD and the same-GPU control (21:25 CDT; single runs)

`train_gpt_pd_pdwd.py` (#36 settings, W ← W − ηλ W R²/mean eig(R²), RTX 6000 Ada g20): **3.27411** at step 3250, first below 3.28 at step 3150.

| Comparison (RTX 6000 Ada unless noted) | Final | pdwd − it |
|---|---|---|
| #36 control, same GPU type (g5) | 3.28209 | **−0.0080 (~180 steps)** |
| PD, decoupled decay (g20) | 3.27710 | **−0.0030** |
| #36 H100 mean (n = 10) | 3.27866 | −0.0046 |
| #36 control, L40S | 3.27996 | −0.0059 |

- **Predeclared mechanism prediction (≤ 3.2751) holds**, as it did in our setup (wd 0.2: −0.0188 vs Muon).
- Step lead vs the same-GPU control: +43 (250), +111 (750), +163 (1250), +206 (1500), +194 (2000), +148 (2500), +93 (3000).
- The lead shrinks in the cooldown but does not vanish: loss gap −0.035 mid-run, −0.008 final. This differs from lower WD, where the gap went from −0.030 to −0.002.
- **The two same-code #36 controls differ by 0.0021** (L40S 3.27996, RTX 6000 Ada 3.28209; the latter is 2.6σ above the H100 mean). Single-run differences carry ~±0.002.
- Against the RTX 6000 Ada control, plain PD is −0.0050, twice the L40S-based −0.0029 used above.

**Next (launched 21:26 CDT):**
- p = 1 geometry (`train_gpt_pd_pdwd1.py`, g20);
- a replicate of p = 2 (g5, RTX 6000 Ada);
- a second #36 control on L40S (g21), which also pairs with pdwd LR 0.02 running on priv-g14 (+173 steps at step 1000).
- A claim needs repeats on matched hardware: Track 3 σ ≈ 0.0013 per run, and our same-code controls differ by 0.002.

**Weight-share probe confirms the mechanism (21:50 CDT; `track3/weight_share_t3.py`, CPU; final Track 3 checkpoints).** Share of ‖W‖²_F along the top 64 input eigendirections, with C measured on each model's own activations (32 held-out sequences); mean over the 12 blocks. Uniform spread gives 0.083 (0.021 for down).

| Kind | Muon (#36 control) | PD, decoupled (LR 0.02) | PD + geometry decay | MuonH (#37) | PD-H |
|---|---|---|---|---|---|
| q | 0.127 | **0.053** | 0.125 | 0.130 | **0.055** |
| k | 0.118 | **0.049** | 0.118 | 0.120 | **0.051** |
| v | 0.094 | **0.051** | 0.101 | 0.088 | **0.047** |
| o | 0.133 | **0.073** | 0.131 | 0.138 | **0.085** |
| up | 0.096 | **0.045** | 0.106 | 0.097 | **0.043** |
| down | 0.033 | **0.011** | 0.029 | 0.034 | **0.012** |

- Along the top 8 (top 1) directions, q gives 0.018 (0.0030) for Muon, 0.005 (0.0006) for PD and 0.016 (0.0023) for geometry decay.
- **Decoupled-decay PD's weights are depleted 2–3× along the most-used inputs (up to 5× on the top direction), below even a uniform spread. Geometry decay restores Muon's profile.**
- **PD-H is depleted the same way.** The hyperball's Frobenius renormalization acts as a uniform decay: new mass goes where PD steps more and is taken from where it steps less. The earlier estimate that the renormalization is too slow to matter was wrong. This explains why PD-H gained only 0.001.

**Our-setup geometry exponent (21:45 CDT; wd 0.2, seed 260925, A6000).** PD − Muon:

| Decay | PD − Muon |
|---|---|
| p = 2 | −0.0188 |
| p = 1 | **−0.0142** |
| Decoupled | +0.0035 |

The ordering matches the random-walk equilibrium: relative steps ∝ r_j^{p/2}, so p = 2 restores PD's full reallocation and p = 1 half of it. The LR-matched decoupled arm (PD@0.007, Muon's shrink per step) is at +0.0038 at step 1200, so PD's higher LR is not the cause; its final follows.
- **LR-matched decoupled arm, final (21:56 CDT).** PD@0.007 at wd 0.2 (shrink per step 1.4e-3, the same as Muon's) gives **−0.0015** vs Muon. That is above −0.010, so by the predeclared reading **the flip is not the extra shrink of PD's higher LR**. Matching the shrink recovers only 0.005 of the 0.021 loss; the decay geometry recovers all of it.
- **The `wdgeo_20260925` cohort is complete:** decoupled +0.0035, LR-matched decoupled −0.0015, geometry p = 1 −0.0142, geometry p = 2 −0.0188 (reference at wd 0.01: −0.0172).

**Geometry decay at LR 0.02 (22:06 CDT; L40S priv-g14): 3.27584.** −0.0041 vs the L40S control (3.27996); −0.0008 vs plain PD at LR 0.02 on L40S (3.27663). Its lead peaked at +232 steps (step 1500) and fell to +76 by step 3000.

**Where geometry decay stands in Track 3 (two same-hardware pairs):**

| Pair | pdwd − PD | pdwd − control | PD − control |
|---|---|---|---|
| LR 0.025, RTX 6000 Ada | −0.0030 | −0.0080 | −0.0050 |
| LR 0.02, L40S | −0.0008 | −0.0041 | −0.0033 |
| Mean | −0.0019 | −0.0061 | −0.0042 |

- The fix restores the weight structure (weight-share table), and in our setup (10% cooldown) it restores the full gain.
- In Track 3, #36's 70% cooldown removes most of its mid-run lead (−0.033 → −0.008 and −0.039 → −0.004). Its final gain over plain PD is **~0.002 on two single-run pairs, not yet established**.
- Plain PD's final gap vs the control *widens* in the cooldown (about −0.002 → −0.005). Decoupled PD has the higher mid-run noise floor, which the cooldown removes; geometry decay's lower floor leaves less to gain.
- **What remains after the decay fix is consistent with the batch effect.** Our setup at half batch keeps 56% of 9% ≈ 5% of steps; Track 3 PD + geometry decay gives ~130 steps ≈ 4%.
- Geometry decay's advantage over PD looks larger at the higher LR (0.025: −0.0030; 0.02: −0.0008). Its relative step on the most-used directions is ∝ r_j ≈ 0.3, which lowers the noise floor but may slow real progress there.
- **Next:** geometry decay at LR 0.03 (g20, after p = 1); the replicate at LR 0.025 on L40S (priv-g14, started 22:07); the L40S #36 control replicate (g21).

### Independent review and decision: the missing 2×2 cell and the cooldown interaction (22:30 CDT)

An independent reviewer (a read-only subagent) read this protocol from the Track 3 port on, plus the Track 3 README and logs. Main points:

1. **Missing control.** By our own algebra, the equilibrium relative step along q_j is √(2ηλ d_j) (random walk) or ηλ d_j (aligned), where d_j is the decay multiplier. It does not depend on the update's u_j. So **Muon with the R²-shaped decay should inherit geometry decay's equilibrium step allocation**, with a weight profile enriched along the top inputs instead of Muon-like.
   - This is the untested cell of {Muon, PD update} × {decoupled, geometry decay}.
   - If Muon + geometry decay comes within 0.002 of PD + geometry decay, the Track 3 gain is a decay-geometry effect, and PD's update matters only before equilibrium and in the cooldown.
   - Geometry decay on ‖W‖² resembles Adam + L2 (decay through the preconditioner), which runs against the AdamW lesson, so it needs strong evidence.
2. **Our-setup noise-floor null was too narrow.** Our 10% cooldown already removed about 60% of the geometry-minus-decoupled gap (0.055 → 0.022). In Track 3, the 70% cooldown removes 75–95% of every lower-decay mid-run lead: WD 0.025, geometry p = 1 and p = 2.
3. **Controls and multiplicity.**
   - RTX 6000 Ada and L40S are the same AD102 chip, so "same-GPU" pairing buys little; **pool the Ada controls**.
   - The two Ada controls average 3.2810, +0.0024 above the H100 mean, which suggests a hardware or 4-GPU offset.
   - Against the pool: pdwd −0.0069 (LR 0.025) and −0.0052 (LR 0.02); plain PD −0.0039 and −0.0044. So **pdwd − PD ≈ −0.002 ± 0.002**, and the −0.0080 headline used the worst control.
   - With 10+ arms screened at a 0.002 threshold, the best single run should regress toward the mean. Require ≥ 2 runs per surviving arm.
   - Do not screen on mid-run leads or on the half-length screen (it never equilibrates).
4. **Defer S∘PD in Track 3.** PD's margin shrank about 4× in transfer, so S∘PD − SOAP-Muon would be ~0.002 and need ≥ 5 pairs. SOAP-Muon's own step reallocation faces the same decay equilibrium. The scripts (`train_gpt_spd_pdwd.py`, `train_gpt_soap.py`) are written and smoke-tested but not launched.
5. **A lower noise floor needs less annealing,** so if pdwd survives, a shorter cooldown (0.5) is its principled next knob.

**Decision (accepted).** Run the missing cell and the cooldown interaction before any further PD tuning.
- **Our setup** (seed 260925, A6000; cohort `wdgeo2_20260925`):
  - Muon@0.007 + geometry decay p = 2 at wd 0.2 (pre- and post-whitening off, so the update is Muon's direction at √min Frobenius norm and only the decay uses R);
  - wd 0.2 with cooldown 0.7 for Muon@0.007, PD@0.01, PD@0.01 + geometry p = 2 and Muon@0.007 + geometry p = 2.
- **Track 3:** #36 Muon with geometry decay (PD's root used only for the decay) on g20 after the p = 1 run. It replaces geometry decay at LR 0.03.

**Readings (fixed now).**
- Our setup, cd 0.1:
  - "decay effect": MG − M_wd0.2 ≤ −0.010 and |MG − PDgeo| ≤ 0.005;
  - "PD-specific": MG − M_wd0.2 ≥ −0.004.
- Our setup, cd 0.7 at wd 0.2: if PDgeo − M collapses to ≥ −0.006, our setup reproduces Track 3's shrinkage and further PD tuning moves there. If it stays ≤ −0.012, the Track 3 shortfall is something else.
- Track 3: if Muon + geometry decay lands within 0.002 of pdwd's mean (3.2750 over the runs so far), the gain is the decay geometry.

**Third #36 control and Track 3 noise (22:43 CDT).** The second L40S control (g21) finished at **3.27724**.
- The three same-code Ada controls (L40S 3.27996 and 3.27724, RTX 6000 Ada 3.28209) have mean **3.27976** and SD **0.0024**, about twice the H100 σ of 0.0013.
- Their spread is 0.010 at step 250, 0.001 at step 1000, then 0.005 from step 1500 to the end. The runs split into persistently different trajectories between steps 1000 and 1500 (GPU nondeterminism; Track 3 does not reseed), and the offset carries through the cooldown.

**Against the pooled control mean:**

| Run | vs pooled control |
|---|---|
| pdwd LR 0.025 | −0.0057 |
| pdwd LR 0.02 | −0.0039 |
| PD LR 0.02 | −0.0031 |
| PD LR 0.025 | −0.0027 |
| PD LR 0.03 | −0.0020 |
| PD WD 0.025 | −0.0018 |
| PD LR 0.035 | −0.0013 |

- **Plain PD: mean −0.0022 over 5 runs, SE ≈ 0.0018. Not established in Track 3.**
- **pdwd: −0.0048 over 2 runs, SE ≈ 0.0022 (about 2σ).** The L40S replicate at LR 0.025 is running and tracks the first run within ~10 steps of lead through step 1750.
- Resolving a 0.003 difference at 2σ needs ~5 runs per arm with this σ. Mid-run step leads carry the same ±0.0025 trajectory offset (~50–100 steps mid-run).

**Geometry decay p = 1 in Track 3 (22:51 CDT; RTX 6000 Ada g20): 3.27248.**
- vs the pooled control mean 3.27976: **−0.0073**; vs p = 2 at the same LR on the same GPU type: −0.0016 (within σ).
- Our setup preferred p = 2 (−0.0188 vs −0.0142), so the exponent is not resolved.
- **The three geometry-decay finals (3.27248, 3.27411, 3.27584) are the three best of all 11 #36-family finals.** The other eight are 5 plain-PD runs over LR/WD settings and 3 controls.
  - If the 11 were exchangeable, P(top three) = 1/C(11,3) ≈ 0.006.
  - Against the 3 controls alone, P(all three geometry runs below all three controls) = 1/20 = 0.05.
  - The arms differ in LR and p, so this is family-level evidence ("geometry decay helps"), not an estimate for one configuration.
- Mean geometry decay vs pooled controls: **−0.0056**; vs the plain-PD mean (3.27760): −0.0035.
- `screen.py` now references the pooled mean of a family's controls: RTX 6000 Ada and L40S share the AD102 chip, and single controls differ by up to 0.005.
- The MG control (Muon + geometry decay) started on g20 at 22:51.

**Replicate of geometry decay p = 2 at LR 0.025 (23:09 CDT; L40S priv-g14): 3.27418.**
- The first run gave 3.27411 (RTX 6000 Ada). Mean of 2: **3.27415, −0.0056 vs the pooled controls** (SE ≈ 0.0022, about 2.5σ).
- The four geometry-decay finals (3.27248, 3.27411, 3.27418, 3.27584) are all below every control and every plain-PD final.
- **Next, launched 23:10 on priv-g14:** geometry decay with cooldown 0.5 (`train_gpt_pd_pdwd_cd0.5.py`). This is the reviewer's knob for a method with a lower noise floor.
  - #36's 0.7 was tuned for Muon.
  - A gain here needs a Muon cd 0.5 control before it counts as PD-specific.

### 2×2 and cooldown results: the decay "kill" was a short-cooldown noise-floor effect (23:19 CDT)

Our setup at wd 0.2, seed 260925, A6000 (`wdgeo2_20260925/report.py`). Final val NLL minus Muon's at the same cooldown:

| | cooldown 0.1 | cooldown 0.7 |
|---|---|---|
| Muon (decoupled) | 3.70855 | 3.69238 |
| PD (decoupled) | +0.0035 | **−0.0232** |
| PD + geometry decay p = 2 | −0.0188 | −0.0217 |
| Muon + geometry decay (MG) | −0.0039 | +0.0045 |

**Predeclared readings.**
- cd 0.1: MG − Muon = −0.0039 ≥ −0.004 → **PD-specific**, just at the bar. Geometry decay helps PD far more (−0.019) than Muon (−0.004).
- cd 0.7: PDgeo − Muon = −0.0217 ≤ −0.012 → **the Track 3 shortfall is something else**. Our setup with Track 3's decay and cooldown does not reproduce the shrinkage.

**Consequences (these revise the 20:45 and 21:25 entries).**
1. Strong decoupled decay does not remove PD's gain once the cooldown is long: with Track 3's decay and cooldown, decoupled PD keeps it all (−0.023, reference −0.017).
   - The wd-0.2 "kill" at cooldown 0.1 was a noise-floor effect. Under strong decay, decoupled PD's mid-run floor is higher (+0.016 at step 1000), and a 10% cooldown cannot remove it.
   - H-WD, as the explanation of Track 3's shrinkage, is **not supported**.
2. Geometry decay's large benefit in our setup is noise-floor reduction: large at cooldown 0.1, none at 0.7.
   - The weight-share probe (PD's weights are depleted along the most-used inputs; geometry decay restores them) stands as a fact about the weights.
   - Its link to the final score is not established.
3. MG gains nothing at cooldown 0.7 (+0.0045). Decay geometry alone does not help Muon.
4. Track 3 has 4 geometry-decay runs against 5 plain-PD runs and 3 controls: pdwd − PD ≈ −0.0035 (SE ≈ 0.0016), pdwd − control ≈ −0.0056. In our setup at cooldown 0.7, geometry decay adds nothing.
   - Either the Track 3 edge depends on a feature our setup lacks (batch, model), or it is a ~2σ fluctuation.
   - The Track 3 MG run (g20, ~00:20) and more pdwd/PD runs decide.
5. **Next suspects for Track 3's small PD gain**, now that decay and cooldown are cleared: the half batch (56% kept at cooldown 0.1, not yet tested with the long cooldown), the model (12L × 768: width 768 kept 72% at cooldown 0.1), and Track 3's LR regime.

**Decision.** Test the combination: our setup at wd 0.2 + cooldown 0.7 + batch 524,288 (2938 steps) for Muon@0.007, PD@0.01 and PD@0.01 + geometry decay. Cohort `t3recipe_20260925`, seed 260925, A6000.
- **Reading:** if PD − Muon ≥ −0.008 there, our setup reproduces the shrinkage, the batch is the driver, and fixes can be screened locally.
- If it stays ≤ −0.015, the remaining difference is the model or Track 3's LR regime. Next would then be a 1M-batch Track 3 pair (Muon vs PD, 1625 steps).

**Decision: batch diagnostic in Track 3 itself (23:24 CDT).** #36 and PD at a 1M-token batch for 1625 steps (same tokens; `train_gpt_simple_bt1M.py`, `train_gpt_pd_bt1M.py`), chained after the current runs: PD on g20, #36 on priv-g14 (both AD102).
- Only the batch, the step count, PD's statistics decay per forward (kept at 0.938 per optimizer step) and the analysis-only save change. LR 0.025 and WD 0.05 are #36's for both.
- This is a diagnostic, not a benchmark entry: Track 3 fixes the batch.
- **Reading:**
  - If PD − Muon ≤ −0.008 at 1M (vs ≈ −0.002 at 0.5M), the batch drives the shrinkage in Track 3's model.
  - If it stays ≥ −0.004, the model or LR regime does.
  - The local recipe cohort (`t3recipe_20260925`) answers the same question in our model.
- Single pair, per-run σ ≈ 0.0024, so a difference of ~0.006 between batch sizes is the smallest this can see.

**Geometry decay with cooldown 0.5 (00:12 CDT, 2026-09-26; L40S priv-g14): 3.27563**, −0.0041 vs the pooled controls.
- Geometry decay with #36's cooldown 0.7 at the same LR: 3.27411 and 3.27418.
- A shorter cooldown does not help (+0.0015, within σ), so no Muon cooldown-0.5 control is needed and cooldown 0.7 stays.
- #36 at 1M batch started on priv-g14 at 00:12.

**Track 3 MG (Muon + PD-geometry decay; 00:15 CDT; RTX 6000 Ada g20): 3.28946, +0.0097 vs the pooled controls.** Its mid-run lead (−0.012 to −0.020 through step 2000, up to +123 steps) was noise floor, and the final is clearly worse than Muon.
- The predeclared test ("MG within 0.002 of pdwd's 3.2750 → the gain is decay geometry alone") fails by 0.0145. **The geometry-decay gain in Track 3 is PD-specific.**
- **Track 3 2×2 vs the pooled controls:**

| | Decoupled decay | Geometry decay |
|---|---|---|
| Muon update | 0 (3 controls) | +0.0097 (1 run) |
| PD update | −0.0022 (5 runs) | −0.0056 (2 runs, p = 2), −0.0073 (p = 1) |

- The interaction is about −0.013: the decay shape helps PD and hurts Muon.
  - The matched pairs, Muon with isotropic decay and PD with R²-shaped decay, do best.
  - This is the original idea's core claim (decay should match the update's geometry), now tested in the setting where the score is the final loss after a 70% cooldown.
- Our setup agrees in sign: MG at cooldown 0.7 is +0.0045 vs Muon. In our setup, though, geometry decay does not beat decoupled PD at cooldown 0.7 (−0.0217 vs −0.0232), while in Track 3 it does (−0.0034 on average).
- Still open: why PD's own gain is smaller in Track 3. The batch diagnostics (local recipe cohort; Track 3 1M-batch pair) are running.

### Our setup reproduces Track 3 under its full recipe; decay exposure explains when PD loses (00:37 CDT)

`t3recipe_20260925` (wd 0.2 + cooldown 0.7 + batch 524,288; 2938 steps; seed 260925, A6000):

| | Final | vs Muon |
|---|---|---|
| Muon@0.007 | 3.64436 | |
| PD@0.01, decoupled decay | 3.64812 | **+0.0038** |
| PD@0.01 + geometry decay | 3.63521 | **−0.0091** |

- Decoupled PD trailed Muon by 0.027–0.038 from step 1000 to 2000; geometry decay led throughout.
- This matches Track 3 (decoupled PD −0.0022 and PD + geometry decay −0.0056 vs controls). By the predeclared reading (PD − Muon ≥ −0.008), **our setup reproduces Track 3's shrinkage.**

**Unifying variable: total decay exposure E = ∫ ηλ dt over the run.** The weights equilibrate once E ≳ 2.

| Run | E | Decoupled PD − Muon | PD + geometry − Muon |
|---|---|---|---|
| Our original setup (wd 0.01) | ~0.1 | −0.017 | – |
| 1M batch, wd 0.2, cooldown 0.7 | 1.9 | −0.023 | −0.022 |
| Track 3 (#36) | 2.6 | −0.002 | −0.006 |
| 1M batch, wd 0.2, cooldown 0.1 | 2.8 | +0.004 | −0.019 |
| 0.5M batch, wd 0.2, cooldown 0.7 | 3.8 | +0.004 | −0.009 |

- The half batch matters mainly because it doubles the steps, and with them E. It may also add noise: even with geometry decay, the 0.5M recipe keeps only half the gain.
- The 22:30 revision ("decay is not the reason") was an artifact of choosing the one long-cooldown case with E < 2. The mechanism stands, with the timescale condition made explicit.

**Decision: screen fixes locally in the recipe setting** (cohort `t3screen_20260926`, same seed and GPU type, 2938 steps, about 63 min per arm). Track 3 runs only to confirm winners. Arms:
- PD + geometry decay: p = 1; LR 0.014; LR 0.007; α = ⅛.
- Muon LR bracket: 0.005 and 0.01. Muon's 0.007 was tuned at wd 0.01 and 1M batch, so a fair baseline needs its best LR in this regime.

**Reading:** a variant "helps" if it beats PD + geometry decay p = 2 (3.63521) by ≥ 0.003. The recipe gain is judged against Muon's best bracketed LR.

**Track 3 batch diagnostic, partial (01:40 CDT).**
- PD at 1M batch (1625 steps; g20 RTX 6000 Ada): final **3.30041**.
- Its #36 control on priv-g14 was killed externally at 00:46:49, at step ~1000 (`track3/logs/aborted/README.md`).
  - Interactive shells holding all 4 GPUs opened on both allocations at 00:46:31–33 (steps 2618555.72 and 2567578.403, same user).
  - **This session makes no further launches on g20 or priv-g14 until the user says so.**
- Up to step 1000 the pair gives PD − #36 = −0.055 (250), −0.022 (500), −0.013 (750), −0.0096 (1000).
  - At 62% of training that is ~−0.010. At 0.5M batch, PD at the same fraction was ≈ −0.001.
  - This is consistent with the exposure rule, E ≈ 1.3 at 1M vs 2.6 at 0.5M.
- The #36 1M-batch control was resubmitted on the gpu partition (g5, RTX 6000 Ada, job 2623627, ~03:05) for the final comparison.

**Local fix screen, finals (01:47 CDT; recipe setting wd 0.2 + cooldown 0.7 + batch 524,288; seed 260925, A6000; `t3screen_20260926`).**

| Arm | Final | vs Muon@0.01 (best bracketed) |
|---|---|---|
| Muon@0.005 | 3.64985 | +0.0095 |
| Muon@0.007 | 3.64436 | +0.0040 |
| **Muon@0.01** | **3.64032** | 0 |
| PD decoupled α ¼ @0.01 | 3.64812 | +0.0078 |
| PD + geometry p = 2, α ¼ @0.01 | 3.63521 | −0.0051 |
| PD + geometry **p = 1**, α ¼ @0.01 | 3.63206 | **−0.0083** |
| PD + geometry p = 2, **α ⅛** @0.01 | 3.63015 | **−0.0102** |
| PD + geometry p = 2, α ¼ @0.007 | 3.64247 | +0.0022 |
| PD + geometry p = 2, α ¼ @0.014 | (step 2750: +0.0005 vs the α ¼ reference) | |

- **Muon's best LR is at the top edge of its bracket** (0.01), so Muon@0.014 is needed. PD's margin is judged against Muon's best LR.
- **Both winners halve the equilibrium reallocation.** With geometry decay in the random-walk equilibrium, relative steps go as r^{p/2} = λ^{−αp/2}.
  - p = 1 with α ¼ and p = 2 with α ⅛ both give λ^{−1/8} instead of λ^{−1/4}, and both beat the reference by 0.003–0.005.
  - The half-batch regime prefers a milder reallocation. Track 3 agrees in direction: p = 1 3.27248 vs p = 2 3.27411 and 3.27418.
- **Next local arms:** Muon@0.014 (bracket); α ⅛ + p = 2 @0.014 (PD LR); α ⅛ + p = 1 (exponent 1/16); decoupled α ⅛ (does geometry decay still matter at the milder α?).
- **Track 3 confirmation:** α ⅛ + geometry decay p = 2 (`train_gpt_pd_pdwd_a0.125.py`), gpu partition L40S (g20 and priv-g14 are paused).
- Round 1 complete (01:51 CDT): PD + geometry p = 2, α ¼ @0.014 **3.63277** (−0.0024 vs @0.01, below the 0.003 bar; −0.0076 vs Muon@0.01).
  - Both methods' best LRs are at the top of their brackets in this regime (Muon 0.01, PD + geometry 0.014).
  - Round 2 (running, ~02:55) tests Muon@0.014 and α ⅛ @0.014.

**02:22 CDT: g20 and priv-g14 back in use.** The user cleared them ("Feel free to use gpus on g20 and priv-g14 towards the goal"). Their interactive steps still hold the GPUs, but all 8 GPUs were idle (0 MiB, 0%). Launches, with `--overlap`:
- g20: geometry decay p = 1 replicate (`train_gpt_pd_pdwd1.py`; first run 3.27248).
- priv-g14: α ⅛ + geometry decay p = 2 replicate (`train_gpt_pd_pdwd_a0.125.py`). Its first run on g21, L40S, led by +127 steps at step 1750.
- Both replicate the equilibrium-exponent-⅛ family, which won the local screen.

**Round 2 and Track 3 α ⅛ (02:53 CDT).**
- **Track 3 α ⅛ + geometry p = 2 (L40S g21): 3.27291**, −0.0069 vs the pooled controls.
  - Equilibrium exponent e = α·p/2 = ⅛ (α ¼ p = 1: 3.27248; α ⅛ p = 2: 3.27291): mean −0.0071.
  - e = ¼ (α ¼ p = 2: 3.27411, 3.27418): mean −0.0056. Same ranking as the local screen.
- **Local round 2 (recipe setting; vs Muon's best bracketed LR).** Muon's bracket is now interior: 0.005 3.64985, 0.007 3.64436, **0.01 3.64032**, 0.014 3.64188.

| PD variant (geometry decay unless noted) | e = α·p/2 | vs Muon@0.01 |
|---|---|---|
| α ¼, p = 2 @0.01 / @0.014 | ¼ | −0.0051 / −0.0076 |
| α ¼, p = 1 @0.01 | ⅛ | −0.0083 |
| α ⅛, p = 2 @0.01 / @0.014 | ⅛ | −0.0102 / −0.0098 |
| **α ⅛, p = 1 @0.01** | 1/16 | **−0.0114** (3.62896) |
| α ⅛, decoupled | | at step 2450, +0.0051 vs Muon@0.007 (final pending) |
| α ¼, decoupled | | +0.0078 |

- Milder equilibrium reallocation keeps helping, with diminishing steps: ¼ → ⅛ → 1/16 gives −0.005 → −0.009/−0.010 → −0.011.
- Geometry decay remains essential at α ⅛.
- **Next:**
  - Local round 3 continues along the exponent direction: α 1/16 with p = 2 (e 1/16 with a Muon-like weight profile), α 1/16 with p = 1 (e 1/32), α ⅛ with p = 1 @0.014.
  - Track 3: α ⅛ with p = 1.

**Track 3 batch diagnostic, final (02:57 CDT).** At 1M batch (1625 steps, same tokens, #36's LR 0.025 and WD 0.05; both on RTX 6000 Ada): PD 3.30041, #36 3.31174 (resubmitted on g5), **PD − #36 = −0.0113**.
- PD led by +41–58 steps throughout.
- At #36's 0.5M batch, plain PD averages −0.0022 over 5 runs.
- By the predeclared reading (≤ −0.008), **the batch drives the shrinkage in Track 3's own model**. Halving the step count halves the decay exposure (E ≈ 1.3 vs 2.6), and PD's gain grows about 5×.
- This confirms the exposure rule in Track 3's architecture, alongside our model's recipe cohort. One pair; the pair SE is ~0.0034, so this is about 3σ from zero and about 2.7σ from the 0.5M value.

**Decoupled α ⅛ in the recipe setting (03:07 CDT): 3.63525, −0.0051 vs Muon@0.01.**
- That is much better than decoupled α ¼ (+0.0078). Geometry decay still adds 0.005–0.006 at α ⅛ (p = 2 −0.0102, p = 1 −0.0114).
- In the random-walk equilibrium, a run is described by two exponents:
  - the weight-profile exponent a_w: w_j ∝ λ_j^{−a_w}, a_w = α(1 − p/2), and a_w = α for decoupled decay;
  - the relative-step exponent e = αp/2 (0 for decoupled decay).
- The six arms, vs Muon's best:

| Arm | (a_w, e) | vs Muon's best |
|---|---|---|
| decoupled α ¼ | (¼, 0) | +0.0078 |
| decoupled α ⅛ | (⅛, 0) | −0.0051 |
| α ¼, p = 2 | (0, ¼) | −0.0051 |
| α ¼, p = 1 | (⅛, ⅛) | −0.0083 |
| α ⅛, p = 2 | (0, ⅛) | −0.0102 |
| α ⅛, p = 1 | (1/16, 1/16) | −0.0114 |

- A flat or nearly flat profile and a small nonzero e do best. The within-step α (the whitening inside the polar) also changes between arms, so this is descriptive rather than a fit.
- Round 3 probes (0, 1/16) with α 1/16, p = 2, and (1/32, 1/32) with α 1/16, p = 1.

**Track 3, e = ⅛ family (03:24 CDT).** The α ⅛ + geometry p = 2 replicate (L40S priv-g14) gave **3.27201**, the best Track 3 run so far.
- Family: 3.27248 (α ¼ p = 1), 3.27291 and 3.27201 (α ⅛ p = 2). **Mean 3.27247, SD 0.0005.**
- **vs the pooled #36 controls** (3.27976, SD 0.0024, n = 3): **−0.0073**, Welch t ≈ 5.2 (df ≈ 2), p ≈ 0.02 one-sided. vs the #36 H100 mean: −0.0062.
- Roughly 150–160 steps at the end-of-run slope (0.0045 per 100 steps).
- These runs vary far less (SD 0.0005) than the #36 controls (0.0024), which split into persistent trajectories between steps 1000 and 1500.
- **Next (launched 03:24, priv-g14): α ⅛ + geometry decay at 3100 steps** (`train_gpt_pd_pdwd_a0.125_s3100.py`). #36's schedule is compressed accordingly. This measures steps-to-3.28 directly. The slope estimate gives ≈ 3.279 at 3100.

**Overnight finals (runs ended 03:47–04:23 CDT; recorded 12:45 CDT).**

*Track 3, #36 settings, vs the pooled controls (3.27976):*

| Configuration | e = α·p/2 | Finals | Mean vs pooled |
|---|---|---|---|
| α ⅛, geometry p = 2 | ⅛ | 3.27201, 3.27291 | **−0.0073** |
| α ¼, geometry p = 1 | ⅛ | 3.27248, 3.27475 | −0.0061 |
| α ⅛, geometry p = 1 | 1/16 | 3.27334 | −0.0064 |
| α ¼, geometry p = 2 | ¼ | 3.27411, 3.27418 | −0.0056 |

- The five runs with e ≤ ⅛ average **3.27310 (−0.0067; SD 0.0010)**. Their worst run (3.27475) is still below every control.
- **3100 steps**, α ⅛ + geometry p = 2 (#36 schedule compressed): **3.27990**, just below 3.28.
  - The steps-to-3.28 saving is therefore ~150 steps on a single run, with no margin.
  - A Track 3-style claim, (3.28 − mean)·√n ≥ 0.004, would sit near 3150 steps (expected ≈ 3.2775) and need n ≈ 3.
  - Our Ada controls run +0.0011 above the #36 H100 mean.

*Local round 3 (recipe setting, vs Muon@0.01):*
- α 1/16 p = 2 (e 1/16, flat profile): −0.0064. α 1/16 p = 1 (e 1/32): −0.0060. α ⅛ p = 1 at LR 0.014: −0.0073.
- All are worse than α ⅛ at LR 0.01 (p = 1 −0.0114, p = 2 −0.0102), so the optimum is near α ⅛, e between 1/16 and ⅛.
- Further α/p tuning now moves the result by less than Track 3's per-run noise.

**3100-step replicates (user direction, 12:54 CDT, 2026-09-26): "submit 4 more jobs for the currently used schedule and 3100 steps; use g20, priv-g14, and two other nodes".**
- Script: `train_gpt_pd_pdwd_a0.125_s3100.py`, α ⅛ + geometry decay p = 2 with #36's schedule compressed to 3100 steps.
- Runs: g20 RTX 6000 Ada (`48883380`), priv-g14 L40S (`c05dde78`), and two A6000 nodes in the gpu partition (`0af9dc11`, `eaaab56d`). The Ada nodes g5, g16 and g21 were full, and the A6000 runs take ~1h50.
- With the first 3100-step run (3.27990, L40S), n = 5.
- **Track 3 criterion:** (3.28 − mean)·√5 ≥ 0.004 needs a mean ≤ 3.27821.
- **The schedule is #36's, tuned for Muon.** For PD + geometry decay, only cooldown 0.5 vs 0.7 has been tried (3250 steps: 0.5 slightly worse, single run). Longer or split cooldowns, decay shape and LR floors are untested.
- The two same-second g20/priv-g14 launches shared one console log. `run_on.sh` now adds the job ID to the console name; the file was replaced by `mv`, since the running launchers still hold the old one.

**3100-step results (14:29 CDT):** 3.27990 (L40S), 3.27880 (L40S), 3.28240 (RTX 6000 Ada), 3.28000 (A6000); the second A6000 run is pending.
- **Mean of 4: 3.28028.** At 3100 steps the configuration does not reach 3.28 on average; the first run was on the lucky side.
- Fitting through the 3250-step mean (3.27246) gives about 0.0052 per 100 steps. Expected: 3125 ≈ 3.2790 (~16 runs for the rule), 3150 ≈ 3.2777 (~3), 3175 ≈ 3.2764 (~2), 3200 ≈ 3.2751 (1).

**3125-step runs (user direction, 14:41 CDT): "start 3125 runs; use g20, priv-g14 and cluster".**
- Script: `train_gpt_pd_pdwd_a0.125_s3125.py`, identical to the 3100 script except `train_steps`.
- Runs: g20 (RTX 6000 Ada), priv-g14 (L40S), and two A6000 jobs (2624272, 2624273) queued in the gpu partition because the Ada nodes were full.
- The user was told that 3125 likely needs ~16 runs at the expected mean (~3.2790) on our GPUs. This first batch estimates the mean.
- **3100-step set complete (14:51 CDT):** 3.27990 (L40S), 3.28240 (RTX 6000 Ada), 3.27880 (L40S), 3.28000 (A6000), 3.27960 (A6000).
  - n = 5, **mean 3.28014**; criterion (3.28 − mean)·√5 = −0.0003 (needs ≥ 0.004).
  - With #36's schedule, 3100 steps is not reached on our GPUs.

### Decision: weight decay × cooldown jointly (user-approved, 2026-09-26 ~15:00 CDT)

**Argument.**
- In the decay equilibrium the relative step is ω ≈ √(2ηλ) and the exposure is E ≈ Σ ω²/2. Lowering WD at a fixed schedule lowers both, which is why plain PD at WD 0.025 lost its mid-run lead in the cooldown.
- Geometry decay already keeps the most-used input directions at low exposure (decay × r̂² ≈ 0.25×), so the global WD mostly sets the bulk's effective LR.
- The cooldown anneals a noise floor whose height WD sets (via ω). **The best cooldown should therefore move with WD** (higher WD → longer cooldown; lower WD → shorter), and the optimum should lie on a diagonal of the (WD, cooldown) plane.
- One-factor sweeps can miss it. Our only cooldown test (0.5 vs 0.7) was at a single WD.

**Design** (cohort `t3wdcd_20260926`). Our model in the Track 3 recipe (batch 524,288, 2938 steps, seed 260925, A6000):
- 2×2 of WD {0.1, 0.3} × cooldown {0.5, 0.9} around the existing center (WD 0.2, cooldown 0.7);
- for PD + geometry decay (α ⅛, p = 2, LR 0.01; center 3.63015) and for Muon (LR 0.01; center 3.64032);
- 8 runs.
- Mapping to Track 3 by per-step shrink: local 0.1 / 0.2 / 0.3 ↔ Track 3 0.025 / 0.05 / 0.075.

**Readings (fixed now).**
- A PD corner "helps" if it beats the PD center by ≥ 0.002 (arms share the seed; paired noise ≈ 0.001).
- The better diagonal (high WD + long cooldown vs low WD + short cooldown) shows whether cooldown should scale with WD.
- The gain is PD-specific if PD − Muon at the best PD corner is ≤ −0.012 (center −0.0102). Otherwise it is a schedule gain Muon shares.
- A PD corner that helps goes to Track 3: 2 runs at 3250 steps on g20 and priv-g14, mapped proportionally.

**3125-step set complete (16:17 CDT):** 3.27942 (L40S), 3.27971 (RTX 6000 Ada), 3.27950 and 3.27951 (A6000).
- n = 4, **mean 3.27954**, spread 0.0003. These runs are far more consistent than the 3100 set (spread 0.0036) or the #36 controls.
- Criterion (3.28 − mean)·√4 = 0.0009 (needs ≥ 0.004). With the rule's assumed σ = 0.0013, about 76 runs would be needed, so 3125 is not claimable with #36's schedule.
- The WD × cooldown grid (running) is the next lever.

**WD × cooldown grid, progress (16:21 CDT).**
- *Moved (user direction):* the two PD arms still queued (jobs 2624286 and 2624287, never started) were cancelled. They now run unchanged on the allocations as `PDgeo2_a0.125_wd0.1_cd0.9_..._ada` (g20, step 2618555.82) and `PDgeo2_a0.125_wd0.3_cd0.5_..._l40s` (priv-g14, step 2567578.414). The unstarted A6000 directories are kept in `t3wdcd_20260926/cancelled_before_start/`.
  - These two cells are compared with the A6000 center across GPU types.
  - PD (0.1, 0.5) runs on A6000 g15, and PD (0.3, 0.9) is still queued in the gpu partition.
- *Muon half complete (LR 0.01, A6000):*

| WD \ cooldown | 0.5 | 0.7 | 0.9 |
|---|---|---|---|
| 0.1 | **3.63867** | | 3.65092 |
| 0.2 | | 3.64032 | |
| 0.3 | 3.66040 | | 3.65623 |

- For Muon, WD 0.3 is too strong at either cooldown (+0.016 / +0.020).
- Low WD needs a short cooldown: (0.1, 0.5) is −0.0017 vs the center, while (0.1, 0.9) is +0.0106. The predicted pairing of lower WD with a shorter cooldown holds for Muon.

**WD × cooldown grid (PD half, 3 of 4 cells, 17:11 CDT):**

| PD + geometry α ⅛ | Final | vs PD center 3.63015 | PD − Muon, same cell |
|---|---|---|---|
| WD 0.1, cooldown 0.5 | 3.63055 | +0.0004 | −0.0081 |
| WD 0.1, cooldown 0.9 | 3.64583 | +0.0157 | −0.0051 |
| WD 0.3, cooldown 0.5 | 3.63844 | +0.0083 | −0.0220 |
| WD 0.3, cooldown 0.9 | running | | |

- No corner beats the center by 0.002. PD is flat along the low-WD / short-cooldown diagonal, and #36's WD and cooldown look near-optimal for PD in this model.
- PD tolerates strong WD far better than Muon with geometry decay (0.3, 0.5: −0.022 vs Muon).

**3150-step runs (user direction, 17:13 CDT): "use g20 and priv-g14 to aim for 3150, and submit more jobs to the cluster for 3150".**
- Script: `train_gpt_pd_pdwd_a0.125_s3150.py`.
- Runs: g20 and priv-g14 (running), plus 4 A6000 jobs queued in the gpu partition (2624721–2624724).
- Expected mean at 3150 ≈ 3.2781, interpolated between the 3125 mean (3.27954, n = 4) and the 3250 mean (3.27246, n = 2). The criterion then needs n ≈ 5, so 6 runs were launched for margin.

**WD × cooldown grid complete (17:50 CDT).** PD + geometry α ⅛ (WD 0.3, cooldown 0.9): 3.64180, PD − Muon −0.0144.

| PD final (PD − Muon) | cooldown 0.5 | cooldown 0.7 | cooldown 0.9 |
|---|---|---|---|
| WD 0.1 | 3.63055 (−0.0081) | | 3.64583 (−0.0051) |
| WD 0.2 | | **3.63015** (−0.0102) | |
| WD 0.3 | 3.63844 (−0.0220) | | 3.64180 (−0.0144) |

- **Reading:** no corner beats the center by ≥ 0.002. **#36's WD and cooldown (the local center) are near-optimal for PD + geometry decay**, flat along the (0.1, 0.5)–(0.2, 0.7) diagonal. Low WD needs a short cooldown for both methods.
- PD beats Muon in every cell. Its margin is largest at strong WD, because Muon degrades more there, which does not help against #36's tuned baseline.
- Remaining schedule knobs (untested): decay shape, a separate cooldown for the aux AdamW, LR floors, warmup, momentum schedule.

**Runs needed at 3150 (estimate, 17:45 CDT).** A weighted linear fit over 3100/3125/3250 puts the expected mean at 3150 at ≈ 3.2778–3.2781. Including per-run noise (~0.001) and fit uncertainty (~0.0005), P(pass) ≈ 50% with 4 runs, 60% with 5, 70% with 6, 80% with 8, 85% with 10. At 3175 (≈ 3.2765), 3 runs pass with ≈ 95%.

**Switch to the original 3250-step schedule (user direction, 18:51 CDT): "switch to the original 3250-step for everything not running; report the earliest step that beats the criteria".**
- **Rationale.** Rule 5 allows early stopping at a step shared by all runs, so one set of full-length runs yields every candidate step.
  - Stopping the 3250 schedule at step 3150 averaged 3.27822 (n = 2), vs 3.27886 (n = 2) for the compressed 3150 schedule; indistinguishable.
  - Only runs of identical code can be pooled.
- **Actions.**
  - Cancelled the three never-started 3150 cluster jobs (2624722–2624724) and submitted three 3250 jobs (2624732–2624734, A6000, queued).
  - Chained 3250 runs after the in-progress compressed 3150 runs: two on priv-g14, one on g20.
  - The running compressed 3150 runs (2624721 cluster, g20 #2, priv-g14 #2) finish as a separate set.
- **Tool:** `track3/earliest_step.py [SCRIPT]`.
  - It pools only runs whose logged code is identical to the script, and only finished runs; every finished run of the script counts.
  - It applies (3.28 − mean)·√n ≥ 0.004 at each late validation step and reports the earliest passing step.
- **Now (2 runs, `train_gpt_pd_pdwd_a0.125.py`):** mean 3.27822 at 3150 (score 0.0025) and 3.27612 at 3175 (0.0055). **Earliest passing step: 3175.** Step 3150 needs a mean of ≲ 3.2782 over ~5 runs.

**Compressed 3150 set complete (19:58 CDT; supporting evidence, not the claim):** 3.27783 (L40S), 3.27989 (RTX 6000 Ada), 3.27886 (A6000), 3.27719 (L40S), 3.27730 (RTX 6000 Ada).
- n = 5, mean 3.27821. **Criterion (3.28 − mean)·√5 = 0.00399**, just short of 0.004; the fifth run needed ≤ 3.27729.
- Per the 18:51 decision, the claim is the 3250-schedule set read at the earliest passing step. That set's new runs started at 19:14 (priv-g14) and 19:58 (g20); the three cluster jobs are still queued.

**3250-schedule set, final (20:16 CDT; `train_gpt_pd_pdwd_a0.125.py`, identical code, n = 3, all L40S):** 3.27201, 3.27209, 3.27291 at 3250.
- At step 3150: 3.27775, 3.27778, 3.27868 (mean 3.27807, score 0.0033, fails).
- At step 3175: mean 3.27599 (score 0.0069, passes).
- **Earliest step passing the Track 3 criterion: 3175.**

**User redirection (20:13 CDT):** "add one more compressed 3150 run on priv-g14; stop the 3250 step runs".
- The three queued 3250 cluster jobs were cancelled, and the g20 3250 run was stopped at step 500 (`track3/logs/aborted/`).
- The priv-g14 3250 run was 2 minutes from its end and was allowed to finish; its chained successor was cancelled at startup.
- Compressed 3150 run #6 started on priv-g14 at 20:16. With 5 runs the set scores 0.00399; the sixth needs ≤ 3.27913 for the set to pass at 3150.

### Track 3 result: PD + geometry decay reaches 3.28 at 3150 steps (21:16 CDT, 2026-09-26)

`track3/train_gpt_pd_pdwd_a0.125_s3150.py`: #36 with its hidden-matrix Muon update replaced by partial data-norm Muon (α ⅛) with PD-geometry weight decay (p = 2). #36's LR, WD, schedule shape, aux Adam, architecture, data and batch are unchanged, and the schedule is compressed to 3150 steps. Six runs of identical code, all reported:

| Run | GPU | Val loss at 3150 |
|---|---|---|
| `e4ff2c15` | L40S | 3.27783 |
| `27abc8e7` | RTX 6000 Ada | 3.27989 |
| `58e8667f` | A6000 | 3.27886 |
| `2124499a` | L40S | 3.27719 |
| `742049d3` | RTX 6000 Ada | 3.27730 |
| `c2be443d` | L40S | 3.27787 |

- **Mean 3.27816, n = 6: (3.28 − mean)·√6 = 0.0045 ≥ 0.004. Passes Track 3's criterion at 3150 steps**, 100 fewer than #36 (3250).
- **Pairwise vs #36** (n = 10, mean 3.2787 at 3250; 0.0045 loss per 100 steps): (0.00054 + 0.0045)/√(1/10 + 1/6) ≈ 0.0098 ≥ 0.004. Significant.
- **Disclosure:**
  - The sixth run was added after five runs scored 0.00399 (optional stopping). All six runs are reported and none is dropped.
  - Runs are on Ada/Ampere GPUs, not H100. Our Ada #36 controls ran about 0.0011 above #36's H100 mean, so the H100 result is likely slightly better.
- **The same configuration with #36's full 3250-step schedule** (3 runs, read at intermediate steps): earliest passing step **3175** (mean 3.27599 at 3175; 3150 scores 0.0033).
- **Compressed sets that do not pass:** 3100 (n = 5, mean 3.28014) and 3125 (n = 4, mean 3.27954).

### Second-order audit: principles behind the progress and a measurement plan (2026-09-26 ~22:50 CDT; planning only, nothing launched)

**User question.** Step back from the leaderboard's axis-by-axis progress (much of it presumably found by coding agents each pushing one axis). Extract the principles, then find a principled way to study activations, gradients and curvature: how close can an efficient optimizer get to a full second-order method, and what is necessary?

**Inputs.**
- Our results so far.
- The Gauss-Newton paper (Abreu et al., ICLR 2026).
- A survey of 23 Track 3 methods (`logs/muon_spectra/second_order_audit_20260926/t3_principles_survey.md`), checked against the records' code and PR texts.
- An independent review of the first draft (read-only subagent).

**Big picture: five levers.**

| Lever | Leaderboard examples | What an ideal second-order step does |
|---|---|---|
| Basis | SOAP-Muon, PMuon, PSGD, Shampoo (gradient Grams); Newton-Muon, PD (activations) | uses the per-layer curvature eigenbasis (input factor ⊗ output factor) |
| Per-direction magnitude | NorMuon and Aurora (rows); Contra/Soft/Dyn/Tempered polar (σ^p); Adam in the eigenbasis; PD's post-multiplication | inverse curvature, shrunk by noise |
| Scale | hyperball, u/w floor, radial brake, Muown, row floors, cautious WD, WD tuning, geometry decay | automatic: Newton/GN steps are scale-equivariant |
| Averaging in time | momentum, EMA-Nesterov, MuLoCo, tail EMA, extrapolation | not second-order; Polyak averaging reaches the optimal noise level without curvature |
| Coupling | Circuit-Muon (V/O gauge) | full GN; per-layer carries most of it (GN paper) |

- **Most-mined lever.** Scale is the most-mined lever (~10 of 23 methods). About a third of the methods are basis changes.
- **Interactions.** Same-slot parts substitute and different-slot parts stack:
  - #44 dropped Contra-Muon, Circuit-Muon and Aurora as neutral once SOAP-Muon covered all matrices.
  - S∘PD stacks on PD.
  - Track 3's decay × update-geometry interaction (about −0.013) is one that a single-axis search misses.

**Principles (working statements).**
1. **Muon as steepest descent.** Muon is steepest descent under a norm that assumes isotropic inputs and outputs. The basis lever replaces those assumptions with measured statistics.
   - Gradient Grams are not curvature. When noise dominates, the per-token terms give an empirical-Fisher factor, so Shampoo's ¼ power acts like C^−¼.
   - When signal dominates, a simple quadratic model gives GᵀG ∝ C² (AdaGrad's object), so the ¼ power acts like C^−½.
   - Gram methods therefore raise their effective power with batch on their own. This is a candidate reason the gradient-Gram basis beat the activation basis in our SOAP test.
   - Activations give the input factor at any batch.
2. **Magnitude must account for noise.** The full inverse is right only at infinite batch.
   - The one-step Wiener rule does not predict the best power.
   - In a noisy-quadratic model with momentum 0.95 and a 70% cooldown (the reviewer's `nqm_alpha2.py`, re-run here), the best power is:
     - ⅜–7/16 at every batch if noise ∝ curvature (κ = 1), with α ¼ 6–19% worse;
     - ¼ at small batch, rising with batch, if κ ≈ ½;
     - here n_j ∝ λ_j^κ along input eigendirections.
   - So κ decides whether PD's α ¼ is a noise effect.
3. **Scale control restores scale-equivariance by hand.** For weights whose outputs are normalized, gradients scale as 1/c and curvature as 1/c². A Newton/GN step's relative size is therefore independent of the weight norm; Muon's and Adam's are not.
4. **The regularizer must match the magnitude map.** Decoupled decay after a non-uniform map P implies a penalty ½λ⟨W, P⁻¹W⟩.
   - Noise-shaped maps (AdamW, NorMuon) penalize noisy directions and can help.
   - Curvature-shaped maps (PD, anything GN-like) penalize the most-used directions and hurt, unless the decay goes through the same map (geometry decay).
   - Any GN-like method will meet this in long runs with decay.
5. **Probably not necessary at moderate batch:** higher-order loss terms and cross-layer curvature (GN paper), and exact inverses.

**Review corrections to the first draft** (all accepted):
- The claim that Grams lose curvature content above the noise scale was wrong (principle 1).
- The claim that polar-terminated updates are immune to decay equilibria was wrong as stated:
  - wide matrices (mlp.proj) are not isotropic;
  - aligned directions settle at relative step ηλ and random-walk ones at √(2ηλ), 40× apart at #36;
  - NorMuon's post-polar map gains 75 steps under decoupled WD (#10 vs #12);
  - the SOAP-Muon record line uses no decoupled decay (norm floors, radius pin).
- The one-step analysis was wrong (principle 2).
- **Confound in our α sweep.** At a fixed LR, PD's output energy E‖ΔWx‖² relative to Muon at matched Frobenius norm is 0.64 / 0.46 / 0.37 / 0.32 for α ⅛ / ¼ / ⅜ / ½ (median over modules, Track 3 run e4ff2c15; verified). α was confounded with the effective step size, so the sweep needs an LR bracket per α.
- **Missed explanations:**
  - Stiff-direction stability: PD may mainly cap a few tens of stiff input directions so the bulk can take a larger LR. The x̄-only arm is the k = 1 case and did not beat tuned Muon.
  - River valley under a long cooldown: one-step and mid-run metrics measure what the cooldown erases.
- **Feasibility:**
  - Checkpoints are rolling, so no mid-run checkpoints exist; reruns must save them.
  - JVP needs the math attention kernel or double-VJP in FP32, the logit quadratic computed as Var_p(dz), and the logit softcap kept.
  - Rank-1 probes cannot see basis errors; use GGN-vector products on each method's actual update.
  - Per-layer GN is conditioned at 1e5 or worse, so use Lanczos or an inner solver, not CG.

**Plan (not launched; needs the user's go-ahead).**
- **M1 (first).**
  - Rerun Muon@0.007 and PD α ¼@0.01 (seed 260925), saving checkpoints with momentum at steps ~300, 700 and 1100.
  - From per-sequence gradients and JVP curvature along C's eigendirections, measure h_j, n_j and s_j, giving κ.
  - Measure the Gram exponent κ_G (slope of v_jᵀE[GᵀG]v_j vs λ_j) at 0.5M and 1M.
  - Parameter-free test: α* ≈ κ_G/8.
- **M2.** Calibration gate: a one-step or short-branch metric must reproduce the known ordering α ¼ > ⅛ > ⅜ ≫ ½ before it is used to measure distance to per-layer GN.
- **Cheap training tests, each about one pair:**
  - PD on the top-k input directions only (k 16/64);
  - PD with error-weighted C, E[‖e_t‖² x_t x_tᵀ];
  - LR-bracketed α sweep;
  - an E-matched half-batch pair: Muon vs decoupled PD α ¼@0.01, 0.5M batch, wd 0.1, cooldown 0.7, E = 1.9.

**Readings (fixed now).**
- κ ≈ 1: the noise route does not explain α ¼; next is the top-k test.
- κ ≤ ½: run the batch × α sweep with predeclared α*(b), and try α decreasing over training.
- Top-k PD with k ≤ 64 keeping ≥ 80% of PD's gain: stiff-direction capping is the mechanism.
- E-matched pair PD − Muon ≤ −0.015: exposure drives the Track 3 shrinkage. ≥ −0.008: batch matters beyond exposure.
- Error-weighted C better than C by ≥ 0.003: per-token output weighting belongs in the input factor.

**Cost.** M1 is two ~50-minute runs plus minutes of measurement per checkpoint. Each training test is about one 4-GPU pair of 30–60 minutes.

**User direction (2026-09-26 ~23:20 CDT): do not center the program on Track 3.**
- Its fixed batch, step budget and strong decay can hide real optimization gains.
- First decide what must be measured about activations, gradients and curvature (spectra, interactions, the assumptions each method makes) and what could be estimated cheaply to approach Gauss-Newton.
- The expanded design is in `SECOND_ORDER_AUDIT.md`: the GN block as a four-index covariance of sampled-label per-sequence gradients, an assumption ladder with tests, and a per-layer Kronecker measurement frame for curvature, signal and noise.
- The Track 3-motivated E-matched half-batch pair is dropped from the first round. The M1 reruns are kept, but with more checkpoints (steps ~50, 200, 500, 900, 1300, end) and SOAP-Muon and S∘PD trajectories.
- Not launched.

**Steps 1 and 2 launched (user go-ahead, 2026-09-26 23:30 CDT).**
- **User direction.** Short-term or branch gains are not a success criterion: the schedule, optimizer state and trajectory confound them. Aim for principled findings, not trial and error.
- **Step 2, trajectories** (`logs/muon_spectra/soaudit_traj_20260926`).
  - Arms: Muon@0.007, PD α ¼@0.01, SOAP-Muon core@0.007 and S∘PD@0.01, all seed 260925, frontier-norm width 512, weak decay.
  - Each arm's config is copied exactly from its original run (waves 4, 5, 8, 10), plus `keep_checkpoints` [10, 50, 100, 200, 500, 900, 1300].
  - Kept checkpoints hold full model and optimizer state. The next step's weights are also saved, so each applied update is recoverable as W[t+1] − W[t]. PD runs also save rank 0's input statistics.
  - Queues: priv-g14 (L40S) runs Muon then SOAP-Muon; g20 (RTX 6000 Ada) runs PD then S∘PD. Started 23:34 CDT.
- **Code.**
  - New config key `keep_checkpoints` (default [], measurement only; validated), whitelisted as measurement in `run_arm.py`. The tiny and qualification stages exercise it (steps 1 and 2).
  - 69 unit tests pass on the frozen copy.
- **Step 1, probe library** `research/adamw_spectra/gn_probe.py`, with `test_gn_probe.py` (7 CPU tests pass).
  - Full-size checks and the first measurement (exact one-sided GN marginals) run on the final Muon and PD checkpoints: `logs/muon_spectra/second_order_audit_20260926/check_tools.py`, dev-gpu job 2624811.
- **Design review** (second independent reviewer). Its corrections and additions are in `SECOND_ORDER_AUDIT.md` §9. The main one is attention's cross-position structure: for keys, the shared-mean input direction is nearly flat, contrary to per-token statistics.

**Step 3 started (user go-ahead, 2026-09-27 00:10 CDT). Predictions for M-A, fixed before any trajectory result is read.**
- **Measurement.** One-sided GN marginals of all 48 hidden matrices at every kept checkpoint (steps 10–1300 and final), for the Muon and PD trajectories now and SOAP-Muon and S∘PD when done.
  - Same 2048 held-out validation sequences for every checkpoint.
  - Sanity gate per checkpoint: V/O gauge and residual scale flat to < 1e-6 of random, or the job aborts.
  - Fit: exact input marginal ≈ a·tr(B)·C_within + b·tr(B)·C_between, where C_within and C_between are the within-sequence and sequence-mean parts of C.
- **Mechanism.**
  - Softmax ignores a common shift of all of a sequence's keys. Without QK-norm, each sequence's key errors sum to zero (verified in a toy: 1e-5), so keys see only within-sequence-centered inputs.
  - Values aggregate across positions coherently.
- **Predictions** (medians over depths, every checkpoint from step 50, both trajectories):
  - k: b/a < 0.3.
  - v: b/a > 1.5.
  - q, o, up and down: b/a between 0.7 and 2.5, since downstream mixing is mild.
  - For k and v, the two-part fit halves the residual of K-FAC's single C or better.
- **What changes next.**
  - If these hold on every trajectory, per-kind input factors become a candidate component: C_within for keys, C_within + b·C_between for values. Both are cheap, since forward-hook statistics give them.
  - Before any full run, the candidate must also account for the existing results: the key exponent 0.74–0.78, centering halving PD's gain, and per-kind α giving nothing.
  - If they fail, the final-checkpoint pattern was specific to post-cooldown weights.

**Gap map result and batch-size study (user goal set 2026-09-27 01:33 CDT; see `logs/muon_spectra/second_order_audit_20260926/OBSERVATIONS.md`).**
- **Gap map at batch 1M** (28 checkpoints, four optimizers).
  - The Newton decrement lives in flat directions (curvature 1e-5 to 1e-4). Their per-direction critical batch is ~1e4 sequences, i.e. about 5M tokens.
  - From step ~100 on, only 10–40% of the decrement is reachable per step at 1M tokens (values 60–95%).
  - The stiff directions oscillate (cos(momentum, gradient) −0.3 to −0.8).
  - The effective step profiles of better optimizers are closer to the noise-aware Newton profile.
- **Prediction.** At larger batch, more of the decrement is reachable, so the preconditioned methods (PD, S∘PD) should gain more over Muon, and the best preconditioning strength may rise.
- **Batch-size study** (cohort `soaudit_batch4m_20260927`).
  - Batch 4,194,304 tokens at the same 1.54B tokens (368 steps), seed 260925. Warmup is held at the same tokens (13 steps); cooldown 0.1.
  - LR brackets: Muon {0.007, 0.014, 0.028} on priv-g14 L40S; PD α ¼ {0.01, 0.02, 0.04} on g20 RTX 6000 Ada; S∘PD {0.01, 0.02, 0.04} on cluster A6000 (jobs 2624858–60).
  - Kept checkpoints at steps 37, 183 and 330, for the gap map at 4M.
  - `run_arm.py` now allows `warmup_steps` as a treatment. The frozen copy passes its unit tests.

**Two-sided data norm implemented (2026-09-27 02:13 CDT; not yet run).**
- **Motivation (audit).**
  - The stiff top-1% Kronecker directions carry ~2% of the update energy yet cause 70–90% of the gradient change in every curvature class, through Gauss-Newton coupling.
  - PD damps the stiff input side but not the output side.
  - In the progress-rate model, two-sided Kronecker scaling reaches 0.63–0.90 of the noise-aware Newton rate, against ≤ 0.37 for input-only.
- **Rule.** D = L polar(L M R) R, with R = C^-α (PD) and L = (B / mean eig + 1e-3 I)^-β. B is an EMA (0.8 per refresh) of the per-token output-error second moment E[e eᵀ] of each body matrix.
  - B comes from an eager, eval-mode pass on 8 local sequences every 10 steps (~0.2% extra compute), with gradients taken with respect to the matrix outputs only.
  - Parameter gradients, PD's input statistics and training randomness are untouched.
  - Labels: "ef" uses the data (empirical Fisher, free in principle from the ordinary backward pass); "gn" samples from the model (Gauss-Newton).
  - The statistics are owner-local and not checkpointed.
- **Config keys:** `data_norm_out_beta` (0 = PD), `data_norm_out_source`, `data_norm_out_refresh`, `data_norm_out_sequences`, `data_norm_out_ema`. They are whitelisted in `run_arm.py`.
- **Tests.** 73 pass, including 4 new ones:
  - identity B reproduces PD exactly;
  - a stiff output direction is damped;
  - the statistics pass has no side effects;
  - config validation.
- **Next.** The one-step two-sided test sets β and the label source. Then full runs at 1M and 4M batches with LR brackets, paired with the trajectory and batch-size cohorts.

**Batch-size result, the exact-GN reference, and a batch-dependent power test (2026-09-27 03:44 CDT).**
- **Batch-size study, closed** (best-LR brackets, 4M batch).
  - Muon 3.9524 (LR 0.014; 0.007 and 0.028 worse); PD α ¼ 3.8847 (0.02; 0.01 and 0.04 worse); S∘PD 3.8478 (0.01; 0.02 at 3.8495).
  - S∘PD − Muon = −0.105 and PD − Muon = −0.068, against −0.029 and −0.018 at 1M. The prediction above holds.
- **Two-sided, one step.** β ¼–½ adds +3% on PD's state (GN labels slightly ahead of data labels) and +6–10% on Muon's. A paired 1M training test is running on A4000 (`soaudit_twosided_20260927`: PD α ¼ control vs α ¼ β ¼ with GN labels, LR 0.01).
- **Correction to the gap map.** Along the updates actually taken, the exact GN curvature is 20–37× the frame-diagonal sum. The frame's stiff, middle and flat parts are coupled at correlation 0.45–0.73, so separate step sizes for them gain ≤ 8% in one step. The frame-diagonal decrement and its "flat directions are 20–100× under-stepped" reading are withdrawn.
- **Exact one-step GN** (`one_step_gn.py`: Lanczos/CG on the exact GN matrix, held-out scoring; OBSERVATIONS.md 03:40).
  - With a fresh 4M-token gradient on 1M-run states, Muon gets 0.36–0.55 of the GN one-step decrease and PD α ½ 0.68–0.70.
  - With a 1M gradient: Muon 0.49–0.68, PD α ½ 0.79–0.87.
  - The gap is not per-matrix allocation and not the output Kronecker factor. K-FAC and EKFAC do worse than Muon.
  - Newton scaling of the momentum is harmful: for M', the best GN direction loses to Muon by 1.4–1.9×.
- **α bracket at 1M (training).** α ¼ @0.01 stays best (3.6891). α ⅛ @0.007 3.6955, α ⅜ @0.01 3.6939, α ⅜ @0.014 3.6979, α ½ @0.01 3.7131; α ⅜ @0.02 and α ½ @0.014/0.02 are running. The earlier one-step prediction (α ⅜ best near LR 0.017) fails so far. One-step direction quality overstates the value of stronger preconditioning at 1M.
- **Decision: test whether the best input power grows with batch.**
  - Hypothesis: curvature scaling is worth more when the gradient is accurate. In one step at the 4M states with a 4M gradient, PD α ½ gets 0.89–0.90 of GN against α ¼ 0.83–0.87.
  - Arms (cohort `soaudit_alpha4m_20260927`, seed 260925, batch 4M, 368 steps; β = 0 code paths are unchanged from `soaudit_batch4m_20260927`):
    - PD α ½ at LR 0.02 and 0.04 on g20 (RTX 6000 Ada), paired with PD α ¼'s Ada bracket;
    - S∘PD α ½ at LR 0.01 and 0.02, plus an S∘PD α ¼ @0.01 hardware control, on priv-g14 (L40S).
  - **Prediction.** At 4M, best-LR PD α ½ beats best-LR PD α ¼ (3.8847) by ≥ 0.005, and S∘PD α ½ beats its L40S α ¼ control by ≥ 0.005.
  - A tie or a loss means the one-step power advantage does not transfer at 4M either. That would put the gap in multi-step dynamics (momentum, sharpness adaptation) rather than in the preconditioner's power.

**Two-sided data norm: first training result and follow-up (2026-09-27 04:23 CDT).**
- **Paired 1M test** (`soaudit_twosided_20260927`, A4000, seed 260925, LR 0.01): two-sided α ¼ β ¼ with GN labels 3.68303, PD α ¼ 3.68824. **TS − PD = −0.0052.**
  - The lead was −0.029 at step 100, −0.016 at 300 and −0.005 to −0.006 from step 550 on, and it survived the cooldown.
  - The A4000 PD control is +0.0013 from the Ada PD run of the same seed.
  - For comparison, S∘PD − PD = −0.011 at 1M. The GN output factor gives about half of SOAP's gain on top of PD, at a fraction of its cost (B^−β recomputed every 10 steps).
  - One step predicted only +3% on PD's state. The training gain is larger than that, perhaps a multi-step effect (damped output-side oscillation).
- **α bracket at 1M, closed except α ½ @0.02.**
  - α ¼ @0.01 is best (3.6891).
  - α ⅜: 3.6939 / 3.6979 / 3.7045 at LR 0.01 / 0.014 / 0.02.
  - α ½: 3.7131 / 3.7125 at 0.01 / 0.014.
  - α ⅛: 3.6955 / 3.6975 at 0.007 / 0.01.
  - The fresh-gradient one-step prediction fails. With the momentum as input, the one-step metric instead ranks α ¼ ≈ α ½, which matches training.
- **Follow-up before any claim** (`soaudit_twosided2_20260927`, A4000):
  - TS at LR 0.007 and 0.014 (bracket);
  - β ½ @0.01;
  - a second-seed pair (PD and TS @0.01, seed 260926).
  - Claim rule as before: every paired Δ < 0 and mean ≤ −0.005 over ≥ 3 pairs on ≥ 2 GPU types. This wave gives 2 pairs on one type.

**Batch-dependent power test: result (2026-09-27 05:17 CDT). The prediction fails.**
- At 4M (best-LR α ¼ baselines):

| Comparison | α ½ | α ¼ | Difference |
|---|---|---|---|
| PD @0.02, Ada | 3.8917 | 3.8847 | +0.0070 (@0.04 still running) |
| S∘PD @0.01, same L40S | 3.8683 | 3.8454 | +0.0229 |

  - The L40S α ¼ control is −0.0024 from the A6000 run of the same arm.
- At 1M the α ½ bracket is also closed: 3.7131 / 3.7125 / 3.7205 at LR 0.01 / 0.014 / 0.02, against α ¼ @0.01 3.6891.
- So stronger input power loses in training at both batch sizes, although it gives better one-step directions on fixed states. That holds with fresh gradients, and at 4M with the momentum as input too.
- α ½ runs at about twice PD α ¼'s gradient norm throughout (4M: 1.2–1.4 vs 0.45–0.8), and PD's states are much sharper than Muon's (top GN Ritz value 134 vs 15 at step 500).
- **Reading.** The state adapts to the preconditioner: sharpness grows along the directions it steps less in, until the edge of stability is reached again. One-step direction quality on a fixed state cannot see that closed loop. Preconditioning strength has to be judged in training (or in a model that includes the stability constraint), not by one-step gains.

**Independent review of the audit's GN claims (2026-09-27 05:41 CDT; read-only subagent, full text summarized here).**
- **Upheld.** The fixed-state, fresh-gradient ranking (GN > PD > Muon, with GN's lead growing with gradient batch). The quadratic model's accuracy at each direction's own optimum. The momentum reversal. The withdrawal of the frame-diagonal decrement.
- **Corrections I accept.**
  1. **GN's advantage exists only at its own optimal scale c.**
     - The optimizers run at 1.6–1.9 c. Interpolating the 0.5c/c/2c line search to 1.75c, GN is worse than Muon on both 4M states, ties on PD's 1M state, and leads 1.35× only on Muon's 1M state.
     - The GN step's trust region is small. A GN optimizer would need a line search or trust region, not a fixed learning rate at the edge of stability.
     - The fixed-state result also does not reproduce the training batch effect for PD vs Muon: PD's share relative to Muon's barely moves with gradient batch. Only GN's lead grows.
  2. **The 20–37× frame-diagonal factor is mostly across matrices.** Muon's cross-matrix coherence is 12–16×, leaving ~2–2.5× within a matrix. K-FAC's poor showing may partly reflect the per-token variant (5–40× off for k and v) and a damping grid that ends at its best value.
  3. **Coherence is not the explanatory variable by itself.** GN on the momentum has low coherence (2.2–2.4) yet loses; K-FAC (8.2) loses to Muon (16.4).
     - Per-block methods are block-Jacobi preconditioners, which stall on global modes without a coarse correction.
     - Abreu et al. report that layerwise GN nearly matches full GN, which contradicts a pure cross-matrix reading.
     - The decisive test: exact block GN per matrix / per layer / per kind, with Gauss-Seidel, against full GN, cross-fitted, with an equal Krylov budget.
  4. **Artifacts to fix.**
     - Gradient clipping (norm 1.0) fires at very different rates across compared arms:

| Arm | Steps clipped |
|---|---|
| TS, 1M | 58% |
| PD, 1M | 12% |
| PD α ½, 4M | 85% |
| PD α ¼, 4M | 8% |
| S∘PD, 4M | 94–96% |

     - With polar updates, clipping only reweights the momentum, but it moves with the treatment. The one-step M′ also mixes an unclipped fresh gradient into a momentum of clipped ones.
     - The 4M-gradient GN references (damping 1e-4..3e-4 ρ) are not converged: 48 → 96 steps adds 9.5%, and shares above 1 on S∘PD states show it. The "top-heavy" GN profile may partly be Krylov truncation.
     - GN variants and scales are selected on the scored sequences; they should be cross-fitted.
     - The k = 32 deflation collapse and the 48-parameter per-matrix fit on 64 + 64 sequences are underpowered negatives.
  5. **An edge-of-stability invariant to test.** Muon's LR × top GN eigenvalue is 0.104 at 1M and 0.108 at 4M. For PD it should be computed in PD's coordinates (R G R). Also score the α ¼ and α ½ directions on each other's 4M states, to separate state adaptation from multi-step dynamics.
- **Decisions (priority order).**
  1. Offline checks: block-GN comparison; the invariant and cross-state α scoring; convergence (48/96/192), cross-fitted scoring, and damping grids extended to plain gradient descent.
  2. A momentum sweep at 4M: Muon with β ≈ 0.81 (the momentum half-life held constant in tokens) and 0.9, a Nesterov arm, a no-clip control, and PD at the best β. `run_arm.py` needs `muon_momentum` and `grad_clip` in TREATMENTS.
  3. Two-sided confirmation: let the running bracket and seed wave finish; add a placebo output factor (B's spectrum in a random basis), data labels, and clipping-matched pairs. The 4M TS pair is already running on g20.
  4. GN-refined Muon only if warm-started CG with momentum or mixed inputs recovers ≥ half the gap within ≤ 4 GN products and holds up at 1.5–2 c. Momentum would then go on the CG solution, not its input.

**Momentum sweep at 4M, and the warm-CG test (2026-09-27 05:44 CDT).**
- **Warm-started CG** (Muon 1M @500, fresh 1M gradient, damping 1e-3 ρ). Muon's direction goes from 0.71 of GN to 0.83 / 0.86 / 0.77 / 0.82 / 0.87 after 1 / 2 / 4 / 8 / 16 GN products; PD α ½ goes from 0.82 to 0.90 / 0.83 / 0.82 / 0.84 / 0.87.
  - The score is not monotone: CG minimizes the curvature-sample quadratic, not the held-out one.
  - It does not meet the reviewer's bar (half the gap robustly within ≤ 4 products), even with a fresh gradient. The GN-refined Muon (ii) stays shelved.
- **Momentum sweep** (`soaudit_mom4m_20260927`, priv-g14 L40S, running). At 4M, β 0.95 averages over ~20 steps = 80M tokens, and the warmup is 13 steps. Arms:
  - Muon @0.014 with β 0.81 (the same momentum half-life in tokens as at 1M) and β 0.9;
  - Muon with Nesterov momentum;
  - Muon without clipping;
  - S∘PD α ¼ @0.01 without clipping. The clipped control is 3.8454; S∘PD clips on 94–96% of steps.
- **Predictions.**
  - If stale averaging limits Muon at 4M, β 0.81 or 0.9 beats β 0.95 (3.9524) by ≥ 0.005.
  - If S∘PD's 4M lead depends on clipping, the no-clip arm loses ≥ 0.005 against 3.8454.
  - Muon without clipping should be within ±0.003 of the clipped run (only 7% of its steps clip).

**4M results: shorter momentum helps Muon, and the two-sided gain grows with batch (2026-09-27 06:18 CDT).**
- **Muon, β 0.81 @0.014** (momentum half-life held at 13.5M tokens, as β 0.95 at 1M; L40S): 3.9437, against β 0.95's 3.9524. **−0.0087.** The prediction (≥ 0.005) holds.
  - The lead is large early (−0.09 to −0.12 at steps 100–150, when β 0.95's ~20-step memory is 5% of the run), about zero at step 300, and −0.010 / −0.009 after the cooldown.
  - One seed and one LR; the LR bracket at β 0.81 is next.
- **Two-sided at 4M, TS α ¼ β ¼ with GN labels @0.02** (Ada, paired with PD α ¼ @0.02 on the same node): 3.8688 vs 3.8847. **−0.0160**, against −0.005 at 1M.
  - Unlike at 1M, the lead builds mid-run: −0.002 at step 100, −0.020 at 200, and −0.016 to −0.018 from 250 to the end.
  - The output-side factor gains ~3× more at 4M, as the input side (PD) did. TS @0.01 is running for the bracket.
  - The 1M bracket is flat: 3.6827 / 3.6830 / 3.6864 at LR 0.007 / 0.01 / 0.014, against PD @0.01 3.6882 on the same A4000s.
  - The second-seed pair (seed 260926) has PD at 3.6918; TS is still running.
  - Placebo, data-label and no-clip controls are running at 1M.
- **Edge-of-stability check in each optimizer's own coordinates** (`eos_invariant.py`, LR × s × λ_max(R G R); partial):

| State | LR × s × λ_max |
|---|---|
| Muon 1M @500 | 0.104 |
| Muon 4M @183 | 0.108 |
| PD 1M @500 | 0.065 |

  - For PD, λ_max(R G R) = 10.8 (the raw value is 134) and s = 0.60.
  - So PD's 9× "sharper" raw state is mostly a coordinate effect: in its own geometry PD sits near the same edge as Muon, slightly below. The remaining states are running.
- **Next** (principled combinations; one factor at a time against existing baselines):
  - PD and TS with β 0.81 at 4M (g20, Ada, paired with the β 0.95 runs);
  - Muon β 0.81 @0.02 (L40S) for its bracket;
  - later, S∘PD with β 0.81.

**Block GN vs full GN; clipping inside S∘PD; the two-sided 4M bracket (2026-09-27 06:42 CDT).**
- **Block-diagonal GN recovers almost all of full GN** (`one_step_blockgn.py`: fresh 4M gradient, 128 curvature sequences, cross-fitted damping and scale, 16 Lanczos steps per block and 96 for full). Decrease ×1e-3:

| State | Muon | PD α ½ | Full GN | Per kind (6 blocks) | Per layer (8 blocks) |
|---|---|---|---|---|---|
| Muon 1M @500 | 14.5 | 18.1 | 23.3 | 22.6 (0.97) | 21.9 (0.94) |
| Muon 4M @183 | 28.9 | 31.8 | 32.2 | 32.2 (1.00) | 30.3 (0.94) |

  - Cross-block coupling (between kinds, or between layers) is not needed for GN's one-step advantage, as Abreu et al. report for layerwise GN.
  - **The "cross-matrix redundancy is the core of the gap" reading is refuted.** Coherence is a property of per-matrix directions, but block GN reaches full GN without cross-block information.
  - Per matrix (48 blocks) and block Gauss-Seidel are running.
- **Clipping is part of S∘PD's 4M result.** S∘PD @0.01 without clipping: 3.8657, against 3.8454 clipped (same L40S). That is +0.020, and +0.09 at steps 50–150.
  - At 4M, S∘PD's gradient norm is ≫ 1, so clipping at 1.0 makes its momentum an average of normalized gradients. Muon and PD α ¼ clip on only 7–8% of steps.
  - Test queued on g20 (`soaudit_clip4m_20260927`): Muon and PD with grad_clip 0.1, i.e. normalized gradients before momentum.
- **Two-sided 4M bracket.** TS 3.8658 (@0.01), 3.8688 (@0.02); best-vs-best −0.019 against PD α ¼ (3.8847). TS with β 0.81 is running.

**Two-sided data norm: second seed and β ½ (2026-09-27 07:17 CDT).**
- Seed 260926, A4000, LR 0.01: TS 3.68197 vs PD α ¼ 3.69182. **Δ = −0.0099.** With seed 260925 (Δ = −0.0052), both pairs are negative; the mean is −0.0075.
- β ½ (seed 260925): 3.68515, Δ = −0.0031. β ¼ is better.
- The placebo basis ran ahead of the real TS on no step: +0.008 to +0.012 against PD through step 300 (final pending).
- Confirmation on a second GPU type (A6000, seeds 260927 and 260926) is submitted: `soaudit_twosided3_20260927`.

**Checks queued (2026-09-27 07:18 CDT).**
- **Is β 0.95 also too long at 1M?** (`soaudit_mom1m_20260927`, A4000) Muon @0.007 with β 0.95 vs β 0.9. If β 0.9 wins at 1M too, the 4M momentum gain is a general mistuning, not a batch effect.
- **Block GN with 48 Lanczos steps per block** (per kind and per layer; Muon and PD 1M states). This checks convergence. On PD's state, per kind reached only 0.77 of full GN with 16 steps per block, against 0.97–1.00 on Muon's states, and PD's states are harder to converge.

**Principle under test: stronger preconditioning needs fresher averaging (2026-09-27 07:20 CDT).**
- **Evidence.**
  - In one step, the best input power rises from ¼ to ½–¾ as the input goes from the stale momentum (c = 0.95) to mostly-fresh g + cM (c ≤ 0.25) (`gnmix/`).
  - In training at β 0.95, α ½ lost to α ¼ at both batch sizes.
  - At 4M, β 0.9 beats β 0.95 by 0.031 for Muon.
- **Test** (`soaudit_mom4m4_20260927`, g20, after the clip tests):
  - PD α ½ @0.02 with β 0.9, against PD α ¼ @0.02 with β 0.9;
  - TS with β_out ½ @0.01 and momentum 0.9, against TS with β_out ¼ and momentum 0.9.
- **Prediction.** At momentum 0.9, α ½ is within ±0.003 of α ¼ or better, where at 0.95 it lost by +0.007 (PD, 4M). If α ½ still loses by ≥ 0.005, staleness is not what limits preconditioning strength, and the sharpness/edge-of-stability explanation becomes more likely.

**Gauss-Seidel result, and momentum 0.81 for PD and TS at 4M (2026-09-27 07:38 CDT).**
- **One Gauss-Seidel sweep of exact per-matrix GN solves gets 0.99 of full GN** on Muon's 1M state (fresh 4M gradient, cross-fitted). Per-matrix Jacobi gets 0.86.
  - What a per-matrix method lacks relative to GN is sequential: each matrix should see the gradient after the matrices before it have moved.
  - This is a principled target. Whether a cheap surrogate exists is open, for example propagating the earlier layers' input shifts into the later layers' updates without extra GN products.
  - Gauss-Seidel on Muon's 4M state and PD's 1M state is queued.
- **β 0.81 at 4M** (@0.02, Ada):

| Optimizer | β 0.81 | β 0.95 | Difference |
|---|---|---|---|
| PD α ¼ | 3.8747 | 3.8847 | −0.010 |
| TS | 3.8550 | 3.8688 | −0.014 |
| Muon | 3.9437 | 3.9524 | −0.009 |

  - At β 0.81 the preconditioner gaps are unchanged: PD − Muon = −0.069 and TS − Muon = −0.089.
  - Muon's best so far is β 0.9 (3.9218). The β 0.9 runs for PD, TS and S∘PD are running.

**β 0.9 at 4M: the preconditioner gaps survive momentum retuning (2026-09-27 08:10 CDT).**

| Optimizer | β 0.95 | β 0.9 | Change |
|---|---|---|---|
| Muon @0.014, L40S | 3.9524 | 3.9218 | −0.031 |
| PD α ¼ @0.02, Ada | 3.8847 | 3.8630 | −0.022 |
| S∘PD @0.01, L40S | 3.8454 | 3.8314 | −0.014 |

- Gaps to Muon at β 0.9: PD −0.059 and S∘PD −0.090, against −0.068 and −0.107 at β 0.95 (same hardware for S∘PD).
- Momentum retuning shrinks the gaps by about 15%, and they stay 2–3× their 1M size.
- Still to come: Muon's LR bracket at β 0.9 (@0.02, queued on priv-g14) and TS at β 0.9 (running).
- **Gauss-Seidel on Muon's 4M state:** 0.97 of full GN, against per-matrix Jacobi 0.91.

**4M with tuned momentum (β 0.9) for all; the 1M two-sided gain without clipping (2026-09-27 08:43 CDT).**
- **4M, β 0.9, best LR tested:**

| Optimizer | Final loss | vs Muon |
|---|---|---|
| Muon (L40S; 3.9218 @0.014, 3.9211 @0.02) | 3.9211 | |
| PD α ¼ @0.02 (Ada) | 3.8630 | −0.058 |
| TS @0.01 (Ada) | 3.8500 | −0.071 |
| S∘PD @0.01 (L40S) | 3.8314 | −0.090 |

  - The 4M batch-size result survives momentum retuning. The gaps are ~15% smaller than at β 0.95 and still 3× their 1M size.
  - TS − PD = −0.013 at β 0.9, against −0.019 best-vs-best at β 0.95.
- **1M two-sided pair without clipping (A4000, seed 260925):** TS 3.68755 vs PD 3.69828, **Δ = −0.0107**, against −0.0052 with clipping.
  - Clipping helps PD more (+0.010 without it) than TS (+0.0045).
  - The two-sided gain does not depend on clipping.
- **Two-sided confirmation so far:** Δ = −0.0052 and −0.0099 (A4000, seeds 260925 and 260926). A6000 seed 260927: PD 3.68869, TS finishing.

**Two-sided confirmation on a second GPU type (2026-09-27 08:43 CDT).**
- A6000, seed 260927, LR 0.01: TS 3.68260 vs PD α ¼ 3.68869, **Δ = −0.0061.**
- Paired Δ so far (TS − PD, 1M, LR 0.01): −0.0052 (seed 260925, A4000; the selection run), −0.0099 (260926, A4000), −0.0061 (260927, A6000).
  - All three are negative; mean −0.0071, SD 0.0025, t = −4.9 (one-sided p ≈ 0.02).
  - The two confirmation pairs alone average −0.0080.
  - The fourth pair (seed 260926 on A6000) is running.
- Once that pair lands, the two-sided output factor meets the project's claim rule: every Δ < 0, mean ≤ −0.005, p < 0.05, ≥ 3 pairs on ≥ 2 GPU types.
- Supporting evidence:
  - It needs the GN output basis: the placebo loses to PD.
  - It holds without clipping (−0.0107).
  - Its gain grows at 4M (−0.013 to −0.019).

**Second seed at 4M (2026-09-27 08:44 CDT).** All 4M results so far are one seed. `soaudit_seed4m_20260927` repeats the tuned-momentum comparison with seed 260926, all four optimizers on one L40S node: Muon @0.014, TS @0.01, PD α ¼ @0.02 and S∘PD @0.01, all at β 0.9. It runs on priv-g14 after the Muon β 0.9 bracket.

**Clipping at 0.1 for PD at 4M; Muon β 0.9 on a second seed (2026-09-27 09:29 CDT).**
- PD α ¼ @0.02 with grad_clip 0.1 (Ada, β 0.95): 3.8746 vs 3.8847 with clip 1.0, **−0.010**.
  - At 0.1 the clip binds on every step (median gradient norm 0.4–0.75 after step 25), so the momentum averages normalized gradients.
  - The gain matches β 0.81's (−0.010) and is half of β 0.9's (−0.022). Muon with clip 0.1 is running on g20.
- Muon @0.014, β 0.9, seed 260926 (L40S): 3.9239, against 3.9218 for seed 260925. The 4M Muon baseline is stable to 0.002 across seeds.

**Decision argument: is the momentum's staleness the curvature-transport term? (2026-09-27 09:29 CDT)**
- **Observation.**
  - On Muon's 1M state at step 500, the next momentum's one-step value is about a third of a fresh gradient's. Cross-fitted decrease ×1e-3 is 4.5 for Muon on M' against 14.5 on a fresh 4M gradient; PD α ½ gives 4.5 against 18.1.
  - Curvature maps gain on fresh gradients but lose on the momentum. Staged Muon and PD gain +71% and +47% on the fresh gradient, and lose −29% and −37% on M'.
- **Model.** The Muon buffer sums gradients taken at past weights. In the GN model, M' = β M_t + g ≈ M*' − H Q_t:
  - M*' is the same batches' gradients at the current weights, and Q_t = Σ_{k≥1} β^k (W_t − W_{t−k}).
  - So M' + H Q_t keeps the momentum's noise averaging and removes its staleness. This is the explicit form of implicit gradient transport (Arnold et al. 2019), and of the STORM/MARS correction applied recursively.
- **Stability caveat, stated before any result.** On a quadratic, full transport turns heavy ball into gradient descent with step η/(1−β).
  - That descent is stable only for ηh < 2(1−β), against heavy ball's 2(1+β).
  - Heavy ball's lag is what keeps stiff directions stable at large effective steps. A fresh, averaged momentum therefore pairs naturally with a curvature-equalizing map (a GN or Newton-like step), not with plain Muon.
  - MARS's small correction weight (γ ≈ 0.025) is consistent with this. A one-step gain is thus not evidence for a training gain, and this test is measurement only.
- **Test** (`transport_test.py`; runs M and PD α ¼ @1M with `muon_track_displacement`, `soaudit_transport_20260927`, step 500):
  - **Replay.** The run's own last 100 training batches are re-evaluated at W_t, with the same data order and bf16 precision, and the logged clip factors. This gives the exact stale-free M*_t, and the staleness S = M_t − M*_t with no sampling noise.
  - **Transport predictions of β S.** −G Q (exact GN), −H Q (true Hessian, by central differences), the per-matrix GN blocks, and K-FAC's −B Q C.
  - **One step.** Cross-fitted decreases of GD, Muon, PD and damped-GN maps on g, M', M*', M' + X, and the replay mean.
- **Predictions.**
  1. The linear model holds over the momentum window: cos(β S, −G Q) ≥ 0.7 in total, and the best multiplier s* is in [0.7, 1.3]. If cos < 0.5, staleness is not a linear curvature effect. The causes would be nonlinearity, the untracked embedding and head drift, or non-GN Hessian terms.
  2. H Q predicts at least as well as G Q.
  3. Per-matrix blocks keep most of it (cos within 0.1 of full G). K-FAC is clearly worse, as frame-diagonal curvature was for real updates.
  4. One step:
     - Muon and PD gain ≥ 2× on M*' over M'.
     - GN maps on M*' beat GN on a fresh 1M gradient, because of less noise and no staleness.
     - M' + G Q recovers most of M*'.
- **What would follow.** If predictions 1 and 4 hold, what an efficient second-order method needs from its momentum is known and measured. That is the fresh averaged gradient, paired with a curvature-equalizing map at a stable step; cheap transports (blocks, Kronecker factors, gradient differences as in MARS) can then be judged against the exact one. If prediction 1 fails, momentum and curvature cannot be separated this way, and the next measurement is how the staleness is distributed over the curvature spectrum.

**Does momentum set Muon's stability edge? (2026-09-27 09:36 CDT; prediction before the measurement)**
- **Observation.** LR × λ_max(G) is 0.104–0.108 for Muon at β 0.95 on every state measured past warmup: 1M at steps 500 and 900, and 4M at step 183. This is close to 2(1 − β) = 0.10.
- **Test** (`eos_invariant_mom.json`, 4M kept states):
  - Muon at step 183 with β 0.9 and 0.81 (LR 0.014), Nesterov (β 0.95), and β 0.9 at LR 0.02;
  - Muon at step 330 for β 0.95, 0.9 and 0.81;
  - PD α ¼ at step 183 with β 0.9 and 0.81.
- **Readings.**
  - If the invariant scales like 2(1 − β) (≈ 0.2 at β 0.9, ≈ 0.38 at β 0.81), momentum's averaging window sets how sharp the state may become. Lower β then buys a sharper, faster-converging state, and a curvature-equalizing map combined with fresher momentum is coherent.
  - If it stays ≈ 0.1 for every β, the edge is set by the update's normalization, not by the momentum.
  - At fixed β, LR 0.02 against 0.014 should leave the invariant unchanged if it is an edge.

**First momentum-edge point, and a sharper test (2026-09-27 09:45 CDT).**
- **Muon 4M @183, β 0.9:** LR × λ_max(G) = 0.098, against 0.108 at β 0.95. The 2(1 − β) reading (≈ 0.2) is refuted. Raw sharpness times LR is not set by the momentum window.
- **The right quantity is in the optimizer's own linearized dynamics.**
  - Muon and PD are nonlinear maps f of the momentum. A perturbation of the weights evolves as heavy ball with operator LR J G, where J = df/dM at the next momentum. J includes Newton-Schulz's normalization and, for PD, the whitening R.
  - Heavy ball is stable iff every eigenvalue of LR J G lies below 2(1 + β): 3.90 at β 0.95, 3.80 at β 0.9, 3.62 at β 0.81. Nesterov's bound is 2(1 + β)/(1 + 2β), 1.34 at β 0.95.
  - `eos_linearized.py` computes LR μ_max(J G) by Arnoldi on the exact GN, with J by forward-mode AD through an FP32 copy of each run's map.
- **Prediction.** If each state sits at the edge of its own linearized dynamics, LR μ_max / threshold ≈ 1 (within ±30%) for Muon at every β and batch size, and for Nesterov against its lower bound.
  - For PD, the same holds if PD also runs at its edge. The raw invariant suggested PD runs below it: 0.064 at 1M, 0.026 at 4M.
  - A ratio well below 1 for the better optimizers would mean their LR is limited by something other than linear stability.

**Linearized edge: the first converged states (2026-09-27 10:16 CDT).** `eos_linearized.py` (40 Arnoldi steps, 256 curvature sequences). LR × μ_max(J G), with J the Jacobian of the run's own map at the next momentum:

| State | LR μ_max | Heavy-ball bound 2(1 + β) | Ratio | Top μ (×LR) |
|---|---|---|---|---|
| Muon 1M @500 | 3.78 | 3.90 | **0.97** | 540, 512, 487 (3.78, 3.58, 3.41) |
| PD α ¼ 1M @500 | 3.28 | 3.90 | 0.84 | 328, 306, 300 |

- J is symmetric to rounding, so the spectrum of J G is real, as found.
- **Prediction met for Muon.** Its state sits at 97% of the stability edge of its own linearized dynamics: heavy ball with the Newton-Schulz Jacobian as preconditioner. Several modes sit near the edge, not one.
- **PD is at 84%, below its edge.** This matches its lower raw invariant.
- The smoke value (≈ 15) came from a 16-sequence curvature set and 8 steps: the sample GN's top eigenvalue is inflated and Arnoldi had not converged. It is superseded.
- **What this frame gives.**
  - The optimizer is a preconditioned heavy ball with preconditioner J(M'). The LR is set by the top of J G's spectrum, and progress in each mode scales with LR μ_i.
  - A Newton method has J G = I: every mode sits at the same distance from the edge. How second-order an optimizer is can then be measured as how much of J G's spectrum sits near the edge, and where its bulk lies.
- **Next.** Muon at β 0.9 and 0.81, Nesterov (bound 1.34), LR 0.02, and PD at 4M. Nesterov is the sharp test: its raw invariant equals heavy ball's, but its linear bound is 2.9× lower.

**Linearized edge: the general claim is refuted (2026-09-27 10:35 CDT).** More states, same measurement:

| State (4M @183 unless noted) | LR μ_max(J G) | Own linear bound | Ratio |
|---|---|---|---|
| Muon β 0.95, 1M @500 | 3.78 | 3.90 | 0.97 |
| Muon β 0.95, 1M @900 | 4.16 | 3.90 | 1.07 |
| Muon β 0.95 | 4.25 | 3.90 | 1.09 |
| Muon Nesterov β 0.95 | 3.74 | 1.345 | 2.78 |
| Muon β 0.81 | 12.16 (next modes 7.8, 7.5) | 3.62 | 3.36 |
| PD α ¼ β 0.95, 1M @500 | 3.28 | 3.90 | 0.84 |

- **Prediction failed** for Nesterov and β 0.81. The states do not sit at their own linear bound in general. At β 0.95 they do, to within 10% in three states. Nesterov sits near heavy ball's number, not its own bound.
- What stays constant across β (0.81–0.95), Nesterov, batch size and LR is the raw LR λ_max(G) ≈ 0.07–0.11.
- Newton-Schulz's Jacobian gain scales like 1/|M'|: at β 0.81 the buffer is ~4× smaller, so LR μ(J G) is ~3–4× larger at equal sharpness. States above their linear bound are held by the map's saturation, a nonlinear limit cycle.
- **The edge is set by the normalized update (LR per step along any rank-one direction), not by linear heavy-ball stability.** The Jacobian frame remains a useful description of the β 0.95 states only. The earlier "prediction met for Muon" entry stands as a measurement, not as a principle.

**Stronger preconditioning with fresher averaging: prediction met (2026-09-27 10:38 CDT).** PD α ½ @0.02 with β 0.9 at 4M (Ada, seed 260925): **3.8589**, against PD α ¼ @0.02 with β 0.9: 3.8630 (Δ = −0.0041).
- At β 0.95 on the same hardware, α ½ lost by +0.0070 (3.8917 vs 3.8847). The swing is 0.011.
- The 07:20 prediction was "within ±0.003 of α ¼ or better at β 0.9". It is met: α ½ now wins.
- **Principle supported: how strongly to precondition depends on how fresh the averaged input is.** Batch size, averaging window and preconditioner power are linked:
  - The flat directions carry the persistent signal. Their per-step noise falls with batch size, so a shorter window suffices.
  - A shorter window keeps their directions current, and stronger amplification of flat directions then pays.
  - One step shows the same thing: the best input power rises from ¼ to ½–¾ as the input gets fresher.
- The TS arm (β_out ½ with momentum 0.9) runs next on g20. One seed; the α ½ vs α ¼ difference at β 0.9 is within the seed spread (~0.002–0.005).

**Two-sided data norm meets the claim rule (2026-09-27 10:56 CDT).** Fourth pair (A6000, seed 260926, 1M, LR 0.01): TS 3.68400 vs PD α ¼ 3.69113, **Δ = −0.0071**.
- All four paired Δ (TS − PD):

| Pair | Hardware | Seed | Δ |
|---|---|---|---|
| Selection | A4000 | 260925 | −0.0052 |
| Confirmation | A4000 | 260926 | −0.0099 |
| Confirmation | A6000 | 260927 | −0.0061 |
| Confirmation | A6000 | 260926 | −0.0071 |

  Mean −0.0071, SD 0.0020, t = −6.95 (one-sided p ≈ 0.003). The three confirmation pairs alone average −0.0077.
- The rule is met: every Δ < 0, mean ≤ −0.005, p < 0.05, ≥ 3 pairs on 2 GPU types.
- Controls:
  - The placebo output basis trails PD.
  - Data labels keep 60% of the gain.
  - Without clipping, Δ = −0.0107.
  - At 4M, β 0.9: −0.013 (seed 260925) and −0.0072 (seed 260926, same hardware).
- **The GN output factor, L = (B / mean + 1e-3)^-¼ applied on the output side of PD's update, is a confirmed improvement over PD at 1M and 4M.**

**4M second seed complete (β 0.9, L40S, seed 260926):** Muon 3.9239, PD α ¼ 3.8617 (−0.062), TS 3.8545 (−0.069), S∘PD 3.8273 (−0.097). Seed 260925 gave −0.059, −0.072 and −0.090.

**Low momentum at 4M:** Muon β 0.5 @0.014 (A6000): 4.1166, about +0.19 behind β 0.9 (3.92, L40S; the A6000 β 0.9 reference is running). Muon's map needs momentum at 4M, so a "fresh-gradient staged Muon" is not viable as is. β 0 is running.

**Next test of the strength principle: both sides at ½, and S∘PD at ½, with fresher averaging (2026-09-27 10:59 CDT).**
- **Why.** PD α ½ beat α ¼ at 4M once momentum was 0.9 (−0.004), after losing at 0.95 (+0.007). The principle predicts the same for:
  - the output side (TS β_out ½; running on g20);
  - both sides at ½ (TS α ½ β_out ½), which is the full Kronecker whitening in front of polar;
  - S∘PD α ½, which lost at β 0.95 (3.8608 @0.02 against α ¼'s 3.8454).
- **Cohort** `soaudit_strength4m_20260927`, 4M, seed 260925, β 0.9, kept checkpoint 183:
  - priv-g14 (L40S): S∘PD α ½ at LR 0.02 and 0.01. Reference: S∘PD α ¼ @0.01, 3.8314 (L40S).
  - g20 (Ada), after TS β_out ½: TS α ½ β_out ½ at LR 0.01 and 0.02. References: TS α ¼ β_out ¼ @0.01, 3.8500; PD α ½ @0.02, 3.8589 (both Ada).
- **Predictions.**
  1. The better S∘PD α ½ run is within +0.003 of S∘PD α ¼, or better.
  2. The better TS α ½ β_out ½ run is within +0.003 of TS α ¼ β_out ¼, or better.
  3. It beats PD α ½ by ≥ 0.005, since the output factor still adds at the stronger setting.
  4. If both sides at ½ lose by ≥ 0.005 to the ¼ settings, the gain from fresher averaging is one-sided (input), or ½ on both sides over-whitens.
- One seed per arm. These are directional tests of a principle, not claims.

**Decision argument: a true Gauss-Newton optimizer in training, as the measured reference for the gap (2026-09-27 11:02 CDT; for review before any implementation).**
- **Why one-step scores cannot answer the question.** Every piece of today's evidence says one-step decreases mostly reward correcting the optimizer's own stiff-mode oscillation:
  - In the stable phase every optimizer's step overshoots ~2× along its own direction (loss minimum at 0.44–0.57 of the step).
  - The current full-batch gradient is dominated by that oscillation in every curvature bin, and the momentum carries almost none of it.
  - The stale-free momentum scores like a fresh gradient in one step (PD α ½: 14.6 vs 12.4 fresh vs 2.6 on the actual momentum), because it re-adds the oscillation that one step can correct.
  - The one-step "GN gap" (e.g. GN 19.6 on a fresh gradient vs Muon 4.6 on its momentum, ×1e-3) therefore bounds nothing about training. Neither the GN paper's claim of a large gap at large batch nor our distance from it has been measured in our setup.
- **What.** GN in training, 4M-token batch (where the preconditioner gaps are largest):
  - Each step, the fresh gradient g of the 48 hidden matrices is mapped to x = −(G_S + λ ρ I)^-1 g by k Krylov (Lanczos/CG) steps on the exact GN matrix G_S of all hidden matrices.
  - G_S is estimated on a subsample S of the step's own batch; ρ is the curvature along g. The GN products are forward-mode + reverse-mode, as in `one_step_gn.py`, and can be split across the 4 GPUs.
  - The update is W ← W(1 − η λ_wd) + η_t x, with η_t the usual schedule shape and peak η ≈ 1 (a Newton step).
  - Embeddings, head and gains use AdamW as now.
  - Variants: k, S and λ; a light gradient average (β ≈ 0.5) before the solve; per-step backtracking on a held-out microbatch as the safety net.
- **Predictions.**
  1. If the oscillating regime is the main inefficiency, GN-train at 4M beats S∘PD (3.83) by ≥ 0.03.
  2. Its steps stop overshooting: loss minimum along the step near 1, midpoint drop ≈ 0.
  3. If GN-train lands within ±0.01 of S∘PD, today's best preconditioners already take most of what a per-step GN step buys at this batch. The GN paper's gap would then come from elsewhere (horizon, batches ≫ 4M, baseline tuning).
- **Cost and risks.**
  - Cost is ~20–30 GN products per step, about 3–4 h per 4M run on one 4-GPU node.
  - Risks: curvature-subsample noise (G_S from ~0.25M tokens), damping and trust region, explicit-attention memory, and FP32 products next to BF16 gradients.
  - A failed run is still informative only if the step logic is sound, so smoke tests at 1M and short 4M runs come first.
- **Alternatives considered.**
  - Keep tuning preconditioning strength: cheap and running, but it cannot say how far is left.
  - Damp the stiff subspace (Newton on the top GN modes): cheaper, but whether it is worth building depends on prediction 1.
  - The GN paper's own code: not available, and a different setup.
- **Review:** an independent critique is requested before implementation.

**Correction to the 11:02 decision argument (2026-09-27 11:06 CDT).**
- "The GN paper's own code: not available" was wrong. It is public (github.com/natalieabreu/full-gauss-newton; JAX-style, so a PyTorch port is needed here).
- Its outer step is not a Krylov solve:
  - The model is linearized at θ_t.
  - The batch's GN quadratic is minimized by N inner Muon steps, each on a fresh sub-batch, with one forward-mode and one reverse-mode product per inner step.
  - A line search then picks α ∈ {2^(−i/2), i = 0..4} on the true loss.
- So curvature comes from the whole batch, at about 3× a normal step's cost. This is cheaper and more faithful than a Krylov solve on a curvature subsample.
- The paper starts all runs after an AdamW warmup of 5% of Chinchilla tokens, and reports its layerwise variant nearly matching full GN in training. This is consistent with our observation that the next gradient carries cross-layer coupling one step late.
- The design will follow the review; this corrects the record only.

**Review of the GN-in-training decision (independent reviewer, 2026-09-27 11:24 CDT). Verdict: go with changes, after two cheap premise checks. Accepted.**
- **The regime.** The GN paper reports GN ≈ SOAP up to a 4M batch at fixed tokens (150M models, 3B tokens). The 5.4× is steps to loss 3.25 at a 240M-token batch. Our one-step data agree: on the 4M states PD α ½ reaches 0.89–0.99 of full GN.
  - Prediction 3 of the 11:02 argument was misread: a 4M GN run within ±0.01 of S∘PD is what the paper predicts, not evidence that the gap lives elsewhere.
  - A 4M run mostly calibrates the method.
- **The premise is untested, and the numbers cut against it.** "One-step GN gains are mostly oscillation correction" is not supported:
  - The Newton decrement of ḡ inside the top-16 GN modes is 3.3e-3 (Muon) and 1.1e-3 (PD), against GN's one-step decrease of 26.3 / 19.5e-3 on M*'. About 85–95% of GN's one-step gain lies outside the 16 stiffest modes.
  - Whether that remainder is semi-stiff oscillation or persistent bulk signal is exactly what a valley-floor re-score decides.
- **Over-statements corrected:**
  - "The oscillation dominates the gradient in every curvature bin" rests on Kronecker-frame bins, which are coupled 20–37×. The supported claim is narrower: 56% (Muon) and 44% (PD) of ḡ's energy lies in the top-16 GN directions, where cos(M, ḡ) = −0.83 / −0.64; elsewhere it is −0.31 / −0.47.
  - "The stored loss is ordered like the final losses" holds across optimizers at 1M. At 4M, β 0.9 stores more than β 0.95 yet finishes better. The cooldown releases stored loss; it is not the inefficiency.
  - The transport cosine 0.86 splits into 0.955 in the top 16 and 0.73 elsewhere (s* 0.55). That is largely period-2 algebra. The Gauss-Seidel cross-layer gain is plausibly the same global mode, since it reverses on M'.
  - "The normalized update sets the edge" is a description, not a tested mechanism.
  - "Stronger preconditioning pays once averaging is fresher" rests on one seed and Δ = −0.004, inside the seed spread.
- **Design changes for GN in training (when it runs):**
  - **Recipe.** Use the paper's: whole-batch linearization, inner Muon over sub-batches, and a line search with candidates ≤ 1 times the schedule on held-out training data. Warm-start from the previous solution.
  - **Why not Newton scale.** The held-out best scale of damped GN is ~0.5 of the Newton step, and 2× that raises the loss in every `gn/` and `gn2/` state.
  - **Damping.** Use Levenberg-Marquardt, or tie it to the top Ritz value, not to ρ (ρ tracks the oscillating part). Add a KL trust region.
  - **Start and weight decay.** All arms start from one shared warmed checkpoint. Match the per-step weight shrink. Put the head in the solve, or measure its share.
  - **Logging.** Predicted vs actual decrease, α, λ, top Ritz value, CG residual, and the hidden-only vs aux-only midpoint drop.
  - **Fairness.** A fixed tuning budget, two seeds for any claim, and steps-to-S∘PD's-loss alongside GN products and wall-clock.
- **Order:**
  1. Premise checks: one-step GN vs S∘PD/TS at a valley-floor state (short LR anneal from the kept 4M@183 and 1M@500 checkpoints), with GN's decrease split by curvature band; and whether the one-step gap reopens with a 16M-token gradient.
  2. A smoke test of the paper-recipe GN at 1M.
  3. 4M hand-offs from shared checkpoints with short-branch tuning.
  4. Full runs on two seeds.
  5. 16M if the gap reopens.

**Low momentum at 4M, complete (A6000, seed 260925, Muon @0.014), 2026-09-27 11:27 CDT:** β 0.9 3.9214 (L40S: 3.9218), β 0.5 4.1166 (+0.195), β 0 4.3660 (+0.445).
- With β 0.95 (3.9524) and β 0.81 (3.9437) from the earlier cohort, Muon at 4M has a sharp optimum near β 0.9.
- Without momentum its map fails.
- A fresh-gradient (staged) variant of Muon's own map is not viable at 4M. Any fresh-gradient second-order method must get its stability from curvature scaling (GN), not from normalization plus averaging.

**Premise check 1, first state: GN's one-step advantage survives the valley floor and grows (2026-09-27 11:50 CDT).**
- **Setup.**
  - Muon 1M: the original state at step 500, and after a 16-step anneal (`anneal_branch.py`) that releases the LR-maintained excess (held-out loss −0.038).
  - `valley_rescore.py`: fresh 4M-token gradient, cross-fitted over two halves of 512 held-out sequences, damped GN from 64 Lanczos steps. Decrease ×1e-3:

| Direction | Oscillating (step 500) | Share of GN | Valley floor (step 516) | Share of GN |
|---|---|---|---|---|
| GD | 5.72 | 0.22 | 1.37 | 0.11 |
| Muon | 14.46 | 0.55 | 4.92 | **0.40** |
| PD α ¼ | 17.06 | 0.65 | 6.69 | 0.54 |
| PD α ½ | 18.09 | 0.69 | 7.63 | **0.62** |
| Two-sided | 17.09 | 0.65 | 6.99 | 0.56 |
| Damped GN | 26.36 | 1 | 12.38 | 1 |

- **Stiff modes.** The exact Newton decrement of g inside the top-16 GN modes falls from 2.11e-3 to 0.13e-3: the anneal relaxed them. The curvature along g falls from 8.1 to 4.7. The top GN eigenvalue does not fall (14.8 → 15.5).
- **Reading.**
  - At the valley floor every one-step decrease is 2–3× smaller. Much of what one-step methods "gain" at an oscillating state is the excess that an anneal would release anyway.
  - But GN's advantage is not oscillation correction: after the stiff modes are relaxed it is relatively larger (Muon 0.55 → 0.40 of GN, PD α ½ 0.69 → 0.62), and it lives outside the stiff modes (top 16: 1% of GN's decrease).
  - The one-step GN decrease at the floor, 12.4e-3, is large against real progress here (~0.7e-3 per step of normal training).
- **Not yet known:** whether this bulk advantage survives the dynamics (in training it is taken one step at a time from a moving, re-oscillating state), how it depends on batch size (the 16M-gradient scores are running), and whether PD and the 4M states agree (running).

**TS with β_out ½ at momentum 0.9 (4M, Ada, seed 260925), 2026-09-27 11:53 CDT:** 3.8515, against TS β_out ¼ at momentum 0.9: 3.8500 (+0.0015).
- This is within the ±0.003 of the 07:20 prediction, but gives no gain.
- The input side at ½ gained (PD α ½: −0.004 at β 0.9); the output side at ½ does not.
- Both are one seed and inside the seed spread. The one-step picture is similar: the two-sided map at ¼ scores below PD α ½ at every state re-scored today.

**S∘PD α ½ with momentum 0.9 at 4M (L40S, seed 260925), 2026-09-27 11:55 CDT:** 3.8260 @0.02, against S∘PD α ¼ β 0.9 @0.01: 3.8314 (**−0.0054**).
- At β 0.95, α ½ lost by +0.015 (3.8608 vs 3.8454). The swing is 0.021.
- Prediction 1 of the 10:59 entry (within +0.003 or better) is met, and it is the best 4M result so far: −0.096 against Muon β 0.9 (3.9218, same hardware).
- With PD (+0.007 → −0.004) this is the second preconditioner whose stronger input power wins once momentum is 0.9. TS's output side at ½ does not (+0.0015).
- The input side's preconditioning strength should rise as averaging gets fresher. One seed each; the α ½ @0.01 bracket runs next.

**Premise checks at 1M: the gap to GN lives in the bulk and widens with batch size (2026-09-27 12:08 CDT).** `valley_rescore.py`, cross-fitted one-step decreases (×1e-3) and shares of damped GN:

| State | Gradient | GN | GD | Muon | PD α ¼ | PD α ½ | Two-sided | Stiff top-16 decrement |
|---|---|---|---|---|---|---|---|---|
| Muon 1M @500 (oscillating) | 4M | 26.4 | 0.22 | 0.55 | 0.65 | 0.69 | 0.65 | 2.11 |
| Muon 1M @500 (oscillating) | 16M | 31.5 | 0.19 | 0.49 | 0.59 | 0.64 | 0.59 | 2.00 |
| Muon 1M, annealed @516 | 4M | 12.4 | 0.11 | 0.40 | 0.54 | 0.62 | 0.56 | 0.13 |
| Muon 1M, annealed @516 | 16M | **20.5** | 0.07 | **0.30** | 0.42 | **0.50** | 0.44 | 0.12 |
| PD α ¼ 1M @500 (oscillating) | 4M | 22.3 | 0.12 | 0.38 | 0.57 | 0.73 | 0.58 | 0.63 |
| PD α ¼ 1M, annealed @516 | 4M | 9.6 | 0.05 | 0.27 | 0.50 | 0.73 | 0.53 | 0.07 |

- **The premise is decided for the 1M states.** GN's one-step advantage is not oscillation correction. At the valley floor the stiff modes carry 1% of it, and the optimizers' shares fall (Muon 0.55 → 0.40, PD α ¼ 0.65 → 0.54).
- **With a 16M-token gradient at the floor, GN's decrease grows 65% (12.4 → 20.5) while Muon's share falls to 0.30 and PD α ½'s to 0.50.** GN turns the extra signal in low-curvature directions into decrease; the optimizers' fixed maps largely cannot. This is the one-step form of the GN paper's large-batch gap, present here from 4M-token gradients on.
- The ordering is stable: GN > PD α ½ > two-sided ≈ PD α ¼ > Muon ≫ GD. Stronger input whitening captures more (α ½ > α ¼ everywhere), matching the training result that α ½ wins once averaging is fresh.
- **Still open:**
  - the 4M states (running);
  - whether the per-step advantage survives the dynamics (GN-trainer runs: a GN arm against a control that takes the same inner Muon steps on true gradients without linearization, which separates curvature feedback from "more, smaller steps");
  - where in the bulk GN's direction differs (block-GN at the floor, running).

**GN in training (the paper's recipe) vs its no-linearization control, 1M from Muon's step-100 state (2026-09-27 12:33 CDT).**
- **Setup.** `gn_train.py`, 2 GPUs, 80 outer steps (100 → 180). Both arms share:
  - 16 inner Muon steps per 1M-token outer step, on the step's own 64k-token sub-batches;
  - inner LR 0.002 (aux 0.0005), warm start, and a line search (α ∈ {1 … 0.25}) on never-seen training sequences.
  - GN takes its inner gradients from the batch's GN model at the inner iterate (linearized at θ_s). The control takes true gradients at the inner iterate.
- **Validation:**

| Step | GN | Control | Muon baseline (one step per batch) | S∘PD baseline |
|---|---|---|---|---|
| 101 | 5.4588 | 5.4597 | 5.4904 (step 100) | 5.3204 (step 100) |
| 120 | 5.0765 | 5.0403 | | |
| 140 | 4.8016 | 4.7161 | | |
| 150 | | | 5.0121 | 4.8266 |
| 160 | 4.5884 | 4.5305 | | |
| 180 | 4.4662 | **4.4184** | ≈ 4.81 (interpolated) | |
| 200 | | | 4.6822 | 4.5062 |

- **The control beats GN at every checkpoint (by 0.04–0.09), at a third of the cost per step (4.1 s vs 12.8 s).** The large early gain of the GN recipe over one-step-per-batch Muon (~0.35 at step 180) comes from taking many small steps per batch; the 1M batch is far above the critical batch this early. Linearization costs ~0.05.
- **Implication for reading the GN paper.** Each outer GN step contains N sequential small-batch inner steps (b_inner = 32 sequences there). Its "iteration complexity" in outer steps therefore mixes second-order information with N× more sequential steps. The fair baseline at equal outer steps and serial work is the no-linearization control.
- **Caveats:**
  - Early training only (steps 100–180, where the critical batch is small).
  - Fixed inner LR: both arms' line search sits at its floor (α = 0.25) after ~step 140, so both over-step.
  - One seed.
- **Next: one damped Newton step per batch.** The reference that measures what a second-order direction buys at a fixed batch is one damped-GN step per batch (Hessian-free style: CG on the batch's GN with warm start and Levenberg-Marquardt damping), run from the same state as a baseline continuing on the same data. That is what the one-step valley-floor results predict about (GN 2.5× Muon and 1.6× PD α ½ per step at the floor).

**Newton-per-batch reference run: design and predictions (2026-09-27 12:37 CDT).**
- **Why.** After the 12:33 finding, the reference for what a second-order direction buys at fixed batch is one damped GN step per batch, with no extra sequential steps.
- **Trainer** (`newton_train.py`, Hessian-free style):
  - batch gradient in BF16;
  - 16 CG iterations on (G + λ I) x = −g for the 48 body matrices, with exact GN products on 32 sequences per rank, warm-started at 0.95 x_{s−1};
  - Levenberg-Marquardt λ, started at 1e-3 × the curvature along g;
  - a line search over α ∈ {1 … 1/16} times the run's schedule factor, so the cooldown anneals the Newton step as it anneals the baselines' LR;
  - baseline-matched weight decay;
  - AdamW on embeddings, head and gains with the checkpoint's moments and the aux schedule.
- **Smoke test** (2 GPUs, from Muon 1M @500, 4 steps): validation 4.0650 → 4.0341, 9 s per step. The line search sat at α = 0.25 and λ rose (0.011 → 0.038) as the reduction ratio was negative.
- **Run.** From Muon's 1M state at step 500 to the end (969 steps, 4× L40S on priv-g14, same data order). The Muon run from the same state is the baseline (3.7047 final); PD (3.6869) and S∘PD are references.
- **Predictions.**
  1. Within ~20 steps Newton falls ~0.03 below Muon's trajectory. This is the LR-maintained excess that the 16-step anneal released (0.038), not evidence of a better direction.
  2. If the valley-floor one-step advantage (GN 2.5× Muon, 1.6× PD α ½ per step) survives the dynamics, the gap keeps widening past 0.03, and the final loss beats PD's, i.e. < 3.687.
  3. If it does not survive, the final gap to Muon is within the ~0.02 that the cooldown releases for everyone. The likely causes would be noise amplification along flat directions at 1M, or curvature-subsample noise.

**Prediction before measuring: does flattening stop paying at the valley floor and at larger batch? (2026-09-27 12:43 CDT)**
- Polar maps set every singular value of the (whitened) input to 1. This is noise-robust, but it discards the gradient's own spectrum, which a Newton step keeps.
- The earlier one-step "partial orthogonalization loses" result was measured at oscillating states.
- **Prediction.** At the annealed Muon 1M state:
  - U S^p V^T with p = ¼ or ½ beats polar (p = 0), and PD's best input power rises above ½ (¾ or 1);
  - both more clearly with a 16M-token gradient than with a 4M one.
  - At the oscillating state polar stays best or ties, as before.
- If so, the spectrum flattening in Muon-family maps and the input power should both be matched to the gradient's signal-to-noise (batch size, state), as momentum freshness already is.
- `valley_rescore.py --extra-maps --no-gn` on both states.

**Strength cohort, first finals (2026-09-27 12:51 CDT).** 4M, β 0.9, seed 260925:

| Arm | LR 0.01 | LR 0.02 | Reference |
|---|---|---|---|
| S∘PD α ½ (L40S) | 3.8534 | **3.8260** | S∘PD α ¼ @0.01: 3.8314 |
| TS α ½ β_out ½ (Ada) | 3.8729 | running | TS α ¼ β_out ¼ @0.01: 3.8500; PD α ½ @0.02: 3.8589 |

- **S∘PD:** prediction 1 is met by the LR 0.02 arm (−0.0054). The bracket's optimum sits at 0.02 or above, so stronger input whitening wants a larger LR.
- **TS with both sides at ½:** at LR 0.01 it loses by +0.023 to ¼/¼, and by +0.014 to PD α ½.
  - Prediction 2 fails at this LR. The LR 0.02 arm decides whether it fails outright.
  - If that arm also loses by ≥ 0.005, then full two-sided whitening at ½ over-whitens, and prediction 4 holds. The input side and the output side would then want different strengths: the output side at ½ gave nothing with α ¼ either (+0.0015).

**Flattening and input power at the valley floor and at larger batch: prediction half refuted (2026-09-27 13:01 CDT).** Muon 1M states, cross-fitted, ×1e-3:

| State · gradient | Polar (p = 0) | p = ¼ | p = ½ | PD α ¼ | PD α ½ | PD α ¾ | PD α 1 | PD α ½, p = ¼ | Two-sided | K-FAC | EKFAC |
|---|---|---|---|---|---|---|---|---|---|---|---|
| @500 · 4M | **14.46** | 11.24 | 8.95 | 17.06 | **18.09** | 17.64 | 16.50 | 14.55 | 17.09 | 8.32 | 10.29 |
| @500 · 16M | **15.40** | 11.69 | 9.17 | 18.54 | 20.15 | **20.24** | 19.47 | 16.01 | 18.65 | 8.65 | 10.72 |
| floor @516 · 4M | **4.92** | 3.44 | 2.47 | 6.69 | **7.63** | 7.52 | 6.90 | 5.78 | 6.98 | 3.13 | 4.45 |
| floor @516 · 16M | **6.21** | 4.13 | 2.85 | 8.64 | 10.14 | **10.38** | 9.87 | 7.31 | 9.09 | 3.47 | 4.99 |

- **Spectrum flattening: refuted.** Full flattening (polar) beats partial flattening at every state and batch size, by 25–55%. The polar map's flattening is not what separates the optimizers from GN.
- **Input power: supported, weakly.** The best α rises from ½ (4M gradients) to ¾ (16M) at both states. This matches the training result (α ½ wins once momentum is fresher) and suggests α should rise further with batch size.
- **Where GN's bulk advantage lives at the floor** (block GN, Muon 1M annealed, 4M gradient, 128 curvature sequences; share of full GN 11.0):

| Solve | Share |
|---|---|
| Per kind | 0.90 |
| Per layer | 0.83 |
| Exact per-matrix GN | 0.83 |
| PD α ½ | 0.69 |
| Muon | 0.45 |

  - Exact per-matrix curvature recovers most of GN's floor advantage. The Gauss-Seidel sweep is still running.
  - The distance from PD α ½ to exact per-matrix GN (0.69 → 0.83) is within-matrix curvature structure that input whitening plus polar misses.
  - The Kronecker-exact routes do worse, not better: K-FAC 0.28 and EKFAC 0.40 of GN (4M gradient).
  - So the useful per-matrix structure is neither the Kronecker factorization nor the gradient's spectrum; it has to be found in the exact per-matrix GN directions (next: compare them with PD's matrix by matrix).
- **4M valley floor too.** Muon 4M annealed (4M gradient): GN 8.1; Muon 0.49, PD α ½ 0.68, two-sided 0.63, EKFAC 0.37, K-FAC 0.24. At the oscillating 4M state these were 0.82 and 0.90.

**Newton per batch at 1M: interim, and the 4M test (2026-09-27 13:13 CDT).** `newton_train.py` v2: Lanczos with 16 steps, held-out joint choice of damping and step, from the step-500 states. Validation:

| Step | Newton from PD's state | PD's own run | Gap | Newton from Muon's state | Muon's own run | Gap |
|---|---|---|---|---|---|---|
| 500 | 4.0284 | 4.0284 | 0 | 4.0650 | 4.0650 | 0 |
| 525 | 3.9783 | ≈ 4.012 | −0.034 | 4.0035 | ≈ 4.047 | −0.044 |
| 550 | 3.9669 | 3.9952 | −0.028 | 3.9912 | 4.0292 | −0.038 |
| 575 | 3.9598 | ≈ 3.983 | −0.023 | | | |
| 600 | 3.9542 | 3.9716 | −0.017 | | | |

- The first ~20 Newton steps release the LR-maintained excess (prediction 1 met). After that each Newton step gains less than the baseline's (PD arm: ~0.22e-3 per step against PD's 0.47e-3), and the gap closes.
- A per-step held-out gain of ~0.2–0.3e-3, against a first-order promise of 2–15e-3, says most of each Newton step's flat-direction content is batch noise.
- **Reading so far.** At 1M, what the Newton step lacks is averaging, not curvature: PD's momentum averages ~20 batches in the flat directions where GN's one-step advantage lives, while Newton uses one fresh 1M batch.
  - This is consistent with GN's one-step advantage growing with gradient size (4M → 16M) and with the paper's gap living at large batch.
  - Final numbers pending; prediction 3 (no durable gain at 1M) is the current trend.
- **The 4M test** (submitted, A6000 queue). Newton from Muon's and PD's 4M states at step 183 (64 curvature sequences per rank), against their own runs (finals 3.9524 and 3.8847).
- **Predictions.**
  1. After the release (0.08–0.095 at 4M), Newton's per-step gain relative to the baseline's is higher than at 1M, since 4× less noise per batch.
  2. If Newton's final beats the baseline at 4M but not at 1M, the realizable gap to second order grows with batch size, as the one-step shares predict. That points to 16M.
  3. If neither, the one-step bulk advantage does not survive the dynamics at these batch sizes, and the lever for our optimizers is averaging combined with stronger input preconditioning (the α ½ at β 0.9 results), not the GN direction itself.

**Premise checks complete: the one-step gap to GN at the valley floor, all states (2026-09-27 13:31 CDT).** Cross-fitted shares of damped exact GN (GN's decrease ×1e-3 in the first column); "floor" means after a 16-step anneal:

| State · gradient | GN | Muon | PD α ¼ | PD α ½ | Two-sided | K-FAC | EKFAC | Stiff top-16 decrement |
|---|---|---|---|---|---|---|---|---|
| Muon 1M @500 · 4M | 26.4 | 0.55 | 0.65 | 0.69 | 0.65 | | | 2.11 |
| Muon 1M @500 · 16M | 31.5 | 0.49 | 0.59 | 0.64 | 0.59 | | | 2.00 |
| Muon 1M floor · 4M | 12.4 | 0.40 | 0.54 | 0.62 | 0.56 | | | 0.13 |
| Muon 1M floor · 16M | 20.5 | 0.30 | 0.42 | 0.50 | 0.44 | | | 0.12 |
| PD 1M @500 · 4M | 22.3 | 0.38 | 0.57 | 0.73 | 0.58 | | | 0.63 |
| PD 1M @500 · 16M | 27.0 | 0.34 | 0.52 | 0.68 | 0.54 | | | 0.57 |
| PD 1M floor · 4M | 9.6 | 0.27 | 0.50 | 0.73 | 0.53 | | | 0.07 |
| PD 1M floor · 16M | 14.9 | 0.22 | 0.42 | 0.64 | 0.45 | | | 0.08 |
| Muon 4M @183 · 4M | 35.3 | 0.82 | 0.88 | 0.90 | 0.87 | 0.50 | 0.58 | 7.26 |
| Muon 4M @183 · 16M | 37.6 | 0.79 | 0.86 | 0.89 | 0.85 | 0.48 | 0.56 | 7.15 |
| Muon 4M floor · 4M | 8.1 | 0.49 | 0.61 | 0.68 | 0.63 | 0.24 | 0.37 | 0.07 |
| Muon 4M floor · 16M | 14.5 | 0.34 | 0.44 | 0.51 | 0.46 | 0.15 | 0.23 | 0.08 |
| PD 4M @183 · 4M | 25.8 | 0.73 | 0.86 | 0.94 | 0.85 | 0.56 | 0.65 | 4.50 |
| PD 4M @183 · 16M | 29.4 | 0.67 | 0.79 | 0.87 | 0.78 | 0.51 | 0.59 | 4.44 |
| PD 4M floor · 4M | 5.35 | 0.40 | 0.59 | 0.75 | 0.61 | 0.31 | 0.43 | 0.04 |
| PD 4M floor · 16M | 10.5 | 0.29 | 0.44 | 0.57 | 0.45 | 0.23 | 0.31 | 0.03 |

- **At oscillating states the optimizers look close to GN, especially at 4M (0.73–0.94).** Most of the one-step decrease there is the LR-maintained excess, which every map releases.
- **At the valley floor, the stiff top-16 modes carry < 1% of GN's decrease, and the optimizers reach much less of it:** Muon 0.22–0.49, PD α ½ 0.50–0.75, K-FAC 0.15–0.31, EKFAC 0.23–0.43.
- **GN's floor decrease roughly doubles from a 4M- to a 16M-token gradient, while the optimizers' shares fall by 0.1–0.15.** The one-step gap lives in the bulk and grows as gradient noise falls, at both 1M and 4M states.
- **The ordering is the same in all 16 rows:** GN > PD α ½ > two-sided ≈ PD α ¼ > Muon > EKFAC > K-FAC > GD.
- **Open: whether this per-step advantage is realizable in training.** At 1M it is not so far. Per-batch Newton releases the excess, then gains less per step than PD or Muon, and PD's own run passes it by step 700. The 4M branches are running.

**Block-diagonal input statistic for PD's MLP-down layers: step probe only, no training test (2026-09-27 ~13:50 CDT).**
- Question (memory): keep only the four 768 × 768 diagonal blocks of the 3072-dim MLP-down input statistic, 4× less memory for that statistic.
- Tool: `track3/blockdiag_probe.py`; figures `track3/figures/blockdiag_mlp_down*.png`.
- Setup: final statistics of Track 3 run 2124499a (α ⅛ + geometry decay, 3150 steps); layers 1, 4, 7, 10; α ⅛.
- cos between the steps polar(M R) R, data-like / random M:
  - full vs block-diagonal: 0.988–0.990 / 0.989–0.991;
  - full vs Muon: 0.980–0.984 / 0.984–0.987.
- **Block-diagonal acts like a weaker PD.**
  - Its step is closest to full-R PD at exponent ≈ 0.065 (cos 0.992–0.994).
  - Its cos with Muon (0.987–0.990) about equals its cos with full PD, so it sits roughly halfway between the two.
  - The difference lives in the bulk, where the diagonal blocks have a flatter spectrum than C.
- **The mean-direction split is not the driver.** The three block-contrast directions are shrunk more than full PD shrinks them (scaled 0.35–0.79 relative to full PD). But keeping C's top 1, 4 or 16 eigen-directions exact raises the cosine by only 0.0002 (top 1) to 0.0016 (top 16).
- **Correction of an in-conversation statement.**
  - The earlier inline test used the statistics of α ¼ run 60f2d520 and normalized each block by its own mean eigenvalue. It gave 0.953 in layer 4, which was read as "as far from PD as Muon".
  - With the full C's mean eigenvalue (free from the blocks), that run gives 0.985–0.987 in every layer, against Muon's 0.971–0.974.
- **Cosine measures fidelity, not quality.** Whether block-diagonal helps is a training question. It mostly reduces to whether weaker preconditioning of MLP-down helps.
  - Locally, a globally weaker α (1/16, p = 2) was worse than α ⅛: −0.0064 vs −0.0102 against Muon.
  - A per-layer exponent has not been tested.

**Where GN's floor advantage lives, and per-batch Newton at 4M (2026-09-27 13:58 CDT).**
- **Decomposition at the valley floor** (Muon 1M annealed, 4M gradient, cross-fitted):

| Direction | Share of full GN |
|---|---|
| Muon | 0.45 |
| PD α ½ | 0.69 |
| Exact per-matrix GN (Jacobi) | 0.83 |
| One Gauss-Seidel sweep of per-matrix GN | 0.97 |
| Full GN | 1 |

  - **Within-matrix shape carries most of it.** GN's direction at Muon's per-matrix norms keeps 0.88 of GN; Muon's direction at GN's norms keeps 0.39.
  - **Which kinds** (swap test, GN's shape at PD's norm in one kind): down +27%, o +17%, up +15%, k and v +7%, q −2% over PD α ½. The reverse swap costs GN down −14%, o and v −5 to −7%.
  - PD's shortfall lies in the matrices that write into the residual stream (down, o) and in up, the same kinds as at the oscillating states. The cross-layer sweep adds the last 0.14.
- **Per-batch Newton at 4M loses too** (from the step-183 states, their own runs as baselines):
  - Muon arm: −0.021 at step 200, +0.019 at 225.
  - PD arm: −0.011 at 200, +0.026 at 225.
  - As at 1M: a short release of the excess, then slower progress than the momentum methods.
- **1M PD arm:** +0.024 at 750, +0.035 at 800, +0.039 at 825 behind PD.
- **Reading.**
  - The one-step bulk advantage of the damped GN direction is large at the floor and grows with gradient size. It does not survive as a per-batch optimizer at 1M or 4M: fresh-batch Newton loses to heavy-ball-averaged polar methods.
  - This matches the GN paper (GN ≈ SOAP up to 4M).
  - The next principled question is not "Newton instead of momentum", but how to combine a low-noise average with the within-matrix curvature structure of down, o and up, which PD misses.

**What GN's step does that PD's does not: energy in the high-variance input directions (2026-09-27 14:00 CDT; prediction before measuring).**
- **Figures:** `logs/muon_spectra/second_order_audit_20260926/figures_floor/rank_grid_*.png`. They show each direction's energy on the 12 × 12 grid of Kronecker-frame rank bins (output B eigenvectors × input C eigenvectors, log2 bins), at the Muon 1M valley-floor state with a fresh 4M gradient.
- **What the grids show:**
  - PD α ½ (and two-sided) put almost no energy (< 1e-4) into the top ~64 input-covariance directions, in every kind. Their energy sits in a stripe of low-variance input directions.
  - GN spreads its energy nearly evenly over the grid. The share in the top input bins is 0.04–0.25 by kind (up 0.23, q 0.25, k 0.17, v 0.15, o 0.12, down 0.04), against 0.00 for PD α ½.
  - Muon sits in between (0.03).
- **Hypothesis.** PD's update is polar(M R) R with R = C^-α. The trailing R, applied after polar flattening, zeroes the high-variance input directions. GN still moves them, because their large gradient offsets their large curvature.
- **Prediction.** At fixed pre-power ½, lowering the post-power (to ¼ or 0) raises PD's floor one-step decrease toward GN, more with the 16M gradient.
  - If it does not, the energy difference is not what limits PD.
  - The gap would then lie in how GN distributes energy across the grid (coupled, non-Kronecker structure), which no pre/post pair reproduces.
  - Test: `valley_rescore.py --split-maps` with pre/post ∈ {(½, 0), (½, ¼), (¼, 0), (¾, ¼), (¾, 0), (1, ¼)}, at the annealed and original Muon 1M states.

**Per-batch Newton in training: decided, runs stopped (2026-09-27 14:01 CDT).**
- Last validation points against each arm's own baseline:
  - 1M from PD's state: +0.048 at step 875 (3.9129 vs ≈ 3.865).
  - 4M from Muon's state: +0.074 at step 250 (4.2735 vs 4.1997).
  - 4M from PD's state: +0.026 at step 225.
- The baselines still hold their LR-maintained excess (≈ 0.035 at 1M, ≈ 0.08 at 4M), which their cooldowns release, so the finals can only widen the gap.
- Prediction 3 of the 13:13 entry is met: the per-step bulk advantage of the damped GN direction does not survive as a per-batch optimizer at 1M or 4M.
- Stopped to free the GPUs; logs, step records and checkpoints are kept in `newton_ref/*_v2`.

**Pre/post whitening split: prediction refuted (2026-09-27 14:14 CDT).** Muon 1M states, shares of damped GN (cross-fitted):

| State · gradient | PD ½/½ | ½/¼ | ½/0 | ¼/0 | ¾/¼ | ¾/0 | 1/¼ |
|---|---|---|---|---|---|---|---|
| @500 · 4M | **0.69** | 0.65 | 0.57 | 0.55 | 0.67 | 0.60 | 0.68 |
| @500 · 16M | **0.64** | 0.60 | 0.51 | 0.50 | 0.62 | 0.55 | 0.63 |
| floor · 4M | **0.62** | 0.56 | 0.43 | 0.41 | 0.59 | 0.49 | 0.61 |
| floor · 16M | **0.50** | 0.44 | 0.33 | 0.31 | 0.46 | 0.38 | 0.49 |

- Lowering PD's trailing whitening (the post power) lowers its one-step decrease at every state. With no trailing R it drops by up to a third. PD's pre = post = ½ is locally optimal in this family.
- So GN's energy in the high-variance input directions is not something a weaker post-whitening can copy. It comes from coupled, non-Kronecker structure. Consistent with this, the Kronecker-exact K-FAC and EKFAC are worse than PD.
- **Next, cheap:** at the valley floor, does GN applied to the momentum (the averaged input) beat PD applied to it? Per-batch Newton lost for lack of averaging, so this decides whether "curvature on the average" is worth a trainer. The anneal branches kept their momentum.

**Strength cohort complete: both sides at ½ win with a larger LR (2026-09-27 14:15 CDT).** 4M, β 0.9, seed 260925:

| Arm | LR 0.01 | LR 0.02 | Reference (same hardware) |
|---|---|---|---|
| S∘PD α ½ (L40S) | 3.8534 | **3.8260** | S∘PD α ¼ @0.01: 3.8314 |
| TS α ½ β_out ½ (Ada) | 3.8729 | **3.8356** | TS α ¼ β_out ¼ @0.01: 3.8500; PD α ½ @0.02: 3.8589 |

- Predictions 1–3 of the 10:59 entry are met at LR 0.02:
  - S∘PD α ½: −0.0054 against α ¼.
  - TS ½/½: −0.014 against TS ¼/¼, and −0.023 against PD α ½.
- Prediction 4 ("both sides at ½ over-whiten") is refuted. The LR 0.01 result (+0.023) was the wrong LR.
- **Principle refined.** Stronger whitening moves the optimal LR up (both ½-strength arms prefer 0.02 over 0.01), so a preconditioner's strength has to be compared across an LR bracket.
  - The earlier single-LR null for β_out ½ with α ¼ (+0.0015 at LR 0.01, 11:53) now carries that caveat.
  - Both the input and output sides want stronger preconditioning at 4M with fresher momentum, paired with a larger LR.
- Best 4M results so far:
  - S∘PD α ½ @0.02: 3.8260;
  - S∘PD α ¼ @0.01: 3.8314;
  - TS ½/½ @0.02: 3.8356.

  Against Muon β 0.9 (3.921) that is −0.086 to −0.095. One seed each.

**GN on the momentum: harmful at an oscillating state, best at the valley floor (2026-09-27 14:31 CDT).** Muon 1M, input M' = β M + g (the saved momentum plus a fresh gradient), cross-fitted, ×1e-3:

| State · input | GN | PD α ½ | Two-sided | PD α ¼ | EKFAC | Muon | K-FAC | GD |
|---|---|---|---|---|---|---|---|---|
| Oscillating @500 · M' (1M fresh) | **2.32** (damping 0.1) | 4.42 | 4.91 | 4.72 | 4.13 | 4.43 | 3.56 | 2.98 |
| Floor @516 · M' (1M fresh) | **8.29** | 5.71 | 5.49 | 4.95 | 4.47 | 3.32 | 2.92 | 0.75 |
| Floor @516 · M' (4M fresh) | **8.70** | 5.97 | 5.74 | 5.18 | 4.63 | 3.49 | 3.03 | 0.82 |

- "Curvature corrections help only on fresh gradients" (morning) holds only at oscillating states. There the momentum's stale stiff components (anti-aligned with the current gradient) turn a Newton-scaled step into a poor one.
- At the valley floor, GN applied to the averaged input is by far the best direction. PD α ½ gets 0.69 of it and Muon 0.40.
- The fresh part barely matters (1M vs 4M: 8.29 vs 8.70), so the momentum does the averaging the per-batch Newton lacked.

**Decision argument: a Newton step on the momentum average (2026-09-27 14:31 CDT).**
- **The two failures and the one success fit together.**
  - Per-batch Newton lost in training for lack of averaging.
  - GN on the momentum fails at oscillating states because of the stale stiff components.
  - At the floor, GN on the momentum wins.
  - Per-batch Newton v2 releases the excess within ~20 steps and then stays floor-like: its line search picks steps that do not re-excite the stiff modes.
- **Test.** `newton_train.py --momentum β`: the Lanczos right-hand side is an EMA of batch gradients (initialized from the checkpoint's momentum buffer), with the same joint held-out choice of damping and step, and AdamW on the aux parameters.
  - From PD's and Muon's 1M states at step 500, β ∈ {0.9, 0.95}, about 150 steps.
  - Baselines: the runs' own trajectories, and per-batch Newton v2 from the same states (which fell behind PD by step 700).
- **Prediction.** After the initial release (~0.03), Newton on the momentum gains at least as much per step as the baseline, so the gap stays ≥ 0.03 or grows through step 650. Per-batch Newton's gap had closed to −0.002 at step 650.
  - If the gap still closes, curvature on the average does not survive the dynamics either. The likely causes would be staleness of the average along Newton's larger flat-direction moves, or noise in the 65k-token curvature estimate.

**Diagonal R for PD: step probe only, no training test (2026-09-27 ~14:50 CDT).**
- Tool: `track3/diag_probe.py`. Same statistics as the block-diagonal probe (run 2124499a), all 12 layers, α ⅛, data-like M.
- **A diagonal R is almost Muon.** cos(full, diagonal) vs cos(full, Muon), by input:

| Input | cos(full, diagonal) | cos(full, Muon) | closest full-R exponent |
|---|---|---|---|
| q/k/v | 0.990–0.995 | 0.990–0.995 | 0–0.005 |
| attn-proj | 0.985–0.993 | 0.984–0.992 | 0–0.025 |
| MLP-fc | 0.991–0.995 | 0.992–0.995 | 0.005–0.015 |
| MLP-down | 0.976–0.987 | 0.975–0.985 | 0.015–0.03 |

  The last column is the full-R exponent the diagonal step most resembles; PD uses ⅛.
- **Why: C's diagonal is nearly flat, its eigenvalues are not.**
  - Middle 90% of channels: 0.39–2.0× the mean (R_jj 0.92–1.13).
  - Middle 90% of eigenvalues: 0.04–3.1× the mean; top eigenvalue 38–2873×.
  - PD acts on directions spread over many channels: the top eigenvector's participation ratio is 13–256 for the 768-dim inputs.
  - Exception: MLP-down layers with one massive-activation neuron, whose diagonal entry reaches 2382× the mean.
- **Tension with IsoMuon (modded-nanogpt PR #370).** Its column-only diagonal metric reportedly gives about half its gain (one-line ablation, no logs). Either their gradient-weighted statistic, roughly E[|δ|² x_j²], is much less flat than E[x_j²], or step probes miss what matters. A diagonal-PD training run would test this picture.

**Decision argument: diagonal-input PD and an output-side factor, Track 3 (2026-09-27 14:54 CDT).**
- User direction: "Let's run these two, and submit them to the cluster."
- **Why now.** IsoMuon (modded-nanogpt PR #370) is a bounded, diagonal, two-sided relative of PD. It gains −0.0042 against its same-GPU #36 baseline (n = 8), with #36's decoupled decay. Its one-line ablations say:
  - row-only or column-only gives about half the gain;
  - activation or output-gradient second moments work about as well as its noise variance;
  - a bounded full-matrix metric does no better than the diagonal.
  Our step probe (entry above) says a diagonal R is nearly Muon.
- **Arms.** Track 3 with #36's schedule compressed to 3150 steps, α ⅛ and PD-geometry decay p = 2. Everything not listed is as in `train_gpt_pd_pdwd_a0.125_s3150.py`.
  1. `track3/train_gpt_pd_pdwd_a0.125_s3150_row.py`, n = 3: full R plus IsoMuon's row factor, update S polar(S M R) R.
     - S = clamp((h/mean h)^½, ½, 2)^−½.
     - h = EMA (β 0.95) of each output row's gradient-noise variance across the step's 8 micro-batch gradients.
     - Decay unchanged (input side only).
  2. `track3/train_gpt_pd_pdwd_a0.125_s3150_diag.py`, n = 2: R = diag((C_jj/mean + 10⁻³)^−⅛), and the decay uses the diagonal R². C is still accumulated in full.
- **Comparison base.** The 6 runs of `train_gpt_pd_pdwd_a0.125_s3150.py`: mean 3.27816 at 3150, SD 0.0010.
  - SE of the difference: 0.0007 at n = 3 and 0.0008 at n = 2.
  - Hardware: gpu-partition A6000, the only free 4-GPU slots. The base includes one A6000 run (3.27886).
- **Expectations.**
  - **Diagonal:** Muon-like. Muon compressed to 3150 steps is estimated at ≈ 3.2855 (#36's 3.2797 at 3250 plus PD's compression cost of 0.0058), so ≈ +0.007 against the base. A final within ~0.002 of the base would mean the step-probe picture misses what matters.
  - **Row:** no firm prediction. IsoMuon's row factor alone was worth ≈ −0.002 on top of Muon. On top of PD it may add up to that, or nothing if it overlaps with PD's input side.
- **Smoke tests** (60 steps, `track3/smoke/smoke_s3150_{base,row,diag}.py`) run before the real runs.

**Newton on the momentum average: prediction failed; averaging alone does not rescue it (2026-09-27 14:58 CDT).** `newton_train.py --momentum β` (EMA right-hand side, initialized from the checkpoint's momentum), 1M, 2 GPUs per arm, from step 500. Validation and gap to the arm's own baseline run:

| Step | PD, β 0.9 | PD, β 0.95 | Per-batch Newton (PD state) | PD baseline | Muon state, β 0.9 | Muon state, β 0.95 | Per-batch Newton (Muon state) | Muon baseline |
|---|---|---|---|---|---|---|---|---|
| 525 | 3.9781 | 3.9824 | 3.9783 | ≈ 4.012 | 4.0093 | 4.0131 | 4.0035 | ≈ 4.047 |
| 550 | 3.9615 (−0.034) | 3.9636 | 3.9669 | 3.9952 | 3.9902 (−0.039) | 3.9934 | 3.9912 | 4.0292 |
| 600 | 3.9474 (−0.024) | 3.9480 | 3.9542 | 3.9716 | 3.9721 (−0.027) | 3.9724 | 3.9761 | 3.9988 |
| 625 / 650 | 3.9419 at 625 (−0.017) | 3.9423 | 3.9487 | 3.9469 at 650 | 3.9611 at 650 (−0.012) | 3.9609 | | 3.9731 |

- The average helps a little (0.004–0.007 below per-batch Newton at equal steps). But after the release both Newton variants gain ~0.22e-3 per step, against ~0.5e-3 for PD and Muon, and the gap keeps closing. The prediction (gap ≥ 0.03 through step 650) fails.
- **Candidate reason: short-horizon bias.** Every successful optimizer steps ~2× past the line-search optimum along its own direction (midpoint argmin 0.44–0.58), while the Newton trainer's held-out line search picks the step that minimizes the next loss.
  - Greedy steps release the valley-wall excess and then move slowly along the floor. Overshooting steps keep a wall oscillation but move faster along the floor.
  - This is the review's "GN at 1.5× the line-search optimum" control.
- **Test.** The same trainer with the chosen step multiplied by κ ∈ {1.5, 2}, on the averaged input (β 0.9), from both states.
- **Prediction.** κ = 2 closes most of the per-step gap to the baselines (≥ 0.35e-3 per step after the release). If it does not, the Newton direction itself, not its step size, is what fails in training at 1M.

**Diagonal / output-side runs: smoke tests passed and runs submitted (2026-09-27 15:02 CDT).**
- **Smoke tests** (60 steps, 4× A6000; all completed). Val loss at step 60:
  - reference script 5.2375;
  - row 5.1901;
  - diagonal 5.3306.
- **Row heats** are finite. After 60 steps, the row factor S spans mostly 0.75–1.41; 19–28% of MLP-fc rows (the 3072 hidden neurons) sit at the clip.
  - The output-side statistic is much less flat than C's diagonal: MLP-fc row heat/mean has 5–95% range 0.07–3.2, against the input diagonal's 0.6–1.9.
- **Submitted** to the gpu partition, 4 GPUs on 48 GB Ada or Ampere nodes, 4 CPUs, 3 h limit.
  - row: 2625179 (g14), 2625180 (g15), 2625181 (pending, nice 100 so the diagonal runs go first);
  - diagonal: 2625182, 2625183 (pending).
- Script hashes (sha256 prefix): row aff966fc452fd72e, diagonal 6536592f8a73d8e9.

**Overshooting Newton steps: a small part of the deficit (2026-09-27 15:21 CDT).** Newton on the momentum (β 0.9), with the line-search step multiplied by κ:

| Arm | 550 | 600 | 625 | 650 | Gain per step after the release |
|---|---|---|---|---|---|
| Muon state, κ 1 | 3.9902 | 3.9721 | 3.9665 | 3.9611 | 0.22e-3 |
| Muon state, κ 1.5 | 3.9929 | 3.9705 | 3.9650 | 3.9582 | 0.25e-3 |
| Muon state, κ 2 | 4.0217 | 3.9901 | 3.9814 | 3.9707 | 0.39e-3 (while recovering from a bad start) |
| Muon baseline | 4.0292 | 3.9988 | | 3.9731 | 0.51e-3 |
| PD state, κ 1 | 3.9615 | 3.9474 | 3.9419 | 3.9381 | 0.22e-3 |
| PD state, κ 1.5 | 3.9606 | 3.9433 | 3.9374 | | 0.24e-3 |
| PD state, κ 2 | 3.9615 | 3.9420 | 3.9341 | | 0.30–0.32e-3 |
| PD baseline | 3.9952 | 3.9716 | ≈ 3.9593 | 3.9469 | 0.47–0.49e-3 |

- Overshooting helps a little but leaves Newton 35–55% slower per step than the momentum-polar baselines. The prediction mostly fails: the greedy step rule is a small part of the deficit.
- **A mechanism consistent with every run.**
  - In flat, persistent directions, a damped Newton step scales with gradient/(curvature + damping), so its steps there stay small and noisy.
  - The polar maps move those directions by ~LR per step regardless of gradient size, and momentum accumulates them.
  - So the normalized-update regime, not local second-order optimality, is what makes progress along the floor at 1M.

**Decision argument: GN's shape inside the normalized regime (2026-09-27 15:21 CDT).**
- At the floor, GN's direction rescaled to Muon's per-matrix norms keeps 0.88 of GN's one-step decrease, against 0.62 for PD α ½ (`gn_floor`). Muon's direction at GN's norms keeps 0.39. The shape is the valuable part.
- **Test** (`newton_train.py --normalize`). The GN direction is computed on the momentum average as before. Each body matrix's step is then normalized to Muon's Frobenius norm (√min(m, n) × the shape factor) times the run's own LR schedule, with no line search and matched weight decay. So only the direction differs from the baseline, not the magnitudes.
  - From Muon's and PD's 1M states at step 500, damping ∈ {1e-2, 1e-3} × ρ, 150 steps.
- **Prediction.** If GN's shape is what the polar maps miss, the normalized-GN arms gain at least as much per step as their baselines, and they keep the ~0.03 head start (the release) or widen it through step 650.
  - If they lose, the shape advantage measured in one step does not transfer to the normalized dynamics either. The gap to GN would then not be a map the normalized regime can take.

**Overshoot arms complete (2026-09-27 15:28 CDT).** The PD-state κ arms reached step 650: κ 1.5 **3.9333** (−0.0136 against the PD baseline's 3.9469), κ 2 **3.9298** (−0.0171); κ 1 was 3.9381 (−0.0088). Gain per step over 625 → 650: κ 1.5 0.16e-3, κ 2 0.17e-3, baseline ≈ 0.50e-3. The overshooting arms' rate falls over time (κ 2: 0.32e-3 over 600 → 625), so the gap keeps closing at every κ. This confirms the 15:21 reading.

**Normalized GN, smoke test: at the baseline's step norms, GN's direction raises the loss (2026-09-27 15:41 CDT).**
- **Setup.** `newton_train.py --normalize d` runs from the 1M step-500 states with momentum 0.95, the runs' own β. The right-hand side is then proportional to the baseline's momentum. Per-matrix step norms:
  - Muon state: Muon's own Newton-Schulz output norm × shape factor.
  - PD state: the nominal √min(m, n) × shape factor, as PD rescales.
  - Both use the run's LR, with no line search.
- **Interrupted.** The smoke outputs went to the session scratchpad on the 20 GB home quota. Their checkpoint writes filled it and stopped both runs after two steps. All outputs now go under `/share/data`.
- **Results, two steps each:**
  - **Muon state, d 1e-2 ρ.** Held-out change +0.0044, then +0.0092 per step, although the first-order term is −0.014. Validation 4.0650 → 4.0671 after one step. cos with Muon's step 0.65.
  - **PD state, d 1e-3 ρ.** +0.155 and +0.188 per step. Validation 4.0284 → 4.1851.
- **Reading.** At the baseline's LR, the curvature along GN's normalized step far exceeds the first-order gain, even though at the floor this shape kept 0.88 of GN's one-step decrease at a *fitted* scale.
- **Two explanations, separable in one step** (`normgn_probe.py`: the run's next momentum; curvature from the trainer's 64 in-batch sequences or from 64/256/1024 fresh ones; Lanczos 16 or 64; the held-out loss at 0.25–2 × the trainer's step):
  - **(a) Estimation noise.** A 64-sequence GN underestimates the curvature along its own damped-inverse direction, so the inverse amplifies noise.
    - Prediction: held-out q / the set's own q ≫ 1 for small sets and small damping, falling toward 1 with 1024 fresh sequences.
    - With 1024 sequences, the normalized step at the trainer's scale lowers the loss and beats Muon's step.
  - **(b) Step size.** The estimate is adequate, but GN's shape at Muon's per-matrix allocation needs a much smaller LR than the polar map. This would be the strength–LR principle again.
    - Prediction: ratios near 1 for every set, and the loss minimum of the normalized step at c ≤ ¼ of the trainer's step. Muon's own minimum should sit near ½, as in the midpoint measurement.

**Normalized-GN probe: not estimation noise; at an oscillating state GN on the momentum is a worse direction at every scale (2026-09-27 16:22 CDT).**
- **Setup.** `normgn_probe.py`, 1M states at step 500, input M' = 0.95 M + g501 (the run's own next momentum). The probe's rebuilt Muon step matches the run's actual W501 − W500 to 1e-5.
- **Held-out loss change at the trainer's step (c = 1), GN's direction rescaled per matrix to the trainer's norms:**

| GN direction | Muon state | PD α ¼ state |
|---|---|---|
| Lanczos 16, any set, any damping | +0.002 to +0.003 | +0.13 to +0.15 |
| Lanczos 64–256, d ≤ 1e-3 ρ, 64 in-batch sequences | −0.0010 to −0.0012 | +0.019 to +0.022 |
| the same, 1024 fresh sequences | −0.0005 | +0.015 |
| trust-region GN at the trainer's radius (per-matrix rescaled / GN's own allocation) | +0.007 / +0.025 | +0.06 / +0.09 |
| **the run's own step** | **−0.0016** | **−0.0015** |

- **Prediction (a), estimation noise: refuted.**
  - Held-out curvature over the set's own falls from 1.5–3 (64 sequences) to 1.03–1.25 (1024) at the Muon state, and to 1.02–1.08 at the PD state.
  - The step does not improve: the most accurate curvature gives the weakest GN step.
  - Lanczos length matters up to 64–128 steps. Damping below 1e-3 ρ changes nothing.
- **Prediction (b), step size: only half right.**
  - At the PD state, GN's best scale is c* = 0.16–0.18 of the trainer's step, against 0.57 for PD's own step.
  - At the Muon state, c* = 0.67–0.73, close to Muon's own 0.55.
  - At both states, the one-step decrease at each direction's own best scale is 3–9× smaller for GN than for the run's own step:
    - Muon state: 0.5–1.3e-3 against 4.9e-3.
    - PD state: 0.6–1.1e-3 against 3.7e-3.
  - This agrees with the cross-fitted table of 14:31 (GN on M' 2.3 vs Muon 4.4 ×1e-3 at step 500).
- **Reading.**
  - At an oscillating state, the run's own direction is co-adapted to that state. It is the best one-step direction and neutral at its own step, with c* ≈ ½ (see OBSERVATIONS).
  - GN on the same averaged input is worse at every scale, because the momentum's stale stiff components mislead a curvature-weighted map. A better curvature estimate does not help.
  - The floor result (GN's shape keeps 0.88 at a fitted scale) does not carry over to the states training actually visits.
  - The 15:21 training test is dropped: its per-step prediction already fails at step 501, whatever the state or estimate.
- **Harness control** (`newton_train.py --normalize 0 --polar-control`, the Muon state, 2× L40S). Muon's own direction inside the Newton trainer reproduces the baseline's per-step training loss to 1e-5 through step 509, and to 1.6e-4 at step 510. Validation at step 550 is 4.02972, against the baseline's 4.02923 (+0.0005 after 50 steps). The baseline does not clip here. All Newton-trainer comparisons therefore used a faithful harness.
- **Figure:** `logs/muon_spectra/second_order_audit_20260926/figures_normgn/normgn_probe.png`.

**Where in training the preconditioners gain: step-equivalent speedup, not loss gaps (2026-09-27 16:25 CDT).**
- **Question.** Is curvature a floor phenomenon (it pays in the cooldown), as the one-step maps on the momentum suggest?
  - On M', at the oscillating 1M state @500, all maps score about equally: Muon 4.4, PD ¼ 4.7, PD ½ 4.4, TS 4.9, GN 2.3.
  - At the floor, they order by curvature strength: GN 8.3, PD ½ 5.7, TS 5.5, PD ¼ 5.0, Muon 3.3 (×1e-3).
- **Answer, from existing curves: no.**
  - 4M, β 0.9, against Muon @0.014: every preconditioner's loss gap is largest early and shrinks steadily (S∘PD α ½: −0.27 at 100, −0.10 at 350, −0.096 final). The cooldown (steps 331–368) adds nothing.
  - 1M: PD − Muon is −0.014 at 1300 and −0.018 final, so the cooldown adds about a quarter.
- **Loss gaps are the wrong lens here.** A flattening loss curve turns a constant speedup into a shrinking gap. Steps each arm needs to reach Muon's validation at Muon's step t (4M):

| Arm | t = 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|
| PD α ¼ @0.02 | 1.15× | 1.16× | 1.18× | 1.19× | 1.18× |
| PD α ½ @0.02 | 1.14× | 1.17× | 1.20× | 1.24× | 1.23× |
| TS ½/½ @0.02 | 1.15× | 1.18× | 1.22× | 1.28× | 1.29× |
| S∘PD α ½ @0.02 | 1.22× | 1.25× | 1.32× | 1.33× | 1.34× |

  - At 1M, PD α ¼ is 1.12× early and 1.08× late.
- **Reading.**
  - At 4M, the curvature-aware maps' speedup grows through the high-LR phase, most for the strongest settings. At 1M it is roughly flat.
  - Their gain is neither a head start nor a cooldown effect. It is a per-step rate advantage in the regime the one-step momentum scores call a tie at step 500 (1M).
  - The one-step score on the momentum at a single state therefore misses what the maps gain over many steps. Candidates: the dynamics (less oscillation or stored excess, and faster floor progress), a state difference (each run's state is co-adapted to its own map), or the batch size (the 4M maps do better).
  - From now on, training comparisons report step-equivalent speedups next to loss gaps.

**Independent review of the GN-in-training line (2026-09-27 16:38 CDT; read-only reviewer, brief in the session scratchpad). Accepted, with corrections to the entries above.**
- **Corrections.**
  1. **Table row.** "Lanczos 16, any set, any damping" (16:22 entry) holds for damping ≤ 1e-2 ρ at the Muon state (+0.002 to +0.003) and 1e-3 ρ at the PD state (+0.13 to +0.15). At 0.1 ρ it is +0.007 to +0.010 (Muon) and +1.1 to +1.2 (PD); at 1e-2 ρ at the PD state, +0.17 to +0.20.
  2. **Trust region: invalid.** The trust-region solve used M' in the sum convention (M ← βM + g, up to 20× the gradient scale) as its linear term. μ/ρ = 0.076 is therefore not the trust region at the trainer's radius; the input should be (1 − β) M'.
  3. **The Muon-state failure is signal, not curvature.** The converged normalized GN step meets *less* held-out curvature than Muon's step (q 0.002–0.005 vs 0.032), and its best scale is larger (≈ 0.7 vs 0.55). What it lacks is first-order gain along the held-out gradient: 5–11× less. "Unregulated curvature" fits only the PD state, where a per-matrix Frobenius norm is not a step size in PD's geometry.
- **The key alternative: the Newton trainers measured their step rule.** Checked from the step logs:
  - The joint line search picks damping and step on the same 64 held-out sequences every step.
  - Muon's own body step, inside the same harness (the polar control), raises that set's loss on 82% of steps (mean +0.0023), yet the run tracks the baseline to 0.0005. A greedy one-step criterion would throttle Muon too.
  - Median Newton step norms over steps 550–650 were 0.005–0.16 (IQR down to 0.001), against 1.33 for Muon's step and 1.92 for PD's; 3–7% of body steps were skipped.
  - The reviewer estimates that AdamW on the embeddings, head and gains supplied a third to ~60% of the late held-out gain.
  - The overshoot arms multiplied an already collapsed step, so they never tested large steps.
  - So the 12:37–15:21 conclusion that "GN-type steps lose per step" may be about greedy step selection, not GN's direction.
- **Decisive control, running** (`newton_train.py --polar-direction`, "greedy Muon"): the same harness and line search, with Muon's own step (peak LR × LINE × schedule) as the only candidate, from both 1M step-500 states, momentum 0.95, 150 steps.
  - If its rate after the release falls to ~0.2–0.3e-3 per step like Newton's, the earlier runs measured the step rule.
  - If it holds ~0.5e-3 per step, GN's direction is at fault and the earlier reading stands.
- **Next, from the review.**
  - **Normalized GN in training, properly:** Lanczos ≥ 64, damping ≤ 1e-3 ρ, 150 steps from Muon's step-500 state, against the polar control. Judged by the rate over 550–650, not by per-step held-out changes. A deficit persisting past 50 steps refutes "switching costs only a transient".
  - **Probe a Newton trainer's own final state:** if GN still offers several e-3 there, the trainer is leaving it unused.
  - **Cooldown test, if run:** a fixed step rule (a fresh held-out slice each step, ≥ 256 sequences, chosen on one half and confirmed on the other); the converged GN direction; a "cooldown then ~20 GN steps" arm to separate a final relaxation from a faster path; a PD pair and 4M @330. Falsifier: with the control within 0.001 of the baseline, B − A > −0.003 falsifies "GN pays at the floor" at 1M.
  - **If the line stops,** curvature goes into the geometry of the normalized map. Targets:
    - within-matrix, non-Kronecker structure in down, o and up (per-matrix GN 0.83 vs PD α ½ 0.69);
    - orthogonalized per-matrix GN on the momentum;
    - a token-weighted input statistic E[w x xᵀ], with w each token's output curvature;
    - a split momentum, with a shorter β on the top input-covariance directions.
    Only candidates that gain at both oscillating and floor states go to 4M training.
- **Calibration caveat, accepted.** The floor share ranks PD α ½ above α ¼ at every state, but training at β 0.95 reverses that at both batch sizes. One-step shares can filter candidates, not judge them.

**Greedy Muon: the Newton-trainer results measured the step rule, not GN (16:47 CDT, 2026-09-27).** `newton_train.py --polar-direction --momentum 0.95`: the same harness and joint held-out line search, with Muon's own step as the only candidate direction. From the 1M step-500 states, 150 steps (2× L40S for the Muon state, 2× Ada for the PD state):

| State · arm | 525 | 550 | 600 | 650 | Rate 550–650 | Median body step norm |
|---|---|---|---|---|---|---|
| Muon · **greedy Muon** | 4.0094 | 3.9906 | 3.9725 | **3.9613** | 0.29e-3 | 0.041 |
| Muon · greedy Newton on momentum, β 0.95 | 4.0131 | 3.9934 | 3.9724 | 3.9609 | 0.33e-3 | 0.029 |
| Muon · greedy Newton on momentum, β 0.9 | 4.0093 | 3.9902 | 3.9721 | 3.9611 | 0.29e-3 | 0.042 |
| Muon · baseline | | 4.0292 | 3.9988 | 3.9731 | 0.56e-3 | 1.33 |
| PD · **greedy Muon** | 3.9801 | 3.9635 | 3.9500 | **3.9402** | 0.23e-3 | ≈ 0 (mostly skipped) |
| PD · greedy Newton on momentum, β 0.95 | 3.9824 | 3.9636 | 3.9480 | 3.9383 | 0.25e-3 | 0.005 |
| PD · greedy Newton on momentum, β 0.9 | 3.9781 | 3.9615 | 3.9474 | 3.9381 | 0.23e-3 | 0.005 |
| PD · baseline | | 3.9952 | 3.9716 | 3.9469 | 0.48e-3 | 1.92 |

- **Muon's direction under the greedy rule reproduces greedy Newton** to ≤ 0.002 at every checkpoint, rate included. The review's alternative is confirmed.
- **What the 12:37–15:21 entries attributed to GN belongs to the step rule.** Those claims were: "Newton-type steps lose per step at 1M/4M", "averaging does not rescue it", and "overshooting is a small part of the deficit".
  - The greedy one-step criterion on a fixed held-out set collapses any direction's step to 1/32–1/4 of the LR, or skips it.
  - Progress then comes from the initial release and from AdamW on the embeddings, head and gains. From the PD state, the body was mostly frozen after the release.
- **Consequences.**
  - GN's direction has not been tested in training with a workable step rule; normalized GN (the LR schedule and Muon's per-matrix norms, Lanczos 64, d 1e-3 ρ) is the first such test and is running.
  - The one-step picture (GN's large decrease at the floor; GN on the momentum weaker at oscillating states) stands.
  - The 12:33 finding that the GN paper's inner-loop recipe gains mostly through more sequential steps used a held-out line search too, with the same kind of rule on both arms. It is not affected as a comparison, but its absolute rates share the throttling.
  - Held-out one-step changes, even on 512 sequences, are not a progress measure for a normalized optimizer at its edge: its own step raises them 82% of the time.

**Decision argument: does the realized speedup keep growing with batch size? A 16M-token cohort (2026-09-27 16:50 CDT).**
- **Why.**
  - The curvature-aware normalized maps' step-equivalent speedup over Muon is ~1.1× at 1M and 1.2–1.34× at 4M, and at 4M it grows through training.
  - The one-step gap to GN also grows with the gradient batch: Muon's share of GN at the floor is 0.27–0.49 with 4M gradients and 0.22–0.34 with 16M.
  - The GN paper's claim is that the distance to a true second-order optimizer is largest at large batch.
  - Whether the maps we have keep tracking that growth in training is the most direct measurement of "the gap at larger batch sizes". The independent review (16:38) suggested it.
- **Design.** `soaudit_batch16m_20260927`: 16,777,216 tokens per step, the same 1.54e9-token budget (92 steps), seed 260925, momentum 0.9, all else as the 4M β 0.9 arms, except:
  - warmup 3 steps (tokens-constant, as 50 → 13);
  - PD/TS statistics refresh every 2 steps (~34M tokens, between the 1M and 4M values; the covariance EMA is already per forward pass);
  - validation every 8 steps;
  - kept checkpoints at 9, 46 and 83 for floor anneals.
- **Arms**, LRs √-scaled from the 4M optima and bracketed:
  - Muon @0.02, 0.028, 0.04;
  - S∘PD α ½ @0.028, 0.04;
  - TS α ½ β_out ½ @0.028, 0.04;
  - PD α ½ @0.028, 0.04.
  - Order: g20 runs Muon @0.028, S∘PD @0.04, TS @0.04, Muon @0.04, S∘PD @0.028. priv-g14 (after the normalized-GN run) runs Muon @0.02, TS @0.028, PD @0.04, PD @0.028.
  - ~50 min per arm on 4 GPUs.
- **Predictions.**
  1. At the best LR each, S∘PD α ½'s step-equivalent speedup over the best Muon at 16M exceeds its 4M value at the same fraction of training (1.32–1.34× over 55–80%).
  2. The final gap S∘PD − Muon is larger in magnitude than at 4M (−0.096).
  3. The best LRs are within one bracket step of the √ scaling (Muon 0.028, S∘PD 0.04).
  - If the speedup plateaus or falls at 16M, the Kronecker-geometry maps stop tracking the growing one-step gap. What large batches need would then lie beyond input/output whitening: the within-matrix non-Kronecker structure and the cross-matrix coupling that the one-step maps locate.
- **Not in this cohort:** β, SOAP's β₂ (kept at the 4M per-step values) and aux LR (0.002 as at 1M and 4M). All arms share them, so the optimizer comparison stays like for like.

**Diagonal-input PD in Track 3: complete; the full matrix carries most of PD's gain (2026-09-27 17:12 CDT).**
- `train_gpt_pd_pdwd_a0.125_s3150_diag.py`, n = 2: finals 3.28294 (L40S, 2625183) and 3.28217 (A6000, 2625182); mean 3.28256.
- Against full PD's 6 runs (3.27816, SD 0.0010): +0.0044, about 5 SE (SE 0.0008).
  - The two runs agree closely all run (3.44736 vs 3.44735 at step 2000).
  - The gap shrinks during the cooldown: +0.020 at step 500, +0.014 at 2000, +0.006 at 3000, +0.0044 at the end.
- **Against Muon:** no Muon run exists on this compressed schedule. #36 shifted by PD's compression effect gives ≈ 3.2855–3.2866, so the diagonal keeps roughly 40% of PD's gain.
  - That is more than the step probe's "≈ Muon".
  - A candidate for the retained part is the few coordinate-aligned outlier inputs (MLP-down massive activations), which the diagonal captures in both the update and the geometry decay. Untested.
- **Reading.** Most of PD's gain needs cross-channel structure. A diagonal R is not a free replacement for the full statistic, so PD's memory cost stays unless a structured approximation (block-diagonal, low-rank plus diagonal) is tested in training.
- **Output-side arm so far:** 2 of 3 runs done (A6000): 3.27681 and 3.27746, mean 3.27713, −0.0010 against full PD (about 1.2 SE). The third run (RTX 6000 Ada, 2625181) is running.

**Floor vs stored excess along training (2026-09-27 17:13 CDT).** 16-step LR anneals (`anneal_branch.py`, the run's own optimizer) at every kept step. Held-out EVAL loss before (raw) and after (floor):

| Batch · step | Muon raw / floor (excess) | PD raw / floor (excess) | Floor gap PD − Muon |
|---|---|---|---|
| 1M · 100 | 5.4348 / 5.2831 (0.152) | 5.2930 / 5.1316 (0.161) | −0.152 |
| 1M · 200 | 4.6016 / 4.5056 (0.096) | 4.4685 / 4.3787 (0.090) | −0.127 |
| 1M · 500 | 3.9670 / 3.9291 (0.038) | 3.9358 / 3.9010 (0.035) | −0.028 |
| 1M · 900 | 3.7738 / 3.7412 (0.033) | 3.7574 / 3.7261 (0.031) | −0.015 |
| 1M · 1300 | 3.6911 / 3.6652 (0.026) | 3.6770 / 3.6505 (0.027) | −0.015 |
| 4M β 0.9 · 37 | 6.1058 / 5.8621 (0.244) | 6.0558 / 5.7684 (0.287) | −0.094 |
| 4M β 0.9 · 183 | 4.2843 / 4.1779 (0.106) | 4.1355 / 4.0508 (0.085) | −0.127 |
| 4M β 0.9 · 330 | 3.9206 / 3.8548 (0.066) | 3.8360 / 3.7890 (0.047) | −0.066 |

(1M: Muon @0.007, PD α ¼ @0.01, β 0.95. 4M: Muon @0.014, PD α ½ @0.02, β 0.9.)
- **1M: PD's gain is all floor.** Both runs store the same excess at every step, and it decays 6× at constant LR (0.15–0.16 → 0.026). The floor lead is built by step ~200 and shrinks to a constant ~0.015 by step 900. From 900 to 1300 both floors fall by the same 0.076.
- **4M: PD also stores less excess once past the early phase.** It stores more at step 37 (0.287 vs 0.244), then less at 183 and 330 (by 0.02). Its floor lead is −0.127 at 183 and −0.066 at 330. The final validation gap, −0.063, matches the step-330 floor gap: the cooldown realizes the floor.
- **Reading.**
  - At 1M the preconditioner's advantage is a floor lead built in the first ~20% of training and then held.
  - At 4M it has two parts: a floor lead, and a smaller stored oscillation later in training. Together they give the growing step-equivalent speedup.
  - Whatever makes PD's steps store less loss in the oscillation at 4M, but not at 1M, is a batch-size-dependent effect worth measuring: the sharpness each optimizer settles at, and how much of its step lies in the stiff directions.
- Figure: `logs/muon_spectra/second_order_audit_20260926/figures_floor/floor_progress.png`; the atlas view "Step rule & floors".

**Normalized GN in training: fails, sharpening the network (17:56 CDT, 2026-09-27).** `newton_train.py --normalize 0.001 --cg 64 --momentum 0.95` from Muon's 1M step-500 state, 2× L40S. GN's direction on the momentum average; each matrix at Muon's own step norm (|NS(M')| × shape factor) and the run's LR; no line search.

| Step | 525 | 550 | 575 | 600 | 625 | 650 | Rate 550–650 |
|---|---|---|---|---|---|---|---|
| Normalized GN | 4.0911 | 4.1040 | 4.1072 | 4.0910 | 4.0845 | **4.0760** | 0.28e-3 |
| Muon baseline | (4.047) | 4.0292 | | 3.9988 | | **3.9731** | 0.56e-3 |
| Muon in the same harness | 4.0443 | 4.0297 | | | | | |

- **Sharpening throughout.** Median top GN Ritz value per 25-step window: 22 (501–525), 60, 64, 83, 86, 101 (626–650). The curvature along the momentum, ρ, rises from 2.6 to ~20. The cosine with Muon's step falls from 0.33 to ~0.07.
- **Both of the review's criteria fail.** The rate after the transient is half Muon's, and the deficit (+0.103 at 650) grows rather than closing. The 15:21 prediction is refuted: GN's shape does not transfer to the normalized regime at the trainer's step norms.
- **Reading.**
  - A normalized first-order optimizer at its edge regulates sharpness through the energy its step puts into the stiff directions (the EoS oscillation).
  - GN's damped inverse removes that energy, so the network sharpens without check. The fixed-norm step then keeps meeting the new curvature, which also pulls ρ and the damping upward.
  - Any second-order step in this regime needs a sharpness-control component, and a step size tied to the curvature rather than a fixed norm. Candidates: a fixed predicted decrease (a KL or G-norm trust region), or the incumbent map's stiff part plus curvature geometry for the bulk. The last is what PD, TS and S∘PD already do.
- **Where the GN-in-training line stands.**
  - Greedy steps are throttled by their rule, whatever the direction.
  - GN's direction at normalized steps sharpens the network and loses.
  - The one-step advantage at the floor is real, but no step rule tested so far realizes it in training.
  - The line pauses here. The 16M cohort and curvature-as-geometry candidates come next.

**Decision argument: two-sided diagonal PD, Track 3 (2026-09-27 17:58 CDT).**
- User direction: "can we run Diagonal Pd with input and output please?"
- **Arm.** `track3/train_gpt_pd_pdwd_a0.125_s3150_diagrow.py` (sha256 prefix 64555361878e963d), n = 2:
  - the diagonal-input arm's R = diag((C_jj/mean + 10⁻³)^−⅛), with its diagonal geometry decay;
  - plus the output-side arm's IsoMuon row factor S, as S polar(S M R) R.
  - No new code path: the output-side script with the diagonal-input pd_root. Both paths were smoke-tested and ran full Track 3 runs, so no separate smoke test.
- **Jobs.** 2625310 (g14) and 2625311 (g15), A6000, started 17:58 CDT.
- **What it isolates.** Diagonal-only against diagonal + output side measures the output factor on a weak input side, as in IsoMuon, where row-only gave about half its gain. It is also the cheap, all-diagonal form of PD.
- **Expectation.** Between diagonal-only (+0.0044 against full PD) and full PD:
  - about +0.0034 if the output factor adds the same −0.001 it adds on top of full PD;
  - nearer +0.002 if it does more when the input side is weak.

**Prediction before measuring: PD with the GN-consistent input statistic (2026-09-27 17:58 CDT).**
- **The candidate.** The exact per-matrix GN is E_t[(d_t d_tᵀ) ⊗ (x_t x_tᵀ)]. Tracing out its output side gives the input factor
  C_w = E_t[|d_t|² x_t x_tᵀ] / E_t[|d_t|²],
  where d_t is the sampled-label backprop error. So C_w is PD's C with each token weighted by its output curvature: the first non-Kronecker correction, still used through the polar map, which keeps the stiff-direction regulation.
  - A contrast arm uses true-label errors (gradient-weighted, as IsoMuon's statistic reportedly is).
- **Probe.** `weighted_input_probe.py` scores PD with C vs C_w vs true-label C_w, at α ¼, ½, ¾. States: Muon's 1M oscillating state @500 and its floor @516. Inputs: g4M, g16M and M'. Scoring is cross-fitted on the valley_rescore sets, so shares of damped GN come from the existing files. Jobs 2625312/2625313 on the gpu partition (A6000).
- **Prediction.**
  1. At the floor, PD-C_w at its best α exceeds PD-C at its best α by ≥ 0.05 of GN's decrease on g4M and g16M. The per-matrix GN reaches 0.83 against PD α ½'s 0.69, and a token weighting is the cheapest part of that gap.
  2. At the oscillating state on M', it is not worse than PD-C by more than 0.1e-3.
  3. The true-label weighting does worse than the sampled-label one.
  - If 1 and 2 hold, a 4M training test at β 0.9 with an LR bracket follows. If C_w ≈ C (median cosine of the normalized matrices > 0.98), token weighting carries no geometry and the within-matrix gap is elsewhere.

**Token-weighted input statistic: no new geometry where the gap is (18:04 CDT).** `weighted_input_probe.py`, Muon's 1M states. Cross-fitted one-step decrease (×1e-3) on g4M:

| State | PD α ½ (C) | PD α ½ (C_w, sampled labels) | PD α ½ (true-label weights) |
|---|---|---|---|
| @500, oscillating | 18.10 | 18.15 | 18.16 |
| @516, floor | 7.615 | 7.663 | 7.651 |

- At the floor, C_w adds +0.004 of GN's decrease (12.38). Prediction 1 (≥ 0.05) fails.
- On the momentum input M': floor 5.705 → 5.813 (+0.013 of GN's 8.29); oscillating state 4.414 → 4.382. Prediction 2 holds, but the effect is small at both states. The true-label weighting is no different (5.822 / 4.372), so prediction 3 fails.
- The median cosine of normalized C and C_w is 0.986 overall, and 0.98–0.997 in down, o and up, the matrices that carry PD's shortfall. It is lower only in v (0.86) and k (0.96).
- The fallback clause applies: PD's within-matrix gap is not token-weighted input curvature. What remains is GN's token-by-token coupling of output and input directions, which no per-side statistic holds.

**Prediction before measuring: PD in the exact GN geometry (2026-09-27 18:04 CDT).**
- **The construction** (`gnpd_probe.py`): GN-PD(p) takes w = (G + μ)^-p b, then P = polar(w) per matrix, then D = −(G + μ)^-p P, rescaled to Muon's norms.
  - This is PD α = p with the exact GN of all hidden matrices replacing I ⊗ C. Matrix functions come from Lanczos (64 steps).
  - The polar step keeps unit singular values in the whitened coordinates, so the step keeps energy in stiff directions (scaled by λ^-p), unlike the damped Newton step (λ^-1).
  - Jobs 2625316/2625317. Muon's 1M states @500 and @516; inputs M' and g4M; damping cross-fitted over {1e-2, 1e-3, 1e-4} ρ.
- **Prediction.**
  1. At the floor on g4M, the better GN-PD reaches ≥ 0.8 of GN's decrease (PD α ½: 0.62). On M' it beats PD α ½ (5.7e-3).
  2. At the oscillating state on M', it is within 0.3e-3 of PD α ½ (4.42) or better, because the polar step keeps the stiff directions' energy.
  - If both hold, a training test follows (two Krylov runs per step, the trainer's LR and norms): does it regulate sharpness like PD, or sharpen like normalized GN?
  - If 1 fails, the exact curvature's geometry is not what PD's shortfall lacks at a fixed shape-normalization, and the gap is in the step magnitudes the polar map discards.

**16M cohort, interim: the loss gap is large, the speedup is not (2026-09-27 18:27 CDT).**
- **Finals so far** (92 steps, 1.54e9 tokens): Muon @0.028 4.9635, Muon @0.02 4.9104, S∘PD α ½ @0.04 **4.5846** (−0.326 vs the best Muon so far).
  - Muon's best is the bracket's bottom, so Muon @0.014 was added (priv-g14, after the first queue).
- **The loss gap misleads here.** `speedup_by_batch.py` computes step-equivalent speedups from each step's training loss on unseen data (16.8M tokens per step), smoothed over 3% of the run, against the best-final Muon at the same batch and β.

| Muon's smoothed loss | 5.4 | 5.2 | 5.0 | 4.6 | 4.2 | 4.0 |
|---|---|---|---|---|---|---|
| 1M, β 0.95: S∘PD α ¼ @0.01 | – | 1.15 | 1.17 | 1.17 | 1.21 | 1.17 |
| 4M, β 0.95: S∘PD α ¼ @0.01 | 1.13 | 1.12 | 1.14 | 1.20 | 1.30 | 1.32 |
| 4M, β 0.9: S∘PD α ½ @0.02 | 1.18 | 1.21 | 1.26 | 1.27 | 1.36 | 1.37 |
| 16M, β 0.9: S∘PD α ½ @0.04 | 1.26 | 1.26 | 1.29 | – | – | – |

  - Against the fraction of training instead, the 16M speedup (1.19 → 1.30×) is below 4M's (1.23 → 1.38×). That is prediction 1 as worded, and it fails.
  - The large 16M loss gap comes from the steep curve at loss ~5.
- **Reading.**
  - The realized speedup depends on where training is, not only on batch size. At 4M it rises from ~1.13–1.2× at loss 5.4 to 1.3–1.37× at loss 4.0.
  - At matched loss, the larger batch gives the larger speedup (16M 1.26–1.29×, 4M 1.18–1.26×, both at loss 5.0–5.4), and 4M's lead over 1M appears below loss ~4.6.
  - With a fixed token budget, a 16M run ends at loss 4.9, before the phase where the maps' advantage is largest. So "the gap grows with batch size" holds at matched loss, but a fixed-token comparison at 16M understates it.
  - Measuring 16M in the low-loss phase would need ~4× the tokens (~370 steps).
- Figure: `logs/muon_spectra/second_order_audit_20260926/figures_floor/speedup_by_batch.png`, whose right panel shows the matched-loss view.

**Output-side arm in Track 3: complete, a small gain on top of full PD (2026-09-27 18:40 CDT).**
- `train_gpt_pd_pdwd_a0.125_s3150_row.py`, n = 3: finals 3.27681 and 3.27746 (A6000), 3.27700 (RTX 6000 Ada); mean 3.27709, SD 0.0003.
- **Against full PD's 6 runs** (3.27816, SD 0.0010): −0.0011, suggestive but not established.
  - t = −1.5 with full PD's SD for both arms, −1.7 pooled, −2.3 Welch.
  - Every output-side run is below the full-PD mean, and below the full-PD runs on the same GPU types (A6000 3.27886; Ada 3.27730 and 3.27989).
- **Lead over the run:** −0.005 to −0.010 early, about −0.0025 from step 1500 to 2500, shrinking through the cooldown to −0.0011. Same pattern as IsoMuon and TS.
- **Track 3 criterion:** passes at 3150 with n = 3 (score 0.0050; full PD needed n = 6). At 3125 the score is 0.0039, just short.
  - A 4th run near the current mean at 3125 (3.27772) would pass at 3125: 25 steps fewer than full PD's claim, 125 fewer than #36.
- Figure: `track3/figures/pd_vs_muon_3150_variants.png` (`plot_pd_vs_muon.py --variants`).

**Prior art note: TS is essentially GO-MUON (2026-09-27 ~19:10 CDT).**
- GO-MUON (Tong Che, "Second-Order Muon Done Right", arXiv:2608.09763, Aug 2026) uses D = P_B polar(P_B M P_A) P_A, rescaled to Muon's Frobenius norm.
  - P_A = A^−¼, where A is the activation second moment.
  - P_B = B̄^−¼, where B̄ is the backpropagated output-gradient second moment E[δ δᵀ].
  - Labels are observed, not sampled. Both factors use OAS shrinkage, B has an EMA of 0.97, and the factors are refreshed every 4 steps.
  - Tested only at toy scale: 3-layer transformers of width 64–128.
- **TS at α = β = ¼ with data labels ("ef") is that update.** The differences are in estimation:
  - TS's default samples labels from the model; data labels gave 60% of TS's gain.
  - TS normalizes by the mean eigenvalue with fixed 10⁻³ damping, not OAS.
  - TS estimates B in a separate 8-sequence pass every 10 steps.
- **PD is GO-MUON's input half.** The report in `logs/muon_spectra/improve_w4_20260925` already cites GO-MUON as the closest prior work for PD. TS results should cite it the same way: they are a scale-up test of GO-MUON's form, with GN labels as the default.

**PD in the exact GN geometry: 2× PD and 1.4× damped GN at the floor; worse than PD at the oscillating state (2026-09-27 19:07 CDT).** `gnpd_probe.py`, Muon's 1M states. Input M' = 0.95 M + g1M; cross-fitted one-step decrease ×1e-3, with damping chosen on the other half for GN and GN-PD:

| State | Muon | PD α ¼ | PD α ½ | damped GN | GN-PD p ¼ | GN-PD p ½ |
|---|---|---|---|---|---|---|
| @500, oscillating | 4.41 | **4.71** | 4.41 | 2.31 | 3.30 | 1.77 |
| @516, floor | 3.31 | 4.94 | 5.71 | 8.28 | **11.78** | 11.65 |

- **Prediction 1 holds, far beyond its threshold.** At the floor, GN-PD reaches 1.42× damped GN's decrease (predicted ≥ 0.8×) and 2.1× PD α ½'s.
- **Prediction 2 fails.** At the oscillating state, GN-PD is below PD at both powers, worse at the stronger power. The exact curvature is misled there as GN is (the momentum's stale stiff components).
- **What exceeding "GN" means.** Damped GN solves the quadratic model with M' as its linear term. On held-out data that is not the optimum, and the polar step in GN's whitened coordinates is more robust to M's mismatch with the current gradient.
  - GN-PD also searches a richer space: two Krylov spaces, from b and from polar(w), and the polar step adds directions outside the first.
  - So the one-step "gap to GN" measured all day understates what a normalized map in the right geometry can take at the floor.
- **Implications.**
  - PD's principle (polar in whitened coordinates) is right, and its Kronecker, input-only whitening is where it falls short. The exact geometry doubles its floor decrease.
  - The dichotomy of the day holds again. Curvature geometry pays at the floor and misleads at the edge on the stale momentum, so where GN-PD can pay in training is the cooldown, or wherever the momentum is made stale-free.
  - The g4M input (the same states) is still computing.

**Decision argument: GN-geometry PD in the cooldown (2026-09-27 19:08 CDT; design from the 16:38 review).**
- **Why.** Every curvature-weighted direction on the momentum loses at oscillating states and wins at the floor. GN-PD's floor win is the largest measured: 2.1× PD α ½ and 1.4× damped GN.
  - In training the state approaches the floor only in the cooldown.
  - The earlier "curvature pays in the cooldown" check (PD vs Muon, 16:25) found little there. PD's floor edge over Muon is 1.7×, GN-PD's is 3.6×, so this is the strongest test of the idea.
- **Design** (`newton_train.py`, momentum 0.95 as the run, 2× A6000 on the gpu partition), from Muon's 1M checkpoint at step 1300 to the end of the schedule (1469; cooldown from ~1322):
  - **A:** Muon's own direction in the harness (`--normalize 0 --polar-control`). It must land within 0.001 of the baseline's final 3.7047. Run in two legs, 1300→1450 then 1450→1469.
  - **B:** GN-PD p ¼ (`--normalize 1e-3 --gnpd 0.25 --cg 64`) from 1300, with the per-matrix step norms and LR of Muon's schedule.
  - **C:** A's state at 1450, then GN-PD for the last 19 steps.
- **Predictions.**
  1. B − A ≤ −0.003 at 1469. B − A > −0.003 falsifies "GN geometry pays at the floor" in training at 1M.
  2. C captures ≥ 80% of B's gain if the benefit is a final relaxation, and less if GN-PD finds a faster path during the cooldown.
  3. Over 1300–1330, before the LR falls much, B loses to A, as at the oscillating state (3.3 vs 4.7 one-step), and recovers as the LR decays.
- **Cost.** Two 64-step Lanczos runs per step (~50 s per step on 2 GPUs): B ≈ 2.3 h, C ≈ 20 min, A ≈ 15 min.
- **Addendum (19:09 CDT): LR for the new direction.** At the floor, GN-PD's cross-fitted best scale is 0.031–0.033 × Muon's per-matrix norm. That is 4× the trainer's LR (0.007), 2× PD α ½'s best (0.015) and 10× Muon's own (0.003). The halves agree (11.6 / 12.0).
  - So at Muon's schedule GN-PD under-steps. The strength–LR principle says stronger whitening wants a larger LR.
  - Added **B3**: GN-PD p ¼ at 3× Muon's schedule (`--lr-scale 3`, weight decay unchanged). B and B3 bracket the LR.
  - The predictions apply to the better of B and B3, and the arms may move to g20 if the gpu partition's A6000s stay unavailable.

**Two-sided diagonal PD in Track 3: complete; the output factor recovers half the diagonal gap (2026-09-27 19:55 CDT).**
- `train_gpt_pd_pdwd_a0.125_s3150_diagrow.py`, n = 2 (A6000): finals 3.27950 and 3.28120; mean 3.28035.
- **Against full PD** (3.27816): +0.0022 (t +2.6 with full PD's SD). **Against diagonal-input only** (3.28256): −0.0022.
  - The output factor is worth −0.0022 on a diagonal input, against −0.0011 on top of full PD. It does more when the input side is weak, the upper end of the expectation.
- **Share of full PD's gain kept**, against the estimated Muon level on this schedule (3.2855–3.2866):
  - full PD + output factor: ~1.15;
  - both diagonal: ~0.7;
  - diagonal input only: ~0.4–0.5.
- The all-diagonal version needs almost no memory, and it approximates IsoMuon with our input statistic (exponent ⅛, unclipped) and geometry decay. It does not reach 3.28 at 3150 on average.
- Figure: `track3/figures/pd_vs_muon_3150_variants.png`.

**GN-PD on a fresh 4M gradient at the oscillating state: best of all maps, including GN (19:59 CDT).** `gnpd_probe.py`, Muon 1M @500, input g4M; cross-fitted ×1e-3 (share of damped GN):
- Muon 14.47 (0.55); PD α ¼ 17.06 (0.65); PD α ½ 18.10 (0.69); damped GN 26.34 (1.00); **GN-PD p ¼ 29.56 (1.12); GN-PD p ½ 30.46 (1.16)**.
- **Reading.**
  - GN-PD's weakness at the oscillating state (3.30 vs PD's 4.71 on M') belongs to the stale momentum input, not to its geometry. On a fresh gradient at the same state it is the best map measured, 1.7× PD α ½.
  - This is the same split the morning's transport study found: curvature maps do well on fresh or stale-free inputs at oscillating states, and badly on the actual momentum.
  - **Caveat (20:01).** At an oscillating state, one-step decreases on a fresh gradient are dominated by correcting the oscillation (15–30e-3 here, against a stored excess of 38e-3 at this state). So this result shows GN-PD corrects the wall better than GN and PD. It does not show faster progress along the floor; the floor results on M' are that measure.
  - What stands between GN-PD's floor value and training at the edge is plausibly the momentum's staleness in the stiff directions (the stale components anti-aligned with the current gradient, 09:29–14:31 CDT entries). That is a hypothesis for the candidates below, not a demonstrated cause.
  - Candidates that address that directly, each with a principled basis:
    - a split momentum: short β on the top curvature directions, long β on the bulk;
    - curvature transport of the momentum (M' + G Q);
    - GN-PD on a fresh gradient with a separate light average for the bulk.

**GN-PD, complete one-step table (20:02 CDT).** `gnpd_probe.py`, Muon 1M; cross-fitted ×1e-3 (share of damped GN):

| State · input | Muon | PD α ¼ | PD α ½ | damped GN | GN-PD p ¼ | GN-PD p ½ |
|---|---|---|---|---|---|---|
| @500 oscillating · M' | 4.41 (1.91) | 4.71 (2.04) | 4.41 (1.91) | 2.31 (1) | 3.30 (1.43) | 1.77 (0.77) |
| @500 oscillating · g4M | 14.47 (0.55) | 17.06 (0.65) | 18.10 (0.69) | 26.34 (1) | 29.56 (1.12) | **30.46 (1.16)** |
| @516 floor · M' | 3.31 (0.40) | 4.94 (0.60) | 5.71 (0.69) | 8.28 (1) | **11.78 (1.42)** | 11.65 (1.41) |
| @516 floor · g4M | 4.88 (0.40) | 6.67 (0.54) | 7.61 (0.62) | 12.33 (1) | 14.64 (1.19) | **14.98 (1.21)** |

- **At the floor**, polar in the exact GN geometry is ~2× PD α ½ and 1.2–1.4× damped GN, on both the averaged and the fresh input. This is the largest one-step gain measured today, and it is a floor measure, so it is progress along the floor rather than wall correction.
- **The one failure is the actual momentum at the oscillating state,** where every curvature-weighted map loses to the plain polar maps.
- **Harness for the cooldown test.** The Muon control inside `newton_train.py` from 1300 reproduces the baseline to the end: 3.7765 at 1350 (3.7763), 3.7405 at 1400 (3.7403), 3.7100 at 1450 (3.7098), **3.7047** at 1469 (3.70465).

**16M cohort complete: a larger speedup at matched loss, a smaller one at matched fraction of training (20:03 CDT).** `soaudit_batch16m_20260927`, 92 steps, 1.54e9 tokens, seed 260925, momentum 0.9. Final validation:

| Optimizer | LR 0.014 | 0.02 | 0.028 | 0.04 |
|---|---|---|---|---|
| Muon | 4.9263 | **4.9104** | 4.9635 | 5.0439 |
| S∘PD α ½ | | | **4.5742** | 4.5846 |
| TS α ½ β_out ½ | | | **4.6677** | 4.6747 |
| PD α ½ | | | **4.6711** | 4.6767 |

- **Gaps to the best Muon:** S∘PD −0.336, TS −0.243, PD −0.239. At 4M: −0.095, −0.085, −0.062.
- **Step-equivalent speedup over Muon @0.02**, from the per-step training loss on unseen data (`speedup_by_batch.py`):
  - By fraction of training (25 → 88%): S∘PD 1.21 → 1.32×, TS 1.10 → 1.22×, PD 1.09 → 1.21×. At 4M: 1.23 → 1.38×, 1.15 → 1.33×, 1.15 → 1.26×.
  - At matched loss (Muon's loss 5.0–5.4), 16M against 4M: S∘PD 1.26–1.31 vs 1.18–1.26; TS 1.18–1.22 vs 1.09–1.18; PD 1.17–1.21 vs 1.08–1.17.
- **Predictions.**
  1. Speedup at the same fraction of training above 4M's: **fails**. The 16M runs end at loss 4.9, still in the early phase, where every speedup is lower.
  2. Final gap larger than at 4M: **holds** (−0.336 vs −0.096), but mostly because the curve is steep at loss ~5.
  3. Best LRs within one bracket step of √ scaling: **holds**. Both optimizers prefer the step below it (Muon 0.02 instead of 0.028; S∘PD 0.028 instead of 0.04).
- **Reading.**
  - The realized advantage of the curvature-aware maps grows with batch size at matched loss, and grows through training at a fixed batch.
  - A fixed-token comparison at a large batch stops before the phase where the advantage is largest, so it understates it in speedup terms and overstates it in loss terms.
  - At 16M the output factor adds nothing over PD (TS ≈ PD). SOAP's per-entry normalization in the Kronecker eigenbasis adds the most (S∘PD − PD = −0.097, against −0.033 at 4M): its share of the gain grows with batch size.

**Decision argument: TS (full input + full output) in Track 3 (2026-09-27 20:03 CDT).**
- User direction: "Let's do full PD and on input and output (TS) too."
- **Arms.** Full PD (α ⅛, PD-geometry decay, 3150-step compressed schedule) plus the TS/GO-MUON full-matrix output factor, update L polar(L M R) R.
  - L = (B/mean eig + 10⁻³ I)^−β. B is the per-token E[e eᵀ] of the loss gradient at each hidden matrix's output.
  - B uses GN labels (sampled from the model) from an eager, eval-mode pass on 8 local sequences every 10 steps, with EMA 0.8 per refresh. It is owner-local, with an FP32 eigh.
  - This is a port of `output_second_moments` from `research/adamw_spectra/muon.py`.
  - Two exponents, n = 2 each, since the output strength was never tuned in Track 3:
    - `track3/train_gpt_pd_pdwd_a0.125_s3150_ts_b0.125.py` (sha256 prefix 5cb373dcae81f56b): β = α = ⅛, matched as in TS's ¼/¼;
    - `..._ts_b0.25.py` (5a77c736eda96f1c): β ¼, TS's validated output value and GO-MUON's quarter power.
  - The decay stays input-side only, as in the other arms.
- **Not a Track 3 submission.** The statistics pass is an extra forward-backward every 10 steps, which rule 2 disallows. A compliant version would take data labels from the training backward itself (worth 60% of TS's gain locally).
- **Comparison base.** Full PD (3.27816, n = 6) and the diagonal output factor (−0.0011, n = 3).
- **Expectation.**
  - In the local setting, TS beat PD by −0.0071 at 1M. Our Track 3 effects have been smaller: the diagonal output factor gave −0.0011.
  - Rough guess: −0.001 to −0.003 against full PD.
  - LR is untuned (#36's 0.025), and stronger preconditioning moved the optimal LR up in the other study.
- **Jobs.** Smoke test 2625383 (60 steps, β ¼). Runs 2625384–2625387 depend on the smoke test's success.
- **Possible follow-up.** The output factor's early lead shrinking through the cooldown is what the decay-equilibrium picture predicts when row-wise steps are not protected by the decay. A two-sided geometry decay, W ← W − ηλ·L^p W R^p/norm, would test that.

**Prediction before measuring: where GN-PD's floor gain comes from (2026-09-27 20:05 CDT).** `blockgnpd_probe.py` runs GN-PD in two per-matrix geometries:
- **Kronecker B ⊗ C.** This is the two-sided map's geometry, at p ¼ and ½.
- **Exact per-matrix GN** from stored per-token factors: G_m V = (1/N) Σ_t d_t (d_tᵀ V x_t) x_tᵀ, with d_t the sampled-label error. It keeps each token's coupling of input and output directions, needs no forward or backward pass per product, and drops only cross-matrix terms. Built from 64 curvature sequences (32k tokens), plus a 128-sequence check at the floor.
- **Prediction.**
  1. At the floor on M', block GN-PD keeps ≥ 70% of full GN-PD's decrease (≥ 8.2 of 11.78). Per-matrix exact GN kept 0.83 of damped GN earlier.
  2. Kronecker GN-PD lands near the two-sided map (≈ 5.5).
  - If 1 holds, an efficient GN-geometry PD is within reach: per-token factors from a subsample, per-matrix Krylov. Its training test would cost about one extra backward pass per statistics refresh.
  - If block ≈ Kronecker, the within-matrix token coupling is not the key, and the gain sits in cross-matrix coupling.

**GN-PD decomposition, first pass: Kronecker gives only the two-sided level; the token-factor block estimator is too noisy to decide (20:29 CDT).** `blockgnpd_probe.py`, Muon 1M; cross-fitted ×1e-3, best of p ¼ / ½:

| State · input | Kronecker B ⊗ C | per-matrix, token factors, 64 seq | same, 128 seq | full GN (gnpd_probe) | PD α ½ |
|---|---|---|---|---|---|
| @500 · M' | 4.76 | 4.44 | – | 3.30 | 4.41 |
| @500 · g4M | 17.32 | 12.83 | – | 30.46 | 18.10 |
| @516 floor · M' | 5.97 | 4.08 | 4.64 | 11.78 | 5.71 |
| @516 floor · g4M | 7.43 | 5.09 | 5.90 | 14.98 | 7.61 |

- **Prediction 2 holds.** GN-PD in the Kronecker geometry sits at the two-sided / PD α ½ level (5.3–6.0 on M' at the floor), so the 2× gain is not in the Kronecker factors.
- **Prediction 1 fails for this estimator, but not as a verdict on the geometry.** The per-matrix operator from stored per-token factors (one sampled label per token) scores below Kronecker, always picks the largest damping (1e-2), and improves with more tokens (64 → 128 sequences: 4.08 → 4.64). That is the signature of estimation noise.
- **The clean test is running** (`exactblock_gnpd_probe.py`): the exact GN with the full softmax Hessian, restricted to each matrix (48 blocks) and to each layer (8 blocks), on the same 256 curvature sequences, at p ¼. It separates within-matrix structure from cross-matrix coupling.

**GN-PD with exact per-layer blocks keeps 82% of full GN-PD at the floor (20:59 CDT).** `exactblock_gnpd_probe.py --blocks layer`: the exact GN restricted to each layer's 6 matrices (8 blocks, 256 curvature sequences, Lanczos 16 per block), p ¼, damping cross-fitted over {1e-3, 1e-4} ρ_b. Input M'; ×1e-3:
- Floor @516: **9.68**. Full GN-PD 11.78; Kronecker GN-PD 5.3–6.0; PD α ½ 5.71; damped GN 8.28.
- Oscillating @500: 2.96. Full GN-PD 3.30; PD α ¼ 4.71.
- Floor @516 on g4M: **13.15**. Full GN-PD 14.64 (p ¼); damped GN 12.33; Kronecker 6.86–7.43; PD α ½ 7.61. Per-layer exact geometry keeps 82–90% of full GN-PD's floor gain on both inputs.
- **Reading.**
  - Cross-layer coupling carries only ~18% of GN-PD's floor gain. The rest lives within layers, in structure the per-matrix Kronecker factors miss.
  - At the oscillating state the per-layer version inherits full GN's weakness on the stale momentum.
  - The per-matrix exact blocks (running) will show whether coupling between a layer's matrices matters or each matrix's own exact curvature suffices.

**TS in Track 3, β ¼: the full output factor gains 3× the diagonal one, and holds (2026-09-27 22:12 CDT).**
- `train_gpt_pd_pdwd_a0.125_s3150_ts_b0.25.py`, n = 2 (A6000, 2625384 and 2625385): finals 3.27567 and 3.27477; mean 3.27522.
- **Against full PD** (3.27816): −0.0029 (t ≈ −3.5 with full PD's SD). Both runs are below every full-PD run, and 0.0032–0.0041 below the A6000 full-PD run.
- **Against PD + the diagonal output factor** (3.27709): −0.0019.
- **The lead does not shrink.** It is −0.0105 at step 500, then a steady −0.0028 to −0.0033 from step 1500 to the end. The diagonal output factor's lead fell from −0.0023 to −0.0011 over the same span.
  - So the full-matrix output structure matters, as on the input side. This matches the other study's placebo, where B's spectrum in a random basis lost.
- **Track 3 criterion (n = 2):** passes at 3100 (score 0.0040; 3125: 0.0059; 3150: 0.0068).
  - This is **not** a legal Track 3 result: the GN statistics pass is an extra forward-backward every 10 steps.
  - A legal version needs output-gradient statistics from the training backward itself, with data labels.
- **β ⅛** (2625386, 2625387) is running. At steps 2000–2500 it leads full PD by −0.0039 to −0.0043, more than β ¼ at the same steps.

**Cooldown test: GN-PD gains 0.003, short of the predeclared threshold (22:24 CDT, 2026-09-27).** From Muon's 1M checkpoint at step 1300 to the end of the schedule (cooldown from ~1322). Validation; 2× RTX 6000 Ada on g20:

| Arm | 1325 | 1350 | 1375 | 1400 | 1425 | 1450 | **1469** | vs A |
|---|---|---|---|---|---|---|---|---|
| Baseline (L40S) | | 3.7763 | | 3.7403 | | 3.7098 | 3.70465 | |
| A: Muon in the harness | 3.7923 | 3.7765 | 3.7601 | 3.7405 | 3.7241 | 3.7100 | **3.7047** | 0 |
| B: GN-PD p ¼, 1× LR | 3.7799 | 3.7664 | 3.7509 | 3.7338 | 3.7182 | 3.7062 | **3.7019** | −0.0028 |
| B3: GN-PD p ¼, 3× LR | 3.8920 | | 3.8465 | 3.8086 | 3.7766 | 3.7501 | **3.7398** | +0.0351 |
| C: A to 1450, then GN-PD 3× | | | | | | 3.7100 | **3.7040** | −0.0007 |

- **The harness is exact.** A reproduces the L40S baseline on Ada to < 0.0001 at every checkpoint.
- **Prediction 1 (B − A ≤ −0.003) fails narrowly.** B − A = −0.0028. That is small, apparently real, and 3% of the cooldown's 0.09 decrease.
- **Prediction 2.** C captures 25% of B's gain, so B's gain is not a final relaxation; it accrues during the cooldown.
- **Prediction 3 (B loses early) fails at 1×.** B led from the start (−0.012 at 1325), and the lead shrank as the control's cooldown released its stored loss. At 3×, B3 lost 0.10 in the first 25 steps (the peak LR at an oscillating state) and never recovered.
- **Reading.**
  - At the trainer's schedule, GN-PD's 2× one-step advantage at the floor becomes ~0.003 in training. Its steps are far below its floor optimum (~4× the peak LR) through most of the cooldown.
  - A larger fixed multiple is destructive while the state is still at Muon's edge.
  - The direction's preferred step size changes as the state moves from the edge to the floor, and a fixed multiple of Muon's schedule cannot follow it. This is the strength–LR principle again, now along the trajectory.
  - A principled rule would tie GN-PD's step to its own curvature scale, not to Muon's schedule. Stepping at the one-step optimum c* is the greedy rule that failed at 16:40. The successful optimizers sit at ~2× their own c* (neutral at c = 1), so a fixed multiple near 2 of a measured c* is the candidate.
  - The cheaper exact-block geometry (per-layer keeps 82–90%) is what would make such a rule affordable.

**GN-PD decomposition complete: ~80% of the floor gain lives within single matrices (22:39 CDT).** `exactblock_gnpd_probe.py --blocks matrix` restricts the exact GN to each of the 48 matrices (256 curvature sequences, Lanczos 16 per block, p ¼). Muon 1M floor @516, input M', ×1e-3 (share of full GN-PD):

| Geometry of the polar map | Decrease | Share |
|---|---|---|
| PD α ½ (input covariance only) | 5.71 | 0.48 |
| Kronecker B ⊗ C (two-sided) | 5.3–6.0 | 0.45–0.51 |
| **exact per-matrix GN blocks** | **9.26** | **0.79** |
| exact per-layer GN blocks | 9.68 | 0.82 |
| full exact GN | 11.78 | 1 |
| (damped Newton, full GN) | 8.28 | 0.70 |

- **Within single matrices, ~80%.** Most of what the exact geometry adds over PD is structure inside each matrix's exact GN block that no Kronecker factorization holds: the token-by-token coupling of input and output directions. Coupling across a layer's matrices adds ~3%, and across layers ~18%.
- **Principle.** PD's (and TS's) assumption that each matrix's curvature factors into separate input and output sides is what costs half of the attainable one-step decrease at the floor. The polar map is fine; its geometry is the approximation.
- **What to estimate.** Each matrix's exact GN block, cheaply and with low noise. The token-factor estimator (one sampled label per token) was too noisy at 32k–65k tokens. Candidates:
  - several label draws per token;
  - more tokens, refreshed like PD's statistics;
  - a low-rank-plus-Kronecker representation of the block.

**TS in Track 3, β ⅛: complete; the best preconditioner so far (2026-09-27 22:50 CDT).**
- `train_gpt_pd_pdwd_a0.125_s3150_ts_b0.125.py`, n = 2 (A6000, 2625386 g1 and 2625387 g14): finals 3.27437 and 3.27476; mean 3.27457.
- **Against full PD** (3.27816): −0.0036 (t ≈ −4.3 with full PD's SD). **Against TS β ¼** (3.27522): −0.0007, not resolved.
  - Its lead is steady at −0.0035 to −0.0045 from step 1000 to the end.
- **Track 3 criterion (n = 2):** passes at 3100 (score 0.0050); 3075 scores 0.0023.
  - As for β ¼, this is **not** a legal Track 3 result, because of the extra GN statistics pass.
- **All TS runs (both β, n = 4):** mean 3.27490, −0.0033 against full PD. All four are on A6000; the one A6000 full-PD run was 3.27886.
- **Reading.** The full-matrix output factor (B = E[e eᵀ], GN labels) gives a durable gain on top of full PD, about 3× the diagonal output factor. The milder matched exponent (β = α = ⅛) is at least as good as ¼.
- **Next, if pursued:**
  - a legal version with output-gradient statistics from the training backward (data labels);
  - more seeds on Ada GPUs;
  - a two-sided geometry decay.

**Second independent review: GN-PD and the cooldown test (22:57 CDT, read-only reviewer; brief in the session scratchpad). Accepted, with corrections.**
- **Verified:** every GN-PD, decomposition and cooldown number reproduces from the JSONs, and in-sample vs cross-fitted differ by ≤ 0.03e-3.
- **Artifacts ruled out:**
  - Muon's per-matrix allocation (GN at Muon's norms scores below GN's own);
  - the damping grid (a 7-point grid gives the same GN);
  - GN's Krylov length (48 → 96 steps: +4%);
  - the polar method (exact SVD everywhere);
  - the scoring sets (disjoint from curvature and gradient sets).
  - Plain powers (G+μ)^-p b without the polar score below GN, so the polar step is what adds the gain.
- **Corrections to the entries above.**
  1. **Model, not loss.** The scores are the held-out GN quadratic model's decrease at each direction's fitted scale, not the true loss. GN-PD's best scale is ~10× Muon's, where the model can drift from the loss (for GN at this floor the true loss sat 5.9e-3 above the model at 2× c*). **GN-PD's true-loss decrease is unverified.**
  2. **One state.** The floor claim rests on step 516 only.
  3. **The 128-step control does not test the search space.** GN-PD lives in the Krylov space of polar(w). The fair control solves GN in the union of both Krylov spaces.
  4. **"~80% within single matrices" is provisional.** The comparison has unmatched Krylov budgets, dampings (picks at the grid edge) and curvature sets (64–128 vs 256 sequences).
  5. **The token-factor operator is biased, not just noisy.** Σ_t d_t (d_tᵀ V x_t) x_tᵀ drops the cross-position terms (t ≠ s) of the per-sequence gradient's covariance. The docstring's "unbiased for the exact GN block" is **false** for every matrix feeding a later attention layer; its 64 → 128 gain extrapolates to about the Kronecker level.
  6. **The cooldown has a simpler reading than "the preferred step grows along the trajectory".**
     - GN-PD's slope along its step, per unit of Muon-normalized step, is 0.33–0.48 of Muon's, and its curvature is 16–29× smaller. So it wins only at large steps.
     - Plugging the cooldown's LRs into the floor quadratic: GN-PD is better at LR 0.007 (−5.5e-3 vs Muon's +1.3e-3). They cross at LR ≈ 0.0038 (~step 1390), and Muon is better below that. This matches B's lead peaking at 1325 and eroding from 1375: annealing to zero hands the gain back.
     - C never tested a final relaxation, since its steps were ≤ 0.1 c*.
     - The trained direction also differs from the probed one: BF16 products, 64 in-batch curvature sequences, damping 1e-3 vs the probe's pick of 1e-4.
     - GN-PD sharpens the network too. Over steps 1451–1469 the top GN Ritz value averages 22.7 in B vs 15.9 in C (B3: 26.9), so the polar step does not keep the stiff-direction regulation it was meant to keep.
- **Plan, following the review.**
  1. **True-loss probe** at the step-516 floor and the step-1300 floor (a 16-step anneal from 1300, step 1316). Held-out true loss along GN-PD, GN and PD at 0.5, 1 and 2× each fitted c*. Supports claim 1 if GN-PD's true decrease is ≥ 1.5× PD's at both floors.
  2. **Which structure matters, before any estimator.** GN-PD whitened by (h + μ)^-p in the Kronecker frame, with the per-pair GN diagonal h per sequence (keeps cross-position terms) vs per token (`Frame` already computes both).
     - If per-sequence h reaches ≥ 0.8 of the exact per-matrix score, a SOAP-like estimator suffices. That fits the 16M finding that SOAP's per-entry normalization carries most of the extra gain.
  3. **Then a floor training test** under one shared step rule for Muon, PD α ½ and GN-PD: step = κ (−â / q̂), with slope â and curvature q̂ along the step from independent fresh sequences, EMA-smoothed, κ ∈ {1, 2} fixed in advance.
     - There is no argmin over a noisy grid and no skipping, so it is not the greedy rule. It separates a compounding rate from a one-time release.
- **Not doing:** more cooldown arms at fixed multiples of Muon's schedule; greedy line searches on small held-out sets; GN-PD training at 4M or 16M (45 s per step); scaling up the token-factor estimator; reading "beats damped GN" as beyond second order; relying on single-seed matched-loss 16M speedups as settled.

**Which structure PD's geometry needs: frame-PD, with true-loss checks (23:29 CDT).** `framepd_probe.py`, Muon 1M floors @516 and @1316 (16-step anneals from 500 and 1300).
- **The construction.** W^-p[X] = U[(UᵀXV) / (h + d mean h)^p]Vᵀ in the Kronecker (B, C) eigenbasis, and D = −W^-p[polar(W^-p[b])] at Muon's norms. Matrix powers are exact in the eigenbasis, with no Krylov approximation. Three choices of h:
  - kfac: λ_B λ_C;
  - ekfac: the per-token GN diagonal;
  - exact: the per-sequence GN diagonal, which keeps cross-position terms.
- **Results.** Model decrease ×1e-3; the true held-out loss at c* is within 0.03 of the model in every case.

| Floor · input | Muon | PD α ¼ | PD α ½ | two-sided ¼/¼ | frame kfac p ½ | frame ekfac p ½ | frame exact p ½ | full GN-PD |
|---|---|---|---|---|---|---|---|---|
| @516 · M' | 3.32 | 4.95 | 5.71 | 5.49 | 7.69 | 7.73 | **7.81** | 11.78 |
| @516 · g4M | 4.92 | 6.69 | 7.63 | 6.98 | 8.50 | 8.54 | **8.60** | 14.98 |
| @1316 · M' | 1.99 | 2.83 | 2.46 | 3.08 | 4.41 | 4.54 | **4.58** | (running) |
| @1316 · g4M | 3.14 | 4.12 | 4.03 | 4.27 | 5.36 | 5.46 | **5.48** | – |

- **The GN model is accurate at the floor.** For every direction here, the true held-out loss at c* equals the model's decrease within ~0.03e-3 (e.g. frame exact p ½ at @516: model 7.81, true −7.83). At 2c* it is ~0, as a quadratic predicts. Review point 1 is answered for these directions; GN-PD's own true-loss check is running.
- **Correction to the 20:29 and 22:39 entries.** "Kronecker GN-PD reaches only the two-sided level (5.3–6.0)" was a Krylov artifact: 32 Lanczos steps approximated (B ⊗ C + μ)^-p poorly in its bulk. Computed exactly in the eigenbasis, the Kronecker geometry at p ½ gives 7.7 (M') and 8.5 (g4M), 1.35× and 1.11× PD α ½, and it holds at the later floor.
  - This is essentially the two-sided map at ½/½, which won in 4M training (3.8356 vs 3.8500 for ¼/¼). The one-step measure and training agree.
  - The exact per-matrix GN-PD (9.26) and the full GN-PD also used truncated Krylov matrix functions. Their numbers are lower bounds on what the exact geometry gives, not upper ones.
- **Per-pair diagonals add almost nothing.** kfac → ekfac → exact gains ≤ 0.12, and cross-position terms (exact vs ekfac) add ≤ 0.08. What lies between frame-PD (7.8 / 8.6) and full GN-PD (11.8 / 15.0) is off-diagonal in the Kronecker frame or across matrices.
  - Review point 5's hypothesis, that cross-position terms are the missing piece, is not supported in this frame.

**Decision argument: Track 3-legal TS and two-sided geometry decay (2026-09-27 23:32 CDT).**
- User direction: "Let's make a legal TS and extend the geometry decay to it."
- **Legal statistic.** TS's output statistic B = E[e eᵀ] now comes from the ordinary backward pass, with data labels, so there is one forward-backward per step.
  - Mechanism: an identity `torch.autograd.Function` on each hidden matrix's output. Its backward returns eᵀe over every 16th position as the gradient of a dummy (d_out, d_out) leaf tensor, so autograd accumulates the statistic with no side effects.
  - Unit-tested under `model.compile` against an eager full-backward-hook reference: relative error 0.2–0.6% (bf16), no graph breaks, parameter gradients unchanged. Script: `track3/smoke/probe_test.py`.
  - The B EMA decays 0.978 per step (TS's 0.8 per 10 steps) over 8192 local tokens per step, 10× TS's 8 sequences per 10 steps. Owner-local, FP32 eigh every 10 steps. Cost: eᵀe is ~2.5% of the step's FLOPs.
- **Arms** (β = α = ⅛, the better TS exponent; everything else as TS β ⅛), n = 2 each:
  - A. `track3/train_gpt_pd_pdwd_a0.125_s3150_tsef_b0.125.py` (sha256 prefix 3770ad7f7ff8a1d8): input-side geometry decay, as before. It isolates the label and estimator change against GN TS β ⅛ (−0.0036).
  - B. `..._tsef_b0.125_geo2.py` (b517cce25a9670b6): two-sided geometry decay, W ← W − ηλ (L²/mean eig L²) W (R²/mean eig R²). It is the output-side extension of the equilibrium argument: with plain row decay, row-wise step reweighting would be equalized at equilibrium.
- **Expectations.**
  - A: between the diagonal output factor (−0.0011) and GN TS (−0.0036); about −0.002 if data labels keep ~60% of the gain, as in the local setting.
  - B: better than A if the output side's relative steps need protecting at equilibrium. GN TS's lead already did not shrink, so the extension may add little.
- **Jobs.** Smoke test 2626217 (geo2, 60 steps). Runs 2626218–2626221 depend on its success.

**GN-PD's floor advantage holds on the true loss at both floors (23:41 CDT).** `gnpd_probe.py --true-loss`, input M'. Held-out true loss change at the fitted scale (c* from one half, loss on the other, averaged), ×1e-3:

| Floor | Muon | PD α ¼ | PD α ½ | damped GN | GN-PD p ¼ | GN-PD vs best PD |
|---|---|---|---|---|---|---|
| @516 (anneal from 500) | −3.31 | −4.95 | −5.73 | −8.11 | **−11.68** | 2.0× |
| @1316 (anneal from 1300) | −1.99 | −2.81 | −2.41 | −5.87 | **−7.31** | 2.6× |

- **The review's criterion is met.** GN-PD's true decrease is ≥ 1.5× PD's at both floors. The true loss matches the GN model at c* within ~1% for every direction.
- **At 2c*:** GN-PD sits at +1.15 (516) and +0.00 (1316); damped GN at +2.65 and +0.31.
- **Share of GN-PD's gain over PD α ½ at the 516 floor, by geometry:**
  - exact Kronecker frame (frame-PD p ½): 34% on M', 13% on g4M;
  - Krylov-limited exact per-matrix GN-PD: 58% on M'.
  - The rest is off-diagonal in the Kronecker frame and across matrices.
- **Next, as the review proposed.** A training test from floor states under one shared, curvature-matched step rule for Muon, PD α ½ and GN-PD: step = κ (−â / q̂), with slope and curvature along each direction measured on independent fresh sequences, EMA-smoothed, κ ∈ {1, 2}. It decides whether GN-PD's floor advantage is a compounding rate or a one-time release.

**Decision argument: a rate or a release? Floor steps under one shared step rule (23:41 CDT; design from the second review).**
- **Why.** GN-PD's true one-step decrease at both floors is 2–2.6× PD's, yet the cooldown test turned it into only 0.003. The review's reading is that GN-PD wins at large steps and the annealing schedule handed the gain back.
  - What remains undecided is whether the advantage compounds step after step, or is a one-time release of a structure that PD's steps leave behind.
  - A fixed multiple of Muon's schedule cannot answer this, since each direction's own curvature scale differs ~10×.
- **Design** (`floor_steps.py`, one GPU per arm, FP32 GN products):
  - **Start and inputs.** From the 516 floor state, for N = 24 steps on the run's own next training batches, with momentum 0.95 continuing from the branch's buffer. Only the hidden matrices move: embeddings, gains and head fixed, no weight decay.
  - **Directions.** Muon (polar(M')); PD α ½ (R from the curvature sequences, refreshed every 8 steps); GN-PD p ¼ (exact GN on 256 held-out curvature sequences, Lanczos 64, damping 1e-4 ρ). All at Muon's per-matrix norms.
  - **Shared step rule.** Scale s_t = κ (−â_t / q̂_t) along each direction. â is the slope on one fresh held-out training slice (64 sequences) and q̂ the GN curvature on an independent slice, each EMA-smoothed (0.7). κ ∈ {1, 2} is fixed in advance.
    - This is not the greedy rule: there is no argmin over a noisy grid and no skipping, and every direction gets the same rule.
  - **Evaluation.** The EVAL held-out loss (512 sequences) every 4 steps.
- **Predictions.**
  1. **If the advantage is a rate:** after 24 steps, GN-PD's cumulative decrease is ≥ 1.5× PD α ½'s at the same κ, and its per-step decrease over steps 12–24 is still ≥ 1.3× PD's.
  2. **If it is a one-time release:** by step ~4 the per-step decreases reach parity, and the cumulative ratio falls toward (first-step gain + 23 × PD's rate) / (24 × PD's rate).
  3. **Stability:** κ 2 moves faster than κ 1 for every direction if the edge regime generalizes, and GN-PD's top GN eigenvalue grows (sharpening) under κ 2 but not under κ 1.
- **Cost.** ~2 min per GN-PD step, so ~50 min per GN-PD arm; ~10 min for the others. Six arms in parallel on g20 and priv-g14.

**Krylov-128 control (23:49 CDT).** `gnpd_probe.py --krylov 128`, @516 floor, M' (×1e-3):
- Damped GN 8.78 (8.28 at 64 steps); GN-PD p ¼ 12.13 (11.78); GN-PD p ½ 11.90 (11.65); PD α ½ 5.71; Muon 3.32.
- GN-PD / GN = 1.38 (1.42 at 64 steps), so GN's Krylov convergence does not explain the gap.
- As the review noted, this does not test the search-space question: GN solved in the union of both Krylov spaces is still open.

**Floor steps, first results: PD's floor advantage over Muon is a one-time release; GN-PD's slope turns positive after one step (2026-09-28 00:23 CDT).** `floor_steps.py` from the @516 floor, 24 steps on the run's next batches, momentum 0.95 continuing, hidden matrices only, the shared step rule s = κ(−â/q̂). EVAL held-out loss (512 sequences; 3.97582 at the start for every arm):

| Arm | k 4 | 8 | 12 | 16 | 20 | 24 | Cumulative |
|---|---|---|---|---|---|---|---|
| Muon κ 1 | 3.96963 | 3.96378 | 3.95962 | 3.95512 | 3.95098 | **3.94729** | −0.0285 |
| PD α ½ κ 1 | 3.96636 | 3.96198 | 3.95769 | 3.95350 | 3.94920 | **3.94560** | −0.0302 |
| Muon κ 2 | 3.97986 | 3.97933 | 3.97954 | 3.97792 | 3.97943 | 3.97766 | +0.0018 |
| PD α ½ κ 2 | 3.98200 | 3.98440 | 3.98458 | 3.98488 | 3.98694 | 3.98696 | +0.0111 |

- **PD vs Muon under the shared rule.** PD's one-step floor advantage (1.72×) becomes only **1.06× cumulative**. All of PD's lead comes in the first 4 steps (−0.0095 vs −0.0062). After that both move at ~0.004 per 4 steps, so the advantage at this state is a **one-time release, not a rate**.
- **κ 2 (each direction's own edge) makes no progress in 24 steps** from the floor. Over one short horizon at a fixed state, the κ 1 steps are what pay.
- **GN-PD** (still running, ~640 s per step in FP32; stopping after its step-4 evaluation):
  - First step: a large negative slope along the direction.
  - From step 2–3, the slope on fresh data turns positive: +0.11 for κ 1; +0.33 then +0.48 for κ 2. GN-PD's map of the momentum stops pointing downhill once the first step has moved the state, the same staleness failure as at oscillating states.
  - Its advantage is also a one-time release, at least with the momentum as input.
- **Reading.** At this floor, the better geometries (PD, GN-PD) exploit, once, structure that the Muon-shaped state left behind. The steady rate afterwards is the same for Muon and PD. The one-step floor ratios therefore measure the first step's release, not a rate. To test a rate advantage, a direction must be run from a state its own dynamics produced.

**Decision argument: SOAP on the two-sided ½/½ map, S∘TS (2026-09-28 00:26 CDT).**
- **Why, from today's measurements.**
  1. The Kronecker two-sided geometry at ½/½ is the best Kronecker geometry for the polar map in one step. Frame-PD with h = λ_B λ_C at p ½ scores 7.7 vs PD α ½'s 5.7 at the floor (M'), with the true loss matching. In 4M training TS ½/½ @0.02 beat PD α ½ (3.8356 vs 3.8589).
  2. SOAP's per-entry normalization in the Kronecker eigenbasis carries a gain that grows with batch size: S∘PD − PD is −0.033 at 4M and −0.097 at 16M. Per-pair GN diagonals add nothing one-step (`framepd_probe.py`), so SOAP's value is plausibly dynamic, not a one-step geometry.
  - Their combination was listed in the morning's plan and never run.
- **Code fix first** (live `muon.py`, all 84 unit tests pass). Until now, SOAP's gradient statistics under data norm used G R even when the output factor was on, while the momentum SOAP normalized was L M R. With the output factor they now use L G R, so both live in the doubly whitened coordinates. Configurations without the output factor (S∘PD, PD, Muon) are unchanged, and no completed run used SOAP with the output factor.
- **Arms** (`soaudit_sts_20260928`, β 0.9, seed 260925, configs as the TS ½/½ arms plus S∘PD's SOAP settings):
  - 4M @0.02 (L40S; references S∘PD α ½ @0.02 3.8260 L40S, TS ½/½ @0.02 3.8356 Ada);
  - 16M @0.028 (Ada; references S∘PD α ½ @0.028 4.5742 Ada, TS ½/½ @0.028 4.6677 L40S).
- **Predictions.**
  1. At 4M, S∘TS ≤ S∘PD − 0.005.
  2. At 16M, S∘TS ≤ S∘PD − 0.01.
  - If S∘TS only matches S∘PD, the output factor's gain is absorbed by SOAP's normalization, which also acts on the output side.

**Floor steps, GN-PD: the one-time release is undone within three steps (2026-09-28 00:27 CDT).** Stopped after the step-4 evaluation as planned (~640 s per step in FP32). EVAL held-out loss:
- **Step 4:** GN-PD κ 1 **3.97181** (−0.0040), GN-PD κ 2 4.00741 (+0.0316). Same state and rule: PD α ½ κ 1 3.96636 (−0.0095), Muon κ 1 3.96963 (−0.0062).
- **Slope along GN-PD's direction on fresh data, steps 1–4:** −0.747, −0.093, +0.112, +0.140. The first step gains ~0.011, matching the one-step probe (predicted −0.0115). Steps 2–4 go uphill.
  - The EMA-smoothed step rule kept a positive scale for two steps after the slope turned (EMA slope still negative). That added to the loss but did not cause the sign change.
- **Sharpening:** the top GN Ritz value rises 15.5 → 18.1 (κ 1) and → 21.9 (κ 2).
- **Verdict on the review's floor test.** Prediction 1 (a compounding rate) is refuted for GN-PD on the momentum. Prediction 2 (a one-time release) holds, and more strongly than expected: the release is undone.
  - After one large step, the momentum's old components no longer fit the state, and GN-PD's curvature-weighted map magnifies them into an uphill direction. This is the same staleness failure as at oscillating states.
  - PD's smaller floor edge is also a one-time release (1.06× cumulative after 24 steps), but it is not undone: its map is milder.
- **Where this leaves the GN-geometry line.**
  - One-step floor ratios measure the structure a Muon-shaped state leaves unexploited, not a training rate.
  - A GN-geometry method needs an input that stays fresh after large steps: the fresh gradient, curvature transport of the momentum, or a split momentum.
  - Its rate would have to be measured on its own trajectory, which costs ~0.5–10 min per step with the exact GN, too expensive at present.
  - The realized training gains remain those of the Kronecker-family maps. The next training test there is S∘TS (`soaudit_sts_20260928`, running).
- **Planned next test (00:28 CDT, not launched; both nodes run S∘TS).** Repeat the floor-step GN-PD κ 1 arm with the curvature-transported momentum M' + G Q (Q ← β Q + β/(1−β) ΔW, one extra GN product per step; the morning's `transport_test.py` construction) as the input.
  - **Prediction:** the slope along GN-PD's direction stays negative after the first step, and the cumulative decrease after 8 steps is ≥ PD α ½'s.
  - **Otherwise:** the uphill turn is not first-order staleness, and GN-PD's floor advantage has no training use at 1M with a momentum input.

**S∘TS: SOAP and output-side whitening are substitutes, not complements (2026-09-28 01:00 CDT).** `soaudit_sts_20260928`, β 0.9, seed 260925. Final validation:

| Batch | PD α ½ | S∘PD α ½ | TS ½/½ | **S∘TS ½/½** |
|---|---|---|---|---|
| 4M @0.02 | 3.8589 | 3.8260 | 3.8356 | **3.8257** |
| 16M @0.028 | 4.6711 | 4.5742 | 4.6677 | **4.6595** |

- **Both predictions fail.** At 4M, S∘TS ties S∘PD (−0.0003, not ≤ −0.005). At 16M it is +0.085 behind S∘PD (not ≤ −0.01), and level with TS and PD alone.
- **Reading.**
  - SOAP's large-batch gain (−0.097 over PD at 16M) needs the output side left un-whitened. Adding the output factor L = B^-½ takes SOAP's gain away: SOAP on TS adds only −0.008 at 16M.
  - A plausible mechanism: after full output whitening, the output-side Gram is near-isotropic, so the output-side eigenbasis SOAP normalizes in is poorly determined. SOAP's normalization then acts on the output side as a noisier substitute for L.
  - This repeats the strength principle: stacking two whitenings of the same side over-whitens it.
- **For the method.** At large batch the best composition measured is input whitening (PD α ½) plus SOAP's per-entry normalization, with no output factor. The TS gains at 1M and 4M (without SOAP) came from the same output-side correction that SOAP supplies.
- One seed and one LR per batch size; this is a direction, not a claim.

**Floor steps with transport and an unsmoothed rule: GN-PD's floor advantage is a larger one-time release, not a rate (2026-09-28 02:29 CDT).** `floor_steps.py` from the @516 floor, 8 steps, κ 1. `--ema 0` is the instantaneous slope/curvature rule; `--transport` feeds the momentum carried to the current weights (M + G D). EVAL held-out loss:

| Arm (κ 1) | k 2 | 4 | 6 | **8** | Cumulative | k 6 → 8 |
|---|---|---|---|---|---|---|
| Muon, unsmoothed | 3.97138 | 3.96915 | 3.96596 | **3.96331** | −0.0125 | −0.0027 |
| PD α ½, unsmoothed | 3.96811 | 3.96497 | 3.96290 | **3.95948** | −0.0163 | −0.0034 |
| PD α ½, transported, unsmoothed | 3.96844 | 3.96585 | 3.96239 | **3.95967** | −0.0162 | −0.0027 |
| GN-PD, unsmoothed | 3.96329 | 3.96168 | 3.96018 | **3.95900** | −0.0168 | −0.0012 |
| **GN-PD, transported, unsmoothed** | 3.96247 | 3.95990 | 3.95679 | **3.95542** | **−0.0204** | −0.0014 |
| GN-PD, transported, smoothed (0.7) | 3.96446 | 3.96381 | 3.96198 | 3.95941 | −0.0164 | −0.0026 |

(The first run's smoothed arms at k 4: Muon 3.96963, PD 3.96636, GN-PD 3.97181.)
- **The earlier "uphill turn" was mostly the step rule.** With the smoothed rule, the second step was large (scale 0.024) on the strength of the first step's slope, and it moved the state far enough to make the momentum stale.
  - With the instantaneous rule, GN-PD's slope stays negative with or without transport (−0.22 and −0.31 at step 3).
  - Transport adds a little: −0.0204 vs −0.0168 at k 8.
- **Predictions of the 00:27 transport test.** Both hold: the slope stays negative, and GN-PD's k-8 decrease is at least PD's. **But the review's "rate" criterion fails.**
  - GN-PD / PD cumulative is 1.46× at k 4 and 1.25× at k 8, below the 1.5× bar.
  - Over steps 6 → 8, GN-PD moves more slowly than PD and Muon.
  - The advantage is front-loaded: a larger one-time release that the right step rule keeps rather than undoes.
- **The transport term is large at the floor:** |G D| / |M| is 2–7. A step with stiff components changes the small floor gradient by several times its size. For the polar maps (Muon, PD) transport changes little (±0.002).
- **Reading.**
  - At a state produced by Muon, the better geometries harvest what that state left unexploited: PD modestly, GN-PD about twice as much. Afterwards all three settle to similar per-step rates.
  - A rate advantage, if GN geometry has one, must be measured on a trajectory the geometry itself shaped. With the exact GN that costs ~10 min per step in FP32, so it would need the cheap estimator first.
  - Two lessons for any second-order step here:
    - the step rule must follow the instantaneous slope and curvature along the direction, since smoothing lags after a large step;
    - a first-order transport of the momentum keeps curvature-weighted maps pointing downhill.

**Decision argument: the large-batch speedup in the late phase, 16M for 4× the tokens (2026-09-28 02:31 CDT).**
- **Why.** At 16M the fixed-token runs (92 steps) end at loss ~4.9. At matched loss their speedup over Muon (1.26–1.31×) already exceeds 4M's, but only down to loss 5.0. At 4M the speedup kept rising into the late phase (1.37× at loss 4.0).
  - Whether the large-batch advantage keeps growing there is the most direct measurement of "the gap at larger batch sizes". It needs a 16M run long enough to reach loss ~4.
- **Design** (`soaudit_b16mlong_20260928`). total_tokens 4 × 1.54e9 = 6.16e9, i.e. 368 steps at 16M, the same step count as the 4M runs. Warmup 3 steps, 10% cooldown, β 0.9, seed 260925. Muon @0.02 (L40S) and S∘PD α ½ @0.028 (Ada), the best LRs of the 92-step brackets. Kept checkpoints at 37, 183 and 330 for floors.
  - One LR each: the optimal LR may shift at the longer horizon, and a single pair cannot rule that out.
- **Predictions.**
  1. At matched loss ≤ 4.2, S∘PD's step-equivalent speedup over Muon at 16M is ≥ its 4M value at the same loss (1.36× at 4.2). The batch-size growth then continues into the late phase.
  2. At 368 steps, the final gap to Muon at 16M exceeds the 4M gap at 368 steps (−0.095).
- **Cost.** About 2.3 h per run on one 4-GPU node, overnight.

**Decision argument: which SOAP side carries the large-batch gain? (2026-09-28 02:33 CDT)**
- **Why.** At 1M, SOAP's output-side basis alone kept 79% of S∘PD's gain over PD (wave 16, 2026-09-25). Tonight SOAP turned out to be a substitute for the GN output factor B^-½ (S∘TS), and it carries a gain that grows with batch size (S∘PD − PD: −0.033 at 4M, −0.097 at 16M).
  - If the large-batch gain is output-side too, the output-side treatment that pays at large batch is normalization in the eigenbasis of the whitened *gradient* Gram, from true-label gradient statistics. It is not the GN output curvature (sampled labels), which TS uses and which adds nothing at 16M.
- **Arms** (`soaudit_soapside16m_20260928`, 16M, 92 steps, β 0.9, seed 260925, PD α ½ @0.028 plus SOAP on one side):
  - S_left∘PD (`soap_basis` left: output side only);
  - S_right∘PD (right: input side only).
  - References: S∘PD 4.5742 (Ada), PD 4.6711 (L40S), both @0.028.
- **Predictions.**
  1. S_left∘PD keeps ≥ ⅔ of S∘PD's gain over PD at 16M (≤ 4.607).
  2. S_right∘PD keeps ≤ ⅓ (≥ 4.639).
  - If both hold, the principle is: at large batch the output side wants gradient-statistics normalization (signal-adaptive, Adam-like), and the input side wants the forward-statistics whitening (PD).
- **Revision (02:35 CDT): 2× the tokens, not 4×.** The 4× cohort failed at qualification before any training: "Insufficient tokens; distributed execution will not recycle data". Only 3.2e9 training tokens exist locally (32 shards); the attempt is kept as `soaudit_b16mlong_20260928_failed_tokens`.
  - The cohort now runs total_tokens 3.08e9 (184 steps at 16M), kept checkpoints 37, 92 and 165.
  - Prediction 1 now applies at the matched losses the runs reach (expected down to ~4.2–4.3). Prediction 2 compares with the 4M gap at matched tokens (−0.095 at 1.54e9 tokens, 368 steps).

**Status at 03:04 CDT, 2026-09-28: two tests still running, results not yet recorded.**
- **The 2× 16M pair** (`soaudit_b16mlong_20260928`, 184 steps).
  - Muon @0.02 at step 118: 5.658 at step 50, 4.8306 at 100.
  - S∘PD α ½ @0.028 at step 85: 5.3467 at step 50.
  - Run-to-run noise at 16M this early is ~0.01: Muon 2× vs the 92-step run at step 50, 5.658 vs 5.6707 with identical settings to step 83.
- **The SOAP-side pair** (`soaudit_soapside16m_20260928`, jobs 2626259/2626260) is pending on the gpu partition.
- **`auto_speedup.sh`** (audit folder, log `auto_speedup.log`) re-runs `speedup_by_batch.py` as each of these finishes. It writes `speedup_by_batch.json` (groups "16M long" and "16M") and `figures_floor/speedup_by_batch.png`.
- The verdicts on the two decision arguments above (late-phase 16M speedup; which SOAP side carries the large-batch gain) are still to be written from those outputs.

**Legal TS + two-sided geometry decay, first run: ahead mid-run, lead gone after the cooldown (2026-09-28 03:10 CDT).**
- `train_gpt_pd_pdwd_a0.125_s3150_tsef_b0.125_geo2.py`, job 2626218 (A6000, g1, 4 GPUs): final 3.27710, −0.0011 against full PD (3.27816). That equals the diagonal output factor (3.27709) and is +0.0025 against GN TS β ⅛ (3.27457). n = 1.
- **Trajectory against full PD:**
  - −0.0082 at step 1000, −0.0064 at 2000, −0.0051 at 2500. That is ahead of GN TS β ⅛ at the same steps (−0.0056, −0.0041, −0.0040).
  - Then the lead shrinks through the cooldown: −0.0035 at 2750, −0.0018 at 3000, −0.0011 at 3150. GN TS's lead holds at −0.0036.
- **Not yet separable.** Data labels (EF) and the two-sided decay changed together. The plain legal-TS arm settles it: 2626219 started 03:07 CDT on g1; 2626223 is queued.
  - Candidate reading: the two-sided decay helps during the high-LR phase and costs in the cooldown, or the EF statistic behaves differently late.
- Second geo2 run: 2626222, queued (projected 06:06 CDT).

**The large-batch speedup continues into the later phase (03:37 CDT, 2026-09-28).** `soaudit_b16mlong_20260928`: 16M, 184 steps, 3.08e9 tokens, β 0.9.

| Step | 50 | 100 | 150 | **184** |
|---|---|---|---|---|
| Muon @0.02 | 5.658 | 4.8306 | 4.3893 | **4.1878** |
| S∘PD α ½ @0.028 | 5.3467 | 4.4694 | 4.1435 | **3.9840** (−0.204) |

- **Step-equivalent speedup over Muon at matched loss** (`speedup_by_batch.py`, per-step training loss on unseen data):

| Muon's smoothed loss | 5.4 | 5.0 | 4.6 | 4.4 | 4.2 | 4.0 |
|---|---|---|---|---|---|---|
| 1M S∘PD α ¼ (β 0.95) | – | 1.17 | 1.17 | 1.19 | 1.21 | 1.17 |
| 4M S∘PD α ¼ (β 0.95) | 1.13 | 1.14 | 1.20 | 1.22 | 1.30 | 1.32 |
| 4M S∘PD α ½ (β 0.9) | 1.18 | 1.26 | 1.27 | 1.32 | 1.36 | 1.37 |
| **16M S∘PD α ½ (β 0.9), 2× run** | 1.26 | 1.31 | **1.34** | **1.37** | – | – |

  - By fraction of training: 1.28× (25%), 1.33× (50%), 1.38× (75%), 1.37× (88%).
- **Prediction 1 holds where it can be tested.** At every loss both batch sizes reach (5.4 to 4.4), 16M's speedup exceeds 4M's by 0.05–0.08, and it keeps rising into the later phase. Muon's smoothed loss ends at ~4.3 before the cooldown, so loss ≤ 4.2 is out of reach.
- **Prediction 2 was not well posed:** the 4M runs have no matched-token point in this range. The final gap at 184 steps is −0.204, against −0.336 at 92 steps and −0.095 at 4M / 368 steps.
- **Reading.** The realized advantage of the best curvature-aware map (input whitening plus SOAP's output-side normalization) grows with batch size at every matched loss, from ~1.2× at 1M to ~1.37× at 16M by loss 4.4. It is not an artifact of short runs stopping early. It is the same direction as the GN paper's claim, measured on real training in our setup.
- **Caveats:** one seed, LRs tuned at 92 steps, and a 2× horizon because only 3.2e9 training tokens exist locally.

**Which SOAP side carries the large-batch gain: mostly the output side, and the input side more than at 1M (04:10 CDT, 2026-09-28).** `soaudit_soapside16m_20260928`, 16M, 92 steps, β 0.9, @0.028. The arms ran on g20 (left, Ada) and priv-g14 (right, L40S) after the gpu-partition jobs stayed pending.

| Arm | Step 50 | Final (92) | Share of SOAP's gain over PD |
|---|---|---|---|
| PD α ½ | 5.4623 | 4.6711 | 0 |
| S_left∘PD (output side only) | 5.4024 | **4.6051** | **68%** (57% at step 50) |
| S_right∘PD (input side only) | 5.4405 | **4.6330** | **39%** (21% at step 50) |
| S∘PD (both) | 5.3573 | 4.5742 | 100% |

- **Prediction 1 holds narrowly** (S_left ≤ 4.607; it reached 4.6051). **Prediction 2 fails narrowly**: S_right keeps 39%, above ⅓.
- The two sides are roughly additive by the end (68% + 39%), less than additive at step 50 (57% + 21%).
- **Reading.**
  - SOAP's gain on top of PD comes mostly from the output side at 16M (68%) as at 1M (79%, 2026-09-25). That output side is the eigenbasis of the input-whitened gradient's output Gram, with per-entry second-moment normalization.
  - The input side adds more at 16M (39%) than PD's forward-statistics whitening leaves room for at 1M.
  - With the S∘TS result: at large batch, the output-side treatment that pays is SOAP's gradient-statistics normalization, not the GN output factor B^-½ (sampled labels). On the input side, forward statistics (PD) and gradient statistics (SOAP right) both contribute.
- One seed, one LR; different GPU types per arm. The S_right – S_left difference (0.028) exceeds the ~0.01 run-to-run noise measured at 16M.

**Prediction before measuring: per-entry signal-to-noise in SOAP's frame vs the GN output frame (2026-09-28 04:13 CDT).** `frame_snr_probe.py` takes 64 micro-batch gradients of 32 sequences each, whitened on the input side as PD α ½ does. In three frames it computes each entry's mean and variance: SOAP (eigenbases of the whitened gradient's Grams), gn_out (B's eigenbasis on the output side), and raw. SNR at batch B is mu² B / (s² m). States: 16M S∘PD @46 (loss ~5.4), 4M PD α ½ @183 (~4.1), 1M PD α ¼ @900 (~3.8).
- **Hypothesis.** Per-entry normalization is informative where an entry's signal exceeds its noise. SOAP's frame concentrates the signal, so more of it clears the noise, increasingly with batch size. The GN output frame ignores the signal.
- **Predictions.**
  1. The fraction of signal energy with SNR > 1 is higher in SOAP's frame than in gn_out at every batch size, and the gap grows from 1M to 16M.
  2. The overlap of SOAP's and GN's top-8 output eigenvectors is below 0.3 (median over matrices).
  3. At a given batch size, the SNR fractions are lower at the later states (1M @900) than at the early one (16M @46): the critical batch grows as loss falls.

**Per-entry SNR does not separate SOAP's frame from the GN output frame (04:14 CDT).** `frame_snr_probe.py`, three states. Share of entries with SNR > 1 at 1M / 4M / 16M tokens:

| State | SOAP frame | GN output frame | raw basis | median top-8 overlap SOAP vs GN output basis |
|---|---|---|---|---|
| 16M S∘PD @46 (loss ~5.4) | 0.35 / 0.46 / 0.49 | 0.39 / 0.49 / 0.53 | 0.54 / 0.63 / 0.65 | 0.44 |
| 4M PD α ½ @183 (~4.1) | 0.24 / 0.35 / 0.39 | 0.25 / 0.36 / 0.40 | 0.35 / 0.46 / 0.49 | 0.58 |
| 1M PD α ¼ @900 (~3.8) | 0.21 / 0.31 / 0.35 | 0.21 / 0.32 / 0.36 | 0.28 / 0.39 / 0.43 | 0.59 |

- **Signal energy.** The share of it in entries with SNR > 1 is 0.95–1.0 in every frame at every batch size.
- **Prediction 1 fails.** SOAP's frame does not hold more high-SNR signal than the GN output frame. It has slightly fewer individually significant entries, and so does GN compared with the raw basis: eigenframes concentrate the signal.
- **Prediction 2 fails.** The two output bases overlap moderately (0.44–0.59), not below 0.3.
- **Prediction 3 holds.** The share of significant entries rises with batch size (+0.14–0.15 from 1M to 16M) and falls later in training (0.35 → 0.21 at 1M from the early to the late state): the critical batch grows as loss falls.
- **Reading.**
  - Per-entry SNR in a frame is not what distinguishes SOAP's output-side normalization from GN's output curvature.
  - What remains is how they scale. SOAP divides each frame entry by its RMS, which is dominated by the signal itself (sign-like equalization of uneven signal magnitudes). B^-½ divides by an output curvature independent of the signal.
  - A test of that would measure the unevenness of the signal magnitudes in the frame across batch sizes, and compare SOAP's normalization with a signal-free variant: SOAP's second moment from noise only, E[(g − ḡ)²] instead of E[g²].
- Figure: `logs/muon_spectra/second_order_audit_20260926/figures_floor/frame_snr.png`.
- **Note and next test (04:15 CDT).** In this code, SOAP's default `soap_second_moment` "update" divides the projected momentum by the root of an EMA of its own squares. That makes it nearly the *sign* of the momentum in the Kronecker eigenbasis, followed by the polar step (`soap_precondition` docstring).
  - So S∘PD's output-side gain is plausibly sign-like equalization of uneven signal magnitudes in the gradient's own eigenbasis, which GN output curvature (B^-½) does not do.
  - **The ready contrast is `soap_second_moment` "gradient":** the projected gradient's second moment, μ² + σ², which becomes SNR weighting where the noise matters.
  - **Planned test, not launched:** S∘PD α ½ @0.028 at 16M in "gradient" mode vs the "update" reference (4.5742).
    - If sign-like equalization is the mechanism, the two coincide where most entries are signal-dominated, and "gradient" mode helps less at 1M.
    - If "gradient" mode wins at 16M, SOAP's value there is SNR weighting, not sign.

**Decision argument: is SOAP's large-batch gain sign-like equalization? (2026-09-28 04:16 CDT)**
- **Design** (`soaudit_soapmode_20260928`). S∘PD α ½ with `soap_second_moment` "gradient" (per-entry second moment of the projected gradient: μ² + σ²) in place of the default "update" (the projected momentum's own squares: nearly sign).
  - 16M @0.028 (Ada; reference "update" 4.5742, Ada) and 4M @0.02 β 0.9 (L40S; reference 3.8260, L40S). Everything else as the references.
- **Predictions.**
  1. At 16M, "gradient" mode lands within 0.01 of "update" mode: most signal entries are high-SNR there, so both normalize by ≈ |μ|.
  2. At 4M, the two differ by more than at 16M. If "gradient" wins at 4M, SNR weighting helps where the noise matters; if it loses, sign-like equalization is what matters even there.
  - If "gradient" mode wins clearly at 16M, SOAP's large-batch value is not sign-like equalization.

**SOAP's second-moment source does not matter at ≥ 4M: sign-like equalization and SNR weighting coincide (04:49 CDT, 2026-09-28).** `soaudit_soapmode_20260928`, S∘PD α ½, β 0.9:

| Batch | "update" (default, ~sign) | "gradient" (μ² + σ²) | Difference |
|---|---|---|---|
| 4M @0.02 (L40S) | 3.8260 | 3.8238 | −0.0022 |
| 16M @0.028 (Ada) | 4.5742 | 4.5686 | −0.0056 |

- **Prediction 1 holds** (within 0.01 at 16M). **Prediction 2 fails:** the modes do not differ more at 4M.
- The 4M gradient-mode run led by ~0.01 through step 250, then converged in the cooldown.
- **Reading, with the per-entry SNR measurement.**
  - At ≥ 1M tokens, 95–100% of the gradient's signal energy sits in entries where the signal dominates the noise. There the momentum's own square and the gradient's second moment both ≈ μ², so the two modes give nearly the same step.
  - SOAP's value in S∘PD is dividing each entry of the gradient's own Kronecker eigenbasis by its signal magnitude: equalizing uneven signal magnitudes, mostly on the output side (68% at 16M, 79% at 1M).
  - The GN output factor B^-½ does not do this. It scales by curvature independent of the signal, and at 16M it adds nothing over PD.
- **Principle from these three tests** (S∘TS, SOAP by side, SOAP by mode). The large-batch gain beyond input whitening comes from signal-magnitude equalization in the gradient's own output-side eigenbasis, not from output-side curvature. Polar equalizes singular values; SOAP's normalization also equalizes the entries of the signal within the eigenbasis. Its relative value grows with batch size as more entries become signal-dominated (0.21 → 0.35 of entries from 1M to 16M at a late state).

**Second seed for the main 16M comparison (04:50 CDT, launched).** `soaudit_seed16m_20260928`: Muon @0.02 (L40S) and S∘PD α ½ @0.028 (Ada) at 16M, 92 steps, seed 260926, otherwise as the seed-260925 arms (4.9104 and 4.5742).
- **Prediction.** The gap S∘PD − Muon is within 0.03 of the seed-260925 gap (−0.336), and the matched-loss speedups within 0.05 of 1.26–1.31×.

**Legal TS: the best Track 3 result so far; the two-sided decay is what fades (2026-09-28 05:20 CDT).**
- `train_gpt_pd_pdwd_a0.125_s3150_tsef_b0.125.py`, job 2626219 (A6000, g1, 4 GPUs): final **3.27342**. n = 1.
  - **−0.0047 against full PD** (3.27816, about 4.5 run SDs). −0.0012 against GN TS β ⅛ (3.27457).
  - Its lead is steady at −0.0048 to −0.0055 from step 1500 to the end, with no cooldown fade.
- **Legal:** the output statistic comes from the ordinary backward pass. A single run passes the Track 3 criterion at **3100** (score 0.0047; 3075: 0.0028).
- **The two-sided geometry decay is what fades**, not the data labels.
  - The geo2 run (2626218) matches legal TS mid-run or leads it (−0.0064 vs −0.0048 at step 2000). It then loses 0.0037 against it through the cooldown (final 3.27710).
  - So extending the geometry decay to the output side hurts the final loss. Our input-side equilibrium argument does not carry over to rows as is.
  - A possible reason: L² decays the low-B output directions harder, and those directions' larger weights help late. Untested.
- **Data labels did not cost anything**, unlike the local 1M result (60% of GN's gain).
  - Here the statistic has 8192 tokens per rank per step with an EMA over steps, against GN TS's 8 sequences every 10 steps. The estimator's sample size may matter more than the label source.
- Pending: 2626222 (geo2, second run; started 05:13 CDT) and 2626223 (legal TS, second run; queued).
- Hygiene: the two aborted 2-GPU partial logs were moved to `track3/logs/aborted/`.
- **Result (05:23 CDT).** Seed 260926: Muon @0.02 **4.9625**, S∘PD α ½ @0.028 **4.6055**, gap **−0.357** (seed 260925: 4.9104 vs 4.5742, −0.336).
  - Matched-loss speedup 1.31 / 1.30 / 1.31× at Muon's loss 5.6 / 5.4 / 5.2 (seed 260925: 1.27 / 1.26 / 1.28×). By fraction of training: 1.27× → 1.33×.
  - **Both predictions hold** (gap within 0.03, speedups within 0.05). The 16M advantage of S∘PD over Muon is reproducible across seeds.

**LR check for the late-phase 16M result (05:24 CDT, launched).** `soaudit_b16mlong_lr_20260928`: the lower neighbours at the 2× horizon, Muon @0.014 (L40S) and S∘PD α ½ @0.02 (Ada), 184 steps, seed 260925. References at the 92-step-best LRs: 4.1878 and 3.9840.
- **Prediction.** The better LR for each optimizer changes the final by < 0.02, and the matched-loss speedup at loss 4.4–4.6 stays ≥ 1.30×.
  - If Muon improves by more than S∘PD at the lower LR, the late-phase speedup was partly an LR artifact.
- **Result (06:26 CDT).** At the 2× horizon: Muon @0.014 **4.1920** (@0.02: 4.1878); S∘PD α ½ @0.02 **3.9790** (@0.028: 3.9840).
  - Each optimizer's better LR changes its final by ≤ 0.005. The best-to-best gap is −0.209.
  - Matched-loss speedup with S∘PD @0.02: 1.25 / 1.30 / 1.32 / 1.37× at Muon's loss 5.4 / 5.0 / 4.6 / 4.4 (@0.028: 1.26 / 1.31 / 1.34 / 1.37×).
  - **The prediction holds.** The late-phase large-batch speedup is not an LR artifact.

**Correction: the second two-sided-decay run ends at −0.0038; the fade replicates, the "hurts" does not (2026-09-28 07:20 CDT).**
- `..._tsef_b0.125_geo2.py`, job 2626222 (A6000, g1): final 3.27431, −0.0038 against full PD.
  - geo2, n = 2: 3.27710 and 3.27431; mean 3.27571, −0.0025. The spread of 0.0028 is large against the usual ~0.001.
- **Correction of the 05:20 entry.** "Extending the geometry decay to the output side hurts the final loss" rested on one run and is withdrawn. With two runs, geo2 ends between full PD and plain legal TS (−0.0047, n = 1), and the difference from legal TS is not resolved.
- **What replicates is the shape.**
  - Both geo2 runs lead far more mid-run than plain legal TS: −0.0064 and −0.0095 at step 2000, against −0.0048.
  - Both lose ~0.005–0.006 of that lead through the cooldown: step 2000 to 3150, −0.0064 → −0.0011 and −0.0095 → −0.0038.
  - Plain legal TS is flat (−0.0048 → −0.0047).
- **Reading, not tested.** A lead that disappears as the LR anneals looks like less LR-maintained excess (an effective step-size effect in noisy directions), which the cooldown removes for every method, rather than more progress on the landscape.
- Legal TS second run: 2626223, started 07:11 CDT on g1.

**Legal TS replicates: −0.0041 against full PD with two runs (2026-09-28 09:10 CDT).**
- `train_gpt_pd_pdwd_a0.125_s3150_tsef_b0.125.py`, n = 2 (both A6000, g1): finals 3.27342 and 3.27462; mean 3.27402.
- **Against full PD** (3.27816): −0.0041, t ≈ −4.9 with full PD's SD. It is at least as good as the non-legal GN TS β ⅛ (3.27456, −0.0036).
- **Its lead is steady** from step ~1500 in both runs. The second run's lead at 3150 is −0.0035.
- **Track 3 criterion, n = 2:** passes at 3100 (score 0.0057). 3075 scores 0.0031; at the same mean, n = 4 would pass there (0.0044).
- **Caveats.**
  - All legal-TS runs are on A6000, while full PD's base is mostly L40S / RTX 6000 Ada (its one A6000 run was 3.27886).
  - LR, β and schedule are untuned for TS (#36's LR 0.025, β = α = ⅛).
- **Status of the other arms:** legal TS + two-sided decay, n = 2: −0.0025, with the mid-run lead fading. GN TS, n = 2 per β: −0.0036 (β ⅛) and −0.0029 (β ¼).

**Decision argument: is exact Gauss-Newton geometry a training rate at large batch? GN-PD against its Kronecker control on its own trajectory at 16M (2026-09-28 09:40 CDT).**
- **Missing premise.** The goal assumes a real per-step gap between our optimizers and a true second-order optimizer, largest at large batch. Nothing so far tests it as a rate:
  - the GN paper's inner-loop recipe gains through extra sequential steps (1M, 12:33 on 09-27);
  - the one-step prizes (GN-PD ≈ 2× PD and 1.2–1.4× damped GN at valley floors, on the true loss) are measured at states co-adapted to other optimizers (principle 4);
  - the only training evidence is 8 floor steps (a one-time release) and one 1M cooldown (GN-PD p ¼ −0.003), both from co-adapted states and short.
- **Proposed mechanism.** At 16M most of the gradient's signal energy is above its noise. A geometry that matches the true curvature, including the within-matrix, non-Kronecker structure where the one-step prize sits (per-matrix exact GN-PD 9.3 vs exact frame-PD 7.8 vs PD 5.7 at the 1M floor), would then keep a per-step advantage along the whole trajectory.
- **Simplest competing explanation.** The prize is a harvest of curvature that the state's optimizer never regulated. On its own trajectory GN-PD regulates its own stiff directions and its rate equals the Kronecker geometry's. Or it sharpens the network, as normalized GN did, and loses.
- **Comparison.** Both arms start from the PD α ½ @0.028 16M state at step 9 (seed 260925; that run ends at 4.6711). They share one harness (`newton_train.py`, 4 GPUs) and everything else: the run's schedule and LR 0.028; the momentum EMA (β 0.9, from the checkpoint's buffer) as the right-hand side; gradient clipping 1.0 as in the trainer; per-matrix norms √min(m,n)·√max(1,m/n) × LR; the trainer's weight decay; AdamW for the aux parameters. The arms differ only in the geometry:
  - `kron`: L polar(L M R) R with L = (B/mean + 1e-3)^-½ and R = (C/mean + 1e-3)^-½. B uses sampled labels and C positions ≥ 1, both from the step's curvature sequences. This is TS ½/½ (real trainer: 4.6677) with instantaneous statistics, and it is GN-PD's exact limit when G = B⊗C.
  - `gnpd`: −(G + μ)^-½ polar((G + μ)^-½ M) per matrix. G is the exact GN of all 48 body matrices on the same curvature sequences (Lanczos 64, μ = 1e-3 ρ). Each step also logs its per-matrix cosine with the `kron` direction and the GN's top Ritz value.
  - References (real trainer, same seed and batch): PD 4.6711, TS ½/½ 4.6677, S∘PD 4.5742, Muon @0.02 4.9104. They are compared by step-equivalent speedup at matched loss.
- **Outcomes that change the next decision.**
  - gnpd beats kron by ≥ 1.1× step-equivalent at matched loss, holding or growing to the end: exact curvature is a large-batch rate. Build a cheap estimator of the within-matrix structure, guided by where the logged cosines differ.
  - gnpd leads in the first ~20 steps, then its speedup falls below 1.05×: a harvest even on its own trajectory. Per-step curvature beyond Kronecker is not where the large-batch gap lives at this scale. The remaining levers are signal equalization (S∘PD) and the extra sequential steps.
  - gnpd loses with a rising Ritz top: the exact geometry removes the stiff-direction regulation. Test p ¼ before concluding.
  - kron differs from the real-trainer TS by > 0.03: harness effects are large. The within-harness comparison stands, but the absolute comparison with S∘PD does not.
- **Prediction (before running).** gnpd − kron at the end is in [−0.04, 0], largest in relative terms within the first 20 steps. gnpd's Ritz top ends above kron's.
- **Cost.** kron ~40 min and gnpd ~2 h on one 4-GPU node. The GN-PD cooldown ran 45 s per step at 1M on 2 GPUs with 64 curvature sequences. No data download. An independent review of this design runs before launch.

**Review of the GN-PD rate test, and smoke tests: a pre-flight before any training arm (2026-09-28 09:59 CDT).** Read-only reviewer (brief: the 09:40 entry).
- **Verdict.** The right experiment: it is the only one that tests whether exact curvature gives a per-step rate. As designed, though, the predicted outcome (gnpd ≈ kron) would be uninterpretable, because the arms differ in more than curvature structure.
- **Accepted corrections.**
  - **Damping scale (most serious).** kron damps each factor at 1e-3 of its own mean eigenvalue. gnpd damps G at 1e-3 ρ, with ρ the momentum's Rayleigh quotient, which sits in the stiff part. That can flatten the bulk, and Lanczos-64 cannot resolve a smaller μ.
  - **Estimator fidelity.** The one-step prize used FP32 products, 256 held-out sequences and cross-fitted μ. The harness uses BF16, 128 in-batch sequences and 1e-3 ρ. A pre-flight must show that the harness direction is reproducible (FP32 / Lanczos 128; a disjoint set of 128 sequences) and that it keeps a one-step advantage over kron. Power ½ vs ¼ is settled there too.
  - **Step length.** Log each step's slope and GN curvature, so the taken step can be compared with the model-optimal one per arm. A difference of more than 1.5× makes the result about the LR, not the geometry.
  - **Harness.** Broadcast the updates from rank 0. The kron-vs-trainer calibration rule (> 0.03) would misfire inside the 0.03–0.05 seed spread, so it is dropped.
  - **Thresholds.** Count speedups from the branch point, (t − 9)/(s − 9), and judge a rate by the incremental speedup between kron's step-46 and step-83 losses. If gnpd beats kron but not S∘PD, the next step is composing it with SOAP, not an estimator.
  - **A better construction.** Nest GN-PD around kron: D = −K^-½ (Ĝ + μ̂)^-½ polar((Ĝ + μ̂)^-½ K^-½ M), with Ĝ = K^-½ G K^-½ and K = blockdiag(B⊗C), trace-matched to G per matrix. D equals kron exactly when G = K. The damping mismatch disappears, and Lanczos spends its steps on G's departure from the Kronecker form, where the spectrum is clustered. It needs FP32 products.
- **Smoke tests** (`second_order_audit_20260926/gnrate_smoke/`, 2–3 steps from PD @9).
  - **Harness.** It reproduces the trainer's step-10 loss (7.17841) and gradient norm (9.707).
  - **β 0.9.** gnpd's first step beats kron's on held-out data (−0.085 vs −0.065). Its second step goes uphill (+0.082, against kron's −0.068), as in the floor steps. The gnpd step is nearly orthogonal to kron's (per-matrix cosine 0.03–0.13) and to Muon's (0.25).
  - **Transport with clipping is inconsistent.** The momentum holds clipped gradients (norms 8–10 clipped to 1), while G D is in raw units. The transport term came out 26× the momentum and made step 11 worse (+0.112). It is not used.
  - **Fresh gradient (β 0) is unstable for both maps at LR 0.028.**
    - kron: −0.160, +0.066, −0.063; top GN eigenvalue 614 → 965 → 2000.
    - gnpd: −0.017, −0.092, +0.080; top GN eigenvalue 614 → 2682 → 909.
  - **Reading.** At 16M the momentum is not only noise averaging. It damps the stiff-direction oscillation that sharpens the network without it (principle 1). The β 0 pair is dropped.
- **Next.** A one-step pre-flight at PD @9 and @46. It uses the harness's own code, 4 ranks and the clipped next momentum, and compares these directions: Muon; PD; kron; harness gnpd (BF16, Lanczos 64, set A); gnpd on a disjoint set B; gnpd FP32 / Lanczos 128; gnpd p ¼; gnpd at μ = 1e-4 ρ; nested gnpd (FP32).
  - Outputs: per-matrix cosines; one-step held-out loss changes at 0.5, 1 and 2× the harness step; the model-optimal multiple c* along each direction; and a Hutchinson estimate of mean eig(G) against μ.
  - Training arms follow from it.

**Pre-flight: at 16M PD states the exact-GN PD direction is not reproducible early and is not a better direction than its Kronecker control; the rate arms are not run (2026-09-28 10:13 CDT).** `gnrate_preflight.py` (4 ranks, the harness's own code) at PD α ½ @9 and @46, on the next clipped momentum. Harness step: LR 0.028 × per-matrix norms. Quality is the GN model's best one-step decrease along a direction, a²/2q (held-out slope a, GN curvature q). c* = −a/q is the model-optimal multiple of the harness step. Figure: `second_order_audit_20260926/figures_gnrate/preflight.png`; data: `gnrate/preflight_pd{9,46}.json`.

| Direction | Quality @9 | c* @9 | True Δ at 1× @9 | Quality @46 | c* @46 | True Δ at 1× / 2× @46 |
|---|---|---|---|---|---|---|
| Muon | 0.116 | 0.16 | **+2.03** | 0.023 | 0.08 | **+0.88** / +1.57 |
| PD α ½ | 0.136 | 2.1 | −0.106 | 0.034 | 0.81 | −0.027 / +0.028 |
| kron (TS ½/½) | **0.136** | 4.9 | −0.052 | **0.037** | 1.56 | −0.029 / −0.028 |
| GN-PD harness (BF16, k 64, set A) | 0.067 | 1.7 | −0.070 | 0.026 | 0.72 | −0.020 / +0.051 |
| GN-PD, disjoint set B | 0.070 | 1.75 | −0.079 | 0.024 | 0.73 | −0.020 / +0.052 |
| GN-PD FP32, k 128 | 0.071 | 1.6 | −0.077 | 0.028 | 0.73 | −0.023 / +0.044 |
| GN-PD p ¼ | 0.106 | 1.15 | −0.124 | 0.031 | 0.47 | −0.002 / +0.121 |
| GN-PD μ 1e-4 ρ | 0.058 | 2.1 | −0.052 | 0.023 | 0.98 | −0.022 / +0.015 |
| nested (Kronecker-whitened, FP32) | 0.002 | 3.4 | −0.001 | 0.026 | 4.9 | −0.010 / −0.017 |

- **The exact-GN direction is a property of the curvature sample, not of the curvature.** Mean per-matrix cosine between set A and the disjoint set B is 0.15 at @9 and 0.40 at @46. Between BF16/k 64 and FP32/k 128 on the same set it is 0.25 and 0.60. Variants on one set agree (0.83–0.96).
  - Different directions score alike: GN-PD's one-step decrease comes from suppressing the stiff part, not from a specific bulk structure.
- **The damping mismatch is real.** The mean eigenvalue of G is 5–6e-4, while μ = 1e-3 ρ ≈ 0.06 (ρ ≈ 58, 10⁵× the mean; top eigenvalue 615 @9, 142 @46). The symmetric GN-PD does not whiten the bulk at all.
- **The Kronecker map is the best direction at both states.** Its quality equals or exceeds PD's (0.136 vs 0.136; 0.037 vs 0.034). GN-PD reaches 0.5–0.75× of it (p ¼: 0.8×).
  - GN-PD's larger decrease at the harness step (−0.070 vs −0.052 @9) is step length: kron's step is 1/4.9 of its optimum, GN-PD's 1/1.7. A same-LR training comparison would have measured that.
- **The nested construction fails with μ̂ = 1e-3.** At @9 its slope is ~0: it concentrates the step where G is flatter than K predicts, doubly amplified.
- **After Kronecker whitening a stiff band remains.** The Lanczos from the whitened momentum finds dozens of Ĝ eigenvalues of 10³–10⁴ (top 15,600 @9, 9,000 @46; unit-mean diagonal blocks), and the whitened momentum's Rayleigh quotient is 4.7 and 30.
- **Co-adaptation is dramatic.** At PD's state Muon's direction has 100–150× the GN curvature of PD's (q 9.5 vs 0.06 @9; 7.0 vs 0.10 @46), and its harness step raises the held-out loss by 2.0 and 0.9.
- **Decision.** The GN-PD rate arms are not launched. Two of the review's failure conditions hold: the disjoint-sample cosine is barely above the gnpd–kron cosine at @9, and c* differs by 2–3× between arms.
  - A per-step exact-GN geometry estimated from 128 sequences has nothing to add over the Kronecker map at 16M oscillating states.
  - The Kronecker factors carry the estimable bulk structure: forward statistics and output gradients averaged over all tokens. The non-Kronecker remainder is a stiff band, where energy regulates sharpness (principle 1), plus sample noise in the flat part.
  - The per-step curvature gap beyond Kronecker is not where the large-batch gains live at this scale. This closes the question left open by principle 5 for 16M: the valley-floor prize at 1M did not transfer, and at oscillating states the Kronecker map wins.

**Decision argument: how does each optimizer's own step meet the curvature at its own state? A step-profile atlas across optimizers, batch sizes and training (2026-09-28 10:20 CDT).**
- **Limitation.** We know the ranking at 16M (S∘PD > TS ≈ PD > Muon) and that one-step scores of foreign directions are biased by co-adaptation (the pre-flight: Muon's step at PD's state has 100–150× PD's curvature). We do not know how the winners' own steps differ in their relation to the curvature each co-adapted landscape presents. The only unbiased per-state comparison is each optimizer's actual step at its own state. The kept checkpoints hold W_{s+1} at 16M (@9, 46, 83), 4M (@37, 183, 330) and 1M (@10 … 1300).
- **Hypotheses.**
  - H1, edge regulation: every optimizer's own step sits at the same place relative to its own curvature (c* ≈ ½–1). Optimizers then differ in slope per unit curvature, and in the sharpness they sustain.
  - H2, SOAP suppresses the oscillation: SOAP's per-entry normalization by temporal second moments suppresses the momentum's components that flip sign across steps (the stiff, edge-of-stability directions). S∘PD's step therefore puts less energy and curvature into the top GN eigenspace than the PD map of the same momentum. The gradient's sign flips along the top Ritz vectors between s and s+1 identify the oscillating directions.
- **Simplest competing explanation.** The winners' steps have the same stiff profile, and they differ only in slope along the bulk.
- **Measurement.** `step_profile_probe.py`, 1 GPU per state. G is the exact GN on 128 held-out sequences (FP32), with the top Ritz pairs from Lanczos 48. For the actual step, the momentum, the held-out gradient, and the Muon and PD maps of the momentum, it reports slope, curvature, c*, quality, the Rayleigh quotient over λ_max, and energy and curvature fractions in the top 1/4/16 Ritz vectors. It also reports gradient sign flips along the top Ritz vectors from s to s+1.
- **Predictions.**
  1. The actual step's c* lies in [0.3, 1.2] for every optimizer, batch and state.
  2. At 16M @46, S∘PD's actual step has a lower top-16 energy fraction than the PD map of its own momentum, and than PD's actual step at PD's state.
  3. Along the top Ritz vectors, the gradient flips sign between s and s+1 (ratio < 0) in most of the top 4, for every optimizer.
- **What changes the next decision.**
  - H2 holds: a cheap explicit version is the next training test. Candidates: an oscillation filter on the output side only, where SOAP_left keeps 68–79%; or PD with the momentum's temporal RMS normalization in the output eigenbasis.
  - H2 fails, and the stiff profiles are alike: SOAP's gain is in the bulk. Look at slope per unit curvature in the bulk.
- **Cost.** About 36 states × ~4 min of GPU time. No training.

**Step profiles: every optimizer runs at the edge of its own geometry; the preconditioned ones let the network sharpen 30–400× in directions they do not step in; at large batch the gradient is mostly a period-2 stiff oscillation (2026-09-28 10:38 CDT).** `step_profile_probe.py`, 34 states. Summary table: `second_order_audit_20260926/step_profile_summary.json`; figure: `figures_gnrate/step_profile.png`.

| Batch, state | Optimizer | Loss | λ_max of G | c* of own step | Own step's energy in top-16 GN vectors | Their share of its curvature | Gradient energy in top 16 | cos(g_s, g_{s+1}) |
|---|---|---|---|---|---|---|---|---|
| 16M @46 | Muon | 5.73 | 4.5 | 0.68 | 1.5e-3 | 0.47 | 0.29 | +0.14 |
| | PD | 5.55 | 146 | 0.81 | 2.0e-5 | 0.58 | 0.73 | −0.22 |
| | S_left∘PD | 5.50 | 261 | 0.81 | 1.0e-5 | 0.58 | 0.84 | +0.15 |
| | S_right∘PD | 5.51 | 324 | 0.80 | 1.0e-5 | 0.65 | 0.82 | −0.31 |
| | S∘PD | 5.45 | 463 | 0.79 | 6.1e-6 | 0.65 | 0.82 | −0.08 |
| | TS | 5.52 | 1029 | 0.80 | 1.7e-6 | 0.57 | 0.86 | −0.09 |
| 16M @83 | Muon / PD / S∘PD / TS | 5.01 / 4.79 / 4.69 / 4.76 | 4.7 / 154 / 490 / 1026 | 0.63 / 0.65 / 0.60 / 0.63 | 1.2e-3 / 2.1e-5 / 4.9e-6 / 2.7e-6 | 0.41 / 0.39 / 0.33 / 0.47 | 0.23 / 0.53 / 0.57 / 0.68 | +0.16 / −0.22 / −0.28 / −0.38 |
| 4M @183 | Muon / PD / S∘PD / TS | 4.36 / 4.21 / 4.15 / 4.18 | 7.6 / 185 / 1123 / 784 | 0.56 / 0.63 / 0.59 / 0.64 | 2.1e-3 / 1.3e-5 / 8.2e-7 / 1.3e-6 | 0.49 / 0.38 / 0.14 / 0.31 | 0.37 / 0.49 / 0.41 / 0.46 | +0.10 / +0.09 / +0.05 / +0.24 |
| 1M @1300 | Muon / PD α¼ / SOAP∘Muon / S∘PD α¼ | 3.83 / 3.82 / 3.81 / 3.81 | 15.7 / 190 / 128 / 911 | 0.52 / 0.66 / 0.65 / 0.60 | 2.9e-4 / 7.9e-6 / 3.6e-5 / 1.3e-6 | 0.30 / 0.25 / 0.31 / 0.31 | 0.19 / 0.33 / 0.27 / 0.28 | +0.81 / +0.81 / +0.77 / +0.83 |

- **Prediction 1 holds: every optimizer runs at the edge of its own geometry.** Its own step's c* lies in 0.45–0.94 at all 34 states, converging to 0.55–0.65 late at every batch size. The tuned LR puts each optimizer at the same relative place, about 1.5–1.8× the GN-model optimum along its own step (principle 4, now measured across optimizers).
- **The network sharpens where the optimizer does not step.**
  - λ_max at each optimizer's own state falls in bands: Muon 5–25, PD and SOAP∘Muon 100–600, S∘PD and TS 500–2250, at every batch size.
  - The energy of the step in the top-16 GN vectors falls in mirrored bands: Muon ~1e-3, PD ~1e-5, S∘PD and TS ~1e-6.
  - The preconditioners decouple the step from the stiff directions, and the landscape then sharpens there without limit. Even so, those directions still carry 14–65% of the step's curvature.
  - Raw sharpness is therefore not a stability quantity for preconditioned methods. It describes the landscape each optimizer carves.
- **Prediction 3 holds at large batch only: a period-2 oscillation.** At 16M and 4M the gradient's component along the top Ritz vectors flips sign between s and s+1 (median ratio −0.3 to −2.4). Late at 1M it does not (ratios +0.3 to +1.4).
  - At 16M the gradient's energy at the preconditioned states is 53–86% in the top-16 directions, and consecutive gradients are anti-correlated (cos −0.08 to −0.38). The bulk stays aligned (+0.18 to +0.63).
  - The stale momentum M_s even points uphill at W_s (PD @46: slope +1.04 along −M). The step taken from M_{s+1} still descends (c* 0.81).
- **Prediction 2 and H2 fail as a mechanism.** S∘PD's step has 3–25× less stiff energy than PD's, but TS has as little or less at 16M and gains far less (4.668 vs 4.574). Stiff suppression is shared by all whitening maps, and it is not SOAP's advantage.
- **One-step progress at own states does not rank the optimizers.**
  - The actual step's held-out GN-model decrease a + q/2 at 16M @83: Muon −0.020, PD −0.023, S∘PD −0.016, TS −0.016. Its quality a²/2q: 0.029, 0.032, 0.028, 0.024.
  - Interpolating PD to S∘PD's loss does not reverse this.
  - The step-equivalent speedups (S∘PD 1.3× over Muon at matched loss) therefore accumulate from what the one-step quadratic model does not see: the drift along the valley floor over many oscillating steps, not the per-step decrease.
- **Reading.** At large batch, the gradient is mostly the edge oscillation of stiff directions. Useful progress is the slow bulk component. Every method stays at its own edge, so per-step one-step measures cannot show the gap. What differs is how cleanly each method extracts and follows the persistent bulk.
  - Spatially, whitening decouples the stiff directions.
  - Temporally, momentum filters the oscillation at the cost of ~β/(1−β) steps of lag. The 4M momentum sweep has an interior optimum at β 0.9: 0.81 +0.022, 0.95 +0.031, and Nesterov, which adds weight to the fresh gradient, +0.071.

**Decision argument: a two-tap pre-filter for the momentum at large batch (2026-09-28 10:38 CDT).**
- **Observable limitation.** At 16M and 4M the gradient is mostly a period-2 oscillation of the stiff directions. Its components along the top GN vectors flip sign every step; 53–86% of its energy sits there at the preconditioned states; consecutive gradients are anti-correlated while the bulk persists (step profiles, 10:38).
  - The momentum's EMA passes a period-2 component with gain (1−β)/(1+β), but it lags the bulk by ~β/(1−β) steps: 9 steps at β 0.9, 10% of a 16M run.
  - The 4M sweep has an interior optimum at β 0.9: 0.81 is +0.022, 0.95 +0.031, and Nesterov, which adds weight to the fresh gradient, +0.071.
- **Mechanism.** If oscillation leakage is what forbids a fresher momentum at large batch, then removing the period-2 component before the average makes a lower β usable, and the fresher momentum follows the bulk (the valley floor) better.
  - The filter is M ← βM + (g_t + g_{t−1})/2. Its response is zero at the Nyquist frequency, near 1 for slow components, and it adds half a step of lag.
- **Simplest competing explanations.**
  - β's optimum is set by batch noise or by lag alone.
  - The oscillation in clipped gradients is not clean period-2.
  - The whitening maps already remove it (their steps hold ~1e-6 of their energy in the top-16 directions), so the filter changes nothing for them.
- **Comparison.** 16M, 92 steps, seed 260925, hardware matched to the references:
  - S∘PD α ½ @0.028 on Ada (reference β 0.9: 4.5742): (a) two-tap β 0.9, (b) two-tap β 0.8, (c) plain β 0.8.
  - Muon @0.02 on L40S (reference 4.9104): (d) two-tap β 0.9. Muon's step carries ~100× more stiff energy, so the filter should matter more there.
- **Predictions.**
  1. (b) beats the reference by ≥ 0.02.
  2. (c) is worse than the reference.
  3. (a) is at or below the reference.
  4. Muon gains more from the filter than S∘PD: (d) − 4.9104 < (a) − 4.5742.
- **Outcomes that change the next decision.**
  - (b) wins clearly: the momentum's role at large batch is oscillation filtering. Next, test the batch dependence (4M, 1M) and a filter confined to the stiff subspace.
  - (a) and (b) are neutral or worse: the oscillation is not the binding constraint on the momentum.
  - (d) gains but (a) does not: the whitening's spatial suppression already does the filtering.
- **Cost.** A small trainer option (`muon_prefilter`, default "none", with unit tests), 4 runs × 30 min on the two nodes. No data.

**Correction to the reading of normalized GN's failure (principle 1), from the step profiles (2026-09-28 10:48 CDT).**
- The 09-27 reading: GN's direction at a fixed step norm removes the stiff-direction energy that regulates sharpness at the edge, so the network sharpens and training falls behind.
- **The step profiles contradict the causal part.** The winning maps also remove stiff-direction energy: PD, TS and S∘PD put 1e-5–1e-6 of their step's energy in the top-16 GN vectors, against Muon's 1e-3. Their states are 30–400× sharper than Muon's (λ_max 150–2250 vs 5–15), and they train better. Sharpening where an optimizer does not step is normal, not a failure.
- **A better-supported candidate** is that the per-step GN direction is dominated by its curvature sample. At 16M, disjoint 128-sequence samples give GN-PD directions with cosine 0.15–0.40, and the normalized-GN run used only 64 sequences. Its rising Ritz value is then a symptom: a sample-specific step leaves the stiff band unregulated in ways no single sample sees.
  - Not yet tested at 1M: the disjoint-sample cosine of the normalized-GN direction itself.
- **Principle 1 therefore reads:** the realized gains come from the geometry of a normalized update. Newton-type directions lose through their step rule (greedy searches) and, per step, through estimation noise in their curvature, not through sharpening as such.

**Decision argument: PD's principle for the unembedding, Adam in input-whitened coordinates (2026-09-28 11:16 CDT).**
- **Observable.** Every method in this study, and in Track 3, trains the head with AdamW (26M parameters, as many as the body). The head's input, the final normalized hidden state, is as anisotropic as the body inputs PD whitens (`head_input_probe.py`, 16M @46 and 1M @1300):
  - top eigenvalue 80–120× the mean;
  - participation ratio 0.026–0.063, i.e. an effective rank of 13–32 of 512;
  - top-8 share 33–52%.
  - For comparison, block08.q is 98–108× with PR 0.03–0.04.
  - Adam's per-coordinate normalization cannot decorrelate inputs. Track 3 showed that diagonal input whitening keeps only 40–50% of PD's gain on the body.
- **Mechanism.** The head's GN is ≈ (softmax Fisher on the vocabulary) ⊗ C_h.
  - Adam's per-entry adaptivity suits the output side: rare tokens need their own step sizes.
  - On the input side it assumes decorrelated coordinates, the head's analog of Muon's isotropic-input assumption.
  - Adam in the coordinates W̃ = W C_h^α is steepest descent under ℓ∞ of ΔW C^α (at high SNR). This gives ΔW = −Adam(G R) R with R = (C_h/mean + 1e-3)^-α: the exact analog of PD's polar(G R) R.
  - The first moment stays in weight space (coordinate-free, as PD's momentum). The second moment is the EMA of (G R)², as in SOAP's gradient mode. The update is norm-matched to its unwhitened version, so only the direction changes.
- **Simplest competing explanations.**
  - The head is not a bottleneck; the body limits progress.
  - The anisotropy sits in directions the loss does not need, such as the mean or massive-activation directions.
  - Adam's second moment already adapts enough.
- **Comparison.** 16M, seed 260925, 92 steps, S∘PD α ½ @0.028 on Ada (reference 4.5742), with the head in whitened coordinates at α ½ and at α ¼. A PD α ½ arm on L40S (reference 4.6711) repeats α ½ if the S∘PD arms gain.
- **Predictions.**
  1. α ½ improves S∘PD by 0.02–0.08, with the gain forming early (the lead at 16M forms in steps 7–15).
  2. α ¼ gains less than α ½.
- **Outcomes that change the next decision.**
  - A gain: the input-whitening principle is general to every linear map. The next steps are the batch dependence at 4M and 1M, and a two-sided (vocabulary-frequency) output factor.
  - No gain: the head is not the bottleneck at this scale. Record that and return to the trajectory questions.
- **Cost.** New options, off by default: `track_head_cov` (model: the head becomes a `StatLinear` with covariance tracking; weights and init hash unchanged) and `head_whitening_alpha` (optimizer). Unit tests, then 2–3 runs × 30 min.

**Two-tap momentum pre-filter, interim: the period-2 gradient component is the stabilizing feedback, not leakage (2026-09-28 11:31 CDT).** `soaudit_prefilter16m_20260928`, 16M, seed 260925. Final validation against the same-hardware reference:

| Arm | Final | Reference | Change | Train-loss gap along the run (steps 10 / 46 / 83) |
|---|---|---|---|---|
| Muon @0.02, two-tap β 0.9 (L40S) | 5.5101 | 4.9104 | **+0.600** | +0.28 / +0.40 / +0.61 |
| PD α ½ @0.028, two-tap β 0.9 (L40S) | 4.8657 | 4.6711 | **+0.195** | −0.03 / +0.08 / +0.18 |
| S∘PD α ½ @0.028, two-tap β 0.8 (Ada) | 4.5655 | 4.5742 (β 0.9) | −0.009 | +0.02 / −0.04 / +0.01 |

- **Predictions 1 and 4 fail, the latter in the opposite direction.**
  - The filter does not make a fresher momentum usable.
  - It hurts in proportion to how much the optimizer steps in the stiff directions: Muon's step holds ~1e-3 of its energy there (+0.60), PD's ~2e-5 (+0.19), and S∘PD's ~6e-6 (neutral).
  - Muon's pre-clip gradient norms stay 2–3× above its reference's all run (step 46: 0.84 vs 0.29). The stiff directions are no longer damped.
- **Reading.** At the edge, a stiff direction's gradient component flips sign each step, and that flip is the restoring force. A two-tap average cancels exactly that force, so in any direction the optimizer steps in, the displacement is no longer corrected and it drifts.
  - The oscillation is the working mechanism of stability, not noise to filter. This is what the momentum's EMA keeps, attenuated by (1−β)/(1+β).
  - The harm grows through the run, and the cooldown does not recover it: this is floor damage, not stored excess.
- **Pending:** S∘PD two-tap β 0.9 (the like-for-like test) and plain S∘PD β 0.8 (β alone).

**Decision argument: the momentum at large batch, a β sweep at 16M across methods (2026-09-28 12:10 CDT).**
- **Observable.** In the two-tap cohort, plain S∘PD at β 0.8 runs ahead of its β 0.9 reference by −0.05 at step 30, −0.107 at 46 and −0.086 at 60 (final pending). That equals S∘PD's whole gain over PD (−0.097).
  - β 0.9 was carried over from the 4M sweep (4M optimum ~0.9; 0.81 +0.022 for Muon) and never tuned at 16M.
  - The two-tap results say the oscillating component is feedback that the momentum must pass.
- **Mechanism.** At batch B the EMA does three things:
  1. averages batch noise (weak at 16M: per-entry SNR is high);
  2. passes the period-2 stabilizing feedback with gain (1−β)/(1+β) (0.05 at β 0.9, 0.11 at 0.8, 0.18 at 0.7);
  3. lags the persistent bulk by ~β/(1−β) steps (9, 4 and 2.3), and a 92-step run cannot afford 9.
  - So the optimal β should fall with batch size: fresher momentum and more feedback, until the oscillation is no longer damped (β 0 was unstable in the harness smoke tests).
- **Simplest competing explanation.** β 0.8 helps only S∘PD, through its SOAP statistics (whose time constant β2 0.9 interacts with β), not through the batch-size mechanism.
- **Comparison.** 16M, seed 260925, 92 steps, hardware-matched references.
  - g20 (Ada): S∘PD α ½ @0.028 at β 0.7, then β 0.6 (β 0.9: 4.5742; β 0.8 pending).
  - priv-g14 (L40S): PD α ½ @0.028 at β 0.8 (β 0.9: 4.6711), then Muon @0.02 at β 0.8 (β 0.9: 4.9104).
- **Predictions.**
  1. S∘PD β 0.8 ends ≥ 0.02 below β 0.9.
  2. The S∘PD optimum lies in 0.6–0.8 and is flat within ±0.02 there.
  3. PD β 0.8 gains too (≥ 0.02).
  4. Muon β 0.8 is within ±0.02 of β 0.9: Muon steps in the stiff directions and needs the damping of a longer average (at 4M, 0.81 was worse).
- **Outcomes that change the next decision.**
  - A general gain for the whitening methods: the momentum's optimal time constant at large batch is a principle to state and use, measured in steps, not tokens.
  - Only S∘PD gains: its SOAP statistics' time constant is the lever; test SOAP's β2.
  - Nothing beyond S∘PD β 0.8: record the single-method result.
- **Cost.** 4 runs × 30 min on the two nodes.

**Where along the curvature spectrum the one-step gap lives: the flattest band is under-stepped, the rest is at the edge (2026-09-28 12:13 CDT).** `step_spectrum_probe.py`, 34 states. A Lanczos run started at each optimizer's own step, its held-out gradient and its momentum. The step lies in its own Krylov space, so its energy, slope g·D and curvature D·G·D split exactly over Ritz nodes. Figure: `figures_gnrate/step_spectrum.png`; bands: `step_spectrum_summary.json`.
- **Per-band c* = −slope/curvature of the own step** (bands of curvature/λ_max):

| State | Optimizer | < 1e-4 | 1e-4–1e-3 | 1e-3–1e-2 | 1e-2–1e-1 | 1e-1–1 | Slope share < 1e-4 |
|---|---|---|---|---|---|---|---|
| 16M @9 | Muon / PD / S∘PD / TS | 9.3 / 5.2 / 5.0 / 6.4 | 4.1 / 4.0 / 3.3 / 5.4 | 2.3 / 2.0 / 2.1 / 3.3 | 0.6 / 1.1 / 0.9 / 2.1 | 0.4 / 0.8 / 0.3 / 0.8 | 0.01 / 0.02 / 0.05 / 0.02 |
| 16M @46 | Muon / PD / S∘PD / TS | 29 / 7.9 / 4.6 / 6.9 | 4.9 / 1.7 / 0.7 / 0.8 | 1.7 / 0.7 / 0.0 / 0.3 | 0.8 / 0.6 / 0.2 / 0.1 | 0.6 / 0.7 / 0.8 / 0.9 | 0.07 / 0.17 / 0.18 / 0.21 |
| 16M @83 | Muon / PD / S∘PD / TS | 21 / 4.6 / 2.9 / 4.6 | — / 0.9 / 0.7 / 0.8 | 1.3 / 0.7 / 0.6 / 0.6 | 0.7 / 0.6 / 0.6 / 0.6 | 0.5 / 0.5 / 0.5 / 0.5 | 0.08 / 0.15 / 0.18 / 0.20 |
| 4M @183 | Muon / PD / S∘PD / TS | 10 / 2.1 / 1.8 / 2.1 | 1.3 / 0.5 / 0.6 / 0.6 | 0.6 / 0.5 / 0.6 / 0.4 | 0.5 / 0.5 / 0.5 / 0.4 | 0.5 / 0.6 / 0.4 / 0.6 | 0.04 / 0.12 / 0.23 / 0.23 |
| 1M @500 | Muon / PD / S∘PD / SOAP∘M | 4.8 / 1.3 / 1.4 / 2.6 | 0.8 / 0.6 / 0.5 / 0.5 | 0.5 / 0.5 / 0.3 / 0.5 | 0.4 / 0.3 / 0.4 / 0.6 | 0.5 / 0.5 / 0.6 / 0.7 | 0.06 / 0.10 / 0.22 / 0.12 |

- **The edge holds band by band from 1e-3·λ_max up.** c* is 0.3–0.9 there for every optimizer after the first steps, so the stiff and middle spectrum is stepped as far as stability allows.
- **The flattest band (< 1e-4·λ_max) is under-stepped.** It holds 95–100% of every step's energy, and the GN model wants it 1.3–29× longer: Muon 4.5–29, PD 1.3–12, TS 2–7, S∘PD 1.4–5.
  - The gap is largest early (16M @9, 4M @37, 1M @100) and shrinks through training.
  - The better optimizers shrink it and draw more of their slope from this band: S∘PD 18–25% vs Muon 4–12% at the mid and late states.
- **Energy, slope and cost.** The step's energy is almost all flat, while most of its slope and its curvature come from the stiffer bands. The stiff-band slope is the edge oscillation: released one step, reversed the next. The flat-band slope is the persistent part. That is why one-step totals do not rank the optimizers, while the flat band's share does.
- **Consequence.** Measured in the exact GN at each optimizer's own state, the per-step gap to a second-order step is the flattest band's under-stepping, not the geometry of the stiff part.
  - A normalized update's length is set by the stiff bands' edge, while its energy lives in the flat band.
  - A method that lengthens the flat-band component without touching the stiff components would close it. Whitening and SOAP already move this way.
- **Caveats.** The per-band split is exact in the GN model on 128 held-out sequences. The flat band's curvature per direction is small and sample-dependent. One seed per state.

**Two-tap cohort complete, and the momentum is a first-order lever at 16M: S∘PD at β 0.8 ends at 4.4724, −0.102 against β 0.9 (2026-09-28 12:18 CDT).** `soaudit_prefilter16m_20260928`, 16M, seed 260925, final validation:

| Arm | Final | Reference | Change |
|---|---|---|---|
| Muon @0.02, two-tap β 0.9 (L40S) | 5.5101 | 4.9104 | +0.600 |
| PD α ½ @0.028, two-tap β 0.9 (L40S) | 4.8657 | 4.6711 | +0.195 |
| S∘PD α ½ @0.028, two-tap β 0.9 (Ada) | 4.6675 | 4.5742 | +0.093 |
| S∘PD, two-tap β 0.8 (Ada) | 4.5655 | 4.5742 | −0.009 |
| **S∘PD, plain β 0.8 (Ada)** | **4.4724** | 4.5742 | **−0.102** |

- **The two-tap filter hurts every method at β 0.9, ranked by stiff stepping:** Muon > PD > S∘PD, as the step profiles order their stiff energy. Against plain β 0.8, the filter costs S∘PD +0.093. Removing the period-2 feedback is harmful wherever the optimizer steps (11:31 reading confirmed).
- **Plain β 0.8 is the result.** −0.03 at step 10, −0.05 at 30, −0.11 at 46, −0.09 at 60 to 83, and −0.10 at the end. The best 16M result so far, ahead of gradient-mode S∘PD (4.5686) by 0.096.
  - β 0.9 was inherited from the 4M sweep and never tuned at 16M.
  - At 16M its lag, ~9 steps, is a tenth of the run, while batch noise is small.
- **Fairness.** Muon's β has not been tuned at 16M either. The β sweep (`soaudit_mom16m_20260928`: S∘PD β 0.7 and 0.6, PD β 0.8, Muon β 0.8) decides how much of this is an S∘PD gain and how much is a 16M momentum effect shared by all.
- **Predictions 1 and 3 of the two-tap argument fail. The momentum argument's prediction 1 holds (≥ 0.02).**

**Whitened head, PD at 16M: α ½ loses, α ¼ gains −0.040 after a large early deficit (2026-09-28 12:25 CDT).** `soaudit_headwhite16m_20260928` (L40S; reference PD α ½ @0.028 β 0.9, 4.6711). Train-loss gap to the reference:

| Head whitening | Step 5 | 10 | 30 | 46 | 60 | 75 | 83 | 92 | Final validation |
|---|---|---|---|---|---|---|---|---|---|
| α ½ | +0.93 | +0.93 | +0.65 | +0.57 | +0.51 | +0.37 | +0.32 | +0.29 | 4.9656 (+0.295) |
| α ¼ | +0.72 | +0.61 | +0.20 | +0.08 | +0.01 | −0.00 | −0.03 | −0.04 | **4.6308 (−0.040)** |

- **The early deficit** is the phase in which the head learns token statistics. The head input's dominant direction is its mean (the output-bias channel; top eigenvalue 80–120× the mean eigenvalue). Whitening suppresses the steps along it, by (λ_top/mean)^−α ≈ 10× at α ½ and 3× at α ¼.
- **Later, whitening helps.** α ¼ crosses the reference near step 70 and is still gaining at the end. The head does benefit from input whitening once the early phase is past, which supports the original premise.
- **Principled fix, next.** Whiten the head's *centered* covariance, as `data_norm_center` does for the body, so the mean direction is not singled out and keeps its Adam step. A CPU probe is measuring how much anisotropy remains after centering.
- **Prediction for centered whitening.** The early deficit mostly disappears, and the late gain holds or grows: α ½ centered ≤ 4.64 on PD.

**Why β 0.8 helps S∘PD at 16M: the stale momentum pays middle-band curvature for no slope (2026-09-28 12:37 CDT).** `step_spectrum_probe.py` on the β 0.8 run's kept states against β 0.9's (same seed and hardware):

| State | β | Own-step slope | Curvature | c* | Quality a²/2q | Band c* (< 1e-4 … 1e-1–1) | Slope share by band |
|---|---|---|---|---|---|---|---|
| @9 | 0.9 / 0.8 | −0.150 / −0.165 | 0.342 / 0.408 | 0.44 / 0.40 | 0.033 / 0.033 | 5.0 3.3 2.1 0.9 0.3 / 5.5 2.6 1.8 0.7 0.3 | same |
| @46 | 0.9 / 0.8 | −0.081 / −0.129 | 0.109 / 0.186 | 0.74 / 0.70 | 0.030 / **0.045** | 4.6 0.7 **0.02 0.16** 0.8 / 4.4 0.8 **0.65 0.45** 0.6 | middle 0.03 / 0.21 |
| @83 | 0.9 / 0.8 | −0.094 / −0.137 | 0.159 / 0.230 | 0.59 / 0.59 | 0.028 / **0.040** | 2.9 0.7 0.6 0.6 0.5 / 2.6 0.6 0.5 0.5 0.5 | similar |

- At the same edge (c* 0.70–0.74 and 0.59), the fresher momentum's step gains 45–59% more one-step slope: its quality rises 45–49%.
- At @46 the β 0.9 step spends curvature in the middle bands (1e-3–1e-1·λ_max) with almost no slope there (band c* 0.02 and 0.16). The 9-step-old momentum's components in those bands no longer match the gradient. At β 0.8 the same bands are productive (c* 0.65 and 0.45, 21% of the slope).
- The flattest band stays under-stepped (c* 2.6–5.5) at both β.
- Within one geometry, one-step quality does track the momentum's benefit, unlike across optimizers, whose states are co-adapted differently. *[Withdrawn 14:13: Muon's β 0.8 step gains the most one-step quality but the least final loss.]*

**β sweep at 16M, interim: the momentum lag costs every whitening method about 0.1 (2026-09-28 12:51 CDT).** `soaudit_mom16m_20260928`, 16M, seed 260925, final validation:

| Method | β 0.9 | β 0.8 | β 0.7 | β 0.6 |
|---|---|---|---|---|
| S∘PD α ½ @0.028 (Ada) | 4.5742 | 4.4724 | **4.4691** | running |
| PD α ½ @0.028 (L40S) | 4.6711 | **4.5475** (−0.124) | — | — |
| Muon @0.02 (L40S) | 4.9104 | running | — | — |

- **Predictions 1–3 hold.** S∘PD's optimum is flat across 0.7–0.8 (−0.003 between them), and PD gains −0.124, as much as S∘PD.
  - The gap S∘PD − PD narrows only slightly, from −0.097 at β 0.9 to −0.075 at β 0.8.
  - PD at β 0.8 now beats S∘PD at β 0.9.
- **PD's gain builds mid-run:** −0.01 at step 10, −0.06 at 30, −0.14 at 46, −0.10 to −0.12 after.
- **Reading.** At 16M the momentum's lag is a first-order cost for the whitening methods, independent of SOAP. The inherited β 0.9 averaged over ~10% of the run. The Muon arm decides whether the effect is universal or specific to methods that barely step in stiff directions.

**Decision argument: stronger input whitening, or a longer step, to use the under-stepped flat band (2026-09-28 12:53 CDT).**
- **Observable.** The flattest band (< 1e-4·λ_max) holds 95–100% of every step's energy and is under-stepped: its band c* is 2.6–5.5 for S∘PD at 16M, at β 0.9 and 0.8 alike. Every stiffer band sits at the edge. A normalized step's length is capped by the stiffer bands, where its small energy meets large curvature.
- **Mechanism.** Stronger input whitening (α ¾) shrinks the step's component along high-variance input directions, which carry the stiff curvature, relative to the flat band. It moves the balance toward the flat band at the same step norm, and the one-step best α rose from ½ to ¾ with 16M gradients. The fresher momentum (β 0.8) also passes more of the stabilizing feedback ((1−β)/(1+β) 0.11 vs 0.05), which may tolerate a longer step: at β 0.9, LR 0.04 lost 0.010 to 0.028.
- **Simplest competing explanation.** The flat band's under-stepping is a one-step model artifact that the trajectory does not reward (the sample-noise caveat). Neither a longer step nor stronger whitening then helps.
- **Comparison.** 16M, seed 260925, β 0.8 (Ada; reference S∘PD α ½ @0.028 β 0.8 4.4724): S∘PD α ¾ @0.028, and S∘PD α ½ @0.04.
- **Predictions.**
  1. α ¾ gains 0.01–0.03 over α ½.
  2. LR 0.04 is within ±0.01 at β 0.8, a flatter LR optimum than at β 0.9.
- **What changes the next decision.** A gain from α ¾ supports whitening strength as a batch-size lever: test α 1 and the per-band c* of the α ¾ step. Neither gaining means the flat band's one-step under-stepping is not a trajectory lever at this scale; record it.
- **Cost.** 2 runs × 30 min on g20, after the seed replicate.

**β sweep at 16M complete: whitening lets the momentum be fresher. PD and S∘PD gain 0.10–0.12, Muon 0.02 (2026-09-28 13:23 CDT).** `soaudit_mom16m_20260928` (+ the prefilter cohort's β 0.8 arm), 16M, seed 260925, final validation:

| Method | β 0.9 | β 0.8 | β 0.7 | β 0.6 |
|---|---|---|---|---|
| S∘PD α ½ @0.028 (Ada) | 4.5742 | 4.4724 | **4.4691** | 4.4891 |
| PD α ½ @0.028 (L40S) | 4.6711 | **4.5475** | — | — |
| Muon @0.02 (L40S) | 4.9104 | **4.8901** | — | — |

- **Predictions.** Prediction 2 holds: S∘PD's optimum lies at 0.7–0.8, flat within 0.003, with β 0.6 +0.020. Prediction 3 holds: PD gains −0.124. Prediction 4 holds at its boundary: Muon gains −0.020, a fifth to a sixth of the whitening methods' gain.
- **Step-equivalent speedup over the best-tuned Muon** (β 0.8, 4.8901) at matched loss:

| Arm | At 25% / 50% / 75% / 88% of training |
|---|---|
| S∘PD β 0.7 | 1.30 / 1.40 / 1.37 / 1.41× |
| S∘PD β 0.8 | 1.29 / 1.36 / 1.36 / 1.40× |
| PD β 0.8 | 1.17 / 1.27 / 1.27 / 1.32× |
| S∘PD β 0.9 | 1.23 / 1.29 / 1.26 / 1.31× |

- **Principle: whitening and a fresh momentum compound at large batch.**
  - The momentum averages batch noise, which is weak at 16M, and damps the stiff directions' period-2 oscillation. It pays with a lag of ~β/(1−β) steps, which at β 0.9 is 10% of a 16M run.
  - Muon steps in the stiff directions and needs the damping, so it cannot shorten the average.
  - The whitening maps keep their steps out of those directions (step profiles: 1e-5–1e-6 of their energy). They can use the fresher average, and it makes their middle curvature bands productive (per-band spectra, 12:37).
  - The inherited β 0.9 hid about 0.1 of the whitening methods' 16M advantage.
- **Caveats.** One seed; the seed-260926 replicate of S∘PD β 0.8 is queued. LR was not re-tuned at β 0.8; the 0.04 arm is queued. At 4M the earlier sweep found β 0.9 best for Muon; the whitening methods were not swept there.


### Independent observer: artifact connections and body–auxiliary premise check (2026-09-28 13:24 CDT)

The user asked a separate observer to study saved evidence and anomalies broadly,
without competing for GPUs used by the main training program. The observer's
notes and new outputs live in `logs/observer_20260928/`. This does not change the
main program's experiment queue or historical conclusions.

- **Missing premise.** The step-profile probe describes consecutive gradients
  but updates only hidden matrices; the river/hill probe loads the whole next
  state. At the same 4M Muon beta .9 state @183 they report gradient cosines
  +0.095 and -0.526. Data source, sample size and precision also differ.
- **Possible connection.** Body–auxiliary co-adaptation may matter to the
  oscillation and curvature story, linking the head intervention to body
  dynamics. Alternatively, shared-sample gradient noise, precision, or the
  different held-out data could explain the discrepancy entirely.
- **Discriminating comparison.** On the actual saved @183/@184 weights, evaluate
  four corners on identical held-out sequences in FP32: base, body update only,
  auxiliary update only, full update. Record per-sequence losses, interaction
  residual, body-gradient inner products, norms, cosines and cross-sequence
  estimates that remove the same-sequence contribution. Do not interpret these
  as population correlations when uncertainty is large.
- **Decision rule.** A material change between body-only and full gradients or
  losses makes auxiliary coupling a missing measurement; little change directs
  attention to sample size/noise. Neither outcome by itself licenses a training
  intervention or a claim about the causal source of optimizer gains.
- **Bounded cost.** CPU only, two threads, one checkpoint pair, 8-sequence pilot
  then up to 32 sequences if the projected total fits 15 minutes. No optimizer
  step, checkpoint mutation, or GPU allocation. Existing saved JSON/algebra
  checks of mean geometry, Lanczos resolution and temporal-filter stability
  also run on CPU. A second checkpoint requires interpreting this result first.
- **Independent peer discussion.** A fresh observer peer supported this premise
  check, emphasizing cancellation-safe reporting, full auxiliary scope, actual
  saved weights, and no mechanism claim from an eight-sequence sample. Details
  are kept in the observer notebook.

**Correction to the head-whitening design: the norm matching confounds the early phase (2026-09-28 13:27 CDT).**
- **The problem.** The whitened-head step was rescaled so that |dW| equals the whitened-coordinate Adam step's norm |u|. Whitening amplifies low-variance input directions by up to (1e-3)^−α (31.6 at α ½), so |u R| ≫ |u|. The matching then shrinks the step, and the head's function-space step, by |u|/|u R|: a much smaller effective head LR.
  - This is not PD's analog. Reparametrized Adam in W̃ = W C^α takes dW = −lr u R unscaled, and its function-space step is comparable to plain Adam's.
- **Two causes of the early deficits, now separable:**
  - suppression of the input's mean, the output-bias channel, which centering removes: the centered, matched α ½ arm has +0.31 at step 10 against +0.93 uncentered;
  - the norm-matching shrink, which the unmatched design removes.
- The α ¼ result (−0.040) came despite both handicaps.
- **Replacement.** `head_whitening_norm: none` (unit-tested), centered, at α ½ and α ¼ on PD (`soaudit_headcenter2_16m_20260928`, priv-g14). The matched centered α ¼ arm was stopped before it started; the matched centered α ½ arm finishes as a record.
- **Prediction.** Early deficit under +0.1 at step 10; final ≤ 4.64 at α ½.

**Independent review of today's 10:20–13:27 entries: corrections accepted (2026-09-28 13:50 CDT).** Read-only reviewer; brief in the session.
- **Per-band spectrum.**
  - The algebra is exact, but the resolution is not. The "flattest band" (< 1e-4·λ_max) is a single Ritz node at every 16M state, the 1e-4–1e-3 band has 0–2 nodes, and 30–36 of the 48 nodes sit in the top decade (< 1% of the energy).
  - The "flat-band c*" is therefore the c* of the residual left after the stiff Krylov directions are removed. For any normalized step at the edge, ill-conditioning forces it above 1.
  - **Withdrawn:** "the flattest band is under-stepped 1.3–29×" and "the per-step gap lives in the flattest band". Also "at the edge in every band ≥ 1e-3": it fails for Muon (1.7 and 1.3) and S∘PD @46 (0.02 and 0.16).
  - The basis and the residual's curvature come from the same 128 sequences, which biases c* up; it should be cross-fitted. The slope has no error bars.
  - "Stale momentum wastes middle-band curvature" rests on @46 alone. At @83, 67% of β 0.8's extra slope is in the top band, where a fresher average mechanically passes more of the period-2 component.
  - What survives: the ordering of the top-band-removed residual's slope share (Muon lowest) and the step-profile facts (edge c* of the whole step, sharpness bands, stiff energy).
- **Momentum result.** The difference between methods is probably real (PD −0.124, S∘PD −0.102 against a 0.02 seed spread), but the mechanism is not shown.
  - **Main confound: clipping.** PD and S∘PD clip on 87–90 of 92 steps; Muon clips only in its first 18–25. Their momentum averages normalized gradients and Muon's does not.
  - Muon's −0.020 comes entirely in the cooldown.
  - The 13:23 caveat was wrong: PD α ¼ *was* swept at 4M (`soaudit_mom4m2`), where β 0.81 was +0.012 worse than 0.9. So the effect may be specific to short runs.
  - Muon's LR curve is sharp, and its β arm was not LR-re-tuned.
  - Running: Muon @0.02 with clip 0.1 at β 0.9 and 0.8 (`soaudit_clip16m_20260928`, priv-g14), and S∘PD β 0.8 at twice the horizon (`soaudit_horizon16m_20260928`, g20).
- **Head whitening was a repair cascade.** There were three designs in two hours, each queued before the previous was read.
  - The centered, matched α ½ arm finished at 4.8880 (+0.217; predicted ≤ 4.64).
  - The 13:27 premise was wrong: PD itself is norm-matched in weight space (√min(m,n)), with a re-tuned LR.
  - The missing control is plain Adam on the head at a head-only LR of ×⅓ and ×3. The early-deficit-then-recovery pattern is what a smaller head LR produces, so α ¼'s −0.040 may be an LR effect.
  - The unmatched cohort was stopped at step 17 and the line paused.
- **Two-tap mechanism, sharpened.** For linear heavy ball with input (g_t + g_{t−1})/2, stability requires ηλ < 2(1−β) = 0.2 at β 0.9, against 3.8 without the filter. The filter delays the restoring force, and the unstable mode has a period of 5–20 steps. The harm was predictable from linear stability, and its ranking by stiff stepping follows.
- **The α ¾ arm is dropped.** Its motivation, flat-band under-stepping, is the withdrawn claim.


### Tiny surrogate qualification: optimizer geometry, batch and momentum (2026-09-28 13:55 CDT)

**User direction.** Establish a small, standalone tiny-GPT-like setup before further full-pipeline work. Use only cluster GPUs with strictly less than 48 GB VRAM. Target the performance ordering Adam < Muon < PD-Muon < TS-Muon < SOAP-Muon in PD coordinates, increasing geometry benefit with batch size, and shorter useful momentum at larger batch. The eventual motivation is substantial simplification/memory or convergence improvements, including SOAP in TS coordinates; those are not substitutes for qualifying the surrogate first. Existing large-model and observer work is separate and is neither launched nor modified by this branch.

**Missing premise.** The existing effects were measured in a roughly 100M transformer with a large vocabulary on FineWeb. A tiny character model has different data repetition, gradient noise and a much smaller auxiliary head. Its ranking and dynamics cannot be assumed to transfer. TS was stronger than PD at 4M in the reference but approximately tied at 16M; the complete chain is sought in at least one informative batch regime, while batch and momentum effects must also reproduce.

**Mechanism and competitor.** Input/output anisotropy, gradient-coordinate normalization and the noise/lag/stability tradeoff could survive small width and vocabulary. Alternatively, corpus overfitting, auxiliary-head dominance in the original, or a short step horizon may prevent a faithful surrogate. A favorable single endpoint, untuned Adam/Muon, or a large-batch momentum gain caused only by too few steps does not establish transfer.

**Independent peer discussion.** A fresh peer supported a staged screen, keeping all three requirements separate. It flagged validation saturation, covariance clocks, the changed body/head balance, and fixed-token versus fixed-step confounding. A separate code audit established the existing optimizer API and the distinction between predictive-label GN output statistics and data-label empirical Fisher statistics. Their guidance is implemented in the following frozen design.

**First candidate and bounded comparison.**
- Tiny Shakespeare characters, Karpathy source pinned by revision and SHA256; contiguous 90/10 train/validation split. Random training windows may repeat and repeated exposure is explicitly logged. No validation gradient/statistic updates.
- Four layers, width 128, four heads, context 128, untied character embeddings/head, no biases, RMSNorm and Q/K RMSNorm; reference initialization. Approximately 0.8M parameters. Standalone loop in `research/tiny_spectra`, reusing only qualified model/optimizer primitives, not `train.py`, `distributed.py`, or their experiment controllers.
- One discovery seed 20260928, batch 8192, horizon 4,194,304 characters (512 updates), physical microbatch 32 sequences, 3.4% warmup and final 10% linear cooldown. AdamW is the original study's Adam baseline; all five body optimizers share auxiliary AdamW LR 0.002, decay 0.01, clipping 1.0, and auxiliary betas (0.9,0.95).
- Exactly three body LRs initially: AdamW {0.0005,0.001,0.002}; Muon/PD/TS/SOAP-PD each {0.004,0.008,0.016}. Plain momentum 0.9; PD alpha 1/4, TS alpha=output power=1/4; SOAP beta2 0.9 and projected-momentum second moments. The common auxiliary LR deliberately isolates the body-optimizer comparison.
- Input covariance sufficient statistics are pooled across microbatches, then committed once per optimizer step with EMA 0.9; stride 4 after position zero. This is a declared small-harness recipe change that holds the statistic clock fixed in steps across batch sizes. Root refresh every 5 steps. TS predictive-label GN moments: 16 training sequences every 5 steps, EMA 0.8; isolated random generator; all positions. Forward BF16, weights/momentum/statistics FP32; the tiny collector explicitly uses FP32 covariance GEMMs (a precision change from the legacy ambient-autocast collector).
- Evaluate fixed validation windows and a fixed training probe throughout; full held-out split at the end. Save paired initial hashes, training-window hashes, actual parameter changes separately from optimizer directions, timing, GPU model/total VRAM, peak allocation, all optimizer/statistic state and source hashes. No resume support is claimed.
- GPU qualification precedes science: CPU scientific-contract checks plus a 12-step smoke (three root refreshes) for all five methods on one A4000 (16GB), capped at 10 minutes. Runtime guard rejects devices >=45 GiB (also excluding nominal 48GB devices whose reported usable memory is slightly smaller).
- Screen: 15 runs, at most four single-A4000 workers concurrently, each at most 30 minutes; projected aggregate target <=2 GPU-hours. Revise the resource plan before launch if qualification predicts more. Retain failures and every LR. A boundary winner permits one predeclared factor-two outward expansion per affected method; never silently expand.

**Reading and next decision.** Endpoint selection uses the fixed held-out bank, with pre-cooldown curves and final full validation also reported. Clear geometry progress without early overfitting advances to Muon/SOAP-PD batch sentinels at 2048 and 32768, then the full ranking/momentum qualification. Failure of Shakespeare is informative: preserve it and consider a compact word/subword dataset or a modestly wider/deeper candidate with a new decision entry, rather than selecting a favorable checkpoint after inspection.

**Completion requirements (not yet satisfied).** Lock a candidate, hyperparameters and loss thresholds using discovery data, then confirm on at least three fresh paired seeds: every adjacent performance comparison in the five-method chain in a nonsaturated regime; widening geometry advantage across at least three batch sizes through common-loss token/step ratios; a lower large-batch momentum optimum after a small LR bracket, with matched-step and extended-large-batch controls. Report uncertainty, ties and reversals. A partial ordering or only one of the dynamical effects does not complete the user goal. Memory/time must be substantially smaller than the original while preserving these phenomena; significant optimizer-memory simplification itself is a later research question.

**The β 0.8 gain for S∘PD replicates on a second seed: −0.107 (2026-09-28 13:56 CDT).** `soaudit_mom16mseed_20260928`, seed 260926, Ada: β 0.8 **4.4985** vs β 0.9 4.6055 (seed 260925: −0.102).
- The gap follows the same path on both seeds: −0.02 at step 10, −0.05 at 20–30, −0.10 at 46, −0.08 to −0.09 at 60–83, −0.10 at the end.
- Prediction (≥ 0.05) holds. The effect is robust across seeds. Its mechanism, and whether it survives the clipping confound and a longer horizon, is what the running arms test.

**Tiny qualification and screen launch (2026-09-28 13:56 CDT).** Nineteen CPU contracts and a five-method paired end-to-end CPU check passed. The frozen 12-step-per-method GPU qualification completed under Slurm 2626581_0 on g18, one NVIDIA RTX A4000 with 16,750,149,632 bytes VRAM. All five methods had identical initial parameter and training-window hashes; snapshots are finite and covariance EMA weights equal 1−0.9^12. The model has 820,864 parameters. Peak allocated memory was 179–239 MB. Conservative projected training cost for the 15-run screen is 0.314 GPU-hours, below the declared limit. Evidence: `logs/tiny_spectra/shakespeare_smoke_20260928/QUALIFICATION_PASSED.json`. The frozen screen is submitted as array 2626585 (15 arms, at most four concurrent A4000s), `logs/tiny_spectra/shakespeare_screen_20260928`. These smoke losses are qualification outputs, not scientific evidence for the target ordering.

**Tiny Shakespeare first screen complete (2026-09-28, Slurm 2626585).** All 15 arms completed with paired hashes and no reported numerical failures or late overfit flags. Best fixed-bank / full-validation NLL: AdamW 1.73803 / 1.74849; Muon 1.55394 / 1.57121; PD 1.52111 / 1.53262; TS 1.51950 / 1.53576; S∘PD 1.51476 / 1.52934. The selected rates are 0.002 for AdamW and 0.016 for all Muon-family arms, all upper boundaries. The requested full chain appears on the selection bank, but the TS–PD gap is tiny and reverses on full validation. Therefore the surrogate is **not qualified**. All methods continue improving through the horizon, so retain the candidate. Per the predeclared boundary rule, run exactly one outward factor-two LR arm per method (AdamW 0.004; others 0.032), same seed and recipe, before dynamics probes. Cost: five A4000 runs, approximately 0.1 GPU-hours. Evidence: `logs/tiny_spectra/shakespeare_screen_20260928/report/README.md`. Fresh-seed scientific confirmation will use full-split held-out NLL as its primary endpoint, with the frozen-bank curve retained, to avoid qualifying a chain confined to the selection subsample.

**Tiny surrogate next decision: baseline bracket closure and batch sentinels (2026-09-28 14:12 CDT).** The five predeclared outward LR arms are complete (`shakespeare_edges_20260928`, Slurm 2626631). PD, TS and S∘PD are now bracketed at LR 0.016; their doubled rate loses. AdamW improves to fixed-bank/full NLL 1.68603/1.69777 at 0.004 and Muon to 1.55106/1.56398 at 0.032, still boundaries. The full-split TS–PD reversal persists. This is a viable but unqualified candidate.

Missing premise: whether its geometry advantage grows with batch, and whether the baseline LR brackets are competent. Extend only the two remaining boundary baselines once more (AdamW 0.008, Muon 0.064, batch8192); this is an explicit second expansion beyond the initial allowance, needed before calling a baseline tuned. In parallel run the planned four Muon/S∘PD sentinels at batches2048 and32768, carrying the current selected LR (0.032/0.016), alpha1/4, momentum0.9 and fixed4,194,304-character exposure. All covariance/refresh/SOAP clocks stay defined in optimizer steps; microbatch maximum32. The hypothesis is a larger S∘PD/Muon step-equivalent gain at32768 than2048; the competitor is a different small-data or auxiliary-dominated regime, or inappropriate carried-over LRs. Discovery thresholds2.4,2.2,2.0 are declared before these trajectories and used only for feasibility, not final confirmation. A missing crossing is censored, not extrapolated. If geometry benefit widens, proceed to momentum/LR qualification; if not, first bracket the sentinels before rejecting transfer. A single LR cannot establish a momentum optimum. Six single-A4000 runs, max4 concurrent, projected aggregate <0.3 GPU-hours; sources frozen separately.

**One-step quality does not explain who gains from the fresher momentum; Muon with clip 0.1 at β 0.9 gains only −0.022 (2026-09-28 14:13 CDT).**
- **Whole-step spectrum probes** on the Muon and PD β 0.8 kept states. The whole-step quantities are exact. Quality a²/2q of each optimizer's own step, β 0.9 → 0.8:

| Method | @9 | @46 | @83 | Final-loss gain from β 0.8 |
|---|---|---|---|---|
| Muon | 0.126 → 0.266 | 0.026 → 0.037 | 0.028 → 0.057 | −0.020 |
| PD | 0.137 → 0.128 | 0.038 → 0.043 | 0.032 → 0.049 | −0.124 |
| S∘PD | 0.033 → 0.033 | 0.030 → 0.045 | 0.028 → 0.040 | −0.102 |

  - The fresher momentum raises one-step quality for every method, Muon's the most, while Muon's training gain is the smallest.
  - **Withdrawn (12:37):** "within one geometry, one-step quality does track the momentum's benefit". As the review noted, a fresher average passes more of the period-2 component and raises one-step slope mechanically.
  - What differs between methods is how much of the extra slope is durable, which a one-step measure cannot see.
- **Clipping confound, first arm.** Muon @0.02 with clip 0.1 at β 0.9 reaches 4.8882, against 4.9104 with clip 1.0 (−0.022). It clips on nearly every step, like the whitening methods. Normalizing Muon's gradients helps about as much as β 0.8 alone (4.8901). The β 0.8 + clip 0.1 arm, running, decides whether the whitening methods' ~0.1 is a clipping effect.

**Tiny batch sentinels complete; endpoint gaps grow, common-loss speedups do not yet (2026-09-28 14:18 CDT).** All six arms completed (Slurm2626637). AdamW0.008 loses to0.004; Muon0.064 loses to0.032 at batch8192, so all five LR choices are bracketed. At batches2048/8192/32768, fixed-bank S∘PD−Muon is −0.02373/−0.03630/−0.08203; full validation differences are −0.02343/−0.03464/−0.08553. This reproduces the growing endpoint loss gap with carried-over rates. It does **not** yet reproduce the stronger rate interpretation: at the predeclared loss2.0 threshold, Muon/S∘PD step ratios are1.151/1.217/1.127. At2.4 and2.2 they are also not monotone. A further exploratory scan over common losses1.7–2.4 likewise does not establish a monotone batch speedup. Preserve that negative result; larger endpoint gaps alone are insufficient. Evidence: `shakespeare_discovery_20260928/DISCOVERY_SPEEDUP.json`.

**Decision: qualify batch × momentum and complete the large-batch ordering.** The large batch has only128 updates, so momentum0.9 has a nine-step lag, while the small batch has2048 updates. The main-study hypothesis predicts fresher momentum benefits whitening more at large batch. The competitor is that the endpoint-gap trend is only loss-curve slope and the tiny setup does not reproduce the useful rate interaction. Run Muon and S∘PD at batches2048 and32768 with momentum{0.7,0.8,0.9,0.95}, each at two adjacent LRs. Small-batch Muon rates{0.016,0.032}, S∘PD{0.008,0.016}; large-batch Muon{0.032,0.064}, S∘PD{0.016,0.032}. Existing identical configurations are reused, never rerun. Also bracket AdamW{0.002,0.004,0.008}, PD/TS{0.008,0.016,0.032} at32768 and momentum0.9 to test whether the fragile TS–PD gap becomes clear there. Geometry power stays1/4 so this tests the same operator across batches.

Exactly37 new A4000 arms, at most4 concurrent, projected aggregate about1 GPU-hour; every arm uses the same fixed4,194,304-character horizon and pooled statistic clock. This implements the independent peer's planned post-screen momentum/LR qualification. Keep the original common-loss thresholds and report the complete ratio-versus-loss curve rather than selecting a favorable crossing. The new discovery decision is whether the best tested momentum shifts down at large batch after allowing LR adjustment, and whether a five-method chain exists on full validation. If these hold, lock recipes/thresholds and run fresh paired seeds plus fixed-step/longer-horizon controls. If not, retain the failure and reconsider the character surrogate or its body/head mismatch. No optimizer simplification or full-size experiment is authorized by these results.

**Independent tiny-surrogate review before confirmation (2026-09-28, approximately14:25 CDT).** The reviewer found no invalidating bug in the standalone training/statistic/kernel paths, but identified two gates beyond finishing array2626669. First, tune momentum fairly: SOAP-PD beta0.8 cannot be compared with PD/TS fixed at0.9 to establish a strict chain. Give PD and TS the chosen lower large-batch momentum at the two adjacent rates before freezing a comparison; also close the visible large-batch Adam lower-LR and PD upper-LR boundaries. At common beta0.9, SOAP-PD's full-validation lead over TS is only0.00091, with the opposite bank ordering. Second, crossing precision is inadequate at small batch: evaluation is every128 updates at2048 but every8 at32768. Some early crossings interpolate across step1→128. Preserve the original nonmonotone ratios as coarse evidence; neither treat them as a precise rejection nor replace them with favorable thresholds. Confirmation will use evaluation every8 updates at every batch and report observed crossing intervals alongside interpolation. Full-split endpoint remains primary for ordering.

Clipping also differs strongly (at batch32768, beta0.9: SOAP-PD113/128 and TS102/128 clipped steps at LR0.016; Muon10/128 at0.032). This can reproduce a recipe-level phenotype but does not isolate momentum lag. The peer's smallest full confirmation design is27 fresh-seed arms (five methods at informative batch plus Muon/SOAP-PD at the other two batches, three seeds), six crossed-momentum arms, and12 replicated fixed512-update controls after an eight-arm discovery LR check for those controls. These are proposed stages, not already launched or evidence of qualification. `logs/tiny_spectra/CONFIRMATION_GATES.json` records the outstanding requirements. All executed model/optimizer/training/data source hashes match across the five tiny cohorts; only launcher/report files changed.

**Large-batch sweep reading and bounded fairness/closure check (2026-09-28, approximately14:29 CDT).** The large-batch part of array2626669 is complete; the small-batch part remains live. Best bank scores are Muon beta0.7/LR0.064 (full1.64573) and SOAP-PD beta0.7/LR0.032 (full1.57998). The0.7–0.8 range is a broad plateau, not a unique established optimum. At beta0.7 SOAP-PD LR0.016 has slightly better full validation but slightly worse bank validation, so retain the original bank-based selection rule. The endpoint-tuned common-loss2.0 ratio at32768 is now1.291, versus1.217 at8192; the2048 comparison is still provisional and coarsely sampled.

A fresh peer endorsed exactly12 additional large-batch arms: PD and TS at shared beta0.7 and LRs{0.016,0.032}; AdamW beta0.9/LR0.001 and PD beta0.9/LR0.064 to close original boundaries; Muon beta0.7/LR0.128 and SOAP-PD beta0.7/LR0.064 to check selected-LR upper boundaries; Muon beta0.6 at LRs{0.032,0.064} and SOAP-PD beta0.6 at{0.016,0.032} to test the lower momentum edge. Same model, exposure, validation bank, geometry and statistic clocks. No fresh-seed claim or confirmation launch follows until these and the small-batch sweep are read. A new boundary winner keeps that bracket open; especially, a Muon0.128 winner would require allowing that LR in the lower-momentum comparison before calling the optimum bracketed. Keep common-momentum and independently selected-recipe rankings distinct.

Resource bound:12 short single-A4000 runs, estimated <0.2 GPU-hours, at most one concurrent. Reduce the existing array's throttle to three before submission and verify at most three current workers, keeping the whole tiny branch at no more than four GPUs. This is a scheduling change only; existing training and frozen sources are preserved.

**Execution qualification on another eligible GPU (2026-09-28, approximately14:33 CDT).** A4000 capacity is constrained while11GB2080Ti nodes are available. A read-only peer inspected installed PyTorch2.11+cu130: sm75 is compiled, BF16 allocation/emulation may be accepted, but NS GEMM support is not established and attention falls back from flash to its mathematical backend. Run exactly one unchanged-code12-step/all-five-method smoke on one2080Ti, at most10 minutes. It exercises actual forward/backward, NS and repeated statistic refreshes. Unsupported or nonfinite execution ends this hardware attempt; no FP16, FP32-NS, package changes or precision-repair branch. If successful, document hardware/backend differences and keep future paired comparisons on one GPU type. Existing scientific runs remain unchanged on A4000. Temporarily cap the existing scientific array at two workers, the fairness array at one and this qualification at one (total≤4); restore the former3+1 cap after this smoke ends. This is instrument/execution qualification, not a new scientific arm or a surrogate result.

**The clipping confound is ruled out: Muon with every gradient normalized still gains nothing from β 0.8 (2026-09-28 14:38 CDT).** `soaudit_clip16m_20260928`, Muon @0.02, 16M, seed 260925, L40S, grad_clip 0.1. Its gradients are clipped on nearly every step, as the whitening methods' are.

| Muon | β 0.9 | β 0.8 | β effect |
|---|---|---|---|
| clip 1.0 (reference) | 4.9104 | 4.8901 | −0.020, all in the cooldown |
| clip 0.1 | 4.8882 | 4.8916 | **+0.003** |

- **Normalizing Muon's gradients helps a little** (−0.022 at β 0.9), about as much as β 0.8 does with clip 1.0. The 16M Muon baseline is within ~0.02 of this local optimum.
- **The difference between methods survives its main confound.** With normalized gradients, the fresher momentum is neutral for Muon, while it gives PD −0.124 and S∘PD −0.102 (−0.107 on a second seed). The asymmetry is tied to the geometry, not to what the momentum averages.
- **Still open.** Muon's LR at β 0.8 (0.014 and 0.028, queued on priv-g14), and the horizon test (S∘PD β 0.8 at 2×, running: −0.09 at step 46, −0.07 at 60). The mechanism by which whitening makes a fresher momentum pay is not shown. The step profiles' stiff-energy bands and the two-tap ranking are consistent with it.

**Decision argument: is the whitening methods' momentum gain a large-batch effect? β 0.8 at 4M (2026-09-28 14:39 CDT).**
- **Observable.** At 16M, β 0.8 gains PD −0.124 and S∘PD −0.10 (two seeds), while Muon gains nothing once clipping is controlled. At 4M, PD α ¼ at β 0.81 was +0.012 worse than β 0.9 (`soaudit_mom4m2`), and Muon's 4M optimum was 0.9.
- **Hypothesis.** The optimal momentum window for the whitening methods shrinks with batch size. Smaller batches need the average for noise; at 16M the gradient's noise is small and the lag dominates.
- **Competing.** The 16M gain is about the short run (92 steps; the 2× horizon arm is running), or it is specific to α ½.
- **Comparison.** 4M, seed 260925, 368 steps, α ½ at β 0.8 against the same-hardware β 0.9 references: S∘PD @0.02 on L40S (3.8260) and PD @0.02 on Ada (3.8589).
- **Predictions.**
  1. Both gain less at 4M than at 16M, by at most 0.03 in either direction.
  2. The gain ranks 16M > 4M, a batch-size dependence of the optimal window.
- **What changes the next decision.**
  - A gain ≥ 0.05 at 4M means the effect is not batch-specific: sweep β at 1M.
  - About 0 at 4M, with the 2× horizon holding at 16M, states a batch-dependent momentum principle for whitening methods.
- **Cost.** 2 runs × ~35 min, after the current queues.

**User correction: prevent evaluation-selection overfitting (2026-09-28).** The user explicitly warned that TS beats PD on selected validation windows but loses on remaining windows, with numerical error much too small to account for the reversal. Treat **every original Tiny Shakespeare evaluation score and every original validation character as development evidence** from now on. Fresh training seeds alone do not make those texts an independent test. Preserve existing artifacts and selected/remaining/full-window reversals. This evaluation issue does not establish a common cause with separate gradient-sign findings.

An independent high-level peer recommended one external, whole-document confirmation panel, frozen before any model score. A separate data-only audit found that MIT's pinned Shakespeare source matches the known tiny opening for11,324 characters. Exclude all nine complete works represented anywhere in the entire old corpus, including its old validation part. Retain **all28 other complete plays**,3,498,575 mechanically extracted characters, with zero shared normalized12/20/32-word and100-letter sequences against the old corpus. Gutenberg independently identifies the same excluded works. Sources, hashes, exclusions and format limitations are in `data/tiny_shakespeare_holdout_audit/README.md`; no model inference or loss calculation was performed.

**Panel policy, fixed before scoring.** Use the audited speech/speaker extraction exactly, the existing65-character vocabulary, separate document boundaries and nonoverlapping128-character contexts with next-token targets. There are17 unsupported digit occurrences; omit and count complete windows touching them rather than substituting or extending the model vocabulary. Use every valid full-text window for final scores. A deterministic curve bank has32 evenly spread valid windows per play; its per-play means are weighted by each play's full valid-token count to estimate the same character-weighted objective. Record equal-play means and per-play differences as prespecified sensitivity analyses. Nearby windows are not independent experimental replications.

**One-shot confirmation discipline.** Freeze configurations, training seeds, horizons, saved checkpoint schedules, common-loss thresholds, uncertainty calculations and the complete comparison family before any model scores on this panel. Finish the whole confirmation training cohort before releasing its test results together. No partial test result may choose a new LR, momentum, checkpoint, play subset or stopping point. Existing submitted sweeps remain development only. A tie, reversal, censored crossing or failure on unseen plays is a failed qualification; do not tune until the panel says the desired ordering. Any subsequent test-guided redesign needs another independently withheld confirmation population. Primary final ordering uses all valid test characters; report both seed uncertainty and play heterogeneity, plus the prespecified macro sensitivity. Original discovery threshold failures remain in the record. The planned fresh-seed confirmation using the old full validation split is superseded by this stronger independent-panel requirement.

**RTX2080Ti execution qualification passed (2026-09-28).** Slurm2626757_0 completed all five12-step methods with unchanged BF16 forward/NS and FP32 parameters/state. Every snapshot tensor is finite, initial weights and training windows match A4000 qualification, and physical VRAM is11,347,623,936 bytes. Peak allocated memory is at most284,509,696 bytes. This qualifies execution on another permitted device, not numerical identity or long-run method comparisons; attention/backend execution differs by GPU architecture. Keep future paired comparisons on the same hardware. Evidence: `logs/tiny_spectra/shakespeare_smoke2080_20260928/QUALIFICATION_PASSED.json`. The scientific array throttle was restored to3, plus fairness1, after this smoke finished.

**Character candidate verdict after the complete declared discovery program (2026-09-28, approximately14:55 CDT).** All75 scientific discovery arms are complete (qualification smokes separate). The setup is useful and cheap, but **not a qualified reproduction**. At shared beta0.7, TS LR0.032 scores bank/full1.564149/1.576896, versus SOAP-PD's bank-selected LR0.0321.565955/1.579976. Thus the last required inequality reverses on both development summaries. Small-batch SOAP-PD beta0.8 and0.7 differ only about0.0013 full NLL; the large-batch low-momentum range is likewise close, with another lower-bound winner at beta0.6/LR0.032. This does not establish a robust shifted optimum. The growing endpoint gap and all coarse common-loss results remain recorded. No independent-panel score was obtained. `logs/tiny_spectra/CHARACTER_CANDIDATE_VERDICT.json` inventories all75 runs. Stop adding batches, horizons, or optimizer powers to this character candidate simply to recover the missing inequalities.

**Independent direction discussion and next transfer candidate.** A fresh peer compared three options: reducing the character batch to512 (which addresses only the momentum premise and adds8192-update runs), selecting an intermediate batch/common momentum (vulnerable to hunting for the missing inequality), or switching once to a small fresh-text subword model. The peer recommended the latter if the existing arms failed the discovery gates, as they have. This is a replacement surrogate test, not a causal attribution of the character failure.

The concrete mismatches are repeated training exposure (4.19M sampled characters from about1M training characters) and an embedding-plus-head share of only2.03% of820,864 parameters. Keeping four layers, width128 and context128 but using4096 subword tokens gives approximately1,852,800 parameters and56.6% in embedding plus head. Fresh TinyStories documents supply more vocabulary structure and allow a nonrecycling training horizon while retaining a model small enough for11/16GB GPUs. All current Shakespeare scores remain development; the28-play MIT panel remains sealed and is not used to choose this new corpus.

**Data-only preparation authorization and bound.** Prepare one pinned TinyStories subset, at most128MiB from the official training text and8MiB from its official validation text, trimmed to complete story boundaries. Fix train/development/test document roles and deduplicate before tokenizer fitting; fit a4096-entry byte-level BPE only on training documents. The test portion is reserved before any model score and is not used for tokenizer training, LR selection or diagnostics. Preserve download range/header checks, prefix hashes, membership hashes, tokenizer JSON/hash/version, token IDs, document boundaries and exclusion counts. Use a private preprocessing-only dependency target if tokenizers is missing; do not modify the shared Python environment. No GPU training follows automatically from preparation. Freeze the subsequent bounded model/hyperparameter comparison before its first run; the intended viability screen is15 arms, five methods times three declared rates, with at most one outward LR check per boundary winner, followed only on success by the fixed batch/momentum screen and one locked independent confirmation family.

**The momentum gain is a phase effect: at twice the horizon, β 0.8 leads by ~0.09 mid-run and ends only −0.010 ahead (2026-09-28 15:02 CDT).**
- `soaudit_horizon16m_20260928`: S∘PD α ½ @0.028 at β 0.8, 16M, 184 steps (3.08B tokens), seed 260925, Ada. Final **3.9740** vs β 0.9 3.9840 (−0.010); β 0.9 @0.02 reached 3.9790.
- **The gap along the run** (β 0.8 − β 0.9, smoothed training loss):

| Step | 10 | 20 | 30 | 46 | 60 | 80 | 100 | 120 | 140 | 160 | 180 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Gap | −0.02 | −0.04 | −0.06 | −0.09 | −0.08 | −0.09 | −0.05 | −0.03 | −0.015 | −0.01 | −0.01 |

- **Reading.** The fresher momentum is faster in the early and middle phase, when the landscape turns quickly; later the longer average catches up.
  - At 1×, the run ends and cools down near the peak of the advantage, so its −0.10 (two seeds) is real but specific to the horizon.
  - The review's horizon concern was right.
  - This matches the common practice of warming momentum up over training, e.g. 0.85 → 0.95 in the modded-nanogpt Muon recipe.
- **Muon's LR at β 0.8** (`soaudit_mlr16m_20260928`): @0.014 4.9042, worse than @0.02 (4.8901); @0.028 running. Muon's non-gain is not an untuned LR at 0.014.
- **Principle, restated.** The optimal momentum window depends on the training phase (short early, long late) more than on batch size alone.
  - At 16M the whitening methods benefit much more than Muon from the short early window: +0.003 for Muon with normalized gradients, against −0.10 for PD and S∘PD at 1×. That asymmetry stands.
  - The 1× step-equivalent speedups at β 0.8 (1.30–1.41×) are horizon-specific. At 2× the final gain is small.
  - The natural next step is a phase-dependent β (a momentum warmup from ~0.8 to 0.9) for the whitening methods.

**Decision argument: a momentum warmup for the whitening methods at 16M (2026-09-28 15:05 CDT).**
- **Observable.** At 2× horizon, S∘PD at β 0.8 leads β 0.9 by ~0.09 over steps 46–80, then loses the lead as β 0.9 catches up (−0.01 at the end). At 1× the run ends near the peak (−0.10, two seeds). Muon gains nothing from the shorter window.
- **Mechanism.** The optimal window is phase-dependent. Early at large batch the landscape rotates quickly and the lag of a long average costs progress; later the direction is steadier, and the longer average's noise and oscillation filtering pays. A window that grows with training should keep both.
- **Simplest competing explanation.** The late catch-up is the cost of the early speed (a trade-off), not a phase mismatch, so any schedule lands between the two constants.
- **Comparison.** S∘PD α ½ @0.028, seed 260925, Ada, with β rising linearly in tokens from 0.8 to 0.9 over the first half of the budget.
  - At 2× horizon, the reference is β 0.9 3.9840 (β 0.8 3.9740).
  - At 1×, the reference is β 0.8 4.4724 (β 0.9 4.5742).
- **Predictions.**
  1. At 2× the schedule ends ≤ 3.965, keeping at least half of the mid-run lead, below both constants.
  2. At 1× it ends within 0.02 of β 0.8.
- **What changes the next decision.** A schedule that beats both constants at 2× gives a practical momentum rule for the whitening methods at large batch; then test its batch dependence (4M). One that lands between the constants means the early speed is traded against the late phase, and a schedule does not help.
- **Code.** `muon_momentum_start` / `muon_momentum_warmup` (train.py `muon_momentum`, applied each step like the LR; off by default), with a unit test; 87 tests pass.

**Fresh-text subword candidate C: numerical plan before any model score (2026-09-28).** The independently reviewed replacement keeps four layers, width128, four heads and context128, with4096 train-only byte-BPE tokens (approximately1,852,800 parameters). Data preparation produced32,939,904 fresh target positions after deduplication, so both the8,388,608-target base horizon and16,777,216-target extended control fit without recycling. The existing character candidate and its negative results remain frozen; no score is taken on its28-play confirmation panel.

Primary discovery uses seed20260928, batch16384 (512updates), shared plain momentum0.8, physical microbatch32 sequences,3.4% warmup and10% cooldown. AdamW body LRs are{0.0005,0.001,0.002}; Muon/PD/TS/SOAP-PD body LRs are{0.008,0.016,0.032}. Auxiliary AdamW is identical in every arm (LR0.002, betas0.9/0.95, decay0.01); body decay0.01 and clipping1.0. PD alpha and TS output power remain1/4; pooled input EMA0.9, root/statistic refresh5, TS predictive-label GN16sequences/EMA0.8, and SOAP projected-momentum second-moment beta2=0.9 are unchanged. This tests a fixed auxiliary recipe rather than unrestricted optimizer superiority. BF16 training with FP32 state; one eligible GPU per arm, at most four concurrent.11GB2080Ti execution is qualified and is the intended common hardware for this candidate; no mixed-hardware selection.

Evaluate development curves every8updates. **Select LRs using full document-respecting development NLL**, never a favorable window bank; retain bank scores and document-group sensitivity as diagnostics. Before launching, fix evaluation coverage to include every within-story target after the first token, including short stories and tails through right-padding/masked targets. This corrects a data-only preparation coverage issue (complete128-blocks alone omitted short stories and about30% of target tokens); preserve v1 metadata and frozen tokens while versioning that evaluation policy. It is changed before any model score. The independent sealed-test documents remain entirely unused for model selection.

The first screen has exactly15arms. Allow at most one factor-two outward LR check for each boundary winner (at most20arms total); preserve all scores, failures and unresolved boundaries. A tied or reversed full-development chain is not a qualified result and ends this candidate's ordering claim under this plan. The screen is a viability check, never confirmation. If it passes, the fixed batch/momentum screen uses batches4096/65536 and Muon/SOAP-PD at momenta{0.8,0.9}, with identical two-adjacent-rate opportunities for both momenta within each method/batch. Existing midpoint beta0.8 then permits a common-momentum comparison at all three batches; that rate comparison is primary, independently selected recipes secondary. No claim of a uniquely tuned momentum optimum follows from an unresolved LR boundary. The meaningful interaction is a larger benefit from the shorter momentum at the larger batch.

The predeclared discovery common-loss thresholds are5.0,4.0,3.0 nats per BPE token, with complete ratio-versus-loss curves, observed crossing intervals and censoring. Different tokenizations' NLLs are not compared numerically. Lock final confirmation criteria from development evidence before any sealed-test score; keep all original threshold outcomes visible. If discovery gates pass, fresh paired seeds and a matched256-update low/high-batch contrast (the large-batch arm doubles base exposure) test replication and the finite-horizon alternative. Their LR treatment and schedules must be frozen before those control results are observed. No power, horizon or corpus search is launched to repair a test-set failure.

Numerical qualification uses CPU data/masking contracts and a12-update smoke of each method on the chosen sub48GB device before the15-arm screen. Bound initial screen at 2 GPU-hours maximum, with actual runtime forecast from qualification; expected cost is much smaller. If qualification predicts excessive runtime or memory, record an execution revision before launching rather than silently changing the scientific recipe.

**Candidate C qualified and screen submitted (2026-09-28, approximately15:22 CDT).** All five12-step smoke arms completed under2626880 on11GB2080Ti. They share initial weights, nonrecycled training windows, validation starts and target-length hashes. Every saved tensor is finite. Full development NLL reconstructs exactly from per-window losses weighted by1,018,977 real targets; masks exclude only right padding and count real EOT tokens. Peak allocated memory is481,521,664 bytes. Conservative projected training cost for the15-arm screen is0.4003 GPU-hours. Evidence: `logs/tiny_spectra/stories_c_smoke_20260928/QUALIFICATION_PASSED.json`. The frozen screen is submitted as2626881 (`stories_c_screen_20260928`), four2080Ti workers maximum. Test documents are not read. These smoke losses are qualification, not ordering evidence.

**Muon's LR at β 0.8 is optimal at 0.02: its non-gain is not an untuned LR (2026-09-28 15:27 CDT).** `soaudit_mlr16m_20260928`, Muon at β 0.8, 16M, seed 260925, L40S: LR 0.014 4.9042, 0.02 4.8901, 0.028 4.9285. Muon's side of the momentum comparison is now controlled for clipping (+0.003 with every gradient normalized) and for LR. The whitening methods' early advantage from a short momentum window stands as a property of the geometry.

**Prospective reference-premise clarification before candidate C dynamics (2026-09-28, approximately15:28 CDT).** The independent main-study result at15:02 changed the reference interpretation: at16M and twice the horizon, the extra gain from SOAP-PD beta0.8 shrinks from about0.10 at1× to0.010 at2×, while its mid-run lead remains. The reference now supports a phase-dependent useful momentum window, not a universal batch-only law. A fresh independent peer agreed that the tiny protocol's13:55 wording (a gain caused only by too few steps does not establish transfer), and any reading that attenuation at2× is automatically failure, were stronger than the observed reference. Preserve that original wording and its history; this dated clarification precedes C batch/momentum results and does not follow a convenient toy outcome.

The required momentum claim is replicated fixed-token batch×momentum interaction, with matched-step and extended-horizon outcomes determining whether it is batch-specific or phase/horizon-dependent. Shrinkage at2× can faithfully reproduce the updated reference and must be reported, not concealed or automatically failed. The controls remain mandatory. Geometry's common-loss speedup is a separate requirement; a failed geometry-rate comparison cannot be relabeled as merely a momentum phase effect. Five-method ordering, independent-panel/seed confirmation, rate metrics, LR fairness and all negative-result retention rules are unchanged.

**Candidate C endpoint-rate rule fixed before dynamics.** If the viability gate passes with bracketed midpoint learning rates eta_star, the two rates for each method at batch4096 are{eta_star/2,eta_star}; at batch65536 they are{eta_star,2*eta_star}. Both momenta0.8 and0.9 receive the identical pair within that method/batch. This uses the same deterministic rule for Muon and SOAP-PD, based only on the midpoint development screen. A remaining boundary is reported as unresolved; the contrast is the prespecified shorter-versus-longer window, not a claim of a global momentum optimum. No endpoint-batch results or sealed-test scores have been observed when fixing this rule.

**S∘PD at β 0.8: LR 0.04 is +0.007 against 0.028 (2026-09-28 15:31 CDT).** `soaudit_strength16m_20260928`: 4.4792 vs 4.4724. The LR optimum stays flat near 0.028 at the fresher momentum; that prediction holds. The α ¾ arm was dropped (13:50).

**Candidate C initial screen complete; dynamics gate closed (2026-09-28, approximately15:33 CDT).** All15 arms completed. Full-development LR-selected losses: AdamW0.0022.837876, Muon0.0082.571820, PD0.0162.543016, TS0.0082.531676, SOAP-PD0.0082.532339. The first four-method chain holds in all four fixed document groups. SOAP-PD−TS is+0.000663 overall, with group differences−0.001608,+0.002156,+0.001913,+0.000228: a small heterogeneous reversal, not the required final inequality. SOAP-PD's selected-window bank favors0.016 but full development favors0.008; retain the full-development rule. A fresh read-only peer verified these results and found no invalidating data/masking/source issue.

Exactly four predeclared edge checks remain: AdamW, Muon, TS and SOAP-PD each at LR0.004; PD is bracketed and receives no added arm. Same dataset, seed, horizon, momentum and hardware. These bring the discovery total to19, within the20-arm cap. No dynamics, fresh-seed confirmation or test scoring follows unless the final viability gate passes. All original screen outcomes remain intact.

**Why whitening lets a shorter momentum window pay: the whitening map filters the stiff oscillation spatially, Muon's polar map passes it on (2026-09-28 15:43 CDT).** `step_profile_probe.py` on the β 0.8 16M states (@9, @46, @83; `step_profile_mom/`), against β 0.9. Energy in the top-16 GN eigenvectors:

| Method | Momentum @9 / @46 / @83, β 0.9 → 0.8 | Own step @9 / @46 / @83, β 0.9 → 0.8 |
|---|---|---|
| Muon | 0.11 → 0.26 / 0.13 → 0.34 / 0.11 → 0.29 | 2.6e-3 → 3.1e-3 / 1.5e-3 → **4.6e-3** / 1.2e-3 → **3.2e-3** |
| PD | 0.18 → 0.35 / 0.34 → 0.34 / 0.34 → 0.33 | 3.3e-5 → 4.1e-5 / 2.0e-5 → 3.5e-5 / 2.1e-5 → 7.3e-5 |
| S∘PD | 0.26 → 0.49 / 0.38 → 0.57 / 0.40 → 0.49 | 7.1e-6 → 8.2e-6 / 6.1e-6 → 8.0e-6 / 4.9e-6 → 8.2e-6 |

- **More oscillation reaches every momentum.** A shorter window passes more of the period-2 stiff oscillation into the momentum for every method.
- **Only Muon's map passes it on.** Muon's polar map carries it into the step: stiff energy 2–3× higher, and at @46 the stiff share of the step's curvature rises from 0.47 to 0.69. The whitening maps keep their steps' stiff energy 50–500× below Muon's.
- **Reading, consistent with every momentum result today:**
  - The edge oscillation must be filtered somewhere.
  - Muon has only the temporal filter, the momentum, and needs a long window despite its lag.
  - The whitening maps add a spatial filter, the input (and SOAP) normalization that decouples the step from stiff directions. So the temporal filter can be short, and the fresher step follows the fast-turning early landscape.
  - The two-tap ranking (Muon > PD > S∘PD) and the clipping control (Muon +0.003 with normalized gradients) fit this.
  - This is a correlation at three states per method, not an intervention.
- **Testable prediction.** A Muon variant whose step is spatially decoupled from the stiff directions, with no other change, would gain from a shorter window as PD does.

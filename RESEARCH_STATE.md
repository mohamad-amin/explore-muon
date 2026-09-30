# Muon / AdamW spectral study: current state

**Tiny-surrogate study:** isolated at the user's request in [tiny_surrogate/](tiny_surrogate/README.md). Its live state and all new decisions are maintained [there](tiny_surrogate/RESEARCH_STATE.md).

**Principles so far (second-order audit, synthesis, 2026-09-28 06:27 CDT).** Each has its evidence in MUON_CASE.md.

1. **The realized gains come from the geometry of a normalized update, not from Newton-type step sizes.**
   - The winners keep the polar (unit-singular-value) step. They improve the coordinates it acts in: input whitening from forward statistics (PD), plus normalization of the gradient's own signal magnitudes in its output-side eigenbasis (SOAP).
   - Greedy or line-searched Newton steps lose because of their step rule (greedy Muon ≈ greedy Newton).
   - GN's direction at a fixed step norm sharpens the network and falls behind. *Corrected 2026-09-28 10:48:* sharpening is not the cause. The winning whitening maps also remove stiff energy, and their states are 30–400× sharper than Muon's. The better-supported cause is that the per-step GN direction is dominated by its curvature sample (`MUON_CASE.md`).
2. **The gain grows with batch size and through training.** Step-equivalent speedup over Muon at matched loss:
   - 1M: ~1.2×; 4M: 1.27–1.36×; 16M: 1.34–1.37× at loss 4.6–4.4.
   - Two seeds and two LRs at 16M agree. This is the GN paper's batch-size claim, measured in training.
   - Loss gaps mislead in both directions: flat curves shrink them, steep curves inflate them.
3. **On the output side at large batch, gradient statistics beat GN curvature.**
   - SOAP's output-side normalization carries 68–79% of its gain on PD. The GN output factor B^-½ adds nothing at 16M.
   - The two are substitutes (S∘TS), and SOAP's second-moment source doesn't matter at ≥ 4M.
   - Per-entry SNR is high for 95–100% of the signal energy in every frame, so the mechanism is sign-like equalization of uneven signal magnitudes, not SNR weighting. The share of significant entries grows with batch size and falls through training.
4. **A state is co-adapted to its optimizer.**
   - Each optimizer's own step is neutral at c = 1 (best at c ≈ ½). Another direction at the same norm meets curvature the dynamics never regulated.
   - One-step scores at a state therefore measure what that state's optimizer left unexploited, not a rate.
   - **Measured across optimizers (step profiles, 2026-09-28 10:38 CDT).**
     - Every optimizer's own step sits at c* 0.45–0.94 of its own GN model, converging to 0.55–0.65 late at 1M, 4M and 16M.
     - The network sharpens where the optimizer does not step. λ_max at own states is Muon 5–25, PD ~100–600, S∘PD and TS ~500–2250, mirroring the steps' stiff energy (1e-3, 1e-5, 1e-6).
     - At 4M and 16M the gradient is mostly a period-2 oscillation of those stiff directions.
     - One-step measures at own states do not rank the optimizers. The speedups accumulate over the trajectory.
   - Better geometries (PD, GN-PD) harvest this once. Rates must be measured on each method's own trajectory.
5. **Exact GN geometry is the largest one-step prize, but not yet a rate.**
   - PD in the exact GN geometry is ~2× PD and 1.2–1.4× damped GN at valley floors, on the true loss. Most of the gain sits within layers, beyond any Kronecker factor or per-pair diagonal.
   - In training it gives a front-loaded one-time gain (1.46× → 1.25× PD over 8 steps), and only with an instantaneous step rule and a transported momentum. Otherwise it turns uphill.
   - Testing it as a rate needs a cheap estimator, so it can run on its own trajectory.
   - **At 16M the question is closed for now (pre-flight, 2026-09-28 10:13 CDT).** At PD's own states @9 and @46:
     - the per-step exact-GN PD direction depends on the 128-sequence curvature sample (cosine 0.15–0.40 between disjoint samples);
     - it is a worse direction than the Kronecker map, at 0.5–0.8× of kron's GN-model optimal decrease;
     - its damping sits 100× above the GN's mean eigenvalue.
     The estimable curvature is the Kronecker bulk. After Kronecker whitening the remainder is a stiff band (10³–10⁴× the mean), where energy regulates sharpness (principle 1), plus sample noise. The rate arms were not run (`MUON_CASE.md`, 10:13).
6. **What the momentum does at large batch, and where the per-step gap does not live (2026-09-28).**
   - *Withdrawn 13:50 (review):* the per-band claims that the flattest band is under-stepped 1.3–29× and that every band above 1e-3·λ_max sits at the edge. With 48 Lanczos nodes, the flattest band is one node, the residual left after the stiff Krylov directions.
   - The period-2 gradient oscillation is the stabilizing feedback. A two-tap filter delays it, and for heavy ball the stable range shrinks about 19×. It costs Muon +0.60, PD +0.19 and S∘PD +0.09 at 16M, in proportion to how much each steps in stiff directions.
   - **A fresher momentum pays early at 16M, and only for the whitening methods.** At 1×, β 0.9 → 0.8 gains PD −0.124 and S∘PD −0.102 (two seeds: −0.102, −0.107; β 0.7 **4.4691**, the best 16M result).
     - Muon gains nothing once clipping is controlled (+0.003 with every gradient normalized), and its LR optimum stays at 0.02 (0.014: 4.9042; 0.028: 4.9285).
     - It is a phase effect: at 2× horizon β 0.8 leads by ~0.09 over steps 46–80, and β 0.9 catches up late (−0.010 at the end). The optimal window is short early and long late.
     - **A momentum warmup keeps part of it (17:05).** S∘PD with β 0.8 → 0.9 over the first half ends at **3.9628** at 2× horizon, the best 2× result. That is below constant β 0.9 (3.9840) and β 0.8 (3.9740).
       - At 1× (92 steps) the same half-budget warmup lands between the constants (4.5152): the whole run is early phase, so the warmup should be defined in steps.
       - **Replicated (19:42).** At 2× horizon the warmup gains −0.021 and −0.024 (S∘PD, two seeds) and −0.027 (PD), each against its own constant β 0.9, with nearly identical gap paths (peak −0.11 at step 46).
     - The phase effect holds at 4M (16:03). PD and S∘PD at β 0.8 lead by 0.07–0.08 around step 80 and end neutral (+0.004, +0.002 of 368 steps). The early phase lasts roughly the first 100–150 steps at both batch sizes.
     - **Dose-response (16:45): an α×β interaction, not the mechanism (corrected 19:56).** The β 0.8 gain at 16M is Muon (α 0) −0.02, PD α ¼ −0.086, PD α ½ −0.124. In steps at ~80: 0, 4 and 6.5 steps (S∘PD 5.5). α also changes the tail amplification, the radii and the auxiliary steps.
       - Whitening strength and window interact: α ¼ is better at β 0.9 and α ½ at β 0.8.
       - Spatial (whitening) and temporal (momentum) filtering of the edge oscillation substitute for each other and must be tuned together.
     - **Mechanism by intervention, PD-top (20:53).** Top-only whitening clamps PD's root at 1: it suppresses the high-variance input directions and does not amplify the tail.
       - Its β 0.8 gain equals PD's (−0.126 vs −0.124; lead 6.5 vs 6.6 steps at step 75), and it keeps 77–89% of PD's gain over Muon at 16M and 4M.
       - The momentum interaction and most of PD's gain come from not stepping along the high-variance input directions. The tail amplification adds 11–23% of the gain and nothing to the interaction.
     - **Candidate mechanism (15:43; weakened 19:56: β 0.8 raises Muon's and PD's stiff step energy by similar factors, and PD's defensible role is added suppression on top of the polar map).** A shorter window leaks more of the stiff period-2 oscillation into every momentum. Muon's polar map passes it into the step (stiff energy 2–3× higher, ~3–5e-3). The whitening maps filter it spatially (steps stay ~1e-5 / 1e-6).
       - Muon has only the temporal filter and needs the long window. Whitening adds a spatial one, so the window can be short.
       - A momentum warmup (0.8 → 0.9) and β 0.8 at 4M are running.
     - The 1× step-equivalent speedups at β 0.8 (1.30–1.41× over the best-tuned Muon) are horizon-specific.
     - **The warmup replicates at 4M (2026-09-28 21:04, 21:19 CDT).** With β 0.8 → 0.9 over the first 120 of 368 steps, PD gives 3.8462 against 3.8589 at β 0.9, and S∘PD gives 3.8166 against 3.8260.
       - That makes five of five pairs against constant β 0.9 (16M 2× ×3, 4M ×2).
       - The 4M ladder is steep: β 0.95 → 0.9 alone gains 0.033–0.035. The earlier "α ½ loses at 4M" was measured at β 0.95.
       - **Caveat (review, 21:48).** Every pair is against β 0.9, which is not the best constant (16M 1×: 0.7 is best, and it beats the 1× warmup by 0.046). "Schedule, not a better constant" is untested; constant 0.85 at 4M and constant 0.7 at 16M 2× are queued (`soaudit_mlaw_20260928`).
       - **Also untested:** the LR warmup is fixed in tokens (3 steps at 16M), so the momentum warmup may stand in for a step-based LR warmup.
     - **Candidate law (review, 21:48): 1 − β* ∝ (q/r)^⅓.** Here q is the gradient's per-step drift and r the per-step noise at the run's batch; this is Cutkosky & Mehta's momentum-error bound for normalized updates.
       - Anchored once (β 0.95 at 1M), it matches the measured optima across batch sizes: 0.90 at 4M, 0.79 at 16M. It also matches the warmup's shape within the 4M run: 0.83 → 0.88 → 0.90.
       - It predicts β ≈ 0.45 at step 9 and ≈ 0.75 at step 46 at 16M. A 0.45 → 0.8 schedule is being tested at 16M 2× against the warmup and constant 0.7.
     - **Settled (2026-09-29 00:43 CDT).** The warmup beats the best constant β tried:
       - 4M: PD −0.010 vs β 0.85; S∘PD −0.009 vs β 0.85;
       - 16M 2×: S∘PD −0.011 … −0.013 vs β 0.8; PD −0.027 vs β 0.9, and constant 0.7 is the worst late;
       - 1M: −0.002 vs β 0.95.
       - The cube-root-law schedule (β 0.45 → 0.8 over 55 steps) ties the warmup at 16M 2× (−0.002, +0.0005). Its early prescription only shapes a transient, and its late values are too short. **The law is not kept as a schedule.**
       - Kept: the best time constant grows along training at every batch size. The endpoint depends on the late value (0.9 at 4M/16M, 0.95 at 1M), not on how short the start is.
     - **Probe readings withdrawn after review (21:48).**
       - The momentum buffer's "noise fraction" (r/(1−β²) over |M|²) is not identifiable. The gradient at W_N is partly the response to the optimizer's own past noise (closed loop), and a signal-free noisy quadratic reproduces every signature: SNR, negative cos(M, μ), lag-1 autocorrelation and q/r.
       - The tail (u < 1) statistics may be leakage from a rotated input basis, so "PD's tail overshoots at large batch" is withdrawn.
       - What stands as measurement: at 1M and 4M the momentum is anti-aligned with the fresh mean gradient along the high-variance input directions, and the per-step gradient changes by about its own size.
   - The unembedding's input is as anisotropic as the body's, and its dominant direction is the input mean. Adam in whitened input coordinates was worse at α ½ (+0.295) and better at α ¼ (−0.040). The line is paused as a repair cascade until a head-LR control for plain Adam exists (13:50).
6a. **PD's tail amplification is SNR-limited: it pays at large batch and costs at small batch (2026-09-28 22:25 CDT).**
   - At the same α ½, hardware and seed, PD − PD-top is −0.056 at 16M, −0.007 at 4M, and +0.011 (Ada) / +0.010 (L40S) at 1M.
   - The sign flips between 4M and 1M, where the per-direction gradient SNR at the run's batch crosses 1 in the weak input directions (`direction_snr_probe.py`; noise per direction must be measured, since the ∝-eigenvalue proxy fails in the deep tail).
   - **SNR-gated PD at 1M beats both fixed choices on the same hardware (2026-09-29 00:31 CDT, α ½, A6000):** gate **3.6955**, PD-top 3.7008, PD 3.7131.
     - The gate amplifies the weak input directions while their SNR is high (tail SNR ≈ 80 at step 50) and cuts back as it falls (w ≈ 0.3–0.4 after step 250, ≈ 0.1 in the cooldown).
     - At 16M its tail weight stays at 1.0 (SNR ~10³), so it should reduce to PD; 4M/16M arms are running.
     - PD-top α ¼ at 1M is +0.014 worse than PD α ¼: mild amplification helps at 1M, strong amplification hurts, which is the partial gain the gate provides.
     - **Caveat (00:50).** At α ¼ the gate is +0.004 worse than PD α ¼ (3.6907 vs 3.6870). The weight SNR/(1+SNR) is not a calibrated optimum: it helps where the base amplification is too strong for the batch and costs where it is about right. The best 1M configuration is still PD α ¼.
   - Principled version, implemented and queued: an SNR-gated PD that interpolates each weak direction between PD-top and PD by its Wiener weight SNR/(1+SNR), estimated online from the across-rank gradient noise (`soaudit_snrgate1m_20260928`).
   - **Leakage check (22:20).** A matrix's own step never flips its gradient: autocorrelation ≥ 0.96, 7–38% of the energy removed in every input bin. The flip along the high-variance inputs comes from all matrices moving together. That is the dynamic side of the earlier coherence finding: the joint step's curvature is 8–17× the block-diagonal sum.
6b. **Per-direction and per-matrix structure of the actual step (2026-09-29; revised 03:46 CDT after review).**
   - **Descriptive, stands.**
     - The collective secant multiplier c* of each optimizer's step departs from ½ early in training in the weak input directions (more for Muon at large batch) and approaches ½ everywhere with training.
     - Every per-matrix method's 48 pieces are positively correlated in output space (mean pairwise ρ̄ 0.15–0.46; one leading mode). The pieces of exact GN from the same input are near zero, some negative.
   - **Withdrawn after review (03:46).**
     - c* ≈ ½ is what any stationary process gives (E[a + q/2] = 0), not the edge of stability, and not evidence that whitening sizes steps correctly.
     - "~90% interactions" restates ρ̄ ≈ 0.2 for 48 pieces.
     - GN's low coherence partly follows from algebra (measured in-sample), and coherence does not track one-step quality. "GN orthogonalizes across matrices, that is the gap" is withdrawn.
   - **Staging the optimizers' own maps across matrices (one step, GN-model scores).**
     - The 16M PD-state results used an unclipped gradient in the momentum input (bug) and are withdrawn.
     - On the clean Muon states staging goes ×0.71 (1M) → ×0.92 (4M) → ×1.36 (16M), confounded by β and loss level. At 1M it pays with a low stale share.
     - Supported: the momentum's stale share sets its sign. Not yet checked: true loss, a clipped momentum at the PD states, smaller curvature batches.
     - The training test waits on those checks (MUON_CASE 03:46).
     - **Corrected checks, 04:20.** In true loss on the clip-weighted momentum, eight layer-wise stages gain ×1.13–1.38 at both 16M states. The 48-stage PD map is inflated by the GN model at Muon's state.
     - **20-step branch, 04:35.** Layer-staged PD from 16M @46, against a same-harness control that reproduces the run:
       - train loss −0.014 → −0.077 over steps 48–66;
       - validation −0.020 / −0.046 / −0.058 / −0.076 at steps 50 / 55 / 60 / 65;
       - pre-clip gradient norm 0.6–0.9 vs 1.3–2.7.
       - The lead grows. Coordinating each layer's step with the earlier layers' output change turns the oscillating trajectory into a steadily descending one. One seed, pre-cooldown; LR ×1.5, an early start, a longer branch and a second seed are next.
     - **Full horizon, 05:27: the lead reverses.** From 16M @9 to step 92, staged − control is +0.07 early, −0.12 at step 50, crosses zero near step 80, and ends **+0.067** in validation (4.7465 vs 4.6795).
       - The line is closed as a recipe. Like the earlier GN-like changes, it gives a front-loaded gain the dynamics take back.
       - Leading hypothesis to test: suppressing the edge oscillation removes its implicit sharpness regularization (sharpness logging next).
     - **Sharpness (06:12).** With the top GN eigenvalue logged along both branches:
       - plain PD's oscillating steps flatten the network, 615 → ~230;
       - the coordinated branch stays 3–6× sharper (1100–1500) and jumps to ~6000 (28×) between steps 70 and 80, at full LR before the cooldown (corrected 07:37; the original said "as the cooldown starts"). Its progress then stalls: step lead 4.1–4.5 over 46–60, 2.6 at 75, −0.1 at 80.
       - The edge oscillation acts as a sharpness regulator, and a step closer to second order removes it.
       - Open: whether the sharpening itself does the damage, or continuing to coordinate in a sharp landscape does. Switch-off tests (coordination until step 30 or 50, then plain PD) are running.
     - **Switch-off (06:59; wording corrected after review, 07:37).** An abrupt switch to plain PD at the scheduled LR does not keep the lead.
       - The first plain step from the 4–5× sharper state is a catapult: gradient norm 9.7 / 18.4 against ~3, and the loss jumps +0.10 / +0.17. Sharpness then relaxes toward plain PD's over 20–40 steps.
       - Each switched run then moves at the control's rate and never recovers. Final validation vs control: coordinated to step 30, +0.047; to step 50, +0.059; throughout, +0.115.
       - Withdrawn after review: "a sharp state only the coordinated step can hold" (it does not hold it either), and "not portable progress" (only the abrupt exit was tested). A gentle exit (plain PD at 0.1× / 0.25× LR from @46) is running.
       - Figure: `staged_switch.png`.
     - **In step units, staging is the only lead that fades (07:37).** The momentum warmup, β 0.8, S∘PD and PD leads hold or grow in steps through the constant phase (warmup: 3 → 7 steps at 16M 2×, 2 → 9 at 4M). The runs' own cooldowns release almost no differential excess; TS's small lead (−0.017 → −0.003) is the exception.
       - The 07:19 "fading" list was made in loss units, where flattening curves shrink every gap. The anneal ledger is dropped; a 16-step anneal also contains about 8.5 full-LR steps of progress.
       - Pattern: changes that shape how the normalized step covers the signal (whitening, equalization, momentum) hold. The one per-step curvature correction, coordination from 32 curvature sequences per rank, grows and then collapses as sharpness runs away. Normalized GN (09-27) and GN-PD floor steps behaved the same way.
       - Running: band-restricted coordination, with the correction kept only on the above-mean input band or only on the bulk (MUON_CASE 07:38), plus a noise probe of the correction.
     - **The staged lead is real progress, and its correction is half sample noise (08:00).**
       - A gentle exit keeps most of the lead: plain PD at 0.1× / 0.25× LR for 4 steps from both @46 states keeps 83% / 68% of the −0.109 validation lead, with no catapult at 0.1×. Coordination gained about 4 steps. The abrupt exit (catapult) and the branch's own later runaway destroy that gain.
       - With 32 curvature sequences per rank, staged directions from disjoint samples have cosine 0.35 at the staged @46 state. The correction is 1.16× the plain direction, so it dominates the step. With 256 sequences the cosine is 0.72.
       - Test submitted: staged vs control with 256 sequences per rank, @9 → 92 (gpu partition, A6000). If noise drives the runaway, the lead should hold past step 70. (Cancelled unstarted, nodes unavailable; rerun on g20.)
     - **Band-restricted and finer coordination (08:45 09-29 → 01:10 09-30): see principle 6c.** The day's intermediate readings (08:45–13:17, including "noise is not the cause" and the β 0.9-only comparisons) are superseded there. Their dated record is in MUON_CASE.
   - Figures: `logs/muon_spectra/second_order_audit_20260926/figures_momentum/` (`secant_*.png`, `twosided_secant.png`, `coupling_*.png`).
6c. **Cross-layer coupling along the shared residual directions is a real second-order gap at large batch (2026-09-29/30; harness runs through full schedules; details MUON_CASE 07:37 09-29 → 01:10 09-30).**
   - **What per-matrix optimizers miss.**
     - From block 1 on, every residual-reading matrix (q/k/v/up) takes its high-variance inputs from nearly the same few residual-stream directions. Top-8 input subspace overlap is 0.79–0.96 between adjacent blocks and 0.93–0.99 between a block's attention and MLP inputs.
     - So the layers' normalized steps add up in output space: the step's curvature is 3.2–5.6× the sum of the layers' own (in-sample), at every batch size.
   - **What fixes it.** A sequential Gauss-Newton sweep that uses the true cross-layer curvature.
     - Each stage's PD/S∘PD step is computed from its momentum plus κ·G·(the earlier stages' corrected steps), on 32 curvature sequences per rank.
     - The correction is kept on each matrix's above-mean input directions (10–15% of them, ~85% of the input variance).
     - Stages run in computation order: q/k/v, then o, then up, then down in each layer (31 GN products per step).
   - **Gains, each against its own harness control** (harness controls sit within ~0.01–0.03 of the real runs):
     - 16M from initialization, S∘PD β 0.8: **4.2770** vs 4.4845 (−0.208), 1.24× in steps. The real run: 4.4724. **With 8 curvature sequences per rank plus an EMA of C: 4.2667**: −0.177 vs a matched control using the same time-averaged root (4.4440), −0.218 vs the old one. The coordination's overhead falls from 34% to ~12–15% of the step (09-30 03:26, 04:32).
     - Branches from @9: PD β 0.9 −0.152/−0.165 (two initializations); PD β 0.8 −0.195; S∘PD β 0.8 −0.177 (layer stages) / **−0.218** (sublayer, 4.2827).
     - 2× horizon, S∘PD β 0.8: from initialization **3.8535** vs its control 3.9772 (−0.124; control within 0.003 of the real run), 1.37× at step 120; the branch from @37 gave 3.8679 (−0.114). The previous best was 3.9628. The cheap version (8 sequences + EMA of C) gives 3.8672 (−0.110 vs the instantaneous-R control; 09-30 05:02).
     - Batch ladder at matched dose: 1M 0.000, 4M −0.022 … −0.028, 16M −0.15 … −0.22.
   - **What is necessary, by control:**
     - the cross-layer direction: keeping only each matrix's reversal of its own band momentum ends +0.110;
     - the sequential structure: Jacobi corrections from plain steps end +0.027;
     - fine enough stages: with the whole correction, layer stages fade to −0.011 while sublayer stages hold −0.172;
     - forward order helps: the reversed sweep keeps 68%;
     - one direction only: a symmetric sweep (forward, then backward, 32 more GN products) is +0.022 against one forward sweep (4.2882 vs 4.2667; 09-30 12:13). Stages closer to the output should yield to earlier ones, not the reverse. A more complete per-step block solve does not pay.
     - Finer than sublayer (per matrix) saturates. 4× fewer curvature sequences keep only 42% unless C (hence R and the band) is averaged over time; with an EMA they keep all of it.
   - **Mechanism (theory note, MUON_CASE 18:28 09-30).**
     - Per-matrix fixed-length steps add coherently along the shared residual directions, so the collective mode is over-stepped roughly by the number of coherent stages, and the edge settles on it (a low, steeply LR-dependent plateau).
     - One forward, band-restricted Gauss-Seidel sweep makes later stages yield. The step direction's curvature falls 3–7× at the same length, and the edge re-forms higher, as a single edge (∝ 1/LR).
     - It stays stable because every step is still one true-loss step.
   - **Why it grows with batch.** At 16M, one step's collective push overwhelms the later layers' own dominant-band momentum through five blocks, and the correction reverses it. At 1M the long-averaged momentum dominates and the correction barely acts.
   - Sharpness settles on plateaus 3–12× the control's without running away when the correction is partial or finely staged. The whole layer-staged correction climbs and runs away.
   - **Not an LR effect (09-30 12:06).**
     - The matched control is best at ×1: ×0.7 is +0.049 and ×1.4 +0.018. The coordinated arm is flat from ×1 to ×1.4 (4.2667, 4.2621).
     - At each arm's best LR, coordination is worth −0.182.
     - The control's sharpness plateau scales like LR^−2.4, the coordinated arm's like LR^−1, as one isolated edge would.
   - **32M (09-30 13:12):** −0.259, with the same 1.3× step ratio as 16M. The cheap version at 2× against a matched control: −0.103 (1.4–1.5× in steps mid-run).
   - **Open:**
     - the real trainer (eight or 32 ordered stages with a GN product between them);
     - whether the edge regulation is being traded for speed in a way that matters at longer horizons;
     - ~~whether the gain survives a tuned aux LR~~ **it does, at ~80% (09-30 18:26):** on the aux ×4 baseline (4.3198) coordination gives **4.1790 (−0.141)**, 1.16–1.23× in steps. That is the best 16M result; together the two are −0.265 against the old control.
6d. **The large-batch gap: distance per step at the edge, the local GN model, and stability (2026-09-30; MUON_CASE 11:40 → 14:55; two reviews and a peer discussion).**
   - **Batch efficiency** (`batch_efficiency.py`; each batch at its tuned body LR/β):
     - With the aux AdamW LR scaled by the square-root rule (1M ×1, 4M ×2, 16M ×4; ×4 is the measured 16M optimum, ×8 is worse), 16M needs **2.84–2.96×** the tokens of 4M, and coordinated 16M **2.40–2.59×**. Unscaled it was 2.9× and 2.4×, so the headline survives.
     - The aux LR was a large first-order confound for absolute numbers: −0.124 at 16M and −0.035 at 4M. It was 0.002 at every batch until 09-30.
     - At 32M both arms need ~1.9× the tokens of 16M (aux unscaled, one pair). Coordination keeps 1.3× in steps there (−0.259).
   - **Step vs path** (one batch, aux frozen, one PD root):
     - Four 4M steps on one 16M batch beat the one step by going further (2.5× the length, uniform over the 48 matrices, cos 0.87). Their direction has ~5/6 of the slope and ~1/3 of the curvature.
     - At matched distance, sequential steps gain nothing.
     - Mid-run, the batch's GN model correctly rates ~2/3 of a four-step path's one-batch gain (0.60–0.73 at @46 and @83). Early (@9) only 0.26–0.49, and for sixteen steps 0.23–0.58.
   - **Frozen-model inner loops are closed at 16M (09-30 15:38–15:58).**
     - GN on the full sub-batch runs away over batches.
     - The full Hessian (central differences) diverges in the first batch: negative curvature is unbounded in a frozen quadratic.
     - 32-sequence corrections, true or GN, stay on a plateau but keep ≤ 0.2 of the gain.
     - Along the path, the GN model's gradient error is linear in Δ (the non-GN Hessian), not the cubic term along the top eigenvector.
     - What makes the true loop stable is the loss beyond second order along the path.
   - **As a rate, edge-tuned steps iterated on a frozen GN model sharpen without the true loop's plateau.**
     - The true loop's λ plateaus (~215); the frozen-model loop's climbs ~15% per batch and runs away by batch ~57.
     - It keeps 0.76 → 0.43 of the true loop's gain over the one step while it lasts.
     - A split that keeps the top 16 GN directions on the plain step fails. So does a cheap simultaneous inner loop: its iterations reverse, a Jacobi-like collective overshoot, and a dose-0 control shows the curvature term is what keeps it near A.
     - Self-stabilization in its cubic, top-eigenvector form is not supported by the residual logs.
   - **Coordination in this lens.** It lowers the curvature of the step direction 3–7× at the same length while every step stays a true-loss step. The edge re-forms higher, and the gain is stable (1.24–1.4× in steps at 16M/32M, and at 2×: −0.103 against a matched control).
   - **Where the top of the whole model's GN spectrum lives (09-30 18:26).**
     - Under PD and S∘PD, at 1M, 4M and 16M, early to late, it is the body's: all-parameter λ1 equals body-only λ1 within 2%, and the head carries ≤ 3% of the top vectors' energy.
     - Under Muon (body λ1 ≈ 6) the head and embeddings hold ~46% of the top, as in arXiv 2607.21716's Adam runs, where the top ≈ the unembedding.
     - The whitening maps sharpen the body one to two orders of magnitude above the head.
7. **Measurement hygiene that changed conclusions:**
   - fixed held-out sets and greedy line searches throttle any direction;
   - smoothed step rules lag after large steps;
   - the stored oscillation excess must be separated from the floor (16-step anneals);
   - Krylov-approximated matrix functions understate geometries;
   - model decreases need true-loss checks.

**Second-order audit (2026-09-27 14:00 CDT; corrected 16:48 CDT; batch size and GN geometry 22:24 CDT):** at a 4M-token batch with tuned momentum (β 0.9), PD − Muon = −0.06, TS − Muon = −0.07, S∘PD − Muon = −0.09 to −0.10 (two seeds); S∘PD α ½ (LR 0.02) reaches 3.826, the best 4M result. The GN output factor (TS) meets the claim rule (4 pairs, mean −0.0071, t = −7). After a short anneal releases the loss the LR maintains, damped exact GN's one-step advantage over every optimizer is large and in the bulk. At this valley floor Muon reaches 0.22–0.49 of it and PD α ½ 0.50–0.75, and the gap widens with 16M-token gradients. Exact per-matrix GN recovers 0.83 of it and one cross-layer Gauss-Seidel sweep 0.97. PD's shortfall is concentrated in down, o and up, and PD zeroes the high-variance input directions that GN still moves. Per-batch damped Newton lost to the momentum methods in training at 1M and 4M. The 16:48 entry shows that this came from its greedy held-out line search, not from GN's direction: Muon's own direction under the same rule does identically badly. The GN paper's inner-loop recipe gains only through extra sequential steps (its no-linearization control does better). Details: dated entries below, `research/adamw_spectra/MUON_CASE.md` and `logs/muon_spectra/second_order_audit_20260926/OBSERVATIONS.md`.

**Track 3 (2026-09-26 21:16 CDT):** PD (α ⅛) with PD-geometry weight decay, on #36, passes the benchmark at 3150 steps (#36: 3250; n = 6, mean 3.27816, score 0.0045; non-H100 GPUs). See the Track 3 section below.

Updated 2026-09-25 21:30 UTC (vs tuned Muon: PD −0.019 at width 512, −0.014 at width 768 (3 fresh seeds, 3 GPU types), −0.015 at 2x horizon; S∘PD −0.031 (width 512), −0.024 (width 768, 1 seed); summary in logs/muon_spectra/improve_w4_20260925/FINDINGS.md;
deflation replication complete; relocation to
`../explore_muon` at 03:06 UTC). Current user instructions
and live job/process records take precedence over the historical notes below.

## Current objective and execution

Study depth at fixed width 512, 8 heads, context 512, global batch 1,048,576
and 1,539,870,720 training tokens. Compare AdamW adaptive-update geometry with
Muon pre-NS momentum and post-NS update geometry, alongside validation NLL.
Measure 24 body matrices every 25 steps plus first/final (60 snapshots).

Eight- and twelve-layer runs are complete for both optimizers. Final validation
NLLs: AdamW8 3.879250005; Muon8 3.720990943; AdamW12 3.839646943;
Muon12 3.664207755. Only the eight-layer pair has a completed comparative report;
the twelve-layer numbers here are a completion record, not a new interpretation.
AdamW20 job **2620278** and Muon20 job **2620623** never started. sacct records
both as cancelled from the user's account at 2026-09-25 15:08 CDT (checked
2026-09-26 21:37 CDT), so depth 20 has no runs; their frozen training code is kept.
No additional experiments were launched by relocation.
The depth-8 frontier-norm variant cohort (allocation 2567578, priv-g14) is complete.
It has no biases, RMSNorm with gains and Q/K norms, with Muon then AdamW; see the
dated entry below.

[Eight-layer comparison](logs/muon_spectra/comparison_depth8_20260925/README.md),
[AdamW protocol](research/adamw_spectra/PROTOCOL.md),
[Muon protocol](research/adamw_spectra/MUON_CASE.md),
[migration verification](migration/move_report.json).

The comparison is a single-seed comparison of optimizer recipes, including
different auxiliary learning rates and effective weight decay. Initial model
hashes and data/token controls match at each completed paired depth. The local
GPT initialization is not verified as the paper's complete initialization.
No causal explanation of the loss or spectral differences is established.

## Retained data and open diagnostic questions

Completed runs retain final model and optimizer state, 60 spectral archives,
per-step metrics, data manifests and frozen sources. The checkpoint was rolling;
intermediate training weights, activations, gradients and singular vectors
were not archived at every spectral sample. Muon final checkpoints contain
momentum for all body matrices, not only the 24 logged panels.

Top-mode energy and related spectral diagnostics are already available over
time. Mean-product attribution, token attribution and cross-fitted descent at
the final checkpoints are done; see the 2026-09-25 spike-origin entry below.
Attributing momentum formation over time still requires temporal measurements.
The noise-threshold and centering variants are not implemented.

This project contains only the spectral-study branch. The separate paused and
sealed head-initialization study in `../last_layer` is outside its scope.

Paper labels: the paper's −0.96 "Final MLP Up" is most likely the MLP down
projection (see the 2026-09-25 check below). Compare local `down` with the
paper's printed "Up" panels.

The notes below preserve the original decisions, failure records and outcomes;
their job-status statements are historical snapshots.

## 2026-09-27 19:55 CDT: Track 3, which parts of PD's preconditioner matter

All arms use #36's schedule compressed to 3150 steps, α ⅛ and PD-geometry decay. Each is compared at step 3150 with full PD (n = 6, 3.27816, SD 0.0010). Figure: `track3/figures/pd_vs_muon_3150_variants.png`.

| Arm | n | Mean at 3150 | vs full PD | Share of PD's gain kept* |
|---|---|---|---|---|
| **Legal TS: full PD + full output factor, β ⅛, B from the training backward (data labels); Track 3-legal (updated 09-28 09:10)** | **2** | **3.27402** (3.27342, 3.27462) | **−0.0041** (t ≈ −4.9) | ~1.5 |
| Legal TS + two-sided geometry decay (updated 09-28 07:20) | 2 | 3.27571 (3.27710, 3.27431) | −0.0025 (large mid-run lead that fades in the cooldown, in both runs) | ~1.3 |
| TS: full PD + full output factor, β ⅛, GN labels (extra statistics pass, not Track 3-legal; added 22:50) | 2 | 3.27457 | −0.0036 (t ≈ −4.3) | ~1.45 |
| TS, β ¼ (same caveat; added 22:50) | 2 | 3.27522 | −0.0029 (t ≈ −3.5) | ~1.38 |
| Full PD + IsoMuon's diagonal output factor | 3 | 3.27709 | −0.0011 (t −1.5 to −2.3) | ~1.15 |
| Diagonal input + diagonal output factor | 2 | 3.28035 | +0.0022 (t +2.6) | ~0.7 |
| Diagonal input only | 2 | 3.28256 | +0.0044 (t +5) | ~0.4–0.5 |

*Against an estimated Muon level on this schedule (3.2855–3.2866: #36 shifted by PD's compression effect). There is no direct Muon run on it.

- **Most of PD's gain needs the full input matrix.** It rests on cross-channel structure: C's diagonal is nearly flat while its eigenvalues span ~3 decades. The training result is larger than the step probe's "diagonal ≈ Muon".
- **The output factor helps in both cases.** It gives −0.0011 on full PD and −0.0022 on the diagonal input, with an early lead that shrinks through the cooldown.
- **The output-side arm passes Track 3's criterion at 3150 with n = 3** (score 0.0050). At 3125 it scores 0.0039, just short.
- **TS, added 22:50.** The full-matrix output factor gives a durable gain of about 0.003 on top of full PD, about 3× the diagonal one. Its lead does not shrink during the cooldown.
  - Both β pass the criterion at 3100 with n = 2.
  - Not a legal Track 3 result: the GN statistics pass is an extra forward-backward every 10 steps. A legal version would take data-label output gradients from the training backward.
  - All TS runs are on A6000.
- **Legal TS, added 09-28 05:20.** The output statistic comes from the ordinary backward pass (a gradient probe, one forward-backward per step).
  - Two runs (09:10): mean 3.27402, −0.0041 against full PD, at least as good as GN TS. It passes the Track 3 criterion at 3100 with n = 2 (score 0.0057). All runs are on A6000.
  - Extending the geometry decay to the output side gives a larger mid-run lead that fades in the cooldown, in both runs: −0.0011 and −0.0038 at the end. The 05:20 claim that it hurts is withdrawn (n = 1).
  - More runs (ideally on L40S / RTX 6000 Ada) are needed before a claim. At the current mean, n = 4 would also pass at 3075.
- **Prior art.** GO-MUON (arXiv:2608.09763, toy scale) is the two-sided quarter-power sandwich. PD is its input half, and TS at α = β = ¼ with data labels is GO-MUON. IsoMuon (modded-nanogpt PR #370) is a clamped two-sided diagonal version.

## 2026-09-27 early morning (CDT): second-order audit, first results

The lab notebook `logs/muon_spectra/second_order_audit_20260926/OBSERVATIONS.md` has the figures and details.

- **Batch size (decisive; `logs/muon_spectra/soaudit_batch4m_20260927`, seed 260925, 1.54B tokens).** At a 4M-token batch (368 steps):

| Optimizer | Final loss by LR | Best |
|---|---|---|
| Muon | 3.9567 (0.007), **3.9524** (0.014), 3.9598 (0.028) | 3.9524 |
| PD α ¼ | 3.8854 (0.01), **3.8847** (0.02); 0.04 running | 3.8847 |
| S∘PD | **3.8478** (0.01), 3.8495 (0.02), 3.8675 (0.04) | 3.8478 |

  - Gaps to Muon: S∘PD −0.105 and PD −0.068 at 4M, against −0.029 and −0.018 at 1M.
  - Curvature-aware preconditioning gains ~3.6–3.8× more at the 4× larger batch. This matches the gap map (85–100% of the frame-diagonal Newton decrement is reachable per step at 4M vs 10–40% at 1M) and the GN paper's batch-size claim.
- **Correction to the gap map.** Along the updates the optimizers actually take, the exact GN curvature is 20–37× the frame-diagonal sum (`one_step_split.py`).
  - The stiff, middle and flat Kronecker-rank parts are strongly coupled (correlation 0.45–0.73).
  - So the earlier "flat directions are 20–100× under-stepped" was an artifact of the diagonal approximation.
  - Every optimizer steps at ~1.6–1.9× the one-step optimum along its own update.
  - Giving the stiff, middle and flat parts separate scales gains ≤ 2% (8% for SOAP-Muon) from step 500 on.
  - Cross-kind coherence alone is 3–4× (q of the whole body ÷ sum over matrix kinds).
- **Output-side factor, one step:** B^−β adds 3% on the PD state and 6–10% on Muon's state, next to 24% for the input side on the PD state. A paired training test is running on 4× A4000 each (`logs/muon_spectra/soaudit_twosided_20260927`): PD α ¼ control vs two-sided α ¼ β ¼ with GN labels, LR 0.01.
- **Gradient persistence (Muon, steps 501–532).** Stiff directions flip sign at lag 1. The flat classes' true-gradient correlation is ~0.2 at lag 1 and ~0.05 by lag 6. The expected gradient is mostly short-lived. PD is running.
- **Next.** The exact one-step GN direction by Lanczos on the exact GN matrix, plain and EKFAC-preconditioned (`one_step_gn.py`), at Muon and PD states for 1M (step 500) and 4M (step 183). It is compared with Muon, PD, two-sided, K-FAC and EKFAC, and with per-matrix norm swaps (shape vs allocation), plus the 48×48 per-matrix GN coupling. Running on priv-g14.

## 2026-09-27 22:24 CDT: second-order audit, batch size, and PD in the exact GN geometry

- **16M-token batch** (`soaudit_batch16m_20260927`, 92 steps, same 1.54e9 tokens, β 0.9). Best finals:
  - Muon @0.02: 4.9104 (bracket 0.014–0.04);
  - S∘PD α ½ @0.028: **4.5742** (−0.336);
  - TS ½/½ @0.028: 4.6677;
  - PD α ½ @0.028: 4.6711.
  - Step-equivalent speedup over Muon (`speedup_by_batch.py`):
    - at matched loss, 16M > 4M > 1M (S∘PD 1.26–1.31× at 16M vs 1.18–1.26× at 4M, at loss 5.0–5.4);
    - at matched fraction of training, 16M < 4M, because 92 steps end at loss 4.9, before the phase where every map's advantage is largest.
  - SOAP's per-entry normalization carries most of the extra gain at 16M: S∘PD − PD is −0.097, against −0.033 at 4M.
- **PD in the exact GN geometry** (`gnpd_probe.py`). GN-PD(p) = −(G+μ)^-p polar((G+μ)^-p b) per matrix at Muon's norms. It reduces to PD α = p when G = I ⊗ C.
  - **At the floor it is the best direction measured:** 11.8 vs PD α ½ 5.7 and damped GN 8.3 on M'; 15.0 vs 7.6 and 12.3 on g4M (×1e-3). That is ~2× PD and 1.2–1.4× damped GN.
    - Review caveats (22:57): these are held-out GN-*model* decreases at each direction's fitted scale; GN-PD's true-loss decrease is unverified; one floor state only.
  - **At the oscillating state on the actual momentum it is worse than PD** (3.3 vs 4.7), like every curvature-weighted map.
  - **Decomposition:** Kronecker B ⊗ C geometry reaches only the two-sided level (5.3–6.0). Exact per-matrix blocks keep 79% (9.3) and exact per-layer blocks 82–90% (9.7 on M', 13.2 on g4M).
    - So ~80% of the doubling appears to live within single matrices, in structure of each matrix's exact GN that no Kronecker factorization holds.
    - Provisional: the settings were unmatched (Krylov budgets, dampings, curvature sets).
  - **The cooldown's simpler reading (review).** GN-PD wins only at large steps: its slope along the step is weaker, its curvature much smaller. As the schedule anneals, the advantage crosses over to Muon near LR 0.0038.
  - Token-weighted C (C_w) adds nothing: cos(C, C_w) is 0.98–0.997 in down, o and up. The per-matrix block from stored token factors (one sampled label per token) was too noisy to decide.
- **Cooldown test** (from Muon's step 1300 to the end). Harness exact (3.7047 vs 3.70465).
  - GN-PD at Muon's schedule: **3.7019** (−0.0028, just short of the predeclared −0.003).
  - At 3× the schedule: 3.7398 (+0.035; the first steps at the edge are destructive).
  - Muon, then GN-PD for the last 19 steps: −0.0007.
  - The floor advantage reaches training only weakly: GN-PD's preferred step grows as the state moves from the edge to the floor, and a fixed multiple of Muon's schedule cannot follow it.
- **Floor steps under one curvature-matched step rule (second review's test, 00:27 CDT, 2026-09-28).** From the step-516 floor, the better geometries' one-step advantages turn out to be one-time releases, not rates.
  - PD α ½ vs Muon: 1.72× in one step, 1.06× cumulative after 24 steps; the whole lead comes in the first 4.
  - GN-PD on the momentum: it gains ~0.011 in its first step, then its direction turns uphill on fresh data (the momentum's stale components magnified by the curvature-weighted map). After 4 steps it is behind both (−0.0040 vs PD −0.0095, Muon −0.0062).
  - One-step ratios at a state measure what that state's optimizer left unexploited. Rates must be measured on each method's own trajectory.
  - **Revised 02:40 CDT.** With an instantaneous (unsmoothed) step rule, GN-PD does not turn uphill; the earlier failure was mostly the smoothed rule's lag. With momentum transport (M + G D) it leads PD after 8 steps (−0.0204 vs −0.0163).
    - The lead is front-loaded: 1.46× at step 4, 1.25× at step 8, a slower rate than PD over steps 6–8. It is a larger one-time release, not a faster rate.
- **S∘TS (SOAP on the two-sided ½/½ map; SOAP statistics patched to L G R).** It ties S∘PD at 4M (3.8257 vs 3.8260) and trails it by 0.085 at 16M (4.6595 vs 4.5742). SOAP's large-batch gain needs the output side un-whitened: SOAP and the output factor are substitutes.
- **Late-phase 16M (2× tokens, 184 steps):** S∘PD α ½ 3.9840 vs Muon 4.1878. At matched loss the speedup is 1.34× at loss 4.6 and 1.37× at 4.4, against 1.27× and 1.32× at 4M and ~1.2× at 1M. The realized gain grows with batch size at every matched loss.
- **SOAP by side at 16M (92 steps, on PD α ½ @0.028).** Output side only: 4.6051, 68% of SOAP's gain over PD. Input side only: 4.6330, 39%. Both: 4.5742.
  - SOAP's large-batch gain is mostly output-side, as at 1M (79%), and the input side adds more at 16M.
- **Per-entry SNR** (`frame_snr_probe.py`): 95–100% of the gradient's signal energy has SNR > 1 at ≥ 1M tokens in SOAP's, GN's and the raw frame alike, so SNR does not separate SOAP from GN on the output side. The share of individually significant entries grows with batch size and falls as training proceeds.
- **SOAP mode** ('gradient' second moment vs the default 'update'): 3.8238 vs 3.8260 at 4M, and 4.5686 vs 4.5742 at 16M, equivalent within noise.
  - Combined with the SNR probe, SOAP's large-batch value is signal-magnitude equalization in the gradient's own output-side eigenbasis, not output curvature.
- **16M, second seed (260926):** S∘PD 4.6055 vs Muon 4.9625 (−0.357; seed 260925 −0.336). Matched-loss speedup 1.30–1.31× (seed 260925 1.26–1.28×). Reproducible.
- **LR check at the 2× 16M horizon:** Muon @0.014 4.1920, S∘PD @0.02 3.9790; the best-to-best gap is −0.209. The matched-loss speedup (1.32–1.37× at loss 4.6–4.4) is robust to the LR choice.
- **Where this leaves the gap to GN.**
  - The right geometry for the normalized map is known to be worth ~2× PD at the floor, mostly within layers.
  - Two obstacles remain before it pays in training: the momentum's staleness at the edge, and a step rule matched to the direction's own curvature scale.
- Atlas: https://claude.ai/artifact/8tZX3HjvacAn7ZhDJSxC8x (view "Step rule & floors"). Details: MUON_CASE.md, entries from 17:58 to 22:24 CDT.

## 2026-09-27 16:48 CDT: second-order audit, the Newton-trainer verdict reversed; where the preconditioners gain

- **Greedy Muon** (independent review, 16:38). The Newton trainers' joint held-out line search, run on Muon's own direction from the 1M step-500 states, reproduces greedy Newton to ≤ 0.002 at every checkpoint:
  - Muon state: 3.9613 vs 3.9609–3.9611 at step 650; 0.29e-3 per step vs Muon's 0.56e-3.
  - PD state: 3.9402 vs 3.9381–3.9383; 0.23e-3 per step vs PD's 0.48e-3.
  - The greedy one-step rule (a fixed 64-sequence set) collapses any direction's step to 1/32–1/4 of the LR. Muon's own step raises that set's loss on 82% of steps while the run tracks its baseline.
  - The "per-batch Newton loses in training" result of 14:00 is therefore about the step rule. GN's direction has not yet been tested with a workable step rule.
- **Normalized GN, one step** (`normgn_probe.py`: GN's direction at the trainer's per-matrix norms and LR). At the oscillating step-500 states it is not better than the run's own step:
  - c = 1: Muon state −0.0005 to −0.0012 vs −0.0016; PD state +0.015 to +0.022 vs −0.0015.
  - More curvature data does not help (64 → 1024 sequences), nor does damping below 1e-3 ρ. Lanczos saturates by 64–128 steps.
  - At the Muon state, GN on the momentum has less curvature *and* less first-order gain than Muon's step.
  - The run's own step is neutral at c = 1 with its optimum at c ≈ ½. Other directions at the same norms meet curvature the dynamics did not regulate: from the PD state, Muon's direction gives +0.108.
  - **Normalized GN in training fails (17:55).** From Muon's 1M step-500 state, GN's direction at Muon's per-matrix norms and LR reaches 4.0760 at step 650, against the baseline's 3.9731.
    - Its rate over 550–650 is half Muon's (0.28 vs 0.56e-3 per step). The top GN eigenvalue rises from ~15 to ~100.
    - GN's damped inverse keeps the step out of the stiff directions, where Muon's edge-of-stability oscillation regulates sharpness, so the network sharpens without check.
    - The GN-in-training line pauses. No step rule tested so far (greedy line search, fixed normalized norm) realizes GN's one-step floor advantage.
- **Where the preconditioners gain.** Read as step-equivalent speedups, since loss gaps shrink as the curve flattens:
  - 4M, β 0.9: the speedup grows through training (S∘PD α ½ 1.22× at step 100 → 1.34× at 300; TS ½/½ 1.15× → 1.29×; PD α ½ 1.14× → 1.23×).
  - 1M: PD α ¼ is flat at ~1.1×.
  - 16-step anneals separate the floor from the loss stored in the oscillation:
    - 1M: equal stored excess (0.035–0.038 at 500, 0.026 at 1300), and a floor lead of −0.028 → −0.015 that is built early and then held.
    - 4M: PD α ½ stores less excess (0.085 vs 0.106 at step 183; 0.047 vs 0.066 at 330), with a floor lead of −0.127 → −0.066.
  - The cooldown realizes the floor and adds little: the 4M final gap is −0.063 vs the floor gap −0.066 at 330.
- **Open, as of 17:58 CDT.**
  - The 16M-token cohort `soaudit_batch16m_20260927` (Muon, S∘PD α ½, TS ½/½, PD α ½, √-scaled LRs, β 0.9; g20 and priv-g14). Does the speedup keep growing with batch size? Muon @0.028 finished at 4.9635 (92 steps).
  - The one-step probe of PD with the GN-consistent input statistic C_w = E[|d_t|² x xᵀ]/E[|d_t|²] (jobs 2625312/3, gpu partition).
  - Atlas: Version 9, with the new view "Step rule & floors" (https://claude.ai/artifact/8tZX3HjvacAn7ZhDJSxC8x).
- Details: MUON_CASE.md entries from 15:41 to 17:58 CDT; OBSERVATIONS.md; figures `figures_normgn/`, `figures_floor/`.

## 2026-09-27 14:00 CDT: second-order audit, the gap to GN in one step and in training

- **Valley-floor premise checks** (review, 11:24; `anneal_branch.py`, `valley_rescore.py`).
  - A 16-step LR anneal from a kept state releases 0.035–0.038 (1M) and 0.078–0.095 (4M) of LR-maintained loss, without lowering the top GN eigenvalue.
  - At these floor states the 16 stiffest GN modes carry < 1% of GN's one-step gain.
  - Shares of damped GN's cross-fitted one-step decrease:
    - Muon 0.22–0.49, PD α ¼ 0.42–0.61, PD α ½ 0.50–0.75, two-sided 0.44–0.63, EKFAC 0.23–0.43, K-FAC 0.15–0.31.
    - Each share falls by 0.1–0.15 from a 4M- to a 16M-token gradient, while GN's decrease roughly doubles.
  - At oscillating states the optimizers look much closer (0.73–0.94 at 4M), because most of the one-step decrease there is the releasable excess.
- **Where the floor gap lives** (Muon 1M floor, 4M gradient):
  - The gap decomposes as PD α ½ 0.69 → exact per-matrix GN 0.83 → one Gauss-Seidel sweep 0.97 → full GN 1.
  - GN's shape at Muon's per-matrix norms keeps 0.88; Muon's shape at GN's norms keeps 0.39.
  - GN's shape inside PD gains most in down (+27%), o (+17%) and up (+15%).
  - Kronecker-frame energy grids: PD α ½ and two-sided put < 1e-4 of their energy in the top ~64 input-covariance directions; GN puts 0.04–0.25 there.
  - A test of weaker post-whitening (PD's trailing R) is running.
- **Spectrum flattening.** Polar beats partial orthogonalization at every state and batch size. The best input power rises from ½ to ¾ with 16M gradients.
- **GN in training.**
  - **The paper's recipe** (inner Muon on a frozen linearization, line search) gains ~0.35 over one-step-per-batch Muon early at 1M. Its no-linearization control (same inner steps on true gradients) does better at every checkpoint, so the gain comes from extra sequential steps, not curvature.
  - **One damped Newton step per batch** (`newton_train.py` v2: Lanczos, held-out joint damping and step choice):
    - It releases the excess in ~20 steps, then gains less per step than PD or Muon.
    - At 1M, PD's own run passes it by step 700 (+0.039 behind at 825).
    - At 4M it leads by 0.011–0.021 at step 200 and trails by 0.019–0.026 at 225.
- **Reading.**
  - The damped GN direction has a large one-step bulk advantage that grows with gradient size, but at 1M and 4M it cannot be taken one batch at a time: fresh-batch Newton lacks the averaging the momentum methods have.
  - This matches the GN paper (GN ≈ SOAP up to 4M).
  - What an optimizer should add is GN-like within-matrix structure (down, o, up; high-variance input directions) on a low-noise average, at a batch-matched preconditioning strength. The training results agree: stronger input whitening pays with fresher momentum (PD α ½ and S∘PD α ½ at β 0.9 at 4M), but both sides at ½ over-whiten (TS α ½ β_out ½: +0.023).
- **Two-sided, confirmed:** TS − PD = −0.0052, −0.0099, −0.0061, −0.0071 (A4000 and A6000; t = −6.95).
- **Running:** the split pre/post whitening test, Newton branches at 4M (PD arm on the cluster, Muon arm on priv-g14), the 1M PD Newton arm (dev chain), and TS α ½ β_out ½ @0.02 (g20).

## 2026-09-27 11:00 CDT: second-order audit, confirmations and the dynamics picture

- **Two-sided data norm (TS) is confirmed** against PD α ¼ at 1M (LR 0.01): Δ = −0.0052, −0.0099, −0.0061 and −0.0071 on A4000 and A6000 (mean −0.0071, t = −6.95). The claim rule is met. At 4M it gains −0.013 and −0.007 on two seeds. Placebo basis: loses. Data labels: 60% of the gain.
- **4M results replicate on a second seed** (β 0.9, seed 260926, L40S): PD −0.062, TS −0.069, S∘PD −0.097 against Muon (3.9239). Seed 260925 gave −0.059, −0.072 and −0.090.
- **How batch size, averaging and preconditioning connect:**
  - Muon β 0.9 vs 0.95: −0.0006 at 1M, −0.031 at 4M.
  - PD α ½ vs α ¼: +0.007 at β 0.95, −0.004 at β 0.9 (4M; one seed, inside the seed spread).
  - Clip 0.1 (normalized gradients) at 4M: −0.019 (Muon), −0.010 (PD).
  - A larger batch lowers per-step noise, so a shorter (fresher) average is best, and a fresher average makes stronger preconditioning of flat directions pay.
  - Muon β 0.5 at 4M is +0.19 behind β 0.9: Muon still needs momentum.
- **Where GN's one-step advantage over per-matrix methods lives: cross-layer coupling.** One Gauss-Seidel sweep over the 48 per-matrix GN solves reaches 0.99–1.01 of full GN in either order; sweeps within layers only reach 0.88; per-matrix Jacobi 0.86 (Muon 1M @500).
- **Staged Muon vs input freshness:** 1.71× (fresh), 1.51× (g + 0.25 M), 1.20× (g + 0.5 M), 0.71× (next momentum). It stops paying near a stale share of 0.7.
- **Stability.**
  - Muon runs at LR × λ_max(GN) ≈ 0.07–0.11 regardless of β (0.81–0.95), Nesterov, batch size and LR. PD in its own coordinates runs at 0.03–0.065.
  - The heavy-ball bound on the linearized dynamics (Newton-Schulz Jacobian as preconditioner) matches only at β 0.95: ratios 0.97–1.09. It is refuted in general (Nesterov 2.8, β 0.81 3.4, β 0.9 1.7). The normalized update sets the edge.
- **Dynamics in the stable phase** (all four optimizers, 1M, steps 500–1300):
  - The loss along each actual step W_t → W_{t+1} is a parabola with its minimum at 0.44–0.57 of the step. Each step overshoots ~2× along its own direction; the valley-crossing oscillation stores 0.002–0.005 of loss at the midpoint.
  - Across optimizers at 1M the stored loss is ordered like the final losses (Muon 0.0053 > PD 0.0039 > SOAP-Muon ≈ S∘PD 0.0030 at step 500). At 4M, lower β stores more yet finishes better, so stored loss (released by the cooldown) is not the inefficiency.
  - An exact replay of the last 100 training batches at the current weights shows two things:
    - The saved momentum is anti-aligned with the current full-batch gradient: cos −0.83 (Muon) and −0.64 (PD) within the top-16 GN modes, −0.31 and −0.47 elsewhere. The Kronecker-bin version ("every bin") overstates this, because those bins are coupled.
    - The stale-free average is 20× the momentum's size.
  - The exact GN transport −G Q predicts the momentum's staleness in direction (cos 0.86: 0.955 in the top 16 modes, 0.73 elsewhere; true Hessian 0.82), overestimates its size 2×, and fails when made per-matrix (0.48) or Kronecker-factored (0.14). Much of this is the algebra of a period-2 cycle. The staleness, like the Gauss-Seidel gain (which reverses on M'), is plausibly carried by the same few global stiff modes.
- **Reading.**
  - The durable gains (PD, TS, S∘PD, larger batch) come from stepping further along the flat, persistent directions without destabilizing the stiff ones.
  - It is untested whether large one-step GN gains on fresh gradients are correction of the stiff oscillation (which training would not keep) or persistent bulk signal. The Newton decrement in the top-16 modes is only 5–15% of GN's one-step gain. Premise checks are next: one-step GN vs S∘PD/TS at a valley-floor state (short anneal), and with a 16M-token gradient.
- **Running:** one-step transport scoring (Muon, PD), low-momentum Muon at 4M (β 0 and the A6000 β 0.9 reference), and TS with β_out ½ at momentum 0.9 (g20).

## 2026-09-27 09:15 CDT: second-order audit, the sequential piece and the two-sided confirmation

- **Gauss-Seidel.** One sequential sweep of exact per-matrix GN solves, where each matrix sees the gradient after the earlier matrices moved, reaches 0.99 / 0.97 / 0.90 of full GN on the Muon 1M, Muon 4M and PD 1M states. Independent per-matrix solves reach 0.86 / 0.91 / 0.65 (the PD numbers are lower bounds; its state needs longer solves).
- **Staged Muon and PD** (their own per-matrix maps in the same sweep, Muon 1M @500):
  - On a fresh 4M gradient: +71% (Muon) and +47% (PD α ½) one-step decrease.
  - On the optimizer's momentum: −29% and −37%.
  - Curvature corrections (GN inverse, sequential coupling) pay only on fresh gradients. Crossover tests (partly fresh inputs) and low-momentum Muon at 4M (β 0.5, 0) are queued. Together they decide whether a staged, low-momentum optimizer is worth building for large batches.
- **Two-sided data norm meets the claim rule on three pairs** (1M, LR 0.01): Δ = −0.0052 (A4000), −0.0099 (A4000), −0.0061 (A6000). Mean −0.0071, t = −4.9. A fourth pair is running.
  - Without clipping, Δ = −0.0107.
  - The placebo basis trails PD by +0.004 to +0.008 through step 1300. It timed out at step 1320; the record is kept.
  - At 4M, −0.013 (β 0.9) and −0.019 (β 0.95).
- **4M with tuned momentum** (β 0.9; Muon's LR bracketed at 0.014 and 0.02):

| Optimizer | Final loss | vs Muon |
|---|---|---|
| Muon | 3.921 | |
| PD | 3.863 | −0.058 |
| TS | 3.850 | −0.071 |
| S∘PD | 3.831 | −0.090 |

  A second seed of the whole comparison is running on priv-g14.

## 2026-09-27 07:10 CDT: second-order audit, after the review (supersedes parts of the 05:05 entry)

An independent review (`research/adamw_spectra/MUON_CASE.md`, 05:41) corrected several readings. Decisive checks since:

- **Where the GN advantage lives** (exact block GN, cross-fitted, fresh 4M gradient). Share of the full-GN one-step decrease:

| State | Muon | PD α ½ | GN per matrix | per layer | per kind |
|---|---|---|---|---|---|
| Muon 1M @500 | 0.62 | 0.78 | 0.86 | 0.94 | 0.97 |
| Muon 4M @183 | 0.90 | 0.99 | 0.91 | 0.94 | 1.00 |

  - Most of the advantage is within single matrices. A smaller part comes from coupling across layers of the same kind.
  - The 05:05 "cross-matrix redundancy is the core of the gap" is withdrawn.
  - GN's one-step lead also exists only near its own optimal step size. At the 1.6–1.9× scale the optimizers run at, it loses on the 4M states: its trust region is small.
- **Momentum was under-tuned at 4M.** Muon @0.014 at 4M with β 0.9: 3.9218, against 3.9524 with β 0.95 (−0.031) and 3.9437 with β 0.81.
  - The 4M preconditioner gaps were measured at β 0.95. They are being re-measured with β 0.9 for PD, TS and S∘PD, and at Muon's LR 0.02.
  - TS with β 0.81 @0.02: 3.8550, still 0.067 below Muon's best so far (β 0.9).
- **Two-sided data norm (GN output factor): a real effect so far.**
  - At 1M, best-vs-best −0.0055 against PD: TS 3.6827–3.6830 over LR 0.007–0.01.
  - At 4M, −0.019 at β 0.95 (TS 3.8658 vs PD 3.8847).
  - A placebo with B's spectrum in a random basis is worse than PD, so the gain needs the GN output eigenbasis.
  - Seed 260926 leads by −0.013 at step 1000; its final is pending.
  - Data-label and no-clip controls are queued.
- **Clipping is part of S∘PD.** At 4M S∘PD clips on 94–96% of steps (norm 1.0), and without clipping it loses 0.020. For polar updates, clipping makes the momentum an average of normalized gradients. Normalized-gradient (clip 0.1) tests for Muon and PD are queued.
- **Edge of stability.** Muon's LR × top GN eigenvalue is 0.104–0.108 across batch sizes and steps. PD, in its own coordinates, sits below that: 0.064 at 1M, 0.026 at 4M.
- **Stronger input power (α ½) loses in training at 1M and 4M**, despite better one-step directions.
- **Negative results.** Partial orthogonalization, per-kind α (no kind wants α ≤ 0), global top-k GN deflation, and warm-started CG from Muon's direction.

## 2026-09-27 05:05 CDT: second-order audit, what the exact Gauss-Newton comparison shows

Details in `logs/muon_spectra/second_order_audit_20260926/OBSERVATIONS.md` (entries 03:40–04:45) and the atlas
(https://claude.ai/artifact/8tZX3HjvacAn7ZhDJSxC8x).

- **Tool.** `one_step_gn.py` computes the exact one-step GN direction for all 48 hidden matrices: Lanczos/CG on the exact GN matrix, held-out scoring, eight optimizer states at 1M and 4M batch. Every other direction is compared with it at its own best scale.
- **Principles so far** (each is a measurement; the training checks are listed with it):
  1. **Curvature information is worth more as the gradient gets more accurate.**
     - With a fresh 4M-token gradient, Muon gets 0.36–0.55 of GN's one-step decrease on 1M-run states. With a 1M gradient it gets 0.49–0.68.
     - Training agrees: preconditioning gains ~3.7× more at a 4M batch.
  2. **Orthogonalization is the robust ingredient.**
     - Full polar beats partial spectral powers U S^p V^T on every state.
     - Frame-diagonal curvature models (K-FAC, EKFAC with the exact diagonal) lose to Muon. Along real updates the Kronecker frame's pairs are coupled 20–37×, so the earlier gap-map reading is withdrawn.
  3. **The gap to GN is redundancy across matrices.**
     - Muon's and PD's 48 per-matrix pieces move the function in nearly the same direction: curvature 16–19× the per-matrix sum. The exact GN step's pieces are complementary: 2–3×.
     - Per-matrix scales do not fix this, and neither does deflating a few global stiff directions (top-32 Ritz vectors plus Newton recover ≤ 0.1 of GN).
     - Within matrices, GN's shape matters most in the MLP down projection.
  4. **Momentum's staleness limits curvature scaling.** Applied to the momentum, Newton scaling loses to Muon and PD, and the expected gradient decorrelates within ~2–10 steps (dense lags).
  5. **The state adapts to the optimizer.**
     - Stronger input power (α ½) gives better one-step directions on a fixed state, but loses in training: at 1M, α ¼ @0.01 3.6891 vs α ½ 3.7125.
     - At 4M: PD α ½ @0.02 3.8917 vs α ¼ 3.8847; S∘PD α ½ @0.01 3.8683 vs α ¼ 3.8478 (A6000). Brackets and an L40S control are running.
     - PD's states are sharper: top GN eigenvalue 134 vs 15 for Muon, and α ½ runs at ~2× PD α ¼'s gradient norm.
- **Training candidate: two-sided data norm** (GN output factor B^−¼ with sampled labels). −0.0052 vs PD at 1M (`soaudit_twosided_20260927`). The LR bracket, β ½ and a second seed are running (`soaudit_twosided2_20260927`, A4000).
- **Running diagnostics.**
  - Fresh-plus-stale inputs g + cM (how much averaging each direction family tolerates).
  - A few warm-started CG steps from Muon's or PD's direction (how many GN products close the gap) (`gnmix/`, `gnwarm/`).

## 2026-09-26 night (CDT): second-order audit (user direction)

- **User direction.** Step back from axis-by-axis tuning, and do not center the work on Track 3.
  - First measure the structure of the per-layer Gauss-Newton (GN) matrix: activations, backprop errors, gradients, their spectra and interactions.
  - Learn which assumptions current optimizers make, and what could be estimated cheaply to approach GN.
  - Short-term or branch gains are not a success criterion.
- **Where it is recorded.**
  - Design: [SECOND_ORDER_AUDIT.md](research/adamw_spectra/SECOND_ORDER_AUDIT.md), after two independent reviews.
  - Protocol entries: MUON_CASE.md, from "Second-order audit".
- **Tools** (`research/adamw_spectra/gn_probe.py`, 8 CPU tests):
  - validated at full size on the final Muon and PD checkpoints (`logs/muon_spectra/second_order_audit_20260926/check_tools_finals.json`);
  - per-sequence gradients sum to the batch gradient (6e-7);
  - the sampled-label second moment matches exact forward-mode GN within Monte Carlo error;
  - exact symmetries are flat (≤ 3e-9 of random).
- **Trajectories** (`logs/muon_spectra/soaudit_traj_20260926`): Muon, PD, SOAP-Muon core and S∘PD reruns with kept checkpoints (steps 10–1300, full state plus next-step weights). Running on priv-g14 and g20 since 23:34 CDT.
- **First measurement (final checkpoints; provisional).**
  - Along the mean input direction, the exact GN input marginal relative to K-FAC's C is:
    - q: ≈ 1.05;
    - k: 0.20 (Muon and PD);
    - v: 3.0 (Muon) and 6.1 (PD);
    - o, up and down: 1.1–1.5.
  - So PD's shared uncentered C is right for queries but misjudges the dominant direction for keys (about 5× too stiff) and values (3–6× too flat).
  - The cause is attention's cross-position structure.
  - The output factor B is much less anisotropic than C.

## 2026-09-25 evening (CDT): PD in the Track 3 benchmark

User goal: improve PD over the tuned Muon baseline, mainly in modded-nanogpt Track 3. Use g20, priv-g14 and
the gpu partition; no cloud VMs; few seeds until the explanations run out. Details and decision arguments:
[MUON_CASE.md](research/adamw_spectra/MUON_CASE.md) (sections from "Track 3 benchmark port" on) and
[track3/README.md](track3/README.md). `track3/screen.py [--steps]` summarizes all Track 3 logs.

- **Track 3 (#36 branch, single runs).** PD beats the same-code control by only 0.002–0.003 at step 3250, about
  45–70 steps, for any LR in 0.02–0.035 and WD in 0.025–0.05. Best: LR 0.02, 3.27663 vs 3.27996 (L40S).
  - PD leads by ~9% of steps early, matching our setup's token saving. At #36's settings the lead is gone
    by step ~1250, about one decay time 1/(ηλ) = 800 steps.
  - Lower decay keeps a large mid-run lead (+185 steps at 2000), but the cooldown erases it.
- **One-factor transfer tests in our setup** (seed 260925, A6000; PD − Muon, reference −0.0172):
  - WD 0.2 (Track 3's shrink per step): **+0.0035**. Muon moves +0.002 from wd 0.01; PD moves +0.023.
  - Batch 524,288: −0.0096 (56% kept).
  - Linear biases: −0.0171.
  - Cooldown 0.7: running.
  - Strong decoupled decay is the main reason PD's gain does not transfer; the smaller batch is secondary.
- **Resolution (00:37 CDT, 2026-09-26): total decay exposure E = ∫ηλ dt decides.**
  - Our setup with Track 3's full recipe (wd 0.2 + cooldown 0.7 + batch 524,288; E ≈ 3.8) reproduces Track 3:
    decoupled PD is +0.0038 vs Muon, and PD + geometry decay −0.0091.
  - Across all runs, decoupled PD keeps its gain for E ≲ 2 and loses it for E ≳ 2.5 (Track 3: E ≈ 2.6).
  - The 23:19 revision below rested on the one long-cooldown case with E ≈ 1.9.
  - **Confirmed in Track 3's own model (02:57 CDT):** at 1M batch (1625 steps, same tokens, E ≈ 1.3), PD − #36 is
    −0.0113. At #36's 0.5M batch (E ≈ 2.6), plain PD averages −0.0022.
  - **Local fix screens** (`logs/muon_spectra/t3screen*_20260926`): milder equilibrium reallocation wins, measured by
    e = α·p/2 (α = PD exponent, p = geometry-decay power). Against Muon's best bracketed LR in the recipe setting:
    e = ¼ −0.005, e = ⅛ −0.008 to −0.010, e = 1/16 −0.011. Round 3 (e = 1/32, and the flat-profile e = 1/16) running.
  - **Track 3 (final, 2026-09-26 ~04:25 CDT):** best configuration α ⅛ + geometry decay p = 2: 3.27201 and 3.27291,
    −0.0073 vs pooled #36 controls and −0.0062 vs the #36 H100 mean.
    - All five runs with e ≤ ⅛ average −0.0067 and all beat every control.
    - At 3100 steps it reaches 3.27990: ~150 fewer steps than #36's 3250, single run, no margin.
    - The local optimum is α ⅛ (e between 1/16 and ⅛); milder settings are worse.
    - **Result (21:16 CDT): passes Track 3's criterion at 3150 steps, 100 fewer than #36.**
      - `track3/train_gpt_pd_pdwd_a0.125_s3150.py`, #36's schedule compressed. n = 6, mean 3.27816, score 0.0045 ≥ 0.004;
        also pairwise-significant vs #36.
      - The sixth run was added after 5 runs scored 0.00399 (disclosed); all runs are on non-H100 GPUs.
      - 3100 (n = 5, mean 3.28014) and 3125 (n = 4, mean 3.27954) do not pass.
      - With the full 3250 schedule read early (n = 3), the earliest passing step is 3175.
    - A local WD × cooldown grid (`logs/muon_spectra/t3wdcd_20260926`) finds #36's WD and cooldown near-optimal for
      PD + geometry decay as well. Untested: decay shape, a separate aux-Adam cooldown, LR floors.
- **Revision (23:19 CDT; superseded by the resolution above).** With Track 3's 70% cooldown, our setup at wd 0.2 keeps PD's full gain even with decoupled
  decay: −0.0232, and PD + geometry decay −0.0217. So the wd-0.2 "kill" below was a noise-floor effect of our 10%
  cooldown, and **strong decay is not why PD's gain shrinks in Track 3**.
  - Muon + geometry decay: −0.0039 at cooldown 0.1, +0.0045 at 0.7.
  - Track 3 geometry decay, 4 runs (two at p = 2, LR 0.025: 3.27411 and 3.27418): −0.0056 vs 3 pooled controls
    (mean 3.27976, SD 0.0024). Plain PD, 5 runs: −0.0022. Unresolved whether this edge is real or needs a
    Track 3-specific explanation.
  - **Track 3 2×2 (00:15 CDT, 2026-09-26):** Muon + PD-geometry decay is **+0.0097** (worse than Muon), while
    PD + geometry decay is −0.0056. The geometry-decay gain is PD-specific: the decay has to match the update's
    geometry.
  - Next suspects: half batch combined with long cooldown (running locally, `t3recipe_20260925`), then the model.
    A Track 3 1M-batch pair is running (g20 PD, priv-g14 #36; ~01:45 CDT).
  - Independent review (22:30 CDT) in MUON_CASE.md.
- **Mechanism (21:25 CDT; superseded as the Track 3 explanation by the revision above).** Under decoupled decay, equilibrium weights shrink exactly where PD steps less, so PD's
  per-direction step sizes are undone. Decaying in PD's geometry, W ← W − ηλ W R²/mean eig(R²), fixes it:
  - Our setup, wd 0.2: PD − Muon goes from +0.0035 to **−0.0188**.
  - Track 3 #36, single run: **3.27411**, vs 3.28209 for the same-GPU control, 3.27710 for plain PD and 3.27866 for
    the H100 mean. It first drops below 3.28 at step 3150.
  - The two same-code #36 controls differ by 0.002, so replicates are running.
  - PD on the no-decay hyperball baseline (#37) gains only 0.001.
- **In flight (20:20 CDT; superseded by the line above for the geometry-decay runs).**
  - PD-geometry decay W ← W − ηλ W R²/mean eig(R²): Track 3 on g20, +66 steps at step 500 vs +38 for PD.
  - PD-H (PD's direction on #37 MuonH, no weight decay): priv-g14. Control #37 on L40S (gpu partition).
  - Same-GPU #36 control: g5.
  - Our-setup arms `logs/muon_spectra/wdgeo_20260925`: geometry decay p = 1 and p = 2, and PD at Muon's LR,
    all at wd 0.2.
  - Half-length A6000 screen of PD variants: `track3/h1625_*`.

## 2026-09-25: Muon improvement search (waves 1–4)

The user's goal, set at about 08:00 UTC: find a principled, solid improvement over Muon (a discovery). Explore the spike branch first, then broaden.
[Protocol, decision arguments, rules and predictions](research/adamw_spectra/MUON_CASE.md) (sections "Muon improvement search" onward).

The cohorts are `logs/muon_spectra/improve_w{1,2,3,4}_20260925`. Each arm is one frontier-norm depth-8 run of about 30–60 minutes on 4 GPUs, from frozen sources with a sha256 manifest.

**Results.** Final val NLL; Δ is paired, same seed and hardware.
- **Mean-input ("spike") branch.** The effects are small or harmful, and seed and hardware noise is about 0.001.
  - Dropping the tracked head: +0.0094.
  - Mean-whitened Muon (E*):
    - −0.0042 (g20) and −0.0022 (A6000);
    - +0.0066 at LR 0.02.
  - E with β = 0.25: −0.0061 (one seed).
- **Muon LR.** 0.01 is the best of {0.01, 0.014, 0.02}.
- **Bias-lag probe.** C's deficit is not a refittable bias offset.
- **Early gains.** Most early gains wash out, and they have the same shape as raising Muon's LR. A step-size / NS-conditioning confound is under test.
- **SOAP-Muon core** (Track-3 reference, matched to its code): 3.68699 vs 3.71457, **Δ = −0.0276** (A6000, seed 260925, single seed). It is worse for the first ~120 steps, then better throughout.

**Update 11:25 UTC.**
- **SOAP-Muon core@0.01 vs Muon@0.01:** −0.0276, −0.0320, −0.0270 (A6000 ×2, L40S). Mean −0.0289, SD 0.0028, p ≈ 0.002. The claim still needs the LR brackets (queued).
- **Input-side-only SOAP:** −0.0214, keeping 78% of the gain with no early deficit.
- **Standard-basis normalization:** −0.0050.
- **Exact-polar Muon:** ≈ −0.003 late.
- **Null arms:**
  - Nesterov: +0.0001.
  - Head kept at Muon's step while the bulk runs at LR 0.02: +0.0093.
- **Queued arms:**
  - partial data-norm Muon (activation second moment, α = ¼ and ⅛; early −0.09 to −0.13 at steps 150–250);
  - activation-basis and column-normalized input-side SOAP;
  - input-whitened polar;
  - input-side SOAP replicas (seeds 260926, 260924 on L40S and g20);
  - S/M LR brackets.

**Update 11:58 UTC.**
- **Partial data-norm Muon** (polar(M C^{−¼}) C^{−¼}, activation second moment, RMS-matched): Δ = −0.0255 vs Muon@0.01 at seed 260925, keeping 92% of SOAP-Muon's gain. It is the principled candidate; replication and LR bracket are in wave 9.
- **Muon's LR:** 0.007 beats 0.01 by 0.008 (A6000), so best-vs-best comparisons use the lower LR; 0.005 is pending.
- **SOAP-Muon:** 4 seeds on 3 GPU types, mean −0.0296 vs Muon@0.01.

**Update 21:30 UTC.**
- **Width-768 verdict:** PD fresh pairs −0.0118, −0.0169, −0.0130 (mean −0.0139, p ≈ 0.006, ≈ 72% of the width-512 gain). Not weakened. The strict per-seed bar is met by 2 of 3 seeds; the third misses by 0.0002.
- **S∘PD at width 768:** −0.0241 (seed 260924, L40S).
- **Damped full data norm** (α = ½, d = 0.1–0.3): ≈ tuned Muon, so α = ¼ is essential. The curvature data are consistent with ¼ but do not derive it uniquely (MUON_CASE.md).

**Update 19:25 UTC.**
- **Width-768 check.** Both methods' LRs were re-bracketed; the carried-over LRs are the interior best (Muon 0.007, PD 0.01). PD vs tuned Muon: −0.0142 (selection), −0.0118 (L40S fresh), −0.0169 (RTX 6000 Ada fresh). The fresh mean −0.0144 keeps ≈ 75% of the width-512 gain.
- **Verdict so far.** Not weakened. The strict "every fresh Δ ≤ −0.012" bar is missed by 0.0002 on one seed, so per the rule a third fresh seed (260927, A6000) is running, due about 21:00.
- **Curvature exponent** at width 768: γ = 0.44, so α = 0.22.
- **Per-kind α = γ/2:** no gain over uniform ¼ (−0.0171 vs −0.0177, 2 seeds).
- **S∘PD LR bracket:** complete, best at 0.01.

**Update 16:20 UTC.**
- **S∘PD vs tuned Muon:** n = 6 on 3 GPU types, mean −0.0313 (fresh seeds −0.0316). Its upper LR check (0.014) is queued.
- **Curvature exponent**, measured directly on tuned Muon's weights: median γ = 0.55, so the curvature-matched α = 0.27, matching the swept optimum ¼.
- **Stacked gain decomposes:** input-side data norm plus output-side SOAP basis (S_left∘PD keeps 79%).
- **Controls:** instrumentation-only Muon = Muon (+0.0007). Output-row normalization adds nothing.
- **Step time** vs Muon on one L40S node (median / mean): PD +6/+9%, SOAP +9/+8%, S∘PD +15/+19%. Wall-clock gains are small with the current unoptimized code.

**Update 16:02 UTC.**
- **2× horizon:** PD − tuned Muon = −0.0164 (L40S) and −0.0145 (RTX 6000 Ada).
- **Token efficiency** (PD at 0.70× and 0.85× budgets): PD reaches tuned Muon's 1× loss at ≈ 0.91× tokens, a ≈ 1.10× speedup. The wall-clock gain is ≈ 1.04× given the +6% step cost, so the gain is modest in compute terms.
- **Centered C** halves PD's gain, so the mean direction matters within the full C.
- **SOAP localization:** its gain is not on MLP-down.
- **Width-768 check (wave 19, launched 15:57):**
  - both LRs re-bracketed on seed 260925 (A6000, 6 arms, about 2.2 h);
  - fresh pairs on L40S (seed 260924) and RTX 6000 Ada (seed 260926) queued;
  - criteria are fixed in MUON_CASE.md.

**Update 15:08 UTC.** Vs tuned Muon@0.007:
- PD: 4 fresh pairs on 3 GPU types, mean −0.0193 (n = 6 overall, −0.0188).
- S∘PD (SOAP-Muon core run in the data-norm coordinates): n = 4 on A6000, −0.0284, −0.0327, −0.0302, −0.0325 (mean −0.0309). L40S and RTX 6000 Ada pairs are queued.
- SOAP-Muon core: −0.0235 (n = 5).
- The α sweep peaks at ¼ (½ gains nothing).
- Magnitude-matched Muon ≈ tuned Muon.

Summary document: `logs/muon_spectra/improve_w4_20260925/FINDINGS.md`.

**Claim met, 14:15 UTC.** Partial data-norm Muon, polar(M C^−¼) C^−¼ at LR 0.01, vs tuned Muon@0.007:
- fresh confirmation pairs: −0.0186 (L40S), −0.0198 (RTX 6000 Ada), −0.0189 (A6000);
- mean −0.0191, one-sided p < 0.001; n = 5 with the selection seeds: mean −0.0185.

Every clause of the predeclared rule in MUON_CASE.md is met. Scope is one model size and one token budget; the horizon and token-efficiency checks are running.

**Update 14:02 UTC.**
- **Tuned Muon:** best LR is 0.007, bracketed on both sides; it beats 0.01 on 3 seeds and 2 GPU types. All claims now use it.
- **Vs tuned Muon:**
  - SOAP-Muon core: −0.0213 (3 seeds).
  - Input-side-only SOAP: −0.0153.
  - Partial data-norm Muon (PD, α = ¼, LR 0.01): −0.0172 and −0.0181 on the selection seeds, −0.0186 on L40S seed 260924.
  - Mean-input whitening: ≥ 0, so the spike branch does not beat tuned Muon.
- **PD brackets:** LR 0.01 is best (0.007 and 0.014 worse); α = ¼ > ⅜ > ½.
- **Magnitude-matched Muon ≈ tuned Muon**, so PD's gain is geometric.
- **SOAP in PD coordinates (S∘PD): −0.0284** on the selection seed. The two mechanisms stack.
- **Running:**
  - fresh-seed confirmation (waves 11 and 14);
  - 2× horizon pairs on priv-g14 and g20 (wave 13);
  - held mechanism arms (centered C, SOAP localization, output rows).

**Wave 4 (launched 10:25 UTC).** The design follows an independent review.
- Controls:
  - exact-polar Muon;
  - Muon LR 0.007;
  - a head-at-stability-limit test (E β = 0.5 at LR 0.02).
- SOAP dissection:
  - input-side basis only;
  - no basis;
  - projected-gradient second moment.
- SOAP replication on seed 260926 (A6000 pair).
- Seed 260924 on RTX 6000 Ada (g20) and L40S (priv-g14), queued after those allocations' current arms.

The claim rule is 3 paired seeds on at least 2 GPU types, mean Δ ≤ −0.005, and one-sided p < 0.05.

The depth-20 jobs 2620278 and 2620623 remain untouched.

## 2026-09-25: NS-budget deflation replication (complete)

The user asked to confirm the deflation paper's NS-budget result "the way they do it" on
priv-g14 and g20. [Decision argument](research/adamw_spectra/MUON_CASE.md),
[cohort](logs/muon_spectra/depth8_nobias_rms_qk_ns3_deflation_20260925/README.md).

- **Design.** Frontier-norm depth-8 Muon, identical to the finished run except the NS map: classic quintic × 3, plain (priv-g14) versus behind the paper's deflation entry (g20). The port matches their public code to 1e-7.
- **Prediction, recorded before launch.** At the reference run's final momentum, three plain steps keep 0.72–0.92 of first-order descent. The paper's gate fires on 11 of 48 matrices (mostly V), so a small late-training effect is expected.
- **Launch.** 07:18 UTC. 33 unit tests pass on the frozen copy. The depth-20 jobs remain pending and untouched.
- **Result (07:50 UTC; one seed per arm).** Val NLL: 5-step reference 3.71310, 3-step plain 3.71941, 3-step deflated 3.71144.
  - Predeclared reading: Δ = −0.0080, "consistent but small".
  - The gain is large early (−0.078 at step 50, −0.048 at 200) and shrinks later. The gate fires on 38% of matrices in steps 1–100 versus 8–17% later.
  - This confirms the paper's direction at our scale. The late-layer, large-scale question stays open.

## 2026-09-25: frontier-norm variant launched (depth 8)

The user chose to adopt three frontier features: no biases, RMSNorm with
learnable gains, and per-head Q/K RMSNorm with gains. Everything else stays
fixed: positions, GELU MLP, heads, initialization, data, seed, batch, tokens,
schedule and the Muon recipe.
[Decision argument](research/adamw_spectra/MUON_CASE.md),
[cohort](logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925/README.md).

- **Code.** `ModelConfig(bias, norm, qk_norm)`; the defaults reproduce the
  recorded 8/12-layer initial-weight hashes exactly. 26 unit tests pass.
- **Pipeline.** Launched detached at 05:47 UTC on allocation 2567578
  (priv-g14, 4× L40S). Muon arm: qualification, 1,469-update run, spike
  diagnostic. Then the AdamW arm: pilots at {6e-4, 1.2e-3, 2.4e-3}, run
  paired by initial-weight hash, diagnostic.
- **Scope.** The old cohorts and the queued depth-20 jobs are untouched; they
  import their own frozen copies.
- **Muon arm complete (06:20 UTC).** One seed; not significance-tested.
  - Validation NLL 3.71310 vs 3.72099 (−0.0079 nats).
  - Momentum ρ1 falls in most matrices (e.g. V 0.83–0.87 → 0.69–0.74, final O 0.66 → 0.57), but the spike remains the token-mean product (35/48 mean readings; final O/down shares 0.96/0.88).
  - QK-norm gives K a nonzero mean term, and its mid/late spikes become first-token driven.
  - Attention sinks strengthen (first-token attention 0.04–0.20 → 0.13–0.32 in blocks 2–7).
  - The bulk's cross-fitted signal is unchanged (about 0.2–0.3).
  
- **AdamW arm complete (07:02 UTC).**
  - LR 1.2e-3 selected (bracketed); initial weights paired with the Muon arm.
  - Validation NLL 3.86532 vs 3.87925 (−0.0139). Muon − AdamW is −0.152 (old −0.158).
  - The first-moment spikes are mostly mean-route (27/48). Outlier channels stay far stronger under AdamW (max/mean-abs 27–83 vs 15–25 for Muon).
  - Conclusion: the architecture change is a small loss gain for both optimizers, not a spike fix.

## 2026-09-25: spike-origin diagnostic at final checkpoints

The user authorized the proposed diagnostic on four GPUs. It ran on allocation
2567578 (priv-g14, 4× L40S); g20 was busy with another project's job. There was
no training and no checkpoint change.
[Decision argument and predeclared readings](research/adamw_spectra/MUON_CASE.md),
[outputs and report](logs/muon_spectra/spike_diagnostics_20260925/README.md).

- **Instrument.** Fresh gradients use 33.5M never-trained tokens per checkpoint at the Muon/AdamW 8- and 12-layer final weights. Gates: validation NLL within 3e-5 of the recorded values; hook reconstruction ≤ 3.5e-3. Failed qualification attempts are kept.
- **Spike origin.** The fresh gradient reproduces most Muon momentum spikes, and those are the token-mean product ē x̄ᵀ. That product explains 0.92–0.98 of u1ᵀḠv1 for final O/down and 0.75–0.81 for final V, where position-0 adds 0.24–0.28.
- **Shared directions.** Q/V/up share the mean-input direction, and late O/down share one residual-gradient direction. K spikes, and the 8-layer early/mid MLP-down spikes, are position-0 (sink) dominated.
- **Outlier channels.** At the final block the mean input concentrates on a few residual outlier channels. AdamW's outlier channels are stronger, and its mean input sits on one or two channels everywhere.
- **Bulk.** Momentum bulk directions carry about 20–30% of their in-sample descent out of sample; only about 2–3 modes per matrix exceed the bulk noise edge. NS forfeits nothing at this scale.

Predeclared outcome: mean route in final O/down/V, so a centering/bias-path
training test is the candidate next step. It needs separate authorization.
The bulk is "mixed", so a noise-edge threshold is not favoured. These results
cover the spike present at every depth at width 512, not the paper's scale
trend. One seed, end-of-cooldown checkpoints.

## 2026-09-25: paper-figure and expert-claim check (read-only)

At the user's request, checked `explain_paper_gist.md` and the attached expert
discussion against the paper's vector figures and the saved 8/12-layer Muon
spectra. No job or training was run.
[Scripts, outputs and figure sources](logs/muon_spectra/paper_claims_check_20260925/README.md).

- **MLP labels.** Fig. 14 and Figs. 1/7/12/13/15/16 swap MLP Up and Down at every depth; the attention panels agree without a swap. The steep matrix is called "projection"/"proj"/"Down" in the text and in Figs. 5, 6 and 14, so it is most likely `c_proj`. Final-block exponents then rank down 0.96, O 0.66, V 0.58, up 0.52.
- **Fig. 6** shows 2.8B data; only its "Layer 20/27" titles are wrong.
- **2.8B spectra.** Final O has σ₁²/‖M‖_F² ≈ 0.92 and the steep matrix ≈ 0.985. Five-step NS keeps about 77% of first-order descent for the steep matrix and 93–95% for final V/O. Rank-1 deflation restores about 100%.
- **Curvature.** The steep law's local slope runs from about the width baseline (−0.3 to −0.5) at small sizes to −1.4 to −1.7 at 1.2–2.8B.
- **Local runs.** The cooldown overlap of the paper's window changes medians by −8% to +4%. At width 512, going from 8 to 12 layers does not lower the final-block medians, whereas the paper's 77M→160M step implies ×0.5–0.8. V carries the largest outlier share at every local depth.

These are readings of published figures plus one-seed local data, not causal
results. The pending depth-20 runs test depth alone. The proposed mean-product,
sink and cross-fitted diagnostics remain unrun.

## 2026-09-25: completed eight-layer AdamW/Muon comparison

At the user's request, compared the completed eight-layer runs and generated
[figures, CSV exports and provenance](logs/muon_spectra/comparison_depth8_20260925/README.md),
including [validation curves](logs/muon_spectra/comparison_depth8_20260925/validation.png),
[median spectral trajectories](logs/muon_spectra/comparison_depth8_20260925/median_all24.png)
and [the full PDF](logs/muon_spectra/comparison_depth8_20260925/comparison.pdf).
Both reached 1,469 updates / 1,539,870,720 tokens and have all 60 snapshots.
Initial-weight hashes, frozen model code, data manifests, global batch,
validation bank and token exposure match. All 24 matrices in each spectral
object were checked against the full archived singular values and energy.

Final validation NLL is **3.720990943 for Muon**, **3.879250005 for AdamW**,
a difference of **−0.158259062 nats/token**; perplexities 41.305 and 48.388
(14.64% lower for this Muon recipe). At the endpoint, the top singular direction
contains 5.96–17.00% of AdamW update energy across the 24 matrices, versus
0.209–0.389% of Muon's post-NS update energy. Muon's pre-NS momentum remains
concentrated (15.26–86.01%). Flattening of post-NS spectra is expected from
orthogonalization and does not establish the cause of the NLL improvement.

This is one seed and two optimizer recipes: AdamW peak LR 0.0012; Muon body
LR 0.01 and auxiliary AdamW LR 0.002, with different effective decay and GPU
counts (one versus four RTX 6000 Ada GPUs). There is no isolated causal NS,
statistical-significance, exact paper-initialization, wall-time speedup or
formal stabilization claim. The predeclared 1100–1300 pre-cooldown summaries
are retained. This establishes an observed loss and geometry difference in
the local eight-layer regime. The next decision is to compare the authorized
deeper paired runs when complete; no extra seed, tuning or job was launched.

## 2026-09-25: Muon depth comparison launched

The user canceled the **16-layer AdamW** cell before launch and authorized
Muon at depths **8/12/20 on4/4/8 GPUs**. Status verified01:50 UTC:
- **8 layers:** scientific training on g20 under **2618555.9**, four48 GiB
  RTX6000 Ada GPUs; reached at least126 updates. Momentum and post-NS archives
  at steps1/25/50/75 each contain24 matrices with512 singular values each.
- **12 layers:** job **2620620**, four48 GB RTX A6000 GPUs on g14; tiny CUDA
  qualification passed and full-size peak-rate qualification is running.
- **20 layers:** replaced pending eight-GPU Muon job2620619 with **2620623**,
  requesting **four96 GB GPUs**, at 2026-09-25 01:55 UTC. The user requested
  moving one depth20 job to four GPUs. Original job2620619 was canceled before
  launch; sources, config, global batch and token horizon are unchanged.
  [Scheduler change](logs/muon_spectra/depth20_w512_20260925_r2/gpu_count_change.json).
The completed AdamW8/12 baselines and queued AdamW20 job2620278 are preserved.
The sealed head-initialization study remains untouched.

Width512, heads8, context512, global batch1,048,576 tokens, seed260924,
initialization recipe, data order, warmup50 and the1,539,870,720-token horizon
remain fixed (1,469 updates). Muon uses body LR.01, plain momentum.95 and
five paper NS polynomials; auxiliary AdamW uses LR.002. Separate archives
record pre-NS FP32 momentum and captured post-NS direction spectra, excluding
LR/decay, every25 steps plus first/final. This is a paper-inspired optimizer
recipe comparison, not isolated NS causality or exact paper reproduction.
[Decision argument](research/adamw_spectra/MUON_CASE.md),
[independent review](research/adamw_spectra/MUON_REVIEW.md).

[Qualification](research/adamw_spectra/MUON_QUALIFICATION.md) records16 passing
unit checks, uneven-batch distributed CPU checks and the initial CUDA failure.
The final implementation computes each NS matrix on one owner and shares its
FP32 direction; strict replica and communicated-owner checks pass. NS is eager
and batched after compiled NS disagreed with its BF16 reference beyond the
fixed check; model compilation remains enabled. Originals are preserved.
Eight-layer full-size peak-rate qualification passed:1.02s/update, .74s for
both spectral panels, 3.44 GiB peak per rank, rough31.5-minute main forecast.
Public jobs independently qualify before science and all pipelines cap at7h45m.
No Muon spectral or comparative-loss conclusion is yet recorded.

[8-layer run](logs/muon_spectra/depth8_w512_20260925_r2/README.md),
[12-layer run](logs/muon_spectra/depth12_w512_20260925_r2/README.md),
[20-layer run](logs/muon_spectra/depth20_w512_20260925_r2/README.md).
Next decision: inspect completed paired trajectories and NLL; do not infer
an optimizer speedup from hardware-dependent elapsed times.

Initialization audit (user question, 2026-09-25): frozen AdamW and Muon
model code, seeds and model configs match at each depth; actual initial-weight
hashes match for the launched8/12 pairs. The initializer is GPT-style normal
std.02, with attention-output/MLP-down std.02/sqrt(2L), zero biases and default
LayerNorm. It does not explicitly apply Appendix A.1's stated initialization
factors sqrt(d_out/d_in) and1/sqrt(d_in) for the head. The paper does not specify
a complete base initialization or experiment revision, so equivalence cannot
be asserted or a unique paper initializer inferred from those factors alone.
The current cohort is a matched local recipe, not verified paper initialization.
See [paper A.1](https://arxiv.org/html/2606.04058v2#A1) and
[frozen initializer](logs/muon_spectra/depth8_w512_20260925_r2/frozen/adamw_spectra/model.py).
No configuration or job was changed for this read-only audit.


## 2026-09-24: submitted fixed-width depth comparison

The user requested two eight-GPU jobs, then clarified that width and batch
must remain fixed to study depth. No jobs from the initial width-scaling
interpretation were submitted. Corrected jobs are **2620279** (12 layers,
48 GB or RTX Pro) and **2620278** (20 layers, 96 GB RTX Pro), both in `gpu`,
eight-hour limits. Status verified 2026-09-25 01:23 UTC: the **12-layer job
completed successfully** on g22 (eight RTX A6000 GPUs), exit 0:0, allocation
elapsed 00:27:45. Its scientific run reached all 1,469 updates and
1,539,870,720 tokens, with final validation NLL **3.8396469429135323**,
all 60 spectra and generated plots. Main invocation time was 1474.68 seconds
(about 24.6 minutes). The **20-layer job remains pending for Resources**, with
no node assigned or training output yet; it still requests eight 96 GB GPUs.
Both fix width 512, 8 heads, context 512, global batch 1,048,576, total tokens
1,539,870,720, seed 260924 and the eight-layer selected LR 0.0012. Thus each
has 1,469 updates and 60 spectral samples; only depth changes under the same
recipe (including its residual initialization scaling with depth). Counts
are 89,603,072 and 114,822,144 parameters. No per-depth LR retuning is added.

[Depth case](research/adamw_spectra/SCALE_UP_CASE.md),
[12-layer job](logs/adamw_spectra/depth12_w512_20260924/README.md),
[20-layer job](logs/adamw_spectra/depth20_w512_20260924/README.md).
The new DDP runner keeps the global mean gradient with actual-token weighting,
handles uneven final microbatches, clips after reduction and distributes the
24 independent SVDs. Three-rank CPU qualification passed against the single
device and on resume; all replicas' parameters/moments were byte-identical
within a run. Distributed resume differed by at most 1.86e-9, so numerical
agreement, not bitwise identity, is claimed. The 12-layer job passed tiny and
full-size eight-GPU qualification and exact replica weight/moment audits
before its full run. The 20-layer job will do the same after allocation.
[12-layer summary](logs/adamw_spectra/depth12_w512_20260924/scientific/summary.json),
[12-layer median spectra](logs/adamw_spectra/depth12_w512_20260924/scientific/plots/median.png).
No depth-effect or spectral-stabilization interpretation is yet recorded.
Frozen sources isolate these
jobs from the now-completed eight-layer g20 run. The current original 32 data
shards suffice; the separate 7.2B-token staging done before clarification is
unused. No sealed initialization outcomes were read.

## 2026-09-24: separate AdamW update-spectrum implementation

The user requested a simple approximately 77M AdamW setup comparable to
*Spectral Scaling Laws of Muon*, measuring the adaptive update rather than
only the first moment. The isolated [implementation](research/adamw_spectra/README.md)
and [protocol](research/adamw_spectra/PROTOCOL.md) are complete with CPU
[qualification](research/adamw_spectra/QUALIFICATION.md). This is a new
descriptive question, not a head-remedy result or a Track-3 modification.
The 76,993,536-parameter reconstruction uses 20N = 1,539,870,720 tokens,
1,048,576 tokens/update and 1,469 updates. Batch size and untied vocabulary
matrices are supported by the plotted horizon and parameter count; exact
architecture details remain documented reconstruction choices.
Subsequent user steering sets spectral sampling to every 25 optimizer updates
plus first/final (60 samples), while training loss remains logged every update.

The subsequent [efficiency update](research/adamw_spectra/EFFICIENCY.md) enables
compiled CUDA execution, fused AdamW, bulk token transfer and cached validation.
Eleven CPU tests, a 53-step comparison against the original implementation,
and tiny compiled/BF16/fused CUDA checks passed. The user then explicitly
requested execution on the other idle allocation, g20. Allocation 2618555
was verified to contain four idle 48 GiB RTX 6000 Ada GPUs. A detached
[frozen pipeline](logs/adamw_spectra/g20_20260924_212246_r2/README.md) completed
under Slurm step **2618555.3**: five-update full-size qualification, three
paired 300-update LR pilots in parallel, then one selected-rate scientific
run and analysis. Full-size qualification now passed; worker **2698379**
exited 0. All three pilots completed successfully; the fixed selection
chose **LR 0.0012**, the upper grid boundary. The scientific eight-layer run
and plotting completed at **2026-09-24 23:45:21 UTC** (18:45 Central).
Slurm accounting confirms **COMPLETED, exit 0:0**, elapsed 02:13:16 for the
pipeline. All workers exited 0; the final checkpoint is at update **1469**
and **1,539,870,720 tokens**. Final held-out validation NLL is
**3.879250004887581**. All **60** spectral snapshots covering the selected
**24** matrices and all plot families are present. The main training
invocation took 6569.51 seconds (about 1h49m); measurement time was 69.48 seconds.
[Summary](logs/adamw_spectra/g20_20260924_212246_r2/77m_seed260924/summary.json),
[median trajectories](logs/adamw_spectra/g20_20260924_212246_r2/77m_seed260924/plots/median.png).
Completion was verified 2026-09-25 01:20 UTC. This is a completed baseline
dataset, not yet an interpretation of stabilization or a depth effect. The
next research decision is to inspect trajectories and compare qualified
depth runs when available. Do not restart the completed baseline. Source hashes, configs, logs and
worker handles are preserved in that run directory. The [selection record](logs/adamw_spectra/g20_20260924_212246_r2/selection.json)
keeps all candidate scores. No extra candidate or second eight-layer seed
is added. No spectral-mechanism conclusion has yet been drawn.
The original step .2 completed its five-update training worker (about 4 s
per steady update, 3.24 GiB peak allocated) but stopped before pilots: FP32
Frobenius accumulation produced 4.77e-4 normalized-energy error, above the
unchanged 2e-4 tolerance. The scalar norm now accumulates in FP64, with no
training-update change; remeasurement of the real checkpoint gives error
below 1e-6. Original outputs and failure are retained in the first attempt.
The corrected [qualification](logs/adamw_spectra/g20_20260924_212246_r2/QUALIFICATION_PASSED.json)
measures 3.94 s/update after startup, 1.17 s per spectral panel collection,
and 3.24 GiB peak allocated GPU memory. Pilot initial-model hashes,
source hashes and training/validation data manifests match. All 12 CPU
tests passed after the precision fix. The rough forecast is 20 minutes for
parallel pilots, then 98 minutes for the main run, plus startup/I/O/analysis.
This is runtime qualification, not a scientific conclusion about the spectra.
The separate sealed
head-initialization study remains untouched. The older research snapshot
below retains its original date and must not be read as current job state.

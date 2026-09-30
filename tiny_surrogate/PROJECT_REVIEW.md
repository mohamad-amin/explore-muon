# Project history and evidence status

Reviewed on 2026-09-28. This account separates the reference study from this
isolated small-model study. Dated launch notes describe history; the live small
study is tracked in [RESEARCH_STATE.md](RESEARCH_STATE.md).

## How it started

The initial question was descriptive: do the layer-dependent spectra discussed
for Muon also appear in AdamW's adaptive updates? The first setup used an
approximately 77M-parameter GPT on FineWeb, with exact singular values recorded
through training. It then became a fixed-width depth comparison between AdamW
and Muon. Eight- and twelve-layer pairs finished; the queued twenty-layer runs
were cancelled before starting.

A central measurement correction was to distinguish the gradient, accumulated
momentum, post-orthogonalization direction, and actual parameter change. A large
singular mode in momentum alone does not establish a training bottleneck.
See the [original AdamW protocol](../research/adamw_spectra/PROTOCOL.md) and
[Muon comparison protocol](../research/adamw_spectra/MUON_CASE.md).

## Stages of the reference work

1. **Describe and diagnose spectra.** Saved-checkpoint probes examined whether
   dominant modes reflect activation means, attention sinks, or more distributed
   gradient structure, and whether the spectral bulk carries signal on fresh
   samples. Architecture and numerical controls followed.
2. **Turn geometry into optimizer comparisons.** Deflation, input-statistic
   preconditioning, and SOAP variants were evaluated by learning progress.
   Partial data-norm Muon (PD) improved on tuned Muon in fresh-seed, hardware,
   width, and longer-horizon comparisons. SOAP in PD coordinates added further
   benefit. This was evidence about concrete recipes, with controlled scope.
3. **Dissect the gains.** Input covariance structure and SOAP's output-side
   gradient normalization emerged as complementary contributors. A separate
   Track 3 benchmark port exposed interactions among geometry, weight decay,
   and cooldown. Those results use a different recipe and comparison criterion.
4. **Audit second-order explanations.** Curvature-aware one-step directions can
   look excellent without producing better sustained training. Sample-dependent
   curvature, line-search rules, and co-adaptation of the state to its optimizer
   changed several initial interpretations. Some finely resolved spectral-band
   claims were explicitly withdrawn when the numerical approximation could not
   resolve those bands.
5. **Study batch and momentum.** The reference records show larger geometry
   benefits at larger batch. Shorter momentum helps the whitening methods early
   in the largest-batch runs, but the extra endpoint gain largely attenuates at
   twice the horizon. The supported target is therefore a fixed-token
   batch–momentum interaction with phase dependence. An isolated, universal
   batch-only mechanism is not established.

The [reference state](../RESEARCH_STATE.md) and dated entries in the
[reference journal](../research/adamw_spectra/MUON_CASE.md) retain the results,
failed predictions, and corrections. Main-study live allocations are managed
separately; this review does not start or change them.

## Why the small study exists

Before pursuing a simpler or substantially cheaper combined optimizer, the
current goal is to reproduce the important reference phenomena in a small
standalone model: the five-method performance order, growing geometry advantage
with batch, and a shorter useful momentum window at larger batch. All new GPU
work here uses devices strictly below 48 GB. The
[local protocol](PROTOCOL.md) contains the original criteria and each decision.

| Candidate | Development program | What survived | What failed |
|---|---|---|---|
| Character GPT, 820,864 parameters | 75 runs on 16 GB A4000s | Growing endpoint SOAP-PD/Muon gaps | Window-dependent TS/PD ranking; TS beats SOAP-PD at the shared lower momentum; weak momentum-optimum shift; common-loss crossings too coarse |
| TinyStories BPE GPT, 1,852,800 parameters | 19 runs on 11 GB RTX 2080 Ti | AdamW → Muon → PD → TS improvements across all four fixed document groups | SOAP-PD does not beat TS on full development; AdamW's LR bracket remains open |
| FineWeb miniature D, 4,861,056 parameters | 36 runs complete on 16 GB A4000s | Desired ordering holds in the two-seed mean | SoapMuon+PD/TS reverses in one seed at the final joint rates; mean lead 0.00369 misses 0.005; SoapMuon+PD LR bracket remains open |

All three candidates are **unqualified** under their original ordering criteria. Their artifacts are preserved in
[the character verdict](logs/tiny_spectra/CHARACTER_CANDIDATE_VERDICT.json) and
[the TinyStories verdict](logs/tiny_spectra/CANDIDATE_C_VERDICT.json).

A subsequent [recipe audit](logs/tiny_spectra/recipe_fidelity_20260928/README.md)
found matching core optimizer algebra but different estimator clocks, sampling,
exposure, and batch relative to model size. The separately declared two-arm
input-clock diagnostic was neutral: TS improved by 0.00051 NLL and SOAP-PD by
0.00041, failing both material-improvement criteria. Its
[complete readout](logs/tiny_spectra/c_clock_diagnostic_20260928/report/README.md)
closes that hypothesis at this setting.

The six exposure-and-batch arms are now complete. SOAP-PD−TS endpoint differences
at LRs 0.004/0.008/0.016 are −0.00146/−0.00104/−0.01136. The largest-rate lead
holds across all four document groups and the fixed late-training interval.
Nevertheless, the [prospective diagnostic gate failed](logs/tiny_spectra/c_scale_diagnostic_20260928/report/README.md):
both methods' minima are at the upper grid boundary, and only one matched rate
shows a material advantage. The diagnostic is closed without grid expansion.
This joint regime test cannot identify separate batch and exposure effects.
It uses all current fresh training data, so a future doubled-horizon test would
need a separate training-data/control plan. The five-method, batch, momentum,
and independent-confirmation requirements remain incomplete.

The fixed 0.016 recipe was then tested on exactly two new development seeds.
SOAP-PD−TS was −0.00517 and +0.00022; their mean was −0.00248, excluding the
selected original seed. The declared replication requirement failed, and the
fixed-recipe lead is closed without additional seeds or recipe repairs. The
[replication report](logs/tiny_spectra/c_fixed_seed_replication_20260928/report/README.md)
retains both seeds, every fixed document group, and all late-trajectory reversals.

Candidate D tests a miniature closer to the reference architecture and task:
eight layers, width 128, two heads, context 512, and a training-only 12,576-token
BPE vocabulary. Its fixed horizon is 96,242,176 fresh training targets. Data
qualification established 333,153,280 available fresh targets, enough for the
declared horizon controls. All five native optimizer implementations passed a
21-update numerical qualification on a 16 GB A4000. A separate CUDA graph speed
attempt failed its fixed numerical agreement test and remains closed.

The user accepted the native per-run cost and requested the optimizer ordering
check with more parallel GPUs. The [completed screen](logs/tiny_spectra/fineweb_d_ordering_20260928/report_final/README.md)
contains 30 initial runs and six predeclared boundary checks, using 22.1128
GPU-hours on 16 GB A4000s. One rate per method was selected jointly across both
seeds using every development target. Mean NLLs follow the desired order:
AdamW 5.130812, Muon 4.564518, PD 4.516786, TS 4.495898, and SoapMuon+PD 4.492208.
However, SoapMuon+PD minus TS is +0.00312 and -0.01050 in the two seeds. Its mean
advantage 0.00369 misses the original 0.005 criterion, and its LR remains an upper
boundary. Candidate D is closed without robust ordering qualification. The
saved common-loss curves also show only a small, incompletely resolved final-pair
rate advantage. All prior numerical, resource and scientific failures remain
recorded.

The separately reviewed [fixed-recipe batch comparison](logs/tiny_spectra/fineweb_d_batch_20260928/report/README.md)
is complete: 12 new runs plus six reused midpoint trajectories, using another
7.7983 GPU-hours. It compares Muon at LRs 0.01 and 0.02 against SoapMuon+PD at
0.02, with two paired seeds and the same model, stream and target exposure.
The endpoint advantage increases with batch against both controls in both seeds.
The common-loss growth criterion remains unresolved by evaluation spacing, and
midpoint monotonicity fails against Muon 0.01. Its original qualification gate
therefore failed. Neither the endpoint trend nor a later momentum result can
replace that failure. The requested [momentum and horizon-control study](logs/tiny_spectra/fineweb_d_momentum_20260929/report_final/README.md)
is now complete: 44 new runs and both automatic CPU analyses finished
successfully, using 31.6628 additional GPU-hours. Both Muon and SoapMuon+PD pass
the frozen fixed-rate momentum-interaction criterion. Momentum 0.8 improves
large-batch loss more than small-batch loss at both rates and in both seeds.
Mean small/large gains are 0.000950/0.041436 for Muon and 0.009200/0.039146 for
SoapMuon+PD. A global optimum or consistent preference reversal is not established.
Matched128-update interactions change sign across seeds for both methods.
Doubling large-batch exposure reduces SoapMuon+PD's gain at the selected common
control LR from 0.049389 to 0.021542; Muon has mixed seedwise attenuation.
This reproduces a fixed-token momentum interaction with phase dependence.
Robust five-method ordering, resolved common-loss batch growth and independent
confirmation remain unfulfilled. No owned job remains live and all independent
panels remain sealed.

## Evaluation overfitting: what is protected and what remains unproved

The selected-window TS/PD reversal is a reproducible sensitivity to the
evaluation sample. Tiny numerical differences do not explain it. It does not
establish a common cause with the separate gradient-sign findings.

All inspected validation scores are **development evidence**, including full
validation and fresh training seeds evaluated on the same text. Full-development
selection prevents choosing a favorable window subset, but it does not undo
adaptive selection over many experiments. The absence of a late training-loss
overfit flag also cannot establish absence of evaluation-selection overfit.

Three document populations were reserved before model scoring: 28 previously
unused complete MIT Shakespeare plays, 5,018 separate TinyStories documents,
and 1,497 FineWeb documents for Candidate D. None has been model-scored.
The original character corpus's nine represented
works were excluded wholly from the play panel. TinyStories train/development/
test assignment and overlap checks preceded train-only tokenizer fitting.

Confirmation must lock the complete recipes, seed family, horizons, checkpoints,
loss thresholds, and uncertainty analysis before training and before any test
score. At least three fresh paired training seeds are required. All declared
results must be released together; ties, reversals and censored crossings remain
visible. Test outcomes cannot choose new hyperparameters or document subsets.
Document variation and training-seed variation are distinct sources of
uncertainty; thousands of windows are not thousands of training replications.

The existing character confirmation code rejects early scoring, changed
configurations/data/sources/weights, partial checkpoint families, and partial
result collection. Thirteen focused synthetic confirmation and story-masking
contracts passed during this review. No real test data entered those tests.
The BPE confirmation adapter is now implemented and independently reviewed:
53 synthetic character/BPE/FineWeb/binding contracts and an earlier actual-loader
synthetic integration passed. The final scientific analysis specification and GPU evaluation
qualification remain outstanding. The code checks a declared family and now binds each canonical panel to one
lock and complete weight family, with attempt history anchored to the panel.
Scientific completeness still requires the protocol and review. These safeguards support a future credible test; they are
not evidence that the current results already generalize.

## Isolation

All new source, results, data, decisions, plots, and scratch work are in this
directory. Historical paths have compatibility links so frozen artifacts remain
usable. The shared Python environment is read-only; `./run` redirects temporary
files and caches here. See the [relocation evidence](migration/relocation.json).

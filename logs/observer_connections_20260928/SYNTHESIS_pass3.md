# Observer synthesis: which signal does an optimizer need?

2026-09-28, updated after the third pass. This is an intermediate result in
the open observer goal. CPU only; no training, GPU use, or main-study changes.

## Third pass: qualify the route, retain the quieter movement

The actual value displacement has now been scored in constant and centered
functional components on two fixed held-out banks. The extra constant benefit
after the centered step changes sign between banks. There is no repeatable
PD-specific local advantage to justify another centering/bias intervention.
The predictive GN cross term explains the positive finite interaction closely;
the two components have almost identical GN correlation (~.56) under Muon
and PD. See `value_split/REPORT.md`.

The saved-weight contractions also expose a necessary invariance correction.
Larger V-coordinate means do not necessarily mean larger functional routes:
an exact within-head V/O change of basis preserves the model function. After
O, the shared constant branch at step500 has summed energies93.96/101.24/
122.44/74.98 for M/PD/S/SPD, removing the old universal amplitude ordering.
Smaller actual joint V/O perturbations survive: PD's energy is5.2× smaller
than Muon's, and SOAP-PD's is9.7× smaller than SOAP-Muon's. These are fixed-mean,
own-state descriptors, not rates. See `value_gauge/REPORT.md` and the exact
counterexample in `value_gauge/IDENTITY.md`. Earlier amplitude claims below
refer to internal coordinates and are superseded as functional interpretations.

The small direction-Gram archive supplies a broader explanation of positive
interference. Its leading normalized mode spans many matrices and closely
aligns with descent slopes; it is repeatable across score halves. It does
not identify a disposable mean channel or the full parameter GN spectrum.
Prior block-GN/deflation/staging results remain constraints. See
`coupling_archive/REPORT.md`.

This closes the fixed architectural premise check without declaring the route
useless. The next distinct question is output-curvature estimator repeatability:
TS used a fixed small local sample while SOAP used global gradients. That
unmeasured estimator difference remains a plausible, falsifiable explanation
to examine before a new optimizer prescription. Current resource/decision
details are in `STATE.md`.

## Follow-up that changes the initial picture

The initial premise below has now been tested further. Fixed independent
input-rank groups show positive weak-tail alignment after one step, but dense
lag measurements change sign within a few steps. The tail is not a steady
slow gradient. See `PERSISTENCE_FINDINGS.md`.

A new architecture connection appears in the learned value projections:
relative gain on the shared input mean versus centered variation ends at
.077 for Muon, .446 for PD, and .603 for SOAP-PD. The larger mean route is
therefore not just a larger input mean; the learned value map treats it
differently. Attention preserves a constant value component exactly, but
data-dependent selection also changes its measured mean. Local usefulness
is still unknown. See `attention_geometry/NOTE.md`.

A controlled80-second CPU experiment then resolved a specific premise.
At Muon4M183→184, body-only and full updates both have positive same-bank
but negative cross-bank body-gradient alignment. Sample-specific components
explain the sign much better than auxiliary scope. Yet a separate finite-step
loss interaction is positive on all16 measured sequences: the auxiliary
step helps alone but makes the joint step worse than body-only. These are
distinct findings, not a mandate to freeze auxiliaries or alter training.
See `body_aux/REPORT.md` and independent review in `persistence_peer/`.

The original first-pass synthesis is preserved as `SYNTHESIS_pass1.md`.
The current next decision is in `STATE.md`: independently score the actual
value update's constant and centered functional components before proposing
a new parameterization or optimizer change.

The most useful connection across the saved work is that **large raw gradient
components, persistent gradient components, and components emphasized by the
actual update are three different objects**. Several current explanations
implicitly identify them. The archive already gives reasons to separate them.

## Evidence that changes the next question

### Raw signal does not describe where the update spends its energy

The new retrospective analysis reads twelve saved frame tensors, covering
Muon, PD, SOAP-Muon and SOAP composed with PD at steps 200/500/900. These are
1M-batch trajectories with independently fitted Kronecker bases and 8192
measurement sequences. Estimated high-SNR coordinates hold >99% of raw
positive signal energy, yet their complement holds 69–87% of body-displacement
energy. The complement also has positive measured slope in all 72
method/kind/checkpoint groups, though the mask-selection dependence prevents
treating those signs as independent statistical tests.

This makes the weak coordinates worth understanding. It does not establish
that they are noise, that all their movement is useful, or that their energy
explains the optimizer ranking. Noise can matter after normalization even
when it barely changes a raw-energy summary.

Separately, the newer SNR probe's estimator has a strong null effect. With
zero-mean Gaussian noise it produces 98.69% and 99.91% estimated signal
energy above SNR=1 at extrapolated 4M and 16M batches. Analytic calculation,
two million simulated entries, and an independent derivation agree. This
invalidates neither the measured training gains nor all gradient signal;
it means that particular statistic cannot resolve the mechanism.

Read `SIGNAL_NOISE_FINDINGS.md`; data in `saved_frame_reweighting*.json`;
calibration in `signal_noise_audit.json`; independent reviews in
`signal_noise_peer/`. The plotted null is `signal_noise_null.png`.

### An output-curvature comparison is also an estimator comparison

The frozen TS implementation estimates each output factor from eight local
sequences per refresh, not from the whole training batch or all ranks. Its
nominal stationary weight ESS is 72 sequences at every batch. SOAP's Gram
uses full globally averaged gradients, whose sampling error falls as batch
grows. Therefore the finding "this SOAP recipe beats this TS recipe at 16M"
does not by itself establish an intrinsic superiority of gradient statistics
over curvature statistics.

There is also useful negative evidence. Input C's clock is largely constant
in tokens, and at 16M TS's factor age is comparable to SOAP's, so simple
stories about a hidden 16-fold C acceleration or uniquely stale B are not
supported. A small/disjoint versus pooled B direction-repeatability check
would isolate the sampling question before another training comparison.
Its outcome is unknown.

Read `estimator_clocks/NOTE.md` and `TABLE.md`. Twenty-seven frozen-source
hashes were checked against nine completed runs.

### Local model improvement survives, but the flat-band explanation is weaker

Across 37 saved states, total step slope and curvature are reproduced to
about 1e-6 relative error. Those are well qualified measurements. A band
split built from 48 Lanczos nodes is a split into feasible Krylov vectors,
with valid within-subspace quadratic scores; it is not an exact projector
onto the full GN spectrum. The inferred flat-band length multiplier still
changes considerably as the prefix grows. Flat c*>1 survives prefixes
32/40/48, so the local opportunity should not simply be dismissed.

More directly, its relative slope fraction fails as a universal training
ranking: at 16M, TS has a larger flat fraction than the better-training
SOAP-PD, and fresher SOAP-PD momentum improves training while reducing that
fraction. Its absolute total model quality, however, improves by about
45–49%. That result does not need a spectral-band interpretation. Ratios
and numerators need to stay visible together.

Read `spectral_claims/NOTE.md` and `TABLES.md`. All 76 audited inputs were
hash-checked unchanged. This audit does not claim full48 is wrong, or that
one-step model quality establishes a training rate.

## A coherent hypothesis, and what would refute it

A candidate picture is that whitening reduces coupling to large oscillatory
gradient components and increases the relative importance of numerous weak
components; successful temporal averaging then needs to distinguish their
sampling noise from their useful persistence. The current SOAP mechanism
could involve signal equalization, noise control, temporal effects, or a
combination. This hypothesis connects the spike, preconditioner, batch and
momentum observations, but does not uniquely predict them yet.

An immediate falsifier of the simplest version would be weak-coordinate
groups that have no independently measured persistence or descent after the
actual update weighting is applied. Conversely, a stable population of weak,
persistent, productive groups would make a raw-energy-based dismissal of
noise control especially implausible. Neither outcome alone establishes a
new optimizer.

## Initial next analysis from the archive (now completed)

Use `persistence/*.pt` internally, without subtracting vectors from separately
measured eigenframes. Those files contain independent sequence sets for
the base and later states in the same saved measurement frame. Group entries
by predeclared input/output rank bins defined from the separate basis data;
retain signed cross-time products, debiased norms and cancellation warnings.
Relate those coarse groups to the existing frame-displacement energy by
matching rank groups only, not signed vectors. The groupwise association is
descriptive across probes/own states, and eigenbasis ambiguity is retained
as a limit. Start with Muon and PD at step500, interpret it, then decide
whether extension to200/900 is warranted. CPU reads only, <=2 threads, no
model calls, and a projected bound of two minutes.

The reason to do this next is substantive: does the portion of the model
movement hidden by the raw-signal summary track persistent structure, or
mostly transient/noisy structure? The previous analysis already establishes
that this portion is large enough to matter. It avoids proposing a method
from a measurement failure alone.

## Status and limits

The observer goal remains active. No robust optimization/architecture
improvement or novelty is claimed. The persistence analysis and CPU
counterfactual described above have completed. The separate pre-existing observer notebook
continues to hold the body–auxiliary premise-check proposal. Its files,
the main code, research state, protocols and scheduler queues remain untouched
by this pass. All new scripts, plots, data and notes are under this directory.

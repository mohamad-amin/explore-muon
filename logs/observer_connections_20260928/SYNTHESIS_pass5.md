# Observer synthesis: which signal does an optimizer need?

2026-09-28, updated after auxiliary attribution and the new momentum-map audit.
This is an intermediate result in the open observer goal. CPU only; no
training, GPU use, or main-study changes. The previous synthesis is preserved
in `SYNTHESIS_pass4.md`.

## Current synthesis: shared progress and differential feedback

Two new results separate questions that were being conflated. The full
body/head/embedding/norm factorial attributes most positive joint-step loss
interaction to embeddings and head, then explains96–98% of it through the
overlap of their separate prediction changes. The mixed representation/readout
term is small at this selected checkpoint. A bank with no input rows absent
from the next training batch still has the embedding interaction, excluding
absent-row momentum as its explanation. The special-channel-removal premise
is closed; see `aux_partition/REPORT.md` and `embedding_clock/REPORT.md`.

That result gives the earlier coupling/value observations a direct finite-loss
meaning. Multiple groups can move toward the same correction and incur a
positive curvature cross term. Removing overlap may remove useful progress:
the post-hoc embedding projection retains55.5% of response energy but only
7.5% of label-linear descent. This is not a tested parameter update and it
does not argue that all coupling is beneficial. It makes reducing coupling
an insufficient objective.

The new momentum profiles meanwhile contain unused same-state map comparisons.
At all18 declared16Mstates, PD maps of the identical lagged momentum suppress
top16Ritz energy much more than ideal Muon maps. Independent-held curvature
per squared norm is also lower everywhere. This is a stronger operator-level
observation than comparing each method's step at its different trained state.
The successful geometry cannot be dismissed as co-adaptation alone.

But the logged M_s is not the input to the actual s→s+1 update. Its Muon/PD
maps are mostly uphill, while the actual fresh-gradient next steps descend.
The saved positive squared-slope quality hides that distinction. Polar itself
also reduces the raw momentum's stiff fraction substantially. The supported
story is extra covariance-dependent spatial redistribution, not an absence
of spatial redistribution under Muon.

Most consequentially, a quiet output in a stiff direction does not measure
the response to a perturbation in that direction. A fixed-norm3×3 example has
zero stiff output projection and arbitrarily large polar derivative. The
actual local feedback object is the joint weight/momentum update Jacobian,
which involves Dphi(M_plus) times the derivative of the incoming gradient.
Clipping, per-matrix norm matching, moving roots and SOAP state matter there;
replacing the gradient derivative by GN is an approximation. The exact
counterexample and source-faithful simplified Jacobian have passed numerical
qualification and independent review in `momentum_maps/`.

This is a falsifiable connection, not a new optimizer discovery: whitening
may permit shorter averaging by reducing *differential* feedback gain while
preserving useful response. Current scalar profiles show output suppression
but cannot establish that causal premise. Auxiliary overlap constrains what
must be preserved if one eventually changes the response. Neither result
licenses removing stiff components, freezing an auxiliary group, or tuning a
local step to a one-step diagnostic. Future work must add genuinely missing
dynamic information or move to a distinct promising artifact, rather than
repeat static energy comparisons.

## Fourth pass: distinguish parameter variability, prediction shape and amplitude

The output-factor sampling premise has now been tested. Four fresh8-sequence
factors produce TS map cosines~.70; repeated labels on identical inputs give
~.74. Pooling helps. Yet the retained directions have predictive-GN cosines
~.99 on independent inputs. The final inverse-root map, not NS alone, expands
their parameter differences. A large Euclidean discrepancy was therefore
insufficient evidence of an optimization bottleneck.

There is a second trap in the reassuring result. At matched parameter norm,
the PD reference produces a functional response2.27–2.41× larger than TS's.
The small GN variation-to-TS–PD ratio (<.006) partly uses this amplitude as its
denominator. Equalizing functional amplitudes post hoc leaves relative shape
uncertainty~.17–.29, not zero. Those extra normalizations were not training
experiments or replacement endpoints. Preserve all three views: weight-space
variation, prediction-space variation, and response magnitude. See
`ts_repeatability/REPORT.md` and `run2/metric_comparison.png` within it.

This closes the immediate covariance-sampling branch without declaring its
noise harmless. A known Fisher half-factor/Rademacher alternative was examined
algebraically and in the literature, but not run. Improving raw estimator
variance alone would not establish a useful nonlinear optimizer change.

Independent scalar reading also matters. The completed clipping control does
not reproduce the whitening methods' momentum benefit. Shorter momentum helps
early in an older4M run but loses by its endpoint, and its advantage narrows in
the newly completed longer16M run. Flatter loss curves account for part of this;
do not confuse loss-gap attenuation with an equal loss of speedup. Records and
source checks are in `clipping_connection/COMPLETION_ADDENDUM.md`.

The new tiny-character artifacts provide a separate sampling example: the
selected validation windows and their complement favor different methods,
verified from saved individual losses. This says where the reversal comes
from, not that it shares a physical mechanism with gradient noise. The old
scores are now development evidence and the main candidate failed its full
qualification. Its new confirmation data remain untouched by this observer.

The next distinct question is attribution of the measured body–auxiliary
finite-loss interaction. It is a more direct observed effect than the abandoned
instantaneous noise-bottleneck story; it still must be separated from generic
useful descent overlap. Current next-step boundaries are in `STATE.md`.

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

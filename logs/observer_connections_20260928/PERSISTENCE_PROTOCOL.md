# Fixed-group persistence: decision argument

2026-09-28, before analysis. Previous goal turn was progress: null calibration,
twelve-frame update reweighting, independent numerical review and two source
audits changed what the existing mechanism claims establish. The present turn
tests the open connection rather than repeating that audit.

## Missing premise

Estimated weak-SNR coordinates contain most actual body-displacement energy.
Their measured weak-group slope is positive but mask-selected on the same
gradient data. We do not know whether coarse independently specified groups
that contain much displacement have persistent expected gradients.

## Hypothesis and alternative

Whitening changes which gradient components the model follows, making weak but
persistent components relevant even when they contain little raw signal energy.
Alternatively, the large tail displacement primarily reflects the dimension
of those groups and contains transient/noisy components. These are not
exhaustive or mutually exclusive; the measurement is descriptive.

## Bounded comparison

First read the Muon and PD step500 `persistence/` tensors. Use the same frame
within each tensor, with a separate sample A at base and sample B at later
states. Define groups before examining values by input/output eigenvalue ranks
`[0:1], [1:8], [8:64], [64:end]`; optionally marginalize those fixed groups.
Pool across layers by matrix kind and across the whole body. Preserve signed
cross-products, debiased squared norms, noise corrections and group dimensions.
Compare group energy in the independently measured saved `frame/` displacement
using only coarse rank groups; do not mix signed vectors across those artifacts.

Report both raw and noise-corrected cosines, with correction-dependent values
flagged. Do not clip cross-products or negative group signal estimates. No
significance claims across coordinates, groups or lags. Record displacement
energy divided by coordinate share to distinguish enrichment from dimension.
Keep future-state sample reuse, approximate sequence independence, unsaved
eigenvectors and body-only measurement scope explicit.

## Interpretation and next decision

A positive cross-time signal in tail groups supports persistence at that lag,
subject to data/uncertainty qualifications. A negative cross-product supports
an opposed component, not by itself stochastic noise or period-two dynamics.
Tiny or correction-dominated group norms make persistence unresolved. Failure
to support the premise is retained and does not trigger an intervention.
Inspect the two states before extending the fixed analysis to steps200/900.

CPU only, two numerical threads, two persistence and two frame reads (about
2.5 GiB), expected under two minutes; no model, gradient or optimizer calls.
All new outputs remain in this observer directory. No main-study source or
protocol is changed. Independent peer discussion is in `persistence_peer/`.

## Extension after the step500 result

At step500, whole-body next-step gradient correlation is negative (Muon
-.198, PD -.251), concentrated in the top input mode. The rank65+ input
tail is positively aligned (.171/.292), with group trace SNR near one,
and contains 87.5%/97.1% of actual body-displacement energy. Five of six
Muon and all six PD matrix-kind tail groups have positive cross-products;
Muon V is the exception. These tail groups also contain90.6% of coordinates,
so energy share alone is not strong enrichment. The coherent input split
justifies the pre-specified extension to saved steps200/900 using exactly
the same grouping and noise-sensitivity rules. No new thresholds selected.

## Dense-lag extension

The extension confirms a positive input-tail next-step product at all six
states, though Muon's step900 value is small. PD's tail correlation is
.267/.292/.341 at200/500/900 versus Muon's .252/.171/.021. This is one seed
on different own trajectories, not causal proof. Next examine the existing
dense-lag reruns based at501, using the same four input rank groups and
signed products at their actual lags1,2,3,6,7,14,15,30,31. The question is
whether one-step persistence survives anywhere near the19-step momentum
mean age, or whether it mainly disappears after a few steps. These are fresh
A4000 reruns, not the same states as the preceding probes; no displacement
artifact is available in their exact basis, so no energy association is
imported. Whole-run horizon stays1469; stop-after532 does not alter schedule.
Two saved1.9GiB tensor reads, CPU only, two threads, expected under one minute.
Retain per-layer products to check whether a pooled sign is broadly shared.

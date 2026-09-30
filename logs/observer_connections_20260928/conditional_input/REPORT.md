# The disjoint-gradient cancellation test is not qualified by the retained provenance

2026-09-28. The proposed two-state saved-gradient test stops at its
historical sample-membership gate. No real gradient tensors were numerically
compared, no predictor was computed, and no model, optimizer, GPU or
training job was run. This is a scoped missing-evidence result, not a
refutation of the cancellation hypothesis or a blocker of the observer goal.

## Question and why the gate matters

The candidate asks whether adding the same opposing momentum history to
two gradient estimates makes their remaining inputs much less repeatable
relative to their norms. It connects the earlier raw-signal/update-weighting
and fixed-weight replay findings, without another parameter-group ablation.

For fixed history H, the absolute difference is unchanged:

    (H+g_A) − (H+g_B) = g_A − g_B.

Any effect must concern relative magnitude or direction; history does not
create new absolute noise. Conditioning on one saved history also does
not measure the unconditional variance of a stochastic momentum process.

The proposed initial pair uses g1M and the complementary part of g4M at
the original Muon/PD step-500 states. The identity
g_B=(4*g4M−g1M)/3 requires the 1M sample to be exactly the first quarter
of the 4M sample, with identical state, sequence boundaries and loss units.
The retained labels by themselves cannot establish that contract.

## What is supported

Matching GN2 result files and execution logs identify the two intended
training arms, step 500, and input labels g1M/g4M/momentum. The training
metadata and frozen sources establish beta .95, plain momentum, clipping
threshold 1.0, and the recurrence

    M_next = .95 * M + clipped_parameter_gradient.

There is no `(1-beta)` EMA multiplier. Body parameters are placed first
in the optimizer, in the model's order; the probe helper enumerates the
same Q/K/V/O/up/down order. Actual binary buffer ownership was not reread
after the earlier gate failed. Ten relevant frozen source comparisons
match their metadata; details are in `source_qualification.json`.

The inspected diagnostic code computes mean-token body gradients, stores
`gd=-b`, constructs `.95*M+fresh`, and rounds retained directions to BF16.
Its current g1/g4 calls use 2048/8192 sequences at the same validation-stream
offset. Those semantics support the intended design conditionally. Full
model clipping cannot be reconstructed from body-only raw gradient records
without auxiliary gradient norms. Even a future qualified calculation
would therefore need the explicit conditional-unclipped scope unless that
missing information were supplied.

## What remains unqualified

The two gradient archives completed around September 27, 04:18 CDT. The
only located producer source was edited at 04:55:59, and its bytecode matches
that later source. The notebook records subsequent feature additions.
There is no matching executed-source snapshot/hash or complete sampling
command in the scoped launch records; the launcher log is empty. The
diagnostic outputs retain arm/step labels but not sample offsets/counts,
input hashes or dependency identities. The workspace has no Git history.

The data-reader bytes and older helper timestamp provide useful corroboration,
but do not prove the historical producer's call arguments. The independent
review therefore leaves exact g1/g4 nesting unresolved. This does **not**
assert that the archived gradients are wrong or that the batches were
nonnested. It means the proposed disjoint-complement interpretation has
not met its declared gate. See [PROVENANCE_REVIEW.md](PROVENANCE_REVIEW.md)
for precise paths, timestamps, current hashes and gate-by-gate limits.

## Precision and conceptual boundary

BF16 storage would require more than checking a nominal dtype. Rounding
intervals must use neighboring BF16 values, including asymmetric spacing
at powers of two. Intervals for a verified complementary mean would be

    [(4*L4−U1)/3, (4*U4−L1)/3].

Adding the same exact conditioned history shifts both intervals. Relative
disagreement and angle bounds must retain possible zero residual norms;
no epsilon or positive clipping can repair an unresolved denominator.
BF16 intervals do not certify every preceding FP32 accumulation error.
The [independent precision note](PRECISION_REVIEW.md) concerns that
mathematical design only. Its three fixed synthetic examples (6,000 latent
interval draws) pass the proposed norm/dot/cosine containment checks and
preserve zero-norm ambiguity. These are arithmetic checks, not empirical
gradient or momentum results.

The useful conceptual distinction survives the failed archive gate:
repeatability of a raw gradient is not automatically repeatability of the
quantity a stateful, nonlinear normalized optimizer transforms. That is
an open hypothesis motivated by several artifacts, not a discovery of its
size or role in these models.

## Decision

Do not execute the proposed de-mixing or cancellation statistics from these
records, quietly replace them by an overlapping comparison, invert a polar
direction to fabricate the missing gradient, or run new model gradients
to repair this archive. A contemporaneous execution/sample record could
reopen the specific gate. Preserve the source uncertainty as uncertainty.

This closes the bounded provenance search and leaves the empirical
conditional-input hypothesis untested. Continue interpreting the qualified
trajectory, functional and geometric evidence without collapsing their
different objects into one local score. In parallel, the newly completed
main PD warmup transfer is direct training evidence and is reviewed
separately in `../warmup_transfer/`; it does not resolve this sampling issue.

Evidence: [qualification protocol](PROTOCOL.md),
[historical review](PROVENANCE_REVIEW.md), `source_qualification.json`,
and the independent [precision review](PRECISION_REVIEW.md). No new optimizer
or architecture improvement is claimed by this pass.

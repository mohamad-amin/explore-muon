# Reference-directed miniature: proposal for independent review

Status: **unreviewed; no data preparation or training authorized by this file**.
The requested independent discussion failed because the agent service reported
an account usage limit. This proposal preserves the full research objective and
all previous negative results. It does not reopen the closed TinyStories lead.

## Question and competing explanations

The character and TinyStories candidates changed depth, context, vocabulary,
data complexity, statistic clocks and optimizer regimes relative to the
successful reference. Their failures do not establish that a small transformer
cannot reproduce the requested phenomena. A direct alternative is to reduce
width and vocabulary deterministically while retaining more of the original
architecture and task.

The hypothesis is that this package retains enough of the reference learning
dynamics to support the five-method order and batch/momentum effects. The
competitor is that those effects need scale or other conditions that still do
not survive the reduction. Several conditions change together, so this would
test transfer of a package, not identify the cause of the previous failures.
There is one proposed architecture, with no width/depth/context search.

## Concrete architecture and budget mapping

| Quantity | Reference | Proposed miniature |
|---|---:|---:|
| Layers | 8 | 8 |
| Width | 512 | 128 |
| Attention heads | 8 | 2 |
| Head dimension | 64 | 64 |
| Context | 512 | 512 |
| Vocabulary | 50,304 | 12,576 train-only BPE entries |
| Parameters | 76,948,992 | 4,861,056 |
| Base batch | 4,194,304 targets | 262,144 targets |
| Base exposure | 1,539,870,720 targets | 96,242,176 targets |
| Updates | 368 | 368 |

Width and vocabulary each decrease fourfold. Batches and exposure decrease
sixteenfold, with exposure rounded upward to a complete 512-target context.
This preserves the reference update counts at all three scaled batches:
65,536 / 262,144 / 1,048,576 targets give 1,469 / 368 / 92 updates.
The miniature has 19.80 rather than 20.01 targets per parameter because position
and normalization parameters do not scale quadratically. Its auxiliary share
is 67.64%, close to the reference. The final midpoint update has 35,328 targets.
These are arithmetic properties, not demonstrated dynamical equivalence.

Keep the reference's no-bias RMSNorm/QK-normalized frontend and initialization.
The proposed midpoint uses momentum 0.9, input/output powers 1/4, SOAP beta2 0.9,
13 warmup updates, 10% cooldown, clipping 1, decay 0.01 and common auxiliary
AdamW LR 0.002. Auxiliary AdamW betas remain (0.9, 0.95).

## Estimator choices requiring review

A four-sequence physical microbatch gives 128 microforwards per midpoint
update, equal to a reference owner's count. Using the original per-forward
covariance retention 0.998 and stride 32 would therefore recover the same
0.998^128 midpoint retention. Rows per input dimension also approximately match.
This requires an explicitly versioned per-forward collector mode; the completed
tiny studies' pooled-per-update mode must remain unchanged in frozen artifacts.
The original mean retention is 0.99 per forward. Input roots refresh every ten
midpoint updates. Statistics collection must remain disabled during evaluation
and output-GN sampling.

The reference uses eight 512-token sequences for an output-statistic update.
Two such sequences in the width-quarter model approximately preserve samples
per output dimension. That is a proposed estimator mapping, not a measured
equivalence; keeping eight is another scientifically different choice. The
independent review must settle this before preparation or model scoring. Output
EMA 0.8 and ten-update midpoint refresh would remain fixed. No clock or sample
budget is to be chosen from the resulting optimizer ranking.

The full batch/momentum design must also specify how reference statistic-refresh
rules transfer at low/high batch. A batch effect must not silently combine a
change in geometry power with a change in batch. Momentum, adaptive directions,
actual writes, and clipping effects remain separate measurements.

## Data contract and feasibility

The parent project's documented read-only FineWeb alias exposes 33 binary
shards totaling 6,600,033,792 bytes. Headers of two training shards and the
validation shard match the existing loader's format and each records 100M
tokens. This inventory read no unrelated study logs or state and changed no
shared file. It does not yet verify document boundaries, tokenizer provenance,
or a new train/development/test assignment.

Preparation would need to verify the GPT-2 decoder assets and recover complete
documents from the shared token stream, including handling shard-boundary
fragments. The shared environment has no tiktoken/transformers installation;
the private tokenizers 0.23.2 package exists. Decoder assets and any preparation
dependency must stay private to this study. Encoding/decoding must be qualified
before treating the recovered text as the reference task.

Before fitting the new BPE, fix whole-document train/development/test roles,
exclude normalized duplicates and long verified overlaps, and preserve all
membership and source hashes. Existing reference evaluation text is development
evidence. A new untouched test population is necessary for the new vocabulary;
neither existing sealed population may be used for model selection. Preserve
all original datasets, tokenizer files and failed-candidate artifacts.

The new train pool must contain at least 192,484,352 unique target positions for
a doubled base horizon, and more if the largest predeclared matched-step control
requires it. Capacity must be measured after encoding/deduplication. Fix the
source-byte limit before preparation; insufficient capacity stops preparation
for review, rather than silently expanding the data. No count of source GPT-2
tokens is a guarantee about the number of new BPE targets.

A new generic document-token loader/scorer contract is needed; FineWeb must not
be mislabeled as TinyStories to bypass the existing identity checks. Every
evaluation must retain short documents and tails with true token denominators.
The new panel needs the same whole-family lock, one-family binding, source/data
verification, and all-results release protections already tested here.

## Proposed staged commitment and stopping rules

1. Review the package and exact estimator/data decisions. No GPU follows from
   this proposal or from idle hardware.
2. Qualify the data and numerical contracts, then a bounded all-five-method
   smoke on one permitted GPU family, preferably 16 GB A4000 for native BF16.
   Twenty-one updates exercise three ten-update root refreshes. A short smoke
   uses one warmup update to exercise peak rates. Limit it to 20 minutes.
3. Forecast time/memory before a scientific screen. If the forecast does not
   demonstrate a substantially cheaper experiment than the reference, revise the
   execution design prospectively; do not quietly shorten the scientific horizon.
4. The proposed screen gives every method three rates on two predetermined
   development seeds, with at most one factor-two outward rate per boundary
   method on both seeds: 30 initial and at most 10 edge arms, no architecture
   variants. Suggested reference-derived centers are AdamW 0.0012, Muon/PD
   0.01, TS/SOAP-PD 0.005, each multiplied by {1/2, 1, 2}. The Muon-family
   factor-two reduction is a relative-weight-step scaling assumption needing
   review, not a known transfer law. Rates and seeds must be fixed before runs.
5. Select by full-development NLL with equal tuning opportunities. A reversed
   chain, unresolved LR bracket, numerical failure, or lack of fresh-seed
   support remains a failed viability result. No additional rates, architecture,
   corpus or powers follow as repairs under this candidate. Exact viability
   margins and the screen resource cap must be settled in review before launch.
6. A successful viability screen only allows a separately bounded full-goal
   plan: all five comparisons, at least three batches and common-loss progress,
   shorter-versus-longer momentum with fair LR allowance, matched-step/extended
   controls, and at least three fresh paired confirmation seeds. Lock the full
   family and uncertainty rules before training and scoring the new test panel.

The proposed screen is larger than earlier single-seed screens to reduce reliance
on a fortunate development seed. Its compute bound is deliberately pending the
new architecture's numerical qualification; no scientific launch is specified
yet. A strict ordering is not to be declared from a favorable window, checkpoint,
single seed, LR envelope, or a weakened baseline.

## Why not simply choose a batch from gradient noise?

The simplified ratio tr(Cov(g))/||E[g]||² measures relative raw-gradient noise
under a specified sampler. Its relation to an efficient training batch depends
on additional assumptions; the original work separates it from a
curvature-weighted quantity and discusses conditioning and horizon limitations.
[McCandlish et al. (2018)](https://arxiv.org/abs/1812.06162).

Later language-model work reports problems with the noise proxy for Adam and
uses direct branched-training comparisons instead.
[Merrill et al. (2025)](https://arxiv.org/abs/2505.23971),
[authors' methodological account](https://allenai.org/blog/critical-batch-size).

Our inference is that a fixed-state noise audit could test statistical mismatch,
but cannot by itself select a batch or guarantee Muon/SOAP performance. The
reference-directed miniature is an alternative to discuss, not a consequence
proved by those papers. The completed state-availability audit and all previous
failed gates remain unchanged.

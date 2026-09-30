# Fixed-state sensitivity of the two-sided output factor

2026-09-28. Prior goal turn: progress, with the qualified value split, V/O
invariance correction and direction-Gram analysis changing the next decision.
Main jobs were checked at14:13; both large and small GPUs are occupied by the
main program. This observer uses CPU only, at most two numerical threads.

## Missing premise and competing explanations

TS used eight owner-local sequences per predictive-label B refresh, while
SOAP used globally averaged gradients. That estimator difference could limit
TS, or the small estimate could already be adequate after inverse powers and
polar normalization. Matrix/eigenvector disagreement alone does not decide it.

Use TS's own16M state at46, stored momentum M46, alpha=beta=.5, damping=.001.
M46 belongs to the preceding actual step at W45; here it is frozen input at
W46. No descent/rate claim is made. Historical B/L/cached R are absent.
The separate saved input-statistics sidecar is rank0 only and after step46;
use it to construct one fixed R for all matrices. It is a proxy for other
owners and for roots cached from45, not an online replay. Its role is to hold
the input geometry fixed while varying B.

## Fixed data and factors

Four disjoint banks of eight sequences, length512, at never-trained stream
offsets2,700,000,000;2,710,000,000;2,720,000,000;2,730,000,000. Each has one
predictive-label draw. Repeat labels independently on A and B, holding their
sequences fixed. Separate label randomness from the combined sequence+label
comparison descriptively; two repeated banks do not estimate a precise variance
decomposition. Hidden output errors differentiate summed token loss over the
whole sequence, including attention paths. B averages e^T e over every token,
including position0. Evaluation mode prevents activation-statistic updates.

Pool RAW factors before trace normalization: AB, CD and ABCD. AB versus CD is
an independent16-versus16 comparison; each versus ABCD is nested and is labeled
as such. For each bank j, form an independent24-sequence anchor B_-j from the
other three first draws, then compare its direction with the direction from
.8 B_-j+.2 B_j. This is an EMA-refresh sensitivity proxy, not historical EMA
or a reconstruction of its nominal72-sequence stationary information budget.

## Mapping and quantities

All48 body matrices, no selected favorable subset. Use the frozen CPU FP32
five-polynomial NS map, then L NS(LMR) R, per-matrix norm sqrt(min(shape)),
and the recorded shape factor. Exclude LR and decay. Compute one additional
explicit CPU BF16-NS map at pooled32 to show numerical scale sensitivity;
CPU BF16 is not a bitwise CUDA replay. Model forwards/statistics are CPU FP32,
different from the training BF16 forward path, explicitly retained as a limit.

Save raw B, R, retained directions and per-matrix/kind/all-body metrics:
trace, trace-normalized B distance/cosine, L distance/cosine, damping-band
fraction, final direction distance/cosine, distance to PD (L=I), and the ratio

    ||D_i−D_j||² / (||D_i−D_PD||² + ||D_j−D_PD||²).

That ratio places sample variation against TS's added geometry, without
calling either distance useful progress. Keep normalization-induced magnitude
changes and the per-layer shape convention visible. No coefficients are fit
and no method is selected by loss.

## Qualification, cost and reading rules

Before the banks, compare the instrument's one-sequence B with the frozen
output_second_moments function under identical labels; relative errors <=1e−5.
Compare hidden weight gradients reconstructed as e^T x to autograd (<=1e−4),
check finite roots/directions and exact target norms to1e−5. Time the largest
input/output root and mapping plus one forward/backward, projecting all banks,
17 B-map sets and the BF16 numerical control. If projection exceeds15 minutes,
stop with qualification retained. A live wall-clock bound also stops work at
15 minutes; no silent subset selection or criterion relaxation.

Descriptive flags, not statistical claims: all six aggregate eight-bank cosines
>=.99 and independent AB/CD>=.995 means stable at this probe's scale. Any
eight-bank cosine<=.90 with improved independent16-bank stability means material
sample sensitivity. Other outcomes are mixed. EMA-proxy and noise-to-geometry
metrics may qualify either flag. Per-layer heterogeneity is always reported.

Stable fresh estimates weaken this particular noise premise at this state.
Unstable fresh estimates do not establish noisy online TS, because history,
state drift, precision and EMA information are different. Independent held-out
loss/curvature scoring would be needed to claim pooling improves optimization;
it is outside this repeatability run. No training follows automatically.

Independent prospective discussion: `../ts_repeatability_peer/NOTE.md`.
Frozen source/schema audit: `../ts_frozen_audit/REPORT.md`. All generated files
stay here; main sources, protocols, checkpoints and queues remain untouched.

## Execution qualification revision, before any bank measurements

Run1 matched the frozen B function and gradient reconstruction exactly but
stopped at its cost gate:1303s forecast, with all48 factors priced as2048×2048.
Only8 factors have that dimension;40 are512×512. The original run, timings
and executed source are preserved. Run2 refines the estimate by timing each
actual shape and applying a30% margin. All48 matrices, banks, comparisons,
readings and the900s cap remain unchanged. No scientific bank outcome had
been observed before this execution-only revision. An independent peer
agreed with the correction; it is not a reduced scientific scope.

## Prospective comparator clarification during execution

The implementation reviewer identified that the script's provisional flag
compares independent16-bank cosine with the worst of six8-bank pairs. Before
any direction-comparison outputs were examined, tighten the reading rule:
"improved independent16 stability" requires exceeding the MEAN of the six
8-bank pair cosines. This avoids calling improvement merely because a larger
sample beats an extreme pair. The executing source and its provisional flag
are preserved; final analysis will report the stricter rule separately.
The stable criterion, material-sensitivity threshold and all computations
remain unchanged. No result motivated or was used to select this clarification.

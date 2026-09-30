# Precision review for the conditional-input repeatability discriminator

2026-09-28. Read the priority reset and current gradient-source/save code.
No historical gradient, momentum, checkpoint or scientific tensor outcome
was inspected. A small synthetic-only interval qualification is recorded in
`precision_peer_synthetic.py` / `.json`; it uses two CPU threads, no models,
no GPU and no project tensor input. Only this review and that qualification
were written.

**Current execution decision:** the root's bounded historical-source/membership
search did not qualify nesting. Therefore no real-gradient arithmetic is
launched. The formulas and already-completed synthetic checks below are
retained for future independently qualified data; they are not empirical
evidence for this project's conditional-input hypothesis.

**Verdict: the proposed arithmetic can support a robust bounded readout,
provided historical nesting/sign/units qualify independently and BF16
uncertainty is propagated before interpreting cancellation.** The bounds
below cover *storage quantization of the latent FP32 probe outputs*, not all
earlier FP32 accumulation error, full-model clipping, or population noise.

## Fix the object and the exact identity first

Let g1 and g4 denote the underlying FP32 mean-gradient outputs before BF16
storage. The archived `gd` is negative input, so negate stored values and
reverse interval endpoints before calling them gradients. Conditional on
qualified nested membership, define

    gA = g1,  gB = (4*g4 − g1)/3,
    h = the fixed, explicitly defined beta*M_s term,
    bA = h + gA,  bB = h + gB.

The same h must be used in both. Beta is the recorded .95 under the
unnormalized-buffer convention; there is no extra `(1-beta)` multiplier.
Specify whether h is the result of FP32 multiplication or a real/FP64
product. If the intended object is "same exact FP32 beta*M", compute that
representable product once, convert it exactly to FP64, and condition on
those bits. Do not silently change this convention while checking a saved
constructed-momentum direction.

In this mathematical conditional object,

    bA − bB = gA − gB = (4/3)*(g1 − g4).

History changes the denominator or direction, not the absolute incoming
perturbation. Evaluate the difference through the shared gradient expression
and reuse it; subtracting two large h-shifted vectors adds unnecessary FP64
cancellation. An ordinary FP32 implementation of the final additions can
introduce its own rounding, which is separate from this mathematical identity.

Even when sample membership qualifies, gB is reconstructed from two computed
FP32 means. Exact-real averaging would give the complementary 3M block
exactly; finite accumulations need not. Without per-microbatch sums, the
BF16 interval does not certify all prior FP32 summation error. Keep that
qualification rather than calling it a perfect raw disjoint-gradient replay.

## BF16 storage bins: predecessor/successor, not a constant relative epsilon

For a finite saved BF16 value y with BF16 neighbors y− and y+, use the
closed round-to-nearest-even preimage

    x in [ (y− + y)/2, (y + y+)/2 ].

Closed endpoints deliberately include either tie assignment; this is a safe
superset. At powers of two the interval is asymmetric. For example, stored
+1 has [0.998046875, 1.00390625], while stored −1 has
[−1.00390625, −0.998046875]. A symmetric "0.39% of stored value" bound is
conservative for ordinary normals but unnecessarily loose near cancellation.

Crucially, `nextafter` on the FP32/FP64 representation gives the wrong
neighbors. Use BF16 bit neighbors, or a BF16-native nextafter with a qualified
implementation. Signed zero has the common interval ±2^−134; BF16 subnormal
spacing is 2^−133. NaN/Inf fails qualification. Either handle the largest
finite overflow cell explicitly or declare it unsupported before reading
outcomes; the supplied synthetic helper takes the latter conservative path.

The synthetic check uses PyTorch's CPU FP32→BF16 conversion and covers
signed zeros, positive/negative powers of two, exact midpoint ties and
subnormals. This confirms the specified rounding model in that helper; it
is not a provenance certificate for every historical GPU conversion or
upstream floating-point operation. The source must exclude stochastic
rounding or an incompatible storage convention independently.

## Propagate coordinate intervals through the common construction

Write g1 in [l1,u1] and g4 in [l4,u4] coordinatewise. Then

    gA in [l1,u1],
    gB in [(4*l4−u1)/3, (4*u4−l1)/3],
    bA in h+[l1,u1],
    bB in h+[(4*l4−u1)/3, (4*u4−l1)/3],
    d=gA−gB in [(4/3)*(l1−u4), (4/3)*(u1−l4)].

Use outward rounding for composed FP64 endpoint operations. Multiplication
by four is exact when representable, but subtraction and division by three
need directed guards. BF16-to-FP64 conversion itself is exact. The h shift
is conditioned and has no statistical uncertainty; any uncertainty in a
reconstructed historical multiply/add is a separately labelled arithmetic
interval, not uncertainty in the saved momentum bits.

The gA and gB errors share g1 and are correlated. Treating their marginal
boxes as independent in the bounds below is conservative, not an assumption
of independent sampling error. The absolute difference must use the direct
formula above, rather than discarding that dependency. Conservative intervals
may be too wide to decide; that is an allowed precision-limited outcome.

## Norm, dot and cosine bounds

For any coordinate box [l,u], define

    near_i = 0 if l_i <= 0 <= u_i, otherwise min(|l_i|,|u_i|),
    far_i  = max(|l_i|,|u_i|).

Then

    ||v|| in [sqrt(sum near_i²), sqrt(sum far_i²)].

These are the exact extrema over that coordinate box before arithmetic
rounding. A zero lower bound is geometry, not a clipped signal estimate.

For two boxes [l,u] and [s,t], each product lies between the minimum and
maximum of {l*s,l*t,u*s,u*t}. Sum these coordinate bounds to obtain a signed
dot interval [Dlo,Dhi]. For a fixed h, the tighter signed pairing is simply
sum min(h*l,h*u) through sum max(h*l,h*u). Retain these signs and the terms
in ||h+g||²=||h||²+||g||²+2<h,g>; this makes cancellation visible.

If both norm lower bounds are strictly positive, form the norm-product
interval [Nlo,Nhi]. Bound cosine by the minimum and maximum of all four
quotients {Dlo/Nlo,Dlo/Nhi,Dhi/Nlo,Dhi/Nhi}, then intersect with [−1,1].
This intersection is a mathematical domain restriction, not statistical
clipping. If either norm lower bound is zero, the angle is unresolved; do
not insert epsilon or report a precise point cosine as though qualified.

A cheaper alternative is an L2-error-ball bound. If ||e_v||<=Ev and
||e_w||<=Ew about stored centers v0,w0, then dot uncertainty is at most
Ev||w0||+Ew||v0||+Ev*Ew, with norm intervals ||v0||±Ev and ||w0||±Ew.
It is valid but often looser. Choose the algorithm before inspecting the
data; do not escalate to progressively tighter fitting when a gate fails.

Use FP64 reductions with an explicit reduction-error guard. For a naive
sum of n products, a conservative gamma bound of order
`gamma_(2n+8)=(2n+8)*u64/(1−(2n+8)*u64)` times the sum of absolute product
bounds is adequate here; take outward square roots/divisions too. Pairwise
summation is usually tighter but should not be assumed to make every error
exactly zero. Chunking matrices is fine if the final sum's guard also covers
combining chunks. This error is normally much smaller than BF16 storage
uncertainty, but that must remain a numerical statement, not an omitted term.

## Relative disagreement: exploit the shared numerator

Use one fixed, symmetric, dimensionless readout

    R(g) = 2||gA−gB|| / (||gA||+||gB||),
    R(b) = 2||gA−gB|| / (||bA||+||bB||).

It lies in [0,2] for nonzero norm sums. Keep the common absolute difference,
all four norms and their intervals beside it. For the denominator sums
Sg and Sb, report

    K = Sg/Sb.

When the difference is nonzero, K=R(b)/R(g) exactly. Bound K directly from
Sg/Sb; dividing separately widened R intervals throws away the shared
numerator and can create a needless precision failure. K remains a defined
norm-sum ratio when d=0, but then it must not be described as a measured
perturbation amplification ratio.

A robust interval for R(b)−R(g) can also reuse d: bound
`2||d||*(1/Sb−1/Sg)` using ordinary interval multiplication, retaining a
signed reciprocal-difference interval. Do not subtract unrelated fitted
SNR estimates or truncate negative quantities.

## Suggested minimal material reading and stopping rule

The root should fix its choice before loading outcomes. My recommendation
is a whole-body claim only, with fixed per-kind summaries descriptive:

- At **both** named states, the lower bound of K exceeds **2**; and
- at both states, the lower bound of R(b)−R(g) exceeds **.10**.

This asks for at least a doubled relative perturbation and a nontrivial
absolute increase in relative disagreement. It prevents celebrating a
large ratio of two practically vanishing numbers. Report both cosine
intervals but do not choose between a favorable angle and norm criterion
after seeing the states. These cutoffs are materiality choices, not a
population hypothesis test. If used, do not call the raw gradients "clean"
merely because K is large; their own R and cosine must support that wording.

Before that scientific reading require finite qualified bins, intended
nested membership/sign/units, a nonzero lower bound for the common absolute
difference, and positive whole-body lower norms for both conditional inputs.
If those fail, or the interval overlaps a material threshold, report
precision-unresolved and stop this bounded comparison. If robust upper
bounds rule out the declared material reading, report it unsupported at
this scope. Keep mixed-state results mixed; no selection of matrices, new
source window, tighter replay, polar map or new GPU measurement follows.

A positive outcome concerns a **conditional unclipped input** given one
fixed saved history. It does not show newly created absolute noise, iid
sampling variance, unconditional momentum variance, improved prediction
response, a bad Muon map, an optimal beta, or a 16M training mechanism.
The two blocks have unequal sizes and contiguous content, not independent
replicated experiments. Unknown full-model gradient clipping prevents
identifying this object with the actual next training step.

## Saved constructed-momentum cross-check

If historical settings identify `momentum:gd` as −(beta*M+g1), its BF16
interval is a useful unit/sign/source check. Compare interval intersection
with the constructed predictor, including the source's FP32 multiply/add
rounding where relevant. Point equality with the midpoint reconstructed
from separately BF16-rounded g1 is not required. Agreement is not proof of
sample nesting, clipping correctness or population repeatability. It also
does not turn the previously stored lagged M into the next optimizer input.

## Synthetic qualification

`precision_peer_synthetic.py` reads no project tensors. Its three fixed
2D examples verify interval containment across 2,000 latent draws each:

- a cancellation example has raw cosine certified above .984 but conditional
  cosine certified below −.612; the **same absolute difference** is reused;
- a reinforcing-history example has denominator amplification below .504;
- a zero-centered predictor has a zero norm lower bound and returns an
  unresolved cosine rather than an epsilon-regularized number.

All norm/dot/cosine containment checks pass, as do sign/power-of-two BF16
preimage checks. These are tests of the arithmetic and terminology, not
predictions or observations about the two historical model states.

# Independent review: what the saved frame-SNR probe establishes

Reviewed 2026-09-28 for the observer task. Read-only source/artifact inspection;
no training or GPU work. Numerical checks used CPU with numerical thread limits
of two. This note does not change any original result or protocol.

## Judgment

The saved observations support the narrower conclusion that **the particular
estimated raw-gradient energy statistic does not distinguish the SOAP and GN
output frames**. The training ablations separately establish useful output-side
normalization and similar final losses for the tested second-moment sources.
Neither establishes that noise weighting is irrelevant or that signal-magnitude
equalization is the unique mechanism. The hypothesis remains plausible.

There are two distinct gaps: the SNR estimator has a strong positive-selection
effect, and its energy weights differ from the optimizer's weights. Their
combination makes the near-100% energy statistic much less diagnostic than its
plain-language description suggests.

## Exact estimator and a null calibration

`logs/muon_spectra/second_order_audit_20260926/frame_snr_probe.py:104-112`
forms, per entry,

    q = max(mean² - s²/K, 0)
    estimated SNR_B = q B/(s² m)

All three JSON artifacts use K=64 and m=16,384 tokens, so the total mean-estimation
sample M=Km is 1,048,576 tokens. Its claimed 4M and 16M results extrapolate noise
variance; they do not improve the precision of that estimated mean.

For a fixed coordinate with iid Gaussian micro-gradients of true mean zero,
T=sqrt(K) mean/s follows Student t with 63 degrees of freedom. Exactly,

    estimated SNR_B > 1  iff  T² > 1 + M/B.

Consequently this zero-signal coordinate clears the threshold with probabilities
16.22%, 26.78%, and 30.66% at B/M=1,4,16. These are **not false discovery rates for
the real data**: they are an estimator calibration under an explicit reference
model. Calling these thresholded entries "individually significant" suggests a
statistical certainty the threshold does not supply.

More surprisingly, the **estimated signal-energy fraction** also approaches one
under this zero-signal null:

| B/M | Null entries above threshold | Null estimated energy above threshold |
|---|---:|---:|
| 1 | 0.162222 | 0.861086 |
| 4 | 0.267798 | 0.986902 |
| 16 | 0.306587 | 0.999078 |

The right column is the ratio of expected energy sums over many coordinates,
not the mean of a single coordinate's ratio. It is the appropriate large-number
reference for pooled independent null coordinates. Correlated coordinates can
change concentration around this reference.

For reproducibility, with nu=K-1 and c=1+M/B, define

    N(x) = BetaSF(x; 3/2, nu/2) - BetaSF(x; 1/2, nu/2).
    null energy fraction = N(c/(nu+c)) / N(1/K).

This follows by decomposing independent Z² and chi-square(nu) into their total
and their Beta(1/2,nu/2) proportion. Verified with scipy.special.betaincc. In the
known-variance/large-K version the fraction is simply
a exp(-(a²-1)/2), a=sqrt(1+M/B).

The untruncated mean²-s²/K is unbiased for squared signal only with an independent
fixed frame and suitable independence assumptions. Clipping negative estimates
to zero creates positive estimated signal under the null. The same positive
fluctuation both supplies the purported signal energy and selects entries above
the SNR threshold. As B grows, the threshold approaches the positive-clipping
boundary, necessarily assigning almost all retained estimated energy to the
"high-SNR" group even if true signal is absent.

The real SOAP-frame entry fractions are 0.205/0.315/0.354 in PD1M_900,
0.240/0.351/0.389 in PD4M_183, and 0.351/0.458/0.493 in SPD16M_46. They exceed the
null reference, and the 1M energy fractions (0.954/0.972/0.988) do as well. **This
review does not imply that actual gradients have no signal.** It establishes
that the 4M/16M energy numbers alone are unable to support the strong conclusion.

## The probe does not measure the optimizer's effective signal weighting

1. **Energy weighting changes.** Raw signal energy weights an entry by mu².
   Even an idealized gradient-RMS-normalized mean instead has squared amplitude
   mu²/(mu²+v), where v is target-batch noise variance. A multitude of weak
   coordinates can be negligible under mu² and substantial under this bounded
   weight. Actual SOAP additionally changes the basis, acts on momentum,
   restores a matrix norm, passes through Newton-Schulz/polar, and maps back
   through the input root. Raw energy dominance does not bound final update
   energy, held-out descent, or curvature cost from the weak coordinates.
   Relevant implementation: `research/adamw_spectra/muon.py:162-183,715-764`.

2. **SNR greater than one is a weak asymptotic condition.** Even if known exactly,
   SNR=1 permits a denominator sqrt(2) times the signal magnitude. Near-sign
   normalization requires SNR much greater than one on the coordinates that
   matter after normalization, not merely greater than one in raw-energy mass.

3. **The SOAP frame is fit and scored on the same micro-gradients.** Probe
   lines 95-106 learn both bases from the same samples used for means and
   variances. The fixed-frame variance correction and t-null do not hold
   literally after this data-dependent basis selection. GN's input basis is
   also the fitted SOAP input basis (line 102); its output basis uses separate
   curvature samples. The raw frame avoids this particular selection effect.

4. **The frame's statistical scale differs from training.** Probe lines 97-100
   use Grams of 32-sequence micro-gradients. Training uses EMA Grams of full
   batch gradients (`muon.py:186-210,739-752`). At a frozen state, the micro-Gram
   has mean part mu mu^T plus micro-noise Gram; the expected full-batch Gram
   has the same mean part plus m/B times that noise Gram. The relative noise
   term therefore differs by factors 64,256,1024 for 1M,4M,16M. The probe's
   SOAP bases should not be assumed to equal checkpoint optimizer bases.
   Likewise, it always uses input exponent 1/2, including the PD1M state whose
   run used 1/4. These are legitimate probe definitions but constrain the
   claimed connection to actual optimizer behavior.

5. **Static sampling noise and temporal prediction error differ.** Fixed-state
   SNR says nothing directly about whether the mean gradient persists after an
   optimizer step. The other saved dynamics measurements report strong
   period-two mean-gradient components. A stiff component can have very high
   static SNR and poor usefulness to old momentum. Whitening may suppress those
   components while emphasizing weak, flatter ones whose data noise still
   matters. This is a useful connection to investigate, not an established
   explanation of SOAP's gain.

6. **Matched final losses do not identify intermediate directions.**
   `MUON_CASE.md:3586-3599` records gradient-mode differences of -0.0022 at 4M
   and -0.0056 at 16M. These meet the predeclared proximity criterion at 16M.
   They do not measure near-equality of updates or show a formal equivalence
   interval across seeds. Momentum and gradient second moments can preserve
   similar noise-dependent relative weights, or the polar step can discard
   parts of their difference, while final losses remain close. Even under a
   stationary iid approximation, momentum's normalized second moment is
   mu²+[(1-beta)/(1+beta)]v, not a noise-free mu². Temporal drift and basis
   changes weaken this approximation further.

## Smallest useful discriminating measurement

First use saved actual SOAP states, if available, to inspect the **actual basis,
denominator, momentum input and final direction**. The current JSON saves only
aggregate/per-matrix fractions, not per-entry means, variances, bases or the
micro-gradients, so it cannot retrospectively repair the estimator or answer
the weighting question by itself.

If additional gradients are later authorized, fix a saved optimizer basis (or
fit one on a separate split). Use independent gradient splits for the mean,
variance, and mask; report untruncated cross-products of independent means for
group-level signal energy. Include an iid null calibration and uncertainty,
and disclose unresolved entries instead of turning clipped estimates into
certainties. Use enough independent mean-estimation data to address the target
batch, or quantify the extrapolation uncertainty explicitly.

Then evaluate weak/strong-coordinate groups **after the normalization** as well
as before it: normalized energy, held-out linear descent, final post-polar
direction angle, and, if already available, curvature cost. Compare frozen
denominators based on total second moment, signal-only and noise-only with
independent scoring. Similarity of these actual update maps on independent
data would be much stronger evidence for a near-sign mechanism than endpoint
loss proximity or raw-energy SNR fractions. No training ablation is warranted
solely to repair the present logical gap.

The appropriate current wording is: "SOAP's extra gain is largely output-side;
the tested momentum- and gradient-second-moment variants perform similarly.
Signal equalization is one candidate explanation. The saved frame-SNR probe
does not yet separate it from noise weighting or temporal effects."

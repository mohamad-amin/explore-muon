# Attention concentration as a quantitative curvature prediction

2026-09-28. Bounded conceptual/source review. No model call, tensor read,
new scientific score, or optimizer proposal. Only this note is written.

**Worth an analytical qualification; a later bounded attention forward could
be informative.** This is a sharper architectural prediction than the old
qualitative “diffuse attention raises value curvature” explanation. It is
not a test of whether the value-mean route is useful for loss, and should
not reopen that closed local branch. The empirical-centering convention
changes the proposed b/a prediction materially.

## Historical coverage

`logs/muon_spectra/second_order_audit_20260926/OBSERVATIONS.md:30` suggested
comparing attention entropy with value excess curvature. The initial
within/between prediction is explicit in
`research/adamw_spectra/MUON_CASE.md:1904–1920`, but its coefficients were
fitted to measured curvature. I found no explicit unfitted
q=mean_row sum(A_row²), harmonic-number, or inverse-participation mapping
in the bounded historical search.

Observer `attention_geometry/NOTE.md` already establishes that post-attention
centered-energy contraction is not an entropy estimate: input correlations,
learned value directions, and selection all contribute. It also reports
that the two-component fit does not universally halve the full-marginal
residual. The new idea should predict a specific part of that structure,
not claim to explain the entire marginal from one scalar.

## Correct identities and finite-T prediction

For one head, put sequence inputs in rows of X and let A be fixed and row
stochastic. Let downstream sampled-label errors h_t at the attention outputs
have zero cross-position covariance and the same within-position covariance
B_out. For the input partial trace, B_out need not be isotropic: only its
trace enters. Conditional on X,A,

    F_in = tr(B_out) C_AX,
    C_AX = Xᵀ Aᵀ A X / T,
    tr(B_preV) = q tr(B_out),     q = ||A||²_F / T.

This uses the archive's mean-token-loss GN units. It is an exact structural
identity under the stated error-covariance premise. C_AX retains actual
input-attention dependence; reducing it to two activation moments introduces
another assumption.

Now assume x_s=mu+epsilon_s, with iid zero-mean epsilon_s of covariance Sigma,
independent of A. Write M=E[mu muᵀ]. Then

    E[C_AX] = M + q Sigma.

But the archived moments use the **empirical sequence mean**, not the latent
mu. For fixed T and this iid model,

    E[C_between] = M + Sigma/T,
    E[C_within]  = (1−1/T) Sigma.

Consequently the coefficients in the archive's fitted convention
F_in ≈ tr(B_preV)[a C_within+b C_between] are predicted to be

    a = (Tq−1) / [q(T−1)],
    b = 1/q,
    b/a = (T−1)/(Tq−1).

Thus **b is 1/q; b/a generally is not**. At T=512, uniform causal attention
has q=H_512/512, giving b approximately 75 and b/a approximately 88.
The finite-T correction is substantial because averaging makes q small.
Complete uniform noncausal attention gives a=0; identity attention gives
a=b=1. These are useful analytic controls before any model measurement.

The old phrase “weight on the sequence mean” sometimes refers to b, whereas
the initial prediction explicitly used b/a. Compare the two saved
coefficients separately. A numerical similarity between an early fitted
ratio near 40 and an order-75 architectural number is motivation, not
confirmation.

## Where the real transformer can depart

- **Attention depends on X.** RMS-normalized inputs are neither iid across
  positions nor independent of A. C_AX is a useful forward-computable
  control that removes this particular approximation.
- **Errors vary with position and correlate across positions.** The general
  input marginal uses a kernel with entries tr Cov(h_t,h_u), giving an
  Xᵀ Aᵀ K A X contraction. Later attention creates off-diagonal K. Equal
  output anisotropy alone does not break the partial-trace formula;
  position dependence and cross-position covariance do.
- **Heads need curvature weights.** The full V input marginal sums head
  contributions weighted by tr(B_out,h). The effective q is their weighted
  average, not necessarily the unweighted head mean. Cross-head covariance
  itself drops out of this input partial trace because V's head rows are
  stacked; it matters for the fuller GN operator. Learned output weights
  and downstream curvature can change the head weights.
- **Across sequences, averaging matters.** A mean of 1/q is not the inverse
  of the appropriate weighted mean q. Correlations between concentration,
  sequence means, and residual covariance also obstruct a single coefficient.
- The archives include position zero; online PD excludes it. This prediction
  concerns the archived marginal, so its forward measurement must include
  all positions. The marginal is a one-draw sampled-label estimate without
  retained independent halves.

## Direction-changing comparison and stopping point

First qualify the formulas and conventions analytically, including the
identity/uniform/causal controls. If pursued later, a small fixed early/late
panel on the original M/PD trajectories could measure A concentration and
C_AX on immutable inputs. Freeze the states and aggregation before reading
attention outcomes. Report both predicted coefficients and full-matrix
shape, retaining the existing fit residuals; do not fit a new multiplier
to make q match b/a.

For the actual multihead archive, a forward-only prediction must explicitly
adopt equal head error traces or retain the range allowed by unknown positive
head weights. Under the common-input iid model, effective q lies between
the headwise q values, yielding a corresponding b/a range. A chosen
unweighted average is not a parameter-free prediction of the full marginal
unless the equal-head premise is stated. The analogous weighting issue
also applies to averaging headwise C_AX matrices.

Agreement of the unfitted q law across that panel would support averaging
geometry as an explanation for the changing *curvature ratio*. Failure of
the scalar law while C_AX tracks the marginal would implicate input-attention
correlations in the scalar reduction. Failure of both would leave the
downstream-error covariance premise or estimator variation central; a
forward-only test cannot identify which. Large systematic failure should
close the simple account, without head selection, extra exponents, or a
fresh optimizer arm. None of these outcomes demonstrates a learning-rate
benefit or a productive mean route.

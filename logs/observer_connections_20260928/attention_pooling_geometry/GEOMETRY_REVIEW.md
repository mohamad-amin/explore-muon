# Attention pooling gives a testable curvature predictor, under explicit assumptions

2026-09-28. Mathematical/source discussion only. No model, score, checkpoint,
attention tensor or empirical fit was read. Only this review is written.
This is a competing lead, not a replacement for the earlier trajectory
priority or permission to reopen a closed mean-route intervention.

**Verdict:** the pooling hypothesis can become a parameter-free *prediction*
rather than a qualitative story. Its simple 1/q version is not an exact law
of the actual transformer. The finite-T correction matters, and downstream
error coherence supplies a distinct competing explanation even at fixed A.

## Exact conditional identity

Fix one attention head and its current inputs X, attention matrix A and all
other parameters. Put Z=AX and Y=Z V^T. V's perturbation does not change this
head's A or X. Let K_tu be the downstream GN metric block between Y_t and Y_u,
with the overall mean-token normalization written separately as 1/T. Then

    Q(Delta V) = (1/T) sum_tu (Delta V z_t)^T K_tu (Delta V z_u).

This is the exact conditional GN block quadratic for V. Define the temporal
partial trace R_tu=tr(K_tu). Its input marginal is exactly

    F_in = X^T A^T R A X / T.

If e_s=sum_t A_ts delta_t is the error reaching V at source position s,
the average per-token error factor B_V satisfies

    tr(B_V) = tr(A^T R A) / T.

Thus the normalized input marginal is X^T S X/tr(S), S=A^T R A. R is PSD;
its off-diagonal entries need not vanish or have one sign. This identity
requires neither iid inputs nor A independent of X. It does **not** reduce
the whole GN block to one Kronecker product when K has general structure.
Model-sampled zero-mean errors can represent this GN covariance; data-label
second moments instead target the empirical Fisher and must be labelled so.

For the stronger assumption K_tu=1[t=u] B, R=tr(B) I. Now

    F_in = tr(B) X^T A^T A X/T,
    B_V = q B,       q=||A||_F^2/T.

This is the precise conditional pooling-operator identity. It concerns the
actual pooled input covariance C_AX, not automatically a within/between
mixture of C_X for each realized sequence.

## Ensemble reduction and the finite-T correction

Assume x_s=mu+epsilon_s, with centered iid epsilon_s of covariance Sigma,
and assume the inputs are independent of the fixed routing/error metric.
The sequence-level mu can be random if its second moment M is independent
of those objects. Conditional on fixed A,R, row stochasticity A1=1 gives

    E[X^T S X]/tr(S) = Sigma + Gamma M,
    Gamma = (1^T R 1) / tr(A^T R A).

The independent, homogeneous error model specializes to Gamma=1/q. For
random A,R, do not casually average ratios: population normalization uses
the ratio of expected numerator and denominator, subject to the required
independences.

The archived decomposition uses the **sample sequence mean**, so

    E[C_between] = M + Sigma/T,
    E[C_within]  = (1-1/T) Sigma.

Consequently the correct coefficients are

    normalized F_in = a C_within + b C_between  [in expectation],
    b = Gamma,
    a = (T-Gamma)/(T-1),
    b/a = (T-1)*Gamma/(T-Gamma).

For Gamma=1/q this becomes a=(Tq−1)/(q(T−1)), b=1/q, and
**b/a=(T−1)/(Tq−1)**. The naive C_within+(1/q)C_between overcounts Sigma
by (1/q−1)/T. A realized-data two-component identity does not follow from
the conditional operator identity: A^T A generally distinguishes positions
in ways those two sample covariances cannot encode.

For uniform causal attention, q=H_T/T. At T=512:

| q | a | b | b/a |
|---:|---:|---:|---:|
| .01331351 | .85496736 | 75.11168 | 87.85327 |

Do not compare an archived fitted b with a proposed b/a, or call both
"gamma." The earlier records used both quantities; verify which the quoted
early/late values denote before comparing them. Under homogeneous independent
errors and causal support, q lies between H_T/T and 1, hence b/a lies between
1 and 87.85 here. More general temporal R need not obey that causal-q bound.

## Assumptions that matter in this model

1. **Routing is input dependent.** A(X) selects correlated features. The
   cross term involving E[A epsilon] need not vanish, and
   E[epsilon^T A^T A epsilon] need not equal E||A||_F² Sigma. Positional
   structure, normalization, sinks and document dependence also violate iid
   epsilon. The exact conditional C_AX identity survives; the scalar mixture
   need not.
2. **Downstream errors mix positions.** Later attention layers generally
   produce off-diagonal R. Gamma then depends on temporal error coherence,
   not just attention collision q. Even diagonal R can be heteroscedastic:
   with weights w_t=tr(K_tt), the relevant q is weighted by w_t.
3. **The last attention layer is a structural control.** If all downstream
   operations are positionwise MLP/norm/head, as in this GPT, its downstream
   R is diagonal conditional on the input for the GN target. Homogeneity is
   still an assumption. Earlier attention layers lack that simplification.
4. **Multiple heads require error weighting.** For the complete V input
   marginal, sum S_h=A_h^T R_h A_h across heads. Under the iid reduction,
   Gamma_eff=sum_h 1^T R_h 1 / sum_h tr(S_h). Even in the homogeneous-error
   case this uses head-error-energy-weighted q_h, not the unweighted average
   of attention collisions. Off-head GN blocks matter to the full operator,
   although the input partial trace retains diagonal head blocks.

A varying fitted mean coefficient could therefore reflect attention
sharpness, downstream temporal coherence, feature-dependent routing, or
several together. A falling coefficient alone does not identify sharpening.

## What would make this a meaningful bounded research branch?

The advance over the old guess is a **zero-fit mapping from independently
measured routing statistics to a specified curvature target**, including
normalization and finite T. Do not infer q from the observed b/a and then
announce a match. Predict both coefficients and full-matrix residuals, not
only the conspicuous mean Rayleigh ratio.

A useful predeclared distinction has two levels:

- Does independently measured q predict the finite-T mixture within the
  existing measurement uncertainty? Violations of its admissible coefficient
  range or large systematic matrix residuals falsify this simple model.
- Does the directly measured pooled-input covariance C_AX predict better
  than that iid mixture under the same error assumption? Improvement would
  locate a failure in feature/routing independence. Failure of both points
  instead toward error heteroscedasticity/coherence or other conditional
  dependence. Weighted variants must be specified before scores, not added
  successively to rescue an attractive account.

The required A/pooled-input/error statistics are not supplied merely by the
old C_within/C_between fits. Their availability and CPU cost must be checked;
this note does not authorize regenerating the full historical marginal panel.

This is distinct from the closed global-mean-route question: it predicts
attention's *curvature geometry*, including centered variation and potentially
sequence-specific means, rather than asserting that a large constant route
is useful. But a successful marginal predictor still would not establish a
better optimizer. Per-kind power and mean-route failures remain constraints,
and the large GN-PD floor prize did not become a sustained rate. Keep this as
a focused architecture/statistics hypothesis until an independently scored
prediction succeeds; no V-only whitening or architecture change follows now.

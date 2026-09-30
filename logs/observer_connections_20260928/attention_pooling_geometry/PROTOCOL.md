# Analytical qualification: attention pooling and value-curvature coherence

2026-09-28. No model, checkpoint, optimizer, training or GPU computation is
authorized by this note. The current step is an analytical connection and
small deterministic numerical qualification, followed only by a separately
specified scalar-archive consistency check if justified. Independent reviews
are `HISTORY_REVIEW.md` and `GEOMETRY_REVIEW.md`.

## Missing connection

The old marginal archive reports large early value-projection sequence
coherence and much smaller late coherence. It suggested diffuse attention
as an explanation but did not test a coefficient prediction from attention
concentration. The candidate is to connect a forward routing statistic to
that specific curvature structure, not to reopen the failed value-loss
route, assume a new optimizer gain, or fit another whitening exponent.

For a single value head with inputs X (T×d), fixed row-stochastic attention A
and output Y=A X V^T, the actual input to the V map after commuting the two
linear operations is AX. A simple equal-query downstream-curvature model
therefore involves X^T A^T A X/T rather than X^T X/T. This identity is known
in the broader weight-sharing/attention-curvature setting; no novelty claim.

## Exact and approximate objects

Let K be the conditional covariance of post-attention errors after tracing
their channel dimension, in the normalization corresponding to mean token
loss. Then the input partial trace has the exact conditional form

    F_in = X^T A^T K A X / T,
    tr(B_pre) = tr(A^T K A)/T.

The special assumption K=tau*I gives F_in=tau*C_AX and
tr(B_pre)=tau*q, where C_AX=X^T A^T A X/T and q=||A||_F^2/T.
This does not require channel-isotropic errors, only equal channel-covariance
trace across queries and zero traced cross-query covariance.

Only under the further ensemble model x_s=mu+epsilon_s, with iid centered
epsilon covariance Sigma and attention/errors independent of those residuals,
does E[C_AX]=M+q*Sigma, where M=E[mu mu^T]. The archive uses empirical
sequence centering:

    C_between = M + Sigma/T,
    C_within = (1-1/T)*Sigma.

Thus the candidate predicted coefficients in
F_in ≈ tr(B_pre)*(a*C_within+b*C_between) are

    b = 1/q,
    a = (T*q-1)/(q*(T-1)),
    b/a = (T-1)/(T*q-1),
    (T-1)*a+b = T.

For uniform causal attention q=H_T/T. At T=512, b≈75 and b/a≈88;
they are not interchangeable. Multihead V needs downstream-curvature-weighted
head/query concentrations; an unweighted forward q adds another assumption.
Real attention depends on X, residual tokens are correlated and downstream
errors couple positions. None of those assumptions is presumed true here.

## Fixed analytical qualification and falsifiers

Use deterministic small FP64 examples only, with at most two numerical
threads and a one-minute cap:

1. Verify the exact partial-trace identity by constructing the full value
   Jacobian for fixed small X,A and a PSD downstream error covariance.
2. Verify the finite-T coefficient relation on exact ensemble second moments
   for identity, uniform causal, full uniform, and one fixed causal mixture
   of attention matrices; no Monte Carlo fitting is necessary.
3. Compare two fixed causal attention matrices with the same q and the same
   X but different C_AX, demonstrating that q is not a sufficient operator
   statistic on arbitrary observed features.
4. Hold X,A fixed while changing downstream positional covariance K; verify
   that equal forward q does not fix the curvature marginal.

All identities must agree to 1e−12 relative/absolute scale as appropriate.
Retain finite-T correction, counterexamples, limits and source hashes. A
failed numerical identity stops and preserves the attempt; no trained model
or optimizer is used to repair it.

If the algebra qualifies, the result is a precise candidate explanation with
known failure conditions. It is not evidence that the model's saved b values
equal a forward attention statistic. Before any real attention measurement,
compare the necessary coefficient relation against the existing archived
fits with their residuals and sample limits. No old fit is silently called
an exact GN population matrix, and no favorable state/layer is selected.

This analytical branch does not replace the separately proposed trajectory
head-start discriminator. It supplies a competing architecture-level
connection for the broader synthesis, not a new training commitment.

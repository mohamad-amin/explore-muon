# Attention spread alone does not describe the retained value-curvature fits

2026-09-28. The exact conditional pooling identity and its finite-context
correction pass deterministic numerical checks. But the simple iid mixture
is not quantitatively supported by the retained value-curvature fits: only
one of 256 cells lies within .10 of its necessary normalized coefficient
relation, and the two-component matrix fits themselves have large residuals.
No model, checkpoint tensor, new gradient, attention capture or optimizer
experiment was run. Do not infer an attention-dependent whitening rule from
this failed scalar closure.

## The connection worth keeping

The archive records that value curvature has a large sequence-coherent
component early and a smaller one late. Its old qualitative explanation was
diffuse attention. A quantitative version would predict the coefficients
from routing statistics rather than fit them to the curvature being explained.

For one value head, fix inputs X, row-stochastic attention A and the current
remaining network. Write Y=A X V^T and let R be the downstream GN covariance
between query positions after tracing the output-channel dimension. With
the archive's mean-token normalization,

    F_in = X^T A^T R A X / T,
    tr(B_preV) = tr(A^T R A)/T.

This conditional identity is exact. It distinguishes the routing matrix
from the downstream temporal-error metric. It is an input partial trace,
not a complete Kronecker description of the GN block.

If downstream queries have identical error covariance and zero cross-query
covariance, R=tau*I. Then F_in=tau*C_AX, where
C_AX=X^T A^T A X/T, and tr(B_preV)=tau*q with q=||A||_F²/T.
The relevant input statistic has been pooled by attention; raw C_X is not
automatically the same object. Weight-sharing/attention-aware curvature is
established prior art, as summarized in [LITERATURE.md](LITERATURE.md).

## Empirical sequence centering changes the coefficient prediction

Only after also assuming x_s=mu+epsilon_s with iid centered residuals,
independent of routing and error covariance, can the marginal reduce to a
two-moment ensemble model. Put M=E[mu mu^T], covariance Sigma. The archive uses

    C_between = M + Sigma/T,
    C_within = (1−1/T)*Sigma.

Thus the coefficients in F_in≈tr(B_preV)(a*C_within+b*C_between) are

    b=1/q,
    a=(T*q−1)/(q*(T−1)),
    b/a=(T−1)/(T*q−1),
    (T−1)*a+b=T.

At context 512, uniform causal attention gives q=.01331351,
**a=.854967, b=75.111679, b/a=87.853270**. A quoted coefficient b must not
be compared with a proposed ratio b/a. The global a also changes scale;
for a positive two-component factor, trace normalization would remove that
global multiplier while retaining b/a and the actual component matrices.

For general downstream R, the iid-input reduction instead has

    Gamma = (1^T R 1)/tr(A^T R A),
    b=Gamma, a=(T−Gamma)/(T−1).

The same coefficient relation remains. Merely replacing q by a weighted
concentration or retuning a downstream scalar cannot repair its failure.
Real input-dependent routing, correlated residual inputs, heterogeneous
errors, estimator noise and fit identifiability are separate possibilities.
For multiple heads, downstream error energies weight the contributions;
an unweighted mean attention concentration adds another assumption.

## Deterministic qualification and counterexamples

`qualify.py` constructs a small full value Jacobian and PSD downstream
covariance, then compares its input partial trace with the formula above.
Relative errors are below 3e−16. Exact ensemble moments verify identity,
uniform causal, full uniform and one causal-mixture cases at T=4 and 512;
coefficient relations agree at numerical precision. No Monte Carlo training
or project tensors are involved.

Two causal routing matrices with identical q=1 and the same fixed inputs
give different C_AX matrices: diag(1,0) and diag(1/3,8/3). Hence q alone
does not determine the conditional operator on arbitrary observed features.
The same fixed X/A with two downstream covariance cases also gives different
normalized input marginals (Frobenius difference .135651); this additional
comparison reuses the exact fixtures in the qualification. Separately,
fixed ensemble moments and attention yield different Gamma under equal,
heterogeneous, coherent and centered error kernels. These are assumption
checks and counterexamples, not empirical explanations of trained models.

## Complete scalar-archive check

The fixed check reads all 32 archived states—M, PD, SOAP-Muon and SOAP-PD at
10/50/100/200/500/900/1300/1469—and all eight V matrices in each. Every state
records 2048 sequences and context 512. It uses the original fit coefficients,
fit residuals and hashes; no new coefficient, attention q or matrix is fitted.

For each cell compute r=[511*a+b]/512−1. Zero is the idealized ensemble-model
relation. The following bands are descriptive scales, not hypothesis tests:

| Reference band | Cells with abs(r) within band | Cells with matrix-fit residual within band | Both |
|---|---:|---:|---:|
| .05 | 1 / 256 | 0 / 256 | 0 / 256 |
| .10 | 1 / 256 | 0 / 256 | 0 / 256 |
| .25 | 6 / 256 | 58 / 256 | 1 / 256 |

Every method's median r at update 10 is negative (−.39 to −.66). By update
100, the median within coefficient a is 4.61, 8.46, 11.06 and 13.19 for
M/PD/S/SPD; corresponding median r is 3.63, 7.46, 10.08 and 12.20. The
idealized causal equal-error model requires a between .855 and 1. These
are large differences in the recorded fits, not numerical boundary effects.

The two-component fit is itself incomplete: median relative matrix residuals
span roughly .22–.72 across the method/stage groups. Sixteen early cells have
nonpositive a. Their signed/unstable b/a values are retained in the CSV and
must not be interpreted as positive effective context lengths.

The coefficient distinction matters even where a is positive. At the final
checkpoint, separately computed medians over layers are:

| Method | a | b | Per-layer b/a |
|---|---:|---:|---:|
| M | 1.956 | 3.187 | 1.633 |
| PD | 3.894 | 6.009 | 1.637 |
| S | 2.548 | 4.572 | 1.716 |
| SPD | 3.879 | 5.829 | 1.655 |

Thus the larger absolute b under PD/SPD does not by itself establish a
proportionally larger **relative** sequence weighting. These are medians of
individual quantities; a ratio of medians is not substituted for the median
ratio. The component matrices and their proportions still differ across
trained states, so similar b/a does not make their normalized factors equal.

All cells and method/stage ranges remain in `archive_cells.csv` and
`archive_check.json`; the [figure](archive_relation.png) shows the complete
panel. The initial metadata-relative-path failure occurred before coefficient
reduction and is preserved in `failed_relative_path/`. The corrected read
resolves paths against the project without rewriting historical metadata.

## What follows, and what does not

The checked algebra produces a falsifiable scalar account; the retained
fits do not support using it as a quantitative model of their full input
marginal. This does not prove that attention averaging contributes nothing
to the coherent component. It is not statistical rejection of a population
law: the marginals use sampled labels, lack independent stored halves, and
their unconstrained fit coefficients can be sensitive to component geometry.
No particular failed independence assumption is identified.

Close this scalar consistency check. Do not infer q from b and call that
independent validation, refit a coefficient to rescue the account, select
favorable layers, regenerate the historical marginal panel, or launch a
value-only optimizer change. The exact conditional operator identity remains
useful for designing a different, explicitly qualified question about
input/routing/error dependence. It is not a new optimizer or a confirmed
mechanism of the momentum gains.

The separate dynamical synthesis also remains relevant: a local component
helpfulness failure cannot veto a possible long-run role. Future priorities
should distinguish retained head starts, endogenous relative rates and
changing coupled feedback, using trajectory-level evidence and adequate
measurement resolution.

Evidence: [protocol](PROTOCOL.md), [archive contract](ARCHIVE_CHECK.md),
[independent results review](RESULTS_REVIEW.md), `qualification.json`,
`fixed_input_downstream_check.json`, `archive_check.json`, `archive_cells.csv`,
and the two independent design reviews in this folder.

# Independent review: the algebra qualifies; the scalar iid-pooling account does not describe the retained fits quantitatively

2026-09-28. Reviewed the analytical producer and recorded qualification,
independently recomputed all archive coefficients/summaries/counts from the
original 32 marginal JSONs, and verified metadata/source hashes. Only scalar
arithmetic and source inspection were used. No new matrices, tensor archives,
model calls, attention capture, optimizer or GPU computation. This is the
only file written.

**Verdict: agree with the bounded conclusion.** The exact conditional
operator identity and finite-T correction are sound. The necessary scalar
coefficient relation is broadly inconsistent with the archived approximate
V fits. That makes the proposed scalar iid account unsupported as a
quantitative description of these retained estimators; it is not a formal
population rejection, a diagnosis of which assumption failed, or evidence
that attention routing is irrelevant.

## Mathematics and synthetic qualification

The full Jacobian in `qualify.py` places pooled input row z_t into the
appropriate V-output-row parameter block. Contracting the diagonal output
blocks of J^T H J/T yields, by indices,

    F_in[j,l] = (1/T) sum_tu sum_a H[(t,a),(u,a)] z_t[j] z_u[l]
              = (X^T A^T R A X/T)[j,l].

The routed pre-V error covariance trace is correspondingly
tr(A^T R A)/T. Thus the code's full-Jacobian and compact expressions have
the same indexing and loss normalization. Its recorded normalized errors
are 2.63e−16 and 2.93e−16. I verified this algebra/source structure; I did
not rerun or create matrix examples in this review.

Independently recalculated all eight existing finite-T cases from their
stored q,T values. They satisfy

    a=(T−1/q)/(T−1), b=1/q, (T−1)a+b=T.

Uniform causal T=512 gives a=.85496736, b=75.11168, b/a=87.85327.
The full-uniform case correctly has a=0 and an undefined b/a, not a
regularized ratio. This confirms why b and b/a cannot be interchanged.

The same-q observed-input counterexample is valid and causal: its pooled
covariances are diag(1,0) and diag(1/3,8/3), with Frobenius separation
sqrt(68)/3=2.74873708. It proves that q alone does not determine the operator
for arbitrary fixed observed X. It does not itself demonstrate empirical
input-dependent routing or refute the iid *ensemble* prediction.

The downstream examples independently reproduce Gamma values
48/25, 258/71, 192/71 and 0 for equal, unequal, coherent and centered temporal
covariances with the same uniform-causal A at T=4. One wording qualification:
this part of the code holds **A and the input ensemble moments** fixed,
not an explicitly sampled realized X. It establishes downstream dependence
of Gamma as intended; the exact fixed-X identity is qualified separately.
No additional toy is necessary to support that narrower statement.

Most importantly, replacing 1/q by a general temporal-coherence Gamma
**does not remove the coefficient relation** under the same iid/input-metric
independence assumptions: a=(T−Gamma)/(T−1), b=Gamma still implies
(T−1)a+b=T. Merely changing to a weighted q or measuring a different scalar
Gamma cannot repair the archive disagreement while leaving those assumptions
intact. This constrains a potential repair cascade before it starts.

## Complete archive coverage and exact numerical reproduction

Independently confirmed:

- exactly four methods M/PD/S/SPD × eight steps
  10/50/100/200/500/900/1300/1469, no duplicate or omitted state;
- every filename agrees with its recorded arm basename and checkpoint step;
- every marginal uses 2048 sequences, with T=512 and eight layers confirmed
  from the corresponding training metadata;
- all 256 V rows are present, one per layer in every state;
- all raw a,b, b/a, fit residual, K-FAC residual, between-share and derived
  coefficient residual fields reproduce exactly;
- all 32 per-state medians, minima/maxima and reference-band counts match;
- 36 input hashes (32 marginals + four metadata files) and the four analysis/
  qualification source/contract hashes match their saved records.

Let r=(511a+b)/512−1. The complete reference counts are:

| Fixed descriptive band | Cells with abs(r) within band | Cells with fit residual within band | Both |
|---|---:|---:|---:|
| .05 | 1 | 0 | 0 |
| .10 | 1 | 0 | 0 |
| .25 | 6 | 58 | 1 |

The lone relation-within-.10 case is SOAP-Muon at step1469, layer8;
its matrix-fit residual is .6180. The lone joint-within-.25 case is
SOAP-Muon at step50, layer7, with r=.14775 and fit residual .22327.
These exceptions are retained, not selected as support. Across all cells,
fit residuals range .14576–.86504; sixteen fitted a values are negative.

The coefficient disagreement is not confined to a single outlier. At step100,
median a is 4.6065/8.4557/11.0558/13.1941 across M/PD/S/SPD. At the final
checkpoint it is 1.9562/3.8944/2.5477/3.8794, while median b is
3.1868/6.0090/4.5723/5.8290. Early step10 has median a .253–.521 and b
44.56–46.49, with wide layer variation. A familiar-looking b magnitude
therefore does not establish the model's required joint relationship.

## Meaning and limitations of this negative check

The archived `fit_exact` is an unconstrained Frobenius least-squares fit of
the Monte-Carlo full-sequence GN input marginal by
tr(B_pre)*(a C_within+b C_between). Current source confirms that scale and
that C_between uses the sample sequence mean. The check therefore uses the
appropriate coefficient convention; it is not accidentally comparing b
with b/a or an unnormalized error factor.

However, fitted a,b are not exact physical mixture weights. Their sizable
matrix residuals mean the target often lies substantially outside the
proposed two-matrix span. Negative a does not establish negative true GN
curvature, and a large coefficient residual cannot by itself locate an
error in attention, downstream covariance, the data model or the estimator.
When one component has little independent matrix energy, coefficient values
can also be sensitive to target error. No split-sample uncertainty, new
fit or condition-number estimate was manufactured in this check.

The 2048-sequence sampled-label measurements are finite estimates. Methods,
layers and checkpoints share data and are not 256 independent trials. The
.05/.10/.25 bands were fixed descriptive scales, not significance levels.
No attention q was measured here. Historical diagnostic-source limitations
remain; checking today's interpretation of an archived field does not
retroactively certify every original measurement byte.

The preserved first attempt failed in relative-path metadata bookkeeping,
before entering the a,b reduction loop. The exact diff adds only conversion
of relative paths to project-root paths in `read`. It changes no state,
coefficient, reference band or scientific criterion. The failed source/log
remain under `failed_relative_path/`.

## Scientific decision

Close the simple scalar iid-pooling account at this retained-fit scope.
Do not launch a forward q capture merely to fit a new effective coefficient,
select a favorable layer, or reinterpret a small mean-direction ratio as
successful full-matrix prediction. The exact routing operator remains a
useful identity and the counterexamples identify why its scalar reduction
needs assumptions. They do not yet motivate a V-only optimizer change.

This is a meaningful constraint on an old qualitative coherence story,
not a refutation of all attention-aware geometry. A distinct future question
would have to name which conditional dependence is being measured and why
existing scalar/fit evidence cannot answer it. It should not silently grow
from this closed check. The separate trajectory-history priority is unchanged.

# Independent review of the completed TS repeatability result

2026-09-28. Reviewed `ts_repeatability/run2/result.json`, `analysis.json`,
`analyze.py`, and retained direction tensors. No model or GPU was used.
`validate.py` independently recomputes every aggregate from all 48 matrix
rows and checks three saved-tensor comparisons on `block01.q` (512 by 512).
It completes in about 1.5 seconds on two CPU threads. Results are retained
in `validation.json`.

## Validation and interpretation

The reported results are correct. The source execution ended at 392.3 seconds,
below the unchanged 900-second cap. Every tested saved-tensor cosine and
PD-relative variation ratio agrees with the JSON to absolute error below
1e-12. The stricter prospective interpretation is material sample sensitivity.

| Comparison | Whole-body direction cosine |
|---|---:|
| Six disjoint eight-sequence pairs | .69649–.70207; mean .69914 |
| Same sequences, new sampled labels | .74057 and .73905 |
| Independent 16-sequence groups | .82212 |
| One-refresh EMA proxies | .98359–.98693 |
| Pooled direction, FP32 versus BF16 NS | .9998447 |

The main new fact is that **sampled predictive labels alone substantially
change the freshly estimated TS direction**, holding model, sequences,
momentum, and input root fixed. This comparison cannot be attributed to
different input examples or an evolving training trajectory. Its replication
on two banks supports that narrow statement. It does not establish a precise
fractional attribution of total estimator variance.

Independent pooling improves repeatability, but the pooled 16-sequence
directions still differ materially. The smaller one-refresh change shows why
fresh-factor instability must not be substituted for online EMA instability.
The CPU NS precision control is far smaller than the measured sampling
differences at the pooled factor; it does not test all forward/backward
precision effects or the sensitivity of every small-sample map.

The independent-eight variation is .442–.449 of the summed squared TS–PD
distances used as the preregistered denominator. For the EMA proxy that ratio
is .0246–.0309. These are finite-sample geometric ratios. Calling the first
number “44% of the training noise” or “44% of TS's useful improvement” would
be incorrect: online history and usefulness were not measured.

All 17 factor sets have zero reported eigenvalues below the relative damping
threshold and zero negative eigenvalues. Therefore the result is not simply
an eigensolver failure or a large pile of modes clamped at the damping floor.
This does not establish good conditioning: eigenvalues above the damping
threshold can still have large relative estimation error.

## Heterogeneity is informative

Average cosines across each type of comparison:

| Projection family | Disjoint eight | Labels only | Independent 16 | EMA proxy |
|---|---:|---:|---:|---:|
| Q | .8340 | .9400 | .8955 | .9919 |
| K | .7900 | .8327 | .8626 | .9882 |
| V | .6998 | .7077 | .7663 | .9728 |
| O | .9145 | .9347 | .9556 | .9974 |
| MLP up | .5291 | .5731 | .7379 | .9805 |
| MLP down | .9375 | .9508 | .9676 | .9981 |

MLP-up and V deserve explanation, rather than treating the whole-body angle
as a generic property of curvature estimators. MLP-up has the largest output
factor dimension, but V's sensitivity shows that dimension alone is not a
sufficient account. Token coupling, the conditional output-error covariance,
and its interaction with the fixed momentum can all matter. No one of these
mechanisms is identified by the table.

On the two repeated-label banks, the median trace-normalized B cosine is
.9486/.9367, whereas the whole-body direction cosine is .7406/.7391.
Those summaries have different weighting and should not be divided to claim
an amplification coefficient, but they motivate inspecting relative factor
error in the coordinates the inverse root actually uses.

## Recommended next decision

**First calibrate the retained factors; do not buy a larger sample yet.**
This can be done without another model forward:

1. Use the pooled C/D factor as an anchor independent of the A/B factors.
   Compare the A0/A1 and B0/B1 label differences, and A0/B0 combined
   differences, after congruence by the anchor inverse root. Normalize each
   factor exactly as the actual TS map does, and preserve the raw trace
   information separately. This asks whether the apparently modest raw-B
   changes are large fractional changes in the relevant output geometry.
2. Report all 48 matrices and projection families. If using anchor eigenvalue
   bands to localize relative error, fix those bands before calculating their
   contributions and do not mistake a noisy independent anchor for truth.
   Compare repeated-label versus cross-sequence errors descriptively.
3. If those measurements support a substantial geometric estimator error,
   the next model measurement should assess **functional relevance of the
   retained direction differences**, before a 72-sequence fresh pool.
   A held-out predictive-GN norm of `D_A-D_B`, relative to the similarly
   measured TS–PD differences, directly tests whether large parameter-space
   variation survives mapping to the model's predictions. This avoids the
   lagged-M descent/rate interpretation and respects the earlier V/O gauge
   lesson. A small fixed set of existing pairs and PD is sufficient; no
   direction should be selected by a favorable loss score.

A single new 72-sequence estimate would have no independent partner at that
sample size, so agreement with nested smaller estimates would again be only
a convergence check. It would also replace historical states by a fixed
state and equal weights, failing to reconstruct the online EMA. Larger
sampling becomes worthwhile after demonstrating that the relevant estimator
variation has a functional consequence and specifying the competing
explanation it would discriminate.

The present result is already a legitimate finding: a previously unmeasured
estimator difference is substantial for fresh TS factors at this retained
state, and predictive-label Monte Carlo randomness is a demonstrated
contributor. No optimizer improvement, intrinsic superiority of SOAP's
target, or failure of historical online TS follows yet.

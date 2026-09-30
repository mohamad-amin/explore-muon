# Fixed-weight replay: data reweighting versus model-history differences

2026-09-28. Retrospective arithmetic on the two saved100-batch replay JSONs
at Muon/PD1Mstep500. No model, GPU, checkpoint alteration or new replay.

The transport archive re-evaluated each past training batch at the same
weights. It retains16gradient projections per batch, historical clipping
factors and projections of actual/replayed momentum. Those arrays can test
whether emphasis on recent data materially changes the dominant stiff
projection, separately from the model-history difference between the actual
and replayed gradients. They cannot establish population noise, stability
or the mechanism of the16Mshort-momentum gain.

Use both complete100×16arrays, ordered newest first, with the archived beta
and historical clipping factors. Let p_k be the raw gradient coordinates,
w_k=beta^k f_k, A=sum(w), gbar=mean(p), and replay=sum(w_k p_k). Exactly

    replay = A*gbar + sum(w_k*(p_k−gbar)).

First reconstruct the archived Mstar and gbar projections. Require relative
error≤1e−5, reflecting FP32 accumulation/projection, without changing inputs
or fitting a scale. Report the weighted residual against both A*gbar and the
actual momentum projection, together with signed inner products and angles.
A small residual relative to the mean but large relative to momentum is not
evidence that data fluctuations are unimportant to the actual optimizer.
Retain all per-mode signed means, variances and the centered lag1..10cosine/
covariance descriptors. No significance tests or independent-lag claims.

The replay includes only100terms. Saved actual M also contains beta^100M400.
Use the existing training-step global clipped-gradient norm logs to bound
that *actual-momentum* tail by the triangle inequality. This does not bound
an unmeasured full stale-free replay tail: re-evaluated older gradients at
W500 can have different norms. If the old logs are incomplete, report the
looser bound beta^100*clip/(1−beta), not an invented actual M400.

Historical clipping factors are held fixed by the original experiment;
the replayed gradients were not re-clipped at W500. The model was trained
on these batches, so the analysis is conditional on a selected finite
history and fixed curvature coordinates. The retained16Ritz coordinates
do not cover all parameters or establish temporal feedback from data-order
correlations. Neither a constant fixed-state mean nor its relation to M
licenses mean removal or an optimizer intervention.

If the mean term explains the large replay/current-momentum difference,
that strengthens the feedback interpretation of those selected directions.
If recent-data reweighting is comparable to that difference, preserve data
composition as a competitor. Either outcome concerns the selected1Mstates,
not a new general mechanism. Close the scalar discriminator without growing
the sample or launching replay. Independent design discussion is with the
reviewer of`../linearized_edge_peer/REVIEW.md`. Cost: seconds of CPU/JSON work,
two numerical threads; all new artifacts remain here.

## Post-hoc boundary check, declared before root's regression

The primary reduction found positive centered correlations across most/all
lags1..10 in both fixed-weight replays. The known100M-token shard layout
provides one external grouping: batches401–500 span419,430,400–524,288,000
and cross the500Mboundary inside batch477. This is a metadata-defined check,
not a searched change point or a special test of the observed lag3maximum.
The independent peer has already checked a related two-block split excluding
the crossing row; retain that history and make no prospective claim.

Use all100rows and exactly one regressor q_k, the fraction of that batch's
input tokens beyond the manifest's500Mboundary. Fit intercept plus q to each
of the16projection coordinates. Preserve q, the explained centered variance,
all original lag1..10descriptors, and residual lag descriptors using both
residual and original variance denominators. Decompose the original weighted
data residual exactly into fitted-boundary and remaining components, retaining
norms, signed cross terms and parallel contributions. No extra strata, fitted
boundary, token-feature scan, beta candidate or model follows.

This tests whether one coarse source-block mean jump accounts for the order
structure. Content, training age and model conditioning remain confounded;
either outcome cannot establish a data-order cause or justify shuffling.

Metric calibration: the q regressor and recency weights are strongly aligned,
so fitting and scoring on the same rows can yield a large weighted share
even for a randomly ordered finite set. Compute exact expectations under
uniform row permutation, preserving these100observed centered vectors. The
expected fitted variance fraction is1/(K−1); the ratio
E[fitted_delta·delta]/E[||delta||²] is cos²(w_centered,q_centered). The expected
weighted residual norm squared is||w_centered||²||P_centered||F²/(K−1).
Retain observed-versus-calibration values. These are descriptive permutation
references, not a claim of row exchangeability, a p-value, or E[the ratio].
No permutations need be simulated. Preserve the pre-calibration output.

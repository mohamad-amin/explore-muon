# Curvature-marginal inventory and coverage decision

2026-09-28. Bounded source/history/JSON-schema review after reading
RESEARCH_GUIDE.md and RESEARCH_STATE.md. No model call, checkpoint tensor,
new scientific scoring, tensor reduction, or source-history reconstruction.
Only this review is written.

**There is a real distinction, but no strong saved-only next experiment.**
The old audit already measured the per-sequence GN input marginal, including
cross-position terms, and tested within/between-sequence fits on every body
matrix. The later C_w experiment was only a per-token marginal; its failure
does not rule out a full GN-marginal input metric. Nor does the failed
per-pair-diagonal hypothesis rule out rotations of that input metric.
However, full marginal matrices survive only for q/k/v. The o/up/down
matrices that motivated the later within-matrix GN gap retain projected
diagonals and scalar summaries, not recoverable full factors. No independent
sample halves or per-sequence sufficient statistics survive in this family.
I would not turn this coverage gap into another Q/K/V local probe or reopen
the already investigated mean/value route.

## What was computed, and what was actually saved

Let `A = logs/muon_spectra/second_order_audit_20260926`.

Producer: `A/measure_marginals.py`; accumulator and definitions:
`research/adamw_spectra/gn_probe.py:337–410`.
The `A/marginals/` directory contains 32 JSONs and corresponding `.pt`
archives: M, PD, SOAP-Muon, and SOAP-PD at steps
10, 50, 100, 200, 500, 900, 1300, and 1469. JSON metadata records 2,048
sequences for each. There are 48 body matrices per state.

The producer uses the validation stream from token offset 2,097,152,
context 512, one sampled-label backward per sequence, and FP64 accumulated
moments of FP32 products. The retained launch scripts, e.g.
`A/job_marg_M.sbatch`, name the complete checkpoint list and producer.
This is the original 1M training-trajectory family, not the later 16M states.

For a sequence of T positions, let g be its sampled-label gradient of
mean token loss. The accumulator computes

    F_in  = T E[g^T g],          F_out = T E[g g^T]
    token_in  = E_t[||e_t||^2 x_t x_t^T]
    token_out = E_t[||x_t||^2 e_t e_t^T]
    C = E_t[x_t x_t^T],         B = E_t[e_t e_t^T].

Here the e_t in the per-token equations have been rescaled out of the
mean-loss 1/T convention. `exact_in` and `exact_out` name F_in and F_out.
They are Monte Carlo estimators of the true GN partial traces, not
deterministically exact matrices from a single label draw. Expanding
g^T g includes t != s products within a sequence; token_in discards those.
The empirical-Fisher versions use true labels and include gradient signal.

`measure_marginals.py:68–80` saves:

| Quantity | Every body matrix? | Full matrix retained? |
|---|---|---|
| C and B eigenvalues, traces | Yes | No, except C under q |
| F_in, token_in, EF_in along C eigenvectors | Yes, full diagonal vectors | F_in only for q/k/v; token_in and EF_in no |
| F_out, token_out, EF_out along B eigenvectors | Yes, full diagonal vectors | No |
| Input mean vector | Yes | Vector retained |
| `exact_in_full` | q, k, v only | Yes, FP32 |
| `C_full`, `C_between_full` | q only, for the shared q/k/v input | Yes, FP32 |

Thus q/k/v share access to each block's saved attention-input C and
C_between through its q entry. The other matrices' eigenvector bases are
not saved here. Their projected diagonals cannot reconstruct a rotated
input metric or its off-diagonal terms.

The JSONs preserve more than eigenvalue ratios. Each matrix has `in`,
`out`, `gradient`, `fit_exact`, and `fit_token`. The side reports include
unit-trace full-matrix Frobenius differences, top-eight eigenspace overlaps,
trace ratios, leading directional ratios, and slopes. The input report
also records curvature along the input mean. Both fits retain a_within,
b_between, residual_fit, residual_kfac, and between_share_of_C.
These full-matrix scalar diagnostics were computed before matrix data
were discarded. Simply proposing those diagnostics again would duplicate
existing coverage.

`A/summarize_marginals.py` is a reporter for `check_tools.py`'s JSON layout
(`runs[].marginals`), not a producer of new factors. It explicitly
distinguishes exact/token cross-position corrections from token/K-FAC
activation-error dependence. `A/plot_marginals.py` and `make_atlas.py`
already consume the fitted coefficients and related diagnostics.

## What the historical tests cover

1. **Within/between-sequence structure was an explicit initial hypothesis.**
   `MUON_CASE.md:1904–1920` predeclared
   F_in ≈ tr(B)[a C_within + b C_between], including k and v predictions
   and a fit-residual criterion. `gn_probe.within_between_fit` fits these
   two full matrices by Frobenius least squares. C_between is
   E_sequence[xbar_sequence xbar_sequence^T]; C_within=C−C_between.
   C_between includes both the global mean outer product and variation of
   sequence means. It is not merely a renamed global-mean rank-one channel.
   The fit is descriptive and unconstrained; its coefficients are not
   automatically a positive-definite optimizer factor.

2. **The architecture pattern was measured across all four trajectories.**
   `A/OBSERVATIONS.md:5–37,50–64` documents flatter key mean directions,
   stronger value sequence coherence, and approximate C proportionality
   for q/o/up/down. These are old observations, not an overlooked new
   attention mechanism. The notebook wording about agreement “along every
   eigenvector” establishes diagonal agreement in that basis; it is not by
   itself a theorem of full-matrix equality. The saved Frobenius/overlap
   diagnostics are the additional relevant existing evidence.

3. **Token weighting was tested and failed its gain criterion.**
   `MUON_CASE.md:2990–3013`, `A/weighted_input_probe.py:42–78`, and
   `A/weighted_probe/` compare C against sampled-label and true-label C_w.
   The script's first docstring calls its per-token operator the exact GN;
   that statement is too broad for matrices feeding downstream attention.
   Its implementation is specifically token_in normalized by error energy,
   with position zero excluded. It never forms per-sequence g^T g.
   The original marginal measurement includes position zero. Therefore a
   later difference between these archives would conflate that mask change
   unless matched; it would not isolate cross-position effects alone.

4. **Per-pair cross-position diagonals were tested separately.**
   `MUON_CASE.md:3286–3320` corrects the token-factor operator's claim of
   unbiasedness and records `A/framepd_probe.py`'s kfac/ekfac/exact test.
   The per-sequence `exact` h retains cross-position terms but stays diagonal
   in the fixed B,C eigenbasis. At the two 1M floor states it adds little
   over the per-token diagonal; that proposed explanation failed there.
   It does not test off-diagonal changes to an input factor or the full
   GN block, and must not be generalized into such a negative result.

5. **A direct full-F_in whitening comparison was not found in this pass.**
   The bounded source search found `exact_in_full` only in the saving code,
   with C_within/C_between used for fitting/reporting. The early protocol
   proposed per-kind GN-marginal factors, but I found no completed training
   arm or scored direction using the saved full F_in as the PD input metric.
   This is a scoped absence finding, not a claim about uninspected external
   execution records or global novelty.

## The precise untested comparison—and why not pursue it from this archive

The distinct mathematical question would be PD with a normalized full
per-sequence partial trace F_in versus matched C and token_in, preserving
the input vector, masks, matrix norm, power, damping, and score data.
That allows input-basis rotations caused by cross-position covariance.
It is neither a within/between coefficient fit nor another per-pair
diagonal in the old C basis.

The useful competing explanation is that a new factor would mostly encode
already known key/value mean or sequence-coherence structure, or finite
sample variation, while the productive GN gap remains output-input coupling
that a partial trace cannot retain. A low residual in a full-matrix fit
would further weaken the case for a distinct factor, but those residuals
have already been computed and should be read before new calculations.

The smallest worthwhile **future qualification** would require a full
F_in and matched token_in/C for the matrices where the loss-relevant GN gap
was observed, with independent halves to establish whether any input-basis
rotation repeats. Those requirements are absent from this saved family for
o/up/down. Selecting q/k/v merely because their full matrices survive would
change the scientific target and largely overlap closed attention/mean and
local Q/K branches. The shared 2,048-sequence archive has no stored
per-sequence factors or split-label realizations from which an independent
repeatability test can be recovered.

Accordingly, **no-go for a new saved-only optimizer-map or functional probe
from these marginal archives in this pass**. Preserve the qualified open
statement: full per-sequence input-marginal whitening has not been ruled
out by token-weighting or frame-diagonal failures. It is not yet a strong,
artifact-supported next method direction. No new model capture, sample
expansion, inferred matrices, or historical-source archaeology is proposed.

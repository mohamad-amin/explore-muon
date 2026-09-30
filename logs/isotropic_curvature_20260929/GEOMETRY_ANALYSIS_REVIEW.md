# Independent geometry-analysis and watcher review

2026-09-29. Source/schema review of `analyze_geometry.py` and
`postprocess_when_complete.py` before postprocessing. No model calls, large
tensor reads, outcome extraction, or producer changes. Both sources parse.
The preceding analyzer's completion/coverage checks were also inspected to
understand the watcher's sequencing.

**Verdict:** the angular-radius and orthonormalized curvature formulas are
correct. Two small fixes are recommended before this analyzer executes:
include the difference matrix in the shared heatmap color limit, and make
the parameter-Gram rank threshold relative to its largest eigenvalue.
Neither changes the measured panel or selects outcomes.

**Pre-execution resolution:** root applied both fixes and positive-norm
guards before postprocessing. The reviewer verified the updated source:
the color limit includes |H−GN|; the rank cutoff is lambda_max×1e−12 with
a nonempty-span guard; Gram eigenvalues and cutoff are archived. No remaining
issue in this bounded review blocks postprocessing. Updated geometry source
SHA256: `fe2ccc9f985da0562d18af20a2e005501ad5c78188b4d98732e8b695c1187041`.
The discussion below preserves what motivated these changes.

## Angular geometry

With row-stored inputs, `white = x @ S_inverse` is the intended symmetric
whitening transform. For each archived direction D, the matrices used for
the two coordinate frames are Q=D and Q=D S. In either chart the unnormalized
physical kick is D x. Therefore

    archived_radius² / ||chart_input||²

is exactly ||Q z||² for that chart's unit-normalized input z. This holds for
all five archived directions, including the direction constructed by
whitened right rotation. Reusing saved radii avoids new model measurements.
The raw and white unit normalizations differ deliberately; these are angular
comparisons, not claims that their re-normalized physical inputs are equal.

The implemented `tr`, `tr2`, effective rank, spherical mean, and relative
variance correspond to

    tr = tr(Q^T Q),
    tr2 = tr((Q^T Q)^2),
    r_eff = tr²/tr2,
    E||Qz||² = tr/n,
    CV²(||Qz||²) = 2/(n+2) (n/r_eff − 1).

`gram.square().sum()` computes tr2 correctly because the Gram is symmetric.
The zero clamp on spherical variance only protects a possible negative
roundoff residue at an exactly flat spectrum. The near-zero denominator
rule returns an absent ratio instead of dividing by a nearly zero spherical
variance; the underlying observed/predicted moments remain reported.

Bank0 uses contexts0–1, bank1 contexts2–3, and `all` uses all four. Population
variance (`ddof=0`) is appropriate for these descriptive empirical moments.
It should not be reinterpreted as an uncertainty estimate from 1024 or2048
independent observations. `input_geometry.csv` reports pooled direction mean,
input-radius CV, and normalized angular second-moment departure from I/n.
That departure is an uncentered angular second-moment diagnostic, not a test
of the complete spherical distribution.

Positive input/direction norms are implicit prerequisites. A clear finite/
positive guard before their divisions would make a pathological zero case
explicit; no such case is inferred here. This is an analysis safeguard, not
a reason to alter the measurements.

## Curvature restriction and recommended rank fix

Let G be the archived parameter Gram, and H_c the coefficient Hessian.
For G=V Λ V^T on its retained positive eigenspace, B=V Λ^-1/2 satisfies
B^T G B=I. Consequently B^T H_c B and B^T GN_c B are the Euclidean-
orthonormalized restrictions. The code implements this correctly, including
the distinction between a finite direction span and full parameter spectra.
The pooled matrices use the same mean-CE convention as the parameter gradient.
The coefficient heatmaps appropriately remain in actual-scale coordinates.

The current cutoff `max(lambda_max,1)*1e-12` introduces an absolute parameter-
unit floor. Rescaling every direction can then change retained rank even
though the physical span is unchanged. Prefer `lambda_max*relative_tol` for
positive lambda_max, explicitly require a nonempty retained span, and archive
the cutoff and Gram eigenvalues. This is a normalization correction, not a
request to change tolerance in response to a measured eigenvalue. Retain the
reported span dimension so a lower-rank restriction is never called a full
five-dimensional spectrum.

## Plots and completion sequencing

The singular spectra correctly divide each singular value by its matrix's
Frobenius norm and distinguish measured gradient, stored M_s, and actual
next write. These are shapes of three different objects; their agreement
would not reconstruct the next optimizer input from M_s.

The heatmap limit currently includes max|H| and max|GN| but omits max|H−GN|.
Opposite-signed entries can make the difference exceed both, silently
saturating the third panel. Include all three matrices in the common maximum
or explicitly show clipping; the first option preserves the intended shared
scale. The norm-dispersion plot correctly labels angular squared-kick CV²
and separates frames. Improved spherical agreement alone does not establish
radial finite-loss curvature or an optimizer benefit.

The watcher waits for explicit producer completion, rejects a producer
failure, and has a four-hour observation bound without restarting anything.
It runs `analyze_atlas.py` first and only starts geometry if that process
exits successfully. That preceding analyzer checks exact state/panel coverage,
finite arrays, labels/scales, bank units, and the joint identities flagged
in the runner review. The geometry script's lighter checks consequently
have a qualified predecessor in this execution path; standalone use should
preserve that prerequisite. Subprocesses run sequentially at one numerical
thread and refuse existing output directories, preserving earlier analyses.

Reviewed source hashes, before the recommended fixes:

- `analyze_geometry.py`: `0eb8f0c0f12eb17ef194b9589be76eaded04959b4d92b2a93a64bbd3487a282c`
- `postprocess_when_complete.py`: `ab28cbe119baef7b060ecd2b5035c161e6ae9b865d90d503ded22e3a62cc9b14`

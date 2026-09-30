# Independent prediction-profile result review

2026-09-28. Read the completed probe record, scalar analysis implementation,
raw per-token arrays, CSV and figure. Independently recomputed all frozen
progress weights and all phase-residual means, primary contrasts and paired
sequence SEs. No model call, new data or alternative endpoint. This is the
only new file.

## Verdict

**The declared negative SOAP-specific residual signature did not appear.**
All twelve method-by-stage-by-bank primary residual contrasts are positive,
opposite to that prediction. This is a substantive negative result for the
proposed readout, not a numerical or implementation failure.

The general mechanism remains unresolved because the Muon progress
interpolation controls are large and the direct nearly matched comparison
varies across banks. The result does not license an opposite common-token
mechanism, prove that all optimizer differences are ordinary Muon progress,
or refute every possible frequency-specific effect. Close this bounded
probe without additional samples, bins, contexts or fitted progress warps.

## Implementation and numerical qualification

- The probe completed all 320 declared score-forwards in 136.03 seconds.
  The first-sequence repeated-forward and scalar-CE discrepancies are
  exactly zero; the forecast was about 200 seconds, below the fixed cap.
- All six method interpolation weights and both Muon-control weights
  recompute exactly from the old-bank calibration values. No new-panel
  refit or extrapolation is present.
- The saved token/frequency hashes and all four frozen helper hashes still
  match. The score support is `[97,1966,3945,3619,2962,3795]`, summing to
  16,384 targets. The primary broad groups contain 5,911 and 6,757 targets.
- Independent reductions of phase-residual global means, grouped contrasts,
  cluster SEs and bin-to-total reconstructions agree with the saved analysis
  to at most 1.4e-17.

The primary contrast is correctly computed as a ratio of pooled rare sums
to rare counts minus the corresponding common ratio, rather than as an
unweighted mean of per-sequence ratios. Its sequence influence term uses
both groups jointly, preserving their covariance and the paired model
losses. The reported `sqrt(n/(n-1)*sum(influence^2))` is the appropriate
nominal sequence-cluster SE with the calibration weights held fixed. It
does not include uncertainty in the old-bank interpolation or correlations
between adjacent sequences; those limitations are correctly disclosed.

## What the readout says

Positive below means that the candidate has less mid-rare improvement
relative to common targets than its fixed Muon secant reference:

| Candidate | Step 500 contrast | Step 900 contrast |
|---|---:|---:|
| PD | +.05353 | +.05884 |
| SOAP | +.03717 | +.02315 |
| SOAP-PD | +.05269 | +.04338 |

Every bank-specific method contrast is also positive. These are not twelve
independent trials: the same inputs and overlapping Muon references enter
many cells. The consistency is descriptive, not a binomial significance
calculation.

The fresh raw same-step profiles help explain why progress correction
matters. At step 500 the rare-minus-common improvement relative to Muon is
-.00693 for PD, -.02645 for SOAP and -.04952 for SOAP-PD. At step 900 the
same raw contrasts are instead +.04909, +.00820 and +.02487. Thus the new
panel itself does not exhibit an invariant rare-target leaning even before
the secant adjustment. This does not erase the older final-checkpoint table,
which described a different stage and score panel.

Muon's own ordinary progress is strongly frequency-dependent and nonlinear:

| Muon interval | Overall NLL change | Mid-rare minus common change |
|---|---:|---:|
| 200→500 | -.69153 | -.94325 |
| 500→900 | -.20342 | -.31061 |
| 900→1300 | -.07111 | -.04266 |

These quantities come from the same saved fresh scores, with no new fit.
They make it unsurprising that a broad progress secant can alter a
frequency-gap interpretation. They do not prove that a particular local
interpolation bias accounts for each candidate's residual.

## The controls prevent a stronger mechanism claim

The own-Muon interpolation contrasts are +.01828 at M500 and **-.06521 at
M900**. The latter is negative in both banks (-.06002/-.07032), and its
magnitude is comparable to or larger than the candidate residuals. A
frequency-shaped residual therefore exists even when only one optimizer's
trajectory is involved. The controls are sensitivity profiles, not rigorous
bounds and not quantities to subtract selectively.

The prespecified direct SOAP500−PD500 comparison gives -.01952 with nominal
sequence SE .01795; bank values are -.04713 and +.00961. It consequently
does not supply the bank-consistent rare advantage that could independently
resolve the secant ambiguity. Their old-bank total NLLs were nearly equal,
but on this new panel their pooled global difference is -.01662; "nearly
matched" was never an exact fresh-panel match.

In particular, do not promote the smaller positive SOAP residual compared
with PD into a rescued successful negative-sign hypothesis. The absolute
predeclared signature failed, the stacked method is not uniformly more
negative than PD across banks/stages, and the direct control varies by bank.

## Scientific decision

The exploratory interpretation that SOAP's advantage is specifically
rare-target learning should no longer carry mechanistic weight on the
basis of the old final table alone. The factual table remains preserved.
The new probe shows that stage, ordinary progress and score-bank composition
materially affect that reading.

It would be equally premature to replace it with "geometry specifically
helps common tokens." Positive residuals against wide secants are not a
causal attribution, and the prescribed controls do not isolate a unique
prediction regime. No equivalence or absence claim follows from this
32-sequence, one-training-seed comparison.

This is the stopping outcome specified in the protocol: the hoped-for
regime-specific premise is unsupported, and its mechanism is unresolved.
Retain all method/control/bank reversals and move to a distinct question
grounded in the main trajectory interventions. Do not enlarge this panel
or search another stratification to obtain the desired sign.

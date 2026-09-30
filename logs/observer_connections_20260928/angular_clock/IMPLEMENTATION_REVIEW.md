# Independent implementation review of the fixed angular-clock audit

2026-09-28. Reviewed `PROTOCOL.md`, `probe.py`, the earlier factorial's arm
mapping/pairing code, and the frozen trainer/model/measurement source. No
geometric outcome, generated result, model or optimizer was evaluated in
this review. The only new file is this review.

**Verdict: go. No critical geometry, state-index, head-shape, decay, or primary
contrast bug found.** Two small pre-run clarifications below do not change
arms, quantities, criteria, or scientific scope and need no further review.

## Checked implementation

- **Fixed family:** extracts the original four ARMS without executing its
  analysis; retains every 9→10, 46→47, 83→84 pair. This is twelve pairs, not a
  selected subset. Token counts are checked at full-batch steps before the
  shortened terminal batch.
- **Timing:** current weights come from the kept post-step checkpoint, next
  weights from its next saved write; LR comes from the next step's record.
  Scalar spectral radii are explicitly labelled incoming step t−1 and not
  silently equated with kept step t.
- **Architecture:** frozen Attention forms `(B,T,8,64)` and normalizes the last
  dimension before transposing heads. Q/K output-row chunks `(8,64,512)`
  therefore match the actual per-head positive-scale symmetry. Whole-matrix
  geometry is supplementary. RMS epsilon and the absence of a prediction-
  distance claim are acknowledged.
- **Written geometry:** computes d from the actual saved next weights in
  FP64, directly computes normalized chord, and separately retains radial
  and tangential components. `2*r*radial + ||d||²` is the signed exact radius-
  square increment. Zero, radial and tangent synthetic cases cover the basic
  identities without importing a model.
- **Decay:** frozen `muon.py` applies decoupled multiplicative decay before
  adding the adaptive step, and the config requires that active decay mode.
  Hence d + eta*wd*w reconstructs the adaptive displacement up to FP32 write
  rounding. The label preserves that qualification. `distributed.py` records
  eta*wd*||w_before||, so the supplementary scalar radius inversion is valid.
- **Pairing and provenance:** compares metadata with the prior qualified
  factorial, rereads completed status, verifies selected JSON byte hashes,
  checks source hashes, confirms checkpoint config/step/dtype/shape, hashes
  consumed body tensors, and checks checkpoint size/mtime for changes.
  Full optimizer/embedding blobs need not be traversed just to hash unused
  contents; the records state precisely which tensors are consumed.
- **Primary outcomes:** constructs every matched head ratio first, then its
  median over all 64 Q heads or all 64 K heads separately. P1/P2 use only
  steps 46/83, each beta or LR as declared. The RMS summaries do not determine
  acceptance. Individual head ratios, extrema, and threshold counts remain.
- **Resources:** no model/optimizer calls, no CUDA access, at most two numerical
  threads, mmap reads, first-pair forecast and wall checks, preserved failure
  output. All outputs stay in the designated audit directory.

## Two pre-run clarifications

1. The protocol says the geometric identities use 1e−10 relative error, but
   the implementation uses normalized relative radius error and **absolute**
   chord-squared error. An absolute chord identity tolerance is reasonable
   near zero; state those two tolerances explicitly before execution. This
   changes no geometric quantity or scientific threshold. Alternatively,
   retain an appropriately guarded relative chord diagnostic as well.
2. The source loop checks every existing file but silently skips absent ones.
   Make the active PD and SOAP dependency files mandatory along with the five
   current required source files (using their actual frozen filenames). This
   closes a provenance assertion gap if an active dependency were missing.
   It is not a claim that any file is presently missing or mismatched.

No additional review or permission cycle is needed after those clarifications.
A failed materiality prediction should be reported as fixed; geometry output
must not be used to choose a different aggregation, state, or threshold.

Pre-run response from the root: both points were addressed. The protocol now
states the absolute chord-squared tolerance explicitly, and the required
source set includes `data_norm_muon.py` and `muon.py`; this frozen implementation
keeps SOAP routines in `muon.py`, so no separate SOAP source file is required.
The execution can proceed under the approved bounded scope.

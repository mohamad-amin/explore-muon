# Does radius adaptation hide a change in effective Q/K step size?

2026-09-28. Retrospective comparison fixed before root reads its geometric
outcomes. No new training or model forward. Independent high-level discussion
is complete in `../next_direction_peer/GEOMETRY_REVIEW.md`.

## Decision argument

The earlier four-arm momentum×LR comparison found nearly the same late
short-momentum loss advantage after increasing nominal body movement 43%.
New main warmup results favor different schedules at 92 and 184 updates.
Neither observation identifies a causal step clock. A possible missing
variable is weight radius: normalized Q/K maps can grow in norm while their
relative angular step falls. If the radius grows with LR, equal timing in
steps may coexist with similar angular movement despite different nominal
parameter distances. This connects the old norm-adaptation discussion to
the new momentum result; it is not a new normalization theory.

The alternative is that increasing LR still produces a comparably larger
Q/K angular step, so radius compensation cannot explain the weak LR
interaction in this family. A second question is whether shorter momentum
changes the angular step despite the identical prescribed direction norm.
Neither result by itself identifies what causes a training advantage.

An independent reviewer inspected selected 2×-horizon Q/K states as a
discovery screen. Those results are not part of the test below. The fixed
test uses the already qualified 1× SOAP-PD beta×LR factorial, all twelve
retained adjacent-state pairs, with no selection of favorable layers/heads.

## Fixed family and quantities

Use the four exact arms listed in `../momentum_lr/analyze.py`: beta {.9,.8}
crossed with LR {.028,.04}, 16M batch, seed 260925, Ada. States 9→10,
46→47 and 83→84; each run is complete. Recheck pairing against the saved
metadata and consumed source hashes. Load only saved tensors, mmap on CPU.

For each of the 48 body matrices retain pre/post norms, actual delta norm,
radial component, tangent component, cosine and normalized-weight chord.
For Q and K also split output rows into all eight heads, exactly as the
forward code does. Headwise Q/K is primary because RMS normalization acts
per head. The other matrix kinds are descriptive context, not individually
scale-invariant functions. This weight-space quotient metric is not a
prediction-space distance, and RMSNorm epsilon breaks exact scale symmetry.

For a flattened weight w, actual next weight w', and d=w'−w, define

    r = ||w||, r' = ||w'||,
    a = <w,d>/r, b² = ||d||²−a²,
    chord = ||w'/r'−w/r||,
    cos(theta) = (r+a)/r'.

Compute chord from normalized vectors, not cancellation of nearly equal
norms. Check ||w'||² = (r+a)²+b² and chord² = 2−2cos(theta), using FP64.
Also retain the decay-corrected displacement d+eta*wd*w, marked as an
approximate adaptive write because FP32 rounding remains. No optimizer
state is reconstructed. Save each matrix/head and fixed kind aggregates.

Supplement with the 24 spectral-panel matrices at steps 1/25/50/75/92.
The frozen trainer records decay_norm=eta*wd*||w_before||; recover that
radius and the nominal relative adaptive norm from those scalars. These
are incoming weights at the numbered update, while kept checkpoints are
after their saved update. Never silently align their different time indices.

Primary contrast is the median of matched headwise chord ratios, over all
64 Q heads and separately all 64 K heads. Keep every head ratio, matrix-kind
RMS summaries, and initial/late scalar radius records. Ratio of RMS chords
is secondary and cannot replace the primary median of paired ratios.

## Predeclared material readings

1. Radius compensation: median high-/low-LR paired head-chord ratio < 1.20 at both 46 and 83,
   separately for Q and K and at both betas. The nominal LR ratio is 1.4286.
2. Momentum-dependent angular step: median short-/long-momentum paired head-chord ratio > 1.10
   at both 46 and 83, separately for Q and K and at both LRs.

These are materiality criteria in a retrospective one-seed family, not
significance or confirmation thresholds. Step 9 and every individual head
remain in the report. If a reading fails, do not rename it from an aggregate,
choose layers, fit a time warp or enlarge the family. Passing reading 1
supports a missing angular-movement alternative to nominal arc; passing
reading 2 establishes a concrete movement difference under equal prescribed
norms. Neither establishes Q/K as the mediator or a universal scheduling rule.

After either outcome close this fixed comparison. Any functional or causal
test would require a separate decision with its own competing explanation.
Do not intervene on weights, gains, momentum, clocks or learning rates here.

## Numerical and resource boundary

CPU only, at most two numerical threads, no gradients or model import.
Check all values finite, masks exactly 48 body matrices and 128 Q/K heads,
positive radii, the radius-squared identity to 1e−10 relative error and
the dimensionless chord-squared identity to 1e−10 absolute error. The latter
avoids dividing by a vanishing chord in the synthetic no-change check.
Hash consumed body tensors and source/JSON inputs. Existing pairing/source
qualification is evidence to verify, not permission to skip provenance.

The work reads at most twelve saved checkpoint/next-weight pairs and a small
fixed set of scalar JSONs. Forecast after the first pair; stop if projected
total exceeds 180 seconds, and check the wall boundary after each pair.
Keep any failed attempt. No GPU, job submission, other-directory output or
sealed-surrogate access. All new outputs stay in this directory.

## Review amendment before extraction

The initial root draft used a ratio of RMS chords. Independent review
recommended the median of matched head ratios, and that is adopted before
any twelve-state outcome is read. The original draft is retained as
`PROTOCOL_draft_rms.md`; both summaries will be exported and only the reviewed
median contrasts determine P1/P2. No threshold or state changes.

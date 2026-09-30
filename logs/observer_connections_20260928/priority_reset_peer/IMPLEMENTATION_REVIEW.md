# Independent implementation review before prediction-profile interpretation

2026-09-28. Reviewed `prediction_profile/PROTOCOL.md`, `probe.py` and its
frozen model/data/probe helpers. Compared the helper bytes with the qualified
`body_aux/source` copies. **No fresh scores, result/status outputs or model
calls were inspected or executed.** This is the only new file.

## Judgment

No critical scientific implementation flaw found. The code implements the
declared ten-state, independently calibrated forward-profile experiment.
The new scoring panel cannot alter checkpoint selection, frequency strata
or progress-interpolation weights. Interpretation should wait for successful
completion, the declared numerical/cost gates and the scalar analysis checks
below.

## Calibration and design

- All calibration NLLs come from archived `scientific/steps` files before
  model scoring starts; hashes and coefficients are saved in `prepared.json`.
  Candidate500 uses M500/900, candidate900 uses M900/1300, and both required
  own-Muon controls use the declared wider brackets. Every coefficient is
  checked to lie in [0,1], with no extrapolation or fresh-panel fitting.
- The additional direct S500−PD500 comparison is justified independently
  by their old-bank mean losses being only .001908 apart. It was written
  into this protocol before fresh scoring. It is a useful supplementary
  nearly matched-mean comparison, not an exactly matched state or a new
  feature selected after observing the panel.
- The exact historical frequency reference is used: 50M training-partition
  tokens starting at 2.5B, with all six original edges and target-token
  lookup. This improves on the earlier proposed 32M-prefix approximation.
  It is data-only and independent of every candidate's score.
- Score offsets 3.12B and 3.16B exceed every selected checkpoint's recorded
  training exposure. The same 32-by-512 input/target arrays are reused for
  all ten states and saved with hashes. These are held-out positions in
  the same training partition, not a separately sealed test population.
  The protocol correctly keeps this distinction.

## Forward execution and preservation

All four imported helpers (`model.py`, `data.py`, `gn_probe.py`, `__init__.py`)
are byte-identical to the previously qualified body-auxiliary copies.
The new probe records their hashes; this independent byte comparison supplies
the source-equality qualification rather than merely recording identifiers.

`P.build_model` explicitly turns off body input-statistic collection and
constructs plain Linear layers. The code separately checks architecture
equality after removing measurement-only tracking flags. Reusing that model
with strict `load_state_dict` for the remaining checkpoints is therefore
appropriate: the model architecture is common, and the tracking buffers
are nonpersistent and irrelevant to the evaluation forward. Eval mode,
disabled parameter gradients and `torch.inference_mode()` prevent training
or statistic updates. All tensors are loaded on CPU.

The source uses CPU FP32 parameters and logits, with no autocast. Token CE
is the same helper used in qualified earlier probes. The first sequence's
repeated forward and scalar-model CE checks use the declared 2e-6 tolerance.
Every scored loss vector must have 512 finite entries. The ten checkpoint
steps, parameter counts and beyond-training score offsets are checked;
model tensor hashes, file size and mtime are retained. No checkpoint object
is passed to an optimizer or mutated.

The chunked frequency count has bounded allocation and verifies exactly
50M observed tokens. `digitize` implements the specified lower-inclusive
bins; all 16,384 scored targets must belong to exactly one of six bins.
Saved losses require only about .63 MiB. One-sequence FP32 logits require
about 103 MB; memory-mapped checkpoint loading and replacement of the model's
weights avoid keeping ten full models or all logits resident.

## Timing boundary

The forecast is conservative and correctly indexed: after two actual score
sequences it counts 318 remaining score forwards, uses the maximum measured
forward duration including the repeated qualification call, and allows nine
additional loads at the first-load cost, all with 25% margin plus overhead.
The two additional numerical-check forwards have already elapsed and are
included in the clock at forecast time. Qualification sequences stay in
the fixed panel.

Torch intra-op/inter-op counts are explicitly capped at 2/1. Environment
variables for other numerical libraries use `setdefault`, so the launcher's
explicit two-thread environment is part of the resource guarantee if the
parent environment otherwise sets larger values.

One narrow wording qualification: the in-script 600-second check runs at
status points, normally every eight sequences, rather than asynchronously
interrupting a forward. Without an external timeout it is a checked wall
boundary with possible one-block overshoot, not a strict OS-level deadline.
This does not change the scientific scope or justify extending the panel;
the parent should retain its external supervision/resource boundary.

## Checks still needed in the analysis, not additional model work

The probe verifies token/bin coverage but does not yet reconstruct total NLL
from each bin's sums; that declared gate belongs in the independent scalar
analysis. Sum in FP64, retain both input banks and all six bins, and report
the same externally defined broad contrast for every fixed state/control.

Use the complete paired per-sequence data when computing the ratio-of-sums
contrast and its nominal cluster SE, including covariance between the two
frequency groups and across models. A bin with sparse/zero sequence support
must not silently become a new sampling unit. The 32 sequences and two
contiguous banks do not create independent training replicates.

The own-Muon interpolation controls are descriptive sensitivity profiles,
not formal bounds. Their inclusion should prevent a strong phase-specific
interpretation when ordinary nonlinear progress produces comparable shapes;
they must not be adaptively subtracted or selectively discarded. The
declared stop on an inconclusive result and prohibition on expanding the
sample/features remain appropriate.

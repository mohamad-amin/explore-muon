# GCP validation run (2026-09-25)

Infrastructure check for the Google Cloud image `explore-muon-v2` (see `cloud/`), not part of the
improvement-search waves. It re-runs an existing arm unchanged on a Spot `a3-highgpu-4g` (4×H100):

- Frozen sources and `frozen_manifest.json`: byte-identical copies from `improve_w9_20260925`.
- Arm `M_lr0.007_s260924_h100`: `config.json` and tiny data copied from
  `improve_w9_20260925/M_lr0.007_s260924_g20` (tuned Muon, LR 0.007, seed 260924, width 512).
- The controller's own gates apply: initial-weight hash and data manifests must equal the
  frontier-norm reference run.

Existing results for this seed and config: RTX 6000 Ada 3.70689 (`improve_w9`), L40S 3.70427
(`improve_w7`). The H100 result is a third hardware point; it is not a new scientific comparison.

## Result (2026-09-25 20:43 UTC)

Spot `a3-highgpu-4g`, `us-east4-b` (us-central1 zones and us-east4-a were stocked out).

- All controller gates passed: tiny CUDA run with replica audit, and five-step full-size qualification
  with initial-weight hash `3c51d98e3b54…` and data manifests equal to the reference.
- Final validation NLL **3.70386** at step 1469, against 3.70689 (RTX 6000 Ada) and 3.70427 (L40S) for
  the same seed and config. That is within the existing cross-hardware spread.
- Steady step 0.341 s. Scientific stage 616 s for 1469 steps with 60 spectra; whole pipeline about
  13 minutes. The VM ran 16 minutes, about $7 at the Spot list price (~$26/h for 4×H100).

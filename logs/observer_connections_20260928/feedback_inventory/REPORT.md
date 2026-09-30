# What the retained tensors can support without a training replay

Subsequent scope correction: this targeted tensor inventory omitted existing
scalar EOS-Jacobian results at older1M/4Mstates. Those measurements are now
audited in`../linearized_edge/REPORT.md`. Missing reconstruction tensors in
the current16Mprofiles do not imply no prior differential experiment exists.

2026-09-28. Metadata-only inspection of22saved tensor files in the GN,
GN-mix, floor and transport archives. All loads used CPU memory mapping;
no tensor/model arithmetic, GPU, or training was run. `inventory.py` and
`result.json` retain paths, file metadata, tensor schemas and source hashes.
This is a targeted inventory, not a claim to have inspected every project
checkpoint. The process completed successfully (session60390, exit0).

## Available information

- The original`gn/` files retain BF16mapped directions and FP16Kronecker
  eigenbases, but no raw`gd` direction. A polar map is not invertible, so its
  input cannot be recovered from the saved mapped output.
- Later`gn2/`,`gn3/`,`gnmix/`,`gnwarm/` and the floor file also retain`gd`
  directions. These provide quantized fresh-gradient or constructed-momentum
  inputs at their particular1M/4Mstates. They permit conditional matrix-map
  sensitivity calculations, with qualification for quantization and mapping
  conventions; they are not the actual16Mnext-step inputs.
- The transport files retain BF16`Mstar`,`replay_mean`,`X_G`,`X_H` for two
  tracked1Mstep500states. The checkpoint supplies M and the accumulated
  displacement Q. These are richer than scalar summaries: they preserve
  specific GN/Hessian products along Q and a replayed stale-free momentum.
  They do not provide arbitrary Hessian products, the full operator, or the
  exact fresh gradient used in every scored constructed next momentum.
- The18new16Mprofiles retain scalar projections, not the top Ritz vectors.
  Their fresh curvature-root matrices are not saved by that probe. A rank0C
  sidecar from an online run is a different statistic from this fresh root,
  and does not recover every owner's historical cached roots or SOAP state.

The currently edited`one_step_gn.py` saves`gd` in every file it would create,
but the older actual files lack that key. The inventory therefore uses tensor
contents as authority rather than retroactively applying the latest source
schema to historical output.

## Decision

A limited matrix sensitivity calculation at older states is feasible. A
source-faithful common-state16Mfeedback measurement cannot be reconstructed
from the inspected scalar profiles alone. Replaying a newly chosen gradient,
root and stiff subspace would require another model experiment and numerical
qualification. It would answer a conditional new question, not recover a
previously measured derivative.

Do not turn this into a repair cascade. The independent priority review
identified a completed momentum×LRfactorial whose full trajectories test a
different explanation at much lower cost. That comparison is completed in
`../momentum_lr/`. The analytic derivative distinction remains valid and the
older transport tensors remain a possible future resource; neither licenses
an automatic replay or curvature-estimator branch now.

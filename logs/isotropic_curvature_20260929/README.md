# Isotropic-curvature trajectory measurements

**Complete:** 18 matrix panels, six joint states, all reductions and independent
numerical verification. Start with [REPORT.md](REPORT.md), the
[trajectory overview](run1/report_tables/trajectory_overview.png),
[finite-remainder overview](run1/report_tables/finite_remainders_overview.png),
and [all 18 numerical rows](run1/report_tables/TABLES.md).

Read [STATE.md](STATE.md) for the completion record and [PROTOCOL.md](PROTOCOL.md)
for the fixed scientific question. This study follows the user's request to
examine [Su's isotropic curvature model](https://arxiv.org/pdf/2511.00674)
through measurements along retained optimization trajectories. Results are
conditional on this recipe family, three snapshots, four score contexts and
selected finite Hessian spans; no new optimizer was trained.

The diagnostic distinguishes a static Hessian quadratic, predictive
Gauss–Newton curvature, and the finite Taylor remainder. It separates
left/output rotations from right/input rotations, and retains a regularized
whitened-coordinate comparison. All data and code are confined here.

- `source/original/`: unchanged original frozen model/data source.
- `source/adamw_spectra/`: separate FP64/explicit-attention diagnostic copy.
- `qualification/result.json`: numerical and source qualification.
- `instrument.py`: derivative, suffix and perturbation definitions.
- `run_atlas.py`: fixed18panel producer with retained raw observations.
- `run1/`: executed source/protocol copies, tokens, source metadata,
  per-panel tensor archives, token arrays and joint-step arrays.
- `run1/analysis/`: complete per-context/bank/pooled tables and signed figures.
- `run1/geometry_analysis/`: displacement moments, gradient/momentum/write
  spectra, and restricted curvature after Gram orthonormalization.
- `run1/transmission_analysis/`: known-Jacobian GN decomposition and raw energies.
- `run1/report_tables/`: compact tables and publication-format PNG/PDF overviews.
- `run1/verification.json`: independent reconstruction of the numerical tables.
- `MANIFEST.json`: final artifact sizes and SHA256 hashes, excluding documented caches.

At each selected matrix, `per_token.npz` axes are score context, direction,
signed multiplier, token for losses; directional slopes/Hessian/GN have no
multiplier axis. The projected Hessian is context×5×5; projected GN is
context×token×5×5. Coordinates are the recorded directions, not an
orthonormal parameter basis. Parameter and input-metric Gram matrices are
stored in `tensors.pt` for that distinction. The joint panel has the analogous
three actual up-weight directions and records individual curves as controls.

The tensors also retain full selected gradients, summed-sequence-loss output
adjoints, stored lagged momentum, actual writes, calibration inputs, score
inputs/preactivations/residuals, input covariance factors and singular values.
At early blocks, an output adjoint at one position includes effects on later
token losses; it is not labeled that position's own loss derivative.

Scripts use the project root's existing `.venv`. This run is complete; do not
restart it. Producers and reductions refuse to overwrite their output
directories. The recorded historical trajectory files remain immutable.
CPU only, two producer threads and one reduction thread; no training or GPU job.

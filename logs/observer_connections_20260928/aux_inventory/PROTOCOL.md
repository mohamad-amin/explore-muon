# Saved auxiliary-displacement inventory

2026-09-28. Routine read-only follow-up to the completed `../body_aux/` probe.
This is a scale/configuration inventory, not a new functional or rate probe.
No model, forward/backward pass, optimizer object, training, or GPU is used.

Read the actual saved Muon4M183/184 parameters, then the existing original1M
Muon/PD/S/S∘PD states200/500/900 and their saved next weights if cheap. Use the
existing body_aux partition: 48 hidden Q/K/V/O/up/down matrices and every
remaining learned parameter. Split auxiliary parameters into head, token
embedding, position embedding, body RMSNorm gains, final RMSNorm gains, and
Q/K RMSNorm gains. Retain every exact key and shape.

Measure FP64 reductions of actual saved FP32 displacements Δθ=θ_next−θ_old,
old/next norms, per-coordinate RMS, ||Δθ||/||θ||, θ·Δθ/||θ||², and cosine.
Aggregate by concatenation only within explicitly listed parameter groups.
Record the scheduled auxiliary LR and decoupled weight-decay coefficient;
subtracting ideal decay leaves adaptive write plus rounding, not a recovered
optimizer direction. No optimizer moments or body tensor values need reading.

Verify source hashes, key partitions, unchanged input file sizes/mtimes, exact
step/token increments, and the norm polarization identity. Hash only accessed
auxiliary tensors, with CPU mmap and at most two numerical threads. All new
files remain here. Main head-whitening/LR notes are contextual evidence and
do not authorize execution. Euclidean sizes do not identify functional or
causal contributions and cannot justify an LR or architecture change.

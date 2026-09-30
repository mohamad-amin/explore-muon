# Four corners of one actual training step

2026-09-28. Independent discussion: `../persistence_peer/NEXT_DECISION.md`.
This follows the completed archive analyses, before any measurement here.

The 4M Muon step183 has conflicting body-gradient descriptions: the
step-profile probe updates only hidden matrices, while river/hill loads the
whole next model; sample and precision also differ. The dense-lag archive
shows changes shared broadly across matrices. Auxiliary changes may drive
some of this, or sample noise/precision may explain the discrepancy.

On one saved183/184 pair, construct base, body-only, auxiliary-only and full
corners. Body is all48 Q/K/V/O/up/down matrices; auxiliary includes every
other learned parameter. Verify the partition and byte-exact reconstruction
of the full next state. Frozen observer source, CPU FP32 eager, same samples
at every corner, per-sequence next-token mean loss. No optimizer is constructed.

Before examining scientific outputs, use a one-sequence implementation/cost
check and compare corner outputs with direct checkpoint loads. If projected
measurement time exceeds15 minutes, stop after qualification and retain it.
Otherwise use exactly two disjoint banks of8 sequences each: never-trained
training-stream token offsets2,500,065,536 and2,600,065,536. The first begins
at the second held-out bank used by step_profile_probe.py; it is a small
subset, not a replication of its128-sequence population estimate.

Record per-sequence paired losses, bank means of body gradients, same-bank
cosines and cross-bank inner products. Decompose gradient change exactly as
body contribution + auxiliary contribution + interaction. Preserve absolute
norms and Gram terms; do not assign percentages when cancellation dominates.
Two banks do not support confident population correlations or precise CIs.
No sample-size escalation based on the observed effect.

Material and reproducible full-versus-body changes would establish that
auxiliary scope matters for this measurement. Negligible changes redirect
attention to sample size and precision. Neither result alone identifies a
training-rate mechanism or authorizes an optimizer/architecture intervention.

CPU only, at most2 numerical threads, no GPU or scheduler allocation. New
files stay here; saved gradients permit follow-up without repeating forwards.
The separate older observer's similar proposal has no live process and no
executed outputs as of13:43; its notes are preserved and not treated as a
running experiment. Main training and its state/protocol remain untouched.

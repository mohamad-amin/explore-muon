# Bounded saved-state inventory: input support and embedding movement

The completed factorial attributes most body–auxiliary interaction to the
embedding group and ordinary predictive overlap. This does not justify a
remedy. A cheap source/state check can clarify one distinct assumption before
any new model experiment: how much token-embedding movement occurs on rows
absent from the actual next training batch, and do those rows even appear in
the scored examples?

At the same saved183→184 pair, read exactly the next4M training input tokens,
the old/new embedding weights, and its stored Adam moments. Count row support.
For absent rows the new gradient is exactly zero in the untied input lookup;
predict their next update from moment decay, bias correction and decoupled WD,
and compare with saved actual displacements. This is arithmetic only, no
model or optimizer execution. Inspect all24 already-scored input sequences
for support overlap. No loss or rate claim follows from inactive movement.

Retain active/inactive row counts, displacement energy, nonzero moment state,
actual/predicted errors, exact token counts and provenance. A mismatch with
the zero-gradient recurrence invalidates the premise or implementation and
must be investigated, not fitted away. Report both absolute error and a
conservative FP32-write-scale bound. CPU two threads, one4M-token read,
expected under60s. This is a bounded follow-up inventory, not a new training
branch or an endorsement of sparse/lazy Adam.

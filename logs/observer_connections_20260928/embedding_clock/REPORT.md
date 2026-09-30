# Absent-row Adam momentum does not explain the measured embedding interaction

2026-09-28. Saved-state/input-support inventory only; no model forward,
backward, optimizer call or GPU. Runtime 0.63 seconds, two CPU threads.

The complete body/head/embedding/norm factorial found that embeddings account
for most body–auxiliary loss interaction at Muon 4M step 183→184. A tempting
alternative is that Adam's old moments move input rows which the current
batch does not observe. This inventory checks both the recurrence and whether
that movement could even reach the scored examples.

The same checkpoint pair as `../aux_partition/REPORT.md` is used. Its token
offset is 767,557,632; exactly the next 4,194,304 input tokens are counted.
The token embedding is untied from the head, so an absent input row has zero
new lookup gradient. Auxiliary AdamW uses LR .002, betas (.9,.95), epsilon
1e-8 and decay .01. The saved moment step is 183; bias corrections for the
next update use 184. No state is modified.

| Input embedding rows | Count | Share of actual displacement energy |
|---|---:|---:|
| Present in next training batch | 47,370 | 96.11097% |
| Absent | 2,934 | 3.88903% |
| Absent with zero second moment | 269 | 0.000000451% |

The absent-row displacement predicted from decaying old moments and
decoupled weight decay agrees with the saved write: maximum absolute error
1.47e-8, relative Frobenius error 3.30e-6. The maximum error is 7.34% of the
predeclared conservative FP32-write bound. This verifies a real standard-Adam
effect; movement without current exposure is not itself useless or harmful.

The original two diagnostic banks contain respectively 4 and 5 input positions
using rows absent from the next training batch. The fresh banks contain
**zero** and **one** such positions out of 2,048 each. In the zero-overlap bank,
the body/embedding interaction is still +0.0135413, and adding the embedding
step after the body increases loss by +0.00601685 (all four sequences).

That is a structural exclusion: moving unused rows of an untied input lookup
cannot change predictions on this bank at all. Absent-row momentum therefore
cannot explain its observed embedding-group interaction. The group also
contains position embeddings, so this does not separately attribute the
effect to token versus position embeddings; that extra split is not needed
for the exclusion. The absent rows can matter on future inputs, and nothing
here licenses sparse/lazy Adam, stopping moment decay, or a training claim.

`result.json` retains optimizer settings, tensor hashes, support counts,
recurrence errors and runtime. `counts.npz` retains row counts and support
masks; `inventory.py` specifies the checkpoint/token source and arithmetic.
No sample extension or model experiment follows this inventory.

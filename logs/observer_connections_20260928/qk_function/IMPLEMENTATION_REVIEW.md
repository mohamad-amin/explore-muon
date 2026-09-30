# Pre-run implementation review of the Q/K functional bridge

2026-09-28. Read the fixed protocol, `probe.py`, qualified helper model-loading
code, and the independent readout discussions. No scientific outcome or
checkpoint tensor was read and no model call was run for this review.

**Verdict: go. No critical mask, indexing, finite-response mathematics,
resource, or provenance bug found.** The protocol explicitly chooses the
KL factorial and QK-last material-loss readings before data, while retaining
my proposed RMS/symmetric-credit quantities descriptively. That is a valid
pre-scoring choice, not a substitution after results. No additional review
or permission cycle is needed.

## State and partition checks

- The four ARMS and their order are taken from the previously qualified
  LR×beta factorial; current metadata must equal that prior record.
- The code uses saved W46 and W47, verifies both step/token indices, and
  never constructs a gradient, lagged momentum or optimizer direction.
- Suffixes `.attn.q.weight` and `.attn.k.weight` select all 16 Q/K weight
  tensors. Q/K normalization gains are excluded from that mask and belong
  to the 68-tensor complement, as declared.
- All 84 parameter names agree across model/base/next state. Each first
  input verifies the exact tensor choice in all four masks. Mask 3 therefore
  reconstructs the full actual next model, including auxiliary writes.
- `build_model` loads by ordinary parameter copy, uses plain linears rather
  than input-statistics buffers, and returns a FP32 CPU model. Later corner
  copies mutate the in-memory model, not the mmap checkpoint tensors.
- Restoring base and requiring an exactly repeated first base logit output
  is appropriate for this deterministic eager CPU evaluation.

## Readout algebra and array mapping

The returned order `(loss, kl, conditional, gram, parts, linear)` matches
exactly the insertion order of the six saved arrays. Shapes match the
four-state, sixteen-sequence, 512-target design.

For log probabilities ell_j, `responses=ell_j−ell_0` differ from raw logit
responses only by a vocabulary-constant shift per token. The code centers
under p0 before covariance, so that irrelevant shift disappears. It retains
all three Q/R/full responses and their complete 3×3 finite covariance,
including signed overlap. Symmetry is imposed by paired filling, and PSD,
additive-response and mixed-response direct identities test the algebra.
These quantities are correctly called finite-response moments, not GN.

The CE/KL identity is correct: the label-linear term is
E_p0[ell_j−ell_0]−(ell_j−ell_0)_y and forward KL is the negative expectation,
so their sum is exactly L_j−L_0. The conditional array order is
`[KL(p_R||p_full), KL(p_Q||p_full)]`: QK-last then complement-last.

`logsumexp(ell_Q+ell_R−ell_0)` is exactly the additive overlap term
L_add−L_Q−L_R+L_0. The direct mixed-logit pairing J and all implemented
relations among interaction, overlap, mixed finite loss and KL have the
correct signs. In particular the code does not mistake a favorable J for
helpful finite mixed loss; both are separately retained.

FP64 CE/response reductions, centered covariance, finite-value tests,
nonnegative KL/PSD tolerances, and the fixed per-token FP32 CE comparison
match the protocol. The synthetic test covers arbitrary distinct logit
shifts across corners and tokens, not merely a common global offset. The
root reported its outcome-free synthetic test passed before scientific
execution. No temperatures, norms, directions or covariances are rescaled
to manufacture an identity or favorable contrast.

## Inputs, provenance and resource boundary

- Both fresh banks are in the 3.2B non-wrapping stream; source manifest,
  maximum training exposure, target-token endpoints and recorded earlier
  intervals are checked before scoring. All states receive identical x/y.
- Completed run state, full metadata equality and all archived source hashes
  are checked, with active model/Muon/PD/training/data files mandatory.
- Helper bytes must match the already qualified confidence helper. Each
  base/next checkpoint's size/mtime and every consumed body tensor hash
  must match the previous angular audit at step 46. Full model tensor hashes
  additionally record the newly consumed auxiliary contents.
- All parameter gradients are disabled; scoring is under inference mode,
  CUDA visibility is empty, and numerical threads are capped at two.
- 64 complete four-corner sequences yield 256 scientific forwards plus one
  restored-base qualification forward. The first-sequence timing includes
  all vocabulary reductions; forecast includes remaining sequences,
  remaining load/hash overhead, a margin and fixed overhead allowance.
- The 600-second boundary is checked after each complete sequence, and cost
  stops preserve partial arrays and original failure information. This is
  a checked compute bound, not proof of scientific usefulness.

The subsequent analysis must preserve raw small/negative-within-tolerance
KLs and treat unresolved ratio denominators as unresolved, not choose an
outcome-dependent floor. Both banks, backgrounds, LR contrasts and all
conditional loss signs must remain visible. A failed bridge reading closes
this fixed question without a per-head or new-state expansion.

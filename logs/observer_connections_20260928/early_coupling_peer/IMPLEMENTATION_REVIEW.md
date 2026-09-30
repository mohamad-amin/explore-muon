# Independent early-split implementation review

2026-09-28. Reviewed `early_split/PROTOCOL.md`, `probe.py`, and frozen
helper bytes. No fresh scientific scores or result/status outputs were
read, and no model/GPU call was made. This is the only new file.

**Verdict:** no critical implementation error found. The code measures the
new exact finite-logit split, independently computes J, and reproduces the
old measurements without expanding states or choosing new outcomes.

## Log-probability construction is gauge-equivalent

For each corner c, write `lp_c = z_c - logsumexp(z_c)`. The implemented

    raw = lp_B + lp_A - lp_0

equals `zadd=zB+zA-z0` minus the per-token scalar
`logsumexp(zB)+logsumexp(zA)-logsumexp(z0)`. Therefore:

- `raw-logsumexp(raw)` is exactly `logsoftmax(zadd)`;
- `logsumexp(raw)` is exactly
  `logsumexp(zadd)-logsumexp(zB)-logsumexp(zA)+logsumexp(z0)`, the
  label-independent additive prediction interaction;
- the synthetic CE uses the true target from those normalized additive
  log probabilities and has the same value as CE computed from raw zadd.

This avoids a large raw-logit storage requirement without changing the
specified path. The code correctly retains `logsumexp(raw)` before
normalizing the synthetic response; it does not mistakenly discard the
overlap term by treating all normalized logsumexp values as zero.

Similarly, `lpF-lpB-lpA+lp0` differs from the raw-logit mixed response only
by a per-token constant. Pairing it with `p0-onehot(y)` removes that
constant. The directly computed J is therefore the desired gauge-invariant
base-residual mixed-response scalar.

## All three identity checks have the correct signs

The code measures interaction I from the four CE corners, mixed loss M
from `CE_full-CE_add`, overlap O from `logsumexp(raw)`, and forward KLs
from the same base probabilities. Its checks are:

    I - O - M = 0,
    M - J - KL_full + KL_add = 0,
    J - I + KL_full - KL_body - KL_aux = 0.

These match the protocol exactly. They are checked per token at 1e-10,
not merely after averaging away errors. CE, log probabilities, overlap and
KL reductions are FP64 after the frozen FP32 model forward. Forward KL
orientation is consistently `KL(p0||pc)`, including the synthetic response.
Nonnegativity is checked without clipping the divergences; final arrays
must be finite.

This direct computation of J now supplies a genuine numerical identity
check in addition to the previous identity-based scalar derivation. It
does not change the distinction between J and finite mixed loss M.

## State, corner and array indexing

The state list is exactly q09/h09 at steps 9 and 46. The lookup into the
old eight-state array uses `(label,step)` tuples, avoiding confusion caused
by the different four-state order in this follow-up. Old inputs occupy
rows 0–7 unchanged; fresh banks occupy rows 8–11 and 12–15.

The original input/target hashes are verified before reuse. Every saved
endpoint model tensor hash must match the completed dose probe's provenance,
and step numbers must be s and s+1. New score offsets are beyond the
checkpoint training prefix. All states use the same 16 input/target arrays,
saved with hashes.

The body/auxiliary partition remains exhaustive and disjoint: 48 body
matrix tensors and 36 remaining tensors. Every corner copies every tensor
from the appropriate base or next state. Masks 0/1/2/3 mean base/body/aux/full;
switching between corners cannot leave a previous corner's parameters
accumulated. Exact initial tensor comparisons and strict key equality
qualify that construction.

On every original input, each per-token CE corner and forward-KL corner is
compared with the correctly indexed old arrays. The 2e-5 maximum and 2e-6
per-sequence-mean tolerances match the predeclared gates. These comparisons
also test source/forward consistency on both original banks across all
four selected states.

## Source, execution and scope

All four helpers are byte-identical to the completed dose-function source.
The qualified builder disables body input-statistic collection and constructs
the shared CPU FP32 evaluation model. Gradient disabling and inference mode
prevent optimizer/statistic updates; checkpoint tensors are only read and
copied into the local model.

Four FP64 vocabulary log-probability arrays are held for one sequence,
plus temporary mixed/additive arrays. This has a larger working-memory and
reduction cost than scalar CE alone, but the declared first-complete-sequence
forecast includes those costs. After one of 64 four-corner sequences it
correctly forecasts 63 remaining sequences and three more paired loads,
with the declared margin and overhead. The 600-second boundary is checked
after each sequence, consistent with the protocol's wording. Numerical
thread limits are as in the qualified dose probe and its launch environment.

The fixed analysis must keep old discovery/reproduction separate from the
two new banks. The three prospective fresh-input predictions and .01
materiality floor remain unchanged. The experiment introduces no subgroup
factorial, extra momentum, state search or response rescaling.

The only new meaning available after successful execution is the finite
loss effect along the declared synthetic-logit path and its state/stage
contrast. zadd is not asserted to be a realizable parameter update, and
neither identity nor sign alone supplies a training-rate mediator or an
auxiliary-group remedy.

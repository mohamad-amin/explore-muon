# Q/K functional relevance cannot be recovered from the current 16M scalar archive

2026-09-28. Bounded read-only inventory of the actual-step profiles/spectra,
direction Grams, relevant producer code, and completed observer reports.
No checkpoint tensors were loaded, no model/gradient/GPU was called, and no
new computation beyond JSON schema/path inspection was made. This is the
only new file.

**Decision: no-go for an artifact-only discriminator of whether the larger
Q/K turns carry substantial actual next-step predictive movement or useful
loss change in the fixed beta×LR family.** The archive preserves whole-body
function-level totals for half the family, but not the Q/K decomposition.
New function-level measurements would be necessary; neither more scalar
reweighting nor older counterfactual Grams can fill that missing information.
This does not automatically justify a new forward factorial or intervention.

## Exact coverage of the fixed family

Root directory below is
`logs/muon_spectra/second_order_audit_20260926/`.
The inventory matched each JSON's `item` field to the full arm path and
steps 9, 46, 83, rather than inferring coverage from a plot or README.

| Actual angular-clock arm | step_profile coverage | step_spectrum coverage | Per-kind/head actual functional terms |
|---|---|---|---|
| `soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada` | 9/46/83 in `step_profile/` | 9/46/83 | Not saved |
| `soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada` | 9/46/83 in `step_profile_mom/` | 9/46/83 | Not saved |
| `soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.04_mom0.9_s260925_ada` | None | None | Not saved |
| `soaudit_strength16m_20260928/SPD_a0.5_b16M_lr0.04_mom0.8_s260925_ada` | None | None | Not saved |

Files follow the precise naming convention
`{subdirectory}/{cohort}__{arm_basename}__{step}.json`. For example:

- `step_profile/soaudit_batch16m_20260927__SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada__46.json`
- `step_profile_mom/soaudit_prefilter16m_20260928__SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada__46.json`
- `step_spectrum/soaudit_prefilter16m_20260928__SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada__46.json`

A search for the two LR-.04 arm basenames in the second-order audit JSONs
found only `strength16m_arms.json`, a treatment/configuration list, not a
functional measurement. This is an inventory of these relevant archived
families, not a claim to have opened every project file.

## What the existing 16M measurements actually contain

`step_profile_probe.py` constructs `actual` from the 48 saved body-matrix
writes W[s+1]−W[s], including decay and parameter-write effects. Therefore
`directions.actual` is the right time-indexed actual body direction; it is
not a mapped lagged momentum. Its saved fields are whole-body:

- `norm`, `slope`, `curvature`, `c_star`, `quality`;
- `curvature_curv_set`, `rayleigh_over_top`;
- `energy_top{1,4,16}` and `curvature_top{1,4,16}`.

Slope and directional GN curvature are scored on a held-out training bank;
Ritz projections use a separate validation curvature bank. These totals
provide no per-kind slopes, Q/K-only quadratic, Q/K-versus-complement cross
term, per-head responses, or logits/JVPs. The top Ritz vectors and the fresh
root matrices are not retained in these JSONs. A parameter-energy share
cannot be multiplied into a functional total to recover the missing terms.

`step_spectrum_probe.py` retains `directions.actual.{norm,theta,energy,slope,
curvature,slope_total,curvature_total}`. These are scalar spectral measures
of the **whole body direction**. They partition by Krylov spectral nodes,
not by parameter kind. Their basis vectors are discarded; no Q/K projection
can be reconstructed from the nodes and weights alone.

The `momentum`, `muon`, and `pd` directions in the profiler use M_s, whereas
the actual write uses the next incoming training gradient through M[s+1].
Using one of those maps as a proxy for the Q/K share of the actual write
would revive the already documented lagged/next-momentum error. Moreover,
the `gradient_pair` next state applies only the body displacement, with
auxiliaries fixed; it is not the full realized next-model gradient.
See `../momentum_maps/REPORT.md` and `../feedback_inventory/REPORT.md`.

The angular audit itself does preserve all twelve states' per-head and
per-matrix weight-space radial/tangent terms in `run1/heads.csv` and
`run1/matrices.csv`. It contains **no** gradient pairing or GN/logit response
of those components. Q/K's positive-scale symmetry supplies an architectural
reason radial scale alone is functionally redundant in the zero-epsilon
limit, but not the magnitude or usefulness of the observed tangent response.

## Why the older direction Grams do not answer this question

`gn/M_lr0.007_s260925_l40s_step000500_gram.json` and
`gn/PD_a0.25_lr0.01_s260925_ada_step000500_gram.json` each retain 48 names
plus vectors `a` and matrices `Q` for `g4M:muon`, `g4M:pd_a0.25`, K-FAC and
GN directions. The main `gn/*.json` files additionally contain half-bank
Grams for constructed-momentum Muon/GN directions at 1M step 500 and 4M
step 183. Their scope is documented in `../coupling_archive/REPORT.md`.

`gram_saved.py` and `one_step_gn.py` establish that these are mapped,
counterfactual directions. In the momentum case the fresh input is built
from a held-out validation gradient and a saved buffer, not the saved actual
SOAP-PD next write. A 48×48 Gram could in principle sum Q/K rows/columns and
retain cross terms **for that same measured direction and state**. It cannot
be transferred to the new 16M SOAP-PD states, new learning rates or different
actual updates by multiplying norms, angles or optimizer labels.

Other nearby artifacts also miss the required identification:

- `one_step_alpha_M.json` / `one_step_alpha_PD.json` and
  `one_step_twosided_*.json` have per-kind `first`/`q` terms, but for freshly
  constructed polar/PD directions at older 1M states. They are neither the
  required states nor actual next writes.
- `one_step_split.json` does use actual writes, at original 1M steps
  200/500/900, but its three components are stiff/middle/flat Kronecker-rank
  pieces spread across all kinds. Its 3×3 Gram cannot isolate Q/K.
- `coupling/` contains only Muon/PD original 1M step-500 tensors. Their
  producer `measure_step_coupling.py` applies actual global stiff/middle/flat/
  auxiliary pieces and saves gradient changes in measurement frames. This
  is not a kindwise actual-step Gram for the twelve target states.
- The completed `../dose_function/REPORT.md` uses actual finite writes at
  16M steps 9/46, but partitions **PD** into body and auxiliary groups. The
  target family here is SOAP-PD, and Q/K is not separately scored there.
- `../aux_partition/REPORT.md` isolates body/head/embedding/norm groups at
  Muon 4M183→184; its body still includes every hidden matrix. The endpoint
  confidence panel likewise contains no update-group perturbations.

## What is missing, and what would change the next decision

Let D_QK be the actual saved Q/K write and D_R its chosen complement. To
assess local functional relevance one needs at least signed slopes for both
pieces and their predictive movement, retaining their cross term. In a GN
view that means q(D_QK), q(D_R), and <D_QK,G D_R>, or the corresponding
logit/JVP responses. The archived scalar

    q(D_body) = q(D_QK) + q(D_R) + 2 <D_QK,G D_R>

does not determine any one of them. Even a large isolated Q/K quadratic
would not establish useful descent; even a small net whole-body quadratic
would not exclude large cancelling Q/K and complement responses.

A finite alternative requires actual base/QK-only/complement/full outputs,
with a precisely declared complement and independent scoring. If complement
means every other parameter, it includes the changing RMS gains, head and
embeddings and reconstructs the actual next model. If it means body only,
that narrower scope must be explicit. A full per-head factorial is not
needed merely to test the missing relevance premise.

This inventory is **not** a proposal to run either measurement now. First
ask whether establishing this functional premise is worth a separate bounded
experiment relative to other open connections. Without it, stop at the
supported structural result: normalized Q/K weights turn farther under
shorter momentum, and their growing radii partly absorb LR changes. Do not
promote that observation to the mediator of the training benefit or design
a norm/angle intervention from it.

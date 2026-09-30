# Prior linearized-edge experiment: independent source and inference audit

2026-09-28. Read current guide/state, the September 27 09:45–10:35 protocol
entries, `eos_linearized.py`, its three JSONs and two completed logs,
`one_step_gn.py`, the gradient helper, actual step records around the probed
checkpoints, and the transport replay implementation. No model, tensor,
GPU or scheduler call. Standard-library arithmetic only. This is the only
new file.

## Verdict and historical correction

**A differential measurement already exists.** The earlier observer wording
must distinguish the missing faithful current-16M feedback replay from an
absence of any differential experiment. September 27 measured a conditional
FP32 map Jacobian composed with sampled GN at older 1M/4M states. The
universal prediction that its leading eigenvalue sits near the appropriate
instantaneous heavy-ball/Nesterov threshold failed.

The failed prediction should not be rescued by citing clipping: the actual
incoming steps of the decisive Muon variants were unclipped. Equally, the
archive does not prove that nonlinear saturation is the cause of stability,
or that differential feedback is irrelevant. It tests a frozen local
surrogate, not the stability of the full time-varying optimizer trajectory.

## Artifact inventory and measured outcomes

All are in `logs/muon_spectra/second_order_audit_20260926/`:

- `eos_linearized.py`: instrument.
- `eos_linearized_a.json`: five completed state records.
- `eos_linearized_b.json`: five additional completed state records.
- `eos_linearized_smoke.json`: two original smoke records, preserved.
- `job_eoslin_a_2625066.log`, `job_eoslin_b_g20.log`: completed measurements.

There is no `eos_linearized/` output directory in the located archive.
The notebook records 40 Arnoldi iterations and 256 curvature sequences for
the completed runs; these settings are also the current script defaults.
The JSON does not record iteration count, sample count, residuals, start
vectors, full Hessenberg matrix or source hash, so the word "converged"
cannot be independently established from these outputs alone.

| State | LR times leading real eigenvalue | Threshold ratio |
|---|---:|---:|
| Muon beta .95, 1M500 | 3.7836 | .9702 |
| Muon beta .95, 1M900 | 4.1560 | 1.0656 |
| Muon beta .95, 4M183 | 4.2518 | 1.0902 |
| Muon beta .9, 4M183, LR .014 | 6.6106 | 1.7396 |
| Muon beta .9, 4M183, LR .02 | 7.0538 | 1.8563 |
| Muon beta .81, 4M183 | 12.1585 | 3.3587 |
| Muon Nesterov beta .95, 4M183 | 3.7414 | 2.7820 |
| PD alpha .25, beta .95, 1M500 | 3.2785 | .8406 |
| PD alpha .25, beta .95, 4M183 | 4.9158 | 1.2605 |
| PD alpha .25, beta .9, 4M183 | 5.9711 | 1.5713 |

The published near-one versus above-bound distinction is real in the saved
numbers. The additional beta .9 and PD records also show why the two
initial favorable examples should not be generalized. The small smoke
records were around 15, about four times the completed values, which is a
reminder of sample/solver sensitivity, not grounds for choosing whichever
record agrees with a story.

## Units, time indexing and what map was differentiated

`eos_linearized.py:99–108` loads the checkpoint's unnormalized momentum
`M_s`, obtains a fresh gradient g, and forms `M_plus=beta*M_s+g`. Nesterov
uses `g+beta*M_plus`, matching the code's lookahead convention. This avoids
the new step-profile instrument's lagged-input mistake: here the input is
a constructed next momentum, not `M_s` alone.

There is no missing `(1-beta)` factor. `fresh_gradient` sums gradients of
per-sequence mean token loss and divides by sequence count. The saved buffer
is a sum of such batch-mean gradients. GN is the operator for the mean
token loss. LR is evaluated for step `s+1` with the appropriate tokens
already consumed. These are consistent units for the code's unnormalized
momentum recurrence.

However, g is a fresh **validation-stream** gradient, not the actual next
training batch. It uses `min(batch_tokens/T,8192)` sequences, giving the
full nominal 1M/4M size for all located records; the cap would become a
different-sized gradient if this code were reused for 16M. No clipping is
applied to this surrogate g, and its norm is not retained.

The map derivative is through a smooth FP32 implementation of the finite
NS polynomial, including its input Frobenius normalization and aspect
multiplier. Training uses BF16 products; the probe is not differentiating
the actual rounding operations. For PD, it **does** differentiate the
per-matrix final Frobenius matching (`:130–132`), unlike the simpler static
step-profile counterfactual. Its root is a fresh input second moment from
the curvature set, with requested alpha and default .001 damping, rather
than the online owner's cached root. The root is held fixed in this
derivative. SOAP state, root changes, auxiliary updates and weight decay
are absent from the operator.

## Clipping does not explain the decisive counterexamples

The actual 4M step-184 gradients at the step-183 checkpoints have these
logged global preclip norms, all with `gradient_clipped=false`:

- Muon beta .81: .5601;
- Muon Nesterov beta .95: .5556;
- Muon beta .9 at LR .014: .5199;
- Muon beta .9 at LR .02: .3891;
- PD beta .9 at LR .02: .6261.

The neighboring step 183 is also unclipped in every case. The beta .95
Muon/PD reference steps and the 1M500/900 references are similarly below
the clip threshold. This does not certify the unlogged probe gradient's
norm or remove sampling/state differences, but it rejects an explanation
that the online local clipping derivative was active and simply omitted
for these observed exceptions. Historical clipping can influence the
saved buffer; that history is already included in M.

## Symmetry/PSD was not qualified by the recorded check

The operator is evaluated with nonsymmetric Arnoldi, which is the suitable
choice. But the prose "J is symmetric to rounding" is not justified by
one random pair's normalized bilinear discrepancy. A small structured
antisymmetric component can be diluted by the ambient dimension. The six
retained leading-real eigenvalues all have zero imaginary parts; that is
not a proof of a real, nonnegative full spectrum.

There is a concrete counterexample for the exact source map, requiring no
network. On diagonal input `diag(1,.5)`, let `s_i=x_i/||x||` and let h be
the composition of the five scalar NS polynomials in the archived recipe.
On the diagonal subspace,

    J_ij = h'(s_i) (delta_ij - s_i s_j) / ||x||.

Direct scalar-chain differentiation of those coefficients gives

    J = [[-1.1300175,  2.2600350],
         [ 1.0112489, -2.0224979]].

This is nonsymmetric and has negative Rayleigh quotients. Finite NS plus
input normalization is not the gradient/Hessian of the nuclear norm merely
because exact polar is. PD's final M-dependent norm matching introduces
another potential loss of symmetry. No claim is made that the trained
buffers realize this particular counterexample; it invalidates the assumed
general guarantee.

This distinction cuts in two directions:

- A near-upper-threshold leading positive eigenvalue never certified full
  local stability, because negative/complex modes were not ruled out.
- If an accurate positive-real eigenvalue lies above the scalar bound,
  it still supplies an unstable mode of the corresponding **frozen**
  heavy-ball linear system; no PSD assumption is needed to make that
  particular mode problematic. Nonsymmetry alone therefore does not erase
  the failed near-edge prediction.

The code sorts eigenvalues by real part and retains only six. No residuals
or negative/complex tail are available for a more complete retrospective
stability test. Do not relabel these numbers as a complete spectral radius.

## Why an instantaneous violation is not proof of saturation

For a fixed derivative A and gradient derivative H, plain momentum has

    [[I-eta A H, -eta beta A],
     [H,                 beta I]].

The real-positive coupled mode bound is `eta*mu < 2(1+beta)`. For the
implemented Nesterov input, the corresponding block has upper row
`[I-eta(1+beta)AH, -eta beta^2 A]`, giving
`eta*mu < 2(1+beta)/(1+2beta)`. The script's thresholds are correct for
these frozen scalar-mode models.

But the real training derivative uses the loss Hessian (and, when active,
the clipping derivative), not GN alone. A and the state vary every step;
R and auxiliary parameters vary too. Stability of a periodic or drifting
trajectory is determined by products of step Jacobians. Individual factors
can have expanding eigenvalues while their product contracts. A normalized
optimizer is also not being linearized around an ordinary stationary
nonzero-momentum fixed point. Thus "the universal instantaneous edge
predictor failed" is supported; "map saturation is established as the
stabilizing cause" is a candidate explanation that these records do not
uniquely identify.

## Connection to the current observer evidence and replay proposal

The three objects are complementary, not replacements:

1. The new 16M static-map audit establishes strong same-state stiff
   suppression by a frozen PD map of lagged momentum.
2. The older EOS archive already measures differential gain for a
   constructed next input, and shows that the simplest instantaneous
   threshold does not organize all successful trajectories.
3. The retained fixed-weight batch replay can separate recent-data
   weighting from changes induced by the model's own parameter history
   inside the stored 16-dimensional probe span.

This makes the proposed replay scalar reduction a useful next discriminator,
with no new model work. Its identity
`Mstar=A*gbar + sum_k beta^k f_k(p_k-gbar)` is sound in the retained span.
Both actual 100-batch replay clip-factor lists are all one, so historical
clipping does not complicate these two specific reductions. Compare the
weighted residual with both `A*gbar` and actual M, retaining their signed
cross terms: large cancellation can make a small residual important for M.

Two details should remain explicit. Replay rows are newest-first, with
k=0 corresponding to batch 500. Mstar includes only the last 100 batches,
whereas full M contains the older tail `beta^100 M_400`; at beta .95 the
coefficient is .0059205. The difference therefore includes an unmeasured
tail as well as model-history effects. Centered lag correlations on those
fixed-weight projections measure batch-order structure, not on-trajectory
feedback or an iid noise spectrum. Neither the 16-dimensional result nor
its correlation can be promoted to a full-space or training-rate claim.

**Decision:** correct the historical omission, retain the old failed
instantaneous-edge prediction with its surrogate qualifications, and finish
the small saved-replay decomposition. Do not repair/relaunch the EOS probe
or construct new momentum candidates merely to recover the old principle.

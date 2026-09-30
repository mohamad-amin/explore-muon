# Peer review: changing the noise used to estimate the same output factor

2026-09-28. Independent, bounded review of the next estimator idea. No model,
GPU, training job, or main-study file was used or changed. The current guide,
state, observer state, frozen repeatability instrument and run-2 review were
read. Live Slurm records showed existing main and tiny-surrogate work active;
this reviewer used one-thread CPU algebra only.

**Recommendation:** a fixed-input categorical-versus-curvature-propagation
repeatability check is scientifically worthwhile, conditional on the parent's
stage-calibration result showing that the original label variation survives
the nonlinear map. The target is held fixed, so this is a cleaner test of an
estimation limitation than substituting SOAP or empirical-label statistics.
Use Rademacher (independent ±1) noise as the practical candidate, or retain
Gaussian as an explicitly diagnostic arm. Gaussian is not uniformly better
than categorical sampling. Rademacher has a useful raw-factor variance
guarantee relative to Gaussian, but neither has a direction-quality guarantee.
No training improvement or novelty follows from this review.

## 1. Identity, including attention and all token positions

Fix weights, evaluation-mode computation, and all teacher-forced input tokens.
At output token t let p_t = softmax(logits_t), D_t = diag(p_t), and
H_t = D_t - p_t p_t^T. Define s_t = sqrt(p_t) and

    L_t = diag(s_t) - p_t s_t^T
    v_t = L_t z_t = sqrt(p_t) * z_t - p_t sum(sqrt(p_t) * z_t).

For zero-mean noise with E[z_t z_t^T] = I,

    L_t L_t^T = D_t - p_t p_t^T = H_t,
    E[v_t] = 0, and 1^T v_t = 0.

Both standard Gaussian and independent Rademacher noise satisfy this. Draw
independent noise across vocabulary entries, output tokens, sequences, and
replicates. Reusing one vector across token positions would introduce unwanted
cross-position covariances and change the target.

The sampled-label score u_t = p_t - onehot(Y_t), with Y_t sampled independently
from p_t, has the same mean and covariance H_t. Its sign is immaterial to its
own Gram, but do not mix signs or noise conventions between tokens arbitrarily.

Stack all hidden outputs of one retained layer into a vector h; let J_t be
the Jacobian of logits_t with respect to h. Backpropagating

    sum_t sum_c logits[t,c] * stop_gradient(v[t,c])

gives e = sum_t J_t^T v_t. The loss is a scalar sum, not a token mean. Detach
both p and the complete v before differentiation. Otherwise derivatives of
the noise map add spurious terms. Thus, with K_t = J_t^T H_t J_t,

    E[e e^T | inputs, weights] = K = sum_t K_t

under all three noise distributions. This statement uses the full Jacobian:
future-token attention paths and shared hidden states are already included.
The pseudo-errors at different hidden positions generally ARE correlated.
Their independence must not be assumed for variance calculations.

Reshape e into N hidden-position rows e_s of width d. The existing statistic

    B_hat = (1/N) sum_s e_s e_s^T

therefore has exactly the same conditional expectation under categorical,
Gaussian, and Rademacher noise. This does not assert that B is the full
parameter GGN: it is the existing output factor, with its existing Kronecker
and token-factorization approximations. Nor does it integrate over new
autoregressively generated prefixes: the input contexts remain fixed. SOAP
gradient statistics and empirical data-label factors are different targets.

## 2. What is and is not improved by Gaussian noise

For any symmetric d-by-d matrix W, define

    Q = (I_N kron W) / N,
    Z = tr(W B_hat) = e^T Q e.

This covers error energy (W=I), each diagonal entry, and each off-diagonal
entry (with symmetric W having one-half in the selected off-diagonal pair).
For Gaussian noise, e is jointly Gaussian, and

    Var_G(Z) = 2 tr(Q K Q K).

For categorical sampling, let a_t(y)=J_t^T[p_t-onehot(y)]. Independent output
tokens imply fourth cumulants add, giving the exact difference

    Var_cat(Z) - Var_G(Z)
      = sum_t { sum_y p_t[y] (a_t(y)^T Q a_t(y))^2
                - [tr(Q K_t)]^2 - 2 tr(Q K_t Q K_t) }.

The bracket can have either sign. High kurtosis of rare categorical outcomes
can favor Gaussian noise, but one cannot infer the sign from entropy or the
size of the vocabulary alone. The downstream Jacobian and the statistic W
matter. Summing this variance identity over a Frobenius-orthonormal basis for
symmetric W gives the corresponding raw-factor Frobenius MSE comparison.
For a single hidden error vector with covariance K, this reduces to

    E_G ||e e^T-K||_F^2 = (tr K)^2 + tr(K^2),
    E_cat ||e e^T-K||_F^2 = E_cat ||e||^4 - tr(K^2).

A scalar Bernoulli counterexample makes the limitation concrete. With
e=Y-p, categorical Var(e^2)=p(1-p)(1-2p)^2, while the matching Gaussian has
Var(e^2)=2[p(1-p)]^2. At p=.5 the former is zero and the latter .125;
at p=.01 the former is .009508 and the latter .000196. Gaussian wins only
when p(1-p)<1/6 in this scalar example. Both directions of the comparison
must remain visible; there is no general Gaussian dominance.

## 3. Why Rademacher is the better practical half-factor comparator

Let T = J^T blockdiag(L_t), so e=Tz, and A=T^T Q T. Keeping exactly this
factor T, independent ±1 coordinates give

    Var_R(Z) = 2 ||A||_F^2 - 2 sum_i A_ii^2,
    Var_G(Z) = 2 ||A||_F^2.

Hence Rademacher has no larger variance for EVERY linear functional of the
raw B, including its trace, entries, and total Frobenius error. This result
requires the same half-factor; Gaussian is invariant to right-orthogonal
rotations of a factor, whereas Rademacher generally is not. This guarantee
is relative to Gaussian, not categorical scores.

Trace normalization, damping, inverse square roots, Newton–Schulz, and final
norm matching are nonlinear. The preceding inequality does not order their
biases, direction repeatability, descent, or training performance. Antithetic
z and -z produce exactly the same B, so they are not two independent probes.

This is established prior art. Martens, Sutskever and Swersky's curvature
propagation paper develops unbiased curvature estimates by propagating random
vectors through curvature factors and analyzes Gaussian versus binary noise;
its Section 4 gives the variance reduction from independent ±1 noise.
[Primary paper](https://www.cs.toronto.edu/~jmartens/docs/Curvature_Propagation.pdf).
KFAC-JAX's official API separately exposes sampled-label `fisher_gradients`
and `fisher_curvature_prop`, the latter using random ±1 vectors with Fisher
half-factors. This confirms that the comparison is an established estimator
choice, not a proposed new curvature target.
[Primary implementation documentation](https://kfac-jax.readthedocs.io/en/latest/api.html).

## 4. Bounded comparison and qualification

Use the already frozen step-46 weights, momentum, common input root, all
48 matrices, and exact saved A/B token banks. For each bank, form two
independent candidate estimates on the same eight sequences. Compare against
the two existing categorical draws for that bank; do not select a bank or
subset of matrices by the size or sign of the observed result. This is four
new B estimates if only one alternative is tested. Do not add Gaussian,
Rademacher, more banks, or more draws after seeing a marginal result merely
to find a favorable comparison.

Before full execution:

- Keep the identity check and a small dense-Jacobian cross-position example.
  Verify the common B expectation by exact enumeration at small dimensions;
  verify the categorical-to-linear-surrogate backward gives the same hidden
  errors for the exact same categorical score vector on one real sequence.
- Check hidden-gradient reconstruction, all-token summed-loss scaling,
  softmax normalization, near-zero sum(v_t), finiteness, factor PSD up to
  numerical tolerance, and exact frozen weight/token/root hashes.
- Preserve model mode, FP32 factor computation, FP64 eigensolver, damping
  .001, exponent, NS convention, final per-matrix norm matching, and primary
  all-matrix aggregation. Reuse the original CPU precision control instead
  of silently moving the new arm to another arithmetic path.
- Price one sequence with both backpropagations and representative shapes,
  include a memory bound for the vocabulary-sized dense noise, and keep the
  existing two-thread CPU/900-second ceiling if the forecast fits. No GPU
  allocation. One sequence at a time bounds logits/noise storage.

Primary outcomes should include same-input replicate distance for raw B
(both absolute scale and common-normalized Frobenius summaries), root/map
stage distances, and final direction distance/cosine, each for A and B as
well as the already fixed whole-body aggregation. Common normalization must
not use each noisy factor's own trace for a claimed unbiased variance
estimate. Independent-pair squared distance divided by two estimates raw
conditional MSE; two pairs remain a noisy premise check, not a precise
population variance estimate. The nonlinear direction has no such
unbiased-target interpretation. Report per-kind and per-matrix heterogeneity
without treating the 48 matrices as independent model replications.

Use the retained pooled categorical direction as an imperfect external
reference, explicitly accounting for whether it reuses A/B. A reference
built from C/D is independent of both candidate and categorical A/B noise,
but also uses other inputs; its distance combines estimation effects and
sequence-sampling differences. Higher repeatability alone is insufficient:
a nonlinear map could become stably biased or collapse toward the PD
baseline. Retain distances to PD, trace/norm information, and map-stage
effects. A common .8/.2 anchor is an optional *predeclared* reporting view;
it must use the same anchor for both noise schemes.

Proceed to a separate held-out local model-quality check only if a material,
consistent A/B gain survives the nonlinear map. Stop this estimator branch
if gains are raw-factor-only, inconsistent across A/B, or small compared
with existing numerical sensitivity. Even a positive premise check cannot
establish that label noise explains TS versus SOAP or that training will
improve. Input-sampling noise, factorization bias, geometry mismatch, and
trajectory adaptation remain distinct explanations.

## 5. Reproducible algebra check

`verify_algebra.py` enumerates all 9 categorical outcomes and 64 Rademacher
outcomes in a two-output-token, two-hidden-position example with a dense
cross-position Jacobian. It checks the half-factor identity, common B
expectation, categorical cumulant formula, and Rademacher variance formula.
The largest absolute discrepancy is 4.44e-16. `algebra_checks.json` preserves
the values and the binary counterexamples. This is a mathematical check,
not a trained-model measurement.

The first invocation used system `python` and failed at importing numpy;
the unchanged script passed under the project's existing `.venv/bin/python`.
No package was installed. All computation used one numerical thread.

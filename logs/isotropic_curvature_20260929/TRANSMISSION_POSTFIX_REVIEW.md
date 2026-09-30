# Independent post-fix transmission source review

2026-09-29. Reviewed `analyze_transmission.py`, `TRANSMISSION_PROTOCOL.md`,
the relevant definitions in `instrument.py`, producer serialization in
`run_atlas.py`, and the diagnostic model's MLP. This was a read-only source
review: no saved tensors, contractions, synthetic qualifications, or model
evaluations were executed. Python AST parsing passed.

Reviewed analyzer SHA256:
`5762ba936e11d1f70e2160f34acc619e15d1e1bf8cbd2fd95e5cc08f1e63974c`.

**Verdict: no blocking correctness issue found.**

- The analytic derivative matches the implemented tanh-approximate GELU.
  Applying `F.linear(dz * gelu_prime(Z), W_down)` computes the exact immediate
  residual-branch tangent for an isolated up-weight perturbation. The unchanged
  residual identity branch contributes zero to this derivative.
- Squared norms sum the feature dimension, retaining context and position.
  GN and energies pool the same context/position selections. The factorization
  uses ratios of pooled quantities, avoiding tokenwise causal attribution
  through subsequent attention.
- The down matrix comes from the producer's base checkpoint, and its hash is
  checked against the producer's consumed model tensor hash. Step, method,
  direction hashes, shapes, and the unique 18-panel coverage are now checked.
- The earlier zero-denominator issue is fixed. Undefined control ratios retain
  empty values and a reason while keeping raw quantities. No positive floor
  substitutes for an undefined ratio.
- The interpretation remains limited to a descriptive GN decomposition.
  It does not explain the Hessian-minus-GN term, finite displacements, joint
  steps, or causal optimizer mediation.

One reporting detail: `contrasts.csv` contains actual/control comparisons.
The protocol's direct raw-versus-whitened comparison should be computed from
`metrics.csv`; both necessary inputs are retained there. This does not require
new model evaluations or producer changes.

Passing this source review does not establish numerical execution success.
The synthetic derivative check, archived radius agreement, finite-value checks,
pooled identity checks, and complete output must still pass when run.

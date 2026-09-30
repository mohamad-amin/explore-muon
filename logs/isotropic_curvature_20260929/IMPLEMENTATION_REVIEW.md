# Independent implementation review

2026-09-29. Reviewed the frozen protocol, `instrument.py`, `qualify.py`, copied
model/data sources, and relevant frozen training source/metadata. No models,
checkpoints, scientific outcomes, or qualification outputs were loaded or
evaluated by this reviewer. Source parsing, hashes and textual comparisons
were the only computations.

**Verdict:** no blocking algebra, index, dtype, or current-input error found
in the reviewed core instrument. Proceed subject to the already planned
engineering qualification and completed-panel resource forecast. This is
not a review of the still-developing full run loop or its final artifacts.

## Exact derivatives and coefficient-space geometry

The nested forward-AD structure is correct. `first(a)` returns `(z(a), z'(a))`;
its JVP returns `((z,z'),(z',z''))`. Therefore the unpacking
`(z,dz),(dz_again,d2z)` is correct. `torch.no_grad()` suppresses reverse-mode
tape construction, not forward AD, so that decorator does not discard the
intended tangents. The duplicate first tangent comparison is a useful check.

For a token with target y and logits z(a), the implemented expressions are

    a_loss = sum(p * dz) − dz_y,
    GN = sum(p * (dz − sum(p * dz))²),
    H = GN + sum(p * d2z) − d2z_y.

These are the exact first and second derivatives of token cross-entropy,
including the network's second-derivative term. They are neither a GN-only
approximation nor mean derivatives broadcast over tokens. The predictive
Gram uses the corresponding centered logit tangents under the same base
probabilities. Its diagonal agrees algebraically with the directional GN.

`projected_hessian` differentiates the aggregate token loss with respect to
the supplied finite-dimensional coefficient vector. In a five-direction
single-site panel it is a 5×5 directional restriction; in the joint panel
it is 3×3. It is not the full parameter Hessian. Its diagonal should agree
with the mean per-token H for the corresponding rays; mixed terms should
be retained without PSD clipping.

`gradient_at_site` differentiates mean token CE, then contracts the output
adjoint with the correct normalized inputs. Thus the returned matrix is the
selected weight's gradient for that mean loss, and its inner product with D
should equal the mean directional slope. The returned `e * y.numel()` is an
unaveraged adjoint of the summed sequence loss at each activation location.
At early sites it must not be called the gradient of that location's own
target loss: later attention couples locations.

## Perturbation placement and parameter indexing

`capture` saves the residual immediately after each block's attention and
the normalized input/pre-GELU output of the up matrix. `suffix` reconstructs
the remaining MLP residual followed by later complete blocks. For one
changed up matrix its input is fixed by untouched upstream parameters;
adding `a * F.linear(cached_input,D)` is therefore the exact weight ray.

`joint_fn` correctly begins at the cached residual after block0 attention,
skips that attention exactly once, and executes attention at subsequent
blocks. It recomputes the normalized current input before every selected
matrix and adds the matching current-input kick. This preserves changes in
later selected inputs caused by earlier perturbations. The direct full-model
parameter-write comparison in `qualify.py` tests precisely this distinction.

For this directory depth, `HERE.parents[1]` is the project root. No previous
deeper observer-directory parent index should be substituted. Checkpoint
pairing asserts states s and s+1, tokens, shape and model-key agreement;
`w1.double() - w0.double()` preserves the exact FP32 saved-write difference
without an intermediate FP32 subtraction.

The body-key list follows model-state order, filtered by the original
metadata's optimizer assignment. Zipping that list with body-group parameter
IDs matches the frozen optimizer construction. Assertions at body positions
4/22/46 identify the up matrices in blocks0/3/7. Stored momentum is replicated
before owner-specific map calculation in the frozen training code, so rank0
contains these momentum buffers even when a different rank owned the map.
The code does not confuse map ownership with buffer availability.

## Frames, precision and source provenance

Left signed permutations are orthogonal and preserve `D.T @ D`; the right
signed permutation implements `D O_R^T`. Applying that same operator to
`P=D S` and multiplying by `S_inverse` gives the declared whitened comparison.
The Gram and map-back checks match these conventions. The tiny negative
eigenvalue clamp is a numerical PSD repair before adding the fixed damping;
it is not an outcome-dependent spectral fit. The run artifacts should retain
the original eigenvalues and regularization, as this helper returns.

The copied diagnostic removes active FP32 normalization and CE casts and
uses explicit causal attention. Model parameters, scalar coefficients and
the explicit loss path are FP64. The original model/data copies match the
frozen training files byte for byte. The ordinary reference comparison
tests model-forward precision differences while computing reference CE in
FP64; it should be described that way, not as exact emulation of training
BF16 loss reduction. Source validation inside `load_pair` checks all original
metadata source hashes. The full run must additionally archive the diagnostic
source and run-loop hashes; original-source validation alone does not freeze
new instrumentation.

Qualification covers cached/full/suffix equality, the changed FP64 numerical
function versus the original FP32 forward, per-token Richardson derivative
checks, a reverse-Hessian cross-check, rotation identities, direct joint
parameter writes, and restoration. These are meaningful engineering checks.
The covariance used by that qualification is an actual cached-input moment,
despite its comment calling it synthetic. This naming issue does not change
the check's mathematics or the later eight-context calibration protocol.

## Remaining execution scope

The reviewed code explicitly restricts CPU numerical threads and hides CUDA.
No optimizer is rerun. Large logit tangents, probabilities, and reverse-AD
graphs can dominate memory; the first complete-panel forecast must include
their storage and the projected Hessian, not just finite forward passes.
Chunking contexts is legitimate if losses and derivatives retain the declared
mean convention. Run-loop pairing, serialization, bank aggregation and
plotting remain to be checked when implemented; they are not certified here.

Reviewed SHA256 values:

| File | SHA256 |
|---|---|
| PROTOCOL.md | `0526c40d9e63ba25bb48790701e3fd112d429c7e4e99af32893359d77a0e69b3` |
| instrument.py | `f8e373c52f8cc699e744e449e2cbff52dd36d3e2572427777c3ce3c84bd34314` |
| qualify.py | `4d9e4f5b87bbb284cca1a55ec179868a016f3201c47a18ffca531510bc8b0f60` |
| source/adamw_spectra/model.py | `c977db1cc482b0cd4db7a5f0c2342e44ea08b8ea92b35673324f94a84c6505c9` |
| source/adamw_spectra/data.py | `99134dc89336636a65ce9eb6774c14e73843a6f9e4be4cf523bccfc0fd612124` |
| source/original/model.py | `7bb63819666a4917ac7919a6ac16540293a9ef074a6015e1729e21ea4f6fb25a` |

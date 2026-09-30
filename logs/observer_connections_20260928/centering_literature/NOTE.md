# Centering preserves a route only when the offset moves with it

2026-09-28. Bounded primary-source review for the observer's proposed functional
V-update split. No model calls, GPUs, scheduler submissions, or main-file edits.
The prior failures of mean-only whitening and body-factor centering remain
evidence against treating centering alone as an improvement. No novelty claim.

## Three precise precedents

**PRONG / Natural Neural Networks.** Desjardins et al. (2015),
[§3.1–3.2, equation 6 and Algorithm 1](https://papers.neurips.cc/paper/5953-natural-neural-networks.pdf#page=3),
write a layer as `V U (x−c)+d`, with statistical coefficients `U,c` held fixed
for ordinary optimization. On refreshing them, they preserve the canonical
weight `W=VU` and bias `b=d−Wc`, then reconstruct `V=W U_new^−1` and
`d=b+W c_new`. The important precedent is joint weight/offset compensation,
not subtracting the mean alone. Their §2.2 factorization approximates the
dependence between activations and backpropagated derivatives; input whitening
does not remove all Fisher correlations.

**Affine K-FAC.** Martens and Grosse (2015),
[§1.1 and §4, Theorem 1 and Corollaries 2–3](https://www.cs.toronto.edu/~rgrosse/publications/icml2015-kfac.pdf#page=6),
include biases as a homogeneous input coordinate. This permits invertible
activation translations within their invariance result. For fixed transforms,
their basic K-FAC trajectory is equivalent with matched initialization,
negligible damping, no momentum, and a parameterization-independent learning
rate. They also describe K-FAC as gradient descent in centered/whitened
activity and derivative coordinates. These are qualified statements about
inverse-curvature updates, not a theorem for every optimizer applied after
whitening.

**Centering plus a compensating shortcut.** Raiko, Valpola and LeCun (2012),
[§2 equations 2–7; §3; §4 “Online Learning”](https://proceedings.mlr.press/v22/raiko12/raiko12.pdf#page=2),
center both activation and slope of hidden nonlinearities, while an explicit
shortcut carries the removed affine part. Equation 7 exactly compensates
changes of the centering coefficients in the shortcut weights. Their Fisher
argument relies on approximately decorrelated factors; centering does not
guarantee diagonal curvature. When they refresh their coefficients they also
reset momentum. The relevant connection is preserving a useful linear route
while changing the coordinates used to learn nonlinear features.

## Local algebra: which operation would actually be tested?

The following identities are this observer's derivations/application, not
empirical claims from the papers. Use column-vector activations.

For an affine value map,

    v = W x + b = W(x−μ) + a,       a = b + Wμ.

At fixed μ, the differentials and ordinary gradients obey

    Δb = Δa − ΔW μ,
    G_center = G_raw − g_b μᵀ,     g_a = g_b.

Thus a zero-mean factor and an independently optimized offset alter weight–bias
coupling. For this project's bias-free V, maintaining `a=Wμ` as a differentiable
constraint gives exactly the original function and gradient. Releasing `a`
adds an affine degree of freedom: the starting function matches, but it is an
architecture/optimizer intervention, not just a relabeling of the existing
bias-free model. Equal initial outputs do not imply equal tangent spaces.

If a stored reference mean moves from μ to μ+h while W is fixed, preserving
the function requires `a_new=a_old+W h`. If W also changes by D while canonical
b remains fixed, the exact compensation is

    a_new − a_old = Dμ + W h + D h.

For an affine model, a stored parameter displacement must obey the same change
of coordinates; first-moment gradients are covectors and instead transform by
the gradient identity above. A moving coordinate system cannot be handled by
reusing a momentum tensor with no stated interpretation. Elementwise squared
moments cannot generally be transported exactly without cross moments.

This also clarifies the covariance issue. Define Σ=E[(x−μ)(x−μ)ᵀ] and
C=E[xxᵀ]. The augmented input factor and its centered form are

    C_aug = [[C, μ], [μᵀ, 1]],
    T = [[I, −μ], [0, 1]],
    T C_aug Tᵀ = diag(Σ, 1).

K-FAC can use this change while preserving bias–weight coupling. Replacing C
with Σ for a bias-free W, with no compensating route, is a different operation.
Even with an offset, zero input mean does not force the exact Fisher/GN
weight–offset cross block to vanish: activation-dependent output curvature
and attention's cross-token dependence can preserve it.

## Consequences for the functional V probe

With the attention matrix A fixed with respect to a local V perturbation and
`A1=1`, the saved V displacement D has the exact functional split

    A(Dx) = Dμ + A[D(x−μ)].

The two-parameter path at one layer is

    v(c,m;x) = W x + c D(x−μ) + m Dμ.

The diagonal `c=m=t` is exactly the ordinary parameter path `(W+tD)x`.
This equality survives nonlinear downstream computation. With multiple
selected V layers, the same statement holds if each split uses that layer's
current input under the perturbed model, rather than a cached baseline input.
Forcing cached activations changes a finite counterfactual into a different
intervention; cached baseline inputs are sufficient only for a local JVP.

For logits z, retain `s_m+s_c=s_all` and
`q_all=q_mm+2q_mc+q_cc`, where `q_ab=J_aᵀ H_loss J_b`.
Centering provides no reason to drop q_mc. Nor must `E[A D(x−μ)]` be zero.
The existing attention-geometry note already finds a material selection mean.

The partition is reference-dependent: changing μ to μ+h adds Dh to the mean
component and subtracts it from the centered component. Their sum is
invariant. Therefore fix μ before scoring, preferably from an independent
activation archive/bank, and label it. A separately retained μ is a robustness
check, not a parameter to tune for a favorable channel score. In a no-GPU
qualification, direct-path and diagonal-split logits/slopes must agree within
the stated precision before either channel is interpreted.

## Where the analogy stops

- **Polar updates:** `polar(M Tᵀ)` does not generally equal a coordinate
  transform of `polar(M)` for nonorthogonal T. Centering is a nonorthogonal
  shear in augmented coordinates. A polar update of `[W,a]` also couples its
  extra column to all singular directions, while a separate vector optimizer
  on a is another rule. PRONG/K-FAC invariance cannot justify either by itself.
- **RMSNorm/QK normalization:** the local identity is valid for x *after*
  RMSNorm and a split on V. Centering the residual *before* RMSNorm changes its
  norm and hence the function. Similarly, Q/K offsets change the subsequent
  per-head RMS normalization and attention weights; they lack V's direct
  constant-through-attention identity. The current model code has no norm
  bias and no normalization on V.
- **Training:** different weight decay, clipping, frame refreshes and momentum
  transport can break equivalence despite a function-preserving refresh.
  A local productive offset is evidence to design controls, not evidence that
  an independently trainable V bias improves a trajectory.

The useful question remains whether the *existing* saved update assigns
productive descent to the constant route, and how strongly that route couples
to centered movement. These frameworks explain how to measure that cleanly;
they do not predict that another centering intervention will win.

## Local evidence and provenance

Read `RESEARCH_GUIDE.md`, the current `RESEARCH_STATE.md`, observer `STATE.md`
and `attention_geometry/NOTE.md` before this review. Primary PDFs and sections
above were opened and checked on 2026-09-28. No secondary article was used.
Current `research/adamw_spectra/model.py:44–54,111–136` confirms the stated
norm and V layout; the archived experiment's config is still authoritative
for an actual probe. `squeue` confirmed main allocations 2567578 (priv-g14)
and 2618555 (g20) running during the review; neither was touched.

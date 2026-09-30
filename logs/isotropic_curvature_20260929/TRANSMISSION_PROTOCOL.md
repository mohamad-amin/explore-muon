# Retrospective architectural transmission decomposition

2026-09-29. Added after the early Muon panels showed a large actual/left-rotation
curvature contrast at identical activation radii. The existing18panel producer
and its fixed measurements remain unchanged. An independent high-level review
supported this one bounded tensor-only interpretation check. It adds no
end-to-end model evaluation, optimizer, training run or GPU use.

For each recorded individual up direction and score input, compute

    dz = D x,
    dr = W_down [gelu'(Z) * dz],

using the implemented tanh-approximate GELU and the checkpoint's CURRENT down
weights. X,Z,D are already archived; down weights are verified against the
original producer's consumed-model tensor hashes. This is the exact immediate
residual-branch tangent caused by the up perturbation. It excludes the unchanged
identity branch and is not the finite displacement of the whole network.

The induced preactivation metric is
diag(gelu'(Z)) W_down^T W_down diag(gelu'(Z)); its rank is at most512 in a
2048-dimensional up-output space. This is an architectural alternative to
isotropic output cost, chosen independently of a fitted curvature response.

Read EVERY method/time/depth/direction and BOTH score banks. Save per-token
preactivation and residual tangent energies, and pool contexts/positions
consistently to report the exact scalar identity

    GN / E||dz||² = (E||dr||²/E||dz||²) * (GN/E||dr||²).

At early blocks, the latter GN is a whole-context quantity: later attention
mixes positions. Do not pair a target token's GN only with its own residual
kick as a causal attribution. Both numerator and denominators remain visible.
Compare how much actual/left and raw/white contrasts remain after this
fixed structural normalization. No fitted correction, pass/fail threshold,
chosen layer, discarded bank, or claim of optimizer mediation follows.

H−GN remains a distinct second-order model-curvature term. This check does
not explain it, establish finite-radius isotropy, or model joint-step radii.
If the normalized GN contrast persists, downstream directional structure
remains. If it shrinks, the immediate architectural projection explains
part of the contrast descriptively. Neither reading makes lower curvature
a sufficient optimization objective; first-order descent remains essential.

Qualification: compare the analytic GELU derivative with autograd on a fixed
synthetic grid; reproduce archived preactivation-kick energies; verify finite
values and the pooled identity; preserve source/input hashes. One numerical
CPU thread after the producer completes; expected under three minutes.

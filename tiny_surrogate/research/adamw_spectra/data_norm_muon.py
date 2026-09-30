"""Minimal reference for partial data-norm Muon (MUON_CASE.md, waves 5-11).

For a linear layer y = W x with input second moment C = E[x x^T], take the steepest-descent
step under the partial data norm ||dW C^alpha||_op:

    dW = s * polar(M R) R,    R = (C / mean_eig(C) + damping I)^(-alpha),

where M is the (Nesterov-free) momentum, polar() is Muon's orthogonalization, and s rescales the
update to Muon's Frobenius norm sqrt(min(m, n)) before Muon's usual shape factor. alpha = 0 is
Muon; alpha = 1/2 is the operator norm measured on the input distribution; alpha = 1/4 was
best here (the loss curvature along an input direction grows sublinearly with its second
moment). `MuonAdamW` implements this with `data_norm_alpha`, an EMA of C from every 32nd
position (training forwards only) and an FP64 eigendecomposition refreshed every 10 steps.
"""
import math

import torch

from .muon import newton_schulz


def inverse_root(cov, alpha, damping=1e-3):
    """(C / mean eigenvalue + damping I)^(-alpha) for a symmetric PSD C (FP64 eigh)."""
    eigenvalues, vectors = torch.linalg.eigh(0.5 * (cov + cov.T).double())
    unit = eigenvalues.clamp_min(0) / eigenvalues.clamp_min(0).mean().clamp_min(1e-30)
    return ((vectors * (unit + damping).pow(-alpha)) @ vectors.T).float()


def data_norm_direction(momentum, cov, alpha=0.25, damping=1e-3, schedule=None):
    """Update direction (before the learning rate) for one matrix of shape (out, in)."""
    root = inverse_root(cov, alpha, damping)
    polar = newton_schulz(momentum.float() @ root, *(() if schedule is None else (schedule,)))
    direction = polar @ root
    rows, cols = momentum.shape
    direction = direction * (math.sqrt(min(rows, cols)) / direction.norm().clamp_min(1e-30))
    return direction * math.sqrt(max(1.0, rows / cols))

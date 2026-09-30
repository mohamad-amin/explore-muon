"""Read-only reconstruction of the direction used by standard PyTorch AdamW."""

import math

import numpy as np
import torch


QUANTILES = (0.1, 0.25, 0.5, 0.75, 0.9)


@torch.no_grad()
def adamw_direction(optimizer, parameter):
    group = next((g for g in optimizer.param_groups
                  if any(p is parameter for p in g["params"])), None)
    if group is None:
        raise ValueError("Parameter is not in this optimizer")
    state = optimizer.state[parameter]
    step = int(state["step"].item())
    return _direction(state, group, step), group


def _direction(state, group, step):
    if group.get("amsgrad") or group.get("maximize") or group.get("differentiable"):
        raise ValueError("Measurement contract covers ordinary AdamW only")
    if step < 1:
        raise ValueError("Call measurement after optimizer.step()")
    beta1, beta2 = group["betas"]
    # First operations allocate private workspaces; never modify state buffers.
    m_hat = state["exp_avg"] / (1 - beta1 ** step)
    v_hat = state["exp_avg_sq"] / (1 - beta2 ** step)
    return m_hat.div_(v_hat.sqrt_().add_(group["eps"]))


@torch.no_grad()
def spectrum(matrix, device="cpu"):
    if matrix.ndim != 2:
        raise ValueError("Expected a matrix")
    work = matrix.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(work).all():
        raise ValueError("Non-finite update matrix")
    # Adam's first direction is nearly elementwise sign. FP32 reductions over
    # million-entry, nearly constant matrices can drift by several 1e-4 in
    # squared norm. Accumulate only this scalar in FP64; keep direct SVD FP32.
    norm = float(torch.linalg.vector_norm(work, dtype=torch.float64).item())
    rank = min(work.shape)
    if norm == 0:
        return {"frobenius_norm": 0.0, "zero_matrix": True,
                "quantiles": {str(q): None for q in QUANTILES},
                "sigma_max": None, "top_two_ratio": None,
                "top_one_energy": None, "stable_rank": None,
                "normalized_energy_sum": None}, np.zeros(rank, dtype=np.float32)
    work = work / norm
    # Direct SVD avoids squaring the condition number in a Gram eigendecomposition.
    if work.shape[0] > work.shape[1]:
        work = work.T.contiguous()
    kwargs = {"driver": "gesvd"} if work.is_cuda else {}
    values = torch.linalg.svdvals(work, **kwargs).cpu().numpy()
    s0, s1 = float(values[0]), float(values[1]) if rank > 1 else 0.0
    return {
        "frobenius_norm": norm, "zero_matrix": False,
        "quantiles": {str(q): float(values[math.ceil(q * rank) - 1]) for q in QUANTILES},
        "sigma_max": s0, "top_two_ratio": s0 / s1 if s1 > 0 else None,
        "top_one_energy": s0 * s0, "stable_rank": 1.0 / (s0 * s0),
        "normalized_energy_sum": float(np.square(values.astype(np.float64)).sum()),
    }, values


@torch.no_grad()
def measure_updates(optimizer, parameters, *, device="cpu", decay_norms=None, quantity="update"):
    rows, arrays = {}, {}
    groups = {id(p): group for group in optimizer.param_groups for p in group["params"]}
    items = list(parameters.items())
    # Fused AdamW stores counters on GPU: read them in one transfer, not 24 syncs.
    steps = torch.stack([optimizer.state[p]["step"] for _, p in items]).cpu().tolist()
    for (name, parameter), step in zip(items, steps):
        group = groups[id(parameter)]
        if group.get("algorithm") == "muon":
            update = (optimizer.state[parameter]["momentum_buffer"] if quantity == "momentum"
                      else optimizer.last_updates[parameter])
        else:
            if quantity != "update":
                raise ValueError("Momentum channel is reserved for Muon")
            update = _direction(optimizer.state[parameter], group, int(step))
        stats, values = spectrum(update, device)
        stats["adaptive_step_norm"] = (group["lr"] * stats["frobenius_norm"]
                                       if quantity == "update" else None)
        stats["weight_decay_step_norm"] = (decay_norms or {}).get(name) if quantity == "update" else None
        stats["quantity"] = quantity
        stats["shape"] = list(parameter.shape)
        rows[name], arrays[name] = stats, values
    return rows, arrays

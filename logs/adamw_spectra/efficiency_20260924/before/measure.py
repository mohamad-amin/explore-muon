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
    if group.get("amsgrad") or group.get("maximize") or group.get("differentiable"):
        raise ValueError("Measurement contract covers ordinary AdamW only")
    state = optimizer.state[parameter]
    step = int(state["step"].item())
    if step < 1:
        raise ValueError("Call measurement after optimizer.step()")
    beta1, beta2 = group["betas"]
    # All arithmetic is out of place: never normalize the optimizer's buffers.
    m_hat = state["exp_avg"] / (1 - beta1 ** step)
    v_hat = state["exp_avg_sq"] / (1 - beta2 ** step)
    return m_hat / (v_hat.sqrt() + group["eps"]), group


@torch.no_grad()
def spectrum(matrix, device="cpu"):
    if matrix.ndim != 2:
        raise ValueError("Expected a matrix")
    work = matrix.detach().to(device=device, dtype=torch.float32)
    if not torch.isfinite(work).all():
        raise ValueError("Non-finite update matrix")
    norm = float(torch.linalg.vector_norm(work).item())
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
def measure_updates(optimizer, parameters, *, device="cpu", decay_norms=None):
    rows, arrays = {}, {}
    for name, parameter in parameters.items():
        update, group = adamw_direction(optimizer, parameter)
        stats, values = spectrum(update, device)
        stats["adaptive_step_norm"] = group["lr"] * stats["frobenius_norm"]
        stats["weight_decay_step_norm"] = (decay_norms or {}).get(name)
        stats["shape"] = list(parameter.shape)
        rows[name], arrays[name] = stats, values
    return rows, arrays

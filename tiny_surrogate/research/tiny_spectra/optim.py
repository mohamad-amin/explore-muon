"""Qualified optimizer kernels for the standalone tiny-model experiments.

Only the production optimizer and model contracts are reused. No production
training, data, scheduling, distributed runner, or launcher is imported here.
"""

from contextlib import nullcontext
from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from research.adamw_spectra.model import StatLinear
from research.adamw_spectra.muon import MuonAdamW


METHODS = ("adamw", "muon", "pd", "ts", "soap", "spd", "sts")


def make_optimizer(model, cfg, device, *, distributed_optimizer=True):
    """Build a body/auxiliary optimizer with the same auxiliary LR in every arm.

    ``cfg`` is a mapping. Required keys are ``method``, ``lr`` and ``aux_lr``.
    Model parameters stay FP32; forward autocast is the runner's responsibility.
    PD/TS inputs use the model's StatLinear buffers. TS output statistics must be
    supplied explicitly with ``update_output_statistics`` before a step.
    """
    cfg = dict(cfg)
    method = cfg["method"]
    if method not in METHODS:
        raise ValueError(f"Unknown optimizer method {method!r}; expected one of {METHODS}")
    lr, aux_lr = float(cfg["lr"]), float(cfg["aux_lr"])
    decay = float(cfg.get("decay", 0.0))
    momentum = float(cfg.get("momentum", 0.9))
    alpha = float(cfg.get("alpha", 0.25))
    output_beta = float(cfg.get("out_beta", 0.25))
    soap_beta2 = float(cfg.get("soap_beta2", 0.9))
    refresh = cfg.get("root_refresh", 10)
    if lr <= 0 or aux_lr <= 0 or decay < 0:
        raise ValueError("Learning rates must be positive and decay nonnegative")
    if not 0 <= momentum < 1 or not 0 <= soap_beta2 < 1:
        raise ValueError("Momentum and SOAP beta2 must lie in [0, 1)")
    if not 0 <= alpha <= 1 or not 0 <= output_beta <= 1:
        raise ValueError("Geometry exponents must lie in [0, 1]")
    if not isinstance(refresh, int) or isinstance(refresh, bool) or refresh < 1:
        raise ValueError("root_refresh must be a positive integer")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Tiny experiments require FP32 parameters and optimizer state")
    body, auxiliary = [], []
    for name, parameter in model.named_parameters():
        (body if name.startswith("blocks.") and parameter.ndim == 2 else auxiliary).append(parameter)
    if len(body) != 6 * model.config.n_layer or not auxiliary:
        raise ValueError("Expected separate Q/K/V/O/up/down body matrices and auxiliary parameters")
    device = torch.device(device)
    implementation = "fused" if device.type == "cuda" else "foreach"
    betas = tuple(cfg.get("betas", (0.9, 0.95)))
    epsilon = float(cfg.get("epsilon", 1e-8))
    if method == "adamw":
        optimizer = torch.optim.AdamW(
            [dict(params=body, lr=lr, lr_scale=1.0, algorithm="adamw", role="body"),
             dict(params=auxiliary, lr=aux_lr, lr_scale=aux_lr / lr,
                  algorithm="adamw", role="auxiliary")],
            lr=lr, betas=betas, eps=epsilon, weight_decay=decay,
            fused=implementation == "fused", foreach=implementation == "foreach",
        )
    else:
        production_config = dict(
            distributed_optimizer=distributed_optimizer,
            learning_rate=lr, aux_learning_rate=aux_lr, weight_decay=decay,
            betas=betas, epsilon=epsilon, muon_momentum=momentum,
            muon_nesterov=False, ns_polynomial=cfg.get("ns_polynomial", "paper5"),
            ns_steps=cfg.get("ns_steps", 5),
            data_norm_alpha=alpha if method in ("pd", "ts", "spd", "sts") else 0.0,
            data_norm_out_beta=output_beta if method in ("ts", "sts") else 0.0,
            data_norm_damping=float(cfg.get("damping", 1e-3)),
            data_norm_refresh=refresh, data_norm_post=True, data_norm_pre=True,
            data_norm_decay="decoupled", data_norm_mode="sandwich",
            soap_precondition=method in ("soap", "spd", "sts"),
            soap_beta2=soap_beta2, soap_denom_power=0.5,
            soap_basis="both", soap_second_moment="update", soap_norm="entry",
        )
        if production_config["data_norm_damping"] <= 0:
            raise ValueError("damping must be positive")
        optimizer = MuonAdamW(model, production_config, device, implementation)
        optimizer.param_groups[0]["role"] = "body"
        optimizer.param_groups[1]["role"] = "auxiliary"
    # Configuration is plain metadata, not hidden tensor state.
    optimizer.tiny_config = cfg
    return optimizer


@torch.enable_grad()
def output_second_moments(model, x, y, source="gn", precision="bf16", generator=None, audit=None):
    """Sample B=E[e e^T] without modifying parameter gradients or input EMAs.

    ``gn`` samples independent predictive labels; ``ef`` uses the data labels.
    Here e is the derivative of the *summed* token loss at a linear output,
    including the influence of later tokens through attention. All positions
    enter B, matching the production GN estimator. Supply a dedicated generator
    for GN so the statistics pass does not perturb the training RNG stream.
    """
    if source not in ("gn", "ef"):
        raise ValueError("Output-statistic source must be 'gn' or 'ef'")
    if precision not in ("bf16", "fp32"):
        raise ValueError("precision must be 'bf16' or 'fp32'")
    if x.shape != y.shape or x.ndim != 2 or x.numel() == 0:
        raise ValueError("x and y must have equal, nonempty [batch, time] shapes")
    modules = {module.weight: module for name, module in model.named_modules()
               if isinstance(module, nn.Linear) and name.startswith("blocks.")}
    if len(modules) != 6 * model.config.n_layer:
        raise ValueError("Expected every Q/K/V/O/up/down linear output")
    outputs = {}
    handles = [module.register_forward_hook(
        lambda mod, args, out, weight=weight: outputs.__setitem__(weight, out))
        for weight, module in modules.items()]
    modes = {module: module.training for module in model.modules()}
    model.eval()
    try:
        context = (torch.autocast("cuda", dtype=torch.bfloat16)
                   if x.device.type == "cuda" and precision == "bf16" else nullcontext())
        with context:
            logits = model(x)
        if source == "gn":
            with torch.no_grad():
                probabilities = logits.float().softmax(dim=-1).flatten(0, 1)
                targets = torch.multinomial(probabilities, 1, generator=generator).view_as(y)
        else:
            targets = y
        if audit is not None:
            import hashlib
            audit["target_sha256"] = hashlib.sha256(targets.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten(), reduction="sum")
        weights = list(outputs)
        errors = torch.autograd.grad(loss, [outputs[weight] for weight in weights])
        moments = {}
        with torch.no_grad():
            for weight, error in zip(weights, errors):
                rows = error.float().reshape(-1, error.shape[-1])
                moments[weight] = rows.T @ rows / rows.shape[0]
        return moments
    finally:
        for handle in handles:
            handle.remove()
        for module, training in modes.items():
            module.training = training


def _external_state(optimizer):
    """Tensor-owning structures that the production state_dict omits."""
    return {
        name: getattr(optimizer, name)
        for name in ("soap", "data_norm", "data_norm_sums")
        if getattr(optimizer, name, None) is not None
    }


def _model_statistics(model):
    return {name: dict(module.named_buffers(recurse=False))
            for name, module in model.named_modules() if isinstance(module, StatLinear)}


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, Mapping):
        # Tensor dictionary keys identify parameters; they are not state storage.
        for key, item in value.items():
            if not (isinstance(key, str) and key == "modules"):
                yield from _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def tensor_memory_breakdown(model, optimizer):
    """Count unique live tensor storage bytes, independently of allocator peaks.

    Categories are disjoint: any shared storage is assigned to its first category.
    Transient forward/backward and matrix-decomposition workspaces are excluded;
    the runner must separately report CUDA peak allocated/reserved memory.
    """
    categories = {
        "model_parameters": list(model.parameters()),
        "parameter_gradients": [p.grad for p in model.parameters() if p.grad is not None],
        "optimizer_standard": optimizer.state,
        "optimizer_external": _external_state(optimizer),
        "model_statistics": _model_statistics(model),
    }
    seen = set()
    result = {}
    for category, values in categories.items():
        byte_count, storage_count = 0, 0
        for tensor in _tensors(values):
            storage = tensor.untyped_storage()
            identity = (str(tensor.device), storage._cdata)
            if identity not in seen:
                seen.add(identity)
                byte_count += storage.nbytes()
                storage_count += 1
        result[category] = {"bytes": byte_count, "storages": storage_count}
    result["total"] = {
        "bytes": sum(value["bytes"] for value in result.values()),
        "storages": sum(value["storages"] for value in result.values()),
    }
    return result


def snapshot_optimizer(model, optimizer):
    """Capture complete supported optimizer/statistic state as CPU-only data.

    State dictionaries use parameter names, never Parameter objects or module
    references. This is an analysis snapshot, not an advertised resume format.
    """
    names = {parameter: name for name, parameter in model.named_parameters()}

    def serialize(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value, Mapping):
            result = {}
            for key, item in value.items():
                if isinstance(key, str) and key == "modules":
                    continue  # references into the live model, not optimizer data
                if isinstance(key, nn.Parameter):
                    key = names[key]
                elif not isinstance(key, (str, int, float, bool, type(None))):
                    raise TypeError(f"Unsupported snapshot dictionary key: {type(key).__name__}")
                result[key] = serialize(item)
            return result
        if isinstance(value, (tuple, list)):
            return type(value)(serialize(item) for item in value)
        if isinstance(value, (str, int, float, bool, type(None))):
            return value
        raise TypeError(f"Unsupported snapshot value: {type(value).__name__}")

    groups = []
    for group in optimizer.param_groups:
        groups.append({key: ([names[p] for p in value] if key == "params" else serialize(value))
                       for key, value in group.items()})
    return {
        "format_version": 1,
        "optimizer_class": type(optimizer).__name__,
        "config": serialize(getattr(optimizer, "tiny_config", {})),
        "param_groups": groups,
        "state": serialize(optimizer.state),
        "external": serialize(_external_state(optimizer)),
        "model_statistics": serialize(_model_statistics(model)),
        "tensor_memory": tensor_memory_breakdown(model, optimizer),
    }

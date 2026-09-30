"""Single-device AdamW study, with gradient accumulation and exact update spectra.

Run from the repository root with: python -m research.adamw_spectra.train --help
"""

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shlex
import shutil
import sysconfig
import time

import numpy as np
import torch

from .data import TokenStream, check_disjoint, token_views
from .measure import measure_updates
from .model import GPT, ModelConfig


DEFAULTS = {
    "model": {}, "seed": 260924, "learning_rate": 0.0006,
    "betas": [0.9, 0.95], "epsilon": 1e-8, "weight_decay": 0.01,
    "batch_tokens": 1048576, "microbatch_sequences": 16,
    "tokens_per_parameter": 20, "total_tokens": None,
    "warmup_steps": 50, "cooldown_fraction": 0.1, "grad_clip": 1.0,
    "train_pattern": "data/fineweb10B/fineweb_train_*.bin",
    "validation_pattern": "data/fineweb10B/fineweb_val_*.bin",
    "validation_tokens": 1048576, "validation_every": 50,
    "spectra_every": 25, "svd_device": "cpu", "checkpoint_every": 100,
    "cpu_threads": 4, "precision": "bf16", "compile": True,
    "optimizer_impl": "auto",
    "optimizer": "adamw", "aux_learning_rate": 0.002, "muon_momentum": 0.95,
    "ns_polynomial": "paper5", "ns_steps": 5, "deflation": False,
    "deflation_window": 0.025, "deflation_oversample": 0.025, "deflation_subspace_iters": 1,
    "deflation_threshold": 0.1, "deflation_pad": 1.01,
    "deflation_mode": "paper", "deflation_head_weight": 1.0, "deflation_power_iters": 1,
    "muon_nesterov": False, "muon_prefilter": "none", "head_whitening_alpha": 0.0, "head_whitening_center": False, "head_whitening_norm": "match",
    "muon_momentum_start": -1.0, "muon_momentum_warmup": 0.0, "data_norm_top_only": False, "momentum_split_top": -1.0, "data_norm_snr_gate": False, "data_norm_snr_ema": 0.9, "mean_whitening": False, "mean_whitening_beta": -1.0,
    "mean_whitening_power": 0.5,
    "mean_bias_adam": False,
    "soap_precondition": False, "soap_beta2": 0.9, "soap_denom_power": 0.5, "soap_basis": "both",
    "soap_second_moment": "update", "soap_norm": "entry",
    "data_norm_alpha": 0.0, "data_norm_damping": 1e-3, "data_norm_refresh": 10, "data_norm_post": True,
    "data_norm_rows": False, "data_norm_center": False, "data_norm_pre": True, "data_norm_mode": "sandwich",
    "soap_layers": "all", "data_norm_alpha_by_kind": {},
    # "geometry": weight decay preconditioned like the PD update, W <- W - lr*wd * W R^p / mean eig(R^p).
    "data_norm_decay": "decoupled", "data_norm_decay_power": 2,
    # Published comparators (modded-nanogpt Track-3 records #15 and #18), with their constants.
    "newton_muon": False, "newton_muon_damping": 0.2, "newton_muon_refresh": 64, "newton_muon_ema": 0.05,
    "newton_muon_down_blocks": 4, "pmuon": False, "pmuon_gamma": 0.3, "pmuon_beta": 0.95,
    # Two-sided data norm (second-order audit): L polar(L M R) R with L = (B / mean eig + damping I)^-beta, B an
    # EMA of the per-token output-error second moment E[e e^T] of each body matrix, from an extra eager pass on
    # `data_norm_out_sequences` local sequences every `data_norm_out_refresh` steps; "ef" uses the data labels
    # (empirical Fisher), "gn" labels sampled from the model (Gauss-Newton). beta = 0 is PD.
    "data_norm_out_beta": 0.0, "data_norm_out_source": "ef", "data_norm_out_refresh": 10,
    "data_norm_out_sequences": 8, "data_norm_out_ema": 0.8, "data_norm_out_placebo": False,
    "muon_track_displacement": False,
    "log_gradient_noise": False,
    # Measurement only (second-order audit): steps whose full checkpoint is kept in out/kept/, plus the
    # next step's weights (so the applied update is W[t+1] - W[t]). Training is unchanged.
    "keep_checkpoints": [],
}


def load_config(path=None, overrides=None):
    raw = json.loads(Path(path).read_text()) if path else {}
    raw.update(overrides or {})
    if set(raw) - set(DEFAULTS):
        raise ValueError(f"Unknown config keys: {sorted(set(raw) - set(DEFAULTS))}")
    config = {**DEFAULTS, **raw}
    model = ModelConfig(**config["model"])
    for key in ("batch_tokens", "microbatch_sequences", "validation_tokens",
                "validation_every", "checkpoint_every", "cpu_threads"):
        if not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["spectra_every"] < 0 or config["warmup_steps"] < 0:
        raise ValueError("Spectral interval and warmup must be nonnegative")
    if not isinstance(config["data_norm_out_placebo"], bool):
        raise ValueError("data_norm_out_placebo must be boolean")
    if not isinstance(config["data_norm_snr_gate"], bool) or not 0.0 <= config["data_norm_snr_ema"] < 1.0 or (
            config["data_norm_snr_gate"] and (config["data_norm_alpha"] <= 0 or config["data_norm_top_only"]
                                              or config["data_norm_center"] or config.get("data_norm_alpha_by_kind")
                                              or config["momentum_split_top"] >= 0)):
        raise ValueError("data_norm_snr_gate is a boolean that needs data_norm_alpha > 0 without top_only, center or "
                         "per-kind alpha; data_norm_snr_ema is in [0, 1)")
    if not isinstance(config["log_gradient_noise"], bool):
        raise ValueError("log_gradient_noise must be boolean")
    if not isinstance(config["muon_track_displacement"], bool):
        raise ValueError("muon_track_displacement must be boolean")
    if config["data_norm_out_beta"] < 0 or config["data_norm_out_source"] not in ("ef", "gn") or \
            config["data_norm_out_refresh"] < 1 or config["data_norm_out_sequences"] < 1 or \
            not 0 <= config["data_norm_out_ema"] < 1:
        raise ValueError("Invalid two-sided data-norm settings")
    kept = config["keep_checkpoints"]
    if not isinstance(kept, list) or any(not isinstance(s, int) or s <= 0 for s in kept) or kept != sorted(set(kept)):
        raise ValueError("keep_checkpoints must be a sorted list of distinct positive steps")
    if config["batch_tokens"] % model.seq_len or config["validation_tokens"] % model.seq_len:
        raise ValueError("Token counts must be whole sequences")
    if not 0 < config["cooldown_fraction"] < 1:
        raise ValueError("Cooldown fraction must be between zero and one")
    for key in ("learning_rate", "epsilon", "grad_clip", "tokens_per_parameter"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be positive and finite")
    if not math.isfinite(config["weight_decay"]) or config["weight_decay"] < 0:
        raise ValueError("Invalid weight decay")
    if len(config["betas"]) != 2 or not all(0 <= b < 1 for b in config["betas"]):
        raise ValueError("Invalid Adam betas")
    if config["total_tokens"] is not None and (config["total_tokens"] <= 0 or
                                                config["total_tokens"] % model.seq_len):
        raise ValueError("Explicit total_tokens must be positive and sequence-aligned")
    if config["precision"] not in ("fp32", "bf16"):
        raise ValueError("Precision must be fp32 or bf16")
    if config["optimizer_impl"] not in ("auto", "single", "foreach", "fused"):
        raise ValueError("optimizer_impl must be auto, single, foreach, or fused")
    if not isinstance(config["compile"], bool):
        raise ValueError("compile must be boolean")
    if config["optimizer"] not in ("adamw", "muon"):
        raise ValueError("optimizer must be adamw or muon")
    if not 0 <= config["muon_momentum"] < 1:
        raise ValueError("Invalid Muon momentum")
    if not math.isfinite(config["aux_learning_rate"]) or config["aux_learning_rate"] <= 0:
        raise ValueError("Invalid auxiliary AdamW learning rate")
    if config["ns_polynomial"] not in ("paper5", "jordan", "svd"):
        raise ValueError("ns_polynomial must be paper5, jordan or svd")
    if not isinstance(config["ns_steps"], int) or config["ns_steps"] < 1 or (
            config["ns_polynomial"] in ("paper5", "svd") and config["ns_steps"] != 5):
        raise ValueError("paper5 has exactly five polynomials (svd keeps ns_steps 5); jordan needs ns_steps >= 1")
    if config["ns_polynomial"] == "svd" and config["deflation"]:
        raise ValueError("Exact polar needs no deflation")
    if not isinstance(config["deflation"], bool) or (config["deflation"] and config["optimizer"] != "muon"):
        raise ValueError("deflation is a boolean Muon option")
    if not (0 < config["deflation_window"] < 1 and 0 <= config["deflation_oversample"] < 1
            and 0 < config["deflation_threshold"] < 1 and config["deflation_pad"] >= 1
            and isinstance(config["deflation_subspace_iters"], int) and config["deflation_subspace_iters"] >= 0):
        raise ValueError("Invalid deflation constants")
    if config["deflation_mode"] not in ("paper", "track1") or not 0 <= config["deflation_head_weight"] <= 1 or (
            not isinstance(config["deflation_power_iters"], int) or config["deflation_power_iters"] < 1):
        raise ValueError("Invalid tracked-deflation settings")
    if not isinstance(config["muon_nesterov"], bool) or not isinstance(config["mean_whitening"], bool):
        raise ValueError("muon_nesterov and mean_whitening are booleans")
    if config["muon_prefilter"] not in ("none", "two_tap") or (config["muon_prefilter"] != "none" and config["muon_nesterov"]):
        raise ValueError("muon_prefilter is none or two_tap, and is not combined with Nesterov")
    if not 0.0 <= config["head_whitening_alpha"] <= 1.0 or (
            config["head_whitening_alpha"] > 0 and (config["optimizer"] != "muon" or not config["model"].get("track_head_cov", False))):
        raise ValueError("head_whitening_alpha is in [0, 1] and needs Muon with model track_head_cov")
    if not isinstance(config["head_whitening_center"], bool) or (config["head_whitening_center"] and config["head_whitening_alpha"] <= 0):
        raise ValueError("head_whitening_center is a boolean that needs head_whitening_alpha > 0")
    if config["head_whitening_norm"] not in ("match", "none"):
        raise ValueError("head_whitening_norm is match or none")
    if (config["muon_momentum_start"] >= 0) != (config["muon_momentum_warmup"] > 0) or not (
            config["muon_momentum_start"] < 1 and 0 <= config["muon_momentum_warmup"] <= 1):
        raise ValueError("muon_momentum_start in [0, 1) and muon_momentum_warmup in (0, 1] go together (both off by default)")
    if not isinstance(config["data_norm_top_only"], bool) or (config["data_norm_top_only"] and config["data_norm_alpha"] <= 0):
        raise ValueError("data_norm_top_only is a boolean that needs data_norm_alpha > 0")
    if not (config["momentum_split_top"] == -1.0 or 0.0 <= config["momentum_split_top"] < 1.0) or (
            config["momentum_split_top"] >= 0 and (config["data_norm_alpha"] <= 0 or config["muon_nesterov"]
                                                   or config["muon_prefilter"] != "none" or config["soap_precondition"])):
        raise ValueError("momentum_split_top is -1 (off) or in [0, 1); it needs data_norm_alpha > 0, no Nesterov, "
                         "no prefilter and no SOAP")
    beta = config["mean_whitening_beta"]
    if not (beta == -1.0 or 0.0 <= beta <= 1.0):
        raise ValueError("mean_whitening_beta is -1 (automatic) or in [0, 1]")
    if not 0.0 < config["mean_whitening_power"] <= 1.0:
        raise ValueError("mean_whitening_power must be in (0, 1]")
    if config["mean_whitening"] and (config["optimizer"] != "muon" or config["deflation"]
                                     or not model.track_input_stats):
        raise ValueError("mean_whitening needs Muon, no deflation and model track_input_stats")
    if not isinstance(config["mean_bias_adam"], bool) or (config["mean_bias_adam"] and (
            not config["mean_whitening"] or config["mean_whitening_beta"] != 0.0 or config["soap_precondition"])):
        raise ValueError("mean_bias_adam is a boolean that needs mean_whitening with beta 0 and no SOAP")
    if not isinstance(config["soap_precondition"], bool) or (config["soap_precondition"] and (
            config["optimizer"] != "muon" or config["deflation"])):
        raise ValueError("soap_precondition is a boolean Muon option without deflation")
    if config["soap_basis"] not in ("both", "left", "right", "right_act", "none"):
        raise ValueError("soap_basis is both, left, right, right_act or none")
    if config["soap_basis"] == "right_act" and config["soap_precondition"] and not model.track_input_cov:
        raise ValueError("soap_basis right_act needs model track_input_cov")
    if config["soap_norm"] not in ("entry", "column"):
        raise ValueError("soap_norm is entry or column")
    if config["soap_second_moment"] not in ("update", "gradient"):
        raise ValueError("soap_second_moment is update or gradient")
    if not (0.0 <= config["data_norm_alpha"] <= 1.0 and config["data_norm_damping"] > 0
            and isinstance(config["data_norm_refresh"], int) and config["data_norm_refresh"] >= 1
            and isinstance(config["data_norm_post"], bool) and isinstance(config["data_norm_rows"], bool)
            and isinstance(config["data_norm_center"], bool) and isinstance(config["data_norm_pre"], bool)
            and config["data_norm_mode"] in ("sandwich", "magnitude_matched")
            and config["data_norm_decay"] in ("decoupled", "geometry") and config["data_norm_decay_power"] in (1, 2)
            and (config["data_norm_decay"] == "decoupled" or config["data_norm_alpha"] > 0)):
        raise ValueError("Invalid data-norm settings")
    if config["data_norm_alpha"] > 0 and (
            config["optimizer"] != "muon" or config["deflation"] or config["mean_whitening"]
            or config["mean_bias_adam"] or not model.track_input_cov
            or (config["soap_precondition"] and config["soap_basis"] == "right_act")):
        raise ValueError("data_norm needs Muon with track_input_cov and no deflation or whitening "
                         "(SOAP runs in whitened coordinates; right_act is not combinable)")
    by_kind = config["data_norm_alpha_by_kind"]
    if not isinstance(by_kind, dict) or not set(by_kind) <= {"q", "k", "v", "o", "up", "down"} or not all(
            0.0 <= float(a) <= 1.0 for a in by_kind.values()) or (by_kind and config["data_norm_alpha"] <= 0):
        raise ValueError("data_norm_alpha_by_kind maps q/k/v/o/up/down to alpha in [0, 1] (needs data_norm_alpha > 0)")
    if config["soap_layers"] not in ("all", "down", "not_down"):
        raise ValueError("soap_layers is all, down or not_down")
    if not (0 <= config["soap_beta2"] < 1 and 0 < config["soap_denom_power"] <= 1):
        raise ValueError("Invalid SOAP constants")
    if not (isinstance(config["newton_muon"], bool) and isinstance(config["pmuon"], bool)
            and config["newton_muon_damping"] > 0 and 0 < config["newton_muon_ema"] <= 1
            and isinstance(config["newton_muon_refresh"], int) and config["newton_muon_refresh"] >= 1
            and isinstance(config["newton_muon_down_blocks"], int) and config["newton_muon_down_blocks"] >= 1
            and 0 < config["pmuon_gamma"] <= 1 and 0 <= config["pmuon_beta"] < 1):
        raise ValueError("Invalid Newton-Muon or PMuon settings")
    if (config["newton_muon"] or config["pmuon"]) and (
            config["optimizer"] != "muon" or config["deflation"] or config["mean_whitening"]
            or config["soap_precondition"] or config["data_norm_alpha"] > 0
            or (config["newton_muon"] and config["pmuon"])):
        raise ValueError("newton_muon and pmuon are Muon options that combine with no other preconditioner")
    if config["newton_muon"] and (not model.track_input_cov or (
            model.n_embd * 4) % config["newton_muon_down_blocks"]):
        raise ValueError("newton_muon needs model track_input_cov and MLP width divisible by its down blocks")
    return config


def token_budget(config, parameter_count, seq_len):
    return config["total_tokens"] or math.ceil(
        config["tokens_per_parameter"] * parameter_count / seq_len) * seq_len


def muon_momentum(config, tokens_before, budget):
    """The Muon momentum at an update: constant, or linear in tokens from muon_momentum_start to muon_momentum over the
    first muon_momentum_warmup fraction of the budget (second-order audit, 2026-09-28: a short window pays early at
    large batch, a long one late)."""
    if config.get("muon_momentum_warmup", 0.0) <= 0:
        return config["muon_momentum"]
    progress = min(1.0, tokens_before / (config["muon_momentum_warmup"] * budget))
    return config["muon_momentum_start"] + (config["muon_momentum"] - config["muon_momentum_start"]) * progress


def learning_rate(config, step, tokens_before, budget):
    """One-based update; preserve full-run schedule in stopped prefix pilots."""
    warmup = min(1.0, step / max(1, config["warmup_steps"]))
    cooldown_start = budget * (1 - config["cooldown_fraction"])
    cooldown = min(1.0, (budget - tokens_before) / (budget - cooldown_start))
    return config["learning_rate"] * warmup * max(0.0, cooldown)


def atomic_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    os.replace(temp, path)


def source_hashes():
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).parent.glob("*.py"))}


def model_hash(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def amp_context(device, config):
    if device.type == "cuda" and config["precision"] == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_optimizer(model, config, device):
    implementation = config["optimizer_impl"]
    if implementation == "auto":
        implementation = "fused" if torch.device(device).type == "cuda" else "foreach"
    if config["optimizer"] == "muon":
        from .muon import MuonAdamW
        return MuonAdamW(model, config, device, implementation), "muon_ns5+adamw_" + implementation
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 betas=tuple(config["betas"]), eps=config["epsilon"],
                                 weight_decay=config["weight_decay"],
                                 foreach=implementation == "foreach", fused=implementation == "fused")
    return optimizer, implementation


def native_compilers():
    """Match this Python's headers, not an inherited cross-environment compiler."""
    resolved = {}
    for variable, fallback in (("CC", "cc"), ("CXX", "c++")):
        executable = shlex.split(sysconfig.get_config_var(variable) or fallback)[0]
        path = shutil.which(executable)
        if path is None:
            raise RuntimeError(f"Compiled execution requires Python's compiler: {executable}")
        os.environ[variable] = resolved[variable] = path
    return resolved


@torch.no_grad()
def evaluate(model, validation_tokens, config, device):
    model.eval()
    count = config["validation_tokens"]
    seq_len = model.config.seq_len
    micro = config["microbatch_sequences"] * seq_len
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    for offset in range(0, count, micro):
        tokens = min(micro, count - offset)
        x, y = token_views(validation_tokens, offset, tokens, seq_len)
        with amp_context(device, config):
            loss = model(x, y)
        total_loss += loss.double() * tokens
    model.train()
    result = float((total_loss / count).item())
    if not math.isfinite(result):
        raise RuntimeError("Non-finite validation loss")
    return result


def save_checkpoint(path, model, optimizer, step, tokens, config, metadata):
    model_device = next(model.parameters()).device
    payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
               "step": step, "tokens": tokens, "config": config,
               "metadata": metadata, "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state(model_device) if model_device.type == "cuda" else None,
               "numpy_rng": np.random.get_state(), "python_rng": random.getstate()}
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    atomic_json(path.with_suffix(".json"), {"step": step, "tokens": tokens})


def run(config, out, *, device="cuda", stop_after=None, resume=False):
    invocation_started = time.perf_counter()
    config = load_config(overrides=config)
    if config["optimizer"] == "muon":
        raise ValueError("Use distributed.py for Muon, including world size 1, to record both spectral objects")
    out = Path(out)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use --device cpu for qualification")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    torch.set_num_threads(config["cpu_threads"])
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    # Avoid TF32 changes in spectral inputs or FP32 reference checks.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train = TokenStream(config["train_pattern"])
    validation = TokenStream(config["validation_pattern"])
    check_disjoint(train, validation)
    model = GPT(ModelConfig(**config["model"])).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    budget = token_budget(config, parameter_count, model.config.seq_len)
    if train.total < budget + 1:
        raise ValueError(f"Need {budget + 1} training tokens, have {train.total}; no recycling")
    if validation.total < config["validation_tokens"] + 1:
        raise ValueError("Insufficient held-out validation tokens")
    if config["warmup_steps"] * config["batch_tokens"] >= budget * (1 - config["cooldown_fraction"]):
        raise ValueError("Warmup overlaps cooldown; choose a meaningful training horizon")
    if out.exists() and any(out.iterdir()) and not resume:
        raise FileExistsError(f"Refusing to overwrite a run: {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "steps").mkdir(exist_ok=True)
    (out / "spectra").mkdir(exist_ok=True)
    optimizer, optimizer_impl = make_optimizer(model, config, device)
    compiled = config["compile"] and device.type == "cuda"
    compilers = native_compilers() if compiled else {}
    measured = model.measured_parameters()
    metadata = {"schema_version": 1, "config": config, "parameter_count": parameter_count,
                "budget_tokens": budget, "total_steps": math.ceil(budget / config["batch_tokens"]),
                "initial_model_sha256": model_hash(model), "source_sha256": source_hashes(),
                "train_manifest": train.manifest, "validation_manifest": validation.manifest,
                "measured_parameters": {k: list(p.shape) for k, p in measured.items()},
                "torch_version": torch.__version__, "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "effective_precision": config["precision"] if device.type == "cuda" else "fp32",
                "effective_optimizer_impl": optimizer_impl, "effective_compile": compiled,
                "compilers": compilers,
                "measurement": "bias-corrected adaptive AdamW direction, excluding LR and weight decay"}
    step, tokens = 0, 0
    if resume:
        saved = torch.load(out / "checkpoint.pt", map_location=device, weights_only=False)
        old = saved["metadata"]
        for field in ("config", "source_sha256", "train_manifest", "validation_manifest",
                      "parameter_count", "budget_tokens", "effective_precision", "device",
                      "torch_version", "device_name", "effective_optimizer_impl", "effective_compile",
                      "compilers"):
            if metadata[field] != old[field]:
                raise ValueError(f"Resume mismatch: {field}")
        metadata = old
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, tokens = saved["step"], saved["tokens"]
        torch.set_rng_state(saved["torch_rng"].cpu())
        if saved["cuda_rng"] is not None and device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"].cpu(), device)
        np.random.set_state(saved["numpy_rng"])
        random.setstate(saved["python_rng"])
        # Preserve speculative records written after the last durable checkpoint.
        orphaned = out / f"uncommitted_{time.time_ns()}"
        for folder in ("steps", "spectra"):
            for path in (out / folder).glob("step*"):
                artifact_step = int(path.name.split(".")[0].removeprefix("step"))
                if artifact_step > step or path.suffix == ".tmp":
                    (orphaned / folder).mkdir(parents=True, exist_ok=True)
                    path.rename(orphaned / folder / path.name)
    else:
        atomic_json(out / "metadata.json", metadata)
        (out / "source").mkdir(exist_ok=True)
        for source in Path(__file__).parent.glob("*.py"):
            (out / "source" / source.name).write_bytes(source.read_bytes())
        (out / "source" / "PROTOCOL.md").write_bytes(Path(__file__).with_name("PROTOCOL.md").read_bytes())
    # Only 8 MiB at the default budget; reuse exactly the same held-out tokens.
    validation_tokens = validation.device_tokens(0, config["validation_tokens"] + 1, device)
    # CPU qualification stays eager; CUDA uses Inductor's default (no autotune sweep).
    train_model = torch.compile(model, dynamic=False) if compiled else model
    if not resume:
        initial_nll = evaluate(train_model, validation_tokens, config, device)
        atomic_json(out / "steps" / "step000000.json",
                    {"step": 0, "tokens": 0, "validation_nll": initial_nll})
    atomic_json(out / "status.json", {"status": "running", "step": step,
                                     "tokens": tokens, "budget_tokens": budget})
    setup_seconds = time.perf_counter() - invocation_started
    limit = min(stop_after or metadata["total_steps"], metadata["total_steps"])
    try:
        while step < limit and tokens < budget:
            step += 1
            lr = learning_rate(config, step, tokens, budget)
            for group in optimizer.param_groups:
                group["lr"] = lr
            if hasattr(optimizer, "momentum") and config["optimizer"] == "muon":
                optimizer.momentum = muon_momentum(config, tokens, budget)
            batch = min(config["batch_tokens"], budget - tokens)
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            begin = time.perf_counter()
            # One H2D transfer per optimizer update rather than one per microbatch.
            batch_tokens = train.device_tokens(tokens, batch + 1, device)
            total_loss = torch.zeros((), device=device)
            micro = config["microbatch_sequences"] * model.config.seq_len
            for offset in range(0, batch, micro):
                count = min(micro, batch - offset)
                x, y = token_views(batch_tokens, offset, count, model.config.seq_len)
                with amp_context(device, config):
                    loss = train_model(x, y)
                (loss * (count / batch)).backward()
                total_loss += loss.detach() * (count / batch)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"],
                                                       error_if_nonfinite=True, foreach=True)
            measure = config["spectra_every"] > 0 and (
                step == 1 or step % config["spectra_every"] == 0 or tokens + batch == budget)
            decay_norms = {}
            if measure:
                norms = torch.stack([p.detach().norm() for p in measured.values()]).cpu().tolist()
                decay_norms = {name: lr * config["weight_decay"] * norm
                               for name, norm in zip(measured, norms)}
            optimizer.step()
            synchronize(device)
            training_seconds = time.perf_counter() - begin
            tokens += batch
            nll, gn = float(total_loss.item()), float(grad_norm.item())
            if not math.isfinite(nll):
                raise RuntimeError("Non-finite training loss")
            row = {"step": step, "tokens": tokens, "batch_tokens": batch, "lr": lr,
                   "train_nll": nll, "gradient_norm_before_clip": gn,
                   "gradient_clipped": gn > config["grad_clip"],
                   "training_seconds": training_seconds}
            begin = time.perf_counter()
            if measure:
                row["matrices"], values = measure_updates(optimizer, measured,
                    device=config["svd_device"], decay_norms=decay_norms)
                path = out / "spectra" / f"step{step:06d}.npz"
                temp = path.with_suffix(".tmp")
                with temp.open("wb") as handle:
                    np.savez_compressed(handle, **values)
                os.replace(temp, path)
            row["measurement_seconds"] = time.perf_counter() - begin
            begin = time.perf_counter()
            if step % config["validation_every"] == 0 or tokens == budget or step == limit:
                row["validation_nll"] = evaluate(train_model, validation_tokens, config, device)
            row["validation_seconds"] = time.perf_counter() - begin
            atomic_json(out / "steps" / f"step{step:06d}.json", row)
            if step % config["checkpoint_every"] == 0 or step == limit:
                save_checkpoint(out / "checkpoint.pt", model, optimizer, step, tokens, config, metadata)
            if step == 1 or step % 10 == 0 or step == limit:
                print(json.dumps({k: v for k, v in row.items() if k != "matrices"}), flush=True)
        summary = {"status": "complete" if tokens == budget else "stopped_at_requested_step",
                   "step": step, "tokens": tokens, "budget_tokens": budget,
                   "setup_seconds": setup_seconds,
                   "invocation_seconds": time.perf_counter() - invocation_started}
        if device.type == "cuda":
            summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
            summary["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
        atomic_json(out / "status.json", summary)
        return summary
    except Exception as exc:
        atomic_json(out / "status.json", {"status": "failed", "step": step,
                    "tokens": tokens, "error": f"{type(exc).__name__}: {exc}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int, help="Stop at this absolute step without changing schedule")
    parser.add_argument("--resume", action="store_true", help="Resume out/checkpoint.pt with identical config")
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    overrides = {k: v for k, v in {"learning_rate": args.learning_rate, "seed": args.seed}.items()
                 if v is not None}
    run(load_config(args.config, overrides), args.out, device=args.device,
        stop_after=args.stop_after, resume=args.resume)


if __name__ == "__main__":
    main()

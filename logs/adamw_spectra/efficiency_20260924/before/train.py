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
import time

import numpy as np
import torch

from .data import TokenStream, check_disjoint
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
    "cpu_threads": 4, "precision": "bf16", "compile": False,
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
    return config


def token_budget(config, parameter_count, seq_len):
    return config["total_tokens"] or math.ceil(
        config["tokens_per_parameter"] * parameter_count / seq_len) * seq_len


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


@torch.no_grad()
def evaluate(model, stream, config, device):
    model.eval()
    count = config["validation_tokens"]
    seq_len = model.config.seq_len
    micro = config["microbatch_sequences"] * seq_len
    total_loss = torch.zeros((), dtype=torch.float64, device=device)
    for offset in range(0, count, micro):
        tokens = min(micro, count - offset)
        x, y = stream.batch(offset, tokens // seq_len, seq_len, device)
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
    config = load_config(overrides=config)
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 betas=tuple(config["betas"]), eps=config["epsilon"],
                                 weight_decay=config["weight_decay"], foreach=False, fused=False)
    measured = model.measured_parameters()
    metadata = {"schema_version": 1, "config": config, "parameter_count": parameter_count,
                "budget_tokens": budget, "total_steps": math.ceil(budget / config["batch_tokens"]),
                "initial_model_sha256": model_hash(model), "source_sha256": source_hashes(),
                "train_manifest": train.manifest, "validation_manifest": validation.manifest,
                "measured_parameters": {k: list(p.shape) for k, p in measured.items()},
                "torch_version": torch.__version__, "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "effective_precision": config["precision"] if device.type == "cuda" else "fp32",
                "measurement": "bias-corrected adaptive AdamW direction, excluding LR and weight decay"}
    step, tokens = 0, 0
    if resume:
        saved = torch.load(out / "checkpoint.pt", map_location=device, weights_only=False)
        old = saved["metadata"]
        for field in ("config", "source_sha256", "train_manifest", "validation_manifest",
                      "parameter_count", "budget_tokens", "effective_precision", "device",
                      "torch_version", "device_name"):
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
        initial_nll = evaluate(model, validation, config, device)
        atomic_json(out / "steps" / "step000000.json",
                    {"step": 0, "tokens": 0, "validation_nll": initial_nll})
    train_model = torch.compile(model) if config["compile"] else model
    atomic_json(out / "status.json", {"status": "running", "step": step,
                                     "tokens": tokens, "budget_tokens": budget})
    started = time.perf_counter()
    limit = min(stop_after or metadata["total_steps"], metadata["total_steps"])
    try:
        while step < limit and tokens < budget:
            step += 1
            lr = learning_rate(config, step, tokens, budget)
            for group in optimizer.param_groups:
                group["lr"] = lr
            batch = min(config["batch_tokens"], budget - tokens)
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            begin = time.perf_counter()
            total_loss = torch.zeros((), device=device)
            micro = config["microbatch_sequences"] * model.config.seq_len
            for offset in range(0, batch, micro):
                count = min(micro, batch - offset)
                x, y = train.batch(tokens + offset, count // model.config.seq_len,
                                   model.config.seq_len, device)
                with amp_context(device, config):
                    loss = train_model(x, y)
                (loss * (count / batch)).backward()
                total_loss += loss.detach() * (count / batch)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"],
                                                       error_if_nonfinite=True)
            measure = config["spectra_every"] > 0 and (
                step == 1 or step % config["spectra_every"] == 0 or tokens + batch == budget)
            decay_norms = ({name: lr * config["weight_decay"] * float(p.detach().norm().item())
                            for name, p in measured.items()} if measure else {})
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
            synchronize(device)
            row["measurement_seconds"] = time.perf_counter() - begin
            begin = time.perf_counter()
            if step % config["validation_every"] == 0 or tokens == budget or step == limit:
                row["validation_nll"] = evaluate(model, validation, config, device)
            row["validation_seconds"] = time.perf_counter() - begin
            atomic_json(out / "steps" / f"step{step:06d}.json", row)
            if step % config["checkpoint_every"] == 0 or step == limit:
                save_checkpoint(out / "checkpoint.pt", model, optimizer, step, tokens, config, metadata)
            if step == 1 or step % 10 == 0 or step == limit:
                print(json.dumps({k: v for k, v in row.items() if k != "matrices"}), flush=True)
        summary = {"status": "complete" if tokens == budget else "stopped_at_requested_step",
                   "step": step, "tokens": tokens, "budget_tokens": budget,
                   "invocation_seconds": time.perf_counter() - started}
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

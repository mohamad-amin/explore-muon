"""Token-exact, single-node DDP execution of the AdamW spectral study."""

import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .data import TokenStream, check_disjoint, token_views
from .measure import measure_updates
from .model import GPT, ModelConfig, StatLinear
from .train import (amp_context, atomic_json, learning_rate, load_config, muon_momentum,
                    make_optimizer, model_hash, native_compilers, source_hashes,
                    synchronize, token_budget)


def partition_sequences(count, world, rank):
    """Return this rank's start/count without dropping, padding, or duplication."""
    base, extra = divmod(count, world)
    return rank * base + min(rank, extra), base + int(rank < extra)


def global_rounds(sequences, world, micro_sequences):
    return math.ceil(math.ceil(sequences / world) / micro_sequences)


def scalar_reduce(value, device, op=dist.ReduceOp.SUM):
    tensor = torch.as_tensor(value, dtype=torch.float64, device=device).clone()
    dist.all_reduce(tensor, op=op)
    return float(tensor.item())


def rng_state(device):
    return {"torch": torch.get_rng_state(), "numpy": np.random.get_state(),
            "python": random.getstate(),
            "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None}


def restore_rng(state, device):
    torch.set_rng_state(state["torch"].cpu())
    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)


def audit_replicas(model, optimizer, out, step, rank, world):
    """Qualification-only exact agreement of all weights and optimizer state."""
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        for value in [parameter] + [optimizer.state[parameter][key]
                                   for key in sorted(optimizer.state[parameter])]:
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    gathered = [None] * world
    dist.all_gather_object(gathered, digest.hexdigest())
    if rank == 0:
        atomic_json(out / f"replica_audit_step{step:06d}.json",
                    {"step": step, "rank_hashes": gathered, "all_equal": len(set(gathered)) == 1,
                     "scope": "all parameter bytes and all optimizer state tensors"})
    if len(set(gathered)) != 1:
        if sum(p.numel() for p in model.parameters()) < 1_000_000:
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "gradients": {name: p.grad for name, p in model.named_parameters()}},
                       out / f"replica_failure_rank{rank}.pt")
            dist.barrier()
        raise RuntimeError("Replicated parameters or optimizer states differ across ranks")


def checkpoint(path, model, optimizer, step, tokens, config, metadata, device, rank, world):
    states = [None] * world if rank == 0 else None
    dist.gather_object(rng_state(device), states, dst=0)
    if rank == 0:
        payload = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "step": step, "tokens": tokens, "config": config,
                   "metadata": metadata, "rank_rng": states}
        temporary = path.with_suffix(".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        atomic_json(path.with_suffix(".json"), {"step": step, "tokens": tokens, "world_size": world})
    dist.barrier()


@torch.no_grad()
def evaluate(model, forward, cached, local_sequences, config, device):
    model.eval()
    sequence = model.config.seq_len
    total = torch.zeros((), device=device, dtype=torch.float64)
    micro = config["microbatch_sequences"] * sequence
    local_tokens = local_sequences * sequence
    for offset in range(0, local_tokens, micro):
        count = min(micro, local_tokens - offset)
        x, y = token_views(cached, offset, count, sequence)
        with amp_context(device, config):
            loss = forward(x, y)
        total += loss.double() * count
    result = scalar_reduce(total, device) / config["validation_tokens"]
    model.train()
    if not math.isfinite(result):
        raise RuntimeError("Non-finite global validation loss")
    return result


def run(config, out, *, device="cuda", stop_after=None, resume=False, audit=False):
    config = load_config(overrides=config)
    rank, world, local_rank = (int(os.environ[k]) for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"))
    device = torch.device("cuda", local_rank) if device == "cuda" else torch.device(device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(config["cpu_threads"])
    dist.init_process_group("nccl" if device.type == "cuda" else "gloo",
                            timeout=datetime.timedelta(minutes=20))
    try:
        return _run(config, Path(out), device, stop_after, resume, rank, world, audit)
    finally:
        dist.destroy_process_group()


def _run(config, out, device, stop_after, resume, rank, world, audit):
    invocation_start = time.perf_counter()
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    train, validation = TokenStream(config["train_pattern"]), TokenStream(config["validation_pattern"])
    check_disjoint(train, validation)
    model = GPT(ModelConfig(**config["model"])).to(device)
    sequence = model.config.seq_len
    count = sum(p.numel() for p in model.parameters())
    budget = token_budget(config, count, sequence)
    steps_total = math.ceil(budget / config["batch_tokens"])
    if config["batch_tokens"] // sequence < world:
        raise ValueError("The full global batch must contain at least one sequence per rank")
    if train.total < budget + 1 or validation.total < config["validation_tokens"] + 1:
        raise ValueError("Insufficient tokens; distributed execution will not recycle data")
    if config["warmup_steps"] * config["batch_tokens"] >= budget * (1 - config["cooldown_fraction"]):
        raise ValueError("Warmup overlaps cooldown")
    if stop_after is not None and stop_after < 1:
        raise ValueError("stop_after must be positive")
    if rank == 0:
        if out.exists() and any(out.iterdir()) and not resume:
            raise FileExistsError(f"Refusing to overwrite {out}")
        for directory in (out, out / "steps", out / "spectra", out / "spectra_momentum"):
            directory.mkdir(parents=True, exist_ok=True)
        if config["keep_checkpoints"]:
            (out / "kept").mkdir(exist_ok=True)
    dist.barrier()
    optimizer, optimizer_impl = make_optimizer(model, config, device)
    compiled = config["compile"] and device.type == "cuda"
    compilers = native_compilers() if compiled else {}
    metadata = {"schema_version": 2, "config": config, "parameter_count": count,
                "budget_tokens": budget, "total_steps": steps_total, "world_size": world,
                "initial_model_sha256": model_hash(model) if rank == 0 else None,
                "source_sha256": source_hashes(), "train_manifest": train.manifest,
                "validation_manifest": validation.manifest,
                "measured_parameters": {k: list(p.shape) for k, p in model.measured_parameters().items()},
                "torch_version": torch.__version__, "device": device.type,
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "effective_precision": config["precision"] if device.type == "cuda" else "fp32",
                "effective_optimizer_impl": optimizer_impl, "effective_compile": compiled,
                "compilers": compilers,
                "gradient_reduction": "DDP mean with micro-loss weight world_size * actual_micro_tokens / actual_global_batch",
                "measurement": "bias-corrected AdamW adaptive update; exact independent SVDs partitioned across ranks"}
    if config["optimizer"] == "muon":
        from .muon import ns_schedule
        metadata.update(
            measurement="captured post-NS update (including shape scaling), excluding LR and decay",
            momentum_measurement="FP32 momentum normalized before BF16 casting; not the rounded BF16 NS input",
            ns_coefficients=ns_schedule(config), nesterov=False,
            ns_execution="eager batched BF16 polynomial; model compilation independent",
            ns_normalization="FP32 Frobenius norm clamped at 1e-7, then BF16 on CUDA",
            muon_shape_scale="sqrt(max(1, original rows / original columns))",
            muon_direction_distribution="single owner per matrix, round-robin within shape; sum disjoint FP32 direction slices",
            parameter_optimizers={name: ("muon" if name.startswith("blocks.") and p.ndim == 2 else "adamw")
                                  for name, p in model.named_parameters()})
        if config["mean_whitening"]:
            metadata["mean_whitening"] = (
                "polar(M P) P with P = I - (1 - beta) v v^T; v = EMA token-mean input (position 0 "
                "excluded, decay 0.99 per training forward; per-rank statistics, broadcast_buffers=False, "
                "so the owning rank's statistics set v and beta); beta "
                + (f"= (c0/(c0+||xbar||^2))^{config['mean_whitening_power']}" if config["mean_whitening_beta"] < 0
                   else f"fixed {config['mean_whitening_beta']}"))
        if config["mean_bias_adam"]:
            metadata["mean_bias_adam"] = (
                "bias-split Muon: the Muon direction excludes v (beta 0); the v-column, the implicit bias "
                "b = W xbar, takes adam(G v) v^T / ||xbar|| at the auxiliary rate (config betas and epsilon, "
                "bias-corrected, raw clipped gradient); Adam state owner-local, not checkpointed")
        if config["data_norm_alpha"] > 0:
            metadata["data_norm"] = (
                f"partial data-norm Muon ({config['data_norm_mode']}{', SOAP in whitened coordinates' if config['soap_precondition'] else ''}): "
                f"polar({'D^-1 ' if config['data_norm_rows'] else ''}M{' R' if config['data_norm_pre'] else ''})"
                f"{' R' if config['data_norm_post'] else ' (no post-multiplication)'}"
                f"{' with D the running RMS of each row of M R (EMA 0.95, owner-local)' if config['data_norm_rows'] else ''}, "
                f"R = C^-{config['data_norm_alpha_by_kind'] or config['data_norm_alpha']} with C the EMA {'centered covariance' if config['data_norm_center'] else ''} "
                f"uncentered input second moment (every 32nd position from 1, decay 0.998 per training "
                f"forward, per-rank), scaled to unit mean eigenvalue plus {config['data_norm_damping']} I, "
                f"FP64 eigh every {config['data_norm_refresh']} steps (owner-local cache); update rescaled to "
                "Muon's Frobenius norm sqrt(min(m, n)) before the shape scale")
        if config["soap_precondition"]:
            metadata["soap_precondition"] = (
                f"SOAP-style momentum preconditioning before NS (Track-3 SOAP-Muon core): eigenbases of "
                f"EMA(GG^T), EMA(G^T G) (beta2 {config['soap_beta2']}, sorted-QR refresh every step; rotated "
                f"sides: {config['soap_basis']}), divided by the EMA second moment of the projected "
                f"{'gradient' if config['soap_second_moment'] == 'gradient' else 'momentum'} "
                f"^{config['soap_denom_power']} ({config['soap_norm']}wise; layers {config['soap_layers']}; right_act = input basis from the "
                "activation second moment), Frobenius norm restored; "
                "statistics owner-local, not checkpointed")
        if config.get("data_norm_top_only", False):
            metadata["data_norm_top_only"] = "input root eigenvalues clamped at 1: only above-mean input directions suppressed (PD-top)"
        if config.get("data_norm_snr_gate", False):
            metadata["data_norm_snr_gate"] = ("PD root factors min(1, f) + max(0, f - 1) w per input direction, w = SNR/(1+SNR) of the "
                                              "direction's log-u bin; SNR from across-rank projected gradient noise, EMA "
                                              f"{config.get('data_norm_snr_ema', 0.9)}; C averaged over ranks at each refresh")
        if config.get("log_gradient_noise", False):
            metadata["log_gradient_noise"] = ("measurement only: per-step across-rank noise energy r and drift q of the mean "
                                              "gradient per hidden matrix (comm hook before the all-reduce, before clipping)")
        if config.get("momentum_split_top", -1.0) >= 0:
            metadata["momentum_split_top"] = (f"input to the preconditioner: (1-b_top) M_short P + (1-b) M (I - P), M_short an EMA with "
                                              f"b_top = {config['momentum_split_top']}, P the projector on above-mean input directions")
        if config.get("muon_momentum_warmup", 0.0) > 0:
            metadata["momentum_schedule"] = (f"linear in tokens from {config['muon_momentum_start']} to {config['muon_momentum']} "
                                             f"over the first {config['muon_momentum_warmup']} of the budget")
        if config.get("head_whitening_alpha", 0.0) > 0:
            metadata["head_whitening"] = (f"unembedding: dW = -Adam(G R) R, R = (C_h / mean + damping)^-{config['head_whitening_alpha']} "
                                          "from the head's input second moment (rank 0; head broadcast), norm-matched; decoupled decay"
                                          + ("; centered covariance (about the EMA mean)" if config.get("head_whitening_center", False) else "")
                                          + ("; no norm matching (pure Adam in whitened coordinates)" if config.get("head_whitening_norm", "match") == "none" else ""))
        if config.get("muon_prefilter", "none") != "none":
            metadata["momentum_prefilter"] = "two_tap: M <- beta M + (g_t + g_{t-1}) / 2 (first step: g_1)"
        if config["muon_nesterov"]:
            metadata["nesterov"] = True
            metadata["ns_input"] = "gradient + momentum * buffer (Nesterov lookahead)"
        if config["deflation"] and config["deflation_mode"] == "track1":
            metadata["ns_deflation"] = (
                "rank-1 head tracked by warm-started power iteration (deflation_power_iters per step, "
                "FP32); entry rest/(||rest||_F+1e-7) + (head_weight/pad) u v^T; head vectors shared "
                "by all-reduce of disjoint owner slices; then the BF16 schedule")
        elif config["deflation"]:
            metadata["ns_deflation"] = (
                "Sun, Kang & Yang (2026) remove-then-restore entry ported from defmuon/optim/"
                "deflation_batched.py@1eb15ce: batched randomized SVD (FP64 CholeskyQR Gram), gate "
                "s[ceil(window*m)]/s1 < threshold, remove pairs above threshold*s1, FP32 entry "
                "rest/(||rest||_F+1e-7) + head/pad, then the BF16 schedule; sketch seeded by step/bank/chunk")
    step, tokens = 0, 0
    if resume:
        saved = torch.load(out / "checkpoint.pt", map_location=device, weights_only=False)
        for key in ("config", "parameter_count", "budget_tokens", "world_size", "source_sha256",
                    "train_manifest", "validation_manifest", "torch_version", "device", "device_name",
                    "effective_precision", "effective_optimizer_impl", "effective_compile", "compilers"):
            if metadata[key] != saved["metadata"][key]:
                raise ValueError(f"Resume mismatch: {key}")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        step, tokens = saved["step"], saved["tokens"]
        metadata = saved["metadata"]
        restore_rng(saved["rank_rng"][rank], device)
        del saved
        if rank == 0:
            orphaned = out / f"uncommitted_{time.time_ns()}"
            for folder in ("steps", "spectra", "spectra_momentum"):
                for path in (out / folder).glob("step*"):
                    if int(path.name.split(".")[0].removeprefix("step")) > step or path.suffix == ".tmp":
                        (orphaned / folder).mkdir(parents=True, exist_ok=True)
                        path.rename(orphaned / folder / path.name)
    elif rank == 0:
        atomic_json(out / "metadata.json", metadata)
        (out / "source").mkdir(exist_ok=True)
        for source in Path(__file__).parent.glob("*.py"):
            (out / "source" / source.name).write_bytes(source.read_bytes())
        for name in ("PROTOCOL.md", "SCALE_UP_CASE.md", "MUON_CASE.md", "MUON_REVIEW.md"):
            (out / "source" / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    # Compile the pure model; DDP remains outside, so no_sync covers forward/backward.
    forward = torch.compile(model, dynamic=False) if compiled else model
    ddp = DDP(forward, device_ids=[device.index] if device.type == "cuda" else None,
              broadcast_buffers=False, gradient_as_bucket_view=True)
    noise_log = None
    if config.get("log_gradient_noise", False) and world > 1:
        # Measurement only (second-order audit, 2026-09-28): each rank's local gradient norm per hidden matrix,
        # taken in the comm hook before the all-reduce. The across-rank variance gives the per-step noise energy
        # of the mean gradient, r = (mean_r |g_r|^2 - |g|^2) / (W - 1); the drift q = |g_t - g_{t-1}|^2 - (r_t + r_{t-1})
        # (independent noise on consecutive batches). Both before clipping. The update is unchanged.
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        tracked = [p for p in model.parameters() if p.ndim == 2 and any(p is q for q in optimizer.param_groups[0]["params"])]
        names = {id(p): n for n, p in model.named_parameters()}
        noise_log = {"params": tracked, "index": {id(p): i for i, p in enumerate(tracked)},
                     "local": torch.zeros(len(tracked), device=device, dtype=torch.float64),
                     "previous": {}, "previous_r": {}, "kinds": [names[id(p)].split(".")[-2] for p in tracked]}

    gate = None
    if config["optimizer"] == "muon" and config.get("data_norm_snr_gate", False):
        if world < 2:
            raise ValueError("data_norm_snr_gate needs at least two data-parallel ranks")
        # SNR-gated tail amplification (second-order audit, 2026-09-28; MUON_CASE decision 22:26 CDT). At every
        # refresh the input second moments are averaged over ranks, each matrix's owner eigendecomposes it and
        # broadcasts (V, u); in the comm hook each rank projects its local gradient on V and keeps the column energies;
        # after the all-reduce the per-column noise of the batch mean is N = (mean_r |g_r V_i|^2 - |g V_i|^2)/(W - 1)
        # and the signal S = |g V_i|^2 - N. S and N are pooled in log-u bins (quarter decades) and EMA-smoothed; the
        # Wiener weight w = SNR / (1 + SNR) of a direction's bin gates PD's tail amplification (muon.snr_gated_root).
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
        body = list(optimizer.param_groups[0]["params"])
        sizes = [p.shape[1] for p in body]
        starts = [sum(sizes[:i]) for i in range(len(body))]
        gate = {"params": body, "offset": {id(p): (a, n) for p, a, n in zip(body, starts, sizes)},
                "local": torch.zeros(sum(sizes), device=device, dtype=torch.float64), "active": False,
                "owners": optimizer.owners(), "modules": optimizer.data_norm["modules"], "bins": {},
                "signal": {p: torch.zeros(24, device=device, dtype=torch.float64) for p in body},
                "noise": {p: torch.zeros(24, device=device, dtype=torch.float64) for p in body},
                "seen": {p: torch.zeros(24, device=device, dtype=torch.bool) for p in body},
                "ema": config["data_norm_snr_ema"], "refresh": config["data_norm_refresh"]}
    if noise_log is not None or gate is not None:
        from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

        def audit_hook(state, bucket):
            for parameter, gradient in zip(bucket.parameters(), bucket.gradients()):
                if noise_log is not None:
                    index = noise_log["index"].get(id(parameter))
                    if index is not None:
                        noise_log["local"][index] = gradient.detach().double().pow(2).sum()
                if gate is not None and gate["active"] and id(parameter) in gate["offset"]:
                    start, n = gate["offset"][id(parameter)]
                    vectors = optimizer.data_norm["basis"][parameter][0]
                    gate["local"][start:start + n] = (gradient.detach().float() @ vectors).double().pow(2).sum(0)
            return default_hooks.allreduce_hook(None, bucket)
        ddp.register_comm_hook(None, audit_hook)
    val_start, val_sequences = partition_sequences(config["validation_tokens"] // sequence, world, rank)
    val_cached = validation.device_tokens(val_start * sequence, val_sequences * sequence + 1, device)
    measured = model.measured_parameters()
    # Eight MLP and sixteen attention matrices: greedy round-robin balances SVD work.
    ordered = sorted(measured.items(), key=lambda kv: (-kv[1].numel() * min(kv[1].shape), kv[0]))
    local_measured = {name: p for index, (name, p) in enumerate(ordered) if index % world == rank}
    if rank == 0 and not resume:
        atomic_json(out / "spectral_rank_assignment.json",
                    {name: index % world for index, (name, _) in enumerate(ordered)})
    if not resume:
        initial = evaluate(model, forward, val_cached, val_sequences, config, device)
        if rank == 0:
            atomic_json(out / "steps/step000000.json", {"step": 0, "tokens": 0, "validation_nll": initial})
    if rank == 0:
        atomic_json(out / "status.json", {"status": "running", "step": step, "tokens": tokens,
                                         "budget_tokens": budget, "world_size": world})
    dist.barrier()
    setup_seconds = time.perf_counter() - invocation_start
    limit = min(stop_after or steps_total, steps_total)
    try:
        while step < limit and tokens < budget:
            step += 1
            lr = learning_rate(config, step, tokens, budget)
            for group in optimizer.param_groups:
                group["lr"] = lr * group.get("lr_scale", 1.0)
            if hasattr(optimizer, "momentum") and config["optimizer"] == "muon":
                optimizer.momentum = muon_momentum(config, tokens, budget)
            batch = min(config["batch_tokens"], budget - tokens)
            global_sequences = batch // sequence
            local_start, local_sequences = partition_sequences(global_sequences, world, rank)
            local_tokens = local_sequences * sequence
            rounds = global_rounds(global_sequences, world, config["microbatch_sequences"])
            optimizer.zero_grad(set_to_none=True)
            synchronize(device)
            begin = time.perf_counter()
            # Empty ranks use an existing sequence only for a zero-weight sync graph.
            cache_start = tokens + local_start * sequence if local_tokens else tokens
            cached = train.device_tokens(cache_start, max(local_tokens, sequence) + 1, device)
            if config["optimizer"] == "muon" and config["data_norm_out_beta"] > 0 and \
                    (step - 1) % config["data_norm_out_refresh"] == 0:
                # Two-sided data norm: output-side statistics from this step's first local sequences.
                from .muon import output_second_moments
                count = max(1, min(config["data_norm_out_sequences"], local_sequences))
                sx, sy = token_views(cached, 0, count * sequence, sequence)
                generator = torch.Generator(device=device).manual_seed(config["seed"] * 1000003 + step * 97 + rank)
                optimizer.update_output_statistics(
                    output_second_moments(model, sx, sy, config["data_norm_out_source"], config, device, generator),
                    config["data_norm_out_ema"])
            total_loss = torch.zeros((), device=device, dtype=torch.float64)
            micro = config["microbatch_sequences"] * sequence
            for iteration in range(rounds):
                offset = iteration * micro
                micro_tokens = min(micro, max(0, local_tokens - offset))
                x, y = token_views(cached, offset if micro_tokens else 0,
                                   micro_tokens or sequence, sequence)
                context = ddp.no_sync() if iteration < rounds - 1 else contextlib.nullcontext()
                with context:
                    with amp_context(device, config):
                        loss = ddp(x, y)
                    (loss * (micro_tokens * world / batch)).backward()
                total_loss += loss.detach().double() * micro_tokens / batch
            noise_row = None
            if noise_log is not None:
                total = noise_log["local"].clone()
                dist.all_reduce(total, op=dist.ReduceOp.SUM)
                noise_log["local"].zero_()
                per = {"r": [], "q": [], "g2": []}
                for index, parameter in enumerate(noise_log["params"]):
                    grad = parameter.grad.detach()
                    g2 = float(grad.double().pow(2).sum())
                    r = max(0.0, (float(total[index]) / world - g2) / (world - 1))
                    previous = noise_log["previous"].get(index)
                    q = (float((grad - previous).double().pow(2).sum()) - r - noise_log["previous_r"][index]
                         if previous is not None else float("nan"))
                    noise_log["previous"][index] = grad.clone()
                    noise_log["previous_r"][index] = r
                    per["r"].append(r); per["q"].append(q); per["g2"].append(g2)
                noise_row = {"r": sum(per["r"]), "g2": sum(per["g2"]),
                             "q": sum(per["q"]) if all(math.isfinite(x) for x in per["q"]) else None,
                             "by_kind": {kind: {key: sum(v for v, k in zip(per[key], noise_log["kinds"]) if k == kind)
                                                for key in ("r", "q", "g2")} for kind in sorted(set(noise_log["kinds"]))}}
            gate_row = None
            if gate is not None:
                if gate["active"]:
                    total = gate["local"].clone()
                    dist.all_reduce(total, op=dist.ReduceOp.SUM)
                    gate["local"].zero_()
                    tail_w, tail_snr = [], []
                    for parameter in gate["params"]:
                        start, n = gate["offset"][id(parameter)]
                        vectors, unit = optimizer.data_norm["basis"][parameter]
                        mean_cols = (parameter.grad.detach().float() @ vectors).double().pow(2).sum(0)
                        noise = ((total[start:start + n] / world - mean_cols) / (world - 1)).clamp_min(0)
                        signal = mean_cols - noise
                        index = gate["bins"][parameter]
                        s_bin = torch.zeros(24, device=device, dtype=torch.float64).index_add_(0, index, signal)
                        n_bin = torch.zeros(24, device=device, dtype=torch.float64).index_add_(0, index, noise)
                        present = torch.zeros(24, device=device, dtype=torch.bool)
                        present[index] = True
                        first = present & ~gate["seen"][parameter]
                        later = present & gate["seen"][parameter]
                        e = gate["ema"]
                        gate["signal"][parameter] = torch.where(first, s_bin, torch.where(later, e * gate["signal"][parameter] + (1 - e) * s_bin, gate["signal"][parameter]))
                        gate["noise"][parameter] = torch.where(first, n_bin, torch.where(later, e * gate["noise"][parameter] + (1 - e) * n_bin, gate["noise"][parameter]))
                        gate["seen"][parameter] |= present
                        snr = gate["signal"][parameter].clamp_min(0) / gate["noise"][parameter].clamp_min(1e-30)
                        w = (snr / (1 + snr))[index]
                        optimizer.data_norm["snr_w"][parameter] = w.float()
                        tail = unit < 1
                        if tail.any():
                            tail_w.append(float(w[tail].mean()))
                            tail_snr.append(float(snr[index][tail].median()))
                    gate_row = {"tail_mean_w": sum(tail_w) / max(len(tail_w), 1),
                                "tail_median_snr": sorted(tail_snr)[len(tail_snr) // 2] if tail_snr else None}
                if (step - 1) % gate["refresh"] == 0:
                    # Rank-averaged input second moments, owner eigendecomposition, broadcast of (V, u).
                    for module in {id(m): m for m in gate["modules"].values()}.values():
                        dist.all_reduce(module.input_cov, op=dist.ReduceOp.SUM)
                        module.input_cov.div_(world)
                    for parameter in gate["params"]:
                        module = gate["modules"][parameter]
                        owner = gate["owners"][parameter]
                        n = parameter.shape[1]
                        vectors = torch.empty(n, n, device=device, dtype=torch.float32)
                        unit = torch.empty(n, device=device, dtype=torch.float32)
                        if rank == owner:
                            cov = (module.input_cov / module.input_cov_weight.clamp_min(1e-12)).double()
                            values, v64 = torch.linalg.eigh(0.5 * (cov + cov.T))
                            values = values.clamp_min(0)
                            vectors.copy_(v64.float())
                            unit.copy_((values / values.mean().clamp_min(1e-30)).float())
                        dist.broadcast(vectors, src=owner)
                        dist.broadcast(unit, src=owner)
                        optimizer.data_norm["basis"][parameter] = (vectors, unit)
                        gate["bins"][parameter] = ((unit.double().clamp_min(1e-12).log10() + 4) / 0.25).floor().clamp(0, 23).long()
                    gate["active"] = True
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"],
                                                       error_if_nonfinite=True, foreach=True)
            sample = config["spectra_every"] > 0 and (
                step == 1 or step % config["spectra_every"] == 0 or tokens + batch == budget)
            decay = {}
            if sample and local_measured:
                norms = torch.stack([p.detach().norm() for p in local_measured.values()]).cpu().tolist()
                decay = {name: lr * config["weight_decay"] * norm for name, norm in zip(local_measured, norms)}
            if config["optimizer"] == "muon":
                optimizer.capture_parameters = set(local_measured.values()) if sample else set()
                optimizer.audit_distribution = audit and (step == 1 or step == limit)
            optimizer.step()
            if audit and (step == 1 or step == limit):
                audit_replicas(model, optimizer, out, step, rank, world)
            synchronize(device)
            training_seconds = scalar_reduce(time.perf_counter() - begin, device, dist.ReduceOp.MAX)
            nll = scalar_reduce(total_loss, device)
            tokens += batch
            if not math.isfinite(nll):
                raise RuntimeError("Non-finite global training loss")
            row = {"step": step, "tokens": tokens, "batch_tokens": batch, "lr": lr,
                   "train_nll": nll, "gradient_norm_before_clip": float(grad_norm.item()),
                   "gradient_clipped": bool(grad_norm.item() > config["grad_clip"]),
                   "training_seconds": training_seconds, "world_size": world,
                   "dummy_sequences": sum(max(0, rounds - math.ceil(partition_sequences(global_sequences, world, r)[1] /
                                                config["microbatch_sequences"])) for r in range(world))}
            if noise_row is not None:
                row["gradient_noise"] = noise_row
            if gate_row is not None:
                row["snr_gate"] = gate_row
            if config["optimizer"] == "muon":
                row["aux_lr"] = optimizer.param_groups[1]["lr"]
                if optimizer.deflation is not None:
                    counts = optimizer.deflation_counts.clone()
                    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                    optimizer.deflation_counts.zero_()
                    matrices, fired, pairs = counts.tolist()
                    row["deflation"] = {"matrices": matrices, "fired": fired, "deflated_pairs": pairs}
                    if optimizer.deflation_mode == "track1":
                        energy = optimizer.deflation_energy.clone()
                        dist.all_reduce(energy, op=dist.ReduceOp.SUM)
                        optimizer.deflation_energy.zero_()
                        row["deflation"]["mean_head_energy"] = float(energy) / max(matrices, 1)
                        autocos = optimizer.deflation_autocos.clone()
                        dist.all_reduce(autocos, op=dist.ReduceOp.SUM)
                        optimizer.deflation_autocos.zero_()
                        row["deflation"]["mean_v_autocos"], row["deflation"]["mean_u_autocos"] = (
                            x / max(matrices, 1) for x in autocos.tolist())
                if optimizer.data_norm is not None:
                    sums = optimizer.data_norm_sums.clone()
                    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                    optimizer.data_norm_sums.zero_()
                    count = max(sums[0].item(), 1)
                    row["data_norm"] = {"matrices": sums[0].item(), "mean_direction_factor": sums[1].item() / count,
                                        "pre_rescale_norm_ratio": sums[2].item() / count}
                if optimizer.whitening is not None:
                    sums = optimizer.whitening_sums.clone()
                    dist.all_reduce(sums, op=dist.ReduceOp.SUM)
                    optimizer.whitening_sums.zero_()
                    row["whitening"] = {"matrices": sums[0].item(), "mean_beta": sums[1].item() / max(sums[0].item(), 1)}
                    if optimizer.bias_adam is not None:
                        bias = optimizer.bias_sums.clone()
                        dist.all_reduce(bias, op=dist.ReduceOp.SUM)
                        optimizer.bias_sums.zero_()
                        row["whitening"]["bias_grad_share"] = bias[1].item() / max(bias[0].item(), 1)
                        row["whitening"]["bias_step_mean_abs"] = bias[2].item() / max(bias[0].item(), 1)
            begin = time.perf_counter()
            for quantity in (("update", "momentum") if config["optimizer"] == "muon" else ("update",)) if sample else ():
                matrix_key = "matrices" if quantity == "update" else "momentum_matrices"
                spectrum_folder = "spectra" if quantity == "update" else "spectra_momentum"
                local_rows, local_values = measure_updates(optimizer, local_measured,
                    device=config["svd_device"], decay_norms=decay, quantity=quantity) if local_measured else ({}, {})
                gathered = [None] * world if rank == 0 else None
                dist.gather_object((local_rows, local_values), gathered, dst=0)
                if rank == 0:
                    row[matrix_key], values = {}, {}
                    for matrix_rows, matrix_values in gathered:
                        if set(row[matrix_key]) & set(matrix_rows):
                            raise RuntimeError("Duplicate distributed spectral matrix")
                        row[matrix_key].update(matrix_rows)
                        values.update(matrix_values)
                    if set(values) != set(measured):
                        raise RuntimeError("Incomplete distributed spectral panel")
                    for name, array in values.items():
                        info = row[matrix_key][name]
                        if (len(array) != min(measured[name].shape) or not np.isfinite(array).all()
                                or np.any(array < 0) or np.any(np.diff(array) > 0)):
                            raise RuntimeError(f"Invalid distributed spectrum: {name}")
                        if not info["zero_matrix"] and abs(info["normalized_energy_sum"] - 1) > 2e-4:
                            raise RuntimeError(f"Spectrum normalization check failed: {name}")
                    path = out / spectrum_folder / f"step{step:06d}.npz"
                    temp = path.with_suffix(".tmp")
                    with temp.open("wb") as handle:
                        np.savez_compressed(handle, **values)
                    os.replace(temp, path)
            row["measurement_seconds"] = scalar_reduce(time.perf_counter() - begin, device, dist.ReduceOp.MAX)
            begin = time.perf_counter()
            if step % config["validation_every"] == 0 or tokens == budget or step == limit:
                row["validation_nll"] = evaluate(model, forward, val_cached, val_sequences, config, device)
            row["validation_seconds"] = scalar_reduce(time.perf_counter() - begin, device, dist.ReduceOp.MAX)
            if rank == 0:
                atomic_json(out / "steps" / f"step{step:06d}.json", row)
                if step == 1 or step % 10 == 0 or step == limit:
                    print(json.dumps({k: v for k, v in row.items() if k not in ("matrices", "momentum_matrices")}), flush=True)
            if step % config["checkpoint_every"] == 0 or step == limit:
                checkpoint(out / "checkpoint.pt", model, optimizer, step, tokens,
                           config, metadata, device, rank, world)
            # Measurement only: kept checkpoints and the next step's weights (applied update = difference).
            if step in config["keep_checkpoints"]:
                checkpoint(out / "kept" / f"step{step:06d}.pt", model, optimizer, step, tokens,
                           config, metadata, device, rank, world)
                if rank == 0:
                    # Rank 0's input statistics (non-persistent buffers, so not in the state dict).
                    stats = {name: {key: buffer.detach().cpu().clone() for key, buffer in module.named_buffers()}
                             for name, module in model.named_modules() if isinstance(module, StatLinear)}
                    if stats:
                        torch.save(stats, out / "kept" / f"step{step:06d}_input_stats_rank0.pt")
                dist.barrier()
            if step - 1 in config["keep_checkpoints"]:
                if rank == 0:
                    torch.save({"model": model.state_dict(), "step": step, "tokens": tokens},
                               out / "kept" / f"step{step:06d}_weights.pt")
                dist.barrier()
        elapsed = scalar_reduce(time.perf_counter() - invocation_start, device, dist.ReduceOp.MAX)
        peak = scalar_reduce(torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
                             device, dist.ReduceOp.MAX)
        summary = {"status": "complete" if tokens == budget else "stopped_at_requested_step",
                   "step": step, "tokens": tokens, "budget_tokens": budget, "world_size": world,
                   "setup_seconds": setup_seconds, "invocation_seconds": elapsed,
                   "peak_cuda_allocated_bytes_per_rank_max": int(peak)}
        if rank == 0:
            atomic_json(out / "status.json", summary)
        dist.barrier()
        return summary
    except Exception as exc:
        # Every failing rank leaves evidence; torchrun terminates peers on failure.
        atomic_json(out / f"failure_rank{rank}.json", {"step": step, "tokens": tokens,
                    "error": f"{type(exc).__name__}: {exc}"})
        if rank == 0:
            atomic_json(out / "status.json", {"status": "failed", "step": step, "tokens": tokens,
                        "error": f"{type(exc).__name__}: {exc}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stop-after", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--audit-replicas", action="store_true", help="Hash all replicas at first/last update for qualification")
    args = parser.parse_args()
    run(load_config(args.config), args.out, device=args.device, stop_after=args.stop_after,
        resume=args.resume, audit=args.audit_replicas)


if __name__ == "__main__":
    main()

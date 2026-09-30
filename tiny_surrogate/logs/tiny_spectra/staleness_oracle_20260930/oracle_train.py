"""Staleness-oracle training (PROTOCOL.md, "Staleness oracle: how much does momentum staleness cost ...").

The training loop is research/tiny_spectra/train.py's, unchanged except for the body momentum that the
optimizer uses at each step (config key "oracle"):

  none       the ordinary momentum (harness control)
  full       sum_{s=t-K+1..t} beta^(t-s) kappa_s g_{B_s}(W_{t-1}): the same batches and clip factors,
             re-evaluated at the current weights
  stiff      the ordinary momentum, with the full oracle's correction kept only inside the top-k
             Gauss-Newton subspace (fixed fresh probe, refreshed every stiff_refresh steps)
  rest       the same correction kept only outside that subspace
  transport  M + G Q: the first-order (Gauss-Newton on the probe) transport of each gradient from the
             weights it was taken at to the current weights, Q_t = beta (Q_{t-1} + S_{t-1} D_t)

The override sets each body momentum buffer to (H - g_t) / beta before optimizer.step(), so the
optimizer's own update produces exactly H. The auxiliary AdamW, PD's input statistics (collected only
for the current batch inside the step window) and the schedule are untouched. Extra gradients use
torch.autograd.grad and never touch parameter .grad or the statistics.
"""
import argparse
from collections import deque
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import torch
from torch.func import functional_call, jvp, vjp
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn import functional as F

from research.tiny_spectra.model import GPT, ModelConfig
from research.tiny_spectra.optim import make_optimizer
from research.tiny_spectra.train import (amp, atomic_json, evaluate, model_hash, save_model_snapshot,
                                         schedule_factor, tensor_hash)

ORACLES = ("none", "full", "stiff", "rest", "rest_matched", "transport", "split", "split1")


def side_effect_state(model):
    """Buffers (incl. statistics), parameter gradients and RNG states, for the replay no-side-effect check."""
    buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    grads = {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}
    rng = [torch.get_rng_state().clone()]
    if torch.cuda.is_available():
        rng.append(torch.cuda.get_rng_state().clone())
    return buffers, grads, rng


def same_state(a, b):
    return (all(torch.equal(a[0][n], b[0][n]) for n in a[0]) and a[0].keys() == b[0].keys()
            and all(torch.equal(a[1][n], b[1][n]) for n in a[1]) and a[1].keys() == b[1].keys()
            and all(torch.equal(x, y) for x, y in zip(a[2], b[2])))


class Body:
    """Flat views of the 48 body matrices."""

    def __init__(self, model):
        self.names = list(model.body_parameters())
        parameters = dict(model.named_parameters())
        self.params = [parameters[n] for n in self.names]
        self.shapes = [p.shape for p in self.params]
        self.sizes = [p.numel() for p in self.params]
        self.dim = sum(self.sizes)

    def flat(self, tensors):
        return torch.cat([t.reshape(-1).float() for t in tensors])

    def split(self, vector):
        out, offset = [], 0
        for shape, size in zip(self.shapes, self.sizes):
            out.append(vector[offset:offset + size].view(shape))
            offset += size
        return out

    def weights(self):
        return self.flat([p.detach() for p in self.params])

    def grads(self):
        return self.flat([p.grad for p in self.params])


def body_gradient(model, body, corpus, positions, cfg, device, micro):
    """Mean-loss body gradient on the given windows at the current weights (training precision)."""
    accumulated = [torch.zeros_like(p, dtype=torch.float32) for p in body.params]
    total = len(positions)
    for chunk in positions.split(micro):
        x, y = corpus.batch("train", len(chunk), cfg["seq_len"], positions=chunk, device=device)
        with amp(device, cfg["precision"]):
            loss = model(x, y)
        grads = torch.autograd.grad(loss * (len(chunk) / total), body.params)
        for a, g in zip(accumulated, grads):
            a.add_(g.float())
    return body.flat(accumulated)


class GaussNewton:
    """Exact GN-vector products of the mean cross entropy on a fixed probe over the body matrices (FP32)."""

    def __init__(self, model, body, corpus, positions, cfg, device, chunk=8):
        params = {n: p.detach() for n, p in model.named_parameters()}
        self.model, self.body_ = model, body
        self.body = {n: params[n] for n in body.names}
        self.others = {n: v for n, v in params.items() if n not in self.body}
        self.inputs = [corpus.batch("train", len(c), cfg["seq_len"], positions=c, device=device)[0]
                       for c in positions.split(chunk)]
        self.count = sum(x.numel() for x in self.inputs)

    def __call__(self, vector):
        v = dict(zip(self.body_.names, self.body_.split(vector)))
        out = torch.zeros_like(vector)
        with sdpa_kernel(SDPBackend.MATH):
            for x in self.inputs:
                f = lambda b: functional_call(self.model, {**self.others, **b}, (x,))
                logits, jv = jvp(f, (self.body,), (v,))
                p = torch.softmax(logits.float(), -1)
                hjv = p * jv - p * (p * jv).sum(-1, keepdim=True)
                _, pullback = vjp(f, self.body)
                (g,) = pullback(hjv)
                out += self.body_.flat([g[n] for n in self.body_.names])
        return out / self.count


def lanczos_top(operator, dim, iterations, k, device, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(dim, generator=generator).to(device)
    q /= q.norm()
    basis, alphas, betas = [q], [], []
    for j in range(iterations):
        w = operator(basis[-1])
        alpha = torch.dot(w, basis[-1])
        w = w - alpha * basis[-1] - (betas[-1] * basis[-2] if betas else 0)
        stack = torch.stack(basis)
        w = w - stack.T @ (stack @ w)
        w = w - stack.T @ (stack @ w)
        beta = w.norm()
        alphas.append(alpha)
        if j == iterations - 1 or beta < 1e-10:
            break
        betas.append(beta)
        basis.append(w / beta)
    m = len(alphas)
    t = torch.diag(torch.stack(alphas))
    if m > 1:
        off = torch.stack(betas[:m - 1])
        t = t + torch.diag(off, 1) + torch.diag(off, -1)
    values, vectors = torch.linalg.eigh(t.double())
    order = torch.argsort(values, descending=True)[:k]
    ritz = (torch.stack(basis[:m]).T @ vectors[:, order].float()).T.contiguous()
    return values[order].float().cpu(), ritz / ritz.norm(dim=1, keepdim=True)


def run(cfg, out):
    oracle = cfg.get("oracle", "none")
    if oracle not in ORACLES:
        raise ValueError(f"oracle must be one of {ORACLES}")
    K = int(cfg.get("oracle_K", 40))
    micro = int(cfg.get("oracle_micro", 32))
    keep_model_every = cfg.get("keep_model_every", 0)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    atomic_json(out / "config.json", cfg)
    atomic_json(out / "status.json", dict(status="initializing", started_unix=started))
    device = torch.device(cfg.get("device", "cuda"))
    torch.set_num_threads(cfg.get("cpu_threads", 2))
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        if properties.total_memory >= 45 * 1024**3:
            raise RuntimeError("This study is restricted to GPUs strictly below nominal 48 GB VRAM")
        hardware = dict(name=properties.name, total_memory_bytes=properties.total_memory,
                        capability=list(torch.cuda.get_device_capability(device)))
        torch.cuda.reset_peak_memory_stats(device)
    else:
        hardware = dict(name="CPU qualification", total_memory_bytes=None)
    torch.manual_seed(cfg["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cfg.get("corpus_kind") != "fineweb_byte_bpe_v1":
        raise ValueError("The oracle trainer supports the Candidate-D FineWeb corpus only")
    from research.tiny_spectra.fineweb import load_training_corpus
    corpus = load_training_corpus(cfg["data_path"], expected_manifest_sha256=cfg["data_sha256"])
    has_cov = cfg["method"] in ("pd", "ts", "spd", "sts")
    mc = ModelConfig(vocab_size=corpus.vocab_size, n_layer=cfg["n_layer"], n_embd=cfg["n_embd"],
                     n_head=cfg["n_head"], seq_len=cfg["seq_len"],
                     track_input_stats=has_cov, track_input_cov=has_cov,
                     cov_stride=cfg["cov_stride"], cov_decay=cfg["cov_ema"],
                     stats_decay=cfg.get("stats_ema", .99), stats_clock=cfg.get("stats_clock", "step"))
    model = GPT(mc).to(device)
    optimizer = make_optimizer(model, cfg, device)
    if cfg["method"] in ("ts", "sts", "soap", "spd", "adamw"):
        raise ValueError("The oracle override is implemented for plain Muon-family body momentum (muon, pd)")
    body = Body(model)
    beta = float(cfg["momentum"])
    total_sequences, remainder = divmod(cfg["total_tokens"], cfg["seq_len"])
    batch_sequences, batch_remainder = divmod(cfg["batch_tokens"], cfg["seq_len"])
    if remainder or batch_remainder or batch_sequences < 1:
        raise ValueError("Budget and effective batch must be positive multiples of context")
    total_steps = math.ceil(total_sequences / batch_sequences)
    generator = torch.Generator().manual_seed(cfg["seed"] + 1729)
    starts = corpus.window_positions("train", total_sequences, cfg["seq_len"], generator=generator)
    # Never-trained blocks beyond the horizon for the curvature probe (the same default stream).
    stream = corpus.window_positions("train", total_sequences + 4096, cfg["seq_len"],
                                     generator=torch.Generator().manual_seed(cfg["seed"] + 1729))
    if not torch.equal(stream[:total_sequences], starts):
        raise RuntimeError("Probe stream does not continue the training stream")
    probe = stream[total_sequences:total_sequences + int(cfg.get("probe_sequences", 64))]
    validation_starts = corpus.evaluation_starts("val", cfg["seq_len"])
    count = min(len(validation_starts), cfg["validation_tokens"] // cfg["seq_len"])
    bank_indices = torch.linspace(0, len(validation_starts) - 1, count).round().long()
    validation_bank = validation_starts[bank_indices]
    train_bank = starts[:min(128, count)].clone()
    parameters = dict(model.named_parameters())
    metadata = dict(model_config=asdict(mc), n_parameters=model.num_parameters(),
                    initial_parameter_sha256=model_hash(model), train_window_sha256=tensor_hash(starts),
                    validation_bank_sha256=tensor_hash(validation_bank), probe_sha256=tensor_hash(probe),
                    corpus=corpus.manifest, hardware=hardware, host=socket.gethostname(),
                    slurm_job_id=os.environ.get("SLURM_JOB_ID"), torch_version=torch.__version__,
                    source_file=str(Path(__file__).resolve()), total_steps=total_steps, oracle=oracle,
                    oracle_K=K, started_unix=started)
    atomic_json(out / "metadata.json", metadata)
    torch.save(dict(train=starts, validation=validation_bank, train_probe=train_bank, curvature_probe=probe),
               out / "windows.pt")
    metrics = (out / "metrics.jsonl").open("x", buffering=1)
    records = []
    initial_val, _ = evaluate(model, corpus, "val", validation_bank, cfg, device)
    records.append(dict(step=0, tokens=0, validation_nll=initial_val, elapsed_seconds=time.time() - started))
    metrics.write(json.dumps(records[-1]) + "\n")
    if keep_model_every:
        save_model_snapshot(model, mc, out / "models/step000000.pt", step=0, tokens=0)
    history = deque(maxlen=max(0, K - 1))        # (step, positions, kappa) of earlier batches
    M_normal = torch.zeros(body.dim, device=device)
    Q = torch.zeros(body.dim, device=device)
    S_previous = 0.0
    last_delta = None
    U = None
    probe_n = int(cfg.get("probe_sequences", 64))
    probe_B = stream[total_sequences + 2048:total_sequences + 2048 + probe_n]   # disjoint diagnostic probe
    refresh = int(cfg.get("stiff_refresh", 2))
    lanczos_its = int(cfg.get("stiff_lanczos", 64))
    k = int(cfg.get("stiff_k", 16))
    diag_every = int(cfg.get("diagnostic_basis_every", 10))
    # Split momentum: a second, shorter EMA used inside or outside a subspace ("gn": top-k GN directions;
    # "band": each matrix's above-mean input eigenvectors from PD's statistics), long beta elsewhere.
    split_basis = cfg.get("split_basis", "gn")
    split_short = cfg.get("split_short", "rest")
    beta_short = float(cfg.get("beta_short", 0.5))
    M_short = torch.zeros(body.dim, device=device)
    bands = None
    # One-buffer split ("split1"): per input eigendirection i of each matrix (PD's statistics), heavy-ball decay
    # beta_i and input gain e_i = (1 - beta_i) / (1 - beta), so that for a fixed basis M1 equals the two-buffer
    # split exactly: M1 <- M1 V diag(beta_i) V^T + g V diag(e_i) V^T. Rule "band": beta_i = beta on the above-mean
    # eigenvalues, beta_short elsewhere.
    M1 = [torch.zeros_like(p, dtype=torch.float32) for p in body.params]
    decays = None
    if oracle == "split" and (split_basis not in ("gn", "band") or split_short not in ("rest", "stiff")):
        raise ValueError("split_basis must be gn or band and split_short rest or stiff")
    science_start = time.perf_counter()
    training_seconds = 0.0
    for step in range(1, total_steps + 1):
        factor = schedule_factor(step, total_steps, cfg["warmup_fraction"], cfg["cooldown_fraction"])
        for group in optimizer.param_groups:
            group["lr"] = cfg["lr"] * factor * group.get("lr_scale", 1.0)
        begin = (step - 1) * batch_sequences
        batch_starts = starts[begin:begin + batch_sequences]
        optimizer.zero_grad(set_to_none=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tick = time.perf_counter()
        model.begin_step_stats()
        accumulated_loss = torch.zeros((), device=device)
        for chunk in batch_starts.split(cfg["microbatch_sequences"]):
            x, y = corpus.batch("train", len(chunk), cfg["seq_len"], positions=chunk, device=device)
            weight = len(chunk) / len(batch_starts)
            with amp(device, cfg["precision"]):
                loss = model(x, y)
            (loss * weight).backward()
            accumulated_loss += loss.detach().float() * weight
        model.finish_step_stats(ema=cfg["cov_ema"] if mc.stats_clock == "step" else None)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"], error_if_nonfinite=True)
        kappa = min(1.0, cfg["grad_clip"] / (float(norm) + 1e-6))
        g_t = body.grads()                         # clipped current-batch body gradient
        M_normal.mul_(beta).add_(g_t)
        M_short.mul_(beta_short).add_(g_t)
        diagnostics = {}
        H, M_full = None, None
        oracle_tick = time.perf_counter()
        check_side_effects = oracle != "none" and step in (2, 3)
        if check_side_effects:
            before_state = side_effect_state(model)
        needs_basis = ((oracle in ("stiff", "rest", "rest_matched") or (oracle == "split" and split_basis == "gn"))
                       and step > 1 and (U is None or (step - 2) % refresh == 0))
        diagnostic_step = step % diag_every == 0
        if needs_basis or (diagnostic_step and oracle in ("none", "full", "transport")):
            values, U = lanczos_top(GaussNewton(model, body, corpus, probe, cfg, device), body.dim, lanczos_its, k,
                                    device, seed=step)
            diagnostics["stiff_eigenvalues"] = values[:4].tolist()
            if diagnostic_step:
                gn = GaussNewton(model, body, corpus, probe, cfg, device)
                diagnostics["ritz_k_relative_residual"] = float((gn(U[-1]) - values[-1] * U[-1]).norm() / values[-1])
                _, U_B = lanczos_top(GaussNewton(model, body, corpus, probe_B, cfg, device), body.dim, lanczos_its, k,
                                     device, seed=step + 7)
                diagnostics["basis_overlap_disjoint_probe"] = float((U @ U_B.T).square().sum() / k)
        if oracle in ("full", "stiff", "rest", "rest_matched") and step > 1:
            M_full = g_t.clone()
            for s, positions, kappa_s in history:
                M_full.add_(body_gradient(model, body, corpus, positions, cfg, device, micro),
                            alpha=beta ** (step - s) * kappa_s)
            if oracle == "full":
                H = M_full
            elif oracle in ("stiff", "rest"):
                correction = M_full - M_normal
                inside = U.T @ (U @ correction)
                H = M_normal + inside if oracle == "stiff" else M_full - inside
                diagnostics["correction_inside_share"] = float(inside.square().sum() / correction.square().sum())
            else:
                # Outside the top-k GN subspace: the stale-free direction at the ordinary momentum's norm.
                inside_normal = U.T @ (U @ M_normal)
                rest_normal = M_normal - inside_normal
                rest_full = M_full - U.T @ (U @ M_full)
                H = inside_normal + rest_full * (rest_normal.norm() / rest_full.norm())
                diagnostics["rest_cos_full_normal"] = float(F.cosine_similarity(rest_full, rest_normal, dim=0))
                diagnostics["rest_norm_full_over_normal"] = float(rest_full.norm() / rest_normal.norm())
            diagnostics["cos_full_normal"] = float(F.cosine_similarity(M_full, M_normal, dim=0))
            diagnostics["norm_full_over_normal"] = float(M_full.norm() / M_normal.norm())
        elif oracle == "split" and step > 1:
            if split_basis == "gn":
                project = lambda v: U.T @ (U @ v)
            else:
                if bands is None or (step - 2) % int(cfg["root_refresh"]) == 0:
                    bands = []
                    for param in body.params:
                        module = optimizer.data_norm["modules"][param]
                        cov = (module.input_cov / module.input_cov_weight.clamp_min(1e-12)).double()
                        values, vectors = torch.linalg.eigh(cov)
                        bands.append(vectors[:, values > values.mean()].float())
                project = lambda v: body.flat([m @ b @ b.T for m, b in zip(body.split(v), bands)])
            ratio = (1 - beta_short) / (1 - beta)          # match the two heavy-ball scales
            long_in, short_in = project(M_normal), project(M_short)
            if split_short == "rest":
                H = long_in + ratio * (M_short - short_in)
            else:
                H = ratio * short_in + (M_normal - long_in)
            diagnostics["split_inside_share_long"] = float(long_in.square().sum() / M_normal.square().sum())
        elif oracle == "split1":
            if decays is None or (step - 2) % int(cfg["root_refresh"]) == 0:
                decays = []
                for param in body.params:
                    module = optimizer.data_norm["modules"][param]
                    cov = (module.input_cov / module.input_cov_weight.clamp_min(1e-12)).double()
                    values, vectors = torch.linalg.eigh(0.5 * (cov + cov.T))
                    unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
                    rule = cfg.get("split_rule", "band")
                    if rule == "band":
                        betas = torch.where(unit >= 1, torch.full_like(unit, beta), torch.full_like(unit, beta_short))
                    else:
                        raise ValueError("unknown split_rule")
                    gains = (1 - betas) / (1 - beta)
                    V = vectors.float()
                    decays.append(((V * betas.float()) @ V.T, (V * gains.float()) @ V.T))
                    long_count = int((betas == beta).sum()) if rule == "band" else 0
                    diagnostics.setdefault("split1_long_dims", 0)
                    diagnostics["split1_long_dims"] += long_count
            parts = []
            for i, (m, g) in enumerate(zip(M1, body.split(g_t))):
                D, E = decays[i]
                M1[i] = m @ D + g @ E
                parts.append(M1[i])
            if step > 1:
                H = body.flat(parts)
        elif oracle == "transport" and step > 1:
            Q = beta * (Q + S_previous * last_delta)
            gn = GaussNewton(model, body, corpus, probe, cfg, device)
            correction = gn(Q)
            H = M_normal + correction
            diagnostics["transport_over_momentum"] = float(correction.norm() / M_normal.norm())
        S_previous = beta * S_previous + kappa
        if U is not None and (needs_basis or diagnostic_step or H is not None):
            share = lambda v: float((U @ v).square().sum() / v.square().sum())
            diagnostics["stiff_share_momentum"] = share(M_normal)
            diagnostics["stiff_share_gradient"] = share(g_t)
            if M_full is not None:
                diagnostics["stiff_share_stale_free"] = share(M_full)
            if H is not None:
                diagnostics["stiff_share_used"] = share(H)
        if check_side_effects:
            unchanged = same_state(before_state, side_effect_state(model))
            diagnostics["replay_side_effect_free"] = unchanged
            if not unchanged:
                raise RuntimeError("The replay/basis block changed statistics, gradients or RNG state")
        if H is not None:
            diagnostics["cos_used_normal"] = float(F.cosine_similarity(H, M_normal, dim=0))
            diagnostics["used_over_normal_norm"] = float(H.norm() / M_normal.norm())
            diagnostics["cos_used_gradient"] = float(F.cosine_similarity(H, g_t, dim=0))
            for p, h, g in zip(body.params, body.split(H), body.split(g_t)):
                optimizer.state[p]["momentum_buffer"].copy_((h - g) / beta)
        diagnostics["cos_momentum_gradient"] = float(F.cosine_similarity(M_normal, g_t, dim=0))
        diagnostics["oracle_seconds"] = time.perf_counter() - oracle_tick
        history.append((step, batch_starts, kappa))
        report = step == 1 or step % cfg["eval_every"] == 0 or step == total_steps
        before_body = body.weights().clone()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        last_delta = body.weights() - before_body
        if U is not None and (needs_basis or diagnostic_step or H is not None):
            energy = float((U @ last_delta).square().sum() / last_delta.square().sum())
            diagnostics["update_stiff_energy"] = energy
            diagnostics["update_stiff_over_chance"] = energy / (k / body.dim)
        if H is not None and step in (2, 3, total_steps // 2, total_steps):
            buffers = body.flat([optimizer.state[p]["momentum_buffer"] for p in body.params])
            diagnostics["override_match"] = float((buffers - H).norm() / H.norm())
        if oracle == "none" and step in (2, total_steps // 2, total_steps):
            buffers = body.flat([optimizer.state[p]["momentum_buffer"] for p in body.params])
            diagnostics["harness_buffer_match"] = float((buffers - M_normal).norm() / max(M_normal.norm(), 1e-30))
        elapsed_step = time.perf_counter() - tick
        training_seconds += elapsed_step
        record = dict(step=step, tokens=min(begin + len(batch_starts), total_sequences) * cfg["seq_len"],
                      train_nll=float(accumulated_loss), lr=cfg["lr"] * factor,
                      gradient_norm_before_clip=float(norm), step_seconds=elapsed_step,
                      elapsed_seconds=time.perf_counter() - science_start, body_step_norm=float(last_delta.norm()),
                      **diagnostics)
        if not math.isfinite(record["train_nll"]):
            raise FloatingPointError("Nonfinite training loss")
        if report:
            record["validation_nll"], _ = evaluate(model, corpus, "val", validation_bank, cfg, device)
            print(json.dumps(record), flush=True)
        records.append(record)
        metrics.write(json.dumps(record) + "\n")
        if keep_model_every and (step % keep_model_every == 0 or step == total_steps):
            save_model_snapshot(model, mc, out / "models" / f"step{step:06d}.pt", step=step, tokens=record["tokens"])
        atomic_json(out / "status.json", dict(status="running", step=step, total_steps=total_steps,
                                             updated_unix=time.time()))
    if cfg.get("smoke"):
        final_full = records[-1]["validation_nll"]
    else:
        final_full, per_sequence = evaluate(model, corpus, "val", validation_starts, cfg, device)
        torch.save(dict(starts=validation_starts, sequence_nll=per_sequence), out / "final_validation.pt")
    torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()}, config=cfg,
                    metadata=metadata), out / "final.pt")
    summary = dict(status="complete", final_validation_nll=records[-1]["validation_nll"],
                   full_validation_nll=final_full, steps=total_steps, tokens=cfg["total_tokens"],
                   training_seconds=training_seconds, total_seconds=time.time() - started,
                   n_parameters=model.num_parameters(), hardware=hardware, oracle=oracle, oracle_K=K,
                   initial_parameter_sha256=metadata["initial_parameter_sha256"],
                   train_window_sha256=metadata["train_window_sha256"])
    atomic_json(out / "summary.json", summary)
    atomic_json(out / "status.json", dict(**summary, finished_unix=time.time()))
    metrics.close()
    print(json.dumps(dict(event="complete", **summary)), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, help="smoke only: truncate the run")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.steps:
        cfg = dict(cfg, total_tokens=args.steps * cfg["batch_tokens"], run_id=cfg["run_id"] + "_smoke")
    if Path(args.out).exists():
        raise FileExistsError(args.out)
    try:
        run(cfg, args.out)
    except Exception as error:
        path = Path(args.out)
        if path.exists():
            atomic_json(path / "failure.json", dict(error=repr(error), traceback=traceback.format_exc(),
                                                   failed_unix=time.time()))
            atomic_json(path / "status.json", dict(status="failed", error=repr(error), failed_unix=time.time()))
        raise


if __name__ == "__main__":
    main()

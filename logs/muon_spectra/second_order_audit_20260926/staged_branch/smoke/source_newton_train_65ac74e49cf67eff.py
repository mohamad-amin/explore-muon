"""One damped Gauss-Newton step per batch (Hessian-free style), as the reference for what a second-order direction buys
at a fixed batch size (after the 12:33 CDT finding that the GN paper's inner-loop recipe mostly buys "more, smaller
steps per batch").

Outer step s (the run's own batch and data order, sequences split over ranks as in distributed.py):
  - g: the batch gradient at theta_s (BF16 autocast, mean token loss), all-reduced;
  - x(d) = -(G + d rho I)^-1 g_body for the 48 body matrices from one k-step Lanczos run on the exact GN matrix
    (forward-mode + reverse-mode products, explicit attention) on the first `--curvature-sequences` of each rank's
    batch, all-reduced, for every d in DAMPINGS (rho = g.G g / g.g);
  - joint line search: theta_body += a x(d) over d in DAMPINGS and a in LINE times the run's schedule factor (warmup,
    constant, linear cooldown to 0, as the baselines' LR), picked by the true loss on held-out training sequences
    (tokens beyond every run's budget). No damping heuristic: the held-out loss chooses damping and step together,
    as the cross-fitted one-step analyses do;
  - with --normalize d instead (GN's shape in the normalized regime, 15:21 CDT decision): no line search; x(d)'s part for
    each body matrix is rescaled to the baseline map's step norm times the run's LR, lr_ref(s) |NS(M)| sqrt(max(1, m/n))
    for Muon runs and lr_ref(s) sqrt(min(m, n)) sqrt(max(1, m/n)) for data-norm runs, so only the direction differs;
  - decoupled weight decay on the body matched to the baseline's per-step shrink (lr_ref(s) * wd);
  - embeddings, gains and head: AdamW with the checkpoint's own state and the run's aux schedule (as in training).
Starts from a kept checkpoint. Validation as in distributed.py, every --validation-every steps.
Added 2026-09-28 (GN-PD as a rate at 16M, 09:40 CDT decision), all off by default: --kron-control (GN-PD's Kronecker
limit, TS p/p with the step's own B and C), --log-kron (per-matrix cosine of the GN-PD step with that map), --clip
(the trainer's global gradient clipping), --sharpness-every (top GN eigenvalue from a fixed random start), --keep
(kept checkpoints) and --transport (the momentum carried to the current weights). Each run stores a copy of this
source and its sha256.

usage: .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 newton_train.py OUT_DIR ARM_DIR:STEP [--cg 16 (Lanczos steps)] [--curvature-sequences 32]
       [--stop-after N]
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.func import functional_call, jvp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream, token_views  # noqa: E402
from research.adamw_spectra.distributed import partition_sequences  # noqa: E402
from research.adamw_spectra.muon import newton_schulz, ns_schedule  # noqa: E402
from research.adamw_spectra.train import learning_rate, token_budget  # noqa: E402

LINE = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
DAMPINGS = (1e-1, 3e-2, 1e-2, 1e-3)     # relative to rho = g.G g / g.g, the curvature along g
HELD_OUT_TRAIN = 2_500_000_000


def bf16():
    return torch.autocast("cuda", dtype=torch.bfloat16)


class GNProduct:
    """v -> G v for the body matrices on this rank's curvature sequences, summed over ranks (mean token loss)."""

    def __init__(self, model, body_keys, xs, total_tokens, micro, fp32=False):
        self.model, self.keys, self.xs, self.total, self.micro = model, body_keys, xs, total_tokens, micro
        self.precision = contextlib.nullcontext if fp32 else bf16     # FP32 products for amplifying maps (pre-flight)

    def __call__(self, v_parts):
        base = {**dict(self.model.named_parameters()), **dict(self.model.named_buffers())}
        weights = [base[k] for k in self.keys]
        out = [torch.zeros_like(w) for w in weights]
        for i in range(0, self.xs.shape[0], self.micro):
            x = self.xs[i:i + self.micro]

            def logits_of(*ws):
                return functional_call(self.model, {**base, **dict(zip(self.keys, ws))}, (x,))
            with torch.no_grad(), P.explicit_attention(), self.precision():
                logits, dz = jvp(logits_of, tuple(w.detach() for w in weights), tuple(v.to(w.dtype) for v, w in zip(v_parts, weights)))
            p = torch.softmax(logits.float(), dim=-1)
            dz = dz.float()
            hdz = (p * (dz - (p * dz).sum(-1, keepdim=True))) / self.total
            del logits, dz, p
            with self.precision():
                z = self.model(x)
            grads = torch.autograd.grad((z.float() * hdz).sum(), weights)
            for acc, g in zip(out, grads):
                acc += g.float()
        flat = torch.cat([o.reshape(-1) for o in out])
        dist.all_reduce(flat)
        return [part.view_as(o) for part, o in zip(flat.split([o.numel() for o in out]), out)]


def dot(a, b):
    return sum(float((x * y).sum()) for x, y in zip(a, b))


@torch.no_grad()
def summed_loss(model, x_all, y_all, micro):
    total = torch.zeros((), device=x_all.device, dtype=torch.float64)
    for i in range(0, x_all.shape[0], micro):
        with bf16():
            total += model(x_all[i:i + micro], y_all[i:i + micro]).double() * x_all[i:i + micro].numel()
    dist.all_reduce(total)
    return float(total)


def kron_roots(model, keys, xs, ys, micro, power, damping, generator):
    """Per body matrix (L, R) = ((B / mean eig + damping)^-power, (C / mean eig + damping)^-power) from this step's
    curvature sequences, summed over ranks: B the sampled-label output factor, C the input second moment over
    positions >= 1 (the trainer's TS and PD statistics, instantaneous rather than EMA)."""
    names = list(P.hidden_linears(model))
    by_key = dict(zip(P.parameter_keys(model, names), names))
    recorder = P.Recorder(model, [by_key[k] for k in keys])
    T = xs.shape[1]
    sums = {k: [0, 0] for k in keys}
    for i in range(0, xs.shape[0], micro):
        P.gradient_passes(model, recorder, xs[i:i + micro], ys[i:i + micro], generator, draws=1)
        for k in keys:
            x = recorder.inputs[by_key[k]][:, 1:].float()
            e = T * recorder.errors[by_key[k]][1].float()
            x, e = x.reshape(-1, x.shape[-1]), e.reshape(-1, e.shape[-1])
            sums[k][0] = sums[k][0] + e.T @ e
            sums[k][1] = sums[k][1] + x.T @ x
        recorder.clear()
    recorder.remove()
    roots = {}
    for k in keys:
        pair = []
        for moment in sums[k]:
            dist.all_reduce(moment)
            values, vectors = torch.linalg.eigh(0.5 * (moment + moment.T).double())
            unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
            pair.append(((vectors * (unit + damping).pow(-power)) @ vectors.T).float())
        roots[k] = tuple(pair)
    return roots


@torch.no_grad()
def pd_roots(model, keys, xs, micro, alpha, damping):
    """PD's input root R = (C / mean eig + damping)^-alpha per body matrix from this step's curvature sequences (C over
    positions >= 1, summed over ranks): the trainer's PD statistic, instantaneous rather than EMA (2026-09-29)."""
    names = list(P.hidden_linears(model))
    by_key = dict(zip(P.parameter_keys(model, names), names))
    recorder = P.Recorder(model, [by_key[k] for k in keys])
    sums = {k: 0 for k in keys}
    for i in range(0, xs.shape[0], micro):
        with bf16():
            model(xs[i:i + micro])
        for k in keys:
            x = recorder.inputs[by_key[k]][:, 1:].float()
            x = x.reshape(-1, x.shape[-1])
            sums[k] = sums[k] + x.T @ x
        recorder.clear()
    recorder.remove()
    roots = {}
    for k in keys:
        moment = sums[k]
        dist.all_reduce(moment)
        values, vectors = torch.linalg.eigh(0.5 * (moment + moment.T).double())
        unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
        roots[k] = ((vectors * (unit + damping).pow(-alpha)) @ vectors.T).float()
    return roots


def kron_direction(roots, keys, parts):
    """-L polar(L M R) R per matrix (polar by SVD): GN-PD's limit when G = B (x) C."""
    out = []
    for k, m in zip(keys, parts):
        left, right = roots[k]
        u, _, vh = torch.linalg.svd(left @ m.float() @ right, full_matrices=False)
        out.append(-(left @ (u @ vh) @ right).reshape(-1))
    return torch.cat(out)


def top_eigenvalue(op, n, k, device, seed=0):
    """Largest GN eigenvalue (top Ritz value of a k-step Lanczos run, full reorthogonalization) from a fixed random
    start that is identical on every rank, so successive steps and arms are measured the same way."""
    start = torch.randn(n, generator=torch.Generator().manual_seed(seed)).to(device)
    V = torch.zeros(k, n, device=device)
    V[0] = start / start.norm()
    alphas, betas = [], []
    for j in range(k):
        w = op(V[j])
        alphas.append(float(w @ V[j]))
        for _ in range(2):
            w -= V[:j + 1].T @ (V[:j + 1] @ w)
        if j + 1 == k:
            break
        betas.append(float(w.norm()))
        V[j + 1] = w / betas[-1]
    Tk = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    off = torch.tensor(betas, dtype=torch.float64)
    return float(torch.linalg.eigvalsh(Tk + torch.diag(off, 1) + torch.diag(off, -1))[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--cg", type=int, default=16)
    parser.add_argument("--curvature-sequences", type=int, default=32, help="per rank")
    parser.add_argument("--damping", type=float, default=1e-3, help="initial lambda relative to the curvature along g")
    parser.add_argument("--warm", type=float, default=0.95)
    parser.add_argument("--micro", type=int, default=16)
    parser.add_argument("--line-sequences", type=int, default=32, help="held-out training sequences per rank")
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--overshoot", type=float, default=1.0,
                        help="take kappa times the line-search step (the successful optimizers step ~2x their own optimum)")
    parser.add_argument("--normalize", type=float, default=None, metavar="DAMPING",
                        help="GN's shape in the normalized regime: the damped GN direction at this relative damping, each "
                             "body matrix's step rescaled to the baseline map's norm (Muon: |NS(M)| sqrt(max(1, m/n)); "
                             "data-norm runs: sqrt(min(m, n)) sqrt(max(1, m/n))) times the run's LR; no line search")
    parser.add_argument("--polar-control", action="store_true",
                        help="with --normalize: Muon's own direction NS(M) in place of GN's (no Lanczos), the harness "
                             "control that should reproduce the Muon baseline")
    parser.add_argument("--polar-direction", action="store_true",
                        help="the step-rule control (review, 16:40 CDT): the joint held-out line search runs on Muon's own "
                             "step, -lr_peak sqrt(max(1, m/n)) NS(M) per matrix, in place of GN's direction ('greedy Muon')")
    parser.add_argument("--gnpd", type=float, default=None, metavar="POWER",
                        help="with --normalize d: PD in the exact GN geometry (gnpd_probe.py, 19:07 CDT), "
                             "-(G + d rho)^-p polar((G + d rho)^-p M) per matrix, two Lanczos runs per step, then the "
                             "normalized step (the run's LR and per-matrix norms)")
    parser.add_argument("--lr-scale", type=float, default=1.0,
                        help="with --normalize: body step = lr-scale x the run's LR (the new direction's own LR; weight "
                             "decay keeps the run's lr x wd)")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="EMA of batch gradients as the Newton right-hand side (0: the batch gradient); initialized "
                             "from the checkpoint's momentum buffer as (1 - beta_run) M")
    parser.add_argument("--kron-control", action="store_true",
                        help="with --normalize d --gnpd p: GN-PD's Kronecker limit L polar(L M R) R (TS p/p) with L, R the "
                             "-p roots of the step's own B and C (damping d, as the trainer's data norm); no Lanczos")
    parser.add_argument("--log-kron", action="store_true",
                        help="with --gnpd: also compute the --kron-control map each step and log each matrix's cosine "
                             "with the step taken")
    parser.add_argument("--clip", type=float, default=None,
                        help="clip the batch gradient's global norm (all parameters) as the trainer does, before the "
                             "momentum average and the aux AdamW (default: no clipping, as in the earlier runs)")
    parser.add_argument("--sharpness-every", type=int, default=0,
                        help="every N steps, the top GN eigenvalue on the curvature sequences (Lanczos 20 from a fixed "
                             "random start), measured before the step")
    parser.add_argument("--keep", default="",
                        help="comma-separated steps whose checkpoint is also kept as kept/stepNNNNNN.pt (post-hoc probes)")
    parser.add_argument("--staged-pd", type=float, default=None, metavar="ALPHA",
                        help="with --normalize (any value) and --momentum b: PD's map (NS polar, R from the step's curvature "
                             "sequences, damping 1e-3) on the momentum average, applied layer by layer; each layer sees the "
                             "average plus clip-factor x G (earlier layers' actual steps) (2026-09-29, MUON_CASE 04:21 CDT)")
    parser.add_argument("--stage-off", action="store_true",
                        help="with --staged-pd: the same PD map with no cross-layer correction (the harness control)")
    parser.add_argument("--transport", action="store_true",
                        help="with --normalize and --momentum b: the direction is computed from the momentum carried to "
                             "the current weights, A + (1 - b) G D with D <- b D + b/(1 - b) dW (floor_steps.py, 02:29 CDT), "
                             "one extra GN product per step; the stored average stays the plain EMA")
    args = parser.parse_args()
    keep = {int(v) for v in args.keep.split(",") if v}
    if args.transport and (args.normalize is None or args.momentum <= 0):
        parser.error("--transport needs --normalize d and --momentum b > 0")
    if args.kron_control and (args.gnpd is None or args.normalize is None or args.polar_control or args.polar_direction):
        parser.error("--kron-control needs --normalize d and --gnpd p, and no polar modes")
    if args.staged_pd is not None and (args.normalize is None or args.momentum <= 0 or args.gnpd is not None
                                       or args.kron_control or args.polar_control or args.polar_direction or args.transport):
        parser.error("--staged-pd needs --normalize and --momentum b, and no other direction mode")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arm, start = args.item.rsplit(":", 1)
    start = int(start)
    kept = Path(arm) / "scientific" / "kept"
    model, saved = P.load_checkpoint(kept / f"step{start:06d}.pt", device)
    config = dict(saved["config"])
    T = model.config.seq_len
    named = dict(model.named_parameters())
    body_keys = [k for k in named if k.startswith("blocks.") and named[k].ndim == 2]
    aux_keys = [k for k in named if k not in body_keys]
    body = [named[k] for k in body_keys]
    aux = [named[k] for k in aux_keys]
    # AdamW for the aux parameters with the checkpoint's own moments (MuonAdamW keeps body first, aux second)
    aux_opt = torch.optim.AdamW(aux, lr=config["aux_learning_rate"], betas=tuple(config["betas"]), eps=config["epsilon"],
                                weight_decay=config["weight_decay"], fused=True)
    saved_state = saved["optimizer"]["state"]
    aux_indices = saved["optimizer"]["param_groups"][1]["params"]
    aux_opt.load_state_dict({"state": {j: saved_state[i] for j, i in enumerate(aux_indices) if i in saved_state},
                             "param_groups": [{**aux_opt.state_dict()["param_groups"][0], "params": list(range(len(aux)))}]})
    average = None
    if args.momentum > 0:
        body_indices = saved["optimizer"]["param_groups"][0]["params"]
        average = torch.cat([(1 - config["muon_momentum"]) * saved_state[i]["momentum_buffer"].float().reshape(-1)
                             for i in body_indices]).to(device)
    displacement = None     # --transport: D = sum_k b^k (theta_s - theta_{s-k}), zero at the start (no history)
    del saved
    budget = token_budget(config, sum(p.numel() for p in named.values()), T)
    steps_total = math.ceil(budget / config["batch_tokens"])
    stop = min(steps_total, start + args.stop_after) if args.stop_after else steps_total
    train = TokenStream(str(REPO / config["train_pattern"]))
    validation = TokenStream(str(REPO / config["validation_pattern"]))
    val_start, val_sequences = partition_sequences(config["validation_tokens"] // T, world, rank)
    val_x, val_y = token_views(validation.device_tokens(val_start * T, val_sequences * T + 1, device), 0, val_sequences * T, T)
    line_x, line_y = token_views(train.device_tokens(HELD_OUT_TRAIN + rank * args.line_sequences * T,
                                                     args.line_sequences * T + 1, device), 0, args.line_sequences * T, T)
    line_count = world * args.line_sequences * T
    peak = config["learning_rate"]
    schedule_ns = ns_schedule(config)
    if rank == 0:
        (args.out / "steps").mkdir(parents=True, exist_ok=True)
        source = Path(__file__).resolve().read_bytes()
        source_name = f"source_newton_train_{hashlib.sha256(source).hexdigest()[:16]}.py"
        (args.out / source_name).write_bytes(source)
        (args.out / "run.json").write_text(json.dumps({"item": args.item, "args": {k: str(v) for k, v in vars(args).items()},
                                                       "world": world, "stop": stop, "config": config,
                                                       "source": source_name,
                                                       "source_sha256": hashlib.sha256(source).hexdigest()}, indent=1) + "\n")
    first = start + 1
    resume = args.out / "checkpoint.pt"
    if resume.exists():     # model, aux AdamW state, CG warm start and damping from the last checkpoint
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        aux_opt.load_state_dict(state["aux"])
        first = state["step"] + 1
        if average is not None and state.get("average") is not None:
            average = state["average"].to(device)
        if state.get("displacement") is not None:
            displacement = state["displacement"].to(device)
        del state
    started = time.time()
    if first == start + 1:
        value = summed_loss(model, val_x, val_y, args.micro) / config["validation_tokens"]
        if rank == 0:
            print(json.dumps({"step": start, "validation_nll": value}), flush=True)
    for s in range(first, stop + 1):
        t0 = time.time()
        tokens_before = (s - 1) * config["batch_tokens"]
        lr_ref = learning_rate(config, s, tokens_before, budget)
        schedule = lr_ref / peak
        local_start, local_sequences = partition_sequences(config["batch_tokens"] // T, world, rank)
        x_all, y_all = token_views(train.device_tokens(tokens_before + local_start * T, local_sequences * T + 1, device),
                                   0, local_sequences * T, T)
        # batch gradient (all parameters), mean over the global batch's tokens
        for p in named.values():
            p.grad = None
        train_total = torch.zeros((), device=device, dtype=torch.float64)
        for i in range(0, local_sequences, args.micro):
            with bf16():
                loss = model(x_all[i:i + args.micro], y_all[i:i + args.micro])
            (loss * (x_all[i:i + args.micro].numel() * world / config["batch_tokens"])).backward()
            train_total += loss.detach().double() * x_all[i:i + args.micro].numel()
        flat = torch.cat([p.grad.reshape(-1) for p in named.values()])
        dist.all_reduce(flat)
        flat /= world
        grad_norm = float(flat.float().norm())
        if args.clip is not None and grad_norm > args.clip:     # the trainer's global clipping, before any state
            flat *= args.clip / grad_norm
        dist.all_reduce(train_total)
        for p, part in zip(named.values(), flat.split([p.numel() for p in named.values()])):
            p.grad = part.view_as(p).clone()
        g = torch.cat([p.grad.float().reshape(-1) for p in body])
        if average is not None:     # Newton on the averaged gradient
            average.mul_(args.momentum).add_(g, alpha=1 - args.momentum)
            g = average.clone()
        sizes = [w.numel() for w in body]
        curv = x_all[:args.curvature_sequences]
        G = GNProduct(model, body_keys, curv, curv.numel() * world, args.micro)

        def op(v):
            return torch.cat([o.reshape(-1) for o in G([part.view_as(w) for part, w in zip(v.split(sizes), body)])])
        extra_pre = {"grad_norm_before_clip": grad_norm}
        if args.transport:          # the momentum carried to the current weights (first order in the GN)
            if displacement is None:
                displacement = torch.zeros_like(g)
            elif float(displacement.norm()) > 0:
                gd = op(displacement)
                extra_pre["transport_ratio"] = float((1 - args.momentum) * gd.norm() / g.norm().clamp_min(1e-30))
                g = g + (1 - args.momentum) * gd
                del gd
        if args.sharpness_every and s % args.sharpness_every == 0:
            extra_pre["gn_top"] = top_eigenvalue(op, sum(sizes), 20, device)
        kron = None
        if args.kron_control or args.log_kron:
            t_kron = time.time()
            roots = kron_roots(model, body_keys, curv, y_all[:args.curvature_sequences], args.micro, args.gnpd,
                               args.normalize, torch.Generator(device=device).manual_seed(1000 * s + rank))
            kron = kron_direction(roots, body_keys, [gi.view_as(w) for gi, w in zip(g.split(sizes), body)])
            del roots
            extra_pre["kron_seconds"] = time.time() - t_kron
        # Lanczos from g (full reorthogonalization): one Krylov basis gives the damped solution for every damping
        solutions, rho, ritz_top = {args.normalize: -g}, float("nan"), float("nan")     # the polar control's placeholder
        stage_log = {}
        if args.staged_pd is not None:      # layer-staged PD (or its control with --stage-off)
            t_stage = time.time()
            roots = pd_roots(model, body_keys, curv, args.micro, args.staged_pd, 1e-3)
            kappa = min(1.0, args.clip / grad_norm) if args.clip is not None else 1.0
            residual = g.clone()
            offsets = [0]
            for n_el in sizes:
                offsets.append(offsets[-1] + n_el)
            blocks = []
            for idx, key in enumerate(body_keys):
                block = key.split(".")[1]
                if not blocks or blocks[-1][0] != block:
                    blocks.append((block, []))
                blocks[-1][1].append(idx)
            direction = torch.zeros_like(g)
            plain_cos = []
            for b_i, (_, members) in enumerate(blocks):
                block_step = torch.zeros_like(g)
                for idx in members:
                    w = body[idx]
                    rows, cols = w.shape
                    r_part = residual[offsets[idx]:offsets[idx + 1]].view_as(w)
                    R = roots[body_keys[idx]]
                    d = newton_schulz((r_part.float() @ R)[None], schedule_ns)[0].float() @ R
                    target = math.sqrt(min(rows, cols)) * math.sqrt(max(1.0, rows / cols))
                    step_part = -(args.lr_scale * lr_ref * target / float(d.norm().clamp_min(1e-30))) * d
                    direction[offsets[idx]:offsets[idx + 1]] = -d.reshape(-1)
                    block_step[offsets[idx]:offsets[idx + 1]] = step_part.reshape(-1)
                    if not args.stage_off and b_i > 0:      # how far staging turned this matrix's direction
                        g_part = g[offsets[idx]:offsets[idx + 1]].view_as(w)
                        d0 = newton_schulz((g_part.float() @ R)[None], schedule_ns)[0].float() @ R
                        plain_cos.append(float((d * d0).sum() / (d.norm() * d0.norm()).clamp_min(1e-30)))
                if not args.stage_off and b_i + 1 < len(blocks):
                    residual += kappa * op(block_step)
            solutions = {args.normalize: direction}
            stage_log = {"stage_seconds": time.time() - t_stage, "kappa": kappa,
                         "staged_cos_plain_mean": sum(plain_cos) / len(plain_cos) if plain_cos else 1.0,
                         "staged_cos_plain_min": min(plain_cos) if plain_cos else 1.0}
            del roots, residual
        elif args.kron_control:       # GN-PD's Kronecker limit (TS p/p), no Lanczos
            solutions = {args.normalize: kron}
        elif args.polar_direction:    # Muon's step at the peak LR; the line search scales it by LINE x schedule
            parts = []
            for w, gi in zip(body, g.split(sizes)):
                rows, cols = w.shape
                polar = newton_schulz(gi.view_as(w)[None], schedule_ns)[0].float().reshape(-1)
                parts.append(-peak * math.sqrt(max(1.0, rows / cols)) * polar)
            solutions = {0.0: torch.cat(parts)}
        elif not args.polar_control:
            def lanczos_run(start, k):
                V = torch.zeros(k, start.numel(), device=device)
                V[0] = start / start.norm()
                alphas_l, betas_l = [], []
                for j in range(k):
                    w = op(V[j])
                    alphas_l.append(float(w @ V[j]))
                    for _ in range(2):
                        w -= V[:j + 1].T @ (V[:j + 1] @ w)
                    if j + 1 == k:
                        break
                    beta = float(w.norm())
                    betas_l.append(beta)
                    V[j + 1] = w / beta
                Tk = torch.diag(torch.tensor(alphas_l, dtype=torch.float64))
                off = torch.tensor(betas_l, dtype=torch.float64)
                return V, Tk + torch.diag(off, 1) + torch.diag(off, -1)

            def matrix_power(V, Tk, start_norm, power, mu):     # (G + mu)^-power applied to the Lanczos start vector
                values, vectors = torch.linalg.eigh(Tk)
                y = vectors @ ((values.clamp_min(0) + mu).pow(-power) * vectors[0] * start_norm)
                return V.T @ y.to(device, torch.float32)
            k = args.cg
            V, Tk = lanczos_run(g, k)
            rho = float(Tk[0, 0])
            ritz_top = float(torch.linalg.eigvalsh(Tk)[-1])
            rhs = torch.zeros(k, dtype=torch.float64)
            rhs[0] = float(g.norm())
            solutions = {}
            if args.gnpd is not None:   # PD in the exact GN geometry: whiten, polar per matrix, whiten again
                mu = args.normalize * rho
                white = matrix_power(V, Tk, float(g.norm()), args.gnpd, mu)
                del V
                parts = []
                for w, wi in zip(body, white.split(sizes)):
                    u, _, vh = torch.linalg.svd(wi.view_as(w).float(), full_matrices=False)
                    parts.append((u @ vh).reshape(-1))
                polar_flat = torch.cat(parts)
                V2, T2 = lanczos_run(polar_flat, k)
                solutions[args.normalize] = -matrix_power(V2, T2, float(polar_flat.norm()), args.gnpd, mu)
                del V2
            else:
                for d in (DAMPINGS if args.normalize is None else (args.normalize,)):
                    y = torch.linalg.solve(Tk + d * rho * torch.eye(k, dtype=torch.float64), rhs)
                    solutions[d] = -(V.T @ y.to(device, torch.float32))
                del V
        # aux parameters: AdamW on the batch gradient, the run's aux schedule
        for group in aux_opt.param_groups:
            group["lr"] = config["aux_learning_rate"] * schedule
        aux_opt.step()
        if args.normalize is not None:
            # GN's shape at the baseline's per-matrix step norms, so only the direction differs from the baseline
            with torch.no_grad():
                base_line = summed_loss(model, line_x, line_y, args.micro)
                before = torch.cat([w.detach().reshape(-1) for w in body]) if args.transport else None
                x = solutions[args.normalize]
                predicted, squares, cosines, kron_cos = 0.0, 0.0, [], []
                kron_parts = kron.split(sizes) if kron is not None else [None] * len(body)
                for w, xi, gi, ki in zip(body, x.split(sizes), g.split(sizes), kron_parts):
                    rows, cols = w.shape
                    scale = math.sqrt(max(1.0, rows / cols))
                    polar = newton_schulz(gi.view_as(w)[None], schedule_ns)[0].float().reshape(-1)
                    if args.polar_control:
                        xi = -polar
                    cosines.append(-float(xi @ polar) / max(float(xi.norm() * polar.norm()), 1e-30))     # vs Muon's step
                    if ki is not None:      # vs the Kronecker limit of the same map
                        kron_cos.append(float(xi @ ki) / max(float(xi.norm() * ki.norm()), 1e-30))
                    if config.get("data_norm_alpha", 0) > 0:    # data-norm maps are rescaled to sqrt(min(m, n))
                        target = math.sqrt(min(rows, cols)) * scale
                    else:                                       # Muon: the norm of its own Newton-Schulz output
                        target = float(polar.norm()) * scale
                    factor = args.lr_scale * lr_ref * target / float(xi.norm().clamp_min(1e-30))
                    w.mul_(1 - lr_ref * config["weight_decay"])
                    w.add_(xi.view_as(w), alpha=factor)
                    predicted += factor * float(gi @ xi)
                    squares += (args.lr_scale * lr_ref * target) ** 2
                if before is not None:      # D <- b D + b/(1 - b) dW (weight decay included in dW)
                    b = args.momentum
                    displacement.mul_(b).add_(torch.cat([w.detach().reshape(-1) for w in body]) - before, alpha=b / (1 - b))
                    del before
                best_loss = summed_loss(model, line_x, line_y, args.micro)
            d_best, alpha, step_norm = args.normalize, lr_ref, math.sqrt(squares)
            extra = {"cos_muon_mean": sum(cosines) / len(cosines), "cos_muon_min": min(cosines), **stage_log}
            if kron_cos:
                extra.update({"cos_kron_mean": sum(kron_cos) / len(kron_cos), "cos_kron_min": min(kron_cos),
                              "cos_kron": dict(zip(body_keys, kron_cos))})
        else:
            # joint line search over (damping, step) for the body on held-out training sequences
            with torch.no_grad():
                originals = [w.detach().clone() for w in body]
                base_line = summed_loss(model, line_x, line_y, args.micro)
                trials = {}
                for d, xs in solutions.items():
                    parts = xs.split(sizes)
                    for a in LINE:
                        a_s = a * schedule
                        for w, w0, xi in zip(body, originals, parts):
                            w.copy_(w0 + a_s * xi.view_as(w0))
                        trials[(d, a_s)] = summed_loss(model, line_x, line_y, args.micro)
                (d_best, alpha), best_loss = min(trials.items(), key=lambda kv: kv[1])
                if best_loss >= base_line:      # no trial improves the held-out loss: skip the body step
                    alpha, best_loss = 0.0, base_line
                alpha *= args.overshoot
                parts = solutions[d_best].split(sizes)
                for w, w0, xi in zip(body, originals, parts):
                    w.copy_(w0 + alpha * xi.view_as(w0))
                    w.mul_(1 - lr_ref * config["weight_decay"])
            step_norm = alpha * float(solutions[d_best].norm())
            predicted = alpha * float(g @ solutions[d_best])    # first-order part of the predicted change
            extra = {}
        actual = (best_loss - base_line) / line_count
        row = {"step": s, "alpha": alpha, "damping": d_best, "rho": rho, "ritz_top": ritz_top,
               "first_order_change": predicted, "held_out_change": actual,
               "line_base": base_line / line_count, "line_best": best_loss / line_count,
               "train_nll": float(train_total) / config["batch_tokens"],
               "step_norm": step_norm, **extra_pre, **extra, "training_seconds": time.time() - t0}
        if s % args.validation_every == 0 or s == stop:
            row["validation_nll"] = summed_loss(model, val_x, val_y, args.micro) / config["validation_tokens"]
        if rank == 0 and (s % args.checkpoint_every == 0 or s == stop or s in keep):
            tmp = args.out / "checkpoint.pt.tmp"
            torch.save({"model": model.state_dict(), "aux": aux_opt.state_dict(), "step": s, "config": config,
                        "average": average.cpu() if average is not None else None,
                        "displacement": displacement.cpu() if displacement is not None else None}, tmp)
            if s in keep:
                (args.out / "kept").mkdir(exist_ok=True)
                torch.save(torch.load(tmp, map_location="cpu", weights_only=False), args.out / "kept" / f"step{s:06d}.pt")
            tmp.replace(resume)
        if rank == 0:
            (args.out / "steps" / f"step{s:06d}.json").write_text(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k != "cos_kron"}), flush=True)
    if rank == 0:
        (args.out / "done.json").write_text(json.dumps({"seconds": time.time() - started, "last_step": stop}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

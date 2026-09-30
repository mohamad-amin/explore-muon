"""Gauss-Newton training in the GN paper's recipe (Abreu et al., ICLR 2026, Algorithm 1 with Muon as inner optimizer),
as a measured reference for the gap to second order in this project's setup (review, 11:24 CDT).

Outer step s (global batch of the run's own training tokens, same data order and sequence split as distributed.py):
  - linearization point theta_s: the model;
  - inner iterate theta~ (a second model), warm-started at the previous outer step's pre-line-search solution;
  - N inner steps, one per sub-batch of the batch: the gradient of the batch's GN model at theta~,
        J^T [grad_z l(z_0) + H_z J (theta~ - theta_s)]   (z_0 = logits at theta_s; one JVP with explicit attention,
                                                          one forward with autograd, one backward),
    averaged over ranks, drives MuonAdamW on theta~ (Muon on the 48 body matrices, AdamW on the rest), no weight decay;
  - line search: theta_{s+1} = theta_s + a (theta~ - theta_s), a in LINE (a <= 1), minimizing the true loss on fixed
    held-out training sequences (tokens beyond the run's budget);
  - decoupled weight decay matched to the baseline's per-step shrink: body *= 1 - lr_ref(s) wd,
    aux *= 1 - aux_lr_ref(s) wd, with lr_ref the run's own schedule.
The inner LR follows the run's schedule shape (warmup, constant, linear cooldown) scaled to --inner-lr.
Starts from a kept checkpoint's model (inner optimizer fresh). Validation as in distributed.py (the run's validation
tokens, BF16), every --validation-every outer steps and at the end.

With --no-linearize (control) the inner steps use true gradients at the inner iterate: small-batch Muon on the same
sub-batches with the same outer line search, which separates curvature feedback from "more, smaller steps per batch".

usage: .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 gn_train.py OUT_DIR ARM_DIR:STEP [--inner-lr 0.007] [--inner-sequences 16]
       [--stop-after N] [--precision bf16|fp32]
"""
import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict
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
from research.adamw_spectra.muon import MuonAdamW  # noqa: E402
from research.adamw_spectra.train import learning_rate, token_budget  # noqa: E402

LINE = (1.0, 2 ** -0.5, 0.5, 2 ** -1.5, 0.25)
HELD_OUT_TRAIN = 2_500_000_000     # training-stream tokens never used by any run (budget 1.54B)


def autocast(precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else torch.autocast("cuda", enabled=False)


def gn_model_gradient(model, keys, params, shadow_params, x, y, total_tokens, precision):
    """This microbatch's share of the gradient of the batch's GN model at theta~ (summed over its tokens, divided by
    the global token count), plus its share of the model's value terms (l_0, first-order, second-order)."""
    tangent = tuple((s.detach() - p.detach()).to(p.dtype) for s, p in zip(shadow_params, params))
    base = {**dict(model.named_parameters()), **dict(model.named_buffers())}

    def logits_of(*weights):
        return functional_call(model, {**base, **dict(zip(keys, weights))}, (x,))
    with torch.no_grad(), P.explicit_attention(), autocast(precision):
        _, jd = jvp(logits_of, tuple(p.detach() for p in params), tangent)
    with autocast(precision):
        logits = model(x)
    z = logits.float()
    p = torch.softmax(z.detach(), dim=-1)
    jd = jd.float()
    grad_z = p.clone()
    grad_z.scatter_add_(-1, y.unsqueeze(-1), -torch.ones(y.shape + (1,), device=z.device))
    hjd = p * (jd - (p * jd).sum(-1, keepdim=True))
    u = (grad_z + hjd) / total_tokens
    grads = torch.autograd.grad((z * u).sum(), params)
    with torch.no_grad():
        l0 = float(torch.nn.functional.cross_entropy(z.detach().flatten(0, 1), y.flatten(), reduction="sum")) / total_tokens
        first = float((grad_z * jd).sum()) / total_tokens
        second = 0.5 * float((jd * hjd).sum()) / total_tokens
    return grads, (l0, first, second)


def true_gradient(shadow, shadow_params, x, y, total_tokens, precision):
    """Control (--no-linearize): the true gradient at the inner iterate on this microbatch (no linearization, no
    curvature feedback), so the inner loop is plain small-batch Muon between the outer line searches."""
    with autocast(precision):
        z = shadow(x).float()
    loss = torch.nn.functional.cross_entropy(z.flatten(0, 1), y.flatten(), reduction="sum") / total_tokens
    grads = torch.autograd.grad(loss, shadow_params)
    return grads, (float(loss), 0.0, 0.0)


@torch.no_grad()
def mean_loss(model, x_all, y_all, micro, precision):
    total = torch.zeros((), device=x_all.device, dtype=torch.float64)
    for i in range(0, x_all.shape[0], micro):
        with autocast(precision):
            total += model(x_all[i:i + micro], y_all[i:i + micro]).double() * x_all[i:i + micro].numel()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--inner-lr", type=float, default=0.007)
    parser.add_argument("--inner-aux-lr", type=float, default=0.002)
    parser.add_argument("--inner-momentum", type=float, default=0.95)
    parser.add_argument("--inner-sequences", type=int, default=16, help="sequences per rank per inner step")
    parser.add_argument("--micro", type=int, default=16)
    parser.add_argument("--line-sequences", type=int, default=32, help="held-out training sequences per rank")
    parser.add_argument("--no-line-search", dest="line_search", action="store_false")
    parser.add_argument("--no-warm-start", dest="warm_start", action="store_false")
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--precision", default="bf16", choices=("bf16", "fp32"))
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--no-linearize", dest="linearize", action="store_false",
                        help="control: inner steps on true gradients at the inner iterate (small-batch Muon + line search)")
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arm, start = args.item.rsplit(":", 1)
    start = int(start)
    kept = Path(arm) / "scientific" / "kept"
    model, saved = P.load_checkpoint(kept / f"step{start:06d}.pt", device)
    config = dict(saved["config"])
    del saved
    shadow, _ = P.load_checkpoint(kept / f"step{start:06d}.pt", device)
    T = model.config.seq_len
    keys = [k for k, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    shadow_params = [p for _, p in shadow.named_parameters()]
    body = {k for k in keys if k.startswith("blocks.") and dict(model.named_parameters())[k].ndim == 2}
    inner_config = {**config, "optimizer": "muon", "learning_rate": args.inner_lr, "aux_learning_rate": args.inner_aux_lr,
                    "muon_momentum": args.inner_momentum, "weight_decay": 0.0, "muon_nesterov": False,
                    "data_norm_alpha": 0.0, "data_norm_out_beta": 0.0, "soap_precondition": False, "deflation": False,
                    "mean_whitening": False, "mean_bias_adam": False, "muon_track_displacement": False}
    inner = MuonAdamW(shadow, inner_config, device, "fused")
    budget = token_budget(config, sum(p.numel() for p in params), T)
    steps_total = math.ceil(budget / config["batch_tokens"])
    stop = min(steps_total, start + args.stop_after) if args.stop_after else steps_total
    train = TokenStream(str(REPO / config["train_pattern"]))
    validation = TokenStream(str(REPO / config["validation_pattern"]))
    val_start, val_sequences = partition_sequences(config["validation_tokens"] // T, world, rank)
    val_tokens = validation.device_tokens(val_start * T, val_sequences * T + 1, device)
    val_x, val_y = token_views(val_tokens, 0, val_sequences * T, T)
    line_tokens = train.device_tokens(HELD_OUT_TRAIN + rank * args.line_sequences * T, args.line_sequences * T + 1, device)
    line_x, line_y = token_views(line_tokens, 0, args.line_sequences * T, T)
    peak = config["learning_rate"]
    if rank == 0:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "steps").mkdir(exist_ok=True)
        (args.out / "run.json").write_text(json.dumps({"item": args.item, "args": {k: str(v) for k, v in vars(args).items()},
                                                       "world": world, "steps_total": steps_total, "stop": stop,
                                                       "config": config}, indent=1) + "\n")

    def validate():
        loss = mean_loss(model, val_x, val_y, args.micro, "bf16")     # token-summed over this rank's share
        dist.all_reduce(loss)
        return float(loss) / config["validation_tokens"]

    resume = args.out / "checkpoint.pt"
    first = start + 1
    if resume.exists():     # model, inner iterate and inner optimizer state from the last checkpoint
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        shadow.load_state_dict(state["shadow"])
        inner.load_state_dict(state["inner"])
        first = state["step"] + 1
        del state
    started = time.time()
    for s in range(first, stop + 1):
        t0 = time.time()
        tokens_before = (s - 1) * config["batch_tokens"]
        lr_ref = learning_rate(config, s, tokens_before, budget)
        schedule = lr_ref / peak
        for group in inner.param_groups:      # body: inner LR; aux: lr_scale = inner aux LR / inner LR
            group["lr"] = args.inner_lr * schedule * group.get("lr_scale", 1.0)
        if not args.warm_start:
            with torch.no_grad():
                for sp, p in zip(shadow_params, params):
                    sp.copy_(p)
        global_sequences = config["batch_tokens"] // T
        local_start, local_sequences = partition_sequences(global_sequences, world, rank)
        tokens = train.device_tokens(tokens_before + local_start * T, local_sequences * T + 1, device)
        x_all, y_all = token_views(tokens, 0, local_sequences * T, T)
        inner_steps = local_sequences // args.inner_sequences
        values = []
        for i in range(inner_steps):
            xs = x_all[i * args.inner_sequences:(i + 1) * args.inner_sequences]
            ys = y_all[i * args.inner_sequences:(i + 1) * args.inner_sequences]
            count = xs.numel() * world
            summed = [torch.zeros_like(p) for p in params]
            terms = [0.0, 0.0, 0.0]
            for j in range(0, xs.shape[0], args.micro):
                if args.linearize:
                    grads, value = gn_model_gradient(model, keys, params, shadow_params, xs[j:j + args.micro],
                                                     ys[j:j + args.micro], count, args.precision)
                else:
                    grads, value = true_gradient(shadow, shadow_params, xs[j:j + args.micro], ys[j:j + args.micro],
                                                 count, args.precision)
                for acc, g in zip(summed, grads):
                    acc += g.float()
                terms = [a + b for a, b in zip(terms, value)]
            flat = torch.cat([g.reshape(-1) for g in summed] + [torch.tensor(terms, device=device, dtype=torch.float32)])
            dist.all_reduce(flat)
            parts = flat[:-3].split([p.numel() for p in params])
            for sp, g in zip(shadow_params, parts):
                sp.grad = g.view_as(sp).clone()
            values.append([float(v) for v in flat[-3:]])
            inner.step()
            inner.zero_grad(set_to_none=True)
        # line search on held-out training sequences
        with torch.no_grad():
            direction = [sp.detach() - p.detach() for sp, p in zip(shadow_params, params)]
            originals = [p.detach().clone() for p in params]
            base_line = mean_loss(model, line_x, line_y, args.micro, "bf16")
            dist.all_reduce(base_line)
            losses = []
            candidates = LINE if args.line_search else (1.0,)
            for a in candidates:
                for p, p0, d in zip(params, originals, direction):
                    p.copy_(p0 + a * d)
                loss = mean_loss(model, line_x, line_y, args.micro, "bf16")
                dist.all_reduce(loss)
                losses.append(float(loss))
            best = min(range(len(candidates)), key=lambda k: losses[k])
            alpha = candidates[best]
            for (k, p), p0, d in zip(model.named_parameters(), originals, direction):
                p.copy_(p0 + alpha * d)
                shrink = lr_ref * config["weight_decay"] if k in body else \
                    config["aux_learning_rate"] * schedule * config["weight_decay"]
                p.mul_(1 - shrink)
            norm = math.sqrt(sum(float(d.pow(2).sum()) for d in direction))
        denominator = world * args.line_sequences * T
        row = {"step": s, "tokens": s * config["batch_tokens"], "lr_ref": lr_ref, "alpha": alpha,
               "line_losses": [x / denominator for x in losses], "line_base": float(base_line) / denominator,
               "inner_steps": inner_steps, "train_nll": sum(v[0] for v in values) / len(values),
               "model_change_first": values[0][1] + values[0][2], "model_change_last": values[-1][1] + values[-1][2],
               "model_change_mean_last_quarter": sum(v[1] + v[2] for v in values[-max(1, len(values) // 4):]) / max(1, len(values) // 4),
               "direction_norm": norm,
               "training_seconds": time.time() - t0}
        if s % args.validation_every == 0 or s == stop or s == start + 1:
            row["validation_nll"] = validate()
        if rank == 0 and (s % args.checkpoint_every == 0 or s == stop):
            tmp = args.out / "checkpoint.pt.tmp"
            torch.save({"model": model.state_dict(), "shadow": shadow.state_dict(), "inner": inner.state_dict(),
                        "step": s, "config": config}, tmp)
            tmp.replace(resume)
        if rank == 0:
            (args.out / "steps" / f"step{s:06d}.json").write_text(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k != "line_losses"}),
                  flush=True)
    if rank == 0:
        (args.out / "done.json").write_text(json.dumps({"seconds": time.time() - started, "last_step": stop}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

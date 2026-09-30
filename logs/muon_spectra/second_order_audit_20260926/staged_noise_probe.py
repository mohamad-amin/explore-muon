"""Is the layer-staged PD correction a property of the curvature or of its sample? (2026-09-29, after the switch test)

newton_train.py --staged-pd computes each layer's step from the momentum average plus kappa x G (earlier layers' actual
steps), with G the GN on the first `--curvature-sequences` of each rank's batch (32 in every staged run so far), and PD's
root R from the same sequences. This probe asks how much of the resulting correction is reproducible across disjoint
curvature samples, and how that changes with 8x more sequences.

torchrun with 4 ranks, the harness's own functions. At a state s (a trainer kept checkpoint ARM:STEP, or a harness
checkpoint --harness FILE whose config/average it carries, with ARM:START naming the run for data and config):
  - the next average A' = beta A + (1 - beta) clip(g), g the step-(s+1) batch gradient, clipped globally at --clip
  - for each curvature size n per rank in --sizes and each of two disjoint sets (A = sequences [0, n), B = [n, 2n) of
    each rank's batch): R from the set (alpha 1/2, damping 1e-3); the plain PD direction polar(A' R) R per matrix; the
    staged direction (8 layer stages), exactly as newton_train.py
  - per matrix: cosines staged_A.staged_B, plain_A.plain_B, correction (staged - plain) A.B, staged_A.plain_A; and the
    correction's size |staged - plain| / |plain| (directions are polar maps, compared after NS)
Writes OUT_JSON with per-matrix values and layer means.

usage: .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 staged_noise_probe.py OUT_JSON ARM_DIR:STEP
       [--harness FILE] [--sizes 32,256] [--beta 0.9] [--clip 1.0]
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from newton_train import P, REPO, GNProduct, TokenStream, bf16, learning_rate, newton_schulz, ns_schedule, \
    partition_sequences, pd_roots, token_budget, token_views  # noqa: E402


def staged(model, body, body_keys, sizes, average, curv, micro, world, lr_ref, kappa, schedule_ns, stage_off):
    """newton_train.py's --staged-pd loop (alpha 1/2, damping 1e-3): per-matrix polar directions (unit-scale), in order."""
    roots = pd_roots(model, body_keys, curv, micro, 0.5, 1e-3)
    G = GNProduct(model, body_keys, curv, curv.numel() * world, micro)

    def op(v):
        return torch.cat([o.reshape(-1) for o in G([part.view_as(w) for part, w in zip(v.split(sizes), body)])])
    offsets = [0]
    for n_el in sizes:
        offsets.append(offsets[-1] + n_el)
    blocks = []
    for idx, key in enumerate(body_keys):
        block = key.split(".")[1]
        if not blocks or blocks[-1][0] != block:
            blocks.append((block, []))
        blocks[-1][1].append(idx)
    residual = average.clone()
    directions = [None] * len(body)
    for b_i, (_, members) in enumerate(blocks):
        block_step = torch.zeros_like(average)
        for idx in members:
            w = body[idx]
            rows, cols = w.shape
            r_part = residual[offsets[idx]:offsets[idx + 1]].view_as(w)
            R = roots[body_keys[idx]]
            d = newton_schulz((r_part.float() @ R)[None], schedule_ns)[0].float() @ R
            target = math.sqrt(min(rows, cols)) * math.sqrt(max(1.0, rows / cols))
            step_part = -(lr_ref * target / float(d.norm().clamp_min(1e-30))) * d
            directions[idx] = d / d.norm().clamp_min(1e-30)
            block_step[offsets[idx]:offsets[idx + 1]] = step_part.reshape(-1)
        if not stage_off and b_i + 1 < len(blocks):
            residual += kappa * op(block_step)
    return directions


def cos(a, b):
    return float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--harness", type=Path, default=None)
    parser.add_argument("--sizes", default="32,256")
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--clip", type=float, default=1.0)
    parser.add_argument("--micro", type=int, default=16)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", rank % torch.cuda.device_count())
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    arm, start = args.item.rsplit(":", 1)
    start = int(start)
    kept = Path(arm) / "scientific" / "kept"
    model, saved = P.load_checkpoint(kept / f"step{start:06d}.pt", device)
    config = dict(saved["config"])
    named = dict(model.named_parameters())
    body_keys = [k for k in named if k.startswith("blocks.") and named[k].ndim == 2]
    body = [named[k] for k in body_keys]
    sizes = [w.numel() for w in body]
    body_indices = saved["optimizer"]["param_groups"][0]["params"]
    average = torch.cat([(1 - config["muon_momentum"]) * saved["optimizer"]["state"][i]["momentum_buffer"].float().reshape(-1)
                         for i in body_indices]).to(device)
    step = start
    if args.harness is not None:        # a harness state: its weights, average and step
        state = torch.load(args.harness, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        average = state["average"].to(device)
        step = state["step"]
        del state
    del saved
    T = model.config.seq_len
    budget = token_budget(config, sum(p.numel() for p in named.values()), T)
    s = step + 1
    tokens_before = (s - 1) * config["batch_tokens"]
    lr_ref = learning_rate(config, s, tokens_before, budget)
    train = TokenStream(str(REPO / config["train_pattern"]))
    local_start, local_sequences = partition_sequences(config["batch_tokens"] // T, world, rank)
    x_all, y_all = token_views(train.device_tokens(tokens_before + local_start * T, local_sequences * T + 1, device),
                               0, local_sequences * T, T)
    for p in named.values():
        p.grad = None
    for i in range(0, local_sequences, args.micro):
        with bf16():
            loss = model(x_all[i:i + args.micro], y_all[i:i + args.micro])
        (loss * (x_all[i:i + args.micro].numel() * world / config["batch_tokens"])).backward()
    flat = torch.cat([p.grad.reshape(-1) for p in named.values()])
    dist.all_reduce(flat)
    flat /= world
    grad_norm = float(flat.float().norm())
    kappa = min(1.0, args.clip / grad_norm)
    flat *= kappa
    parts = dict(zip(named, flat.split([p.numel() for p in named.values()])))
    g = torch.cat([parts[k].float().reshape(-1) for k in body_keys])
    for p in named.values():
        p.grad = None
    average = args.beta * average + (1 - args.beta) * g
    for p in model.parameters():
        p.requires_grad_(True)
    schedule_ns = ns_schedule(config)
    results = {"item": args.item, "harness": str(args.harness) if args.harness else None, "step": step,
               "grad_norm": grad_norm, "kappa": kappa, "lr_ref": lr_ref, "names": body_keys, "sizes": {}}
    for n in [int(v) for v in args.sizes.split(",")]:
        sets = {"A": x_all[:n], "B": x_all[n:2 * n]}
        dirs = {}
        for label, curv in sets.items():
            dirs[("staged", label)] = staged(model, body, body_keys, sizes, average, curv, args.micro, world, lr_ref, kappa,
                                             schedule_ns, False)
            dirs[("plain", label)] = staged(model, body, body_keys, sizes, average, curv, args.micro, world, lr_ref, kappa,
                                            schedule_ns, True)
        rows = []
        for idx, key in enumerate(body_keys):
            sa, sb = dirs[("staged", "A")][idx], dirs[("staged", "B")][idx]
            pa, pb = dirs[("plain", "A")][idx], dirs[("plain", "B")][idx]
            ca, cb = sa - pa, sb - pb
            first = float(ca.norm()) < 1e-12 and float(cb.norm()) < 1e-12     # the first layer is never corrected
            rows.append({"name": key, "staged_AB": cos(sa, sb), "plain_AB": cos(pa, pb),
                         "correction_AB": None if first else cos(ca, cb),
                         "staged_plain_A": cos(sa, pa), "correction_size_A": float(ca.norm() / pa.norm().clamp_min(1e-30))})
        layers = {}
        for r in rows:
            layers.setdefault(r["name"].split(".")[1], []).append(r)
        def mean(values):
            values = [v for v in values if v is not None]
            return sum(values) / len(values) if values else float("nan")
        summary = {b: {k: mean([r[k] for r in rs]) for k in ("staged_AB", "plain_AB", "correction_AB",
                                                             "staged_plain_A", "correction_size_A")}
                   for b, rs in layers.items()}
        results["sizes"][n] = {"rows": rows, "layers": summary}
        if rank == 0:
            print(json.dumps({"n_per_rank": n, "layers": {b: {k: round(v, 3) for k, v in d.items()} for b, d in summary.items()}}),
                  flush=True)
    results["seconds"] = time.time() - started
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

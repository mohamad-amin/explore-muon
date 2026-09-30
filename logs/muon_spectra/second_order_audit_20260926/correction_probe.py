"""Is the band-restricted cross-layer correction just a shrink of the momentum's own top-band component? (review,
2026-09-29 13:17 CDT: the cheapest discriminator between cross-layer coordination and a curvature-free shrink.)

At a state (as band_probe.py), run the staged loop (--stage-band top, 32 curvature sequences per rank: the harness's) and,
for every corrected matrix, take the total correction c = residual - A' it received (kappa x GN of earlier layers' steps,
projected on its above-mean input band). Report per matrix: cos(c, -A'_top) and |c| / |A'_top| with A'_top = A' P_top
(the momentum's own top-band part), the share of A' in the top band, and the residual's top-band norm relative to A'_top.
If c ~ -gamma A'_top (cosine near 1), a curvature-free shrink of the top band is the right control.

usage: .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 correction_probe.py OUT_JSON ARM_DIR:STEP
       [--harness FILE] [--band top]
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
from band_probe import blocks_of  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--harness", type=Path, default=None)
    parser.add_argument("--band", default="top")
    parser.add_argument("--curvature", type=int, default=32)
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
    model, saved = P.load_checkpoint(Path(arm) / "scientific" / "kept" / f"step{start:06d}.pt", device)
    config = dict(saved["config"])
    named = dict(model.named_parameters())
    body_keys = [k for k in named if k.startswith("blocks.") and named[k].ndim == 2]
    body = [named[k] for k in body_keys]
    sizes = [w.numel() for w in body]
    offsets = [0]
    for n_el in sizes:
        offsets.append(offsets[-1] + n_el)
    body_indices = saved["optimizer"]["param_groups"][0]["params"]
    average = torch.cat([(1 - config["muon_momentum"]) * saved["optimizer"]["state"][i]["momentum_buffer"].float().reshape(-1)
                         for i in body_indices]).to(device)
    step = start
    if args.harness is not None:
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
    schedule_ns = ns_schedule(config)
    curv = x_all[:args.curvature]
    roots, projectors = pd_roots(model, body_keys, curv, args.micro, 0.5, 1e-3, band=args.band)
    G = GNProduct(model, body_keys, curv, curv.numel() * world, args.micro)

    def op(v):
        return torch.cat([o.reshape(-1) for o in G([part.view_as(w) for part, w in zip(v.split(sizes), body)])])
    blocks = blocks_of(body_keys)
    residual = average.clone()
    rows = []
    for b_i, (_, members) in enumerate(blocks):
        block_step = torch.zeros_like(average)
        for idx in members:
            w = body[idx]
            m_rows, m_cols = w.shape
            a = average[offsets[idx]:offsets[idx + 1]].view_as(w).float()
            r = residual[offsets[idx]:offsets[idx + 1]].view_as(w).float()
            Pm = projectors[body_keys[idx]]
            if b_i > 0:
                c = r - a
                a_top = a @ Pm
                rows.append({"name": body_keys[idx], "layer": b_i,
                             "cos_c_minus_atop": float(-(c * a_top).sum() / (c.norm() * a_top.norm()).clamp_min(1e-30)),
                             "c_over_atop": float(c.norm() / a_top.norm().clamp_min(1e-30)),
                             "atop_share": float(a_top.norm() ** 2 / a.norm().clamp_min(1e-30) ** 2),
                             "rtop_over_atop": float((r @ Pm).norm() / a_top.norm().clamp_min(1e-30)),
                             "cos_rtop_atop": float(((r @ Pm) * a_top).sum() / ((r @ Pm).norm() * a_top.norm()).clamp_min(1e-30))})
            R = roots[body_keys[idx]]
            d = newton_schulz((r @ R)[None], schedule_ns)[0].float() @ R
            target = math.sqrt(min(m_rows, m_cols)) * math.sqrt(max(1.0, m_rows / m_cols))
            block_step[offsets[idx]:offsets[idx + 1]] = (-(lr_ref * target / float(d.norm().clamp_min(1e-30))) * d).reshape(-1)
        if b_i + 1 < len(blocks):
            correction = op(block_step)
            for later in [i for _, ms in blocks[b_i + 1:] for i in ms]:
                w = body[later]
                part = correction[offsets[later]:offsets[later + 1]].view_as(w)
                correction[offsets[later]:offsets[later + 1]] = (part.float() @ projectors[body_keys[later]]).reshape(-1)
            residual += kappa * correction
    summary = {}
    for key in ("cos_c_minus_atop", "c_over_atop", "atop_share", "rtop_over_atop", "cos_rtop_atop"):
        vals = [r[key] for r in rows]
        summary[key] = {"mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["name"].split(".")[-1], []).append(r)
    kinds = {k: {key: sum(r[key] for r in rs) / len(rs) for key in ("cos_c_minus_atop", "c_over_atop", "rtop_over_atop")}
             for k, rs in by_kind.items()}
    if rank == 0:
        print(json.dumps({"kappa": round(kappa, 3), "summary": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in summary.items()},
                          "by_kind": {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in kinds.items()}}), flush=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"item": args.item, "harness": str(args.harness) if args.harness else None, "step": step,
                                        "kappa": kappa, "band": args.band, "rows": rows, "summary": summary, "by_kind": kinds,
                                        "seconds": time.time() - started}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

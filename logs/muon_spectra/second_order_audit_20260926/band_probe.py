"""What does band-restricted coordination change in the step? (2026-09-29, after the band arms, MUON_CASE 08:45 CDT)

At a state (a trainer kept checkpoint ARM:STEP, or a harness checkpoint --harness FILE with ARM:START naming the run),
torchrun with 4 ranks, the harness's own functions (newton_train.py). From the next average A' (the step's clipped
batch gradient into the saved average, as the harness does), the directions of five maps, each from the curvature set A
(the harness's: the first n sequences of each rank's batch) and from a disjoint set B:
    plain   PD's map with the step's R (--stage-off)
    full    layer-staged PD (the whole cross-layer correction)
    top     the correction kept on each matrix's above-mean input directions (--stage-band top)
    bulk    the correction kept on the rest (--stage-band bulk)
    half    the whole correction at half strength (--stage-scale 0.5)
For each map's set-A step at the harness's per-matrix step norms (lr_ref sqrt(min(m, n)) sqrt(max(1, m/n))):
  - the true held-out loss change at 0.5x, 1x and 2x the step (held-out training sequences, disjoint from both sets);
  - the GN curvature q = D.G D on set A and its layer-block-diagonal part, sum_b D_b.G_bb D_b; their ratio is the
    step's cross-layer coherence (1: layers' output changes independent; > 1: they add up);
  - the first-order slope on the held-out sequences and c* = -slope / q (the GN model's optimal multiple);
  - the per-matrix cosine with plain and the A.B agreement (set B's direction).

usage: .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 band_probe.py OUT_JSON ARM_DIR:STEP
       [--harness FILE] [--curvature 32] [--held 32] [--beta 0.9] [--clip 1.0]
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
from newton_train import HELD_OUT_TRAIN, P, REPO, GNProduct, TokenStream, bf16, learning_rate, newton_schulz, \
    ns_schedule, partition_sequences, pd_roots, summed_loss, token_budget, token_views  # noqa: E402

MAPS = {"plain": dict(off=True), "full": dict(), "top": dict(band="top"), "bulk": dict(band="bulk"), "half": dict(scale=0.5),
        "mean": dict(band="mean"), "top8": dict(band="top8")}      # mean / top8 added 12:5x CDT


def blocks_of(body_keys):
    blocks = []
    for idx, key in enumerate(body_keys):
        block = key.split(".")[1]
        if not blocks or blocks[-1][0] != block:
            blocks.append((block, []))
        blocks[-1][1].append(idx)
    return blocks


def staged_step(model, body, body_keys, sizes, offsets, average, curv, micro, world, lr_ref, kappa, schedule_ns,
                off=False, band=None, scale=1.0):
    """newton_train.py's --staged-pd loop (alpha 1/2, damping 1e-3) with its --stage-band / --stage-scale options:
    the full step vector (harness step norms)."""
    if band is None:
        roots, projectors = pd_roots(model, body_keys, curv, micro, 0.5, 1e-3), None
    else:
        roots, projectors = pd_roots(model, body_keys, curv, micro, 0.5, 1e-3, band=band)
    G = GNProduct(model, body_keys, curv, curv.numel() * world, micro)

    def op(v):
        return torch.cat([o.reshape(-1) for o in G([part.view_as(w) for part, w in zip(v.split(sizes), body)])])
    blocks = blocks_of(body_keys)
    residual = average.clone()
    step = torch.zeros_like(average)
    for b_i, (_, members) in enumerate(blocks):
        block_step = torch.zeros_like(average)
        for idx in members:
            w = body[idx]
            rows, cols = w.shape
            r_part = residual[offsets[idx]:offsets[idx + 1]].view_as(w)
            R = roots[body_keys[idx]]
            d = newton_schulz((r_part.float() @ R)[None], schedule_ns)[0].float() @ R
            target = math.sqrt(min(rows, cols)) * math.sqrt(max(1.0, rows / cols))
            block_step[offsets[idx]:offsets[idx + 1]] = (-(lr_ref * target / float(d.norm().clamp_min(1e-30))) * d).reshape(-1)
        step += block_step
        if not off and b_i + 1 < len(blocks):
            correction = op(block_step)
            if projectors is not None:
                for later in [i for _, ms in blocks[b_i + 1:] for i in ms]:
                    w = body[later]
                    part = correction[offsets[later]:offsets[later + 1]].view_as(w)
                    correction[offsets[later]:offsets[later + 1]] = (part.float() @ projectors[body_keys[later]]).reshape(-1)
            residual += (kappa * scale) * correction
    return step, op


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--harness", type=Path, default=None)
    parser.add_argument("--curvature", type=int, default=32, help="sequences per rank and set")
    parser.add_argument("--held", type=int, default=32, help="held-out training sequences per rank")
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
    held_x, held_y = token_views(train.device_tokens(HELD_OUT_TRAIN + (1 << 22) + rank * args.held * T, args.held * T + 1,
                                                     device), 0, args.held * T, T)
    held_count = world * args.held * T
    # held-out slope: gradient of the held-out loss w.r.t. the body at the state
    for p in named.values():
        p.grad = None
    for i in range(0, args.held, args.micro):
        with bf16():
            loss = model(held_x[i:i + args.micro], held_y[i:i + args.micro])
        (loss * (held_x[i:i + args.micro].numel() / held_count)).backward()
    held_grad = torch.cat([w.grad.float().reshape(-1) for w in body])
    dist.all_reduce(held_grad)
    for p in named.values():
        p.grad = None
    base = summed_loss(model, held_x, held_y, args.micro) / held_count
    sets = {"A": x_all[:args.curvature], "B": x_all[args.curvature:2 * args.curvature]}
    blocks = blocks_of(body_keys)
    results = {"item": args.item, "harness": str(args.harness) if args.harness else None, "step": step,
               "grad_norm": grad_norm, "kappa": kappa, "lr_ref": lr_ref, "held_base": base, "maps": {}}
    steps = {}
    for name, opts in MAPS.items():
        dA, opA = staged_step(model, body, body_keys, sizes, offsets, average, sets["A"], args.micro, world, lr_ref, kappa,
                              schedule_ns, **opts)
        dB, _ = staged_step(model, body, body_keys, sizes, offsets, average, sets["B"], args.micro, world, lr_ref, kappa,
                            schedule_ns, **opts)
        steps[name] = dA
        q = float(dA @ opA(dA))     # GN products need autograd (reverse pass inside), so not under no_grad
        diag = 0.0
        for _, members in blocks:
            mask = torch.zeros_like(dA)
            for idx in members:
                mask[offsets[idx]:offsets[idx + 1]] = 1
            piece = dA * mask
            diag += float(piece @ opA(piece))
        slope = float(held_grad @ dA)
        with torch.no_grad():
            changes = {}
            originals = [w.detach().clone() for w in body]
            for mult in (0.5, 1.0, 2.0):
                for w, w0, part in zip(body, originals, dA.split(sizes)):
                    w.copy_(w0 + mult * part.view_as(w0))
                changes[str(mult)] = summed_loss(model, held_x, held_y, args.micro) / held_count - base
            for w, w0 in zip(body, originals):
                w.copy_(w0)
        per_ab = [float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30)) for a, b in zip(dA.split(sizes), dB.split(sizes))]
        results["maps"][name] = {"q": q, "q_layer_diagonal": diag, "coherence": q / diag if diag > 0 else None,
                                 "slope": slope, "c_star": -slope / q if q > 0 else None, "true_change": changes,
                                 "agreement_AB_mean": sum(per_ab) / len(per_ab), "step_norm": float(dA.norm())}
        if rank == 0:
            print(json.dumps({"map": name, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in results["maps"][name].items()
                                              if k != "true_change"}, "true_change": {k: round(v, 5) for k, v in changes.items()}}),
                  flush=True)
    for name in MAPS:
        if name != "plain":
            per = [float((a * b).sum() / (a.norm() * b.norm()).clamp_min(1e-30))
                   for a, b in zip(steps[name].split(sizes), steps["plain"].split(sizes))]
            results["maps"][name]["cos_plain_mean"] = sum(per) / len(per)
    results["seconds"] = time.time() - started
    if rank == 0:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

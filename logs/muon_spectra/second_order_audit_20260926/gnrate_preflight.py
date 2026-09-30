"""Pre-flight for the GN-PD rate test (review, 2026-09-28 09:59 CDT): at the 16M PD states, is the harness's GN-PD
direction reproducible, and does it, or the kron-nested construction, beat its Kronecker control in one step?

torchrun with 4 ranks, the harness's own code (newton_train.py). From a kept PD checkpoint at step s0:
  - A' = beta A + (1 - beta) clip(g): the next momentum. A = (1 - beta) M is the checkpoint's, g the run's step-(s0+1)
    16M batch gradient clipped globally at the trainer's 1.0 (as newton_train.py --clip 1.0 --momentum 0.9)
  - curvature sets per rank: A = sequences [0, 32) of the rank's batch (the harness's), B = [32, 64) (disjoint)
  - directions, each matrix rescaled to the harness step lr sqrt(min(m, n)) sqrt(max(1, m/n)):
      muon    polar(A') (SVD)
      pd      polar(A' R) R, R = (C_A / mean + 1e-3)^-1/2
      kron    L polar(L A' R) R, TS 1/2/1/2 from set A: the harness control
      gnpd_*  -(G + mu)^-p polar((G + mu)^-p A'), G the exact GN of the 48 body matrices, mu = d rho with
              rho = A'.G A' / A'.A': A_bf16_k64_p0.5_d1e-3 (the harness), B_bf16_k64_p0.5_d1e-3 (disjoint set),
              A_fp32_k128_p0.5_d1e-3, A_bf16_k64_p0.25_d1e-3, A_bf16_k64_p0.5_d1e-4
      nested  -K^-1/2 (Gh + mu)^-1/2 polar((Gh + mu)^-1/2 K^-1/2 A'), Gh = K^-1/2 G K^-1/2 with
              K = blockdiag(s_m B_m (x) C_m) from kron's damped factors, s_m matching each diagonal block of Gh to unit
              mean eigenvalue (Hutchinson, 8 probes); FP32, Lanczos 64, mu = 1e-3. Equal to kron when G = K.
  - per-matrix cosines between all directions
  - on held-out training sequences (tokens beyond every budget, 32 per rank per set), per direction at the harness
    step: slope a and GN curvature q (set H1), c* = -a / q in units of that step; the true loss change at 0.5, 1 and
    2 x the step (set H2, paired)
  - mean eig(G) on set A (Hutchinson, 8 probes, BF16) against mu = 1e-3 rho

usage: .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 gnrate_preflight.py OUT_JSON ARM_DIR:STEP
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream, token_views  # noqa: E402
from research.adamw_spectra.distributed import partition_sequences  # noqa: E402
from research.adamw_spectra.train import learning_rate, token_budget  # noqa: E402
from newton_train import HELD_OUT_TRAIN, GNProduct, bf16, kron_roots  # noqa: E402

KINDS = ("attn.q", "attn.k", "attn.v", "attn.o", "mlp.up", "mlp.down")


def lanczos(op, start, k):
    V = torch.zeros(k, start.numel(), device=start.device)
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
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    off = torch.tensor(betas, dtype=torch.float64)
    return V, T + torch.diag(off, 1) + torch.diag(off, -1)


def matrix_power(V, T, start_norm, power, mu):
    values, vectors = torch.linalg.eigh(T)
    y = vectors @ ((values.clamp_min(0) + mu).pow(-power) * vectors[0] * start_norm)
    return V.T @ y.to(V.device, torch.float32)


def polar(x):
    u, _, vh = torch.linalg.svd(x.float(), full_matrices=False)
    return u @ vh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--curvature-sequences", type=int, default=32, help="per rank and set")
    parser.add_argument("--held-sequences", type=int, default=32, help="per rank and held-out set")
    parser.add_argument("--micro", type=int, default=16)
    parser.add_argument("--beta", type=float, default=0.9)
    parser.add_argument("--clip", type=float, default=1.0)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    log = {}

    def note(key, value):
        log[key] = value
        if rank == 0:
            print(json.dumps({key: value, "t": round(time.time() - started)}), flush=True)

    arm, s0 = args.item.rsplit(":", 1)
    s0 = int(s0)
    model, saved = P.load_checkpoint(Path(arm) / "scientific" / "kept" / f"step{s0:06d}.pt", device)
    config = dict(saved["config"])
    T = model.config.seq_len
    named = dict(model.named_parameters())
    keys = [k for k in named if k.startswith("blocks.") and named[k].ndim == 2]
    body = [named[k] for k in keys]
    shapes = [w.shape for w in body]
    sizes = [w.numel() for w in body]
    state = saved["optimizer"]["state"]
    body_indices = saved["optimizer"]["param_groups"][0]["params"]
    average = torch.cat([(1 - config["muon_momentum"]) * state[i]["momentum_buffer"].float().reshape(-1)
                         for i in body_indices]).to(device)
    del saved, state

    def split(v):
        return [part.view(shape) for part, shape in zip(v.split(sizes), shapes)]

    def flat(parts):
        return torch.cat([p.reshape(-1) for p in parts])

    # the next step's 16M batch gradient, clipped as the trainer does, into the momentum average
    budget = token_budget(config, sum(p.numel() for p in named.values()), T)
    s = s0 + 1
    tokens_before = (s - 1) * config["batch_tokens"]
    lr = learning_rate(config, s, tokens_before, budget)
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
    grad = torch.cat([p.grad.reshape(-1) for p in named.values()])
    dist.all_reduce(grad)
    grad /= world
    grad_norm = float(grad.float().norm())
    clip = min(1.0, args.clip / grad_norm)
    offsets, g_parts, position = dict(), [], 0
    for name, p in named.items():
        offsets[name] = (position, p.numel())
        position += p.numel()
    g_body = torch.cat([grad[offsets[k][0]:offsets[k][0] + offsets[k][1]].float() for k in keys]) * clip
    for p in named.values():
        p.grad = None
    del grad
    b = args.beta * average + (1 - args.beta) * g_body
    del average, g_body
    note("gradient", {"step": s, "lr": lr, "grad_norm_before_clip": grad_norm, "clip_factor": clip})

    n = args.curvature_sequences
    sets = {"A": (x_all[:n], y_all[:n]), "B": (x_all[n:2 * n], y_all[n:2 * n])}
    ops = {}
    for label, (xs, _) in sets.items():
        for precision in ("bf16", "fp32"):
            G = GNProduct(model, keys, xs, xs.numel() * world, args.micro if precision == "bf16" else args.micro // 2,
                          fp32=precision == "fp32")
            ops[(label, precision)] = (lambda G_: lambda v: flat(G_(split(v))))(G)
    held = []
    for j in range(2):
        first = HELD_OUT_TRAIN + (j * world + rank) * args.held_sequences * T
        hx, hy = token_views(train.device_tokens(first, args.held_sequences * T + 1, device), 0, args.held_sequences * T, T)
        held.append([(hx[i:i + 8], hy[i:i + 8]) for i in range(0, args.held_sequences, 8)])

    # Kronecker factors from set A (the harness's statistics)
    roots = kron_roots(model, keys, sets["A"][0], sets["A"][1], args.micro, 0.5, 1e-3,
                       torch.Generator(device=device).manual_seed(1000 * s + rank))
    targets = [lr * math.sqrt(min(m, k)) * math.sqrt(max(1.0, m / k)) for m, k in shapes]

    def harness_step(direction):
        """Each matrix of a descent direction rescaled to the harness step."""
        return flat([d * (t / float(d.norm().clamp_min(1e-30))) for d, t in zip(split(direction), targets)])

    directions = {}
    parts = split(b)
    directions["muon"] = harness_step(flat([-polar(m) for m in parts]))
    directions["pd"] = harness_step(flat([-(polar(m @ roots[k][1]) @ roots[k][1]) for k, m in zip(keys, parts)]))
    directions["kron"] = harness_step(flat([-(roots[k][0] @ polar(roots[k][0] @ m @ roots[k][1]) @ roots[k][1])
                                            for k, m in zip(keys, parts)]))
    note("kron_done", True)

    # mean eigenvalue of G on set A (Hutchinson, Rademacher, identical on every rank)
    gen = torch.Generator().manual_seed(7)
    probes = [(torch.randint(0, 2, (sum(sizes),), generator=gen).float() * 2 - 1).to(device) for _ in range(8)]
    mean_eig = sum(float(v @ ops[("A", "bf16")](v)) for v in probes) / (len(probes) * sum(sizes))
    note("mean_eig_G_A", mean_eig)

    def gnpd(op, first, p, d):
        V, Tk = first
        rho = float(Tk[0, 0])
        w = matrix_power(V, Tk, float(b.norm()), p, d * rho)
        polar_flat = flat([polar(x) for x in split(w)])
        V2, T2 = lanczos(op, polar_flat, V.shape[0])
        z = matrix_power(V2, T2, float(polar_flat.norm()), p, d * rho)
        del V2
        return harness_step(-z), rho, float(torch.linalg.eigvalsh(Tk)[-1])

    ritz = {}
    first = lanczos(ops[("A", "bf16")], b, 64)
    for label, p, d in (("gnpd_A_bf16_k64_p0.5_d1e-3", 0.5, 1e-3), ("gnpd_A_bf16_k64_p0.25_d1e-3", 0.25, 1e-3),
                        ("gnpd_A_bf16_k64_p0.5_d1e-4", 0.5, 1e-4)):
        directions[label], rho, top = gnpd(ops[("A", "bf16")], first, p, d)
        ritz[label] = {"rho": rho, "ritz_top": top}
        note(label, ritz[label])
    del first
    first = lanczos(ops[("B", "bf16")], b, 64)
    directions["gnpd_B_bf16_k64_p0.5_d1e-3"], rho, top = gnpd(ops[("B", "bf16")], first, 0.5, 1e-3)
    ritz["gnpd_B_bf16_k64_p0.5_d1e-3"] = {"rho": rho, "ritz_top": top}
    note("gnpd_B", ritz["gnpd_B_bf16_k64_p0.5_d1e-3"])
    del first
    first = lanczos(ops[("A", "fp32")], b, 128)
    directions["gnpd_A_fp32_k128_p0.5_d1e-3"], rho, top = gnpd(ops[("A", "fp32")], first, 0.5, 1e-3)
    ritz["gnpd_A_fp32_k128_p0.5_d1e-3"] = {"rho": rho, "ritz_top": top}
    note("gnpd_fp32", ritz["gnpd_A_fp32_k128_p0.5_d1e-3"])
    del first

    # nested: K^-1/2 = s^-1/2 (L (x) R) per matrix, s matching Gh's diagonal blocks to unit mean eigenvalue
    op32 = ops[("A", "fp32")]

    def whiten(v, scales):
        return flat([sc ** -0.5 * (roots[k][0] @ x @ roots[k][1]) for k, x, sc in zip(keys, split(v), scales)])
    unit = [1.0] * len(keys)
    block = torch.zeros(len(keys), dtype=torch.float64)
    for v in probes:
        gv = whiten(op32(whiten(v, unit)), unit)
        for m, (gi, vi) in enumerate(zip(split(gv), split(v))):
            block[m] += float(gi.reshape(-1) @ vi.reshape(-1)) / vi.numel()
    scales = (block / len(probes)).clamp_min(1e-12).tolist()
    note("nested_scales_by_kind", {kind: sum(sc for k, sc in zip(keys, scales) if kind in k) / 8 for kind in KINDS})

    def op_hat(v):
        return whiten(op32(whiten(v, scales)), scales)
    b_hat = whiten(b, scales)
    V, Tk = lanczos(op_hat, b_hat, 64)
    ritz["nested"] = {"rho_hat": float(Tk[0, 0]), "ritz_top_hat": float(torch.linalg.eigvalsh(Tk)[-1]),
                      "ritz_hat": [float(x) for x in torch.linalg.eigvalsh(Tk)]}
    w = matrix_power(V, Tk, float(b_hat.norm()), 0.5, 1e-3)
    del V
    polar_flat = flat([polar(x) for x in split(w)])
    V2, T2 = lanczos(op_hat, polar_flat, 64)
    z = matrix_power(V2, T2, float(polar_flat.norm()), 0.5, 1e-3)
    del V2
    directions["nested"] = harness_step(-whiten(z, scales))
    note("nested", {k: v for k, v in ritz["nested"].items() if k != "ritz_hat"})

    # cosines per matrix between all directions, summarized by kind
    labels = list(directions)
    cos = {}
    for i, a in enumerate(labels):
        for c in labels[i + 1:]:
            per = [float(x.reshape(-1) @ y.reshape(-1) / (x.norm() * y.norm()).clamp_min(1e-30))
                   for x, y in zip(split(directions[a]), split(directions[c]))]
            cos[f"{a}|{c}"] = {"mean": sum(per) / len(per),
                               "by_kind": {kind: sum(v for k, v in zip(keys, per) if kind in k) / 8 for kind in KINDS},
                               "per_matrix": per}
    note("cos_summary", {pair: round(v["mean"], 3) for pair, v in cos.items()})

    # one-step scores on held-out training sequences
    def reduce(values):
        t = torch.tensor(values, dtype=torch.float64, device=device)
        dist.all_reduce(t)
        return t.tolist()

    @torch.no_grad()
    def held_loss(batches):
        total = sum(float(P.token_losses(model(x), y).mean(1).sum()) for x, y in batches)
        return reduce([total, sum(x.shape[0] for x, _ in batches)])

    base_total, base_count = held_loss(held[1])
    base = base_total / base_count
    scores = {}
    originals = [w.detach().clone() for w in body]
    for label, d in directions.items():
        first_sum = q_sum = count = 0.0
        dmap = dict(zip(keys, split(d)))
        for x, y in held[0]:
            t = P.directional_terms(model, x, y, dmap)
            first_sum += float(t["first"].sum())
            q_sum += float(t["q"].sum())
            count += x.shape[0]
        first_sum, q_sum, count = reduce([first_sum, q_sum, count])
        a, q = first_sum / count, q_sum / count
        changes = {}
        for c in (0.5, 1.0, 2.0):
            with torch.no_grad():
                for w, w0, di in zip(body, originals, split(d)):
                    w.copy_(w0 + c * di)
            total, cnt = held_loss(held[1])
            changes[str(c)] = total / cnt - base
        with torch.no_grad():
            for w, w0 in zip(body, originals):
                w.copy_(w0)
        scores[label] = {"slope": a, "curvature": q, "c_star": -a / q if q > 0 else float("nan"),
                         "model_change_at_1": a + 0.5 * q, "true_change": changes}
        note(f"score_{label}", {k: (round(v, 5) if isinstance(v, float) else {kk: round(vv, 5) for kk, vv in v.items()})
                                for k, v in scores[label].items()})
    if rank == 0:
        args.out.write_text(json.dumps({"item": args.item, "step": s, "lr": lr, "log": log, "ritz": ritz, "scores": scores,
                                        "cos": cos, "seconds": time.time() - started}, indent=1) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

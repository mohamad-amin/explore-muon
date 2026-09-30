"""Step vs path at 16M (2026-09-30 11:40 CDT; revised after review 12:0x CDT, MUON_CASE): what the endpoints of one 16M
step and of K sequential smaller-batch steps on the same batch (true loss, or the batch's GN model at the start) reach,
and how the local model at the start rates each displacement.

torchrun with 4 ranks. From a kept 16M state theta (ARM:STEP) and endpoints label=DIR (a newton_train.py run from it,
DIR/checkpoint.pt) or label=FILE.pt (a state dict, e.g. the real run's kept next-step weights), D is each endpoint's
body-matrix displacement (auxiliary parameters stay at theta; the path runs froze them):
  - the gradient g_V at theta on the validation tokens used for every loss below (unclipped), and the batch gradient g_B
    on the run's next 16M batch (the one every path consumed; in-sample);
  - G D for every endpoint, G the exact GN of the mean token loss on --curvature-sequences held-out sequences per rank;
    a_i = g_V.D_i, b_i = g_B.D_i, Q_ij = D_i.G D_j;
  - the true validation loss along theta + c D_i for c in --cs (with c = +-0.05 giving the true directional curvature
    by finite differences, against Q_ii), and m_i(c) = L + c a_i + c^2 Q_ii / 2;
  - the model's minimizer x* in the span of all D's, and the true loss at x*/2, x*, 2x* and at x* rescaled to the
    norms of the first two endpoints;
  - each D with the top --project GN eigendirections at theta removed (Lanczos, --krylov steps from a fixed random
    start), and its true loss: the oscillation-phase-free part;
  - per body matrix: |D_i|, cosines between endpoints, and the share of |D_i|^2 on the input band (C's above-mean
    eigenvectors at theta from --band-sequences per rank).

usage: .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 path_probe.py OUT_JSON ARM:STEP \
           label=DIR_OR_FILE [...] [--cs ...]
"""
import argparse
import json
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
from newton_train import HELD_OUT_TRAIN, GNProduct, bf16, pd_roots, summed_loss  # noqa: E402


def body_gradient(model, keys, x_all, y_all, micro, total_tokens):
    named = dict(model.named_parameters())
    for p in model.parameters():
        p.grad = None
    for i in range(0, x_all.shape[0], micro):
        with bf16():
            loss = model(x_all[i:i + micro], y_all[i:i + micro])
        (loss * (x_all[i:i + micro].numel() / total_tokens)).backward()
    flat = torch.cat([named[k].grad.float().reshape(-1) for k in keys])
    dist.all_reduce(flat)
    for p in model.parameters():
        p.grad = None
    return flat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("endpoints", nargs="+")
    parser.add_argument("--cs", default="-0.05,0.05,0.25,0.5,0.75,1,1.25,1.5,2,3")
    parser.add_argument("--curvature-sequences", type=int, default=128, help="held-out GN sequences per rank")
    parser.add_argument("--band-sequences", type=int, default=32, help="sequences per rank for C and the input band")
    parser.add_argument("--krylov", type=int, default=32)
    parser.add_argument("--project", type=int, default=16)
    parser.add_argument("--micro", type=int, default=16)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.time()
    arm, start = args.item.rsplit(":", 1)
    start = int(start)
    model, saved = P.load_checkpoint(Path(arm) / "scientific" / "kept" / f"step{start:06d}.pt", device)
    config = dict(saved["config"])
    del saved
    T = model.config.seq_len
    named = dict(model.named_parameters())
    keys = [k for k in named if k.startswith("blocks.") and named[k].ndim == 2]
    sizes = [named[k].numel() for k in keys]
    theta = {k: v.detach().clone() for k, v in named.items()}
    theta_body = torch.cat([theta[k].float().reshape(-1) for k in keys])
    labels, D = [], []
    for spec in args.endpoints:
        label, source = spec.split("=", 1)
        path = Path(source)
        state = torch.load(path / "checkpoint.pt" if path.is_dir() else path, map_location="cpu", weights_only=False)
        state = state.get("model", state)
        state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
        labels.append(label)
        D.append(torch.cat([state[k].to(device).float().reshape(-1) for k in keys]) - theta_body)
        del state
    cs = [float(c) for c in args.cs.split(",")]
    n = len(D)

    train = TokenStream(str(REPO / config["train_pattern"]))
    validation = TokenStream(str(REPO / config["validation_pattern"]))
    tokens_before = start * config["batch_tokens"]
    b_start, b_count = partition_sequences(config["batch_tokens"] // T, world, rank)
    bx, by = token_views(train.device_tokens(tokens_before + b_start * T, b_count * T + 1, device), 0, b_count * T, T)
    base_h = HELD_OUT_TRAIN + 50_000_000     # beyond the harness's line and fixed-root sequences
    per_rank = args.curvature_sequences + args.band_sequences
    h_tokens = train.device_tokens(base_h + rank * per_rank * T, per_rank * T + 1, device)
    cx, _ = token_views(h_tokens, 0, args.curvature_sequences * T, T)
    px, _ = token_views(h_tokens, args.curvature_sequences * T, args.band_sequences * T, T)
    v_start, v_count = partition_sequences(config["validation_tokens"] // T, world, rank)
    vx, vy = token_views(validation.device_tokens(v_start * T, v_count * T + 1, device), 0, v_count * T, T)

    g_val = body_gradient(model, keys, vx, vy, args.micro, config["validation_tokens"])
    g_batch = body_gradient(model, keys, bx, by, args.micro, config["batch_tokens"])
    del bx, by
    _, band = pd_roots(model, keys, px, args.micro, 0.5, 1e-3, band="top")
    gn = GNProduct(model, keys, cx, world * args.curvature_sequences * T, args.micro)

    def op(v):
        return torch.cat([o.reshape(-1) for o in gn([part.view_as(named[k]) for part, k in zip(v.split(sizes), keys)])])
    GD = [op(d) for d in D]
    a_val = [float(g_val @ d) for d in D]
    a_batch = [float(g_batch @ d) for d in D]
    Q = [[float(D[i] @ GD[j]) for j in range(n)] for i in range(n)]
    del GD

    def set_body(vec):
        with torch.no_grad():
            for k, part in zip(keys, (theta_body + vec).split(sizes)):
                named[k].copy_(part.view_as(named[k]).to(named[k].dtype))

    def restore():
        with torch.no_grad():
            for k in keys:
                named[k].copy_(theta[k])

    def val_at(vec):
        set_body(vec)
        value = summed_loss(model, vx, vy, args.micro) / config["validation_tokens"]
        restore()
        return value

    base = summed_loss(model, vx, vy, args.micro) / config["validation_tokens"]
    lines = {}
    for i, label in enumerate(labels):
        true = [val_at(c * D[i]) - base for c in cs]
        eps = 0.05
        fd = None
        if -eps in cs and eps in cs:
            fd = (true[cs.index(eps)] + true[cs.index(-eps)]) / eps ** 2     # true directional 2nd derivative
        norm = float(D[i].norm())
        lines[label] = {"c": cs, "norm": norm, "true": true,
                        "model_val": [c * a_val[i] + 0.5 * c * c * Q[i][i] for c in cs],
                        "model_batch": [c * a_batch[i] + 0.5 * c * c * Q[i][i] for c in cs],
                        "curvature_fd": fd, "curvature_gn": Q[i][i],
                        "c_star_model": -a_val[i] / Q[i][i] if Q[i][i] > 0 else None}

    # the model's minimizer in the span of the D's
    Qm = torch.tensor(Q, dtype=torch.float64)
    am = torch.tensor(a_val, dtype=torch.float64)
    ridge = 1e-6 * float(Qm.diagonal().abs().max())
    x = -torch.linalg.solve(Qm + ridge * torch.eye(n, dtype=torch.float64), am)
    combo = sum(float(x[i]) * D[i] for i in range(n))
    span = {"coefficients": dict(zip(labels, x.tolist())), "norm": float(combo.norm())}
    for scale in (0.5, 1.0, 2.0):
        span[f"true_at_{scale}"] = val_at(scale * combo) - base
        span[f"model_at_{scale}"] = float(scale * (x @ am) + 0.5 * scale * scale * (x @ Qm @ x))
    for label in labels[:2]:
        target = float(D[labels.index(label)].norm())
        scale = target / max(span["norm"], 1e-30)
        span[f"true_at_norm_of_{label}"] = val_at(scale * combo) - base
        span[f"model_at_norm_of_{label}"] = float(scale * (x @ am) + 0.5 * scale * scale * (x @ Qm @ x))
    del combo

    # top GN eigendirections at theta (Lanczos, full reorthogonalization), projected out of each D
    generator = torch.Generator(device=device).manual_seed(20260930)
    start_vec = torch.randn(sum(sizes), device=device, generator=generator)
    dist.broadcast(start_vec, 0)
    k = args.krylov
    V = torch.zeros(k, sum(sizes), device=device)
    V[0] = start_vec / start_vec.norm()
    alphas, betas = [], []
    for j in range(k):
        w = op(V[j])
        alphas.append(float(w @ V[j]))
        for _ in range(2):
            w -= V[:j + 1].T @ (V[:j + 1] @ w)
        if j + 1 == k:
            break
        beta = float(w.norm())
        betas.append(beta)
        V[j + 1] = w / beta
    Tk = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    for j, b in enumerate(betas):
        Tk[j, j + 1] = Tk[j + 1, j] = b
    ritz, Y = torch.linalg.eigh(Tk)
    top = Y[:, -args.project:].to(device, torch.float32)
    Z = top.T @ V       # the top Ritz vectors, rows
    del V
    projected = {}
    for i, label in enumerate(labels):
        coeff = Z @ D[i]
        rest = D[i] - Z.T @ coeff
        projected[label] = {"removed_energy_share": float(coeff.square().sum() / D[i].square().sum().clamp_min(1e-30)),
                            "true_at_1": val_at(rest) - base, "slope_val": float(g_val @ rest)}

    per_matrix = {}
    offsets = [0]
    for s_ in sizes:
        offsets.append(offsets[-1] + s_)
    for j, key in enumerate(keys):
        parts = [d[offsets[j]:offsets[j + 1]].view_as(named[key]) for d in D]
        per_matrix[key] = {
            "norm": {labels[i]: float(parts[i].norm()) for i in range(n)},
            "band_share": {labels[i]: float((parts[i] @ band[key]).square().sum() / parts[i].square().sum().clamp_min(1e-30))
                           for i in range(n)},
            "cos": {f"{labels[i]}|{labels[l]}": float((parts[i] * parts[l]).sum() / (parts[i].norm() * parts[l].norm()).clamp_min(1e-30))
                    for i in range(n) for l in range(i + 1, n)}}
    if rank == 0:
        result = {"item": args.item, "endpoints": args.endpoints, "labels": labels, "base_validation": base,
                  "slope_val": dict(zip(labels, a_val)), "slope_batch": dict(zip(labels, a_batch)),
                  "gram_Q": {f"{labels[i]}|{labels[j]}": Q[i][j] for i in range(n) for j in range(n)},
                  "cos_total": {f"{labels[i]}|{labels[l]}": float(D[i] @ D[l]) / max(1e-30, float(D[i].norm() * D[l].norm()))
                                for i in range(n) for l in range(i + 1, n)},
                  "lines": lines, "span_optimum": span, "projected_top": projected,
                  "ritz_top": ritz[-args.project:].tolist(), "per_matrix": per_matrix,
                  "grad": {"val_norm": float(g_val.norm()), "batch_norm": float(g_batch.norm()),
                           "cos_val_batch": float(g_val @ g_batch) / float(g_val.norm() * g_batch.norm())},
                  "args": {k_: str(v) for k_, v in vars(args).items()}, "seconds": time.time() - started}
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        for label in labels:
            li = lines[label]
            print(f"{label:8s} |D| {li['norm']:.3f} a_val {result['slope_val'][label]:+.5f} a_batch {result['slope_batch'][label]:+.5f} "
                  f"Q {li['curvature_gn']:.5f} fd {li['curvature_fd']} c*_model {li['c_star_model']}", flush=True)
            print("   true ", " ".join(f"{c:g}:{v:+.4f}" for c, v in zip(cs, li["true"])), flush=True)
            print("   model", " ".join(f"{c:g}:{v:+.4f}" for c, v in zip(cs, li["model_val"])), flush=True)
            print("   top-removed", json.dumps(projected[label]), flush=True)
        print("span", json.dumps(span), flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

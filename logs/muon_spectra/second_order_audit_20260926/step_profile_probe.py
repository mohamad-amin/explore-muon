"""How does each optimizer's actual step meet the curvature at its own state? (2026-09-28 decision, 10:20 CDT)

A state is co-adapted to its optimizer (principle 4), so the only unbiased per-state comparison across optimizers is
each optimizer's own step at its own state. At a kept state s whose next weights W_{s+1} were kept, on 1 GPU:
  - G: the exact GN of the mean token loss on --curvature-sequences held-out validation sequences (FP32), hidden
    matrices only; a --krylov-step Lanczos run from a fixed random start gives its top Ritz pairs
  - directions: actual (W_{s+1} - W_s, the run's own step with its weight decay), momentum (the checkpoint's buffer M),
    gradient (-g_s, held-out), and three maps of M: muon polar(M), pd polar(M R) R (R: alpha 1/2 from C) and pdtop
    (R's eigenvalues clamped at 1: only above-mean input directions suppressed; added 2026-09-28 20:0x CDT)
  - per direction, on held-out training sequences beyond every 1x budget: slope a and exact GN curvature q (set 1),
    c* = -a/q, quality a^2/2q, the Rayleigh quotient q/|d|^2 over the top Ritz value, and the fractions of |d|^2 and of
    d.G d (curvature set) in the top 1, 4 and 16 Ritz vectors
  - the gradient at W_{s+1} against the gradient at W_s (same held-out sequences, set 2), along each of the top 16 Ritz
    vectors and in the rest: a sign flip along the stiffest directions is the edge-of-stability oscillation

usage: step_profile_probe.py OUT_JSON ARM_DIR:STEP [--curvature-sequences 128] [--krylov 48]
"""
import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream, token_views  # noqa: E402
from one_step_gn import CURV, Flat, GaussNewton, lanczos  # noqa: E402

HELD_OUT_TRAIN = 2_500_000_000
TOPS = (1, 4, 16)


def polar(x):
    u, _, vh = torch.linalg.svd(x.float(), full_matrices=False)
    return u @ vh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--curvature-sequences", type=int, default=128)
    parser.add_argument("--held-sequences", type=int, default=128)
    parser.add_argument("--krylov", type=int, default=48)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    started = time.time()
    arm, step = args.item.rsplit(":", 1)
    step = int(step)
    kept = Path(arm) / "scientific" / "kept"
    model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
    config = saved["config"]
    T = model.config.seq_len
    layers = P.hidden_linears(model)
    names = list(layers)
    keys = P.parameter_keys(model, names)
    named = dict(model.named_parameters())
    body_keys = [k for k in named if k.startswith("blocks.") and named[k].ndim == 2]
    body_indices = saved["optimizer"]["param_groups"][0]["params"]
    by_key = {k: saved["optimizer"]["state"][i]["momentum_buffer"].to(device).float() for k, i in zip(body_keys, body_indices)}
    momentum = OrderedDict((n, by_key[k]) for n, k in zip(names, keys))
    del saved
    nxt, _ = P.load_checkpoint(kept / f"step{step + 1:06d}_weights.pt", device, config["model"])
    next_layers = P.hidden_linears(nxt)
    actual = OrderedDict((n, (next_layers[n].weight - layers[n].weight).detach().float()) for n in names)
    del nxt, next_layers
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for n in names:     # the GN product's reverse pass and the gradients need the hidden weights
        layers[n].weight.requires_grad_(True)
    flat = Flat(OrderedDict((n, layers[n].weight) for n in names))
    val = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    train = TokenStream(str(REPO / config["train_pattern"]))
    curv = [val.batch(CURV + first * T, 8, T, device) for first in range(0, args.curvature_sequences, 8)]
    held = []
    for j in range(2):
        first = HELD_OUT_TRAIN + j * args.held_sequences * T
        hx, hy = token_views(train.device_tokens(first, args.held_sequences * T + 1, device), 0, args.held_sequences * T, T)
        held.append([(hx[i:i + 8], hy[i:i + 8]) for i in range(0, args.held_sequences, 8)])

    # input root for the PD map (alpha 1/2, positions >= 1)
    recorder = P.Recorder(model)
    sums, count = {n: 0 for n in names}, 0
    with torch.no_grad():
        for x, _ in curv:
            model(x)
            for n in names:
                xi = recorder.inputs[n][:, 1:].double()
                sums[n] = sums[n] + xi.reshape(-1, xi.shape[-1]).T @ xi.reshape(-1, xi.shape[-1])
    recorder.remove()
    R, R_top = {}, {}
    for n in names:
        values, vectors = torch.linalg.eigh(sums[n])
        unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
        factors = (unit + 1e-3).pow(-0.5)
        R[n] = ((vectors * factors) @ vectors.T).float()
        R_top[n] = ((vectors * factors.clamp_max(1.0)) @ vectors.T).float()     # PD-top: only above-mean directions suppressed
    del sums

    def gradient(batches):
        out = OrderedDict((n, torch.zeros_like(layers[n].weight)) for n in names)
        for x, y in batches:
            loss = P.token_losses(model(x), y).mean(1).sum() / args.held_sequences
            for n, g in zip(names, torch.autograd.grad(loss, [layers[n].weight for n in names])):
                out[n] += g.float()
        return out
    g_now = gradient(held[1])

    # top of the GN spectrum
    op = GaussNewton(model, names, curv, flat)
    start = torch.randn(sum(flat.sizes), generator=torch.Generator().manual_seed(0)).to(device)
    V, Tk = lanczos(op, start, args.krylov)
    theta, Y = torch.linalg.eigh(Tk)
    order = theta.argsort(descending=True)
    theta, Y = theta[order], Y[:, order]
    ritz = (Y[:, :max(TOPS)].T.float().to(device) @ V)          # top Ritz vectors, (16, n)
    del V
    top = float(theta[0])
    print(json.dumps({"stage": "lanczos", "ritz_top8": [round(float(t), 3) for t in theta[:8]],
                      "seconds": round(time.time() - started)}), flush=True)

    directions = OrderedDict()
    directions["actual"] = actual
    directions["momentum"] = OrderedDict((n, -m) for n, m in momentum.items())
    directions["gradient"] = OrderedDict((n, -g) for n, g in g_now.items())
    directions["muon"] = OrderedDict((n, -polar(m)) for n, m in momentum.items())
    directions["pd"] = OrderedDict((n, -(polar(m @ R[n]) @ R[n])) for n, m in momentum.items())
    directions["pdtop"] = OrderedDict((n, -(polar(m @ R_top[n]) @ R_top[n])) for n, m in momentum.items())
    result = {"item": args.item, "step": step, "batch_tokens": config["batch_tokens"], "ritz": [float(t) for t in theta],
              "directions": {}}
    for label, d in directions.items():
        dk = OrderedDict((k, d[n]) for n, k in zip(names, keys))
        a = q = 0.0
        for x, y in held[0]:
            t = P.directional_terms(model, x, y, dk)
            a += float(t["first"].sum())
            q += float(t["q"].sum())
        a, q = a / args.held_sequences, q / args.held_sequences
        v = flat.flat(d)
        norm2 = float(v @ v)
        gv = op(v)
        q_curv = float(v @ gv)
        proj = ritz @ v
        entry = {"norm": math.sqrt(norm2), "slope": a, "curvature": q, "c_star": -a / q if q > 0 else float("nan"),
                 "quality": a * a / (2 * q) if q > 0 else float("nan"), "rayleigh_over_top": q_curv / norm2 / top,
                 "curvature_curv_set": q_curv}
        for k in TOPS:
            entry[f"energy_top{k}"] = float(proj[:k].pow(2).sum() / norm2)
            entry[f"curvature_top{k}"] = float((theta[:k].float().to(device) * proj[:k].pow(2)).sum() / max(q_curv, 1e-30))
        result["directions"][label] = entry
        print(json.dumps({"direction": label, **{kk: round(vv, 5) for kk, vv in entry.items()}}), flush=True)

    # the gradient after the run's step, along the stiffest directions and in the rest
    with torch.no_grad():
        for n in names:
            layers[n].weight.add_(actual[n])
    g_next = gradient(held[1])
    with torch.no_grad():
        for n in names:
            layers[n].weight.sub_(actual[n])
    a0, a1 = flat.flat(g_now), flat.flat(g_next)
    p0, p1 = ritz @ a0, ritz @ a1
    rest0, rest1 = a0 - ritz.T @ p0, a1 - ritz.T @ p1
    result["gradient_pair"] = {
        "top_projection_now": p0.tolist(), "top_projection_next": p1.tolist(),
        "top_energy_fraction_now": float(p0.pow(2).sum() / (a0 @ a0)), "top_energy_fraction_next": float(p1.pow(2).sum() / (a1 @ a1)),
        "cos_rest": float(rest0 @ rest1 / (rest0.norm() * rest1.norm())), "cos_all": float(a0 @ a1 / (a0.norm() * a1.norm())),
        "norm_now": float(a0.norm()), "norm_next": float(a1.norm())}
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    gp = result["gradient_pair"]
    print(json.dumps({"flips_top4": [round(x / y, 2) if abs(y) > 0 else None for x, y in zip(gp["top_projection_next"][:4], gp["top_projection_now"][:4])],
                      "cos_rest": round(gp["cos_rest"], 3), "seconds": round(result["seconds"])}), flush=True)


if __name__ == "__main__":
    main()

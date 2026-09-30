"""Persistent vs oscillating parts of the gradient, from two consecutive iterates.

At a kept step t (weights W_t, momentum M_t) and the kept next-step weights W_{t+1}, the full-batch gradient on the same
held-out sequences at both points: g_t and g_{t+1} (sampling noise cancels in their difference). With a period-2 limit
cycle in the stiff modes, p = (g_t + g_{t+1}) / 2 is the persistent ("river") part and tau = (g_t - g_{t+1}) / 2 the
oscillating ("hill") part. Binned by the exact per-pair GN diagonal h in each matrix's Kronecker frame (as in
transport_test.py) and projected on the top exact GN eigenvectors (random-start Lanczos at W_t): energies of g_t,
g_{t+1}, p, tau, the saved momentum M_t and the applied update dW = W_{t+1} - W_t, and their cross terms.

usage: river_hill.py OUT_JSON ARM_DIR:STEP [--sequences 8192]
"""
import argparse
import json
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
from research.adamw_spectra.data import TokenStream  # noqa: E402
from one_step_gn import BASE, CURV, Flat, GaussNewton, lanczos, statistics  # noqa: E402
from transport_test import frame_bins, frame_products  # noqa: E402

REF = BASE + 100000 * 512     # held out from the gradient, curvature, scoring and transport sequences


def full_gradient(model, names, stream, sequences, T, device, micro=16):
    layers = P.hidden_linears(model)
    weights = [layers[n].weight for n in names]
    total = OrderedDict((n, torch.zeros_like(layers[n].weight)) for n in names)
    for first in range(0, sequences, micro):
        x, y = stream.batch(REF + first * T, micro, T, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        for n, g in zip(names, torch.autograd.grad(loss * (micro / sequences), weights)):
            total[n] += g.float()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--sequences", type=int, default=8192)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    parser.add_argument("--ritz-steps", type=int, default=32)
    parser.add_argument("--ritz", type=int, default=16)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
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
    body = [p for name, p in model.named_parameters() if name.startswith("blocks.") and p.ndim == 2]
    if any(p is not layers[n].weight for p, n in zip(body, names)):
        raise RuntimeError("optimizer body order differs from the hidden-matrix order")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for n in names:
        layers[n].weight.requires_grad_(True)
    M = OrderedDict((n, saved["optimizer"]["state"][i]["momentum_buffer"].to(device).float()) for i, n in enumerate(names))
    del saved
    nxt, _ = P.load_checkpoint(kept / f"step{step + 1:06d}_weights.pt", device, config["model"])
    next_layers = P.hidden_linears(nxt)
    dW = OrderedDict((n, next_layers[n].weight.detach().float() - layers[n].weight.detach().float()) for n in names)
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    curv = [stream.batch(CURV + first * T, 8, T, device) for first in range(0, args.curvature_sequences, 8)]
    stats = statistics(model, names, curv, T, device)
    edges, frames = frame_bins(stats, names)
    flat = Flat(OrderedDict((n, layers[n].weight) for n in names))
    op = GaussNewton(model, names, curv, flat)
    gen = torch.Generator(device=device).manual_seed(0)
    V, Tk = lanczos(op, torch.randn(sum(flat.sizes), device=device, generator=gen), args.ritz_steps)
    theta, Y = torch.linalg.eigh(Tk)
    order = torch.argsort(theta, descending=True)[:args.ritz]
    probes = (V.T @ Y[:, order].to(V.device, torch.float32)).T.contiguous()
    del V, op
    g_t = full_gradient(model, names, stream, args.sequences, T, device)
    for parameter in nxt.parameters():
        parameter.requires_grad_(False)
    for n in names:
        next_layers[n].weight.requires_grad_(True)
    g_next = full_gradient(nxt, names, stream, args.sequences, T, device)
    p = OrderedDict((n, 0.5 * (g_t[n] + g_next[n])) for n in names)
    tau = OrderedDict((n, 0.5 * (g_t[n] - g_next[n])) for n in names)
    vectors = {"g_t": g_t, "g_next": g_next, "p": p, "tau": tau, "M": M, "dW": dW}
    pairs = [("g_t", "g_t"), ("g_next", "g_next"), ("g_t", "g_next"), ("p", "p"), ("tau", "tau"), ("M", "M"),
             ("M", "p"), ("M", "tau"), ("dW", "dW"), ("dW", "p"), ("dW", "tau"), ("M", "g_t")]
    totals = {f"{a}*{b}": float(sum((vectors[a][n] * vectors[b][n]).sum() for n in names)) for a, b in pairs}
    result = {"arm": arm, "step": step, "beta": config["muon_momentum"], "lr": None, "sequences": args.sequences,
              "names": names, "bins": {"edges": edges.tolist()}, "totals": totals,
              "frame_bins": frame_products(frames, names, vectors, pairs),
              "ritz": {"values": theta[order].tolist(),
                       "projections": {k: (probes @ flat.flat(v)).tolist() for k, v in vectors.items()}},
              "seconds": time.time() - started}
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"item": args.item, "p/g energy": round(totals["p*p"] / totals["g_t*g_t"], 4),
                      "tau/g energy": round(totals["tau*tau"] / totals["g_t*g_t"], 4),
                      "cos(g_t, g_next)": round(totals["g_t*g_next"] / (totals["g_t*g_t"] * totals["g_next*g_next"]) ** 0.5, 4),
                      "cos(M, p)": round(totals["M*p"] / (totals["M*M"] * totals["p*p"]) ** 0.5, 4),
                      "cos(M, tau)": round(totals["M*tau"] / (totals["M*M"] * totals["tau*tau"]) ** 0.5, 4),
                      "cos(dW, p)": round(totals["dW*p"] / (totals["dW*dW"] * totals["p*p"]) ** 0.5, 4),
                      "cos(dW, tau)": round(totals["dW*tau"] / (totals["dW*dW"] * totals["tau*tau"]) ** 0.5, 4),
                      "seconds": round(result["seconds"])}), flush=True)


if __name__ == "__main__":
    main()

"""Where along the curvature spectrum does each optimizer's own step spend its energy, gain its slope and pay its
curvature? Spectral measures of the step, the gradient and the momentum against the exact GN (2026-09-28).

At a kept state s whose next weights W_{s+1} were kept, on 1 GPU (exact GN of the mean token loss on
--curvature-sequences held-out validation sequences, FP32, hidden matrices only):
  - for each direction D (actual step W_{s+1} - W_s; the held-out gradient -g; the checkpoint's momentum -M), a
    --krylov-step Lanczos run started at D gives Gauss quadrature nodes theta_j (Ritz values) and Ritz vectors z_j with
    D in their span, so exactly
        |D|^2 = sum_j (D.z_j)^2,   D.G D = sum_j theta_j (D.z_j)^2,   g.D = sum_j (D.z_j)(g.z_j)
    (the last because D lies in the Krylov space): D's energy, curvature and slope distributed over curvature levels.
  - g is the held-out gradient on --held-sequences training sequences beyond every 1x budget.

usage: step_spectrum_probe.py OUT_JSON ARM_DIR:STEP [--curvature-sequences 128] [--krylov 48]
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
from research.adamw_spectra.data import TokenStream, token_views  # noqa: E402
from one_step_gn import CURV, Flat, GaussNewton, lanczos  # noqa: E402

HELD_OUT_TRAIN = 2_500_000_000


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
    for n in names:
        layers[n].weight.requires_grad_(True)
    flat = Flat(OrderedDict((n, layers[n].weight) for n in names))
    val = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    train = TokenStream(str(REPO / config["train_pattern"]))
    curv = [val.batch(CURV + first * T, 8, T, device) for first in range(0, args.curvature_sequences, 8)]
    hx, hy = token_views(train.device_tokens(HELD_OUT_TRAIN, args.held_sequences * T + 1, device), 0, args.held_sequences * T, T)
    g = OrderedDict((n, torch.zeros_like(layers[n].weight)) for n in names)
    for i in range(0, args.held_sequences, 8):
        loss = P.token_losses(model(hx[i:i + 8]), hy[i:i + 8]).mean(1).sum() / args.held_sequences
        for n, gi in zip(names, torch.autograd.grad(loss, [layers[n].weight for n in names])):
            g[n] += gi.float()
    g_flat = flat.flat(g)
    op = GaussNewton(model, names, curv, flat)
    result = {"item": args.item, "step": step, "batch_tokens": config["batch_tokens"], "directions": {}}
    for label, d in (("actual", actual), ("gradient", OrderedDict((n, -v) for n, v in g.items())),
                     ("momentum", OrderedDict((n, -v) for n, v in momentum.items()))):
        v = flat.flat(d)
        V, Tk = lanczos(op, v, args.krylov)
        theta, Y = torch.linalg.eigh(Tk)
        norm = float(v.norm())
        d_on = Y[0, :] * norm                              # D.z_j (D = |D| V[0])
        g_on = Y.T @ (V @ g_flat).double().cpu()            # g.z_j
        del V
        result["directions"][label] = {"norm": norm, "theta": theta.tolist(), "energy": (d_on ** 2).tolist(),
                                       "slope": (d_on * g_on).tolist(), "curvature": (theta * d_on ** 2).tolist(),
                                       "slope_total": float(g_flat @ v), "curvature_total": float(v @ op(v))}
        e = result["directions"][label]
        print(json.dumps({"direction": label, "norm": round(norm, 4), "slope": round(sum(e["slope"]), 6),
                          "slope_check": round(e["slope_total"], 6), "curvature": round(sum(e["curvature"]), 6),
                          "curvature_check": round(e["curvature_total"], 6), "theta_max": round(max(e["theta"]), 3),
                          "seconds": round(time.time() - started)}), flush=True)
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()

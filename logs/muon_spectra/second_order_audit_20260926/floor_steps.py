"""Floor steps under one shared, curvature-matched step rule: does GN-PD's floor advantage compound? (23:41 CDT decision)

From a floor state (a kept anneal-branch checkpoint), for --steps steps on the run's own next training batches with the
momentum continuing (M <- beta M + g, beta the run's), each step moves the hidden matrices along one direction family
(embeddings, gains and head fixed; no weight decay), every direction at Muon's per-matrix norms:
  muon    -polar(M) x shape factor
  pd      -polar(M R) R, R = (C / mean eig + 1e-3)^-1/2 from the curvature sequences, refreshed every 8 steps
  gnpd    -(G + mu)^-p polar((G + mu)^-p M) per matrix (gnpd_probe.py), exact GN of all hidden matrices on the 256
          held-out curvature sequences, Lanczos --krylov, p = 1/4, mu = 1e-4 rho
Shared step rule: s_t = kappa (-a_t / q_t) along the direction D_t, with a_t the slope <grad L, D_t> on one fresh
held-out training slice and q_t the GN curvature D_t^T G D_t on an independent fresh slice (each 64 sequences, new
every step, tokens beyond every run's budget), both EMA-smoothed (--ema). No argmin over a grid, no skipping.
Logs per step: a, q, s, and for gnpd the top Ritz value of its first Lanczos run (sharpening); every --eval-every steps
the held-out loss on the EVAL sequences (the probes' scoring set).

usage: floor_steps.py OUT_JSON ARM_DIR:STEP --direction muon|pd|gnpd --kappa 2 [--steps 24]
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
from research.adamw_spectra.data import TokenStream  # noqa: E402
from one_step_gn import CURV, EVAL, Flat, GaussNewton, lanczos, muon_norm, polar  # noqa: E402
from gnpd_probe import input_roots, matrix_power  # noqa: E402
from valley_rescore import fresh_gradient  # noqa: E402

HELD_OUT_TRAIN = 2_500_000_000


def slope_and_curvature(model, batches, direction):
    first = q = 0.0
    count = 0
    for x, y in batches:
        t = P.directional_terms(model, x, y, direction)
        first += float(t["first"].sum())
        q += float(t["q"].sum())
        count += x.shape[0]
    return first / count, q / count


@torch.no_grad()
def eval_loss(model, batches):
    total = count = 0
    for x, y in batches:
        total += float(P.token_losses(model(x), y).mean(1).sum())
        count += x.shape[0]
    return total / count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--direction", required=True, choices=("muon", "pd", "gnpd"))
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--ema", type=float, default=0.7)
    parser.add_argument("--krylov", type=int, default=64)
    parser.add_argument("--power", type=float, default=0.25)
    parser.add_argument("--damping", type=float, default=1e-4)
    parser.add_argument("--slice-sequences", type=int, default=64)
    parser.add_argument("--eval-every", type=int, default=4)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    parser.add_argument("--transport", action="store_true",
                        help="direction from the curvature-transported momentum M + G D, D <- beta D + beta/(1-beta) dW "
                             "(first-order correction of the momentum's staleness; 00:27 CDT plan)")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    started = time.time()
    arm, step0 = args.item.rsplit(":", 1)
    step0 = int(step0)
    kept = Path(arm) / "scientific" / "kept"
    model, saved = P.load_checkpoint(kept / f"step{step0:06d}.pt", device)
    config = saved["config"]
    T = model.config.seq_len
    layers = P.hidden_linears(model)
    names = list(layers)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for n in names:
        layers[n].weight.requires_grad_(True)
    beta = config["muon_momentum"]
    M = OrderedDict((n, saved["optimizer"]["state"][i]["momentum_buffer"].to(device).float()) for i, n in enumerate(names))
    del saved
    flat = Flat(OrderedDict((n, layers[n].weight) for n in names))
    val = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    train = TokenStream(str(REPO / config["train_pattern"]))
    curv = [val.batch(CURV + first * T, 8, T, device) for first in range(0, args.curvature_sequences, 8)]
    evals = [val.batch(EVAL + first * T, 8, T, device) for first in range(0, 512, 8)]
    batch_sequences = config["batch_tokens"] // T
    op = GaussNewton(model, names, curv, flat) if (args.direction == "gnpd" or args.transport) else None
    displacement = OrderedDict((n, torch.zeros_like(M[n])) for n in names) if args.transport else None
    roots = None
    a_ema = q_ema = None
    rows = []
    result = {"arm": arm, "start": step0, "args": {k: str(v) for k, v in vars(args).items()}, "beta": beta, "rows": rows}
    rows.append({"k": 0, "eval_loss": eval_loss(model, evals)})
    print(json.dumps(rows[-1]), flush=True)
    for k in range(1, args.steps + 1):
        t0 = time.time()
        s = step0 + k
        g = fresh_gradient(model, names, train, (s - 1) * config["batch_tokens"], batch_sequences, T, device)
        for n in names:
            M[n].mul_(beta).add_(g[n])
        row = {"k": k, "step": s}
        source = M
        if args.transport and k > 1:     # the momentum's past gradients carried to the current weights (first order)
            gd = flat.dict(op(flat.flat(displacement)))
            source = OrderedDict((n, M[n] + gd[n]) for n in names)
            row["transport_ratio"] = math.sqrt(sum(float(gd[n].pow(2).sum()) for n in names) / sum(float(M[n].pow(2).sum()) for n in names))
        if args.direction == "muon":
            D = OrderedDict((n, -polar(source[n]) * math.sqrt(max(1.0, M[n].shape[0] / M[n].shape[1]))) for n in names)
        elif args.direction == "pd":
            if roots is None or (k - 1) % 8 == 0:
                roots = input_roots(model, names, curv)
            D = OrderedDict()
            for n in names:
                r = roots[n][0.5]
                d = polar(source[n] @ r) @ r
                D[n] = -d * (muon_norm(M[n].shape) / d.norm().clamp_min(1e-30))
        else:
            b_flat = flat.flat(source)
            V, Tk = lanczos(op, b_flat, args.krylov)
            rho = float(Tk[0, 0])
            row["rho"], row["ritz_top"] = rho, float(torch.linalg.eigvalsh(Tk)[-1])
            mu = args.damping * rho
            w = flat.dict(matrix_power(V, Tk, float(b_flat.norm()), args.power, mu))
            del V
            p_flat = flat.flat(OrderedDict((n, polar(w[n])) for n in names))
            V2, T2 = lanczos(op, p_flat, args.krylov)
            z = flat.dict(matrix_power(V2, T2, float(p_flat.norm()), args.power, mu))
            del V2
            D = OrderedDict((n, -z[n] * (muon_norm(z[n].shape) / z[n].norm().clamp_min(1e-30))) for n in names)
        base = HELD_OUT_TRAIN + k * 2 * args.slice_sequences * T
        slice_a = [train.batch(base + first * T, 8, T, device) for first in range(0, args.slice_sequences, 8)]
        slice_b = [train.batch(base + (args.slice_sequences + first) * T, 8, T, device) for first in range(0, args.slice_sequences, 8)]
        a, _ = slope_and_curvature(model, slice_a, D)
        _, q = slope_and_curvature(model, slice_b, D)
        a_ema = a if a_ema is None else args.ema * a_ema + (1 - args.ema) * a
        q_ema = q if q_ema is None else args.ema * q_ema + (1 - args.ema) * q
        scale = args.kappa * max(0.0, -a_ema) / max(q_ema, 1e-30)
        with torch.no_grad():
            for n in names:
                layers[n].weight.add_(D[n], alpha=scale)
                if displacement is not None:
                    displacement[n].mul_(beta).add_(D[n], alpha=scale * beta / (1 - beta))
        row.update({"slope": a, "curvature": q, "slope_ema": a_ema, "curvature_ema": q_ema, "scale": scale,
                    "seconds": time.time() - t0})
        if k % args.eval_every == 0 or k == args.steps:
            row["eval_loss"] = eval_loss(model, evals)
        rows.append(row)
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({key: (round(v, 6) if isinstance(v, float) else v) for key, v in row.items()}), flush=True)
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()

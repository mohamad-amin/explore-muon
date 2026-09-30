"""One-step re-score at a state (e.g. after anneal_branch.py): exact damped GN vs the optimizers' maps on large fresh
gradients, cross-fitted, with GN's Newton decrement split by curvature (premise checks of the 11:24 CDT review).

For a kept state and each input b in --inputs (fresh held-out gradients: g4M = 8192 sequences, g16M = 32768):
  directions  gd, muon, pd_a0.25, pd_a0.5, two_sided, K-FAC and EKFAC (closed forms of one_step_gn.py) and damped GN
              -(G + d rho I)^-1 b from --krylov Lanczos steps on the exact GN of all hidden matrices (d in DAMPINGS)
  scoring     cross-fitted on the two halves of the held-out scoring sequences: scale (and damping for GN) fitted on
              one half, decrease measured on the other, averaged over both directions of the fit
  stiff part  the exact Newton decrement of b inside the top-16 GN modes, sum_i <b, u_i>^2 / (2 lambda_i) (random-start
              Lanczos eigenvectors), against the damped GN decrement
  quadrature  b^T (G + mu I)^-1 b = |b|^2 sum_i w_i / (theta_i + mu) from the Lanczos run on b (w_i = Y_1i^2): the
              decrement per Ritz-value band, log-binned, at the damping GN's cross-fit picks
Gradients use BF16 autocast (the training precision); GN products FP32. One-step and local.

usage: valley_rescore.py OUT_JSON ARM_DIR:STEP [--inputs g4M,g16M,momentum_g1M] [--krylov 64]
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
from one_step_gn import (BASE, CURV, EVAL, GRAD, Flat, GaussNewton, closed_form_directions, krylov_direction,  # noqa: E402
                         lanczos, muon_norm, polar, statistics)
from one_step_blockgn import cross_fit, halves_score  # noqa: E402

G16 = BASE + 130000 * 512       # 16M-token gradients (32768 sequences), held out from every other region
DAMPINGS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)
MAPS = ("gd", "muon", "pd_a0.25", "pd_a0.5", "two_sided")
# spectrum flattening and input power: U S^p V^T in place of polar's U V^T, and PD at alpha 3/4 and 1
EXTRA_MAPS = ("muon_p0.25", "muon_p0.5", "pd_a0.5_p0.25", "pd_a0.5_p0.5", "pd_a0.75", "pd_a1")
# PD with separate pre- and post-whitening powers: -polar(g R_pre) R_post rescaled to Muon's norm (PD is pre = post = a)
SPLIT_POWERS = ((0.5, 0.0), (0.5, 0.25), (0.25, 0.0), (0.75, 0.25), (0.75, 0.0), (1.0, 0.25))


def fresh_gradient(model, names, stream, offset, sequences, T, device, micro=16):
    layers = P.hidden_linears(model)
    weights = [layers[n].weight for n in names]
    total = OrderedDict((n, torch.zeros_like(layers[n].weight)) for n in names)
    for first in range(0, sequences, micro):
        x, y = stream.batch(offset + first * T, micro, T, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        for n, g in zip(names, torch.autograd.grad(loss * (micro / sequences), weights)):
            total[n] += g.float()
    return total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--inputs", default="g4M,g16M")
    parser.add_argument("--krylov", type=int, default=64)
    parser.add_argument("--ritz-steps", type=int, default=32)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    parser.add_argument("--eval-sequences", type=int, default=512)
    parser.add_argument("--extra-maps", action="store_true", help="also score EXTRA_MAPS")
    parser.add_argument("--split-maps", action="store_true", help="also score PD with separate pre/post powers")
    parser.add_argument("--no-gn", dest="gn", action="store_false", help="skip the Lanczos GN solve and quadrature")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    started = time.time()
    arm, step = args.item.rsplit(":", 1)
    step = int(step)
    kept = Path(arm) / "scientific" / "kept"
    model, _ = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
    T = model.config.seq_len
    layers = P.hidden_linears(model)
    names = list(layers)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for n in names:
        layers[n].weight.requires_grad_(True)
    flat = Flat(OrderedDict((n, layers[n].weight) for n in names))
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    curv = [stream.batch(CURV + first * T, 8, T, device) for first in range(0, args.curvature_sequences, 8)]
    evals = [stream.batch(EVAL + first * T, 8, T, device) for first in range(0, args.eval_sequences, 8)]
    halves = [evals[:len(evals) // 2], evals[len(evals) // 2:]]
    stats = statistics(model, names, curv, T, device)
    op = GaussNewton(model, names, curv, flat)
    gen = torch.Generator(device=device).manual_seed(0)
    V, Tr = lanczos(op, torch.randn(sum(flat.sizes), device=device, generator=gen), args.ritz_steps)
    theta, Y = torch.linalg.eigh(Tr)
    order = torch.argsort(theta, descending=True)[:16]
    top_values = theta[order]
    probes = (V.T @ Y[:, order].to(V.device, torch.float32)).T.contiguous()
    del V
    base_loss = sum(float(P.token_losses(model(x), y).mean(1).sum()) for x, y in evals) / args.eval_sequences
    result = {"arm": arm, "step": step, "eval_loss": base_loss, "top16_eigenvalues": top_values.tolist(), "inputs": {}}
    print(json.dumps({"item": args.item, "stage": "setup", "eval_loss": round(base_loss, 5),
                      "top4": [round(float(x), 2) for x in top_values[:4]], "seconds": round(time.time() - started)}), flush=True)
    sizes = {"g4M": (GRAD, 8192), "g16M": (G16, 32768), "g1M": (GRAD, 2048)}
    saved_state = torch.load(kept / f"step{step:06d}.pt", map_location="cpu", weights_only=False)
    beta = saved_state["config"]["muon_momentum"]
    buffers = OrderedDict((n, saved_state["optimizer"]["state"][i]["momentum_buffer"].to(device).float())
                          for i, n in enumerate(names)) if "optimizer" in saved_state else None
    del saved_state
    for label in args.inputs.split(","):
        t0 = time.time()
        if label.startswith("momentum"):
            # the optimizer's next momentum M' = beta M + g (g: a fresh gradient of the size after the underscore)
            offset, sequences = sizes[label.split("_")[1]]
            g = fresh_gradient(model, names, stream, offset, sequences, T, device)
            b = OrderedDict((n, beta * buffers[n] + g[n]) for n in names)
        else:
            offset, sequences = sizes[label]
            b = fresh_gradient(model, names, stream, offset, sequences, T, device)
        b_flat = flat.flat(b)
        directions = closed_form_directions(b, stats)
        families = {m: {m: directions[m]} for m in MAPS + (EXTRA_MAPS if args.extra_maps else ())}
        if args.split_maps:
            for pre, post in SPLIT_POWERS:
                d_split = OrderedDict()
                for n, gn_ in b.items():
                    g_ = gn_.float()
                    d = polar(g_ @ stats[n][f"R{float(pre)}"])
                    if post > 0:
                        d = d @ stats[n][f"R{float(post)}"]
                    d_split[n] = -d * (muon_norm(g_.shape) / d.norm().clamp_min(1e-30))
                families[f"pd_pre{pre:g}_post{post:g}"] = {f"pd_pre{pre:g}_post{post:g}": d_split}
        # per-matrix Kronecker curvature: K-FAC and EKFAC (exact per-pair GN diagonal in the frame), damping cross-fitted
        families["kfac"] = {k: v for k, v in directions.items() if k.startswith("kfac_")}
        families["ekfac"] = {k: v for k, v in directions.items() if k.startswith("ekfac_")}
        if args.gn:
            V, Tk = lanczos(op, b_flat, args.krylov)
            rho = float(Tk[0, 0])
            families["gn"] = {f"gn_d{d:g}": flat.dict(krylov_direction(V, Tk, float(b_flat.norm()), args.krylov, d * rho))
                              for d in DAMPINGS}
            del V
        else:
            rho = float("nan")
        entry = {"rho": rho, "families": {}}
        for family, candidates in families.items():
            scored = {k: halves_score(model, halves, d) for k, d in candidates.items()}
            entry["families"][family] = {"cross_fit": cross_fit(scored), "raw": scored}
        # stiff part: exact Newton decrement of b inside the top-16 GN modes
        proj = probes @ b_flat
        entry["stiff16_decrement"] = float(0.5 * (proj.pow(2) / top_values.to(proj.device, torch.float32)).sum())
        if not args.gn:
            result["inputs"][label] = entry
            args.out.write_text(json.dumps(result, indent=1) + "\n")
            cf = {f: round(v["cross_fit"]["cross_fitted"] * 1e3, 3) for f, v in entry["families"].items()}
            print(json.dumps({"item": args.item, "input": label, "cross_fitted_x1e3": cf, "seconds": round(time.time() - t0)}), flush=True)
            continue
        # Lanczos quadrature of the damped decrement at the damping GN's cross-fit picked (first fit)
        pick = entry["families"]["gn"]["cross_fit"]["picks"]["fit0"]["label"]
        mu = float(pick.split("_d")[1]) * rho
        values, vectors = torch.linalg.eigh(Tk)
        weights = vectors[0].pow(2) * float(b_flat.norm()) ** 2
        contrib = 0.5 * weights / (values.clamp_min(0) + mu)
        edges = torch.logspace(-4, 3, 15, dtype=torch.float64)
        index = torch.bucketize(values.clamp_min(1e-12), edges)
        entry["quadrature"] = {"mu": mu, "total": float(contrib.sum()), "edges": edges.tolist(),
                               "by_band": torch.bincount(index, weights=contrib, minlength=len(edges) + 1).tolist(),
                               "above_top16_threshold": float(contrib[values >= float(top_values[-1])].sum()),
                               "ritz": values.tolist(), "weights": weights.tolist()}
        result["inputs"][label] = entry
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        cf = {f: round(v["cross_fit"]["cross_fitted"] * 1e3, 3) for f, v in entry["families"].items()}
        print(json.dumps({"item": args.item, "input": label, "rho": round(rho, 3), "cross_fitted_x1e3": cf,
                          "gn_pick": pick, "stiff16_x1e3": round(entry["stiff16_decrement"] * 1e3, 3),
                          "quad_total_x1e3": round(entry["quadrature"]["total"] * 1e3, 3),
                          "quad_above_top16_x1e3": round(entry["quadrature"]["above_top16_threshold"] * 1e3, 3),
                          "seconds": round(time.time() - t0)}), flush=True)
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()

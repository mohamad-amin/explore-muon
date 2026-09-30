"""Why does GN's direction at the baseline's step norms raise the loss in training? One-step probe of the curvature
estimate behind it (after the 15:21 CDT normalized-GN decision; smoke: +0.004 per step at Muon's step-500 state).

At a kept state s, the input is the run's next momentum M' = beta M + g, with g the gradient on the run's own batch s+1
(BF16, as in training). For each curvature set
  batch64          the 64 sequences of that batch the 2-GPU Newton trainers use (in-sample for g)
  fresh64/256/1024 fresh held-out sequences (the CURV region)
one Lanczos run of --krylov steps from M' on the exact GN of the hidden matrices gives x(k, d) = -(G_set + d rho I)^-1 M'
for k in KRYLOV and d in DAMPINGS. Each x is also rescaled per matrix to the normalized-GN trainer's step,
s = lr |NS(M')| sqrt(max(1, m/n)) x / |x| per matrix (data-norm runs: lr sqrt(min(m, n)) sqrt(max(1, m/n))). For x, s and
the references (Muon's own step, and M' at Muon's per-matrix norms):
  held_out   first-order term and exact GN curvature q on the two halves of 512 held-out sequences
  own_q      the curvature the set itself assigns to the same direction (x: y^T T y; s: one more product)
  losses     the true held-out loss at c s, c in SCALES (s and references only; hidden matrices moved, the rest fixed)
If small sets underestimate the curvature along their own step (held-out q / own q >> 1), the inverse is amplifying
estimation noise, and a better (averaged) curvature estimate is what a GN-shaped step needs.

The references also include the step the run actually took (W_{s+1} - W_s, hidden matrices) when it was kept.

usage: normgn_probe.py OUT_JSON ARM_DIR:STEP [--sets batch64,fresh64,fresh256,fresh1024 | ""] [--krylov 64]
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
from research.adamw_spectra.distributed import partition_sequences  # noqa: E402
from research.adamw_spectra.muon import newton_schulz, ns_schedule  # noqa: E402
from research.adamw_spectra.train import learning_rate, token_budget  # noqa: E402
from one_step_gn import CURV, EVAL, Flat, GaussNewton, lanczos, loss_along  # noqa: E402
from one_step_blockgn import halves_score  # noqa: E402
from valley_rescore import fresh_gradient  # noqa: E402

KRYLOV = (16, 64)
DAMPINGS = (1e-1, 1e-2, 1e-3)
SCALES = (0.25, 0.5, 1.0, 2.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--sets", default="batch64,fresh64,fresh256,fresh1024")
    parser.add_argument("--krylov", type=int, default=max(KRYLOV), help="Lanczos steps per set")
    parser.add_argument("--krylov-list", default=",".join(str(k) for k in KRYLOV), help="truncations evaluated")
    parser.add_argument("--eval-sequences", type=int, default=512)
    parser.add_argument("--dampings", default=",".join(f"{d:g}" for d in DAMPINGS), help="relative to rho")
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
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for n in names:
        layers[n].weight.requires_grad_(True)
    flat = Flat(OrderedDict((n, layers[n].weight) for n in names))
    beta = config["muon_momentum"]
    momentum = OrderedDict((n, saved["optimizer"]["state"][i]["momentum_buffer"].to(device).float())
                           for i, n in enumerate(names))
    del saved
    # the run's own next batch (step s+1) and its LR, as newton_train.py takes them
    budget = token_budget(config, sum(p.numel() for p in model.parameters()), T)
    tokens_before = step * config["batch_tokens"]
    lr = learning_rate(config, step + 1, tokens_before, budget)
    train = TokenStream(str(REPO / config["train_pattern"]))
    batch_sequences = config["batch_tokens"] // T
    g = fresh_gradient(model, names, train, tokens_before, batch_sequences, T, device)
    b = OrderedDict((n, beta * momentum[n] + g[n]) for n in names)
    b_flat = flat.flat(b)
    # the normalized-GN trainer's per-matrix step norms, and Muon's own step
    schedule = ns_schedule(config)
    targets, muon_step = OrderedDict(), OrderedDict()
    for n, m in b.items():
        rows, cols = m.shape
        scale = math.sqrt(max(1.0, rows / cols))
        polar = newton_schulz(m[None], schedule)[0].float()
        muon_step[n] = -lr * scale * polar
        targets[n] = lr * scale * (math.sqrt(min(rows, cols)) if config.get("data_norm_alpha", 0) > 0 else float(polar.norm()))

    def normalized(d):
        return OrderedDict((n, d[n] * (targets[n] / d[n].norm().clamp_min(1e-30))) for n in names)

    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    evals = [stream.batch(EVAL + first * T, 8, T, device) for first in range(0, args.eval_sequences, 8)]
    halves = [evals[:len(evals) // 2], evals[len(evals) // 2:]]
    base_loss = sum(float(P.token_losses(model(x), y).mean(1).sum()) for x, y in evals) / args.eval_sequences

    def held_out(d):
        (f0, q0), (f1, q1) = halves_score(model, halves, d)
        return {"first": [f0, f1], "q": [q0, q1]}

    def cosine(d, ref):
        num = sum(float((d[n] * ref[n]).sum()) for n in names)
        return num / math.sqrt(sum(float(d[n].pow(2).sum()) for n in names) * sum(float(ref[n].pow(2).sum()) for n in names))

    result = {"arm": arm, "step": step, "lr": lr, "beta": beta, "seq_len": T, "base_loss": base_loss,
              "batch_sequences": batch_sequences, "references": {}, "sets": {}}
    references = OrderedDict(muon=muon_step, momentum_at_muon_norms=normalized(OrderedDict((n, -b[n]) for n in names)))
    next_path = kept / f"step{step + 1:06d}_weights.pt"
    if next_path.exists():      # the step the run actually took from this state (hidden matrices, its own decay included)
        nxt, _ = P.load_checkpoint(next_path, device, config["model"])
        next_layers = P.hidden_linears(nxt)
        references["actual"] = OrderedDict((n, (next_layers[n].weight - layers[n].weight).detach().float()) for n in names)
        del nxt, next_layers
    for label, d in references.items():
        result["references"][label] = {"held_out": held_out(d), "losses": loss_along(model, evals, d, SCALES),
                                       "cos_muon": cosine(d, muon_step)}
    print(json.dumps({"item": args.item, "stage": "references", "lr": lr, "T": T, "base": round(base_loss, 5),
                      "losses": {k: [round(v - base_loss, 5) for v in r["losses"]] for k, r in result["references"].items()},
                      "seconds": round(time.time() - started)}), flush=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    for label in filter(None, args.sets.split(",")):
        t0 = time.time()
        if label == "batch64":      # the first 32 sequences of each of the two ranks' partitions, as in newton_train.py
            curv = []
            for rank in range(2):
                first, _ = partition_sequences(batch_sequences, 2, rank)
                curv += [train.batch(tokens_before + (first + i) * T, 8, T, device) for i in range(0, 32, 8)]
        else:
            count = int(label.removeprefix("fresh"))
            curv = [stream.batch(CURV + first * T, 8, T, device) for first in range(0, count, 8)]
        op = GaussNewton(model, names, curv, flat)
        V, Tk = lanczos(op, b_flat, args.krylov)
        rho = float(Tk[0, 0])
        entry = {"rho": rho, "ritz": torch.linalg.eigvalsh(Tk).tolist(), "directions": {}}
        for k in (int(k) for k in args.krylov_list.split(",")):
            if k > args.krylov:
                continue
            for damping in (float(d) for d in args.dampings.split(",")):
                rhs = torch.zeros(k, dtype=torch.float64)
                rhs[0] = float(b_flat.norm())
                y = torch.linalg.solve(Tk[:k, :k] + damping * rho * torch.eye(k, dtype=torch.float64), rhs)
                x_flat = -(V[:k].T @ y.to(V.device, torch.float32))
                x = flat.dict(x_flat)
                s = normalized(x)
                s_flat = flat.flat(s)
                entry["directions"][f"k{k}_d{damping:g}"] = {
                    "x": {"own_first": float(b_flat @ x_flat), "own_q": float(y @ Tk[:k, :k] @ y), "held_out": held_out(x)},
                    "s": {"own_q": float(s_flat @ op(s_flat)), "held_out": held_out(s), "cos_muon": cosine(s, muon_step),
                          "losses": loss_along(model, evals, s, SCALES)}}
            # trust region at the trainer's radius: |x(mu)| = r (global), mu > -theta_min, from the Ritz decomposition of T_k.
            # The linear term is the gradient-scale average (1 - beta) M', not the sum-convention M' (review, 16:38 CDT)
            radius = math.sqrt(sum(t * t for t in targets.values()))
            theta, Y = torch.linalg.eigh(Tk[:k, :k])
            b_norm = (1 - beta) * float(b_flat.norm())
            w = Y[0] * b_norm
            lo, hi = -float(theta[0]) + 1e-12, float(theta[-1]) + b_norm / radius
            for _ in range(200):
                mu = 0.5 * (lo + hi)
                lo, hi = (mu, hi) if float((w / (theta + mu)).pow(2).sum().sqrt()) > radius else (lo, mu)
            y = -(Y @ (w / (theta + mu)))
            x_flat = V[:k].T @ y.to(V.device, torch.float32)
            x = flat.dict(x_flat)
            s = normalized(x)
            entry["directions"][f"k{k}_tr"] = {
                "mu": mu, "mu_over_rho": mu / rho, "radius": radius,
                "x": {"own_first": float(b_flat @ x_flat), "own_q": float(y @ Tk[:k, :k] @ y), "held_out": held_out(x),
                      "cos_muon": cosine(x, muon_step), "losses": loss_along(model, evals, x, SCALES)},
                "s": {"own_q": float(flat.flat(s) @ op(flat.flat(s))), "held_out": held_out(s), "cos_muon": cosine(s, muon_step),
                      "losses": loss_along(model, evals, s, SCALES)}}
        del V
        result["sets"][label] = entry
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        brief = {key: {"s_loss@1": round(v["s"]["losses"][SCALES.index(1.0)] - base_loss, 5),
                       "s_q_ratio": round(sum(v["s"]["held_out"]["q"]) / 2 / max(v["s"]["own_q"], 1e-30), 2),
                       "x_q_ratio": round(sum(v["x"]["held_out"]["q"]) / 2 / max(v["x"]["own_q"], 1e-30), 2)}
                 for key, v in entry["directions"].items()}
        print(json.dumps({"item": args.item, "set": label, "rho": round(rho, 3), "directions": brief,
                          "seconds": round(time.time() - t0)}), flush=True)
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()

"""Per-input-direction secant step multiplier of the optimizer's actual step (2026-09-29 01:03 CDT).

At a kept state N with next weights W_{N+1}, the actual step is D = W_{N+1} - W_N (every matrix). For each hidden matrix,
in the input eigenbasis V of W_N (C from held-out sequences), and in each log-u bin b of input directions:
  slope     a_b = <mu_N, D>_b                  (first-order loss change of the bin's part of the step; < 0 for descent)
  curvature q_b = <mu_{N+1} - mu_N, D>_b       (secant: the change of the bin's gradient along the bin's step, which
                                                includes every other matrix's step, i.e. the collective response)
  c*_b = -a_b / q_b                            (the multiple of the bin's step that the secant model says is optimal;
                                                1 = right length, < 1 = overshoot, > 1 = undershoot)
mu_N and mu_{N+1} are mean gradients over the same K micro-batches (fresh validation sequences, BF16 autocast); the inner
products with the fixed D are unbiased. Pooled over matrices, and per matrix kind. Also the share of the step's squared
norm and of the total slope in each bin.

Caveat (review 21:48): in the tail the all-advanced mu_{N+1} may leak through a rotated input basis; the high-variance
bins (u >= 1) are robust.

usage: secant_probe.py OUT_JSON ARM_DIR:STEP [--micro 32] [--micro-batches 48]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402
from one_step_gn import CURV, GRAD  # noqa: E402

EDGES = [10 ** (k / 2) for k in range(-10, 5)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--micro", type=int, default=32)
    parser.add_argument("--micro-batches", type=int, default=48)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    started = time.time()
    arm, step = args.item.rsplit(":", 1)
    step = int(step)
    kept = Path(arm) / "scientific" / "kept"
    model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
    config = saved["config"]
    del saved
    nxt, _ = P.load_checkpoint(kept / f"step{step + 1:06d}_weights.pt", device, config["model"])
    T = model.config.seq_len
    layers = P.hidden_linears(model)
    next_layers = P.hidden_linears(nxt)
    names = list(layers)
    steps = {n: (next_layers[n].weight - layers[n].weight).detach().float() for n in names}
    for m in (model, nxt):
        for parameter in m.parameters():
            parameter.requires_grad_(False)
    for n in names:
        layers[n].weight.requires_grad_(True)
        next_layers[n].weight.requires_grad_(True)
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))

    recorder = P.Recorder(model)
    sums = {n: 0 for n in names}
    with torch.no_grad():
        for first in range(0, args.curvature_sequences, 8):
            x, _ = stream.batch(CURV + first * T, 8, T, device)
            model(x)
            for n in names:
                xi = recorder.inputs[n][:, 1:].double()
                xi = xi.reshape(-1, xi.shape[-1])
                sums[n] = sums[n] + xi.T @ xi
    recorder.remove()
    V, U = {}, {}
    for n in names:
        values, vectors = torch.linalg.eigh(sums[n])
        values = values.clamp_min(0)
        V[n] = vectors.float()
        U[n] = (values / values.mean().clamp_min(1e-30)).float()
    del sums

    mean_a = {n: torch.zeros_like(layers[n].weight) for n in names}
    mean_b = {n: torch.zeros_like(layers[n].weight) for n in names}
    K = args.micro_batches
    for k in range(K):
        x, y = stream.batch(GRAD + k * args.micro * T, args.micro, T, device)
        for m, lay, acc in ((model, layers, mean_a), (nxt, next_layers, mean_b)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(x, y)
            for n, g in zip(names, torch.autograd.grad(loss, [lay[n].weight for n in names])):
                acc[n] += g.float() / K

    cols = {}
    for n in names:
        d = steps[n] @ V[n]
        ga = mean_a[n] @ V[n]
        gb = mean_b[n] @ V[n]
        cols[n] = {"a": (ga * d).sum(0), "q": ((gb - ga) * d).sum(0), "d2": d.pow(2).sum(0)}

    def pooled(masks):
        a = sum(float(cols[n]["a"][m].sum()) for n, m in masks.items())
        q = sum(float(cols[n]["q"][m].sum()) for n, m in masks.items())
        d2 = sum(float(cols[n]["d2"][m].sum()) for n, m in masks.items())
        count = sum(int(m.sum()) for m in masks.values())
        if count == 0:
            return None
        return {"count": count, "slope": a, "curvature": q, "step_sq": d2, "c_star": -a / q if q != 0 else None}

    kinds = sorted({n.split(".")[1] for n in names})
    total = pooled({n: torch.ones_like(U[n], dtype=torch.bool) for n in names})
    result = {"arm": arm, "step": step, "micro_batches": K, "all": total, "bins": [], "top_u_ge_1": {}, "tail_u_lt_1": {}}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        row = pooled({n: (U[n] >= lo) & (U[n] < hi) for n in names})
        if row:
            row.update({"u_lo": lo, "u_hi": hi, "slope_share": row["slope"] / total["slope"], "step_share": row["step_sq"] / total["step_sq"]})
            result["bins"].append(row)
    for kind in kinds + ["all"]:
        sel = names if kind == "all" else [n for n in names if n.endswith("." + kind)]
        result["top_u_ge_1"][kind] = pooled({n: U[n] >= 1 for n in sel})
        result["tail_u_lt_1"][kind] = pooled({n: U[n] < 1 for n in sel})
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"item": args.item, "c_star_all": round(total["c_star"], 3),
                      "c_star_top": {k: round(v["c_star"], 3) for k, v in result["top_u_ge_1"].items() if v and v["c_star"]},
                      "c_star_tail": {k: round(v["c_star"], 3) for k, v in result["tail_u_lt_1"].items() if v and v["c_star"]},
                      "seconds": round(result["seconds"])}), flush=True)
    for row in result["bins"]:
        cs = row["c_star"]
        print(f"  u [{row['u_lo']:.0e}, {row['u_hi']:.0e}) n {row['count']:5d}  c* {cs if cs is None else round(cs, 3)}  "
              f"slope share {row['slope_share']:+.3f}  step share {row['step_share']:.3f}", flush=True)


if __name__ == "__main__":
    main()

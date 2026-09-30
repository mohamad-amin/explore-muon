"""Which coupling flips the gradient? Advance subsets of matrices to W_{N+1} and measure one probed matrix's
one-step response per input eigendirection (2026-09-28 22:46 CDT; follows direction_drift_self.py).

For each probed hidden matrix n (blocks --blocks), with everything else at W_N, the matrices advanced are:
  self    n only (n's input basis stays exact)
  block   the six matrices of n's block (q, k, v, o, up, down)
  kind    n's kind in all eight blocks
  others  every matrix except n
  all     every matrix
Only "self" keeps n's input distribution; the other modes can leak through a rotated basis in the tail, so read them in
the high-variance input directions (u >= 1), where the signal is 1e4-1e5x the tail's.

Per mode: K micro-batch gradients of n (same sequences) before and after, projected on V_N; pooled over probed matrices
(and per kind) in u >= 1: one-step autocorrelation of the mean gradient and the energy ratio, bias-corrected with the
micro-batch (co)variances.

usage: direction_coupling_probe.py OUT_JSON ARM_DIR:STEP [--blocks 1,4,8] [--micro 32] [--micro-batches 32]
"""
import argparse
import json
import math
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


def response(before, after, K):
    ma, mb = before.mean(0), after.mean(0)
    va, vb = before.var(0, unbiased=True), after.var(0, unbiased=True)
    cab = ((before - ma) * (after - mb)).sum(0) / (K - 1)
    return {"sa": ma.pow(2).sum(0) - va.sum(0) / K, "sb": mb.pow(2).sum(0) - vb.sum(0) / K,
            "dab": (ma * mb).sum(0) - cab.sum(0) / K}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--blocks", default="1,4,8")
    parser.add_argument("--micro", type=int, default=32)
    parser.add_argument("--micro-batches", type=int, default=32)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    parser.add_argument("--modes", default="self,block,kind,others,all")
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
    new = {n: next_layers[n].weight.detach().clone() for n in layers}
    old = {n: layers[n].weight.detach().clone() for n in layers}
    del nxt, next_layers
    blocks = [int(b) for b in args.blocks.split(",")]
    names = [n for n in layers if int(n[5:7]) in blocks]
    for parameter in model.parameters():
        parameter.requires_grad_(False)
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
    batches = [stream.batch(GRAD + k * args.micro * T, args.micro, T, device) for k in range(args.micro_batches)]
    K = args.micro_batches

    def gradients(n):
        weight = layers[n].weight
        weight.requires_grad_(True)
        cols = []
        for x, y in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            cols.append(torch.autograd.grad(loss, [weight])[0].float() @ V[n])
        weight.requires_grad_(False)
        return torch.stack(cols)

    def subset(n, mode):
        block, kind = n.split(".")
        return {"self": [n], "block": [m for m in layers if m.split(".")[0] == block],
                "kind": [m for m in layers if m.split(".")[1] == kind],
                "others": [m for m in layers if m != n], "all": list(layers)}[mode]

    modes = args.modes.split(",")
    cols = {mode: {} for mode in modes}
    for n in names:
        before = gradients(n)
        for mode in modes:
            moved = subset(n, mode)
            with torch.no_grad():
                for m in moved:
                    layers[m].weight.copy_(new[m])
            cols[mode][n] = response(before, gradients(n), K)
            with torch.no_grad():
                for m in moved:
                    layers[m].weight.copy_(old[m])
        del before

    def pooled(mode, masks):
        t = {k: 0.0 for k in ("sa", "sb", "dab")}
        count = 0
        for n, mask in masks.items():
            for k in t:
                t[k] += float(cols[mode][n][k][mask].sum())
            count += int(mask.sum())
        if count == 0:
            return None
        return {"count": count, "autocorr_one_step": t["dab"] / math.sqrt(max(t["sa"], 1e-30) * max(t["sb"], 1e-30)),
                "energy_ratio": t["sb"] / max(t["sa"], 1e-30)}

    result = {"arm": arm, "step": step, "blocks": blocks, "matrices": names, "micro_batches": K, "modes": {}}
    for mode in modes:
        entry = {"top_u_ge_1": pooled(mode, {n: U[n] >= 1 for n in names}),
                 "all": pooled(mode, {n: torch.ones_like(U[n], dtype=torch.bool) for n in names}), "bins": [], "top_by_kind": {}}
        for kind in sorted({n.split(".")[1] for n in names}):
            entry["top_by_kind"][kind] = pooled(mode, {n: U[n] >= 1 for n in names if n.endswith("." + kind)})
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            row = pooled(mode, {n: (U[n] >= lo) & (U[n] < hi) for n in names})
            if row:
                entry["bins"].append({"u_lo": lo, "u_hi": hi, **row})
        result["modes"][mode] = entry
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"item": args.item, "seconds": round(result["seconds"])}), flush=True)
    for mode in modes:
        e = result["modes"][mode]
        top = e["top_u_ge_1"]
        kinds = " ".join(f"{k}:{v['autocorr_one_step']:+.2f}/{v['energy_ratio']:.2f}" for k, v in e["top_by_kind"].items() if v)
        print(f"  {mode:6s} top (u>=1) autocorr {top['autocorr_one_step']:+.3f} energy {top['energy_ratio']:.3f} | {kinds}", flush=True)


if __name__ == "__main__":
    main()

"""Leakage-free one-step response per input eigendirection: advance one matrix at a time (2026-09-28 22:0x CDT;
review 21:48, item 2).

direction_drift_probe.py advances every matrix to W_{N+1}, so each matrix's input distribution (and its input
eigenbasis) moves too; with signal falling 1e4-1e5x from the top to the tail of the input spectrum, a 1e-5 leak of the
top through a rotated basis can double a tail column. Here, for each probed hidden matrix, only that matrix is set to
its W_{N+1} value and everything else stays at W_N. Its input distribution is then exactly W_N's (the q/k/v/up inputs
come from earlier blocks; o's input is the attention output of the unchanged q/k/v; down's input is the unchanged up's
activation), so its input basis V_N is exact for both gradients. What is measured is the matrix's self-response to its
own step (cross-matrix coupling is excluded by construction).

Per probed matrix, K micro-batch gradients (same sequences) at W_N and at the one-matrix-advanced model, projected on
V_N; per input-eigenvalue bin (pooled over probed matrices): one-step autocorrelation of the mean gradient and the
energy ratio |mu_after|^2 / |mu_N|^2 (both bias-corrected with the micro-batch (co)variances).

usage: direction_drift_self.py OUT_JSON ARM_DIR:STEP [--blocks 1,4,8] [--micro 32] [--micro-batches 32]
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--blocks", default="1,4,8")
    parser.add_argument("--micro", type=int, default=32)
    parser.add_argument("--micro-batches", type=int, default=32)
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
    blocks = [int(b) for b in args.blocks.split(",")]
    names = [n for n in layers if int(n[5:7]) in blocks]
    new_weights = {n: next_layers[n].weight.detach().clone() for n in names}
    del nxt, next_layers
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
        values = values.flip(0).clamp_min(0)
        V[n] = vectors.flip(-1).float()
        U[n] = (values / values.mean().clamp_min(1e-30)).float()
    del sums
    batches = [stream.batch(GRAD + k * args.micro * T, args.micro, T, device) for k in range(args.micro_batches)]
    K = args.micro_batches

    def gradients(n):
        weight = layers[n].weight
        weight.requires_grad_(True)
        s1 = torch.zeros_like(weight)
        s2 = torch.zeros_like(weight)
        cols = []
        for x, y in batches:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            g = torch.autograd.grad(loss, [weight])[0].float() @ V[n]
            cols.append(g)
        weight.requires_grad_(False)
        return torch.stack(cols)

    cols = {}
    for n in names:
        before = gradients(n)
        original = layers[n].weight.detach().clone()
        with torch.no_grad():
            layers[n].weight.copy_(new_weights[n])
        after = gradients(n)
        with torch.no_grad():
            layers[n].weight.copy_(original)
        ma, mb = before.mean(0), after.mean(0)
        va, vb = before.var(0, unbiased=True), after.var(0, unbiased=True)
        cab = ((before - ma) * (after - mb)).sum(0) / (K - 1)
        sa = ma.pow(2).sum(0) - va.sum(0) / K
        sb = mb.pow(2).sum(0) - vb.sum(0) / K
        dab = (ma * mb).sum(0) - cab.sum(0) / K
        cols[n] = {"u": U[n], "sa": sa, "sb": sb, "dab": dab}
        del before, after

    def pooled(mask_by_name):
        t = {k: 0.0 for k in ("sa", "sb", "dab")}
        count = 0
        for n, mask in mask_by_name.items():
            for k in t:
                t[k] += float(cols[n][k][mask].sum())
            count += int(mask.sum())
        if count == 0:
            return None
        return {"count": count, "autocorr_one_step": t["dab"] / math.sqrt(max(t["sa"], 1e-30) * max(t["sb"], 1e-30)),
                "energy_ratio": t["sb"] / max(t["sa"], 1e-30)}

    result = {"arm": arm, "step": step, "blocks": blocks, "matrices": names, "micro_batches": K, "bins": [],
              "all": pooled({n: torch.ones_like(cols[n]["u"], dtype=torch.bool) for n in names})}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        row = pooled({n: (cols[n]["u"] >= lo) & (cols[n]["u"] < hi) for n in names})
        if row:
            result["bins"].append({"u_lo": lo, "u_hi": hi, **row})
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"item": args.item, "all": result["all"], "seconds": round(result["seconds"])}), flush=True)
    for row in result["bins"]:
        print(f"  u [{row['u_lo']:.0e}, {row['u_hi']:.0e}) n {row['count']:5d}  autocorr {row['autocorr_one_step']:+.2f}  "
              f"energy ratio {row['energy_ratio']:.3g}", flush=True)


if __name__ == "__main__":
    main()

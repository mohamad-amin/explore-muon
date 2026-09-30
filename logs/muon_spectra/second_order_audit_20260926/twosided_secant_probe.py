"""Two-sided secant map: the collective step multiplier c* over input x output eigendirections (2026-09-29 01:20 CDT).

Question: which curvature information does each optimizer's step capture? A Kronecker-structured second-order step
(K-FAC-like, per matrix) would make the step multiplier uniform across the grid of input eigendirections (input second
moment C) x output eigendirections (the Gauss-Newton output factor B, from model-sampled labels). secant_probe.py showed
that PD makes c* uniform across input directions (Muon leaves its weak inputs 1.1-3.8x under-stepped at 16M); this asks
the same of the output side, for Muon (polar only), PD (input whitening + polar), S∘PD (adds SOAP's normalization in the
gradient's eigenbasis), TS (adds the GN output factor B^-beta) and SOAP∘Muon.

At a kept state N with next weights W_{N+1}, the actual step is D = W_{N+1} - W_N (every matrix). For each hidden matrix,
V = eigenbasis of C (positions >= 1, held-out sequences), U = eigenbasis of B (sampled labels, same sequences);
u = eig(C)/mean, b = eig(B)/mean. In the rotated frame, per entry (i, j) = (output direction, input direction):
  a_ij = (U^T mu_N V)_ij (U^T D V)_ij,   q_ij = (U^T (mu_{N+1} - mu_N) V)_ij (U^T D V)_ij
with mu_N, mu_{N+1} mean gradients over the same K micro-batches (fresh validation sequences, BF16). Entries are pooled
into decade bins of b (rows) and u (columns); per cell c* = -sum a / sum q (1 = the secant-optimal length, 1/2 = the edge
of stability), the step's energy share and the slope (first-order gain) share. Pooled over all matrices and per kind.

Caveats: the collective response mu_{N+1} includes every matrix's step (the relevant one for the dynamics); cells with
little energy are leakage-prone (rotated bases after the step) and noisy, so read the cells that carry the step.

usage: twosided_secant_probe.py OUT_JSON ARM_DIR:STEP [--micro 32] [--micro-batches 48]
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

EDGES = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0, float("inf")]
LABELS = ["<1e-3", "1e-3-1e-2", "1e-2-1e-1", "1e-1-1", "1-10", ">=10"]


def bin_index(values):
    index = torch.zeros_like(values, dtype=torch.long)
    for k, edge in enumerate(EDGES[1:-1], start=1):
        index += (values >= edge).long()
    return index


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

    # input factor C (positions >= 1) and sampled-label output factor B, as in frame_snr_probe.py
    gen = torch.Generator(device=device).manual_seed(1)
    recorder = P.Recorder(model)
    sums = {n: {"C": 0, "B": 0} for n in names}
    for first in range(0, args.curvature_sequences, 8):
        x, y = stream.batch(CURV + first * T, 8, T, device)
        P.gradient_passes(model, recorder, x, y, gen, draws=1)
        for n in names:
            xi = recorder.inputs[n][:, 1:].double().reshape(-1, recorder.inputs[n].shape[-1])
            es = (T * recorder.errors[n][1]).reshape(-1, recorder.errors[n][1].shape[-1]).double()
            sums[n]["C"] = sums[n]["C"] + xi.T @ xi
            sums[n]["B"] = sums[n]["B"] + es.T @ es
    recorder.remove()
    V, U, u_bin, b_bin = {}, {}, {}, {}
    for n in names:
        cv, cvec = torch.linalg.eigh(sums[n]["C"])
        bv, bvec = torch.linalg.eigh(sums[n]["B"])
        cv, bv = cv.clamp_min(0), bv.clamp_min(0)
        V[n], U[n] = cvec.float(), bvec.float()
        u_bin[n] = bin_index((cv / cv.mean().clamp_min(1e-30)).float())
        b_bin[n] = bin_index((bv / bv.mean().clamp_min(1e-30)).float())
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

    nb = len(LABELS)
    grids = {}

    def add(key, n):
        d = U[n].T @ steps[n] @ V[n]
        a = (U[n].T @ mean_a[n] @ V[n]) * d
        q = (U[n].T @ (mean_b[n] - mean_a[n]) @ V[n]) * d
        cell = (b_bin[n][:, None] * nb + u_bin[n][None, :]).reshape(-1)
        g = grids.setdefault(key, {k: torch.zeros(nb * nb, device=device, dtype=torch.float64) for k in ("a", "q", "e", "n")})
        g["a"].index_add_(0, cell, a.reshape(-1).double())
        g["q"].index_add_(0, cell, q.reshape(-1).double())
        g["e"].index_add_(0, cell, d.pow(2).reshape(-1).double())
        g["n"].index_add_(0, cell, torch.ones_like(cell, dtype=torch.float64))

    for n in names:
        add("all", n)
        add(n.split(".")[1], n)

    def summarize(g):
        a, q, e, count = (g[k].view(nb, nb) for k in ("a", "q", "e", "n"))
        cstar = torch.where(q.abs() > 0, -a / q, torch.full_like(a, float("nan")))
        return {"c_star": [[None if x != x else round(float(x), 4) for x in row] for row in cstar.tolist()],
                "energy_share": (e / e.sum()).tolist(), "slope_share": (a / a.sum()).tolist(), "count": count.tolist(),
                "c_star_total": float(-a.sum() / q.sum())}

    result = {"arm": arm, "step": step, "micro_batches": K, "rows": "output bins (B eigenvalue / mean)",
              "cols": "input bins (C eigenvalue / mean)", "labels": LABELS,
              "grids": {key: summarize(g) for key, g in grids.items()}, "seconds": time.time() - started}
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    g = result["grids"]["all"]
    print(json.dumps({"item": args.item, "c_star_total": round(g["c_star_total"], 3), "seconds": round(result["seconds"])}), flush=True)
    print("  rows = output bins " + " | ".join(LABELS) + "; cols = input bins; entries c* (energy share %)", flush=True)
    for i, label in enumerate(LABELS):
        cells = []
        for j in range(nb):
            c = g["c_star"][i][j]
            e = g["energy_share"][i][j]
            cells.append("   .      " if e < 1e-4 else f"{c:5.2f}({100 * e:4.1f})")
        print(f"  {label:>10s}  " + "  ".join(cells), flush=True)


if __name__ == "__main__":
    main()

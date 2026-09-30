"""PD with a token-weighted input statistic: one-step probe (review of 16:38 CDT, "curvature as the map's geometry").

The exact per-matrix GN is G = E_t[(d_t d_t^T) (x) (x_t x_t^T)] over tokens t, with d_t the (sampled-label) backprop
error at the matrix's output and x_t its input. K-FAC splits it as E[d d^T] (x) E[x x^T]; PD keeps only an input factor
and lets polar handle the output side. Tracing out the output side of the exact GN gives the input factor
  C_w = E_t[|d_t|^2 x_t x_t^T] / E_t[|d_t|^2],
i.e. PD's C with each token weighted by its output curvature: the first non-Kronecker correction to PD's statistic.
At a kept state, for each input b (fresh g4M, g16M, or the next momentum M' = beta M + g1M), directions
  muon                  -polar(b) x shape factor
  pd_a{a}               -polar(b R) R, R = (C/mean + 1e-3)^-a  (positions >= 1, PD's statistic)
  pdw_a{a}              the same with C_w
  pdw_self_a{a}         C_w with true-label errors instead of sampled labels (gradient-weighted, IsoMuon-like)
all rescaled to Muon's per-matrix norm, scored like valley_rescore.py: cross-fitted on the two halves of the 512 held-out
EVAL sequences (scale fitted on one half, decrease measured on the other). Same inputs and scoring sets as the
valley_rescore.py / mom_*.json runs, so shares of damped GN can be taken from those files.

usage: weighted_input_probe.py OUT_JSON ARM_DIR:STEP [--inputs g4M,momentum_g1M] [--curvature-sequences 256]
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
from one_step_gn import CURV, EVAL, GRAD, inverse_roots, muon_norm, polar  # noqa: E402
from one_step_blockgn import cross_fit, halves_score  # noqa: E402
from valley_rescore import G16, fresh_gradient  # noqa: E402

ALPHAS = (0.25, 0.5, 0.75)


def weighted_statistics(model, names, batches, T, device):
    """C (positions >= 1), C_w with sampled-label errors, C_w with true-label errors, from the curvature sequences."""
    gen = torch.Generator(device=device).manual_seed(1)
    recorder = P.Recorder(model)
    sums = {n: {"C": 0, "Cw": 0, "Cs": 0, "w": 0.0, "ws": 0.0} for n in names}
    count = 0
    for x, y in batches:
        P.gradient_passes(model, recorder, x, y, gen, draws=1)
        for n in names:
            xi = recorder.inputs[n][:, 1:].double()
            e_true = (T * recorder.errors[n][0])[:, 1:].double()
            e_samp = (T * recorder.errors[n][1])[:, 1:].double()
            flat = xi.reshape(-1, xi.shape[-1])
            w = e_samp.reshape(-1, e_samp.shape[-1]).pow(2).sum(-1)
            ws = e_true.reshape(-1, e_true.shape[-1]).pow(2).sum(-1)
            sums[n]["C"] = sums[n]["C"] + flat.T @ flat
            sums[n]["Cw"] = sums[n]["Cw"] + (flat * w[:, None]).T @ flat
            sums[n]["Cs"] = sums[n]["Cs"] + (flat * ws[:, None]).T @ flat
            sums[n]["w"] += float(w.sum())
            sums[n]["ws"] += float(ws.sum())
        count += x.shape[0] * (x.shape[1] - 1)
    recorder.remove()
    stats = {}
    for n in names:
        C = sums[n]["C"] / count
        Cw = sums[n]["Cw"] / sums[n]["w"]
        Cs = sums[n]["Cs"] / sums[n]["ws"]
        stats[n] = {"R": inverse_roots(C, ALPHAS), "Rw": inverse_roots(Cw, ALPHAS), "Rs": inverse_roots(Cs, ALPHAS)}
        # how different the statistics are: cosine of the normalized matrices and the top-eigenvector overlap
        cw = Cw / Cw.trace()
        c = C / C.trace()
        stats[n]["cos_C_Cw"] = float((cw * c).sum() / (cw.norm() * c.norm()))
        stats[n]["cos_C_Cs"] = float(((Cs / Cs.trace()) * c).sum() / ((Cs / Cs.trace()).norm() * c.norm()))
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--inputs", default="g4M,momentum_g1M")
    parser.add_argument("--curvature-sequences", type=int, default=256)
    parser.add_argument("--eval-sequences", type=int, default=512)
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
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    curv = [stream.batch(CURV + first * T, 8, T, device) for first in range(0, args.curvature_sequences, 8)]
    evals = [stream.batch(EVAL + first * T, 8, T, device) for first in range(0, args.eval_sequences, 8)]
    halves = [evals[:len(evals) // 2], evals[len(evals) // 2:]]
    stats = weighted_statistics(model, names, curv, T, device)
    result = {"arm": arm, "step": step, "cos_C_Cw": {n: stats[n]["cos_C_Cw"] for n in names},
              "cos_C_Cs": {n: stats[n]["cos_C_Cs"] for n in names}, "inputs": {}}
    print(json.dumps({"item": args.item, "stage": "statistics", "cos_C_Cw_median": sorted(result["cos_C_Cw"].values())[len(names) // 2],
                      "seconds": round(time.time() - started)}), flush=True)
    saved = torch.load(kept / f"step{step:06d}.pt", map_location="cpu", weights_only=False)
    beta = saved["config"]["muon_momentum"]
    buffers = OrderedDict((n, saved["optimizer"]["state"][i]["momentum_buffer"].to(device).float()) for i, n in enumerate(names))
    del saved
    sizes = {"g4M": (GRAD, 8192), "g16M": (G16, 32768), "g1M": (GRAD, 2048)}
    for label in args.inputs.split(","):
        t0 = time.time()
        if label.startswith("momentum"):
            offset, sequences = sizes[label.split("_")[1]]
            g = fresh_gradient(model, names, stream, offset, sequences, T, device)
            b = OrderedDict((n, beta * buffers[n] + g[n]) for n in names)
        else:
            offset, sequences = sizes[label]
            b = fresh_gradient(model, names, stream, offset, sequences, T, device)
        directions = {"muon": OrderedDict()}
        for a in ALPHAS:
            for key in ("pd", "pdw", "pdw_self"):
                directions[f"{key}_a{a:g}"] = OrderedDict()
        for n, gn_ in b.items():
            g_ = gn_.float()
            target = muon_norm(g_.shape)
            directions["muon"][n] = -polar(g_) * math.sqrt(max(1.0, g_.shape[0] / g_.shape[1]))
            for a in ALPHAS:
                for key, root in (("pd", "R"), ("pdw", "Rw"), ("pdw_self", "Rs")):
                    r = stats[n][root][a]
                    d = polar(g_ @ r) @ r
                    directions[f"{key}_a{a:g}"][n] = -d * (target / d.norm().clamp_min(1e-30))
        entry = {}
        for k, d in directions.items():
            scored = {k: halves_score(model, halves, d)}
            entry[k] = cross_fit(scored)["cross_fitted"]
        result["inputs"][label] = entry
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({"item": args.item, "input": label, "cross_fitted_x1e3": {k: round(v * 1e3, 3) for k, v in entry.items()},
                          "seconds": round(time.time() - t0)}), flush=True)
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()

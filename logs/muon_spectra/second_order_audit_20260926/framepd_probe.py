"""Which curvature structure does PD's polar map need? PD whitened by the per-pair GN diagonal in the Kronecker frame
(second review, 22:57 CDT, Q3), with a true-loss check (Q1).

Per matrix, with U, V the eigenbases of the sampled-label output factor B and the input factor C and h[i, j] a curvature
per frame pair (one_step_gn.statistics / gn_probe.Frame):
  W^-p[X] = U [(U^T X V) / (h + d mean h)^p] V^T,   framepd = -W^-p[polar(W^-p[b])], rescaled to Muon's norm
with h
  kfac     lam_B[i] lam_C[j]                       Kronecker only (the two-sided / Kronecker GN-PD geometry, control)
  ekfac    token mean of (u_i^T e~)^2 (v_j^T x)^2    per-token GN diagonal (drops cross-position terms)
  exact    T x sequence mean of (u_i^T g v_j)^2     per-sequence exact GN diagonal (keeps them)
p in (1/4, 1/2), d in DAMPINGS chosen by cross-fitting. If exact reaches most of the exact per-matrix GN-PD score
(exactblock_gnpd_probe.py) while ekfac and kfac stay at the Kronecker level, a SOAP-like estimator of per-pair
second moments of per-sequence sampled-label gradients is what PD's geometry needs, and cross-position terms are the
missing structure.

Every family's cross-fitted pick is also checked on the true held-out loss: the loss at 0.5, 1 and 2 c* on one half of
the held-out sequences, with c* fitted on the other half (hidden matrices moved, everything else fixed), averaged over
both directions of the fit. References: Muon, PD alpha 1/4 and 1/2, two-sided (closed forms of one_step_gn.py).

usage: framepd_probe.py OUT_JSON ARM_DIR:STEP [--inputs momentum_g1M,g4M] [--curvature-sequences 256]
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
from one_step_gn import CURV, EVAL, GRAD, closed_form_directions, loss_along, muon_norm, polar, statistics  # noqa: E402
from one_step_blockgn import cross_fit, halves_score  # noqa: E402
from valley_rescore import G16, fresh_gradient  # noqa: E402

POWERS = (0.25, 0.5)
DAMPINGS = (1e-1, 1e-2, 1e-3)


def frame_diagonals(model, names, batches, stats, T):
    frames = {n: P.Frame(stats[n]["U"], stats[n]["V"], stats[n]["lam_B"].double(), stats[n]["lam_C"].double(), T) for n in names}
    gen = torch.Generator(device=batches[0][0].device).manual_seed(2)
    recorder = P.Recorder(model)
    for x, y in batches:
        P.gradient_passes(model, recorder, x, y, gen, draws=1)
        for n in names:
            frames[n].add(recorder.inputs[n], recorder.errors[n][0], [recorder.errors[n][1]])
    recorder.remove()
    out = {}
    for n in names:
        s = frames[n].summary()
        out[n] = {"kfac": s["kfac"].float(), "ekfac": s["ekfac"].float(), "exact": s["exact"].float()}
    return out


def true_loss_check(model, halves, bases, direction, raw):
    """Held-out true loss change at 0.5, 1, 2 c*: c* fitted on one half, loss measured on the other, averaged."""
    rows = []
    for fit, test in ((0, 1), (1, 0)):
        f, q = raw[fit]
        c = -f / q if q > 0 else 0.0
        losses = loss_along(model, halves[test], direction, [0.5 * c, c, 2 * c])
        rows.append([l - bases[test] for l in losses])
    return {"c_star": [-raw[0][0] / raw[0][1], -raw[1][0] / raw[1][1]], "true_change_at_0.5_1_2": [0.5 * (a + b) for a, b in zip(*rows)]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--inputs", default="momentum_g1M,g4M")
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
    with torch.no_grad():
        bases = [sum(float(P.token_losses(model(x), y).mean(1).sum()) for x, y in h) / (len(h) * h[0][0].shape[0]) for h in halves]
    stats = statistics(model, names, curv, T, device)
    diag = frame_diagonals(model, names, curv, stats, T)
    saved = torch.load(kept / f"step{step:06d}.pt", map_location="cpu", weights_only=False)
    beta = saved["config"]["muon_momentum"]
    buffers = OrderedDict((n, saved["optimizer"]["state"][i]["momentum_buffer"].to(device).float()) for i, n in enumerate(names))
    del saved
    result = {"arm": arm, "step": step, "inputs": {}}
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
        closed = closed_form_directions(b, stats)
        families = {"muon": {"muon": closed["muon"]}, "pd_a0.25": {"pd_a0.25": closed["pd_a0.25"]},
                    "pd_a0.5": {"pd_a0.5": closed["pd_a0.5"]}, "two_sided": {"two_sided": closed["two_sided"]}}
        for kind in ("kfac", "ekfac", "exact"):
            for p in POWERS:
                fam = families[f"framepd_{kind}_p{p:g}"] = {}
                for d in DAMPINGS:
                    direction = OrderedDict()
                    for n in names:
                        U, V = stats[n]["U"], stats[n]["V"]
                        h = diag[n][kind].clamp_min(0)
                        w = (h + d * h.mean()).pow(-p)
                        m = b[n].float()
                        white = U @ ((U.T @ m @ V) * w) @ V.T
                        pol = polar(white)
                        z = U @ ((U.T @ pol @ V) * w) @ V.T
                        direction[n] = -z * (muon_norm(m.shape) / z.norm().clamp_min(1e-30))
                    fam[f"framepd_{kind}_p{p:g}_d{d:g}"] = direction
        entry = {"families": {}}
        for family, candidates in families.items():
            scored = {k: halves_score(model, halves, dd) for k, dd in candidates.items()}
            cf = cross_fit(scored)
            pick = cf["picks"]["fit0"]["label"]
            entry["families"][family] = {"cross_fit": cf, "raw": scored,
                                         "true_loss": true_loss_check(model, halves, bases, candidates[pick], scored[pick])}
        result["inputs"][label] = entry
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({"item": args.item, "input": label,
                          "model_x1e3": {f: round(v["cross_fit"]["cross_fitted"] * 1e3, 3) for f, v in entry["families"].items()},
                          "true_x1e3_at_0.5_1_2": {f: [round(x * 1e3, 3) for x in v["true_loss"]["true_change_at_0.5_1_2"]]
                                                   for f, v in entry["families"].items()},
                          "seconds": round(time.time() - t0)}), flush=True)
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()

"""PD in the exact Gauss-Newton geometry: one-step probe (after the 18:0x CDT finding that token-weighting C barely moves it).

PD with input power a is Muon in whitened coordinates for the Kronecker metric I (x) C: vec(D) = -(I (x) C)^-a vec(polar(M C^-a)).
Replacing I (x) C by the exact GN of all hidden matrices G gives
  GN-PD(p):  w = (G + mu I)^-p b;  P = polar(w) per matrix;  D = -(G + mu I)^-p vec(P), each matrix rescaled to Muon's norm,
which reduces to PD alpha = p when G = I (x) C. The polar step keeps unit singular values in the whitened coordinates (so
the step keeps energy in every direction, scaled by the curvature to the -p), unlike the damped Newton step (-1), which
at normalized norms sharpened the network in training (17:55 CDT). Matrix functions by Lanczos (k steps, full
reorthogonalization): one run from b (shared by every p, mu and the damped GN reference), one run from vec(P) per (p, mu).

Scored like valley_rescore.py (cross-fitted on the two halves of 512 held-out EVAL sequences; for GN and GN-PD the
damping is chosen on one half too). Inputs: fresh g4M and the next momentum M' = beta M + g1M; references Muon,
PD alpha 1/4 and 1/2 (C from the curvature sequences, positions >= 1) and damped GN.

usage: gnpd_probe.py OUT_JSON ARM_DIR:STEP [--inputs g4M,momentum_g1M] [--krylov 64] [--curvature-sequences 256]
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
from one_step_gn import CURV, EVAL, GRAD, Flat, GaussNewton, inverse_roots, lanczos, muon_norm, polar  # noqa: E402
from one_step_blockgn import cross_fit, halves_score  # noqa: E402
from valley_rescore import G16, fresh_gradient  # noqa: E402

POWERS = (0.25, 0.5)
DAMPINGS = (1e-2, 1e-3, 1e-4)          # relative to rho = b.G b / b.b
GN_DAMPINGS = (1e-1, 1e-2, 1e-3, 1e-4)


def matrix_power(V, T, start_norm, power, mu):
    """(G + mu I)^-power applied to the Lanczos start vector (norm start_norm), in the Krylov space of V, T."""
    values, vectors = torch.linalg.eigh(T)
    y = vectors @ ((values.clamp_min(0) + mu).pow(-power) * vectors[0] * start_norm)
    return V.T @ y.to(V.device, torch.float32)


def input_roots(model, names, batches):
    recorder = P.Recorder(model)
    gen = torch.Generator(device=batches[0][0].device).manual_seed(1)
    sums, count = {n: 0 for n in names}, 0
    for x, y in batches:
        P.gradient_passes(model, recorder, x, y, gen, draws=0)
        for n in names:
            xi = recorder.inputs[n][:, 1:].double()
            flat = xi.reshape(-1, xi.shape[-1])
            sums[n] = sums[n] + flat.T @ flat
        count += x.shape[0] * (x.shape[1] - 1)
    recorder.remove()
    return {n: inverse_roots(sums[n] / count, (0.25, 0.5)) for n in names}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--inputs", default="g4M,momentum_g1M")
    parser.add_argument("--krylov", type=int, default=64)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    parser.add_argument("--eval-sequences", type=int, default=512)
    parser.add_argument("--true-loss", action="store_true",
                        help="also check each family's pick on the true held-out loss at 0.5, 1, 2 c* (review, 22:57 CDT)")
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
    roots = input_roots(model, names, curv)
    op = GaussNewton(model, names, curv, flat)
    saved = torch.load(kept / f"step{step:06d}.pt", map_location="cpu", weights_only=False)
    beta = saved["config"]["muon_momentum"]
    buffers = OrderedDict((n, saved["optimizer"]["state"][i]["momentum_buffer"].to(device).float()) for i, n in enumerate(names))
    del saved
    result = {"arm": arm, "step": step, "krylov": args.krylov, "inputs": {}}
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
        b_flat = flat.flat(b)
        V, Tk = lanczos(op, b_flat, args.krylov)
        rho = float(Tk[0, 0])
        families = {"muon": {}, "pd_a0.25": {}, "pd_a0.5": {}, "gn": {}}
        muon, pd = OrderedDict(), {0.25: OrderedDict(), 0.5: OrderedDict()}
        for n, m in b.items():
            m = m.float()
            muon[n] = -polar(m) * math.sqrt(max(1.0, m.shape[0] / m.shape[1]))
            for a in (0.25, 0.5):
                r = roots[n][a]
                d = polar(m @ r) @ r
                pd[a][n] = -d * (muon_norm(m.shape) / d.norm().clamp_min(1e-30))
        families["muon"]["muon"] = muon
        families["pd_a0.25"]["pd_a0.25"] = pd[0.25]
        families["pd_a0.5"]["pd_a0.5"] = pd[0.5]
        for d in GN_DAMPINGS:
            families["gn"][f"gn_d{d:g}"] = flat.dict(-matrix_power(V, Tk, float(b_flat.norm()), 1.0, d * rho))
        for p in POWERS:
            fam = families[f"gnpd_p{p:g}"] = {}
            for d in DAMPINGS:
                mu = d * rho
                w = flat.dict(matrix_power(V, Tk, float(b_flat.norm()), p, mu))
                polar_flat = flat.flat(OrderedDict((n, polar(w[n])) for n in names))
                V2, T2 = lanczos(op, polar_flat, args.krylov)
                z = flat.dict(matrix_power(V2, T2, float(polar_flat.norm()), p, mu))
                del V2
                fam[f"gnpd_p{p:g}_d{d:g}"] = OrderedDict((n, -z[n] * (muon_norm(z[n].shape) / z[n].norm().clamp_min(1e-30))) for n in names)
        del V
        entry = {"rho": rho, "families": {}}
        if args.true_loss:
            from framepd_probe import true_loss_check
            with torch.no_grad():
                bases = [sum(float(P.token_losses(model(x), y).mean(1).sum()) for x, y in h) / (len(h) * h[0][0].shape[0]) for h in halves]
        for family, candidates in families.items():
            scored = {k: halves_score(model, halves, d) for k, d in candidates.items()}
            entry["families"][family] = {"cross_fit": cross_fit(scored), "raw": scored}
            if args.true_loss:
                pick = entry["families"][family]["cross_fit"]["picks"]["fit0"]["label"]
                entry["families"][family]["true_loss"] = true_loss_check(model, halves, bases, candidates[pick], scored[pick])
        result["inputs"][label] = entry
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        cf = {f: round(v["cross_fit"]["cross_fitted"] * 1e3, 3) for f, v in entry["families"].items()}
        gn = cf["gn"]
        print(json.dumps({"item": args.item, "input": label, "rho": round(rho, 3), "cross_fitted_x1e3": cf,
                          "share_of_gn": {f: round(v / gn, 3) for f, v in cf.items()} if gn > 0 else None,
                          "picks": {f: v["cross_fit"]["picks"]["fit0"]["label"] for f, v in entry["families"].items()},
                          "true_x1e3_at_0.5_1_2": {f: [round(x * 1e3, 3) for x in v["true_loss"]["true_change_at_0.5_1_2"]]
                                                   for f, v in entry["families"].items() if "true_loss" in v},
                          "seconds": round(time.time() - t0)}), flush=True)
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()

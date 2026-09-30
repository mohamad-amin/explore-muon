"""Where does GN-PD's gain come from? PD in three curvature geometries, from cheapest to exact (20:1x CDT).

GN-PD(p) = -(G + mu)^-p polar((G + mu)^-p b), each matrix at Muon's norm (gnpd_probe.py), with G:
  kron       per-matrix Kronecker GN, B (x) C (sampled-label output factor, input factor): the two-sided map's geometry
  block      the per-matrix GN from stored per-token factors, G_m V = (1/N) sum_t d_t (d_t^T V x_t) x_t^T, with x_t the
             matrix's input and d_t its sampled-label backprop error at token t. CORRECTION (review, 23:0x CDT): this drops
             the cross-position terms (t != s) of the per-sequence gradient covariance, so it is biased, not unbiased, for the
             exact GN block of every matrix that feeds a later attention layer. No
             forward or backward passes per product: the token-level coupling of inputs and outputs, no cross-matrix
             terms
  full       from gnpd_probe.py's JSON (the exact GN of all hidden matrices), for comparison
The kron and block operators act per matrix; matrix powers by per-matrix Lanczos (--krylov steps, full
reorthogonalization); damping mu_m = d rho_m (rho_m = b_m.G_m b_m / b_m.b_m), d in DAMPINGS, chosen by cross-fitting.
Scored like gnpd_probe.py (cross-fitted on the two halves of 512 held-out EVAL sequences) on the same inputs, so the
numbers compare directly with its muon, pd, gn and gnpd families.

usage: blockgnpd_probe.py OUT_JSON ARM_DIR:STEP [--inputs momentum_g1M,g4M] [--token-sequences 64] [--krylov 32]
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
from one_step_gn import CURV, EVAL, GRAD, muon_norm, polar  # noqa: E402
from one_step_blockgn import cross_fit, halves_score  # noqa: E402
from valley_rescore import G16, fresh_gradient  # noqa: E402

POWERS = (0.25, 0.5)
DAMPINGS = (1e-2, 1e-3, 1e-4)


def token_factors(model, names, batches, T):
    """Per matrix: inputs X (N x n) and sampled-label errors D (N x m) at positions >= 1, over all curvature sequences."""
    gen = torch.Generator(device=batches[0][0].device).manual_seed(1)
    recorder = P.Recorder(model)
    xs, ds = {n: [] for n in names}, {n: [] for n in names}
    for x, y in batches:
        P.gradient_passes(model, recorder, x, y, gen, draws=1)
        for n in names:
            xs[n].append(recorder.inputs[n][:, 1:].reshape(-1, recorder.inputs[n].shape[-1]).float())
            e = T * recorder.errors[n][1]
            ds[n].append(e[:, 1:].reshape(-1, e.shape[-1]).float())
    recorder.remove()
    return {n: (torch.cat(xs[n]), torch.cat(ds[n])) for n in names}


def lanczos(op, start, steps):
    V = torch.zeros(steps, start.numel(), device=start.device)
    V[0] = start.reshape(-1) / start.norm()
    alphas, betas = [], []
    for j in range(steps):
        w = op(V[j])
        alphas.append(float(w @ V[j]))
        for _ in range(2):
            w -= V[:j + 1].T @ (V[:j + 1] @ w)
        if j + 1 == steps:
            break
        beta = float(w.norm())
        if beta < 1e-12:
            V = V[:j + 1]
            break
        betas.append(beta)
        V[j + 1] = w / beta
    k = len(alphas)
    Tk = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    off = torch.tensor(betas[:k - 1], dtype=torch.float64)
    return V[:k], Tk + torch.diag(off, 1) + torch.diag(off, -1)


def power_apply(V, Tk, start_norm, power, mu):
    values, vectors = torch.linalg.eigh(Tk)
    y = vectors @ ((values.clamp_min(0) + mu).pow(-power) * vectors[0] * start_norm)
    return V.T @ y.to(V.device, torch.float32)


def gnpd_matrix(op, b, power, damping, steps):
    """-(G + mu)^-p polar((G + mu)^-p b) for one matrix b (m x n) and its per-matrix operator op on flat vectors."""
    shape = b.shape
    V, Tk = lanczos(op, b, steps)
    rho = float(Tk[0, 0])
    mu = damping * rho
    w = power_apply(V, Tk, float(b.norm()), power, mu).view(shape)
    p_ = polar(w)
    V2, T2 = lanczos(op, p_, steps)
    return -power_apply(V2, T2, float(p_.norm()), power, mu).view(shape)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--inputs", default="momentum_g1M,g4M")
    parser.add_argument("--token-sequences", type=int, default=64)
    parser.add_argument("--krylov", type=int, default=32)
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
    curv = [stream.batch(CURV + first * T, 8, T, device) for first in range(0, args.token_sequences, 8)]
    evals = [stream.batch(EVAL + first * T, 8, T, device) for first in range(0, args.eval_sequences, 8)]
    halves = [evals[:len(evals) // 2], evals[len(evals) // 2:]]
    factors = token_factors(model, names, curv, T)
    kron = {}
    for n in names:
        X, Dm = factors[n]
        kron[n] = (Dm.T @ Dm / Dm.shape[0], X.T @ X / X.shape[0])     # B (m x m), C (n x n)

    def block_op(n, shape):
        X, Dm = factors[n]
        N = X.shape[0]

        def op(v):
            V = v.view(shape)
            s = ((Dm @ V) * X).sum(-1)                  # d_t^T V x_t per token
            return (Dm.T @ (s[:, None] * X) / N).reshape(-1)
        return op

    def kron_op(n, shape):
        B, C = kron[n]

        def op(v):
            return (B @ v.view(shape) @ C).reshape(-1)
        return op

    saved = torch.load(kept / f"step{step:06d}.pt", map_location="cpu", weights_only=False)
    beta = saved["config"]["muon_momentum"]
    buffers = OrderedDict((n, saved["optimizer"]["state"][i]["momentum_buffer"].to(device).float()) for i, n in enumerate(names))
    del saved
    result = {"arm": arm, "step": step, "token_sequences": args.token_sequences, "krylov": args.krylov,
              "tokens": int(factors[names[0]][0].shape[0]), "inputs": {}}
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
        families = {}
        for geometry, make in (("kron", kron_op), ("block", block_op)):
            for p in POWERS:
                fam = families[f"{geometry}pd_p{p:g}"] = {}
                for d in DAMPINGS:
                    direction = OrderedDict()
                    for n in names:
                        m = b[n].float()
                        z = gnpd_matrix(make(n, m.shape), m, p, d, args.krylov)
                        direction[n] = z * (muon_norm(m.shape) / z.norm().clamp_min(1e-30))
                    fam[f"{geometry}pd_p{p:g}_d{d:g}"] = direction
        entry = {"families": {}}
        for family, candidates in families.items():
            scored = {k: halves_score(model, halves, d) for k, d in candidates.items()}
            entry["families"][family] = {"cross_fit": cross_fit(scored), "raw": scored}
        result["inputs"][label] = entry
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({"item": args.item, "input": label,
                          "cross_fitted_x1e3": {f: round(v["cross_fit"]["cross_fitted"] * 1e3, 3) for f, v in entry["families"].items()},
                          "picks": {f: v["cross_fit"]["picks"]["fit0"]["label"] for f, v in entry["families"].items()},
                          "seconds": round(time.time() - t0)}), flush=True)
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()

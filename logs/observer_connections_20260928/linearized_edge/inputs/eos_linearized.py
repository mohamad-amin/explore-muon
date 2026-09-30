"""Edge of stability of each optimizer's own linearized dynamics.

Muon and PD update W <- W - lr f(M'), M' = beta M + g, with f the run's map (Muon: shape factor times Newton-Schulz;
PD: NS(M' R) R rescaled to Muon's norm, R = (C / mean eig + 1e-3 I)^-alpha). Linearized around the next momentum M'
(the saved buffer plus a fresh gradient of the run's batch size), a perturbation obeys heavy ball with the operator
lr J G: J = df/dM at M' (per matrix, by forward-mode AD through an FP32 copy of the map), G the exact GN matrix of
all hidden matrices. Heavy ball is stable iff every eigenvalue mu of lr J G lies in (0, 2(1 + beta)) (Nesterov:
2(1 + beta) / (1 + 2 beta)). If the states sit at the edge of their own dynamics, lr mu_max is at that threshold.
mu_max by Arnoldi on J G (full reorthogonalization); J is checked for symmetry on random pairs.

usage: eos_linearized.py OUT_JSON ARM_DIR:STEP:MAP [...]   (MAP: muon | pd:ALPHA)
"""
import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch.func import jvp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402
from research.adamw_spectra.muon import ns_schedule  # noqa: E402
from research.adamw_spectra.train import learning_rate, token_budget  # noqa: E402
from one_step_gn import CURV, GRAD, Flat, GaussNewton, fresh_gradient, inverse_roots  # noqa: E402


def ns32(x, schedule):
    """The run's Newton-Schulz map in FP32 (training runs the products in BF16)."""
    x = x / x.norm().clamp_min(1e-7)
    transpose = x.shape[0] > x.shape[1]
    if transpose:
        x = x.T
    for a, b, c in schedule:
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transpose else x


def arnoldi(op, start, steps):
    """Hessenberg matrix of op on the Krylov space of start (full reorthogonalization); returns its eigenvalues."""
    n = start.numel()
    V = torch.zeros(steps + 1, n, device=start.device)
    H = torch.zeros(steps + 1, steps, dtype=torch.float64)
    V[0] = start / start.norm()
    for j in range(steps):
        w = op(V[j])
        for _ in range(2):
            coeff = V[:j + 1] @ w
            w -= V[:j + 1].T @ coeff
            H[:j + 1, j] += coeff.double().cpu()
        H[j + 1, j] = float(w.norm())
        if H[j + 1, j] < 1e-12:
            H = H[:j + 1, :j + 1]
            break
        V[j + 1] = w / H[j + 1, j]
    values = torch.linalg.eigvals(H[:H.shape[1], :])
    return values[torch.argsort(values.real, descending=True)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("items", nargs="+")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    for item in args.items:
        started = time.time()
        arm, step, kind = item.split(":", 2)
        step = int(step)
        alpha = float(kind.split(":")[1]) if kind.startswith("pd") else 0.0
        kept = Path(arm) / "scientific" / "kept"
        model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
        config = saved["config"]
        beta, nesterov = config["muon_momentum"], config.get("muon_nesterov", False)
        schedule = ns_schedule(config)
        T = model.config.seq_len
        layers = P.hidden_linears(model)
        names = list(layers)
        body = [p for name, p in model.named_parameters() if name.startswith("blocks.") and p.ndim == 2]
        if any(p is not layers[n].weight for p, n in zip(body, names)):
            raise RuntimeError("optimizer body order differs from the hidden-matrix order")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for n in names:
            layers[n].weight.requires_grad_(True)
        state = saved["optimizer"]["state"]
        buffers = OrderedDict((n, state[i]["momentum_buffer"].to(device).float()) for i, n in enumerate(names))
        del saved
        sequences = config["batch_tokens"] // T
        g = fresh_gradient(model, names, stream, GRAD, min(sequences, 8192), T, device)
        # the map's input at the next step: M' = beta M + g (Nesterov: g + beta M')
        inputs = OrderedDict()
        for n in names:
            m_next = beta * buffers[n] + g[n]
            inputs[n] = g[n] + beta * m_next if nesterov else m_next
        curv = [stream.batch(CURV + first * T, 8, T, device) for first in range(0, args.curvature_sequences, 8)]
        roots = {}
        if alpha > 0:
            recorder = P.Recorder(model)
            sums, count = {n: 0 for n in names}, 0
            with torch.no_grad():
                for x, _ in curv:
                    model(x)
                    for n in names:
                        inp = recorder.inputs[n][:, 1:].reshape(-1, recorder.inputs[n].shape[-1]).double()
                        sums[n] = sums[n] + inp.T @ inp
                    count += x.shape[0] * (T - 1)
            recorder.remove()
            roots = {n: inverse_roots(sums[n] / count, (alpha,))[alpha] for n in names}

        def update_map(n):
            shape = inputs[n].shape
            scale = math.sqrt(max(1.0, shape[0] / shape[1]))
            if alpha > 0:
                r = roots[n]

                def f(x):
                    d = ns32(x @ r, schedule) @ r
                    return d * (math.sqrt(min(shape)) * scale / d.norm().clamp_min(1e-30))
            else:
                def f(x):
                    return ns32(x, schedule) * scale
            return f
        maps = {n: update_map(n) for n in names}
        flat = Flat(OrderedDict((n, layers[n].weight) for n in names))

        def J(v):
            parts = flat.dict(v)
            return torch.cat([jvp(maps[n], (inputs[n],), (parts[n],))[1].reshape(-1) for n in names])
        gen = torch.Generator(device=device).manual_seed(0)
        u = torch.randn(sum(flat.sizes), device=device, generator=gen)
        w = torch.randn(sum(flat.sizes), device=device, generator=gen)
        symmetry = float((u @ J(w) - J(u) @ w) / (u.norm() * J(w).norm()).clamp_min(1e-30))
        gn = GaussNewton(model, names, curv, flat)
        values = arnoldi(lambda v: J(gn(v)), torch.randn(sum(flat.sizes), device=device, generator=gen), args.steps)
        count_params = sum(p.numel() for p in model.parameters())
        lr = learning_rate(config, step + 1, step * config["batch_tokens"], token_budget(config, count_params, T))
        threshold = 2 * (1 + beta) / (1 + 2 * beta) if nesterov else 2 * (1 + beta)
        top = values[0]
        entry = {"map": kind, "alpha": alpha, "beta": beta, "nesterov": nesterov, "lr": lr,
                 "mu_top": [[float(x.real), float(x.imag)] for x in values[:6]],
                 "invariant": lr * float(top.real), "threshold": threshold, "ratio": lr * float(top.real) / threshold,
                 "j_symmetry": symmetry, "batch_tokens": config["batch_tokens"], "seconds": time.time() - started}
        results[f"{Path(arm).name}:{step}"] = entry
        print(json.dumps({"item": f"{Path(arm).name}:{step}", "map": kind, "beta": beta, "nesterov": nesterov,
                          "lr": round(lr, 5), "lr_mu": round(entry["invariant"], 3), "threshold": round(threshold, 3),
                          "ratio": round(entry["ratio"], 3), "mu_top": [[round(a, 2), round(b, 2)] for a, b in entry["mu_top"][:3]],
                          "j_symmetry": round(symmetry, 4), "seconds": round(entry["seconds"])}), flush=True)
        args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()

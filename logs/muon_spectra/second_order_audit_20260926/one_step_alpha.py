"""Which data-norm power alpha gives the best update direction on a given state, free of the learning rate?

At a kept checkpoint, from the saved momentum updated with a fresh 1M-token gradient (M' = 0.95 M + g) and the input second moments C (the run's own EMA statistics when
saved, else measured on probe data), build PD's direction for each alpha exactly as the optimizer does (exact
polar instead of Newton-Schulz):  D = polar(M R) R,  R = (C / mean eig + 1e-3 I)^-alpha, rescaled to Muon's
Frobenius norm sqrt(min(m, n)), times Muon's shape factor sqrt(max(1, rows / cols)); alpha = 0 is Muon.
On held-out sequences, one forward-mode pass gives <g, D> and the exact GN quadratic q(D). The best one-step
decrease along D at its own optimal scale c* = -<g,D>/q(D) is R = <g,D>^2 / (2 q(D)): no learning rate involved.
Reported for the whole body and per matrix kind, with c* relative to the step size actually used (lr_next).
A one-step diagnostic of direction quality, not a training result.

usage: one_step_alpha.py OUT_JSON ARM_DIR STEP [STEP ...] [--alphas 0,0.125,...] [--sequences 256]
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402
from research.adamw_spectra.train import learning_rate, token_budget  # noqa: E402

OFFSET = 2 * 1048576 + 20000 * 512      # beyond the gap-map sequences


def polar(a):
    u, _, vh = torch.linalg.svd(a.double(), full_matrices=False)
    return (u @ vh).float()


def root(cov, alpha, damping=1e-3):
    values, vectors = torch.linalg.eigh(0.5 * (cov + cov.T).double())
    unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
    return ((vectors * (unit + damping).pow(-alpha)) @ vectors.T).float()


def input_moments(model, stream, T, device, sequences=256, micro=8):
    """E[x x^T] of every hidden matrix's input over probe tokens (position 0 excluded, as PD's statistics)."""
    recorder = P.Recorder(model)
    sums = {n: 0 for n in recorder.layers}
    count = 0
    with torch.no_grad():
        for first in range(0, sequences, micro):
            x, _ = stream.batch(OFFSET + first * T, micro, T, device)
            model(x)
            for n in recorder.layers:
                inp = recorder.inputs[n][:, 1:].reshape(-1, recorder.inputs[n].shape[-1]).double()
                sums[n] = sums[n] + inp.T @ inp
            count += x.shape[0] * (T - 1)
    recorder.remove()
    return {n: s / count for n, s in sums.items()}


def fresh_gradient(model, stream, offset, T, device, sequences=2048, micro=8):
    """Mean-loss gradient of every hidden matrix on `sequences` fresh sequences (one training batch of 1M
    tokens): the gradient the optimizer adds to its momentum at the next step (M <- 0.95 M + g)."""
    recorder = P.Recorder(model)
    total = None
    for first in range(0, sequences, micro):
        x, y = stream.batch(offset + first * T, micro, T, device)
        grads, _ = P.gradient_passes(model, recorder, x, y, draws=0)
        total = {n: g.clone() for n, g in grads.items()} if total is None else {n: total[n] + grads[n] for n in grads}
    recorder.remove()
    return {n: g / sequences for n, g in total.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("arm", type=Path)
    parser.add_argument("steps", nargs="+", type=int)
    parser.add_argument("--alphas", default="0,0.0625,0.125,0.25,0.375,0.5,0.75")
    parser.add_argument("--sequences", type=int, default=256)
    parser.add_argument("--micro", type=int, default=8)
    args = parser.parse_args()
    alphas = [float(a) for a in args.alphas.split(",")]
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    results = {"arm": str(args.arm), "alphas": alphas, "steps": {}}
    for step in args.steps:
        kept = args.arm / "scientific" / "kept"
        model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
        T = model.config.seq_len
        config = saved["config"]
        layers = P.hidden_linears(model)
        names = list(layers)
        state = saved["optimizer"]["state"]
        momentum = {n: state[i]["momentum_buffer"].to(device) for i, n in enumerate(names)}
        fresh = fresh_gradient(model, stream, 2 * 1048576 + 60000 * 512, T, device)
        momentum = {n: 0.95 * momentum[n].float() + fresh[n] for n in names}   # the next step's momentum
        stats_path = kept / f"step{step:06d}_input_stats_rank0.pt"
        if stats_path.exists():
            stats = torch.load(stats_path, map_location="cpu", weights_only=False)
            module_names = {id(m): nm for nm, m in model.named_modules()}
            covs = {n: (stats[module_names[id(layers[n])]]["input_cov"] /
                        stats[module_names[id(layers[n])]]["input_cov_weight"].clamp_min(1e-12)).to(device)
                    for n in names}
            source = "run EMA statistics"
        else:
            covs = input_moments(model, stream, T, device)
            source = "probe data"
        count = sum(p.numel() for p in model.parameters())
        lr = learning_rate(config, step + 1, step * config["batch_tokens"], token_budget(config, count, T))
        batches = [stream.batch(OFFSET + first * T, args.micro, T, device) for first in range(0, args.sequences, args.micro)]
        per_alpha = {}
        for alpha in alphas:
            directions = {}
            for n in names:
                m = momentum[n].float()
                rows, cols = m.shape
                if alpha == 0:
                    d = polar(m)
                else:
                    r = root(covs[n], alpha)
                    d = polar(m @ r) @ r
                    d = d * (math.sqrt(min(rows, cols)) / d.norm().clamp_min(1e-30))
                directions[n] = -d * math.sqrt(max(1.0, rows / cols))   # descent direction (update = lr * this)
            groups = {"body": names, **{k: [n for n in names if n.endswith("." + k)] for k in P.KINDS}}
            row = {}
            for group, members in groups.items():
                first = q = 0.0
                for x, y in batches:
                    terms = P.directional_terms(model, x, y, {n: directions[n] for n in members})
                    first += float(terms["first"].sum()); q += float(terms["q"].sum())
                first /= args.sequences; q /= args.sequences
                best_scale = -first / q if q > 0 else float("nan")
                row[group] = {"first": first, "q": q, "best_decrease": first * first / (2 * q) if first < 0 else 0.0,
                              "best_scale_over_lr": best_scale / lr if lr > 0 else float("nan"),
                              "decrease_at_lr": -(lr * first + 0.5 * lr * lr * q)}
            per_alpha[str(alpha)] = row
            print(json.dumps({"step": step, "alpha": alpha, "body": row["body"]}), flush=True)
        results["steps"][str(step)] = {"lr_next": lr, "covariance_source": source, "per_alpha": per_alpha}
        args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()

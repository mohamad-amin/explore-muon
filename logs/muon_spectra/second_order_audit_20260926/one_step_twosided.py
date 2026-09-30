"""Does a Gauss-Newton output factor improve the update direction on a given state? (learning-rate free)

At a kept checkpoint, from the saved momentum updated with a fresh 1M-token gradient (M' = 0.95 M + g): two-sided data-norm directions
    D = L polar(L M R) R,   R = (C / mean eig + 1e-3 I)^-alpha,   L = (B / mean eig + 1e-3 I)^-beta,
rescaled to Muon's Frobenius norm sqrt(min(m, n)) and Muon's shape factor, for a grid of (alpha, beta), with B
either the Gauss-Newton output factor (per-token errors under model-sampled labels) or the empirical-Fisher one
(data labels; free from the ordinary backward pass). C and B are measured on probe sequences at the checkpoint.
The exact one-step decrease at the direction's own optimal scale, <g,D>^2 / (2 q(D)), is measured on held-out
sequences (body and per kind). Also Muon (alpha = beta = 0). A one-step diagnostic, not a training result.

usage: one_step_twosided.py OUT_JSON ARM_DIR STEP [STEP ...] [--sequences 256]
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

PROBE = 2 * 1048576 + 30000 * 512       # statistics sequences
EVAL = 2 * 1048576 + 34000 * 512        # held-out evaluation sequences


def polar(a):
    u, _, vh = torch.linalg.svd(a.double(), full_matrices=False)
    return (u @ vh).float()


def root(cov, power, damping=1e-3):
    if power == 0:
        return None
    values, vectors = torch.linalg.eigh(0.5 * (cov + cov.T).double())
    unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
    return ((vectors * (unit + damping).pow(-power)) @ vectors.T).float()


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
    parser.add_argument("--sequences", type=int, default=256)
    parser.add_argument("--stat-sequences", type=int, default=256)
    parser.add_argument("--micro", type=int, default=8)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    grid = [(0.0, 0.0), (0.25, 0.0), (0.5, 0.0)] + [(a, b, src) for src in ("gn", "ef")
                                                   for a in (0.0, 0.25, 0.5) for b in (0.125, 0.25, 0.5)]
    results = {"arm": str(args.arm), "steps": {}}
    for step in args.steps:
        kept = args.arm / "scientific" / "kept"
        model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
        T = model.config.seq_len
        names = list(P.hidden_linears(model))
        state = saved["optimizer"]["state"]
        momentum = {n: state[i]["momentum_buffer"].to(device).float() for i, n in enumerate(names)}
        fresh = fresh_gradient(model, stream, 2 * 1048576 + 60000 * 512, T, device)
        momentum = {n: 0.95 * momentum[n].float() + fresh[n] for n in names}   # the next step's momentum
        # statistics: C (input), B_gn (sampled labels), B_ef (data labels), all per token
        recorder = P.Recorder(model)
        gen = torch.Generator(device=device).manual_seed(0)
        sums = {n: {"C": 0, "gn": 0, "ef": 0} for n in names}
        count = 0
        for first in range(0, args.stat_sequences, args.micro):
            x, y = stream.batch(PROBE + first * T, args.micro, T, device)
            P.gradient_passes(model, recorder, x, y, gen, draws=1)
            for n in names:
                xi = recorder.inputs[n].reshape(-1, recorder.inputs[n].shape[-1]).double()
                et = (T * recorder.errors[n][0]).reshape(-1, recorder.errors[n][0].shape[-1]).double()
                es = (T * recorder.errors[n][1]).reshape(-1, recorder.errors[n][1].shape[-1]).double()
                sums[n]["C"] = sums[n]["C"] + xi.T @ xi
                sums[n]["ef"] = sums[n]["ef"] + et.T @ et
                sums[n]["gn"] = sums[n]["gn"] + es.T @ es
            count += x.numel()
        recorder.remove()
        batches = [stream.batch(EVAL + first * T, args.micro, T, device) for first in range(0, args.sequences, args.micro)]
        rows = {}
        for entry in grid:
            alpha, beta = entry[0], entry[1]
            source = entry[2] if len(entry) > 2 else "none"
            directions = {}
            for n in names:
                m = momentum[n]
                rows_n, cols_n = m.shape
                r = root(sums[n]["C"] / count, alpha)
                l = root(sums[n][source] / count, beta) if source != "none" else None
                a = m if r is None else m @ r
                a = a if l is None else l @ a
                d = polar(a)
                d = d if r is None else d @ r
                d = d if l is None else l @ d
                d = d * (math.sqrt(min(rows_n, cols_n)) / d.norm().clamp_min(1e-30))
                directions[n] = -d * math.sqrt(max(1.0, rows_n / cols_n))
            groups = {"body": names, **{k: [n for n in names if n.endswith("." + k)] for k in P.KINDS}}
            row = {}
            for group, members in groups.items():
                first = q = 0.0
                for x, y in batches:
                    terms = P.directional_terms(model, x, y, {n: directions[n] for n in members})
                    first += float(terms["first"].sum()); q += float(terms["q"].sum())
                first /= args.sequences; q /= args.sequences
                row[group] = {"first": first, "q": q, "best_decrease": first * first / (2 * q) if first < 0 else 0.0}
            key = f"a{alpha}_b{beta}_{source}"
            rows[key] = row
            print(json.dumps({"step": step, "direction": key, "body": row["body"]}), flush=True)
        results["steps"][str(step)] = rows
        args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()

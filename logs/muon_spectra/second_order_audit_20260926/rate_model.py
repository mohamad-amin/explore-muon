"""Progress-rate model on measured spectra (gap-map frame files). A model, not a result.

River picture: the expected gradient along each direction persists over the steps considered (checked separately
by measure_persistence.py), so the loss keeps falling at a rate set by how the update is spread over directions.
For a direction with curvature h, squared signal s and per-sequence noise n, a step eta along -(g + xi) with
Var xi = n / b_eff gives an expected one-step decrease
    eta s - (1/2) h eta^2 (s + n / b_eff).
For a preconditioner shape P (eta_i = eta P_i) and the best global eta per matrix, the rate is
    R(P) = (sum P s)^2 / (2 sum h P^2 (s + n / b_eff)),
and the per-direction optimum (noise-aware Newton, eta_i = (1/h) SNR / (1 + SNR)) gives
    R* = (1/2) sum (s / h) SNR / (1 + SNR),   SNR = s b_eff / n,
so efficiency = R(P) / R* is in [0, 1] (Cauchy-Schwarz). b_eff = batch sequences x averaging gain (1 for no
averaging; momentum 0.95 gives up to (1 + beta) / (1 - beta) = 39 if the signal persists that long).
Rules: iso (P = 1), curv^-p, input^-q (PD-like), output^-q, kfac^-q, and each optimizer's measured profile.

usage: rate_model.py FRAME_DIR STEP OUT_JSON [--source M] [--summary SUMMARY.pt]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

KINDS = ("q", "k", "v", "o", "up", "down")
EDGES = np.logspace(-12, 6, 73)


def rules_for(m, h):
    lb = m["lam_B"].double().numpy()[:, None]
    lc = m["lam_C"].double().numpy()[None, :]
    lb, lc = np.broadcast_to(lb, h.shape), np.broadcast_to(lc, h.shape)
    rules = {"iso": np.ones_like(h)}
    for p in (0.25, 0.5, 0.75, 1.0):
        rules[f"curv^-{p}"] = np.clip(h, 1e-30, None) ** -p
    for q in (0.25, 0.5, 1.0):
        rules[f"input^-{q}"] = np.clip(lc, 1e-30, None) ** -q
        rules[f"output^-{q}"] = np.clip(lb, 1e-30, None) ** -q
        rules[f"kfac^-{q}"] = np.clip(lb * lc, 1e-30, None) ** -q
    return rules


def profile_P(profile, h):
    idx = np.clip(np.searchsorted(EDGES, h), 0, len(profile) - 1)
    valid = np.where(np.isfinite(profile))[0]
    filled = profile.copy()
    for i in range(len(filled)):
        if not np.isfinite(filled[i]):
            filled[i] = profile[valid[np.argmin(np.abs(valid - i))]]
    return filled[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("step", type=int)
    parser.add_argument("out", type=Path)
    parser.add_argument("--source", default="M")
    parser.add_argument("--summary", type=Path, default=None)
    args = parser.parse_args()
    path = next(Path(args.frame_dir).glob(f"{args.source}_*_step{args.step:06d}.pt"))
    data = torch.load(path, weights_only=False)
    N = data["meta"]["sequences"]
    profiles = {}
    if args.summary:
        summary = torch.load(args.summary, weights_only=False)
        for run, steps in summary.items():
            if args.step in steps:
                for kind, b in steps[args.step]["bins"].items():
                    gD, s, c = b["gD"].numpy(), b["s"].numpy(), b["count"].numpy()
                    with np.errstate(divide="ignore", invalid="ignore"):
                        profiles[(run, kind)] = np.where((s > 0) & (gD > 0) & (c > 50), gD / s, np.nan)
    gains = {"b": 1, "b x 39 (momentum, persistent)": 39}
    batches = {"0.26M": 512, "1M": 2048, "4M": 8192, "16M": 32768}
    result = {}
    for bname, b in batches.items():
        for gname, gain in gains.items():
            b_eff = b * gain
            totals = {}
            per_kind = {k: {} for k in KINDS}
            ideal_total = 0.0
            ideal_kind = {k: 0.0 for k in KINDS}
            for name, m in data["matrices"].items():
                kind = name.split(".")[1]
                h = np.clip(m["exact"].double().numpy(), 1e-30, None)
                g = m["mean"].double().numpy()
                n = m["noise"].double().numpy()
                s = np.clip(g * g - n / N, 0, None)
                noise = n / b_eff
                snr = s / np.clip(noise, 1e-300, None)
                ideal = 0.5 * (s / h * snr / (1 + snr)).sum()
                ideal_total += ideal
                ideal_kind[kind] += ideal
                rules = rules_for(m, h)
                for run in ("M", "PD", "S", "SPD"):
                    if (run, kind) in profiles:
                        rules[f"measured:{run}"] = profile_P(profiles[(run, kind)], h)
                for rule, P in rules.items():
                    num = (P * s).sum() ** 2
                    den = 2 * (h * P * P * (s + noise)).sum()
                    rate = num / den if den > 0 else 0.0
                    totals[rule] = totals.get(rule, 0.0) + rate
                    per_kind[kind][rule] = per_kind[kind].get(rule, 0.0) + rate
            key = f"{bname} | {gname}"
            result[key] = {"ideal_rate": ideal_total,
                           "efficiency": {r: v / ideal_total for r, v in totals.items()},
                           "efficiency_by_kind": {k: {r: v / ideal_kind[k] for r, v in per_kind[k].items()}
                                                  for k in KINDS}}
            order = sorted(totals, key=lambda r: -totals[r])
            print(f"{key:40s} ideal rate {ideal_total:.3e}  " +
                  "  ".join(f"{r}:{totals[r] / ideal_total:.3f}" for r in order), flush=True)
    args.out.write_text(json.dumps({"meta": {**data["meta"], "source": args.source}, "result": result},
                                   indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()

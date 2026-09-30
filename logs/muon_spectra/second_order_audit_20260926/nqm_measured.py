"""Noisy-quadratic model built from measured spectra (gap-map frame files), to compare update rules on the
landscape the network actually has, at several batch sizes, before touching training code. A model, not a result.

Each frame pair (u_i v_j of a hidden matrix) is an independent direction with exact GN curvature h, squared signal
s (debiased, clipped at 0) and per-sequence gradient noise n, measured at one checkpoint. The local quadratic model:
    loss = sum h theta^2 / 2, initial E[theta^2] = s / h^2 (so the initial excess loss is the Newton decrement
    sum s / (2h)); minibatch gradient h theta + xi, Var xi = n / b (b sequences);
    momentum m <- beta m + g (sum form, as our Muon), theta <- theta - eta * P * m;
    schedule: constant, then linear cooldown over the last 10% of steps; eta tuned per rule on a grid.
The exact second moments are propagated (no sampling). Rules differ only in the per-direction preconditioner P
(normalized to geometric mean 1; the tuned eta sets the scale):
    curv^-p        P = h^-p                      exact per-direction curvature (p = 1: Newton)
    input^-q       P = lam_C^-q                  PD-like, input side only (PD alpha 1/4 ~ q = 1/2)
    kfac^-q        P = (lam_B lam_C)^-q          two-sided Kronecker power
    wiener         P = (1/h) SNR / (1 + SNR)     noise-aware Newton at batch b
    measured:<run> P = that optimizer's measured effective step profile (gap map, per curvature bin, shape only)
Output: fraction of the Newton decrement removed after a fixed token budget, per kind and in total.

usage: nqm_measured.py FRAME_DIR STEP OUT_JSON [--source M] [--per-kind 20000]
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch

KINDS = ("q", "k", "v", "o", "up", "down")
EDGES = np.logspace(-12, 6, 73)


def load_directions(frame_dir, run, step, per_kind, rng):
    path = next(p for p in Path(frame_dir).glob(f"{run}_*_step{step:06d}.pt") if re.match(rf"{run}_", p.name))
    data = torch.load(path, weights_only=False)
    N = data["meta"]["sequences"]
    out = {}
    for kind in KINDS:
        rows = {k: [] for k in ("h", "s", "n", "lb", "lc", "w")}
        for name, m in data["matrices"].items():
            if name.split(".")[1] != kind:
                continue
            h = m["exact"].double().numpy().ravel()
            g = m["mean"].double().numpy().ravel()
            n = m["noise"].double().numpy().ravel()
            lb = np.repeat(m["lam_B"].double().numpy(), m["lam_C"].numel())
            lc = np.tile(m["lam_C"].double().numpy(), m["lam_B"].numel())
            s = np.clip(g * g - n / N, 0, None)
            rows["h"].append(h); rows["s"].append(s); rows["n"].append(n); rows["lb"].append(lb); rows["lc"].append(lc)
        cat = {k: np.concatenate(v) for k, v in rows.items() if v}
        total = len(cat["h"])
        pick = rng.choice(total, size=min(per_kind, total), replace=False)
        sample = {k: v[pick] for k, v in cat.items()}
        sample["weight"] = total / len(pick)            # each sampled direction stands for this many pairs
        out[kind] = sample
    return out, data["meta"]


def measured_profiles(summary_path):
    summary = torch.load(summary_path, weights_only=False)
    profiles = {}
    for run, steps in summary.items():
        for step, entry in steps.items():
            for kind, b in entry["bins"].items():
                gD, s = b["gD"].numpy(), b["s"].numpy()
                with np.errstate(divide="ignore", invalid="ignore"):
                    m = np.where((s > 0) & (gD > 0) & (b["count"].numpy() > 50), gD / s, np.nan)
                profiles[(run, step, kind)] = m
    return profiles


def profile_P(profile, h):
    idx = np.clip(np.searchsorted(EDGES, h), 0, len(profile) - 1)
    valid = np.where(np.isfinite(profile))[0]
    filled = profile.copy()
    for i in range(len(filled)):                      # nearest valid bin
        if not np.isfinite(filled[i]):
            filled[i] = profile[valid[np.argmin(np.abs(valid - i))]]
    return filled[idx]


def simulate(h, s, n, P, b, steps, etas, beta=0.95, cooldown=0.1):
    """Exact second-moment recursion for (theta, m) per direction and eta; returns final excess loss per eta."""
    P = P / np.exp(np.mean(np.log(P)))
    E = etas[:, None]
    tt = np.broadcast_to(s / h ** 2, (len(etas), len(h))).copy()     # E theta^2
    tm = np.zeros_like(tt)                                          # E theta m
    mm = np.zeros_like(tt)                                          # E m^2
    q = n / b
    stable = steps - int(cooldown * steps)
    dead = np.zeros(len(etas), bool)
    for t in range(steps):
        lr = E * (1.0 if t < stable else (steps - t) / (steps - stable))
        # m' = beta m + h theta + xi ; theta' = theta - lr P m'
        a, c = beta, h                     # m' = a m + c theta + xi
        mm_new = a * a * mm + 2 * a * c * tm + c * c * tt + q
        tm_new = a * tm + c * tt           # E[theta m'] (xi independent of theta)
        k = lr * P
        tt_new = tt - 2 * k * tm_new + k * k * mm_new
        tm = tm_new - k * mm_new           # E[theta' m']
        tt, mm = tt_new, mm_new
        if t % 25 == 0:
            bad = ~np.isfinite(tt).all(1) | (tt.max(1) > 1e30)
            dead |= bad
            tt[bad] = tm[bad] = mm[bad] = 0
    loss = 0.5 * (h * tt).sum(1)
    loss[dead] = np.inf
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_dir", type=Path)
    parser.add_argument("step", type=int)
    parser.add_argument("out", type=Path)
    parser.add_argument("--source", default="M", help="whose checkpoint supplies the landscape")
    parser.add_argument("--per-kind", type=int, default=20000)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--batches", default="2048:900,8192:225,32768:56",
                        help="batch_sequences:steps pairs at equal tokens (2048 sequences = 1M tokens)")
    args = parser.parse_args()
    args.batches = [tuple(int(v) for v in item.split(":")) for item in args.batches.split(",")]
    rng = np.random.default_rng(0)
    dirs, meta = load_directions(args.frame_dir, args.source, args.step, args.per_kind, rng)
    profiles = measured_profiles(args.summary) if args.summary else {}
    results = {"meta": {**meta, "source": args.source, "per_kind": args.per_kind}, "runs": {}}
    for batch, steps in args.batches:   # same tokens across batch sizes
        rules = {f"curv^-{p}": (lambda d, p=p: (d["h"] + 1e-12) ** -p) for p in (0, 0.25, 0.5, 0.75, 1.0)}
        rules.update({f"input^-{q}": (lambda d, q=q: d["lc"].clip(1e-12) ** -q) for q in (0.25, 0.5, 1.0)})
        rules.update({f"kfac^-{q}": (lambda d, q=q: (d["lb"] * d["lc"]).clip(1e-24) ** -q) for q in (0.25, 0.5, 1.0)})
        rules["wiener"] = lambda d, b=batch: (1 / d["h"]) * (d["s"] * b / d["n"]) / (1 + d["s"] * b / d["n"]) + 1e-30
        for run in ("M", "PD", "S", "SPD"):
            rules[f"measured:{run}"] = lambda d, run=run, kind=None: None      # filled per kind below
        table = {}
        for rule, make in rules.items():
            total_initial, total_final, per_kind = 0.0, 0.0, {}
            for kind, d in dirs.items():
                if rule.startswith("measured:"):
                    key = (rule.split(":")[1], args.step, kind)
                    if key not in profiles:
                        break
                    P = profile_P(profiles[key], d["h"])
                else:
                    P = make(d)
                P = np.asarray(P, float)
                coarse = np.logspace(-9, 3, 13)
                losses = simulate(d["h"], d["s"], d["n"], P, batch, steps, coarse)
                best = int(np.argmin(losses))
                edge = best in (0, len(coarse) - 1)
                fine = coarse[best] * np.logspace(-0.5, 0.5, 7)
                fine_losses = simulate(d["h"], d["s"], d["n"], P, batch, steps, fine)
                j = int(np.argmin(fine_losses))
                initial = 0.5 * (d["s"] / d["h"]).sum()
                per_kind[kind] = {"removed": 1 - fine_losses[j] / initial, "eta": fine[j], "edge": edge}
                total_initial += d["weight"] * initial
                total_final += d["weight"] * fine_losses[j]
            else:
                table[rule] = {"removed_total": 1 - total_final / total_initial, "per_kind": per_kind}
        results["runs"][f"b{batch}"] = table
        print(f"batch {batch} seq ({batch * 512 / 1e6:.2f}M tokens), {steps} steps:")
        for rule, r in sorted(table.items(), key=lambda kv: -kv[1]["removed_total"]):
            print(f"  {rule:14s} removed {r['removed_total']:.3f}   " +
                  " ".join(f"{k}:{v['removed']:.2f}{'!' if v['edge'] else ''}" for k, v in r["per_kind"].items()), flush=True)
    args.out.write_text(json.dumps(results, indent=1, default=float) + "\n")


if __name__ == "__main__":
    main()

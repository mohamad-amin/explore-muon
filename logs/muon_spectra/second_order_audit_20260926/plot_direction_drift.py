"""Figure for direction_drift_probe.py: per input eigendirection bin, the one-step autocorrelation of the mean
gradient, its energy ratio |mu_{N+1}|^2 / |mu_N|^2 (from the pooled q/S and autocorrelation), and the momentum's cosine
with the current mean gradient (2026-09-28 21:1x CDT).

The energy ratio x solves q/S = 1 + x - 2 rho sqrt(x): sqrt(x) = rho + sqrt(rho^2 + q/S - 1).
Along one eigendirection of a quadratic, g_{N+1} = (1 - eta lambda) g_N: autocorrelation sign(1 - eta lambda) and
x = (1 - eta lambda)^2. An exact Newton step gives x -> 0; the edge of stability (eta lambda = 2) gives rho = -1, x = 1.

usage: plot_direction_drift.py OUT_PNG TAG [TAG ...]   (TAG = json stem in direction_drift/)
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def energy_ratio(row):
    rho, qs = row["autocorr_one_step"], row["drift_over_signal"]
    disc = rho * rho + qs - 1
    return (rho + math.sqrt(disc)) ** 2 if disc >= 0 else float("nan")


def main():
    out, tags = sys.argv[1], sys.argv[2:]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    cmap = plt.get_cmap("plasma")
    for j, tag in enumerate(tags):
        r = json.loads((HERE / "direction_drift" / f"{tag}.json").read_text())
        rows = [b for b in r["bins"] if b["count"] >= 20]
        u = [(b["u_lo"] * b["u_hi"]) ** 0.5 for b in rows]
        color = cmap(j / max(1, len(tags) - 1) * 0.9)
        style = "--" if tag.startswith("M") else "-"
        label = f"{tag} (β {r['beta']})"
        axes[0].plot(u, [b["autocorr_one_step"] for b in rows], style, marker="o", ms=3, color=color, label=label)
        axes[1].plot(u, [energy_ratio(b) for b in rows], style, marker="o", ms=3, color=color, label=label)
        axes[2].plot(u, [b["cos_momentum_mean"] for b in rows], style, marker="o", ms=3, color=color, label=label)
    axes[0].set(xscale="log", ylim=(-1, 1), xlabel="input eigenvalue / mean (u)", ylabel="corr(mean grad at N, at N+1)",
                title="One-step autocorrelation of the mean gradient")
    axes[0].axhline(0, color="grey", lw=0.8)
    axes[1].set(xscale="log", yscale="log", xlabel="u", ylabel="|mean grad at N+1|² / |at N|²",
                title="Gradient energy after one step (Newton → 0, EoS → 1)")
    axes[1].axhline(1, color="grey", lw=0.8)
    axes[2].set(xscale="log", xlabel="u", ylabel="pooled cos(M, mean grad at N)", title="Momentum vs the current gradient")
    axes[2].axhline(0, color="grey", lw=0.8)
    axes[0].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()

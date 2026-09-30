"""Figure for the step-vs-path probe (2026-09-30, MUON_CASE 12:30 CDT): the true validation change along each displacement
against its length, with the local GN model's parabola; and the per-matrix length ratios of the sequential paths.

usage: plot_path_probe.py PROBE_JSON OUT_PNG
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHOW = [("k1", "one 16M PD step", "#2e86c1", "-"), ("k1c", "one coordinated 16M step", "#c0392b", "-"),
        ("k4d", "4 × 4M at matched distance (LR/4)", "#7f8c8d", "-"),
        ("k4", "4 × 4M steps (LR 0.02)", "#27ae60", "-"), ("k4lin", "  … on the start's GN model", "#27ae60", "--"),
        ("k16", "16 × 1M steps (LR 0.01)", "#8e44ad", "-"), ("k16lin", "  … on the start's GN model", "#8e44ad", "--")]


def main():
    r = json.load(open(sys.argv[1]))
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
    for label, name, color, style in SHOW:
        li = r["lines"][label]
        pts = [(c * li["norm"], t, m) for c, t, m in zip(li["c"], li["true"], li["model_val"]) if c > 0]
        xs = [p[0] for p in pts]
        ax[0].plot(xs, [p[1] for p in pts], style, marker="o", ms=3.5, color=color, lw=1.8, label=name)
        if label in ("k1", "k1c", "k4", "k16"):
            ax[0].plot(xs, [p[2] for p in pts], ":", color=color, lw=1.2)
    ax[0].axhline(0, color="k", lw=0.7)
    ax[0].set(xlabel="length of the displacement ‖c D‖ (body matrices)", ylabel="validation change from the start",
              ylim=(-0.14, 0.12), xlim=(0, 40),
              title="Along each displacement (dotted: the local GN model's parabola)")
    ax[0].legend(fontsize=7.5, loc="upper left")

    names = ["k1", "k1c", "k4d", "k4dlin", "k4", "k4lin", "k16", "k16lin"]
    texts = ["16M step", "coordinated", "4×4M matched dist.", "  on GN model", "4×4M", "  on GN model", "16×1M", "  on GN model"]
    at1 = [r["lines"][n]["true"][r["lines"][n]["c"].index(1.0)] for n in names]
    best = [min(r["lines"][n]["true"]) for n in names]
    colors = ["#2e86c1", "#c0392b", "#7f8c8d", "#bdc3c7", "#27ae60", "#a9dfbf", "#8e44ad", "#d2b4de"]
    y = range(len(names))
    ax[1].barh([v - 0.2 for v in y], at1, height=0.4, color=colors, label="at the endpoint (c = 1)")
    ax[1].barh([v + 0.2 for v in y], best, height=0.4, color=colors, alpha=0.5, hatch="//", label="at the line's best c")
    ax[1].set_yticks(list(y), texts, fontsize=8)
    ax[1].invert_yaxis()
    ax[1].axvline(0, color="k", lw=0.7)
    ax[1].set(xlabel="validation change from the start",
              title="One batch, used by one step or by K sequential steps")
    ax[1].legend(fontsize=7.5)

    kinds = ["q", "k", "v", "o", "up", "down"]
    pm = r["per_matrix"]
    for kind, color in zip(kinds, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]):
        keys = sorted([k for k in pm if k.split(".")[-2] == kind], key=lambda k: int(k.split(".")[1]))
        layers = [int(k.split(".")[1]) for k in keys]
        ax[2].plot(layers, [pm[k]["norm"]["k4"] / pm[k]["norm"]["k1"] for k in keys], "-o", ms=3, color=color, label=kind)
        ax[2].plot(layers, [pm[k]["norm"]["k16"] / pm[k]["norm"]["k1"] for k in keys], "--o", ms=3, color=color)
    ax[2].set(xlabel="layer", ylabel="‖D_path‖ / ‖16M step‖ per matrix", ylim=(0, 5.5),
              title="The paths go further uniformly (solid: 4×4M, dashed: 16×1M)")
    ax[2].legend(fontsize=7.5, ncol=3)
    fig.tight_layout()
    fig.savefig(sys.argv[2], dpi=130)
    print("wrote", sys.argv[2])


if __name__ == "__main__":
    main()

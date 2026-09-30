"""Figures for step_coupling_probe.py (2026-09-29 01:5x CDT).

(1) coupling_corr.png: per state, the 48 x 48 correlation of the matrices' logit changes, Cov_p(J_l D_l, J_m D_m) /
    sqrt(var var), ordered by block then kind; (2) coupling_parts.png: where each step's GN curvature comes from
    (own, same layer, same kind in other layers, other kinds in other layers), and the coherence sum Q / trace Q.

usage: plot_step_coupling.py OUT_PREFIX TAG [TAG ...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    prefix, tags = sys.argv[1], sys.argv[2:]
    data = {t: json.loads((HERE / "step_coupling" / f"{t}.json").read_text()) for t in tags
            if (HERE / "step_coupling" / f"{t}.json").exists()}
    tags = [t for t in tags if t in data]
    ncol = 4
    nrow = (len(tags) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 4.0 * nrow), squeeze=False)
    for k, t in enumerate(tags):
        r = data[t]
        names = r["names"]
        order = sorted(range(len(names)), key=lambda i: (names[i].split(".")[0], r["kinds"].index(names[i].split(".")[1])))
        c = np.array(r["corr"])[np.ix_(order, order)]
        ax = axes[k // ncol][k % ncol]
        im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
        for b in range(1, len(r["blocks"])):
            ax.axhline(b * len(r["kinds"]) - 0.5, color="k", lw=0.3)
            ax.axvline(b * len(r["kinds"]) - 0.5, color="k", lw=0.3)
        ax.set_title(f"{t}: coherence {r['coherence']:.1f}, GN c* {r['c_star_gn']:.2f}", fontsize=8)
        ax.set_xticks([i * len(r["kinds"]) + 2.5 for i in range(len(r["blocks"]))], [b.replace("block0", "L") for b in r["blocks"]], fontsize=6)
        ax.set_yticks([i * len(r["kinds"]) + 2.5 for i in range(len(r["blocks"]))], [b.replace("block0", "L") for b in r["blocks"]], fontsize=6)
    for k in range(len(tags), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.colorbar(im, ax=axes, shrink=0.5, label="correlation of logit changes (kinds q,k,v,o,up,down within each layer)")
    fig.savefig(f"{prefix}_corr.png", dpi=120, bbox_inches="tight")

    fig, ax = plt.subplots(1, 2, figsize=(15, 4.2))
    parts = ["diagonal", "same_layer_offdiag", "same_kind_other_layers", "other_kind_other_layers"]
    bottoms = np.zeros(len(tags))
    for part, color in zip(parts, ["#444", "#e0843a", "#3a9d5a", "#6a8fd6"]):
        vals = np.array([data[t]["parts"][part] / data[t]["total_q"] for t in tags])
        ax[0].bar(range(len(tags)), vals, bottom=bottoms, color=color, label=part.replace("_", " "))
        bottoms += vals
    ax[0].set_xticks(range(len(tags)), tags, rotation=60, fontsize=7)
    ax[0].set(ylabel="share of the step's GN curvature", title="Where the step's curvature comes from")
    ax[0].legend(fontsize=7)
    ax[1].bar(range(len(tags)), [data[t]["coherence"] for t in tags], color=["#c0392b" if t.startswith("M") or t.startswith("S1") else "#2e86c1" for t in tags])
    ax[1].set_xticks(range(len(tags)), tags, rotation=60, fontsize=7)
    ax[1].set(ylabel="Σ Q / trace Q", title="Coherence of the step's pieces (1 = independent; 48 = all identical)")
    fig.tight_layout()
    fig.savefig(f"{prefix}_parts.png", dpi=120)
    print("wrote", prefix)


if __name__ == "__main__":
    main()

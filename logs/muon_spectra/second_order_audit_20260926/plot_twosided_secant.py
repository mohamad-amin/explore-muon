"""Heatmaps for twosided_secant_probe.py: c* over output (rows, GN output factor B) x input (columns, input second
moment C) decade bins, one panel per optimizer and state; the cell's share of the step energy is printed in it
(2026-09-29 01:3x CDT). Colour: log2(c* / 0.5), so white = the edge of stability, red = under-stepped (c* > 1/2),
blue = beyond the edge; cells with < 0.2% of the step are left blank.

usage: plot_twosided_secant.py OUT_PNG ROW_SPEC [ROW_SPEC ...]   ROW_SPEC = "label:TAG,TAG,..." (json stems)
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    out, specs = sys.argv[1], sys.argv[2:]
    rows = [(s.split(":")[0], s.split(":")[1].split(",")) for s in specs]
    ncol = max(len(t) for _, t in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(3.3 * ncol, 3.0 * len(rows)), squeeze=False)
    for i, (label, tags) in enumerate(rows):
        for j in range(ncol):
            ax = axes[i][j]
            if j >= len(tags) or not (HERE / "twosided_secant" / f"{tags[j]}.json").exists():
                ax.axis("off")
                continue
            r = json.loads((HERE / "twosided_secant" / f"{tags[j]}.json").read_text())
            g = r["grids"]["all"]
            c = np.array([[np.nan if x is None else x for x in row] for row in g["c_star"]], dtype=float)
            e = np.array(g["energy_share"], dtype=float)
            z = np.where((e >= 2e-3) & (c > 0), np.log2(np.clip(c, 1e-3, None) / 0.5), np.nan)
            im = ax.imshow(z, cmap="RdBu_r", vmin=-2, vmax=2, origin="lower")
            for a in range(z.shape[0]):
                for b in range(z.shape[1]):
                    if e[a, b] >= 2e-3:
                        ax.text(b, a, f"{c[a, b]:.2f}\n{100 * e[a, b]:.0f}%", ha="center", va="center", fontsize=6)
            labels = r["labels"]
            ax.set_xticks(range(len(labels)), [x.replace("-", "–") for x in labels], rotation=60, fontsize=6)
            ax.set_yticks(range(len(labels)), labels, fontsize=6)
            ax.set_title(f"{tags[j]}  (c* {g['c_star_total']:.2f})", fontsize=8)
            if j == 0:
                ax.set_ylabel(f"{label}\noutput bin (B eig / mean)", fontsize=7)
            if i == len(rows) - 1:
                ax.set_xlabel("input bin (C eig / mean)", fontsize=7)
    fig.colorbar(im, ax=axes, shrink=0.6, label="log2(c* / ½): 0 = edge, + = under-stepped")
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()

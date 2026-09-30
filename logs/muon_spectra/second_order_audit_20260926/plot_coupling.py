"""Which part of an update moves the gradient along each curvature class? (measure_step_coupling.py output)

Per K-FAC curvature bin of the target direction (the direction whose gradient changes), unbiased squared gradient
change caused by each part of the actual update (stiff / middle / flat hidden-matrix parts by K-FAC rank, and aux =
embeddings, head and norm gains), with the full Hessian (H) and with Gauss-Newton only (G).
usage: plot_coupling.py COUPLING_DIR FIGURE_PATH
"""
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

KINDS = ("q", "k", "v", "o", "up", "down")
PARTS = ("stiff", "middle", "flat", "aux")
PART_COLOR = {"stiff": "tab:red", "middle": "tab:orange", "flat": "tab:blue", "aux": "tab:gray"}
EDGES = np.logspace(-10, 6, 33)


def main(folder, figure):
    files = sorted(Path(folder).glob("*.pt"))
    fig, axes = plt.subplots(len(KINDS), 2 * len(files), figsize=(4.2 * 2 * len(files), 2.3 * len(KINDS)),
                             sharex=True, squeeze=False)
    x = np.concatenate([[EDGES[0] / 2], np.sqrt(EDGES[1:] * EDGES[:-1]), [EDGES[-1] * 2]])
    for f, path in enumerate(files):
        d = torch.load(path, weights_only=False)
        run = re.match(r"(M|PD|S|SPD)_", path.name).group(1)
        step = d["meta"]["step"]
        for row, kind in enumerate(KINDS):
            members = [n for n in d["kfac"] if n.endswith("." + kind)]
            for c, product in enumerate(("H", "G")):
                ax = axes[row, 2 * f + c]
                totals = {}
                for part in PARTS:
                    acc = np.zeros(len(EDGES) + 1)
                    count = np.zeros(len(EDGES) + 1)
                    for n in members:
                        idx = np.digitize(d["kfac"][n].numpy().ravel(), EDGES)
                        a = d["products"][part][product]["A"][n].double().numpy().ravel()
                        b = d["products"][part][product]["B"][n].double().numpy().ravel()
                        acc += np.bincount(idx, a * b, minlength=len(EDGES) + 1)
                        count += np.bincount(idx, minlength=len(EDGES) + 1)
                    totals[part] = acc
                whole = sum(np.clip(v, 0, None) for v in totals.values())
                ok = (count > 100) & (whole > 0)
                for part in PARTS:
                    share = np.clip(totals[part], 0, None) / np.where(whole > 0, whole, 1)
                    ax.plot(x[ok], share[ok], color=PART_COLOR[part], label=part)
                ax.set_xscale("log"); ax.set_ylim(-0.02, 1.02)
                if row == 0:
                    ax.set_title(f"{run} step {step}: share of |Δg|² ({'Hessian' if product == 'H' else 'GN only'})",
                                 fontsize=8)
                if 2 * f + c == 0:
                    ax.set_ylabel(kind)
                if row == len(KINDS) - 1:
                    ax.set_xlabel("K-FAC curvature of the direction")
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("Which part of the actual update changes the gradient along each curvature class")
    fig.tight_layout()
    fig.savefig(figure, dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:3])

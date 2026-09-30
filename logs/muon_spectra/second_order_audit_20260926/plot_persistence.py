"""Gradient persistence per curvature class (measure_persistence.py output), for looking at.

For each checkpoint t and later weights t', per bin of K-FAC curvature (lam_B lam_C) of the step-t frame:
    corr = sum g(t) g(t') / sqrt(sum s(t) sum s(t'))     (independent sequence sets; s = debiased squared signal)
1 = the expected gradient in that class is unchanged; 0 = decorrelated; negative = sign flip (oscillation).
usage: plot_persistence.py PERSISTENCE_DIR FIGURE_PATH
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
EDGES = np.logspace(-10, 6, 49)
COLOR = {"M": "tab:blue", "PD": "tab:red", "S": "tab:green", "SPD": "tab:purple"}
LABEL = {"M": "Muon", "PD": "PD α¼", "S": "SOAP-Muon", "SPD": "S∘PD"}


def binned(m, label_a, label_b, N):
    kfac = torch.outer(m["lam_B"].double(), m["lam_C"].double()).numpy().ravel()
    ga = m[f"{label_a}:signal"].double().numpy().ravel()
    gb = m[f"{label_b}:signal"].double().numpy().ravel()
    sa = ga * ga - m[f"{label_a}:noise"].double().numpy().ravel() / N
    sb = gb * gb - m[f"{label_b}:noise"].double().numpy().ravel() / N
    idx = np.digitize(kfac, EDGES)
    cross = np.bincount(idx, ga * gb, minlength=len(EDGES) + 1)
    na = np.bincount(idx, sa, minlength=len(EDGES) + 1)
    nb = np.bincount(idx, sb, minlength=len(EDGES) + 1)
    count = np.bincount(idx, minlength=len(EDGES) + 1)
    return cross, na, nb, count


def main(folder, figure):
    files = sorted(Path(folder).glob("*.pt"))
    data = []
    for path in files:
        match = re.match(r"(M|PD|S|SPD)_.*_t(\d+)\.pt", path.name)
        if match:
            data.append((match.group(1), int(match.group(2)), torch.load(path, weights_only=False)))
    x = np.concatenate([[EDGES[0] / 2], np.sqrt(EDGES[1:] * EDGES[:-1]), [EDGES[-1] * 2]])
    fig, axes = plt.subplots(len(KINDS), 2, figsize=(11, 2.4 * len(KINDS)), sharex=True, sharey=True, squeeze=False)
    styles = {200: ":", 500: "--", 900: "-"}
    for run, t, d in data:
        N = d["meta"]["sequences"]
        labels = [k.split(":")[0] for k in next(iter(d["matrices"].values())) if k.endswith(":signal")]
        later = [lab for lab in labels if lab not in ("t", "t+1")]
        for row, kind in enumerate(KINDS):
            for col, target in enumerate(("t+1", later[0] if later else None)):
                if target is None:
                    continue
                total = None
                for name, m in d["matrices"].items():
                    if name.split(".")[1] != kind:
                        continue
                    parts = binned(m, "t", target, N)
                    total = parts if total is None else tuple(a + b for a, b in zip(total, parts))
                cross, na, nb, count = total
                ok = (count > 200) & (na > 0) & (nb > 0)
                corr = np.where(ok, cross / np.sqrt(np.clip(na * nb, 1e-300, None)), np.nan)
                ax = axes[row, col]
                ax.plot(x[ok], corr[ok], styles.get(t, "-"), color=COLOR[run], lw=1.2,
                        label=f"{LABEL[run]} t={t}" + ("" if col == 0 else f"→{target[2:]}"))
                ax.axhline(0, color="k", lw=0.6, ls=":")
                ax.axhline(1, color="gray", lw=0.4, ls=":")
                ax.set_xscale("log"); ax.set_ylim(-1.1, 1.3)
                if col == 0:
                    ax.set_ylabel(kind)
                if row == 0:
                    ax.set_title(["gradient at t vs t+1 (one step)", "gradient at t vs a later kept step (300-400 steps)"][col])
                if row == len(KINDS) - 1:
                    ax.set_xlabel("K-FAC curvature λ_B λ_C of the direction")
    axes[0, 0].legend(fontsize=6, ncol=2)
    axes[0, 1].legend(fontsize=6, ncol=2)
    fig.suptitle("Does the expected gradient along a direction persist? (correlation of true gradients, by curvature class)")
    fig.tight_layout()
    fig.savefig(figure, dpi=105)
    plt.close(fig)


if __name__ == "__main__":
    main(*sys.argv[1:3])

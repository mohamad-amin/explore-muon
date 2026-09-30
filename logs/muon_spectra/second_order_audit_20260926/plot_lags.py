"""Gradient persistence against lag (measure_persistence.py output on the dense-lag runs), for looking at.

Frame and gradient at t (sequence set A); gradients at t + lag from later weights (disjoint set B). Per class of
K-FAC curvature rank of the step-t frame (share of each matrix's pairs, stiffest first):
    corr(lag) = sum g(t) g(t+lag) / sqrt(sum s(t) sum s(t+lag))   (s = debiased squared signal)
Odd lags are filled markers and even lags open ones, so a period-2 oscillation shows as a zig-zag.
usage: plot_lags.py LAGS_DIR FIGURE_PATH [JSON_PATH]
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

KINDS = ("q", "k", "v", "o", "up", "down")
CLASSES = (("top 0.1%", 0.0, 0.001), ("0.1-1%", 0.001, 0.01), ("1-10%", 0.01, 0.10), ("10-50%", 0.10, 0.50),
           ("50-100%", 0.50, 1.01))
COLORS = ("tab:red", "tab:orange", "tab:olive", "tab:cyan", "tab:blue")
LABEL = {"M": "Muon", "PD": "PD α¼"}


def class_sums(m, target, N):
    kfac = torch.outer(m["lam_B"].double(), m["lam_C"].double()).flatten()
    share = torch.argsort(torch.argsort(kfac, descending=True)).double() / kfac.numel()
    ga, gb = m["t:signal"].double().flatten(), m[f"{target}:signal"].double().flatten()
    sa = ga * ga - m["t:noise"].double().flatten() / N
    sb = gb * gb - m[f"{target}:noise"].double().flatten() / N
    out = []
    for _, lo, hi in CLASSES:
        mask = (share >= lo) & (share < hi)
        out.append(np.array([float((ga * gb)[mask].sum()), float(sa[mask].sum()), float(sb[mask].sum())]))
    return np.stack(out)


def main(folder, figure, json_path=None):
    runs = {}
    for path in sorted(Path(folder).glob("*_t*.pt")):
        match = re.match(r"(M|PD)_t(\d+)\.pt", path.name)
        if match:
            runs[match.group(1)] = (int(match.group(2)), torch.load(path, weights_only=False))
    fig, axes = plt.subplots(len(KINDS), len(runs), figsize=(5.2 * len(runs), 2.3 * len(KINDS)), sharex=True,
                             sharey=True, squeeze=False)
    table = {}
    for col, (run, (t, d)) in enumerate(sorted(runs.items())):
        N = d["meta"]["sequences"]
        first = next(iter(d["matrices"].values()))
        targets = [k.split(":")[0] for k in first if k.endswith(":signal") and not k.startswith("t:")]
        lags = [1 if tg == "t+1" else int(tg[2:]) - t for tg in targets]
        order = np.argsort(lags)
        lags = [lags[i] for i in order]; targets = [targets[i] for i in order]
        for row, kind in enumerate(KINDS):
            corr = np.zeros((len(CLASSES), len(lags)))
            for j, tg in enumerate(targets):
                total = sum(class_sums(m, tg, N) for name, m in d["matrices"].items() if name.split(".")[1] == kind)
                corr[:, j] = total[:, 0] / np.sqrt(np.clip(total[:, 1] * total[:, 2], 1e-300, None))
            table[f"{run}:{kind}"] = {"lags": lags, **{c[0]: corr[i].tolist() for i, c in enumerate(CLASSES)}}
            ax = axes[row, col]
            lag_arr = np.array(lags)
            for i, (name, _, _) in enumerate(CLASSES):
                ax.plot(lag_arr, corr[i], "-", color=COLORS[i], lw=1, alpha=0.7)
                odd = lag_arr % 2 == 1
                ax.plot(lag_arr[odd], corr[i][odd], "o", color=COLORS[i], ms=4, label=name if row == 0 else None)
                ax.plot(lag_arr[~odd], corr[i][~odd], "o", mfc="none", color=COLORS[i], ms=4)
            ax.axhline(0, color="k", lw=0.6, ls=":")
            ax.set_xscale("log")
            ax.set_ylim(-1.05, 1.05)
            if col == 0:
                ax.set_ylabel(kind)
            if row == 0:
                ax.set_title(f"{LABEL[run]}, t = {t}")
            if row == len(KINDS) - 1:
                ax.set_xlabel("lag (steps); filled = odd, open = even")
    axes[0, 0].legend(fontsize=7, title="K-FAC curvature rank", title_fontsize=7)
    fig.suptitle("How long does the expected gradient persist? corr(g(t), g(t+lag)) by curvature class")
    fig.tight_layout()
    fig.savefig(figure, dpi=110)
    plt.close(fig)
    if json_path:
        Path(json_path).write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:4])

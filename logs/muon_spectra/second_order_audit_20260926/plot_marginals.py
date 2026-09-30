"""Figures for the one-sided GN marginals (measure_marginals.py output), for looking at, not scoring.

usage: plot_marginals.py MARGINALS_DIR FIGURE_DIR
Each figure is small multiples: rows = matrix kind, columns = trajectory; color = training step.
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

KINDS = ("q", "k", "v", "o", "up", "down")
LABEL = {"M": "Muon", "PD": "PD α¼", "S": "SOAP-Muon", "SPD": "S∘PD"}


def load(folder):
    runs = defaultdict(dict)
    for path in sorted(Path(folder).glob("*.pt")):
        match = re.match(r"(M|PD|S|SPD)_.*_step(\d+)\.pt", path.name)
        if not match:
            continue
        run, step = match.group(1), int(match.group(2))
        runs[run][step] = {"arrays": torch.load(path, weights_only=False)["arrays"],
                           "report": json.loads(path.with_suffix(".json").read_text())["matrices"]}
    return runs


def colors(steps):
    cmap = plt.get_cmap("viridis")
    return {s: cmap(i / max(len(steps) - 1, 1)) for i, s in enumerate(sorted(steps))}


def names(kind, depths=range(1, 9)):
    return [f"block{d:02d}.{kind}" for d in depths]


def ratio_along_c(runs, out):
    """Exact GN curvature along each eigenvector of C, relative to K-FAC (tr(B) * eigenvalue)."""
    order = [r for r in ("M", "PD", "S", "SPD") if r in runs]
    fig, axes = plt.subplots(len(KINDS), len(order), figsize=(4.2 * len(order), 2.4 * len(KINDS)),
                             sharex="row", sharey=True, squeeze=False)
    for col, run in enumerate(order):
        palette = colors(runs[run])
        for row, kind in enumerate(KINDS):
            ax = axes[row, col]
            for step, data in sorted(runs[run].items()):
                ratios = []
                for n in names(kind):
                    a = data["arrays"][n]
                    ratios.append((a["in_exact_in"] / (a["trB"] * a["lam_C"].clamp_min(1e-30))).numpy())
                r = np.median(np.stack(ratios), 0)
                idx = np.arange(1, len(r) + 1)
                ax.plot(idx, r, color=palette[step], lw=1, label=f"{step}")
            ax.axhline(1, color="k", lw=0.6, ls=":")
            ax.set_xscale("log"); ax.set_yscale("log"); ax.set_ylim(0.05, 20)
            if row == 0:
                ax.set_title(LABEL[run])
            if col == 0:
                ax.set_ylabel(f"{kind}\nexact / K-FAC")
            if row == len(KINDS) - 1:
                ax.set_xlabel("eigenvector of C (rank)")
    axes[0, -1].legend(title="step", fontsize=6, title_fontsize=7, ncol=2, loc="upper right")
    fig.suptitle("Exact GN input marginal along C's eigenvectors, relative to K-FAC (median over 8 depths)")
    fig.tight_layout()
    fig.savefig(out / "ratio_along_C.png", dpi=110)
    plt.close(fig)


def curvature_vs_variance(runs, out, depth=4):
    """Log-log: exact curvature vs input variance along C's eigenvectors, one depth, all steps."""
    order = [r for r in ("M", "PD", "S", "SPD") if r in runs]
    fig, axes = plt.subplots(len(KINDS), len(order), figsize=(4.2 * len(order), 2.6 * len(KINDS)), squeeze=False)
    for col, run in enumerate(order):
        palette = colors(runs[run])
        for row, kind in enumerate(KINDS):
            ax = axes[row, col]
            n = f"block{depth:02d}.{kind}"
            for step, data in sorted(runs[run].items()):
                a = data["arrays"][n]
                lam = a["lam_C"].numpy(); mean = lam.mean()
                ax.loglog(lam / mean, a["in_exact_in"].numpy() / (a["trB"] * mean), ".", ms=1.5,
                          color=palette[step], alpha=0.7)
                ax.loglog(lam[:1] / mean, a["in_exact_in"].numpy()[:1] / (a["trB"] * mean), "o", ms=4,
                          mfc="none", color=palette[step])
            lim = ax.get_xlim()
            ax.loglog(lim, lim, "k:", lw=0.8)
            if row == 0:
                ax.set_title(f"{LABEL[run]} (block {depth})")
            if col == 0:
                ax.set_ylabel(f"{kind}\nexact curvature")
            if row == len(KINDS) - 1:
                ax.set_xlabel("input variance λ_C / mean")
    fig.suptitle("Exact GN curvature vs input variance along C's eigenvectors; dotted = K-FAC; circle = top eigenvector")
    fig.tight_layout()
    fig.savefig(out / f"curvature_vs_variance_block{depth}.png", dpi=110)
    plt.close(fig)


def over_training(runs, out):
    """Mean input direction and the within/between fit across training, per kind (thin lines = depths)."""
    order = [r for r in ("M", "PD", "S", "SPD") if r in runs]
    panels = [("mean direction: exact / K-FAC", lambda r: r["in"]["mean_direction"]["exact_over_kfac"], True),
              ("mean direction share of tr C", lambda r: r["in"]["mean_direction"]["share_of_C"], False),
              ("fit: weight on C_within (a)", lambda r: r["fit_exact"]["a_within"], True),
              ("fit: weight on C_between (b)", lambda r: r["fit_exact"]["b_between"], True),
              ("residual: fit / K-FAC", lambda r: r["fit_exact"]["residual_fit"] / r["fit_exact"]["residual_kfac"], False),
              ("gradient noise scale (sequences)", lambda r: r["gradient"]["noise_scale_sequences"], True)]
    kind_color = dict(zip(KINDS, plt.get_cmap("tab10").colors))
    fig, axes = plt.subplots(len(panels), len(order), figsize=(4.4 * len(order), 2.5 * len(panels)),
                             sharex=True, squeeze=False)
    for col, run in enumerate(order):
        steps = sorted(runs[run])
        for row, (title, get, logy) in enumerate(panels):
            ax = axes[row, col]
            for kind in KINDS:
                values = np.array([[get(runs[run][s]["report"][n]) for s in steps] for n in names(kind)])
                for line in values:
                    ax.plot(steps, line, color=kind_color[kind], lw=0.4, alpha=0.35)
                ax.plot(steps, np.median(values, 0), color=kind_color[kind], lw=2, label=kind)
            ax.set_xscale("log")
            if logy:
                ax.set_yscale("log")
            if title.startswith("mean direction: exact") or title.startswith("fit"):
                ax.axhline(1, color="k", lw=0.6, ls=":")
            if row == 0:
                ax.set_title(LABEL[run])
            if col == 0:
                ax.set_ylabel(title, fontsize=8)
            if row == len(panels) - 1:
                ax.set_xlabel("training step")
    axes[0, -1].legend(fontsize=7, ncol=3)
    fig.suptitle("Over training: thick = median over depths, thin = each depth")
    fig.tight_layout()
    fig.savefig(out / "over_training.png", dpi=110)
    plt.close(fig)


def spectra(runs, out, depth=4):
    """Eigenvalue spectra of C (input) and B (output), normalized by their mean, one depth, over training."""
    order = [r for r in ("M", "PD", "S", "SPD") if r in runs]
    fig, axes = plt.subplots(2 * len(order), len(KINDS), figsize=(2.9 * len(KINDS), 2.3 * 2 * len(order)),
                             sharey="row", squeeze=False)
    for i, run in enumerate(order):
        palette = colors(runs[run])
        for j, kind in enumerate(KINDS):
            n = f"block{depth:02d}.{kind}"
            for side, row in (("lam_C", 2 * i), ("lam_B", 2 * i + 1)):
                ax = axes[row, j]
                for step, data in sorted(runs[run].items()):
                    lam = data["arrays"][n][side].clamp_min(1e-30).numpy()
                    ax.loglog(np.arange(1, len(lam) + 1), lam / lam.mean(), color=palette[step], lw=1)
                if row == 0:
                    ax.set_title(kind)
                if j == 0:
                    ax.set_ylabel(f"{LABEL[run]}\n{'C (input)' if side == 'lam_C' else 'B (output)'}", fontsize=8)
    fig.suptitle(f"Spectra of C and B / mean, block {depth}, over training (color = step)")
    fig.tight_layout()
    fig.savefig(out / f"spectra_block{depth}.png", dpi=110)
    plt.close(fig)


def main(folder, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    runs = load(folder)
    ratio_along_c(runs, out)
    curvature_vs_variance(runs, out)
    over_training(runs, out)
    spectra(runs, out)
    print(sorted((r, sorted(s)) for r, s in runs.items()))


if __name__ == "__main__":
    main(*sys.argv[1:3])

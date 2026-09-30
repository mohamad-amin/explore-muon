"""Input- and output-side energy profiles of saved update directions (one_step_gn.py *_directions.pt).

For each matrix and direction D (unit Frobenius norm): the energy on each eigenvector of the input second moment C,
||D v_j||^2, and on each eigenvector of the output factor B, ||u_i^T D||^2, binned by log2 rank and summed over the
eight layers of each kind. Shows where along the input and output spectra each direction puts its step.

usage: profile_gn.py DIRECTIONS_PT FIGURE_PATH [--input g4M]
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

KINDS = ("q", "k", "v", "o", "up", "down")


def binned(values):
    ranks = np.arange(1, len(values) + 1)
    bins = np.floor(np.log2(ranks)).astype(int)
    total = np.bincount(bins, weights=values)
    count = np.bincount(bins)
    return 2.0 ** np.arange(len(total)), total, count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("figure", type=Path)
    parser.add_argument("--input", default="g4M")
    args = parser.parse_args()
    data = torch.load(args.path, map_location="cpu", weights_only=False)
    frames = data["frames"]
    source = args.input if args.input in data["directions"] else next(iter(data["directions"]))
    directions = data["directions"][source]
    labels = list(directions)
    fig, axes = plt.subplots(2, len(KINDS), figsize=(3.2 * len(KINDS), 6.0), squeeze=False)
    colors = {"gd": "gray", "muon": "tab:blue", "pd_a0.25": "tab:orange", "pd_a0.5": "tab:brown"}
    for j, kind in enumerate(KINDS):
        members = [n for n in frames if n.endswith("." + kind)]
        for label in labels:
            inp = out = None
            for n in members:
                d = directions[label][n].double()
                d = d / d.norm()
                V, U = frames[n]["V"].double(), frames[n]["U"].double()
                e_in = ((d @ V) ** 2).sum(0).numpy()
                e_out = ((U.T @ d) ** 2).sum(1).numpy()
                inp = e_in if inp is None else inp + e_in
                out = e_out if out is None else out + e_out
            color = colors.get(label, "tab:red" if label.startswith("gn") else "tab:green")
            for i, energy in enumerate((inp, out)):
                x, total, count = binned(energy / len(members))
                axes[i, j].plot(x, total / count, "o-", ms=3, color=color, label=label if j == 0 and i == 0 else None)
        for i, side in enumerate(("input eigvec of C (rank)", "output eigvec of B (rank)")):
            ax = axes[i, j]
            ax.set_xscale("log", base=2)
            ax.set_yscale("log")
            ax.set_title(f"{kind}: energy per {'input' if i == 0 else 'output'} direction", fontsize=8)
            ax.set_xlabel(side, fontsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{args.path.stem} ({source}): where each unit-norm direction puts its energy along the input and output spectra")
    fig.tight_layout()
    fig.savefig(args.figure, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()

"""Within-matrix shape of the exact GN step vs the optimizers' directions (one_step_gn.py *_directions.pt files).

For every hidden matrix and saved direction D:
  - singular values of D C^1/2 (input-whitened: PD alpha 1/2 is flat here up to damping) and of B^1/2 D C^1/2
    (both sides whitened: the per-matrix K-FAC metric), normalized by the largest;
  - participation ratio (sum s^2)^2 / sum s^4 of those spectra (effective rank, as a fraction of the dimension);
  - cosine between directions (GN vs Muon, PD alpha 1/4, PD alpha 1/2, K-FAC).
C and B come from the saved frames (eigenvectors in FP16, eigenvalues). Writes a JSON summary and a figure.

usage: analyze_gn_shape.py DIRECTIONS_PT FIGURE_PATH [--block 4] [--input momentum]
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

KINDS = ("q", "k", "v", "o", "up", "down")


def half_power(vectors, values):
    values = values.clamp_min(0)
    return (vectors * values.sqrt()) @ vectors.T


def participation(s):
    p = s ** 2
    return float(p.sum() ** 2 / (p ** 2).sum() / len(s))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path)
    parser.add_argument("figure", type=Path)
    parser.add_argument("--block", type=int, default=4)
    parser.add_argument("--input", default="momentum")
    args = parser.parse_args()
    data = torch.load(args.path, map_location="cpu", weights_only=False)
    frames = data["frames"]
    source = args.input if args.input in data["directions"] else next(iter(data["directions"]))
    directions = data["directions"][source]
    labels = list(directions)
    gn_label = next((x for x in labels if x.startswith("gnp_")), next(x for x in labels if x.startswith("gn_")))
    summary = {"input": source, "labels": labels, "gn": gn_label, "matrices": {}}
    roots = {}
    for n, f in frames.items():
        roots[n] = (half_power(f["V"].double(), f["lam_C"].double()), half_power(f["U"].double(), f["lam_B"].double()))
    for n in frames:
        c_half, b_half = roots[n]
        entry = {}
        for label in labels:
            d = directions[label][n].double()
            s_c = torch.linalg.svdvals(d @ c_half)
            s_bc = torch.linalg.svdvals(b_half @ d @ c_half)
            entry[label] = {"pr_c": participation(s_c), "pr_bc": participation(s_bc),
                            "s_c": (s_c / s_c[0]).float().tolist(), "s_bc": (s_bc / s_bc[0]).float().tolist()}
        g = directions[gn_label][n].double()
        entry["cos_gn"] = {label: float((directions[label][n].double() * g).sum() /
                                        (directions[label][n].double().norm() * g.norm()).clamp_min(1e-30)) for label in labels}
        summary["matrices"][n] = entry
    # per-kind means
    kinds = {}
    for kind in KINDS:
        members = [n for n in frames if n.endswith("." + kind)]
        kinds[kind] = {label: {"pr_c": float(np.mean([summary["matrices"][n][label]["pr_c"] for n in members])),
                               "pr_bc": float(np.mean([summary["matrices"][n][label]["pr_bc"] for n in members])),
                               "cos_gn": float(np.mean([summary["matrices"][n]["cos_gn"][label] for n in members]))}
                       for label in labels}
    summary["kinds"] = kinds
    out_json = args.figure.with_suffix(".json")
    out_json.write_text(json.dumps(summary) + "\n")
    for kind in KINDS:
        print(kind, " ".join(f"{label}: pr_c {v['pr_c']:.2f} pr_bc {v['pr_bc']:.2f} cos_gn {v['cos_gn']:+.2f} |" for label, v in kinds[kind].items()))
    fig, axes = plt.subplots(2, len(KINDS), figsize=(3.1 * len(KINDS), 5.6), squeeze=False)
    colors = {"gd": "gray", "muon": "tab:blue", "pd_a0.25": "tab:orange", "pd_a0.5": "tab:brown"}
    for j, kind in enumerate(KINDS):
        n = f"block{args.block:02d}.{kind}"
        for i, key in enumerate(("s_c", "s_bc")):
            ax = axes[i, j]
            for label in labels:
                s = np.array(summary["matrices"][n][label][key])
                color = colors.get(label, "tab:red" if label.startswith("gn") else "tab:green")
                ax.plot(np.arange(1, len(s) + 1), s, color=color, lw=1.2 if label == gn_label else 0.9,
                        label=label if j == 0 and i == 0 else None)
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim(1e-3, 1.5)
            ax.set_title(f"{n}: {'D C^½' if key == 's_c' else 'B^½ D C^½'}", fontsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle(f"{args.path.stem} ({source}): singular spectra in whitened coordinates (normalized)")
    fig.tight_layout()
    fig.savefig(args.figure, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()

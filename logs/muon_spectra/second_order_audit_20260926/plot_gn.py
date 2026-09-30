"""Figures for one_step_gn.py: how close each direction gets to the exact Gauss-Newton step, and what the GN step
looks like (per-matrix allocation, singular spectra, energy in the Kronecker frame, coupling between matrices).

usage: plot_gn.py GN_DIR FIGURE_DIR
"""
import json
import math
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KINDS = ("q", "k", "v", "o", "up", "down")
RUN_LABEL = {"M": "Muon", "PD": "PD α¼", "S": "SOAP-Muon", "SPD": "S∘PD"}
INPUT_LABEL = {"g1M": "fresh gradient, 1M tokens", "g4M": "fresh gradient, 4M tokens", "momentum": "next momentum M'"}


def run_name(path):
    match = re.match(r"(M|PD|S|SPD)_.*_step(\d+)\.json", path.name) or re.match(r"(M|PD)_step(\d+)\.json", path.name)
    if not match:
        return (path.stem, 0)
    batch = "4M" if "_b4M_" in path.name else "1M"
    return (f"{RUN_LABEL.get(match.group(1), match.group(1))} ({batch} run)", int(match.group(2)))


def reference(entry):
    return entry["best"].get("gnall", entry["best"].get("gn"))


def ladder_labels(entry):
    best, ref = entry["best"], reference(entry)
    labels = [("gd", "gradient"), ("gd@muon_norms", "gradient, Muon's per-matrix norms"), ("muon", "Muon"),
              ("pd_a0.25", "PD α¼"), ("pd_a0.5", "PD α½"), ("two_sided", "two-sided α¼ β¼"),
              (best["kfac"], f"K-FAC ({best['kfac'].split('_')[1]})"), (best["ekfac"], f"EKFAC exact diag ({best['ekfac'].split('_')[1]})"),
              (f"muon@{ref}_norms", "Muon shape, GN's per-matrix norms"),
              (f"{ref}@muon_norms", "GN shape, Muon's per-matrix norms")]
    if "gn" in best:
        labels.append((best["gn"], f"GN, plain CG ({best['gn'][3:]})"))
    if "gnpow" in best:
        labels.append((best["gnpow"], f"GN^-p, plain Krylov ({best['gnpow'][5:]})"))
    if "gnp" in best:
        labels.append((best["gnp"], f"GN, EKFAC-preconditioned CG ({best['gnp'][4:]})"))
    if "gni" in best:
        labels.append((best["gni"], f"GN, input-whitened CG ({best['gni'][4:]})"))
    return labels


def figure_ladder(results, out):
    rows = len(results)
    inputs = sorted({s for r in results for s in r[1]["inputs"]}, key=list(INPUT_LABEL).index)
    fig, axes = plt.subplots(rows, len(inputs), figsize=(5.2 * len(inputs), 3.6 * rows), squeeze=False)
    for i, (name, r) in enumerate(results):
        for j, source in enumerate(inputs):
            ax = axes[i, j]
            if source not in r["inputs"]:
                ax.axis("off")
                continue
            entry = r["inputs"][source]
            scores = entry["scores"]
            labels = [(k, t) for k, t in ladder_labels(entry) if k in scores]
            ref = scores[reference(entry)]["best_decrease"]
            values = [scores[k]["best_decrease"] / ref for k, _ in labels]
            colors = ["#999999" if k.startswith("gd") else "#d62728" if k.startswith("gn") or "@gn" in k or k.startswith("muon@gn")
                      else "#1f77b4" for k, _ in labels]
            ax.barh(range(len(labels)), values, color=colors)
            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels([t for _, t in labels], fontsize=7)
            ax.axvline(1, color="k", lw=0.6, ls=":")
            for y, v in enumerate(values):
                ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=6)
            ax.set_xlim(0, max(1.15, max(values) * 1.1))
            ax.set_title(f"{name[0]} step {name[1]}: {INPUT_LABEL[source]}\nGN best = {ref * 1e3:.2f}e-3", fontsize=8)
            if i == rows - 1:
                ax.set_xlabel("one-step decrease at own best scale ÷ exact GN's")
    fig.suptitle("How much of the exact Gauss-Newton one-step decrease does each direction get? (held-out sequences)")
    fig.tight_layout()
    fig.savefig(out / "gn_ladder.png", dpi=110)
    plt.close(fig)


def figure_krylov(results, out):
    inputs = sorted({s for r in results for s in r[1]["inputs"]}, key=list(INPUT_LABEL).index)
    fig, axes = plt.subplots(len(results), len(inputs), figsize=(4.6 * len(inputs), 3.2 * len(results)), squeeze=False)
    for i, (name, r) in enumerate(results):
        for j, source in enumerate(inputs):
            ax = axes[i, j]
            if source not in r["inputs"]:
                ax.axis("off")
                continue
            scores = r["inputs"][source]["scores"]
            by_k = {}
            for label, s in scores.items():
                match = re.fullmatch(r"(gn[pi]?)_k(\d+)_d([0-9.e+-]+)", label)
                if match:
                    by_k.setdefault((match.group(1), int(match.group(2))), []).append((float(match.group(3)), s["best_decrease"]))
            for (family, k), points in sorted(by_k.items()):
                points.sort()
                ax.plot([p[0] for p in points], [p[1] * 1e3 for p in points], "o-" if family == "gn" else "s--", ms=3,
                        label=f"{ {'gn': 'plain', 'gnp': 'EKFAC-prec.', 'gni': 'input-whitened'}[family]} CG, {k} steps")
            for label, style in (("muon", "b--"), ("pd_a0.25", "r--"), (r["inputs"][source]["best"]["kfac"], "g--")):
                ax.axhline(scores[label]["best_decrease"] * 1e3, color=style[0], ls="--", lw=0.8, label=label)
            ax.set_xscale("log")
            ax.set_xlabel("damping ÷ curvature along the (preconditioned) input")
            ax.set_ylabel("best one-step decrease (×1e-3)")
            ax.set_title(f"{name[0]} step {name[1]}: {INPUT_LABEL[source]}", fontsize=8)
            ax.legend(fontsize=6)
    fig.suptitle("Exact GN direction vs damping and Krylov depth")
    fig.tight_layout()
    fig.savefig(out / "gn_krylov.png", dpi=110)
    plt.close(fig)


def figure_ritz(results, out):
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for name, r in results:
        for source, entry in r["inputs"].items():
            if "ritz_values" in entry:
                ritz = np.array(entry["ritz_values"])
                ax.plot(np.arange(1, len(ritz) + 1), ritz, "-", lw=1, label=f"{name[0]} {name[1]}, {source}")
    ax.set_yscale("log")
    ax.set_xlabel("Ritz value index (Krylov space of the input)")
    ax.set_ylabel("GN eigenvalue estimate")
    ax.legend(fontsize=6)
    ax.set_title("Top of the GN spectrum seen from the input (Lanczos Ritz values)")
    fig.tight_layout()
    fig.savefig(out / "gn_ritz.png", dpi=110)
    plt.close(fig)


KIND_SHAPES = {"q": (512, 512), "k": (512, 512), "v": (512, 512), "o": (512, 512), "up": (2048, 512), "down": (512, 2048)}


def bin_counts(d):
    """Pairs per log2 rank bin (bin b holds ranks 2^b .. 2^(b+1) - 1, capped at bin 11), as in one_step_gn.structure."""
    bins = np.minimum(np.floor(np.log2(np.arange(1, d + 1))), 11).astype(int)
    return np.bincount(bins, minlength=12)


def matrix_grid(structure, value):
    layers = sorted({int(n[5:7]) for n in structure})
    grid = np.full((len(layers), len(KINDS)), np.nan)
    for n, s in structure.items():
        grid[int(n[5:7]) - 1, KINDS.index(n.split(".")[1])] = value(s)
    return grid


def figure_allocation(results, out):
    for name, r in results:
        entries = r["inputs"]
        fig, axes = plt.subplots(len(entries), 4, figsize=(13, 3.2 * len(entries)), squeeze=False)
        for i, (source, entry) in enumerate(entries.items()):
            st = entry["structure"]
            best = entry["best"]
            muon = matrix_grid(st["muon"], lambda s: s["norm"])
            for j, (label, title) in enumerate((("pd_a0.25", "PD α¼"), (best["kfac"], "K-FAC"), (best["ekfac"], "EKFAC"),
                                                (reference(entry), "exact GN"))):
                grid = matrix_grid(st[label], lambda s: s["norm"])
                ratio = (grid / np.sqrt((grid ** 2).sum())) / (muon / np.sqrt((muon ** 2).sum()))
                ax = axes[i, j]
                im = ax.imshow(np.log2(ratio), cmap="RdBu_r", vmin=-4, vmax=4, aspect="auto")
                ax.set_xticks(range(len(KINDS)))
                ax.set_xticklabels(KINDS, fontsize=7)
                ax.set_yticks(range(grid.shape[0]))
                ax.set_yticklabels([f"L{k + 1}" for k in range(grid.shape[0])], fontsize=7)
                for (y, x), v in np.ndenumerate(np.log2(ratio)):
                    ax.text(x, y, f"{v:+.1f}", ha="center", va="center", fontsize=5)
                ax.set_title(f"{title} ({source}): log2 share of step vs Muon", fontsize=7)
            fig.colorbar(im, ax=axes[i, -1], fraction=0.05)
        fig.suptitle(f"{name[0]} step {name[1]}: per-matrix allocation of the step (each total normalized to 1), relative to Muon")
        fig.tight_layout()
        fig.savefig(out / f"gn_allocation_{name[0].split()[0]}_{'4M' if '4M' in name[0] else '1M'}_{name[1]}.png", dpi=110)
        plt.close(fig)


def figure_singular(results, out, block=4):
    for name, r in results:
        entries = r["inputs"]
        fig, axes = plt.subplots(len(entries), len(KINDS), figsize=(3.0 * len(KINDS), 2.7 * len(entries)), squeeze=False)
        for i, (source, entry) in enumerate(entries.items()):
            st = entry["structure"]
            best = entry["best"]
            for j, kind in enumerate(KINDS):
                ax = axes[i, j]
                n = f"block{block:02d}.{kind}"
                for label, title, color in (("gd", "gradient", "gray"), ("muon", "Muon", "tab:blue"), ("pd_a0.25", "PD α¼", "tab:orange"),
                                            (best["kfac"], "K-FAC", "tab:green"), (reference(entry), "exact GN", "tab:red")):
                    s = np.array(st[label][n]["singular_values"])
                    ax.plot(np.arange(1, len(s) + 1), s / s[0], color=color, lw=1, label=title)
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_ylim(1e-4, 1.5)
                ax.set_title(f"{n} ({source})", fontsize=7)
                if j == 0:
                    ax.set_ylabel("σ_i / σ_1")
        axes[0, 0].legend(fontsize=6)
        fig.suptitle(f"{name[0]} step {name[1]}: singular value spectra of the update directions (block {block})")
        fig.tight_layout()
        fig.savefig(out / f"gn_singular_{name[0].split()[0]}_{'4M' if '4M' in name[0] else '1M'}_{name[1]}.png", dpi=110)
        plt.close(fig)


def figure_frame_energy(results, out):
    for (name, r), source in ((nr, s) for nr in results for s in nr[1]["inputs"]):
        entry = r["inputs"][source]
        st = entry["structure"]
        labels = (("muon", "Muon"), ("pd_a0.25", "PD α¼"), (entry["best"]["kfac"], "K-FAC"), (reference(entry), "exact GN"))
        fig, axes = plt.subplots(len(KINDS), len(labels), figsize=(3.1 * len(labels), 2.6 * len(KINDS)), squeeze=False)
        for i, kind in enumerate(KINDS):
            example = next(s for n, s in st["muon"].items() if n.split(".")[1] == kind)
            rows, cols = len(example["singular_values"]), None
            shape = next((n for n in st["muon"] if n.split(".")[1] == kind))
            for j, (label, title) in enumerate(labels):
                grid = sum(np.array(s["rank_grid_energy"]) for n, s in st[label].items() if n.split(".")[1] == kind)
                d_out, d_in = KIND_SHAPES[kind]
                count = np.outer(bin_counts(d_out), bin_counts(d_in))
                density = np.where(count > 0, grid / np.maximum(count, 1), np.nan) / (grid.sum() / count.sum())
                ax = axes[i, j]
                im = ax.imshow(np.log10(np.clip(density, 1e-4, None)), cmap="RdBu_r", vmin=-3, vmax=3, origin="lower", aspect="auto")
                ax.set_title(f"{title}: {kind}", fontsize=7)
                ax.set_xlabel("input eigvec rank bin: log2(rank)", fontsize=6)
                ax.set_ylabel("output eigvec rank bin", fontsize=6)
        fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, label="log10 energy per pair ÷ the kind's mean")
        fig.suptitle(f"{name[0]} step {name[1]} ({source}): energy density of each direction in the Kronecker frame\n"
                     "(rank bins of the eigenvectors of B (rows) and C (columns); bin 0 = stiffest)")
        fig.savefig(out / f"gn_frame_energy_{name[0].split()[0]}_{'4M' if '4M' in name[0] else '1M'}_{name[1]}_{source}.png", dpi=105)
        plt.close(fig)


def figure_gram(results, out):
    for name, r in results:
        entry = r["inputs"].get("momentum")
        if not entry or "gram" not in entry:
            continue
        fig, axes = plt.subplots(1, len(entry["gram"]), figsize=(6.2 * len(entry["gram"]), 5.4), squeeze=False)
        lines = []
        for j, (label, halves) in enumerate(entry["gram"].items()):
            a = (np.array(halves["A"][0]) + np.array(halves["B"][0])) / 2
            Q = (np.array(halves["A"][1]) + np.array(halves["B"][1])) / 2
            d = np.sqrt(np.diag(Q))
            corr = Q / np.outer(d, d)
            ax = axes[0, j]
            im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_title(f"{label}: correlation of per-matrix output changes", fontsize=8)
            ticks = [i for i in range(len(a)) if i % 6 == 0]
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_xticklabels([f"L{i // 6 + 1}" for i in ticks], fontsize=6)
            ax.set_yticklabels([f"L{i // 6 + 1}" for i in ticks], fontsize=6)
            fig.colorbar(im, ax=ax, fraction=0.046)
            ones = np.ones(len(a))
            full, diag = ones @ Q @ ones, d.dot(d)
            # per-matrix scales fitted on one half, scored on the other
            cross = []
            for fit, test in (("A", "B"), ("B", "A")):
                af, Qf = np.array(halves[fit][0]), np.array(halves[fit][1])
                at, Qt = np.array(halves[test][0]), np.array(halves[test][1])
                c = -np.linalg.lstsq(Qf + 1e-9 * np.trace(Qf) / len(af) * np.eye(len(af)), af, rcond=None)[0]
                single = -(ones @ at) / (ones @ Qt @ ones)
                cross.append((float(c @ at + 0.5 * c @ Qt @ c), float(-(ones @ at) ** 2 / (2 * ones @ Qt @ ones)), single))
            kinds = np.kron(np.ones(len(a) // 6), np.arange(6)).astype(int)
            S = np.zeros((len(a), 6))
            S[np.arange(len(a)), kinds] = 1
            Qk = S.T @ Q @ S
            lines.append(f"{label}: q(sum) / sum q(matrix) = {full / diag:.2f}; q(sum) / sum q(kind) = {full / np.trace(Qk):.2f}; "
                         f"per-matrix scales (cross-validated) {-np.mean([c[0] for c in cross]) * 1e3:.3f}e-3 vs single "
                         f"{-np.mean([c[1] for c in cross]) * 1e3:.3f}e-3")
        fig.suptitle(f"{name[0]} step {name[1]}, next momentum: exact GN coupling between the 48 matrices' pieces of the step\n" +
                     "\n".join(lines), fontsize=8)
        fig.tight_layout()
        fig.savefig(out / f"gn_gram_{name[0].split()[0]}_{'4M' if '4M' in name[0] else '1M'}_{name[1]}.png", dpi=110)
        plt.close(fig)
        print("\n".join(lines))


def main(folder, figures):
    folder, figures = Path(folder), Path(figures)
    figures.mkdir(parents=True, exist_ok=True)
    results = [(run_name(p), json.loads(p.read_text())) for p in sorted(folder.glob("*.json"))]
    results = [(n, r) for n, r in results if r.get("inputs")]
    for (name, step), r in results:
        for source, entry in r["inputs"].items():
            scores = entry["scores"]
            ref = scores[reference(entry)]["best_decrease"]
            row = " ".join(f"{k}:{scores[k]['best_decrease'] / ref:.2f}" for k, _ in ladder_labels(entry) if k in scores)
            print(f"{name} {step} {source}: GN {ref * 1e3:.3f}e-3 (best {reference(entry)}) | {row}")
            for label, line in entry.get("line_search", {}).items():
                s = scores[label]
                predicted = [s["first"] * c + 0.5 * s["q"] * c * c for c in line["scales"]]
                actual = [v - r["eval_loss"] for v in line["loss"]]
                print(f"    line {label:12s} quadratic {np.round(np.array(predicted) * 1e3, 3)} true {np.round(np.array(actual) * 1e3, 3)} (x1e-3)")
    figure_ladder(results, figures)
    figure_krylov(results, figures)
    figure_ritz(results, figures)
    figure_allocation(results, figures)
    figure_singular(results, figures)
    figure_frame_energy(results, figures)
    figure_gram(results, figures)


if __name__ == "__main__":
    main(*sys.argv[1:3])

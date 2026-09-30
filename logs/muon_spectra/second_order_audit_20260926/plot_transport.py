"""Figures for transport_test.py: how stale the momentum is, whether curvature transport predicts it, what the momentum
does as a filter across the curvature spectrum, gradient signal and noise by curvature, and the one-step table.

usage: plot_transport.py OUT_DIR RESULT_JSON [RESULT_JSON ...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

KINDS = ("q", "k", "v", "o", "up", "down")
COLORS = dict(zip(KINDS, ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")))


def short(result):
    return f"{Path(result['arm']).name.split('_track')[0]} @{result['step']}"


def matrix_panels(result, out):
    names = result["names"]
    preds = result["staleness"]["predictions"]
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), sharex=True)
    x = np.arange(len(names))
    colors = [COLORS[n.split(".")[1]] for n in names]
    ratio = [preds["G"]["matrix"][n]["target_norm"] for n in names]
    beta = result["beta"]
    # |beta S| relative to |beta M| per matrix: the stale fraction of the momentum's own size
    frame = result["frame_bins"]
    msq = [sum(frame[n]["M*M"]) for n in names]
    axes[0].bar(x, [r / (beta * np.sqrt(m)) for r, m in zip(ratio, msq)], color=colors)
    axes[0].set_ylabel("|M - M*| / |M|")
    axes[0].set_title(f"{short(result)}: staleness of the momentum per matrix (exact replay of {result['replay']} batches)")
    for label, marker in (("G", "o"), ("H", "s"), ("G_block", "^"), ("kfac", "x")):
        axes[1].plot(x, [preds[label]["matrix"][n]["cos"] for n in names], marker, label=label, ms=5)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_ylabel("cos(beta S, -X Q)")
    axes[1].set_ylim(-0.2, 1.0)
    axes[1].legend(ncol=4, fontsize=8)
    for label, marker in (("G", "o"), ("H", "s")):
        axes[2].plot(x, [preds[label]["matrix"][n]["s_star"] for n in names], marker, label=label, ms=5)
    axes[2].axhline(1, color="k", lw=0.5, ls="--")
    axes[2].set_ylabel("best multiplier s*")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=90, fontsize=7)
    axes[2].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / f"{Path(result['arm']).name}_{result['step']}_matrices.png", dpi=110)
    plt.close(fig)


def binned(result, kinds=KINDS):
    """Per kind: arrays over curvature bins of the pooled sums."""
    names = result["names"]
    frame, noise = result["frame_bins"], result["noise"]
    K = result["replay"]
    edges = np.array(result["bins"]["edges"])
    out = {}
    for kind in kinds + ("all",):
        members = [n for n in names if kind == "all" or n.endswith("." + kind)]
        acc = {}
        for n in members:
            for key, values in frame[n].items():
                acc[key] = acc.get(key, 0) + np.array(values, dtype=float)
            acc["h"] = acc.get("h", 0) + np.array(result["bins"]["h_sum"][n], dtype=float)
            acc["sumsq"] = acc.get("sumsq", 0) + np.array(noise["sum_sq_batches"][n], dtype=float)
        acc["noise"] = (acc["sumsq"] - K * acc["gbar*gbar"]) / (K - 1)
        out[kind] = acc
    centers = np.concatenate([[edges[0] / 1.5], np.sqrt(edges[1:] * edges[:-1]), [edges[-1] * 1.5]])
    return centers, out


def curvature_panels(result, out):
    centers, acc = binned(result)
    beta = result["beta"]
    geometric = (1 - beta ** result["replay"]) / (1 - beta)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax = axes.flat
    for kind in KINDS + ("all",):
        a = acc[kind]
        ok = a["count"] > 50
        c = centers[ok]
        style = dict(color=COLORS.get(kind, "k"), lw=2.5 if kind == "all" else 1.2, label=kind)
        ax[0].loglog(c, (a["gbar*gbar"] / a["count"])[ok], **style)
        ax[0].loglog(c, (a["noise"] / a["count"])[ok], ls=":", color=style["color"], lw=style["lw"])
        ax[1].semilogx(c, np.log10(np.maximum(a["gbar*gbar"] / np.maximum(a["noise"], 1e-300), 1e-6))[ok], **style)
        ax[2].semilogx(c, (a["M*gbar"] / a["gbar*gbar"])[ok] / geometric, **style)
        ax[2].semilogx(c, (a["Mstar*gbar"] / a["gbar*gbar"])[ok] / geometric, ls=":", color=style["color"], lw=style["lw"])
        ax[3].semilogx(c, (a["bS*bS"] / (beta ** 2 * a["M*M"]))[ok], **style)
        ax[4].semilogx(c, (-a["bS*GQ"] / np.sqrt(a["bS*bS"] * a["GQ*GQ"]))[ok], **style)
        ax[4].semilogx(c, (-a["bS*HQ"] / np.sqrt(a["bS*bS"] * a["HQ*HQ"]))[ok], ls=":", color=style["color"], lw=style["lw"])
        ax[5].loglog(c, (a["count"] / a["count"].sum())[ok], **style)
    ax[0].set_title("signal gbar^2 (solid) and 1M-batch noise (dotted) per pair", fontsize=10)
    ax[1].set_title("log10 SNR of one 1M-token batch", fontsize=10)
    ax[1].axhline(0, color="k", lw=0.5)
    ax[2].set_title("<M, gbar> / (|gbar|^2 sum beta^k): M solid, fresh M* dotted", fontsize=10)
    ax[2].axhline(1, color="k", lw=0.5)
    ax[2].axhline(0, color="k", lw=0.5, ls="--")
    ax[3].set_title("stale fraction |M - M*|^2 / |M|^2", fontsize=10)
    ax[4].set_title("cos(beta S, -G Q) solid, cos(beta S, -H Q) dotted", fontsize=10)
    ax[4].axhline(0, color="k", lw=0.5)
    ax[5].set_title("share of frame pairs per bin", fontsize=10)
    for a in ax:
        a.set_xlabel("exact per-pair GN diagonal h (Kronecker frame)")
        a.grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"{short(result)}: gradient signal, noise, momentum filter and staleness across the curvature spectrum")
    fig.tight_layout()
    fig.savefig(out / f"{Path(result['arm']).name}_{result['step']}_curvature.png", dpi=110)
    plt.close(fig)


def one_step_panel(results, out):
    maps = ("gd", "muon", "pd_a0.25", "pd_a0.5", "gn")
    fig, axes = plt.subplots(1, len(results), figsize=(8 * len(results), 5), squeeze=False)
    for ax, result in zip(axes[0], results):
        inputs = list(result["one_step"])
        x = np.arange(len(inputs))
        width = 0.16
        for i, m in enumerate(maps):
            values = [result["one_step"][s]["families"][m]["cross_fit"]["cross_fitted"] * 1e3 for s in inputs]
            ax.bar(x + (i - 2) * width, values, width, label=m)
        ax.set_xticks(x)
        ax.set_xticklabels(inputs, rotation=30, fontsize=8)
        ax.set_ylabel("held-out one-step decrease x1e3 (cross-fitted)")
        ax.set_title(short(result))
        ax.grid(alpha=0.3, axis="y")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "one_step.png", dpi=110)
    plt.close(fig)


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    results = [json.loads(Path(p).read_text()) for p in sys.argv[2:]]
    for result in results:
        if "frame_bins" in result:
            matrix_panels(result, out)
            curvature_panels(result, out)
    complete = [r for r in results if r.get("one_step")]
    if complete:
        one_step_panel(complete, out)


if __name__ == "__main__":
    main()

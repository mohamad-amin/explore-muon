"""Gap-map analysis of measure_frame.py outputs: bins every direction (pair u_i v_j of every hidden matrix) by its
exact GN curvature h and summarizes, per matrix kind and checkpoint:
  decrement   sum of s / h           (s = debiased squared signal): where a Newton step would gain
  b*          sum n / sum s          per-direction critical batch (sequences) vs the training batch (2048)
  m_eff       sum(-D g) / sum s      how far the optimizer's direction D (update minus decay) moved along the
                                     signal, per unit gradient; compared in shape with the noise-aware ideal
  m_ideal     sum(s * eta*) / sum s  eta* = (1/h) * SNR / (1 + SNR), SNR = s * batch / n (one step, batch 2048)
  overshoot   sum(h D^2) / (2 sum(-D g))   < 1: the step lowers the quadratic model in this bin; 1/2 = optimal size
  cos(M, g)   momentum vs the gradient at W[t] (negative = oscillation)
2D view: the same sums over (output-rank bin, input-rank bin) of the frame, rank bins 1, 2-3, 4-7, ...

usage: analyze_frame.py FRAME_DIR OUT_DIR   (writes summary.pt and figures)
"""
import math
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
RUNS = ("M", "PD", "S", "SPD")
LABEL = {"M": "Muon", "PD": "PD α¼", "S": "SOAP-Muon", "SPD": "S∘PD"}
COLOR = {"M": "tab:blue", "PD": "tab:red", "S": "tab:green", "SPD": "tab:purple"}
EDGES = torch.logspace(-12, 6, 73, dtype=torch.float64)   # curvature bins
BATCH = 2048                                              # training batch in sequences
QUANTITIES = ("count", "s", "n", "dec", "h", "gD", "DD", "hDD", "Mg", "MM", "gg", "eta_s")


def rank_bins(n):
    return torch.clamp(torch.floor(torch.log2(torch.arange(1, n + 1, dtype=torch.float64))), max=11).long()


def summarize_file(path):
    data = torch.load(path, weights_only=False)
    meta, mats = data["meta"], data["matrices"]
    N = meta["sequences"]
    out = {k: {q: torch.zeros(len(EDGES) + 1, dtype=torch.float64) for q in QUANTITIES} for k in KINDS}
    out2d = {k: {q: torch.zeros(12, 12, dtype=torch.float64) for q in ("dec", "DD", "s", "count")} for k in KINDS}
    for name, m in mats.items():
        kind = name.split(".")[1]
        h = m["exact"].double().clamp_min(1e-300)
        g = m["mean"].double()
        n = m["noise"].double()
        s = g * g - n / N
        D = (m["update"] - m["decay"]).double()
        M = m["momentum"].double()
        snr = (s.clamp_min(0) * BATCH / n.clamp_min(1e-300))
        eta = snr / (1 + snr) / h
        index = torch.bucketize(h.flatten(), EDGES)
        for q, v in (("count", torch.ones_like(h)), ("s", s), ("n", n), ("dec", s / h), ("h", h),
                     ("gD", -g * D), ("DD", D * D), ("hDD", h * D * D), ("Mg", M * g), ("MM", M * M),
                     ("gg", g * g), ("eta_s", eta * s)):
            out[kind][q] += torch.bincount(index, weights=v.flatten(), minlength=len(EDGES) + 1)
        ri, rj = rank_bins(h.shape[0]), rank_bins(h.shape[1])
        cell = (ri[:, None] * 12 + rj[None, :]).flatten()
        for q, v in (("dec", s / h), ("DD", D * D), ("s", s), ("count", torch.ones_like(h))):
            out2d[kind][q] += torch.bincount(cell, weights=v.flatten(), minlength=144).view(12, 12)
    return meta, out, out2d


def load_all(folder):
    summary = defaultdict(dict)
    for path in sorted(Path(folder).glob("*.pt")):
        match = re.match(r"(M|PD|S|SPD)_.*_step(\d+)\.pt", path.name)
        if match:
            meta, out, out2d = summarize_file(path)
            summary[match.group(1)][int(match.group(2))] = {"meta": meta, "bins": out, "bins2d": out2d}
            print("summarized", path.name, flush=True)
    return summary


def centers():
    e = EDGES.numpy()
    mids = np.sqrt(e[1:] * e[:-1])
    return np.concatenate([[e[0] / 2], mids, [e[-1] * 2]])


def profiles_at(summary, step, out):
    x = centers()
    cols = ["decrement share", "critical batch b* (sequences)", "effective step / ideal (shape)",
            "overshoot  hD²/(2·(−gD))", "cos(momentum, gradient)"]
    fig, axes = plt.subplots(len(KINDS), len(cols), figsize=(4.0 * len(cols), 2.3 * len(KINDS)), squeeze=False)
    for row, kind in enumerate(KINDS):
        for run in RUNS:
            if step not in summary.get(run, {}):
                continue
            b = {q: v.numpy() for q, v in summary[run][step]["bins"][kind].items()}
            ok = b["count"] > 50
            dec = b["dec"] / max(b["dec"][b["dec"] > 0].sum(), 1e-300)
            axes[row, 0].plot(x[ok], np.clip(dec[ok], 1e-6, None), color=COLOR[run], label=LABEL[run])
            with np.errstate(divide="ignore", invalid="ignore"):
                bstar = b["n"] / b["s"]
                axes[row, 1].plot(x[ok & (b["s"] > 0)], bstar[ok & (b["s"] > 0)], color=COLOR[run])
                meff = b["gD"] / b["s"]
                mideal = b["eta_s"] / b["s"]
                ratio = meff / mideal
                good = ok & (b["s"] > 0) & np.isfinite(ratio) & (ratio > 0)
                weights = np.where(good, np.clip(b["dec"], 0, None), 0)
                if weights.sum() > 0:
                    order = np.argsort(ratio[good])
                    cum = np.cumsum(weights[good][order]) / weights[good].sum()
                    median = ratio[good][order][np.searchsorted(cum, 0.5)]
                    axes[row, 2].plot(x[good], ratio[good] / median, color=COLOR[run])
                over = b["hDD"] / (2 * b["gD"])
                pos = ok & (b["gD"] > 0)
                axes[row, 3].plot(x[pos], over[pos], color=COLOR[run])
                cos = b["Mg"] / np.sqrt(b["MM"] * b["gg"])
                axes[row, 4].plot(x[ok], cos[ok], color=COLOR[run])
        axes[row, 1].axhline(BATCH, color="k", ls=":", lw=0.8)
        axes[row, 2].axhline(1, color="k", ls=":", lw=0.8)
        axes[row, 3].axhline(1, color="k", ls=":", lw=0.8); axes[row, 3].axhline(0.5, color="gray", ls=":", lw=0.6)
        axes[row, 4].axhline(0, color="k", ls=":", lw=0.8)
        for col in range(len(cols)):
            ax = axes[row, col]
            ax.set_xscale("log")
            if col in (0, 1, 2, 3):
                ax.set_yscale("log")
            if row == 0:
                ax.set_title(cols[col], fontsize=9)
            if col == 0:
                ax.set_ylabel(kind)
            if row == len(KINDS) - 1:
                ax.set_xlabel("exact curvature h of the direction")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle(f"Gap map at step {step}: directions binned by exact GN curvature (all depths pooled per kind)")
    fig.tight_layout()
    fig.savefig(out / f"gap_profiles_step{step}.png", dpi=105)
    plt.close(fig)


def over_training(summary, out):
    """Scalar summaries per kind vs step: decrement-weighted overshoot, fraction of decrement reachable at the
    batch (b* < 2048), and decrement-weighted cos(momentum, gradient)."""
    panels = ["decrement reachable at batch 2048 (share)", "overshoot, decrement-weighted",
              "cos(momentum, gradient), stiffest 1% of decrement", "cos(momentum, gradient), all"]
    fig, axes = plt.subplots(len(panels), len(KINDS), figsize=(3.2 * len(KINDS), 2.6 * len(panels)), squeeze=False)
    for col, kind in enumerate(KINDS):
        for run in RUNS:
            steps = sorted(summary.get(run, {}))
            if not steps:
                continue
            reach, over, cos_stiff, cos_all = [], [], [], []
            for step in steps:
                b = {q: v.numpy() for q, v in summary[run][step]["bins"][kind].items()}
                dec = np.clip(b["dec"], 0, None)
                with np.errstate(divide="ignore", invalid="ignore"):
                    bstar = b["n"] / b["s"]
                    reach.append(dec[(b["s"] > 0) & (bstar < BATCH)].sum() / max(dec.sum(), 1e-300))
                    ob = b["hDD"] / (2 * b["gD"])
                    use = (b["gD"] > 0) & np.isfinite(ob)
                    over.append((ob[use] * dec[use]).sum() / max(dec[use].sum(), 1e-300))
                    cum = np.cumsum(dec[::-1])[::-1] / max(dec.sum(), 1e-300)
                    stiff = cum <= 0.01 + 1e-12
                    cos_stiff.append(b["Mg"][stiff].sum() / np.sqrt(b["MM"][stiff].sum() * b["gg"][stiff].sum() + 1e-300))
                    cos_all.append(b["Mg"].sum() / np.sqrt(b["MM"].sum() * b["gg"].sum()))
            for row, values in enumerate((reach, over, cos_stiff, cos_all)):
                axes[row, col].plot(steps, values, "-o", ms=3, color=COLOR[run], label=LABEL[run])
        for row in range(len(panels)):
            ax = axes[row, col]
            ax.set_xscale("log")
            if row == 1:
                ax.set_yscale("log"); ax.axhline(1, color="k", ls=":", lw=0.8)
            if row >= 2:
                ax.axhline(0, color="k", ls=":", lw=0.8)
            if row == 0:
                ax.set_title(kind)
            if col == 0:
                ax.set_ylabel(panels[row], fontsize=8)
            if row == len(panels) - 1:
                ax.set_xlabel("training step")
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Gap map over training (per kind, all depths pooled)")
    fig.tight_layout()
    fig.savefig(out / "gap_over_training.png", dpi=105)
    plt.close(fig)


def two_d(summary, step, out):
    """Where the decrement lives on the input (C) and output (B) sides, and where each optimizer spends its update."""
    runs = [r for r in RUNS if step in summary.get(r, {})]
    fig, axes = plt.subplots(len(KINDS), 1 + len(runs), figsize=(2.6 * (1 + len(runs)), 2.4 * len(KINDS)), squeeze=False)
    for row, kind in enumerate(KINDS):
        ref = summary[runs[0]][step]["bins2d"][kind]
        dec = np.clip(ref["dec"].numpy(), 0, None)
        axes[row, 0].imshow(np.log10(dec / dec.sum() + 1e-6), origin="lower", cmap="magma", vmin=-4, vmax=0)
        axes[row, 0].set_ylabel(f"{kind}\noutput rank bin (B)")
        if row == 0:
            axes[row, 0].set_title(f"decrement ({LABEL[runs[0]]})", fontsize=8)
        for col, run in enumerate(runs, start=1):
            dd = summary[run][step]["bins2d"][kind]["DD"].numpy()
            axes[row, col].imshow(np.log10(dd / dd.sum() + 1e-6), origin="lower", cmap="viridis", vmin=-4, vmax=0)
            if row == 0:
                axes[row, col].set_title(f"update energy: {LABEL[run]}", fontsize=8)
        for col in range(1 + len(runs)):
            axes[row, col].set_xticks(range(0, 12, 3), [f"{2 ** k}" for k in range(0, 12, 3)], fontsize=6)
            axes[row, col].set_yticks(range(0, 12, 3), [f"{2 ** k}" for k in range(0, 12, 3)], fontsize=6)
            if row == len(KINDS) - 1:
                axes[row, col].set_xlabel("input rank bin (C)", fontsize=7)
    fig.suptitle(f"Step {step}: share of decrement and of update energy over (output rank, input rank), log10")
    fig.tight_layout()
    fig.savefig(out / f"gap_2d_step{step}.png", dpi=105)
    plt.close(fig)


def reachable_vs_batch(summary, out, run="M"):
    """Share of the Newton decrement in directions whose critical batch is below b, as a function of b (tokens)."""
    batches = np.logspace(np.log10(64), np.log10(262144), 25)          # sequences
    steps = sorted(summary.get(run, {}))
    cmap = plt.get_cmap("viridis")
    fig, axes = plt.subplots(1, len(KINDS), figsize=(3.2 * len(KINDS), 3.0), sharey=True)
    for col, kind in enumerate(KINDS):
        ax = axes[col]
        for i, step in enumerate(steps):
            b = {q: v.numpy() for q, v in summary[run][step]["bins"][kind].items()}
            dec = np.clip(b["dec"], 0, None)
            with np.errstate(divide="ignore", invalid="ignore"):
                bstar = np.where(b["s"] > 0, b["n"] / b["s"], np.inf)
            share = [(dec[bstar < B]).sum() / max(dec.sum(), 1e-300) for B in batches]
            ax.plot(batches * 512 / 1e6, share, color=cmap(i / max(len(steps) - 1, 1)), label=f"{step}")
        for B, style in ((1.05, ":"), (4.19, "--")):
            ax.axvline(B, color="k", ls=style, lw=0.8)
        ax.set_xscale("log"); ax.set_title(kind); ax.set_xlabel("batch (M tokens)")
        if col == 0:
            ax.set_ylabel(f"reachable share of decrement\n({LABEL[run]} checkpoints)")
    axes[-1].legend(title="step", fontsize=6, title_fontsize=7)
    fig.suptitle("Share of the Newton decrement whose per-direction critical batch is below b (dotted 1M, dashed 4M)")
    fig.tight_layout()
    fig.savefig(out / f"reachable_vs_batch_{run}.png", dpi=105)
    plt.close(fig)


def main(folder, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cached = out / "frame_summary.pt"
    if folder == "cached" and cached.exists():
        summary = torch.load(cached, weights_only=False)
    else:
        summary = load_all(folder)
        torch.save(dict(summary), cached)
    available = sorted({s for r in RUNS for s in summary.get(r, {})})
    chosen = [s for s in (100, 500, 1300) if s in available] or available
    for step in chosen:
        profiles_at(summary, step, out)
        two_d(summary, step, out)
    over_training(summary, out)
    for run in RUNS:
        if summary.get(run):
            reachable_vs_batch(summary, out, run)


if __name__ == "__main__":
    main(*sys.argv[1:3])

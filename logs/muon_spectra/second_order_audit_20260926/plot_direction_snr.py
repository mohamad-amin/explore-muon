"""Figure for direction_snr_probe.py (v2): per input eigendirection bin, the per-step gradient SNR at the run's own
batch, the momentum buffer's noise fraction, and its cosine with the current mean gradient (2026-09-28 21:1x CDT).

usage: plot_direction_snr.py OUT_PNG TAG [TAG ...]   (TAG = json stem in direction_snr_v2/)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
NAMES = {1048576: "1M", 4194304: "4M", 16777216: "16M"}


def main():
    out, tags = sys.argv[1], sys.argv[2:]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    cmap = plt.get_cmap("viridis")
    for j, tag in enumerate(tags):
        r = json.loads((HERE / "direction_snr_v2" / f"{tag}.json").read_text())
        batch = NAMES[r["batch_train"]]
        rows = [b for b in r["bins_all"] if b["count"] >= 20]
        u = [(b["u_lo"] * b["u_hi"]) ** 0.5 for b in rows]
        color = cmap(j / max(1, len(tags) - 1))
        style = "--" if tag.startswith("M") else "-"
        label = f"{tag} (β {r['beta']})"
        axes[0].plot(u, [b[f"median_snr_{batch}"] for b in rows], style, marker="o", ms=3, color=color, label=label)
        axes[1].plot(u, [b["momentum_noise_fraction"] for b in rows], style, marker="o", ms=3, color=color, label=label)
        axes[2].plot(u, [b["median_cos_momentum"] for b in rows], style, marker="o", ms=3, color=color, label=label)
    axes[0].set(xscale="log", yscale="log", xlabel="input eigenvalue / mean (u)", ylabel="median per-step SNR at the run's batch",
                title="Gradient SNR per input direction")
    axes[0].axhline(1, color="grey", lw=0.8)
    axes[1].set(xscale="log", ylim=(-0.02, 1.02), xlabel="u", ylabel="noise share of |M|² (white-noise estimate)",
                title="How much of the momentum buffer is noise")
    axes[2].set(xscale="log", xlabel="u", ylabel="median cos(M, current mean gradient)", title="Momentum vs the current gradient")
    axes[2].axhline(0, color="grey", lw=0.8)
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()

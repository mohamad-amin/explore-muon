"""Figure for secant_probe.py: the collective secant step multiplier c* per input-eigenvalue bin, per state; the step's
energy share and slope share per bin (2026-09-29 01:2x CDT).

usage: plot_secant.py OUT_PNG TAG [TAG ...]   (TAG = json stem in secant/)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def main():
    out, tags = sys.argv[1], sys.argv[2:]
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    cmap = plt.get_cmap("viridis")
    for j, tag in enumerate(tags):
        r = json.loads((HERE / "secant" / f"{tag}.json").read_text())
        rows = [b for b in r["bins"] if b["count"] >= 20 and b["c_star"] is not None and b["step_share"] > 1e-4]
        u = [(b["u_lo"] * b["u_hi"]) ** 0.5 for b in rows]
        color = cmap(j / max(1, len(tags) - 1))
        style = "--" if tag.startswith("M") else "-"
        label = f"{tag} (all {r['all']['c_star']:.2f})"
        axes[0].plot(u, [b["c_star"] for b in rows], style, marker="o", ms=3, color=color, label=label)
        axes[1].plot(u, [b["step_share"] for b in rows], style, marker="o", ms=3, color=color, label=tag)
        axes[2].plot(u, [b["slope_share"] for b in rows], style, marker="o", ms=3, color=color, label=tag)
    axes[0].axhline(0.5, color="C3", lw=0.9, ls=":")
    axes[0].axhline(1.0, color="grey", lw=0.9, ls=":")
    axes[0].text(1.2e-3, 0.52, "edge of stability (ηλ = 2)", color="C3", fontsize=8)
    axes[0].text(1.2e-3, 1.02, "secant-optimal length", color="grey", fontsize=8)
    axes[0].set(xscale="log", yscale="log", xlabel="input eigenvalue / mean (u)", ylabel="c* = −⟨μ_N, D⟩ / ⟨μ_N+1 − μ_N, D⟩",
                title="Collective step multiplier per input direction")
    axes[0].legend(fontsize=7, loc="upper right")
    axes[1].set(xscale="log", xlabel="u", ylabel="share of |D|²", title="Where the step's energy is")
    axes[2].set(xscale="log", xlabel="u", ylabel="share of −⟨μ, D⟩", title="Where the first-order gain is")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()

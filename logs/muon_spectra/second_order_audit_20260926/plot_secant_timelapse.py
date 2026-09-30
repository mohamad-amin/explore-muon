"""Time-lapse of the collective secant step multiplier c*(u) at 1M, one panel per optimizer, lines coloured by step,
plus where the step's energy sits (2026-09-29 01:3x CDT).

usage: plot_secant_timelapse.py OUT_PNG
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OPTS = [("M", "Muon"), ("S", "SOAP∘Muon"), ("PD", "PD α ¼"), ("SPD", "S∘PD α ¼")]
STEPS = (10, 50, 100, 200, 500, 900, 1300)


def main():
    fig, axes = plt.subplots(2, len(OPTS), figsize=(4.4 * len(OPTS), 7.6), sharey="row")
    cmap = plt.get_cmap("plasma")
    for j, (tag, label) in enumerate(OPTS):
        for k, step in enumerate(STEPS):
            path = HERE / "secant_timelapse" / f"{tag}1M_{step}.json"
            if not path.exists():
                continue
            r = json.loads(path.read_text())
            rows = [b for b in r["bins"] if b["count"] >= 20 and b["c_star"] is not None and b["step_share"] > 2e-3]
            u = [(b["u_lo"] * b["u_hi"]) ** 0.5 for b in rows]
            color = cmap(k / (len(STEPS) - 1) * 0.9)
            axes[0][j].plot(u, [b["c_star"] for b in rows], marker="o", ms=3, color=color, label=f"@{step} ({r['all']['c_star']:.2f})")
            axes[1][j].plot(u, [b["step_share"] for b in rows], marker="o", ms=3, color=color)
        axes[0][j].axhline(0.5, color="C3", lw=0.9, ls=":")
        axes[0][j].axhline(1.0, color="grey", lw=0.9, ls=":")
        axes[0][j].set(xscale="log", yscale="log", title=f"{label}, 1M: c* per input direction", xlabel="u")
        axes[1][j].set(xscale="log", xlabel="u", title="share of the step's |D|²")
        axes[0][j].legend(fontsize=7, title="step (whole-step c*)", title_fontsize=7)
    axes[0][0].set_ylabel("c* (½ = edge, 1 = secant-optimal)")
    axes[1][0].set_ylabel("share of |D|²")
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=130)
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()

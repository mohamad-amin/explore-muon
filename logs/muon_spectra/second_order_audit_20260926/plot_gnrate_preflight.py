"""Figure for the GN-rate pre-flight (gnrate_preflight.py) at the 16M PD states @9 and @46.

Panels per state: (1) direction quality, the GN model's best one-step decrease a^2 / 2q along each direction (held-out
slope a, GN curvature q); (2) c* = -a / q, the model-optimal multiple of the harness step (log scale; 1 = the harness
step is the model optimum); (3) true held-out loss change at 0.5, 1, 2 x the harness step (symlog); (4) mean per-matrix
cosines between directions.

usage: plot_gnrate_preflight.py OUT_PNG preflight_pd9.json preflight_pd46.json
"""
import json
import sys
from pathlib import Path

import numpy as np

SHORT = {"muon": "Muon", "pd": "PD α½", "kron": "kron (TS ½/½)", "gnpd_A_bf16_k64_p0.5_d1e-3": "GN-PD harness",
         "gnpd_B_bf16_k64_p0.5_d1e-3": "GN-PD set B", "gnpd_A_fp32_k128_p0.5_d1e-3": "GN-PD FP32 k128",
         "gnpd_A_bf16_k64_p0.25_d1e-3": "GN-PD p¼", "gnpd_A_bf16_k64_p0.5_d1e-4": "GN-PD μ 1e-4ρ",
         "nested": "nested (K-whitened)"}


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(sys.argv[1])
    runs = [json.loads(Path(p).read_text()) for p in sys.argv[2:]]
    fig, axes = plt.subplots(len(runs), 4, figsize=(22, 5.2 * len(runs)))
    for row, r in zip(np.atleast_2d(axes), runs):
        labels = list(r["scores"])
        names = [SHORT.get(k, k) for k in labels]
        a = np.array([r["scores"][k]["slope"] for k in labels])
        q = np.array([r["scores"][k]["curvature"] for k in labels])
        quality = a * a / (2 * q)
        colors = ["tab:grey" if k == "muon" else "tab:blue" if k in ("pd", "kron") else
                  "tab:purple" if k == "nested" else "tab:orange" for k in labels]
        ax = row[0]
        ax.barh(names, quality, color=colors)
        ax.invert_yaxis()
        ax.set_xlabel("a² / 2q: GN model's best one-step decrease")
        ax.set_title(f"16M PD state @{r['step'] - 1}: direction quality")
        ax = row[1]
        ax.barh(names, [r["scores"][k]["c_star"] for k in labels], color=colors)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.axvline(1, color="k", lw=0.8)
        ax.set_xlabel("c* = −a/q (multiple of the harness step, LR 0.028)")
        ax.set_title("model-optimal step length")
        ax = row[2]
        width = 0.27
        y = np.arange(len(labels))
        for j, (c, shade) in enumerate((("0.5", 0.45), ("1.0", 0.7), ("2.0", 1.0))):
            ax.barh(y + (j - 1) * width, [r["scores"][k]["true_change"][c] for k in labels], height=width,
                    color=plt.cm.Greens(shade), label=f"{c}× step")
        ax.set_yticks(y, names)
        ax.invert_yaxis()
        ax.set_xscale("symlog", linthresh=0.01)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("true held-out loss change")
        ax.set_title("one step on held-out sequences")
        ax.legend(fontsize=7)
        ax = row[3]
        cos = np.eye(len(labels))
        for i, x in enumerate(labels):
            for j, z in enumerate(labels):
                key = f"{x}|{z}" if f"{x}|{z}" in r["cos"] else f"{z}|{x}"
                if i != j:
                    cos[i, j] = r["cos"][key]["mean"]
        im = ax.imshow(cos, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(len(labels)), names, rotation=70, fontsize=7)
        ax.set_yticks(range(len(labels)), names, fontsize=7)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{cos[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if cos[i, j] < 0.5 else "black")
        ax.set_title(f"mean per-matrix cosine (mean eig G {r['log']['mean_eig_G_A']:.1e}, "
                     f"ρ {r['ritz']['gnpd_A_bf16_k64_p0.5_d1e-3']['rho']:.0f})")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=100)


if __name__ == "__main__":
    main()

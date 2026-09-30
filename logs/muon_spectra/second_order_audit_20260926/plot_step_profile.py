"""Figure and table for step_profile_probe.py: each optimizer's own step against the curvature at its own state.

usage: plot_step_profile.py OUT_PNG OUT_JSON [STEP_PROFILE_DIR]
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TOTAL = {16777216: 92, 4194304: 368, 1048576: 1468}
BATCH = {16777216: "16M", 4194304: "4M", 1048576: "1M"}
COLORS = {"Muon": "tab:grey", "PD": "tab:blue", "TS": "tab:green", "S∘PD": "tab:red", "S_left∘PD": "tab:orange",
          "S_right∘PD": "tab:purple", "SOAP∘Muon": "tab:brown"}
MARKERS = {"16M": "o", "4M": "s", "1M": "^"}


def optimizer(arm):
    name = arm.split("/")[-1]
    for prefix, label in (("SleftPD", "S_left∘PD"), ("SrightPD", "S_right∘PD"), ("SPD", "S∘PD"), ("TS", "TS"),
                          ("PD", "PD"), ("M_", "Muon"), ("S_", "SOAP∘Muon")):
        if name.startswith(prefix):
            return label
    return name


def main():
    out_png, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    folder = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "step_profile"
    rows = []
    for f in sorted(folder.glob("*.json")):
        r = json.loads(f.read_text())
        arm, step = r["item"].rsplit(":", 1)
        b = r["batch_tokens"]
        d = r["directions"]
        gp = r["gradient_pair"]
        flips = [n / c if abs(c) > 1e-12 else float("nan")
                 for n, c in zip(gp["top_projection_next"][:4], gp["top_projection_now"][:4])]
        rows.append({"batch": BATCH[b], "optimizer": optimizer(arm), "step": int(step), "fraction": int(step) / TOTAL[b],
                     "lambda_max": r["ritz"][0], "lambda_2": r["ritz"][1], "ritz": r["ritz"][:16],
                     **{f"{k}_{label}": d[label][k] for label in d for k in
                        ("c_star", "quality", "curvature_top1", "curvature_top16", "energy_top1", "energy_top16",
                         "rayleigh_over_top", "slope", "curvature", "norm")},
                     "flips_top4": flips, "cos_rest": gp["cos_rest"], "grad_top_energy_now": gp["top_energy_fraction_now"],
                     "grad_top_energy_next": gp["top_energy_fraction_next"]})
    out_json.write_text(json.dumps(rows, indent=1) + "\n")
    for r in sorted(rows, key=lambda r: (r["batch"], r["step"], r["optimizer"])):
        print(f"{r['batch']:>3} {r['optimizer']:>10} @{r['step']:<5} λmax {r['lambda_max']:8.2f}  actual: c* {r['c_star_actual']:5.2f} "
              f"q-share top1 {r['curvature_top1_actual']:.2f} top16 {r['curvature_top16_actual']:.2f} "
              f"E top16 {r['energy_top16_actual']:.1e} | pd-map q-share top16 {r['curvature_top16_pd']:.2f} c* {r['c_star_pd']:5.2f} "
              f"| muon-map top16 {r['curvature_top16_muon']:.2f} | flips {[round(x, 2) for x in r['flips_top4']]} cos_rest {r['cos_rest']:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["batch"], r["optimizer"])].append(r)
    for (batch, opt), items in groups.items():
        items.sort(key=lambda r: r["step"])
        x = [r["fraction"] for r in items]
        style = dict(color=COLORS.get(opt, "k"), marker=MARKERS[batch], ls="-" if batch == "16M" else "--" if batch == "4M" else ":")
        label = f"{opt} {batch}"
        axes[0, 0].plot(x, [r["lambda_max"] for r in items], label=label, **style)
        axes[0, 1].plot(x, [r["c_star_actual"] for r in items], label=label, **style)
        axes[0, 2].plot(x, [r["curvature_top16_actual"] for r in items], label=label, **style)
        axes[1, 0].plot(x, [r["energy_top16_actual"] for r in items], label=label, **style)
        axes[1, 1].plot(x, [r["curvature_top16_pd"] - r["curvature_top16_actual"] for r in items], label=label, **style)
        axes[1, 2].plot(x, [np.nanmedian(r["flips_top4"]) for r in items], label=label, **style)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("top GN eigenvalue at each optimizer's own state")
    axes[0, 1].set_title("c* of the optimizer's own step (1 = model-optimal length)")
    axes[0, 1].axhline(1, color="k", lw=0.6)
    axes[0, 1].axhline(0.5, color="k", lw=0.6, ls=":")
    axes[0, 2].set_title("share of the own step's GN curvature in the top 16 eigenvectors")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title("share of the own step's energy in the top 16 eigenvectors")
    axes[1, 1].set_title("PD map of the same momentum minus the actual step: top-16 curvature share")
    axes[1, 1].axhline(0, color="k", lw=0.6)
    axes[1, 2].set_title("gradient after / before the step, along the top 4 eigenvectors (median)")
    axes[1, 2].axhline(0, color="k", lw=0.6)
    for ax in axes.flat:
        ax.set_xlabel("fraction of training")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)


if __name__ == "__main__":
    main()

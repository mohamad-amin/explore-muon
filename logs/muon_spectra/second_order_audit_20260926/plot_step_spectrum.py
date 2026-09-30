"""Figure for step_spectrum_probe.py: where each optimizer's own step spends its energy, gains its slope and pays its
curvature along the GN spectrum, at its own state.

Each panel: cumulative share (from the flattest Ritz node up) against the node's curvature over the state's top Ritz
value, per optimizer. Rows: the actual step's energy, its slope (share of g.D), its curvature (share of D.G D); and the
held-out gradient's energy. Columns: 16M @46, 16M @83, 4M @183, 1M @500.

usage: plot_step_spectrum.py OUT_PNG OUT_JSON [STEP_SPECTRUM_DIR]
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BATCH = {16777216: "16M", 4194304: "4M", 1048576: "1M"}
COLORS = {"Muon": "tab:grey", "PD": "tab:blue", "TS": "tab:green", "S∘PD": "tab:red", "S_left∘PD": "tab:orange",
          "S_right∘PD": "tab:purple", "SOAP∘Muon": "tab:brown"}
COLUMNS = (("16M", 46), ("16M", 83), ("4M", 183), ("1M", 500))


def optimizer(arm):
    name = arm.split("/")[-1]
    for prefix, label in (("SleftPD", "S_left∘PD"), ("SrightPD", "S_right∘PD"), ("SPD", "S∘PD"), ("TS", "TS"),
                          ("PD", "PD"), ("M_", "Muon"), ("S_", "SOAP∘Muon")):
        if name.startswith(prefix):
            return label
    return name


def cumulative(theta, weights, top):
    order = np.argsort(theta)
    x = np.clip(np.array(theta)[order] / top, 1e-12, None)
    w = np.array(weights)[order]
    total = w.sum()
    return x, np.cumsum(w) / (total if abs(total) > 1e-30 else 1.0)


def main():
    out_png, out_json = Path(sys.argv[1]), Path(sys.argv[2])
    folder = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / "step_spectrum"
    runs = []
    for f in sorted(folder.glob("*.json")):
        if f.name == "summary.json":
            continue
        r = json.loads(f.read_text())
        arm, step = r["item"].rsplit(":", 1)
        runs.append({"batch": BATCH[r["batch_tokens"]], "opt": optimizer(arm), "step": int(step), "r": r})
    summary = []
    for run in runs:
        d = run["r"]["directions"]
        top = max(max(v["theta"]) for v in d.values())
        row = {"batch": run["batch"], "opt": run["opt"], "step": run["step"], "top": top}
        for label, v in d.items():
            theta = np.array(v["theta"]) / top
            e, s, c = np.array(v["energy"]), np.array(v["slope"]), np.array(v["curvature"])
            for cut in (1e-4, 1e-3, 1e-2, 1e-1):
                low = theta < cut
                row[f"{label}_energy_below_{cut:g}"] = float(e[low].sum() / e.sum())
                row[f"{label}_slope_below_{cut:g}"] = float(s[low].sum() / s.sum()) if abs(s.sum()) > 1e-30 else None
                row[f"{label}_curv_below_{cut:g}"] = float(c[low].sum() / c.sum())
        summary.append(row)
    out_json.write_text(json.dumps(summary, indent=1) + "\n")
    for row in sorted(summary, key=lambda r: (r["batch"], r["step"], r["opt"])):
        print(f"{row['batch']:>3} @{row['step']:<5} {row['opt']:>10}: step energy <1e-3·λmax {row['actual_energy_below_0.001']:.2f} "
              f"<1e-2 {row['actual_energy_below_0.01']:.2f} | slope <1e-3 {row['actual_slope_below_0.001']:.2f} <1e-2 {row['actual_slope_below_0.01']:.2f} "
              f"| curvature <1e-2 {row['actual_curv_below_0.01']:.2f} <1e-1 {row['actual_curv_below_0.1']:.2f} "
              f"| gradient energy <1e-2 {row['gradient_energy_below_0.01']:.2f}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = (("actual", "energy", "own step: energy"), ("actual", "slope", "own step: slope g·D"),
            ("actual", "curvature", "own step: curvature D·G·D"), ("gradient", "energy", "held-out gradient: energy"))
    fig, axes = plt.subplots(len(rows), len(COLUMNS), figsize=(5.2 * len(COLUMNS), 3.6 * len(rows)), squeeze=False)
    for j, (batch, step) in enumerate(COLUMNS):
        for run in runs:
            if run["batch"] != batch or run["step"] != step:
                continue
            d = run["r"]["directions"]
            top = max(max(v["theta"]) for v in d.values())
            for i, (label, field, title) in enumerate(rows):
                x, y = cumulative(d[label]["theta"], d[label][field], top)
                axes[i, j].step(x, y, where="post", color=COLORS.get(run["opt"], "k"), label=run["opt"], lw=1.6)
                axes[i, j].set_xscale("log")
                axes[i, j].set_title(f"{batch} @{step}: {title}", fontsize=9)
                axes[i, j].set_xlabel("curvature / top Ritz value", fontsize=8)
                axes[i, j].set_ylabel("cumulative share", fontsize=8)
                axes[i, j].axhline(1, color="k", lw=0.5)
    for ax in axes.flat:
        if ax.lines:
            ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=100)


if __name__ == "__main__":
    main()

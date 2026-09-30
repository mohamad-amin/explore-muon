"""Validation-NLL difference from tuned Muon (LR 0.007) over training, per seed and hardware.

Reads step rows only. Run from the project root:
.venv/bin/python logs/muon_spectra/improve_w4_20260925/plot_best.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = Path(__file__).resolve().parent.parent
PAIRS = {  # panel: (baseline Muon@0.007 arm, {label: arm})
    "seed 260925, A6000": ("improve_w4_20260925/M_lr0.007_s260925_a6000", {
        "Muon@0.01": "improve_w2_20260925/M_lr0.01_s260925_a6000_gpusvd",
        "SOAP-Muon core@0.007": "improve_w8_20260925/S_lr0.007_s260925_a6000",
        "data-norm Muon α=¼ @0.01": "improve_w5_20260925/PD_a0.25_lr0.01_s260925_a6000",
        "SOAP∘data-norm α=¼ @0.01": "improve_w10_20260925/SPD_a0.25_lr0.01_s260925_a6000",
        "magnitude-matched Muon": "improve_w10_20260925/MM_a0.25_lr0.01_s260925_a6000",
        "mean-whitened E(β=0.25)": "improve_w2_20260925/E_b0.25_lr0.01_s260925_a6000_gpusvd"}),
    "seed 260924, L40S": ("improve_w7_20260925/M_lr0.007_s260924", {
        "Muon@0.01": "depth8_w512_nobias_rms_qk_20260925/muon",
        "SOAP-Muon core@0.007": "improve_w7_20260925/S_lr0.007_s260924",
        "data-norm Muon α=¼ @0.01": "improve_w9_20260925/PD_a0.25_lr0.01_s260924"}),
    "seed 260927, A6000 (fresh)": ("improve_w11_20260925/M_lr0.007_s260927_a6000", {
        "SOAP-Muon core@0.007": "improve_w11_20260925/S_lr0.007_s260927_a6000",
        "data-norm Muon α=¼ @0.01": "improve_w11_20260925/PD_a0.25_lr0.01_s260927_a6000",
        "SOAP∘data-norm α=¼ @0.01": "improve_w14_20260925/SPD_a0.25_lr0.01_s260927_a6000"}),
}
COLORS = {"Muon@0.01": "0.5", "SOAP-Muon core@0.007": "tab:red", "data-norm Muon α=¼ @0.01": "tab:blue",
          "SOAP∘data-norm α=¼ @0.01": "tab:purple", "magnitude-matched Muon": "tab:gray",
          "mean-whitened E(β=0.25)": "tab:green"}


def curve(arm):
    out = {}
    for path in glob.glob(str(LOGS / arm / "scientific/steps/step*.json")):
        row = json.loads(Path(path).read_text())
        if "validation_nll" in row:
            out[row["step"]] = row["validation_nll"]
    return out


def main():
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(6 * len(PAIRS), 4.6), sharey=True)
    for ax, (title, (base_arm, arms)) in zip(axes, PAIRS.items()):
        base = curve(base_arm)
        for label, arm in arms.items():
            c = curve(arm)
            steps = sorted(s for s in c if s in base and s >= 300)
            if not steps:
                continue
            diffs = [c[s] - base[s] for s in steps]
            tag = f" ({diffs[-1]:+.4f})" if steps[-1] == 1469 else f" (at {steps[-1]})"
            ax.plot(steps, diffs, color=COLORS[label], label=label + tag,
                    linestyle="--" if "Muon" in label and "data-norm" not in label else "-")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(f"{title}: vs tuned Muon@0.007")
        ax.set_xlabel("step")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="lower left")
    axes[0].set_ylabel("val NLL difference (nats/token)")
    axes[0].set_ylim(-0.07, 0.03)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / "best_vs_tuned_muon.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()

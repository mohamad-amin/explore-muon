"""Validation-NLL difference from Muon@0.01 over training, same seed (260925) and A6000.

Reads step rows only. Run from the project root:
.venv/bin/python logs/muon_spectra/improve_w4_20260925/plot_family.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOGS = Path(__file__).resolve().parent.parent
ARMS = {  # label: (arm directory, color, style)
    "Muon@0.007": ("improve_w4_20260925/M_lr0.007_s260925_a6000", "0.4", "--"),
    "exact-polar Muon": ("improve_w4_20260925/X_svd_lr0.01_s260925_a6000", "0.6", ":"),
    "mean-whitened E*": ("improve_w2_20260925/E_auto_lr0.01_s260925_a6000_gpusvd", "tab:olive", "-"),
    "mean-whitened E(β=0.25)": ("improve_w2_20260925/E_b0.25_lr0.01_s260925_a6000_gpusvd", "tab:green", "-"),
    "SOAP-Muon core (S)": ("improve_w3_20260925/S_soap_lr0.01_s260925_a6000_gpusvd", "tab:red", "-"),
    "S, input-side basis only": ("improve_w4_20260925/S_right_lr0.01_s260925_a6000", "tab:orange", "-"),
    "S, standard basis": ("improve_w4_20260925/S_none_lr0.01_s260925_a6000", "tab:brown", ":"),
    "S, input basis, column norm": ("improve_w6_20260925/S_rightcol_lr0.01_s260925_a6000", "tab:pink", "--"),
    "S, activation basis": ("improve_w6_20260925/S_rightact_lr0.01_s260925_a6000", "tab:purple", "--"),
    "data-norm α=1/4 (sandwich)": ("improve_w5_20260925/PD_a0.25_lr0.01_s260925_a6000", "tab:blue", "-"),
    "data-norm α=1/8 (sandwich)": ("improve_w5_20260925/PD_a0.125_lr0.01_s260925_a6000", "tab:cyan", "-"),
    "input-whitened polar α=1/2": ("improve_w6_20260925/PDin_a0.5_lr0.01_s260925_a6000", "navy", "--"),
}
BASE = "improve_w2_20260925/M_lr0.01_s260925_a6000_gpusvd"


def curve(arm):
    out = {}
    for path in glob.glob(str(LOGS / arm / "scientific/steps/step*.json")):
        row = json.loads(Path(path).read_text())
        if "validation_nll" in row:
            out[row["step"]] = row["validation_nll"]
    return out


def main():
    base = curve(BASE)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for label, (arm, color, style) in ARMS.items():
        c = curve(arm)
        steps = sorted(s for s in c if s in base and s >= 50)
        if not steps:
            continue
        diffs = [c[s] - base[s] for s in steps]
        final = f" ({diffs[-1]:+.4f})" if steps[-1] == 1469 else f" (at {steps[-1]})"
        for ax in axes:
            ax.plot(steps, diffs, color=color, linestyle=style, label=label + final)
    for ax, lo in zip(axes, (50, 400)):
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xlim(lo, 1469)
        ax.set_xlabel("step")
        ax.set_ylabel("val NLL − Muon@0.01 (nats/token)")
        ax.grid(alpha=0.3)
    axes[1].set_ylim(-0.045, 0.02)
    axes[0].set_title("Seed 260925, A6000: whole run")
    axes[1].set_title("From step 400 (stable phase and cooldown)")
    axes[1].legend(fontsize=7, loc="lower left")
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / "family_vs_muon_s260925.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()

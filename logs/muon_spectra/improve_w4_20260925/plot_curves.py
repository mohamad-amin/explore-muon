"""Training curves (absolute validation and training NLL) for tuned Muon and the new methods.

Reads step rows only. Run from the project root:
.venv/bin/python logs/muon_spectra/improve_w4_20260925/plot_curves.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS = Path(__file__).resolve().parent.parent
STYLE = {"Muon (tuned, LR 0.007)": ("k", "-"), "SOAP-Muon core": ("tab:red", "-"),
         "data-norm Muon (α=¼)": ("tab:blue", "-"), "SOAP∘data-norm": ("tab:purple", "-")}


def rows(arm):
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(LOGS / arm / "scientific/steps/step*.json")))]


def val(arm):
    return sorted((r["step"], r["validation_nll"]) for r in rows(arm) if "validation_nll" in r and r["step"] > 0)


def train(arm, window=25):
    data = sorted((r["step"], r["train_nll"]) for r in rows(arm) if "train_nll" in r)
    steps = np.array([s for s, _ in data]); losses = np.array([l for _, l in data])
    kernel = np.ones(window) / window
    return steps[window - 1:], np.convolve(losses, kernel, mode="valid")


def panel(ax, arms, title, zoom=None, kind="val"):
    for label, arm in arms.items():
        color, style = STYLE[label]
        if kind == "val":
            pts = val(arm)
            if not pts:
                continue
            x, y = zip(*pts)
            tag = f" (final {y[-1]:.4f})" if x[-1] in (1469, 2562, 2938) else f" (at step {x[-1]})"
        else:
            x, y = train(arm)
            tag = ""
        ax.plot(x, y, color=color, linestyle=style, lw=1.4, label=label + tag)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("step (1M tokens each)")
    ax.set_ylabel("validation NLL" if kind == "val" else "training NLL (25-step mean)")
    if zoom:
        ax.set_xlim(*zoom[0]); ax.set_ylim(*zoom[1])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5)


def main():
    w512 = {"Muon (tuned, LR 0.007)": "improve_w11_20260925/M_lr0.007_s260927_a6000",
            "SOAP-Muon core": "improve_w11_20260925/S_lr0.007_s260927_a6000",
            "data-norm Muon (α=¼)": "improve_w11_20260925/PD_a0.25_lr0.01_s260927_a6000",
            "SOAP∘data-norm": "improve_w14_20260925/SPD_a0.25_lr0.01_s260927_a6000"}
    w768 = {"Muon (tuned, LR 0.007)": "improve_w19_20260925/M_w768_lr0.007_s260924",
            "SOAP-Muon core": "improve_w24_20260925/S_w768_lr0.007_s260924",
            "data-norm Muon (α=¼)": "improve_w19_20260925/PD_w768_a0.25_lr0.01_s260924",
            "SOAP∘data-norm": "improve_w24_20260925/SPD_w768_a0.25_lr0.01_s260924"}
    long = {"Muon (tuned, LR 0.007)": "improve_w13_20260925/M_lr0.007_2x_s260924",
            "data-norm Muon (α=¼)": "improve_w13_20260925/PD_a0.25_lr0.01_2x_s260924"}
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    panel(axes[0, 0], w512, "Width 512 (77M), fresh seed 260927, A6000: full run", zoom=((0, 1469), (3.6, 6.5)))
    panel(axes[1, 0], w512, "Width 512: last 700 steps (cooldown from step 1323)", zoom=((770, 1469), (3.66, 3.95)))
    panel(axes[0, 1], w768, "Width 768 (134M), seed 260924, L40S: full run", zoom=((0, 2562), (3.4, 6.5)))
    panel(axes[1, 1], w768, "Width 768: last 1300 steps (cooldown from step 2306)", zoom=((1260, 2562), (3.43, 3.75)))
    panel(axes[0, 2], w512, "Width 512, seed 260927: training loss", zoom=((25, 1469), (3.6, 6.5)), kind="train")
    panel(axes[1, 2], long, "2× tokens (2938 steps), seed 260924, L40S: last 1700 steps",
          zoom=((1240, 2938), (3.55, 3.85)))
    fig.suptitle("Training curves: tuned Muon vs data-norm Muon, SOAP-Muon and their combination", fontsize=12)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / "training_curves.png"
    fig.savefig(out, dpi=120)
    print(out)


if __name__ == "__main__":
    main()

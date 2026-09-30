"""The reversal of layer-staged PD and the sharpness it leaves: loss gap staged - control, and the top GN eigenvalue on the
step's curvature sequences for both branches (2026-09-29 05:5x CDT). Optional extra branches (the switch-off tests,
06:58 CDT) are drawn in both panels with a marker at their switch step.

usage: plot_staged_sharpness.py OUT_PNG CONTROL_DIR STAGED_DIR [EXTRA_DIR:SWITCH_STEP ...]   (dirs under staged_branch/)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent


def rows(name):
    out = {}
    for f in sorted((HERE / "staged_branch" / name / "steps").glob("step*.json")):
        r = json.loads(f.read_text())
        out[r["step"]] = r
    return out


def main():
    out, control, staged = sys.argv[1:4]
    extras = [(e.split(":")[0], int(e.split(":")[1])) for e in sys.argv[4:]]
    c, s = rows(control), rows(staged)
    steps = sorted(set(c) & set(s))
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4))
    ax[0].plot(steps, [s[t]["train_nll"] - c[t]["train_nll"] for t in steps], color="C0", lw=1.5, label="train loss, staged − control")
    vs = [t for t in steps if "validation_nll" in s[t] and "validation_nll" in c[t]]
    ax[0].plot(vs, [s[t]["validation_nll"] - c[t]["validation_nll"] for t in vs], "o", color="C0", label="validation, staged − control")
    for k, (name, switch) in enumerate(extras):
        e = rows(name)
        es = sorted(set(e) & set(c))
        color = ["#e0843a", "#3a9d5a", "#8e44ad"][k % 3]
        ax[0].plot(es, [e[t]["train_nll"] - c[t]["train_nll"] for t in es], color=color, lw=1.3, label=f"coordinated to step {switch}, then plain")
        ax[0].axvline(switch, color=color, lw=0.8, ls=":")
        ev = [t for t in es if "validation_nll" in e[t] and "validation_nll" in c[t]]
        ax[0].plot(ev, [e[t]["validation_nll"] - c[t]["validation_nll"] for t in ev], "o", ms=4, color=color)
        ek = [t for t in es if "gn_top" in e[t]]
        ax[1].plot(ek, [e[t]["gn_top"] for t in ek], "o-", color=color, label=f"coordinated to {switch}, then plain")
    ax[0].axhline(0, color="grey", lw=0.8)
    ax[0].set(xlabel="step (16M tokens)", ylabel="loss difference", title="Layer-staged PD: a mid-run lead that reverses" + (" and cannot be banked" if extras else ""))
    ax[0].legend(fontsize=8)
    ks = [t for t in steps if "gn_top" in c[t] and "gn_top" in s[t]]
    ax[1].plot(ks, [c[t]["gn_top"] for t in ks], "o-", color="k", label="control (plain PD, oscillating)")
    ax[1].plot(ks, [s[t]["gn_top"] for t in ks], "o-", color="C0", label="staged (coordinated steps)")
    ax[1].set(xlabel="step", ylabel="top GN eigenvalue (curvature batch)", title="Sharpness left by each optimizer", yscale="log")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()

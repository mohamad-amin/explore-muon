"""Four-panel summary figure for FINDINGS.md (reads step rows and probe JSON only).

.venv/bin/python logs/muon_spectra/improve_w4_20260925/plot_summary.py
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGS = Path(__file__).resolve().parent.parent


def curve(arm):
    out = {}
    for path in glob.glob(str(LOGS / arm / "scientific/steps/step*.json")):
        row = json.loads(Path(path).read_text())
        if "validation_nll" in row:
            out[row["step"]] = row["validation_nll"]
    return out


def diff(arm, base):
    a, b = curve(arm), curve(base)
    steps = sorted(s for s in a if s in b and s >= 300)
    return steps, [a[s] - b[s] for s in steps]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9))
    # (a) width 512, seed 260927 (fresh): methods vs tuned Muon
    ax = axes[0, 0]
    base = "improve_w11_20260925/M_lr0.007_s260927_a6000"
    for label, arm, color in (("SOAP-Muon core", "improve_w11_20260925/S_lr0.007_s260927_a6000", "tab:red"),
                              ("partial data-norm Muon (PD)", "improve_w11_20260925/PD_a0.25_lr0.01_s260927_a6000", "tab:blue"),
                              ("SOAP∘PD", "improve_w14_20260925/SPD_a0.25_lr0.01_s260927_a6000", "tab:purple")):
        steps, d = diff(arm, base)
        ax.plot(steps, d, color=color, label=f"{label} ({d[-1]:+.4f})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="(a) width 512, fresh seed 260927: vs tuned Muon@0.007", xlabel="step", ylabel="Δ val NLL")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # (b) alpha sweep at seed 260925 vs Muon@0.007
    ax = axes[0, 1]
    m7 = curve("improve_w4_20260925/M_lr0.007_s260925_a6000")[1469]
    arms = {0.125: "improve_w5_20260925/PD_a0.125_lr0.01_s260925_a6000", 0.25: "improve_w5_20260925/PD_a0.25_lr0.01_s260925_a6000",
            0.375: "improve_w10_20260925/PD_a0.375_lr0.01_s260925_a6000", 0.5: "improve_w10_20260925/PD_a0.5_lr0.01_s260925_a6000"}
    xs = sorted(arms)
    ys = [curve(arms[a])[1469] - m7 for a in xs]
    ax.plot([0] + xs, [0] + ys, "o-", color="tab:blue")
    gamma = json.loads((LOGS / "gamma_probe_20260925/gamma.json").read_text())
    g512 = gamma["M_w512_lr0.007_s260925_allkinds"]["median_gamma"]
    ax.axvline(g512 / 2, color="tab:green", ls="--", label=f"α = γ/2 = {g512 / 2:.2f} (measured γ = {g512:.2f})")
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="(b) data-norm exponent α (seed 260925): vs tuned Muon", xlabel="α  (0 = Muon, ½ = full data norm)",
           ylabel="Δ final val NLL")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    # (c) curvature vs input second moment (log-log), width 512
    ax = axes[1, 0]
    mats = gamma["M_w512_lr0.007_s260925_allkinds"]["matrices"]
    for name in ("block04.q", "block04.v", "block04.up", "block04.o", "block04.down"):
        rows = [r for r in mats[name]["rows"] if r["curvature"] > 0]
        lam = np.array([r["eigenvalue"] for r in rows]); cur = np.array([r["curvature"] for r in rows])
        ax.loglog(lam / lam.mean(), cur / cur.mean(), "o-", ms=3, label=f"{name} (γ={mats[name]['gamma']:.2f})")
    x = np.logspace(-3, 1.5, 10)
    ax.loglog(x, x ** 1.0, "k:", lw=0.8, label="slope 1 (full data norm)")
    ax.set(title="(c) loss curvature along input eigendirections (width 512)", xlabel="λ_j / mean λ",
           ylabel="curvature / mean")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
    # (d) width 768: PD vs tuned Muon, per seed
    ax = axes[1, 1]
    pairs = (("seed 260925 (selection, A6000)", "improve_w19_20260925/PD_w768_a0.25_lr0.01_s260925_a6000", "improve_w19_20260925/M_w768_lr0.007_s260925_a6000"),
             ("seed 260924 (fresh, L40S)", "improve_w19_20260925/PD_w768_a0.25_lr0.01_s260924", "improve_w19_20260925/M_w768_lr0.007_s260924"),
             ("seed 260926 (fresh, RTX 6000 Ada)", "improve_w19_20260925/PD_w768_a0.25_lr0.01_s260926_g20", "improve_w19_20260925/M_w768_lr0.007_s260926_g20"),
             ("seed 260927 (fresh, A6000)", "improve_w22_20260925/PD_w768_a0.25_lr0.01_s260927_a6000", "improve_w22_20260925/M_w768_lr0.007_s260927_a6000"))
    for label, pd, m in pairs:
        steps, d = diff(pd, m)
        if steps:
            tag = f" ({d[-1]:+.4f})" if steps[-1] == 2562 else f" (at {steps[-1]})"
            ax.plot(steps, d, label=label + tag)
    ax.axhline(0, color="k", lw=0.8)
    ax.set(title="(d) width 768 (134M): PD@0.01 vs tuned Muon@0.007", xlabel="step", ylabel="Δ val NLL")
    ax.set_ylim(-0.06, 0.01)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(__file__).resolve().parent / "plots" / "summary.png"
    fig.savefig(out, dpi=130)
    print(out)


if __name__ == "__main__":
    main()

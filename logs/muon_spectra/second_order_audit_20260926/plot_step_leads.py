"""Step leads along training for every change the audit compared at large batch (2026-09-29 07:4x CDT, after review).

The lead of A over B at step t: the (interpolated) step at which B's smoothed training loss reaches A's smoothed loss at
t, minus t (step_lead.py's rule; each step's training loss is on unseen data, centered running mean of +-half steps).
Loss gaps shrink as curves flatten; step leads do not, so they separate a lasting rate from a fading lead. Harness
branches (layer-staged PD and its switch-offs) are compared with the harness control on the same code.

usage: plot_step_leads.py OUT_PNG
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def curve_from(folder, half):
    rows = {}
    for f in folder.glob("step*.json"):
        r = json.loads(f.read_text())
        if "train_nll" in r:
            rows[r["step"]] = r["train_nll"]
    steps = np.array(sorted(rows))
    loss = np.array([rows[s] for s in steps])
    smooth = np.array([loss[max(0, i - half):i + half + 1].mean() for i in range(len(loss))])
    return steps, smooth


def leads(a, b, half):
    sa, la = curve_from(a, half)
    sb, lb = curve_from(b, half)
    out = []
    for t in sa[half:len(sa) - half]:
        target = float(np.interp(t, sa, la))
        below = np.where(lb <= target)[0]
        if len(below) == 0 or below[0] == 0:
            out.append((t, np.nan))
            continue
        i = below[0]
        t0, t1, l0, l1 = sb[i - 1], sb[i], lb[i - 1], lb[i]
        tb = t0 + (target - l0) / (l1 - l0) * (t1 - t0) if l1 != l0 else t1
        out.append((t, tb - t))
    return np.array(out)


def run(path):
    return ROOT / path / "scientific" / "steps"


def branch(name):
    return HERE / "staged_branch" / name / "steps"


PANELS = {
    "16M, 1× horizon (92 steps; cooldown from 84)": [
        ("PD vs Muon", run("soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
         run("soaudit_batch16m_20260927/M_b16M_lr0.02_mom0.9_s260925_l40s"), "#2e86c1", "-"),
        ("S∘PD vs PD", run("soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada"),
         run("soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"), "#8e44ad", "-"),
        ("TS vs PD", run("soaudit_batch16m_20260927/TS_a0.5_b0.5gn_b16M_lr0.028_mom0.9_s260925_l40s"),
         run("soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"), "#7f8c8d", "-"),
        ("PD β 0.8 vs β 0.9", run("soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s"),
         run("soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"), "#27ae60", "-"),
        ("layer-staged PD vs control (harness)", branch("staged_9_sharp"), branch("control_9_sharp"), "#c0392b", "-"),
        ("coordinated to 50, then plain", branch("switch_50"), branch("control_9_sharp"), "#e0843a", ":"),
    ],
    "momentum warmup 0.8 → 0.9 vs constant 0.9": [
        ("PD, 16M 2× (184 steps; cooldown from 166)", run("soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_l40s"),
         run("soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_l40s"), "#27ae60", "-"),
        ("S∘PD, 16M 2×, seed 260926", run("soaudit_warmrep_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260926_ada"),
         run("soaudit_warmrep_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260926_ada"), "#8e44ad", "-"),
        ("PD, 4M (368 steps; cooldown from 332)", run("soaudit_warm4m_20260928/PD_a0.5_b4M_lr0.02_momwarm0.8to0.9in120_s260925_ada"),
         run("soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada"), "#2e86c1", "-"),
    ],
}


def main():
    out = sys.argv[1]
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, (title, rows) in zip(axes, PANELS.items()):
        for label, a, b, color, style in rows:
            if not a.exists() or not b.exists():
                continue
            half = 5 if "4M" in label else 3
            d = leads(a, b, half)
            if len(d):
                ax.plot(d[:, 0], d[:, 1], style, color=color, lw=1.6, label=label)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set(xlabel="step", ylabel="lead of A over B (steps)", title=title)
        ax.legend(fontsize=7)
    axes[0].set_ylim(-8, 20)
    fig.suptitle("Leads measured in steps: every change holds or grows through the constant phase, except layer-staged coordination",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()

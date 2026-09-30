"""Momentum schedules against constant β at 16M 2× and 4M: smoothed training-loss gap to the β 0.9 run, and the
schedules themselves (2026-09-28 22:4x CDT; MUON_CASE decision "test the cube-root momentum law", 21:49 CDT).

usage: plot_momentum_law.py OUT_PNG
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
GROUPS = {
    "S∘PD 16M 2× (Ada)": ("soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada", {
        "β 0.8": "soaudit_horizon16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.8_s260925_ada",
        "warmup 0.8→0.9 (half)": "soaudit_momwarm16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada",
        "law 0.45→0.8 in 55": "soaudit_mlaw_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.45to0.8in55_s260925_ada",
        "β 0.7": "soaudit_mlaw_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.7_s260925_ada"}),
    "PD 16M 2× (L40S)": ("soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_l40s", {
        "warmup 0.8→0.9 (half)": "soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_l40s",
        "law 0.45→0.8 in 55": "soaudit_mlaw_20260928/PD_a0.5_b16M_T2x_lr0.028_momwarm0.45to0.8in55_s260925_l40s",
        "β 0.7": "soaudit_mlaw_20260928/PD_a0.5_b16M_T2x_lr0.028_mom0.7_s260925_l40s"}),
    "PD 4M (Ada)": ("soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada", {
        "β 0.8": "soaudit_mom4mb08_20260928/PD_a0.5_b4M_lr0.02_mom0.8_s260925_ada",
        "β 0.95": "soaudit_alpha4m_20260927/PD_a0.5_b4M_lr0.02_s260925_ada",
        "warmup 0.8→0.9 in 120": "soaudit_warm4m_20260928/PD_a0.5_b4M_lr0.02_momwarm0.8to0.9in120_s260925_ada",
        "β 0.85": "soaudit_mlaw_20260928/PD_a0.5_b4M_lr0.02_mom0.85_s260925_ada"}),
    "S∘PD 4M (L40S)": ("soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s", {
        "β 0.8": "soaudit_mom4mb08_20260928/SPD_a0.5_b4M_lr0.02_mom0.8_s260925_l40s",
        "β 0.95": "soaudit_alpha4m_20260927/SPD_a0.5_b4M_lr0.02_s260925_l40s",
        "warmup 0.8→0.9 in 120": "soaudit_warm4m_20260928/SPD_a0.5_b4M_lr0.02_momwarm0.8to0.9in120_s260925_l40s",
        "β 0.85": "soaudit_mlaw_20260928/SPD_a0.5_b4M_lr0.02_mom0.85_s260925_l40s"}),
}


def curve(arm, half):
    rows, val = {}, {}
    for f in (ROOT / arm / "scientific" / "steps").glob("step*.json"):
        r = json.loads(f.read_text())
        if "train_nll" in r:
            rows[r["step"]] = r["train_nll"]
        if "validation_nll" in r:
            val[r["step"]] = r["validation_nll"]
    steps = sorted(rows)
    smooth = {t: sum(rows[x] for x in range(t - half, t + half + 1) if x in rows) /
              max(1, sum(1 for x in range(t - half, t + half + 1) if x in rows)) for t in steps}
    return smooth, val


def main():
    fig, axes = plt.subplots(1, len(GROUPS), figsize=(5.2 * len(GROUPS), 4.4))
    for ax, (title, (ref, arms)) in zip(axes, GROUPS.items()):
        half = 3 if "16M" in title else 5
        base, base_val = curve(ref, half)
        for label, arm in arms.items():
            if not (ROOT / arm / "scientific" / "steps").exists():
                continue
            c, v = curve(arm, half)
            steps = sorted(set(c) & set(base))
            if not steps:
                continue
            final = ""
            if v and base_val and max(v) == max(base_val):
                final = f" (final {v[max(v)] - base_val[max(base_val)]:+.4f})"
            ax.plot(steps, [c[t] - base[t] for t in steps], label=label + final, lw=1.6 if "law" in label else 1.1)
        ax.axhline(0, color="grey", lw=0.8)
        ax.set(title=f"{title}: minus β 0.9", xlabel="step", ylabel="smoothed training loss difference")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=130)
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()

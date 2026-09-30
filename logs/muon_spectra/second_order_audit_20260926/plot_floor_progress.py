"""Floor progress vs stored excess along training, and step-equivalent speedups.

(1) For each run and each kept step t with a 16-step anneal branch (anneal_branch.py), the held-out loss before the
    branch (raw, the loss the run is at) and after it (the floor under the oscillation), on the same EVAL sequences.
(2) The stored excess (raw - floor) against step.
(3) Steps each arm needs to reach the reference arm's validation loss at the reference's step t, as a speedup t / that.

usage: plot_floor_progress.py OUT_PNG
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
ANNEALS = {   # label: (color, [branch dirs]) with the branch step in the directory name (..._a<step>)
    "Muon 1M @0.007": ("C0", sorted(HERE.glob("valley/M_lr0.007_s260925_l40s_a*"))),
    "PD α¼ 1M @0.01": ("C3", sorted(HERE.glob("valley/PD_a0.25_lr0.01_s260925_ada_a*"))),
    "Muon 4M β0.9 @0.014": ("C0", sorted(HERE.glob("valley/M_b4M_lr0.014_mom0.9_s260925_l40s_a*"))),
    "PD α½ 4M β0.9 @0.02": ("C3", sorted(HERE.glob("valley/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada_a*"))),
}
SPEEDUP_REF = ROOT / "soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s"
SPEEDUP_ARMS = {
    "PD α¼ @0.02": ROOT / "soaudit_mom4m3_20260927/PD_b4M_lr0.02_mom0.9_s260925_ada",
    "PD α½ @0.02": ROOT / "soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada",
    "TS ½/½ @0.02": ROOT / "soaudit_strength4m_20260927/TS_a0.5_b0.5gn_b4M_lr0.02_mom0.9_s260925_ada",
    "S∘PD α½ @0.02": ROOT / "soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s",
}


def curve(arm):
    out = {}
    for f in (Path(arm) / "scientific" / "steps").glob("step*.json"):
        r = json.loads(f.read_text())
        if "validation_nll" in r:
            out[r["step"]] = r["validation_nll"]
    return out


def steps_to(c, target):
    s = np.array(sorted(c))
    v = np.array([c[k] for k in s])
    for i in range(1, len(s)):
        if v[i] <= target < v[i - 1]:
            f = (v[i - 1] - target) / (v[i - 1] - v[i])
            return float(np.exp(np.log(max(s[i - 1], 1)) + f * (np.log(s[i]) - np.log(max(s[i - 1], 1)))))
    return float("nan")


def anneal_points(dirs):
    points = []
    for d in dirs:
        f = d / "anneal.json"
        if not f.exists():
            continue
        r = json.loads(f.read_text())
        points.append((r["start_step"], r["loss"][0], r["loss"][-1]))     # held-out before the branch, after its last step
    return sorted(points)


def main():
    out = Path(sys.argv[1])
    fig, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for row, batch in enumerate(("1M", "4M")):
        ax_loss, ax_excess = axes[row, 0], axes[row, 1]
        for label, (color, dirs) in ANNEALS.items():
            if batch not in label:
                continue
            pts = anneal_points(dirs)
            if not pts:
                continue
            steps, raw, floor = (np.array(x) for x in zip(*pts))
            ax_loss.plot(steps, raw, "o--", color=color, alpha=0.6, label=f"{label}: raw")
            ax_loss.plot(steps, floor, "s-", color=color, label=f"{label}: floor (16-step anneal)")
            ax_excess.plot(steps, raw - floor, "o-", color=color, label=label)
        ax_loss.set_xlabel("step")
        ax_loss.set_ylabel("held-out loss (EVAL sequences)")
        ax_loss.set_title(f"{batch}: raw loss and the floor under it")
        ax_loss.legend(fontsize=7)
        ax_excess.set_xlabel("step")
        ax_excess.set_ylabel("stored excess = raw - floor")
        ax_excess.set_title(f"{batch}: loss stored in the oscillation")
        ax_excess.legend(fontsize=7)
    ax = axes[0, 2]
    ref = curve(SPEEDUP_REF)
    ts = [t for t in sorted(ref) if 50 <= t <= 330]
    for label, arm in SPEEDUP_ARMS.items():
        c = curve(arm)
        ax.plot(ts, [t / steps_to(c, ref[t]) for t in ts], "o-", label=label)
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xlabel("Muon's step t (4M, β 0.9, @0.014)")
    ax.set_ylabel("speedup: t / steps the arm needs for Muon's loss at t")
    ax.set_title("4M: step-equivalent speedup over Muon")
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    m = curve(ROOT / "soaudit_traj_20260926/M_lr0.007_s260925_l40s")
    p = curve(ROOT / "soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada")
    ts = [t for t in sorted(m) if 50 <= t <= 1300]
    ax.plot(ts, [t / steps_to(p, m[t]) for t in ts], "o-", color="C3", label="PD α¼ @0.01 vs Muon @0.007")
    ax.axhline(1, color="k", lw=0.5)
    ax.set_xlabel("Muon's step t (1M, β 0.95)")
    ax.set_ylabel("speedup")
    ax.set_title("1M: step-equivalent speedup over Muon")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()

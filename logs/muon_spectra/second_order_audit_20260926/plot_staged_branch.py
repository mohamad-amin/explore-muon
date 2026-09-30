"""Figure for the staged-PD branch tests (newton_train.py --staged-pd): train loss, the staged - control gap, and the
pre-clip gradient norm per step, with the original run for reference (2026-09-29 04:4x CDT).

usage: plot_staged_branch.py OUT_PNG RUN_ARM CONTROL_DIR BRANCH_DIR [BRANCH_DIR ...]   (dirs under staged_branch/)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def rows(path):
    out = {}
    for f in sorted(path.glob("step*.json")):
        r = json.loads(f.read_text())
        out[r["step"]] = r
    return out


def main():
    out, run_arm, control, branches = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    run = rows(ROOT / run_arm / "scientific" / "steps")
    ctrl = rows(HERE / "staged_branch" / control / "steps")
    data = {b: rows(HERE / "staged_branch" / b / "steps") for b in branches}
    steps = sorted(ctrl)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    axes[0].plot(steps, [run[s]["train_nll"] for s in steps if s in run][:len(steps)], color="grey", lw=1, label="original run")
    axes[0].plot(steps, [ctrl[s]["train_nll"] for s in steps], color="k", lw=1.4, label=f"{control} (no coordination)")
    for b, d in data.items():
        st = sorted(set(d) & set(ctrl))
        axes[0].plot(st, [d[s]["train_nll"] for s in st], lw=1.6, label=b)
        axes[1].plot(st, [d[s]["train_nll"] - ctrl[s]["train_nll"] for s in st], lw=1.6, label=f"{b} − control (train)")
        vs = [s for s in st if "validation_nll" in d[s] and "validation_nll" in ctrl[s]]
        axes[1].plot(vs, [d[s]["validation_nll"] - ctrl[s]["validation_nll"] for s in vs], "o", ms=5, label=f"{b} − control (validation)")
        axes[2].plot(st, [d[s]["grad_norm_before_clip"] for s in st], lw=1.4, label=b)
    axes[2].plot(steps, [ctrl[s]["grad_norm_before_clip"] for s in steps], color="k", lw=1.4, label="control")
    axes[1].axhline(0, color="grey", lw=0.8)
    axes[0].set(xlabel="step (16M tokens each)", ylabel="training loss on the run's batch", title="Layer-staged PD vs its control")
    axes[1].set(xlabel="step", ylabel="loss difference", title="Staged − control")
    axes[2].set(xlabel="step", ylabel="pre-clip gradient norm", title="Gradient left behind by the step")
    for ax in axes:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()

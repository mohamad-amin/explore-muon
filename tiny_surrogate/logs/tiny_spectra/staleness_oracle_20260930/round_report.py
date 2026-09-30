"""Round readout for the 1-GPU (p1_*) runs: validation curves, difference from the control, step-equivalent ratios.

Usage (from tiny_surrogate/): ./run logs/tiny_spectra/staleness_oracle_20260930/round_report.py <set>
Sets are defined in SETS below. The step-equivalent ratio of an arm at step s is the reference's step reaching the
same validation loss (linear interpolation on the reference's curve) divided by s; > 1 means faster. It is
reported in the constant-LR phase (after warmup, before the cooldown), where both runs share the schedule, and at
the last evaluation where the arm's loss lies inside the reference's range. Final losses are the complete-event
full validation NLL (console log) when the run has finished, else the last bank evaluation.
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = Path("logs/tiny_spectra/staleness_oracle_20260930")
SETS = {
    "262k": dict(control="p1_pd262k_control", second="p1_pd262k_beta08", arms=[
        ("p1_pd262k_control", "β 0.9 (control)", "black"), ("p1_pd262k_beta08", "β 0.8", "#7f7f7f"),
        ("p1_pd262k_warm", "β 0.8→0.9 (first third)", "#bcbd22"),
        ("p1_pd262k_split05", "split, short β 0.5", "#2ca02c"), ("p1_pd262k_split07", "split, short β 0.7", "#17becf"),
        ("p1_pd262k_splitsched", "split, short β 0.5→0.9", "#1f77b4")]),
    "1m2x": dict(control="p1_pd1m2x_control", second="p1_pd1m2x_beta08", arms=[
        ("p1_pd1m2x_control", "β 0.9 (control)", "black"), ("p1_pd1m2x_beta08", "β 0.8", "#7f7f7f"),
        ("p1_pd1m2x_warm", "β 0.8→0.9 (first third)", "#bcbd22"), ("p1_pd1m2x_split05", "split, short β 0.5", "#2ca02c"),
        ("p1_pd1m2x_splitsched", "split, short β 0.5→0.9", "#1f77b4")]),
    "a05": dict(control="p1_pd1m_a05_control", second="p1_pd1m_a05_beta08", arms=[
        ("p1_pd1m_a05_control", "α ½, β 0.9", "black"), ("p1_pd1m_a05_beta08", "α ½, β 0.8", "#7f7f7f"),
        ("p1_pd1m_a05_split05", "α ½, split 0.5", "#2ca02c"),
        ("p1_pd1m_auxx1_control", "α ¼, β 0.9", "#444444"), ("p1_pd1m_auxx1_split05", "α ¼, split 0.5", "#98df8a")]),
    "twopole": dict(control="p1_pd1m_auxx1_control", second="p1_pd1m_auxx1_beta08", arms=[
        ("p1_pd1m_auxx1_control", "β 0.9 (control)", "black"), ("p1_pd1m_auxx1_beta08", "β 0.8", "#7f7f7f"),
        ("p1_pd1m_auxx1_split05", "split, short β 0.5", "#2ca02c"),
        ("p1_pd1m_auxx1_twopole_w05", "uniform two-pole 0.5/0.9, w 0.5", "#d62728"),
        ("p1_pd1m_auxx1_twopole_w025", "uniform two-pole 0.5/0.9, w 0.25", "#ff7f0e"),
        ("p1_pd1m_auxx1_bimaxwell", "Bi-Maxwell 0.85/0.98, w 0.44, on at 35%", "#9467bd")]),
}
for aux in ("x1", "x2", "x4", "x8"):
    SETS["aux" + aux] = dict(control=f"p1_pd1m_aux{aux}_control", second=f"p1_pd1m_aux{aux}_beta08", arms=[
        (f"p1_pd1m_aux{aux}_control", "β 0.9 (control)", "black"), (f"p1_pd1m_aux{aux}_beta08", "β 0.8", "#7f7f7f"),
        (f"p1_pd1m_aux{aux}_split05", "split, short β 0.5", "#2ca02c")])


def load(run):
    path = C / "runs" / run / "metrics.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    val = [(r["step"], r["validation_nll"]) for r in rows if "validation_nll" in r and r.get("step", 0) > 0]
    lr = {r["step"]: r["lr"] for r in rows if "lr" in r}
    final = None
    for log in sorted((C / "console").glob(run + "_*.log")):
        for line in log.read_text().splitlines():
            if line.startswith('{"event": "complete"'):
                final = json.loads(line)["full_validation_nll"]
    return dict(val=val, lr=lr, final=final, last_step=max(lr) if lr else 0)


def step_equivalent(ref, points):
    rs, rv = map(np.array, zip(*ref))
    order = np.argsort(rv)
    rv, rs = rv[order], rs[order]
    out = []
    for s, v in points:
        out.append((s, float(np.interp(v, rv, rs)) / s if rv[0] <= v <= rv[-1] else np.nan))
    return out


def constant_phase(lr):
    peak = max(lr.values())
    steps = sorted(s for s, x in lr.items() if abs(x - peak) < 1e-12)
    return (steps[0], steps[-1]) if steps else (0, 0)


def main():
    name = sys.argv[1]
    spec = SETS[name]
    data = {run: load(run) for run, _, _ in spec["arms"]}
    table = {}
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for ref_key, ax in (("control", axes[1, 0]), ("second", axes[1, 1])):
        ref = data.get(spec[ref_key])
        if not ref or not ref["val"]:
            continue
        lo, hi = constant_phase(ref["lr"])
        for run, label, color in spec["arms"]:
            d = data.get(run)
            if not d or not d["val"] or run == spec[ref_key]:
                continue
            ratios = step_equivalent(ref["val"], d["val"])
            ax.plot(*zip(*ratios), color=color, label=label)
            inside = [r for s, r in ratios if lo <= s <= hi and not np.isnan(r)]
            last = [(s, r) for s, r in ratios if not np.isnan(r)]
            entry = table.setdefault(run, dict(label=label))
            entry[f"ratio_vs_{ref_key}_constant_median"] = float(np.median(inside)) if inside else None
            entry[f"ratio_vs_{ref_key}_last_in_range"] = last[-1] if last else None
        ax.axhline(1, color="0.5", lw=0.8)
        ax.axvspan(hi, ref["last_step"], color="0.9")
        ax.set(title=f"Step-equivalent ratio vs {spec[ref_key]} (>1 = faster; grey: cooldown)", xlabel="step",
               ylim=(0.8, 1.3))
    control = data.get(spec["control"])
    for run, label, color in spec["arms"]:
        d = data.get(run)
        if not d or not d["val"]:
            continue
        s, v = map(np.array, zip(*d["val"]))
        axes[0, 0].plot(s, v, color=color, label=label)
        entry = table.setdefault(run, dict(label=label))
        entry.update(final=d["final"], last_step=d["last_step"], last_val=float(v[-1]))
        if control and control["val"] and run != spec["control"]:
            cval = dict(control["val"])
            common = [x for x in s if x in cval]
            axes[0, 1].plot(common, [v[list(s).index(x)] - cval[x] for x in common], color=color, label=label)
    base = table.get(spec["control"], {}).get("final")
    second = table.get(spec["second"], {}).get("final")
    for run, entry in table.items():
        if entry.get("final") is not None:
            entry["final_minus_control"] = entry["final"] - base if base is not None else None
            entry["final_minus_second"] = entry["final"] - second if second is not None else None
    lows = [x for d in data.values() if d and d["val"] for _, x in d["val"]]
    axes[0, 0].set(title="Validation NLL (bank)", xlabel="step", ylim=(min(lows) - 0.02, min(lows) + 1.0) if lows else None)
    axes[0, 1].set(title="Difference from the control", xlabel="step", ylim=(-0.12, 0.08))
    axes[0, 1].axhline(0, color="0.5", lw=0.8)
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.suptitle(f"Round readout: {name}")
    fig.tight_layout()
    (C / "figures").mkdir(exist_ok=True)
    fig.savefig(C / "figures" / f"round_{name}.png", dpi=110)
    (C / f"round_{name}.json").write_text(json.dumps(table, indent=1, default=float) + "\n")
    for run, entry in table.items():
        print(f"{run:32s} final {entry.get('final')!s:>20s}  d_ctrl {entry.get('final_minus_control')!s:>24s}  "
              f"d_second {entry.get('final_minus_second')!s:>24s}  const_vs_ctrl {entry.get('ratio_vs_control_constant_median')!s:>20s}  "
              f"const_vs_second {entry.get('ratio_vs_second_constant_median')!s:>20s}  last_vs_second {entry.get('ratio_vs_second_last_in_range')!s}")


if __name__ == "__main__":
    main()

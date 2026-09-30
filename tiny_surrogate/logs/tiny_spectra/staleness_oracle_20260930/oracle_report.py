"""Staleness-oracle readout: final NLLs, curves, step-equivalent speedups and oracle diagnostics."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

C = Path("logs/tiny_spectra/staleness_oracle_20260930")
ARMS = [("pd1m_control", "β 0.9 (control)", "black"), ("pd1m_beta08", "β 0.8", "#7f7f7f"),
        ("pd1m_beta05", "β 0.5", "#bcbd22"), ("pd1m_beta0", "β 0", "#8c564b"),
        ("pd1m_oracle_full", "stale-free oracle", "#d62728"),
        ("pd1m_oracle_rest_matched", "stale-free outside top-16, matched norm", "#e377c2"),
        ("pd1m_split_gn_rest", "split: β 0.5 outside top-16 GN, 0.9 inside", "#1f77b4"),
        ("pd1m_split_gn_stiff", "split: β 0.5 inside top-16 GN, 0.9 outside", "#9467bd"),
        ("pd1m_split_band_rest", "split: β 0.5 below-mean inputs, 0.9 above", "#2ca02c"),
        ("pd1m_split_band_stiff", "split: β 0.5 above-mean inputs, 0.9 below", "#ff7f0e")]


def curve(run):
    path = C / "runs" / run / "metrics.jsonl"
    if not path.exists():
        return None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    return rows


def step_equivalent(control_steps, control_loss, steps, loss):
    """For each arm point, the control step reaching the same validation loss (linear interpolation)."""
    out = []
    order = np.argsort(control_loss)
    cl, cs = np.array(control_loss)[order], np.array(control_steps)[order]
    for s, value in zip(steps, loss):
        if value < cl[0] or value > cl[-1]:
            out.append((s, np.nan))
            continue
        out.append((s, float(np.interp(value, cl, cs)) / s))
    return out


def main():
    table, curves = {}, {}
    for run, label, color in ARMS:
        rows = curve(run)
        if not rows:
            continue
        curves[run] = rows
        summary = C / "runs" / run / "summary.json"
        table[run] = dict(label=label, final=json.loads(summary.read_text())["full_validation_nll"]
                          if summary.exists() else None, last_step=rows[-1]["step"])
    control = curves.get("pd1m_control")
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for run, label, color in ARMS:
        rows = curves.get(run)
        if not rows:
            continue
        val = [(r["step"], r["validation_nll"]) for r in rows if "validation_nll" in r and r["step"] > 0]
        s, v = map(np.array, zip(*val))
        axes[0, 0].plot(s, v, color=color, label=label)
        if control and run != "pd1m_control":
            cval = {r["step"]: r["validation_nll"] for r in control if "validation_nll" in r}
            common = [x for x in s if x in cval]
            axes[0, 1].plot(common, [v[list(s).index(x)] - cval[x] for x in common], color=color, label=label)
            cs, cv = map(np.array, zip(*sorted((k, w) for k, w in cval.items() if k > 0)))
            ratios = step_equivalent(cs, cv, s, v)
            axes[1, 0].plot([a for a, b in ratios], [b for a, b in ratios], color=color, label=label)
            late = [b for a, b in ratios if a >= 40 and not np.isnan(b)]
            table[run]["step_equivalent_median_after_40"] = float(np.median(late)) if late else None
        norms = [(r["step"], r["gradient_norm_before_clip"]) for r in rows if "gradient_norm_before_clip" in r]
        axes[1, 1].plot(*zip(*norms), color=color, label=label, alpha=0.8)
        for key in ("cos_used_normal", "used_over_normal_norm", "transport_over_momentum", "correction_inside_share"):
            values = [r[key] for r in rows if key in r]
            if values:
                table[run][key + "_median"] = float(np.median(values))
    axes[0, 0].set(title="Validation NLL (bank, every 4 steps)", xlabel="step", ylim=(5.3, 7.0))
    axes[0, 1].set(title="Difference from the control", xlabel="step")
    axes[0, 1].axhline(0, color="0.5", lw=0.8)
    axes[1, 0].set(title="Step-equivalent ratio vs control (>1 = faster)", xlabel="step", ylim=(0.5, 2.0))
    axes[1, 0].axhline(1, color="0.5", lw=0.8)
    axes[1, 1].set(title="Pre-clip gradient norm", xlabel="step", yscale="log")
    for ax in axes.flat:
        ax.legend(fontsize=8)
    fig.tight_layout()
    (C / "figures").mkdir(exist_ok=True)
    fig.savefig(C / "figures" / "oracle.png", dpi=120)
    if control and "pd1m_control" in table and table["pd1m_control"]["final"] is not None:
        base = table["pd1m_control"]["final"]
        for run in table:
            if table[run]["final"] is not None:
                table[run]["final_minus_control"] = table[run]["final"] - base
    (C / "results.json").write_text(json.dumps(table, indent=1) + "\n")
    print(json.dumps(table, indent=1, default=str))


if __name__ == "__main__":
    main()

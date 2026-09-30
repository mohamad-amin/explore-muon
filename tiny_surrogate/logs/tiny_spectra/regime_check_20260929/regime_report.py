"""Figures, summary and the predeclared decision checks for the regime check (PROTOCOL.md review addendum).

Reads analysis/<run>/{edge,momentum,gains}.json; writes analysis/figures/*.png and analysis/verdict.json.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COHORT = Path("logs/tiny_spectra/regime_check_20260929")
RUNS = [("muon_b262144_lr0.01_s20261001", "Muon 262K", "#1f77b4", "-", "262K"),
        ("pd_b262144_lr0.01_s20261001", "PD 262K", "#d62728", "-", "262K"),
        ("muon_b1048576_lr0.01_s20261001", "Muon 1M", "#1f77b4", "--", "1M"),
        ("pd_b1048576_lr0.02_s20261001", "PD 1M", "#d62728", "--", "1M")]
PRE_COOLDOWN = {"262K": [120, 183, 280], "1M": [46, 65, 83]}


def load(base, run, name):
    path = base / run / name
    return json.loads(path.read_text()) if path.exists() else None


def mid(edge):
    return edge["windows"][-1]["summary"] if edge and edge.get("windows") else None


def main():
    base = COHORT / (sys.argv[1] if len(sys.argv) > 1 else "analysis")
    figures = base / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    data = {run: dict(edge=load(base, run, "edge.json"), momentum=load(base, run, "momentum.json"),
                      gains=load(base, run, "gains.json")) for run, *_ in RUNS}
    verdict = dict(a={}, b={}, c={}, checks={})

    # ------------------------------------------------------------------ edge figure
    fig, axes = plt.subplots(2, 3, figsize=(21, 10))
    ax = axes[0, 0]
    for run, label, color, style, _ in RUNS:
        edge = data[run]["edge"]
        if not edge or not edge["sharpness"]:
            continue
        T = edge["total_steps"]
        x = np.array([r["step"] for r in edge["sharpness"]]) / T
        ax.plot(x, [r["lambda_max"] for r in edge["sharpness"]], style, color=color, label=label)
    ax.set(title="Top Gauss-Newton eigenvalue, body matrices", xlabel="fraction of training", yscale="log")
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for run, label, color, style, _ in RUNS:
        edge = data[run]["edge"]
        if not edge or not edge["sharpness"] or not run.startswith("muon"):
            continue
        T = edge["total_steps"]
        x = np.array([r["step"] for r in edge["sharpness"]]) / T
        ax.plot(x, [r["lr_times_lambda"] for r in edge["sharpness"]], style, color=color, label=label)
    ax.axhspan(0.07, 0.11, color="0.85", zorder=0, label="reference Muon band")
    ax.set(title="Muon: LR × top eigenvalue (description only)", xlabel="fraction of training", yscale="log")
    ax.legend(fontsize=8)
    ax = axes[0, 2]
    for run, label, color, style, _ in RUNS:
        edge = data[run]["edge"]
        if not edge:
            continue
        T = edge["total_steps"]
        for window in edge["windows"]:
            steps = window["steps"]
            x = np.array([s["step"] for s in steps]) / T
            c = np.array([np.nanmean([s["c_star_A"], s["c_star_B"]]) for s in steps])
            ax.plot(x, c, style, color=color, marker=".", label=label if window is edge["windows"][0] else None)
    ax.axhspan(0.3, 0.75, color="0.9", zorder=0)
    ax.axhline(0.5, color="0.5", lw=0.8)
    ax.set(title="c* of the body step (½ = edge; grey = E1 band)", xlabel="fraction of training", ylim=(-1, 3))
    ax.legend(fontsize=8)
    ax = axes[1, 0]
    for i, (run, label, color, style, _) in enumerate(RUNS):
        edge = data[run]["edge"]
        if not edge:
            continue
        for k, window in enumerate(edge["windows"]):
            stiff = [p["stiff_cos_cross"] for p in window["pairs"]]
            rest = [p["rest_cos_cross"] for p in window["pairs"]]
            x0 = i * 3 + k * 1.2
            ax.boxplot([stiff], positions=[x0], widths=0.45, patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=0.8), showfliers=False)
            ax.boxplot([rest], positions=[x0 + 0.5], widths=0.45, patch_artist=True,
                       boxprops=dict(facecolor="white"), showfliers=False)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.axhline(-0.3, color="0.5", lw=0.8, ls=":")
    ax.set(title="Consecutive fixed-probe gradients across disjoint probes\nfilled = top-16 GN, open = rest; "
                 "per run: early window, mid window", xticks=[i * 3 + 0.85 for i in range(4)],
           xticklabels=[r[1] for r in RUNS], ylim=(-1.05, 1.05))
    ax = axes[1, 1]
    for i, (run, label, color, style, _) in enumerate(RUNS):
        edge = data[run]["edge"]
        if not edge:
            continue
        for k, window in enumerate(edge["windows"]):
            ax.bar(i * 3 + k, window["summary"]["body_loss_increase_fraction"], color=color, alpha=0.5 + 0.4 * k)
    ax.axhline(0.3, color="0.5", lw=0.8, ls=":")
    ax.axhline(0.5, color="0.5", lw=0.8)
    ax.set(title="Share of body-only steps raising the probe loss (early, mid)", xticks=[i * 3 + 0.5 for i in range(4)],
           xticklabels=[r[1] for r in RUNS], ylim=(0, 1))
    ax = axes[1, 2]
    for i, (run, label, color, style, _) in enumerate(RUNS):
        edge = data[run]["edge"]
        if not edge:
            continue
        for k, window in enumerate(edge["windows"]):
            ax.bar(i * 3 + k, window["summary"]["stiff_energy_over_chance_median"], color=color, alpha=0.5 + 0.4 * k)
    ax.axhline(1, color="0.5", lw=0.8)
    ax.set(title="Stiff energy of the steps ÷ chance (top-16 GN), early and mid", yscale="log",
           xticks=[i * 3 + 0.5 for i in range(4)], xticklabels=[r[1] for r in RUNS])
    fig.tight_layout()
    fig.savefig(figures / "edge.png", dpi=120)
    plt.close(fig)

    for batch in ("262K", "1M"):
        muon = next(r for r in RUNS if r[4] == batch and r[0].startswith("muon"))[0]
        pd = next(r for r in RUNS if r[4] == batch and r[0].startswith("pd"))[0]
        m = mid(data[muon]["edge"])
        if not m:
            continue
        c_star = np.nanmean([m["c_star_A_median"], m["c_star_B_median"]])
        e1 = 0.3 <= c_star <= 0.75
        e2 = m["stiff_cos_cross_median"] <= -0.3 and m["stiff_cos_cross_median"] <= m["rest_cos_cross_median"] - 0.3
        e3 = m["body_loss_increase_fraction"] >= 0.3
        entry = dict(c_star=float(c_star), E1=bool(e1), E2=bool(e2), E3=bool(e3), present=bool(e2 and (e1 or e3)),
                     stiff_cos_cross=m["stiff_cos_cross_median"], rest_cos_cross=m["rest_cos_cross_median"],
                     body_loss_increase=m["body_loss_increase_fraction"])
        p = mid(data[pd]["edge"])
        if p:
            entry["pd"] = dict(c_star=float(np.nanmean([p["c_star_A_median"], p["c_star_B_median"]])),
                               stiff_cos_cross=p["stiff_cos_cross_median"], rest_cos_cross=p["rest_cos_cross_median"],
                               body_loss_increase=p["body_loss_increase_fraction"])
            entry["pd_over_muon_stiff_energy"] = p["stiff_energy_median"] / m["stiff_energy_median"]
        verdict["a"][batch] = entry

    # ------------------------------------------------------------------ momentum figure
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    for ci, cls in enumerate(("gn_top", "gn_rest", "all")):
        ax = axes[ci]
        for ri, (run, label, color, style, _) in enumerate(RUNS):
            momentum = data[run]["momentum"]
            if not momentum:
                continue
            for si, r in enumerate(momentum):
                c = r["classes"][cls]
                sig = max(c["signal"], 1e-30)
                x = ri * 6 + si
                ax.bar(x - 0.2, c["staleness"] / sig, width=0.4, color=color)
                ax.bar(x + 0.2, c["noise"] / sig, width=0.4, color=color, alpha=0.35, hatch="//")
        ax.set(title=f"Momentum error ÷ signal, {cls}: solid staleness, hatched noise\n"
                     "(5 states per run, early to final)", yscale="log", xticks=[ri * 6 + 2 for ri in range(4)],
               xticklabels=[r[1] for r in RUNS])
        ax.axhline(1, color="0.5", lw=0.8)
    fig.tight_layout()
    fig.savefig(figures / "momentum_error.png", dpi=120)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8), sharey=True)
    for ax, cls in zip(axes, ("gn_top", "gn_rest")):
        for run, label, color, style, _ in RUNS:
            momentum = data[run]["momentum"]
            if not momentum:
                continue
            T = data[run]["edge"]["total_steps"] if data[run]["edge"] else max(r["step"] for r in momentum)
            x = np.array([r["step"] for r in momentum]) / T
            ax.plot(x, [r["classes"][cls]["cos_momentum_fresh"] for r in momentum], style, color=color, marker="o",
                    label=label + ": momentum")
            ax.plot(x, [r["classes"][cls]["cos_stale_free_fresh"] for r in momentum], style, color=color, marker="x",
                    alpha=0.5)
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set(title=f"cos with the fresh mean gradient, {cls} (o momentum, x stale-free)",
               xlabel="fraction of training", ylim=(-1.05, 1.05))
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "momentum_cos.png", dpi=120)
    plt.close(fig)

    for run, label, color, style, batch in RUNS:
        momentum = data[run]["momentum"]
        if not momentum:
            continue
        states, checks = [], []
        for r in momentum:
            top, rest, whole = r["classes"]["gn_top"], r["classes"]["gn_rest"], r["classes"]["all"]
            gate = (0.7 <= whole["replay_over_fresh_spread"] <= 1.4
                    and whole["replay_gap_over_fresh"] <= 2 * whole["replay_gap_expected"]
                    and (r.get("ns_step_check") is None or r["ns_step_check"]["median_cos"] >= 0.999))
            cond = (top["cos_momentum_fresh"] <= -0.2 and top["cos_stale_free_fresh"] >= 0.3
                    and top["staleness"] > top["noise"])
            checks.append(dict(step=r["step"], gate=bool(gate), ns=r.get("ns_step_check"),
                               final=r.get("final_momentum_check"), norm_check=r["norm_check_max_relative"],
                               spread_ratio=whole["replay_over_fresh_spread"],
                               replay_gap=whole["replay_gap_over_fresh"], replay_gap_expected=whole["replay_gap_expected"],
                               kappa_mean=r["kappa_mean"], clipped_fraction=r["clipped_fraction"]))
            if r["step"] in PRE_COOLDOWN[batch]:
                states.append(dict(step=r["step"], condition=bool(cond), gate=bool(gate),
                                   top_cos_momentum=top["cos_momentum_fresh"], top_cos_stale_free=top["cos_stale_free_fresh"],
                                   rest_cos_momentum=rest["cos_momentum_fresh"],
                                   top_staleness_over_signal=top["staleness"] / max(top["signal"], 1e-30),
                                   top_noise_over_signal=top["noise"] / max(top["signal"], 1e-30),
                                   all_staleness_over_signal=whole["staleness"] / max(whole["signal"], 1e-30),
                                   all_noise_over_signal=whole["noise"] / max(whole["signal"], 1e-30),
                                   momentum_stiff_share_over_chance=r["momentum_stiff_share_over_chance"]))
        passing = sum(1 for s in states if s["condition"] and s["gate"])
        top_median = float(np.median([s["top_cos_momentum"] for s in states])) if states else None
        rest_median = float(np.median([s["rest_cos_momentum"] for s in states])) if states else None
        verdict["b"][run] = dict(states=states, passing_states=passing, present=passing >= 2,
                                 top_cos_median=top_median, rest_cos_median=rest_median,
                                 top_below_rest_by_0p2=(top_median is not None and top_median <= rest_median - 0.2))
        verdict["checks"][run] = checks
    # Same (b) rule on the exact rebuilds (momentum from step 1, the trainer's 4-sequence replay microbatch),
    # run after the planned K = 40 rebuild failed the Newton-Schulz gate at 1M.
    verdict["b_exact"] = {}
    for run, label, color, style, batch in RUNS:
        exact = load(base, run, "momentum_exact.json")
        if not exact:
            continue
        states = []
        for r in exact:
            top, rest, whole = r["classes"]["gn_top"], r["classes"]["gn_rest"], r["classes"]["all"]
            gate = (0.7 <= whole["replay_over_fresh_spread"] <= 1.4
                    and whole["replay_gap_over_fresh"] <= 2 * whole["replay_gap_expected"]
                    and (r.get("ns_step_check") is None or r["ns_step_check"]["median_cos"] >= 0.999))
            cond = (top["cos_momentum_fresh"] <= -0.2 and top["cos_stale_free_fresh"] >= 0.3
                    and top["staleness"] > top["noise"])
            states.append(dict(step=r["step"], condition=bool(cond), gate=bool(gate), ns=r.get("ns_step_check"),
                               top_cos_momentum=top["cos_momentum_fresh"], rest_cos_momentum=rest["cos_momentum_fresh"],
                               top_staleness_over_signal=top["staleness"] / max(top["signal"], 1e-30),
                               top_noise_over_signal=top["noise"] / max(top["signal"], 1e-30)))
        passing = sum(1 for x in states if x["condition"] and x["gate"] and x["step"] in PRE_COOLDOWN[batch])
        verdict["b_exact"][run] = dict(states=states, passing_states=passing, present=passing >= 2)
    for method in ("muon", "pd"):
        small = verdict["b"].get(f"{method}_b262144_lr0.01_s20261001")
        large = verdict["b"].get(next(r[0] for r in RUNS if r[4] == "1M" and r[0].startswith(method)))
        if small and large and small["top_cos_median"] is not None and large["top_cos_median"] is not None:
            verdict["b"][f"{method}_1M_no_weaker_than_262K"] = large["top_cos_median"] <= small["top_cos_median"] + 0.05

    # ------------------------------------------------------------------ gains figure
    fig, axes = plt.subplots(1, 3, figsize=(21, 5))
    for run, label, color, style, _ in RUNS:
        gains = data[run]["gains"]
        if not gains:
            continue
        trajectory = gains["trajectory"]
        T = max(r["step"] for r in trajectory) or 1
        x = np.array([r["step"] for r in trajectory]) / T
        for ax, key in ((axes[0], "ln1.weight"), (axes[1], "ln2.weight")):
            values = np.array([[r[n]["max_over_median"] for n in r if n.endswith(key)] for r in trajectory])
            ax.plot(x, values.max(1), style, color=color, label=label + " (worst layer)")
            ax.plot(x, np.median(values, 1), style, color=color, alpha=0.4)
        for f in gains["fold"]:
            xs = f["step"] / T
            axes[2].scatter([xs], [np.median([r["muon_ns_data_cos"] for r in f["rows"]])], color=color, marker="o")
            axes[2].scatter([xs], [np.median([r["pd025_data_cos"] for r in f["rows"]])], color=color, marker="^")
            axes[2].scatter([xs], [np.median([r["pd05_algebra_data_cos"] for r in f["rows"]])], color=color,
                            marker="s", facecolors="none")
    axes[0].set(title="ln1 gain spread: max ÷ median (solid worst layer, faint median layer)",
                xlabel="fraction of training")
    axes[1].set(title="ln2 gain spread: max ÷ median", xlabel="fraction of training")
    axes[2].set(title="Step with vs without the gains folded in, data-metric cosine (median over q/k/v/up)\n"
                      "o Muon (Newton-Schulz), ▲ PD α¼, □ PD α½ algebra check", xlabel="fraction of training")
    axes[2].axhline(0.99, color="0.5", lw=0.8, ls=":")
    axes[2].axhline(0.95, color="0.5", lw=0.8, ls=":")
    for ax in axes[:2]:
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figures / "gains.png", dpi=120)
    plt.close(fig)

    for run in ("muon_b262144_lr0.01_s20261001", "pd_b262144_lr0.01_s20261001"):
        gains = data[run]["gains"]
        if not gains:
            continue
        folds = [f for f in gains["fold"] if 120 <= f["step"] <= 280]
        rows = [r for f in folds for r in f["rows"]]
        if not rows:
            continue
        muon_cos = float(np.median([r["muon_ns_data_cos"] for r in rows]))
        algebra = float(np.min([r["pd05_algebra_data_cos"] for r in rows]))
        verdict["c"][run] = dict(
            muon_ns_data_cos_median=muon_cos, muon_ns_cos_median=float(np.median([r["muon_ns_cos"] for r in rows])),
            muon_svd_data_cos_median=float(np.median([r["muon_svd_data_cos"] for r in rows])),
            pd025_data_cos_median=float(np.median([r["pd025_data_cos"] for r in rows])),
            algebra_min=algebra, algebra_ok=algebra >= 0.999,
            gain_max_over_median=float(np.max([r["gain_max_over_median"] for r in rows])),
            outcome=("void" if algebra < 0.999 else "deprioritize" if muon_cos >= 0.99
                     else "proceed" if muon_cos <= 0.95 else "weak"))

    (base / "verdict.json").write_text(json.dumps(verdict, indent=1) + "\n")
    print(json.dumps(verdict, indent=1)[:12000])


if __name__ == "__main__":
    main()

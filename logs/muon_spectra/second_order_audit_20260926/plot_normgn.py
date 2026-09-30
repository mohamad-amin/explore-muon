"""Figures for normgn_probe.py: is the normalized GN step's loss rise estimation noise or step size?

Per state (rows):
  (1) held-out loss change along each step at c times the trainer's step (Muon's step, the run's actual step, the
      momentum at Muon's norms, and GN's normalized step for each curvature set at Lanczos 64 and its best damping)
  (2) held-out curvature over the set's own curvature along the same step, against the curvature set, per damping
      (normalized steps s solid, raw Newton steps x dashed): above 1 means the set underestimates it
  (3) every step's held-out first-order term against its curvature at the trainer's scale; the dotted line is where c = 1
      is neutral (c* = 1/2, the regime the optimizers settle in), steps below it gain at c = 1

usage: plot_normgn.py OUT_PNG PROBE_JSON [...]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SET_COLORS = {"batch64": "#d62728", "fresh64": "#ff7f0e", "fresh256": "#2ca02c", "fresh1024": "#1f77b4"}
DAMPING_STYLES = {"0.1": "o", "0.01": "s", "0.001": "^"}


def mean(pair):
    return 0.5 * (pair[0] + pair[1])


def main():
    out = Path(sys.argv[1])
    probes = [json.loads(Path(p).read_text()) for p in sys.argv[2:]]
    fig, axes = plt.subplots(len(probes), 3, figsize=(17, 5.2 * len(probes)), squeeze=False)
    scales = np.array([0.25, 0.5, 1.0, 2.0])
    for row, r in zip(axes, probes):
        base = r["base_loss"]
        name = Path(r["arm"]).name
        ax = row[0]
        for label, style in (("actual", dict(color="k", lw=2.5)), ("muon", dict(color="gray", ls="--", lw=1.5)),
                             ("momentum_at_muon_norms", dict(color="purple", ls=":", lw=1.5))):
            if label in r["references"]:
                ax.plot(scales, np.array(r["references"][label]["losses"]) - base, marker="o", label=label, **style)
        for set_label, entry in r["sets"].items():
            candidates = {k: v for k, v in entry["directions"].items() if k.startswith("k64")}
            best = min(candidates, key=lambda k: candidates[k]["s"]["losses"][2])
            ax.plot(scales, np.array(candidates[best]["s"]["losses"]) - base, marker="s", color=SET_COLORS.get(set_label),
                    label=f"GN normalized, {set_label}, {best}")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xscale("log", base=2)
        ax.set_xticks(scales, [str(s) for s in scales])
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("c (multiple of the trainer's step)")
        ax.set_ylabel("held-out loss change (hidden matrices moved)")
        ax.set_title(f"{name} @ {r['step']}: loss along each step")
        ax.legend(fontsize=7)
        ax = row[1]
        sets = [s for s in SET_COLORS if s in r["sets"]]
        for k, ls in (("k16", ":"), ("k64", "-")):
            for damping, marker in DAMPING_STYLES.items():
                key = f"{k}_d{damping}"
                ratios_s = [mean(r["sets"][s]["directions"][key]["s"]["held_out"]["q"]) /
                            r["sets"][s]["directions"][key]["s"]["own_q"] for s in sets]
                ratios_x = [mean(r["sets"][s]["directions"][key]["x"]["held_out"]["q"]) /
                            r["sets"][s]["directions"][key]["x"]["own_q"] for s in sets]
                ax.plot(range(len(sets)), ratios_s, ls=ls, marker=marker, color="C0", label=f"s, {key}")
                ax.plot(range(len(sets)), ratios_x, ls=ls, marker=marker, color="C3", alpha=0.6, label=f"x, {key}")
        ax.axhline(1, color="k", lw=0.5)
        ax.set_xticks(range(len(sets)), sets)
        ax.set_yscale("log")
        ax.set_ylabel("held-out q / the set's own q along the step")
        ax.set_title("does the curvature set underestimate its own step's curvature?")
        ax.legend(fontsize=6, ncol=2)
        ax = row[2]
        points = []
        for label in ("actual", "muon", "momentum_at_muon_norms"):
            if label in r["references"]:
                h = r["references"][label]["held_out"]
                points.append((label, -mean(h["first"]), mean(h["q"]), "k", "*"))
        for set_label, entry in r["sets"].items():
            for key, v in entry["directions"].items():
                h = v["s"]["held_out"]
                points.append((f"{set_label} {key}", -mean(h["first"]), mean(h["q"]), SET_COLORS.get(set_label),
                               DAMPING_STYLES[key.split("_d")[1]] if key.startswith("k64") else "x"))
        for label, gain, q, color, marker in points:
            ax.scatter(q, gain, color=color, marker=marker, s=60 if marker == "*" else 30)
            if marker == "*":
                ax.annotate(label, (q, gain), fontsize=7, xytext=(4, 4), textcoords="offset points")
        qs = np.logspace(np.log10(min(p[2] for p in points) / 2), np.log10(max(p[2] for p in points) * 2), 50)
        ax.plot(qs, qs / 2, color="k", ls=":", lw=1, label="neutral at c = 1 (-first = q / 2)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("held-out curvature q along the step (c = 1)")
        ax.set_ylabel("held-out first-order gain -<g, s> (c = 1)")
        ax.set_title("gain vs curvature at the trainer's step (above the line: loss falls at c = 1)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main()

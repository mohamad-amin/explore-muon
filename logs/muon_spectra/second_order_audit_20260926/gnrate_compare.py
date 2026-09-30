"""GN-PD as a rate at 16M: its own trajectory against its Kronecker control (2026-09-28 09:40 CDT decision).

Arms are newton_train.py runs from the PD alpha 1/2 @0.028 16M state at step 9; steps 1-9 are that run's own (shared
prefix). Each step's training loss before the update is on unseen data, so it estimates the population loss; it is
smoothed as in speedup_by_batch.py (centered running mean over ~3% of the run). Speedup of A over B at step t =
t / the first step at which A's smoothed loss reaches B's at t (A faster: > 1). References are the real-trainer 16M
arms (seed 260925).

Panels: smoothed loss; speedups over Muon @0.02 and over the Kronecker control; the GN's top eigenvalue (Lanczos 20
from a fixed random start, before each step); each matrix kind's cosine between the GN-PD step and the Kronecker map.

usage: gnrate_compare.py OUT_JSON OUT_PNG LABEL=RUN_DIR [LABEL=RUN_DIR ...]   (the first run is the Kronecker control)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
START_ARM = "soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"
START = 9
TOTAL = 92          # steps of a 1x 16M run
REFERENCES = {
    "Muon @0.02": "soaudit_batch16m_20260927/M_b16M_lr0.02_mom0.9_s260925_l40s",
    "PD α½ @0.028": START_ARM,
    "TS ½/½ @0.028": "soaudit_batch16m_20260927/TS_a0.5_b0.5gn_b16M_lr0.028_mom0.9_s260925_l40s",
    "S∘PD α½ @0.028": "soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada",
}
KINDS = ("attn.q", "attn.k", "attn.v", "attn.o", "mlp.up", "mlp.down")


def rows_of(directory):
    out = {}
    for f in Path(directory).glob("step*.json"):
        r = json.loads(f.read_text())
        out[r["step"]] = r
    return out


def smooth(steps, loss, total):
    width = max(1, int(round(0.03 * total)) | 1)
    pad = width // 2
    return np.convolve(np.pad(loss, pad, mode="edge"), np.ones(width) / width, mode="valid")


def curve(rows, total):
    steps = np.array(sorted(s for s in rows if "train_nll" in rows[s]))
    loss = np.array([rows[s]["train_nll"] for s in steps])
    return steps, smooth(steps, loss, total)


def speedup(steps_a, loss_a, steps_b, loss_b, first=START + 1):
    """t / (first step at which A's smoothed loss reaches B's at t), for t in B's steps >= first; NaN if not reached."""
    out = []
    for t, target in zip(steps_b, loss_b):
        if t < first:
            continue
        hit = np.nonzero(loss_a <= target)[0]
        if len(hit) == 0:
            out.append((int(t), float("nan"), float(target)))
            continue
        j = hit[0]
        if j > 0:   # interpolate between the two steps around the crossing
            s0, s1, l0, l1 = steps_a[j - 1], steps_a[j], loss_a[j - 1], loss_a[j]
            reach = s0 + (l0 - target) / max(l0 - l1, 1e-12) * (s1 - s0)
        else:
            reach = steps_a[j]
        out.append((int(t), float(t / max(reach, 1e-9)), float(target)))
    return out


def main():
    out_json, out_png = Path(sys.argv[1]), Path(sys.argv[2])
    runs = dict(item.split("=", 1) for item in sys.argv[3:])
    prefix = {s: r for s, r in rows_of(ROOT / START_ARM / "scientific" / "steps").items() if s <= START}
    total = TOTAL
    curves, finals, extras = {}, {}, {}
    for label, arm in REFERENCES.items():
        rows = rows_of(ROOT / arm / "scientific" / "steps")
        curves[label] = curve(rows, total)
        vals = [(s, r["validation_nll"]) for s, r in rows.items() if "validation_nll" in r]
        finals[label] = max(vals)[1] if vals else None
    for label, run in runs.items():
        own = rows_of(Path(run) / "steps")
        rows = {**prefix, **own}
        curves[label] = curve(rows, total)
        vals = [(s, r["validation_nll"]) for s, r in own.items() if "validation_nll" in r]
        finals[label] = max(vals)[1] if vals else None
        extras[label] = {
            "last_step": max(own) if own else None,
            "gn_top": {s: own[s]["gn_top"] for s in sorted(own) if "gn_top" in own[s]},
            "validation": {s: own[s]["validation_nll"] for s in sorted(own) if "validation_nll" in own[s]},
            "cos_kron_by_kind": {kind: {s: float(np.mean([v for k, v in own[s]["cos_kron"].items() if kind in k]))
                                        for s in sorted(own) if "cos_kron" in own[s]} for kind in KINDS},
            "seconds_per_step": float(np.median([own[s]["training_seconds"] for s in own])) if own else None,
        }
    control = next(iter(runs))
    result = {"finals": finals, "extras": extras, "speedup_over_muon": {}, "speedup_over_control": {}}
    ms, ml = curves["Muon @0.02"]
    cs, cl = curves[control]
    for label, (s, l) in curves.items():
        if label != "Muon @0.02":
            result["speedup_over_muon"][label] = speedup(s, l, ms, ml)
        if label != control:
            result["speedup_over_control"][label] = speedup(s, l, cs, cl)
    out_json.write_text(json.dumps(result, indent=1) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    ax = axes[0, 0]
    for label, (s, l) in curves.items():
        style = "--" if label in REFERENCES else "-"
        ax.plot(s, l, style, label=f"{label} ({finals[label]:.4f})" if finals[label] else label)
    ax.axvline(START, color="grey", lw=0.8)
    ax.set_ylim(None, 7.5)
    ax.set_xlabel("step (16M tokens each)")
    ax.set_ylabel("training loss before the step (unseen data, smoothed)")
    ax.set_title("16M: GN-PD on its own trajectory from PD's step-9 state")
    ax.legend(fontsize=7)
    ax = axes[0, 1]
    for label, rows in result["speedup_over_muon"].items():
        pts = [(t, v) for t, v, _ in rows if np.isfinite(v)]
        if pts:
            ax.plot(*zip(*pts), "--" if label in REFERENCES else "-", label=f"{label} / Muon")
    for label, rows in result["speedup_over_control"].items():
        if label in runs:
            pts = [(t, v) for t, v, _ in rows if np.isfinite(v)]
            if pts:
                ax.plot(*zip(*pts), "k-", lw=2, label=f"{label} / {control}")
    ax.axhline(1, color="grey", lw=0.8)
    ax.set_xlabel("step t of the slower run")
    ax.set_ylabel("step-equivalent speedup")
    ax.set_title("speedup at matched loss")
    ax.legend(fontsize=7)
    ax = axes[1, 0]
    for label in runs:
        top = extras[label]["gn_top"]
        if top:
            ax.plot(list(map(int, top)), list(top.values()), "o-", label=label)
    ax.set_yscale("log")
    ax.set_xlabel("step")
    ax.set_ylabel("top GN eigenvalue (curvature sequences)")
    ax.set_title("sharpness in the GN")
    ax.legend(fontsize=7)
    ax = axes[1, 1]
    for label in runs:
        for kind, series in extras[label]["cos_kron_by_kind"].items():
            if series:
                ax.plot(list(map(int, series)), list(series.values()), label=f"{label}: {kind}")
    ax.set_xlabel("step")
    ax.set_ylabel("cos(GN-PD step, Kronecker map)")
    ax.set_title("where the exact geometry departs from Kronecker")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    print(json.dumps({"finals": finals, "seconds_per_step": {k: v["seconds_per_step"] for k, v in extras.items()}}))


if __name__ == "__main__":
    main()

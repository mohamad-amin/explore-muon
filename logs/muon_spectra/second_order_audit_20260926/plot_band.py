"""Band-restricted coordination (newton_train.py --staged-pd 0.5 --stage-band top|bulk) against the full layer-staged
correction and the same-harness control, 16M @9 -> 92 (2026-09-29 08:4x CDT): step lead over the control, top GN
eigenvalue, and how far the coordinated direction turns from plain PD's (mean per-matrix cosine).

usage: plot_band.py OUT_PNG [NAME=DIR ...]   (DIR relative to this folder; defaults: the three 32-sequence arms)
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("psl", HERE / "plot_step_leads.py")
psl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psl)

DEFAULT = {"full correction (both bands)": ("staged_branch/staged_9_sharp", "#c0392b"),
           "above-mean input band only": ("band/band_top", "#2e86c1"),
           "bulk input band only": ("band/band_bulk", "#27ae60")}
CONTROL = "staged_branch/control_9_sharp"


def rows(path):
    out = {}
    for f in sorted((HERE / path / "steps").glob("step*.json")):
        r = json.loads(f.read_text())
        out[r["step"]] = r
    return out


def main():
    out = sys.argv[1]
    arms = dict(DEFAULT)
    colors = ["#8e44ad", "#e0843a", "#7f8c8d", "#16a085"]
    for k, spec_ in enumerate(sys.argv[2:]):
        name, path = spec_.split("=", 1)
        arms[name] = (path, colors[k % len(colors)])
    control = rows(CONTROL)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for name, (path, color) in arms.items():
        d = psl.leads(HERE / path / "steps", HERE / CONTROL / "steps", 3)
        r = rows(path)
        final = r.get(92, {}).get("validation_nll")
        label = f"{name} (final {final - control[92]['validation_nll']:+.3f})" if final is not None else name
        ax[0].plot(d[:, 0], d[:, 1], color=color, lw=1.7, label=label)
        ks = [t for t in sorted(r) if "gn_top" in r[t]]
        ax[1].plot(ks, [r[t]["gn_top"] for t in ks], "o-", color=color, label=name)
        cs = [t for t in sorted(r) if "staged_cos_plain_mean" in r[t]]
        ax[2].plot(cs, [r[t]["staged_cos_plain_mean"] for t in cs], color=color, lw=1.3, label=name)
    ks = [t for t in sorted(control) if "gn_top" in control[t]]
    ax[1].plot(ks, [control[t]["gn_top"] for t in ks], "o-", color="k", label="control (plain PD)")
    ax[0].axhline(0, color="grey", lw=0.8)
    ax[0].set(xlabel="step (16M tokens)", ylabel="lead over the control (steps)",
              title="Coordination restricted to one input band keeps its lead")
    ax[1].set(xlabel="step", ylabel="top GN eigenvalue", yscale="log", title="Sharpness: a new plateau, or a runaway")
    ax[2].set(xlabel="step", ylabel="mean cosine with plain PD's direction", title="How far coordination turns the step")
    for a in ax:
        a.legend(fontsize=7)
    fig.text(0.01, 0.01, "Legend: final validation minus the control's (4.6811). Step leads are undefined once the control "
             "never reaches the arm's loss before its end.", fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()

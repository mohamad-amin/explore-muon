"""Above-mean band coordination composed with the momentum (2026-09-29 14:3x CDT, the review's decisive test): validation
curves of the harness controls and the band arms at beta 0.9 and 0.8, with the real-trainer references, and the step
lead of each band arm over its own control.

usage: plot_coordination_beta.py OUT_PNG
"""
import importlib.util
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
spec = importlib.util.spec_from_file_location("psl", HERE / "plot_step_leads.py")
psl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(psl)

ARMS = [("β 0.9 control (harness)", "staged_branch/control_9_sharp", "#7f8c8d", "--"),
        ("β 0.9 + above-mean band", "band/band_top", "#2e86c1", "--"),
        ("β 0.8 control (harness)", "band/control_b08", "#7f8c8d", "-"),
        ("β 0.8 + above-mean band", "band/band_top_b08", "#c0392b", "-"),
        ("β 0.8 + 8 largest directions", "band/band_top8_b08", "#e0843a", "-")]
REFS = [("real PD β 0.8", "soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s"),
        ("real S∘PD β 0.7 (previous best)", "soaudit_mom16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.7_s260925_ada")]
PAIRS = [("β 0.9: band vs its control", "band/band_top", "staged_branch/control_9_sharp", "#2e86c1"),
         ("β 0.8: band vs its control", "band/band_top_b08", "band/control_b08", "#c0392b"),
         ("β 0.8: 8 directions vs control", "band/band_top8_b08", "band/control_b08", "#e0843a")]


def rows(folder):
    out = {}
    for f in sorted(folder.glob("step*.json")):
        r = json.loads(f.read_text())
        out[r["step"]] = r
    return out


def main():
    out = sys.argv[1]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for label, path, color, style in ARMS:
        r = rows(HERE / path / "steps")
        vs = [t for t in sorted(r) if "validation_nll" in r[t]]
        if not vs:
            continue
        final = r[vs[-1]]["validation_nll"]
        ax[0].plot(vs, [r[t]["validation_nll"] for t in vs], style, marker="o", ms=3, color=color,
                   label=f"{label}: {final:.4f}" + (f" (at {vs[-1]})" if vs[-1] != 92 else ""))
    for label, path in REFS:
        r = rows(ROOT / path / "scientific" / "steps")
        if 92 in r and "validation_nll" in r[92]:
            ax[0].axhline(r[92]["validation_nll"], color="k", lw=0.8, ls=":")
            ax[0].annotate(f"{label} {r[92]['validation_nll']:.4f}", (60, r[92]["validation_nll"]), fontsize=7, va="bottom")
    ax[0].set(xlabel="step (16M tokens)", ylabel="validation NLL", ylim=(4.3, 5.6), title="Cross-layer coordination composes with a fresh momentum")
    ax[0].legend(fontsize=7)
    for label, arm, control, color in PAIRS:
        a, c = HERE / arm / "steps", HERE / control / "steps"
        if not a.exists() or not c.exists():
            continue
        d = psl.leads(a, c, 3)
        ax[1].plot(d[:, 0], d[:, 1], color=color, lw=1.8, label=label)
    ax[1].axhline(0, color="grey", lw=0.8)
    ax[1].set(xlabel="step", ylabel="lead over its own control (steps)", title="Step lead of the coordinated arm")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out)


if __name__ == "__main__":
    main()

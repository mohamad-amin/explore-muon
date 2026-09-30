"""Ledger of the cross-layer coordination variants (2026-09-29): final validation of each harness arm minus its own
same-harness control, grouped by setting. Arms that are still running are skipped.

usage: plot_coordination_ledger.py OUT_PNG
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent

GROUPS = [
    ("16M PD β 0.9", "staged_branch/control_9_sharp", [
        ("full correction", "staged_branch/staged_9_sharp"), ("full, 256 curvature seq.", "band/staged_big_256"),
        ("full, half strength", "band/staged_half"), ("bulk band", "band/band_bulk"), ("input mean only", "band/band_mean"),
        ("8 largest directions", "band/band_top8"), ("above-mean band", "band/band_top")]),
    ("16M PD β 0.9, 2nd init.", "band/control_s2", [("above-mean band", "band/band_top_s2")]),
    ("16M PD β 0.8", "band/control_b08", [("8 largest directions", "band/band_top8_b08"), ("above-mean band", "band/band_top_b08")]),
    ("16M S∘PD β 0.8", "band/spd_control_b08", [
        ("oracle per-matrix reversal", "band/spd_reversal_b08"), ("Jacobi (plain-step corrections)", "band/spd_jacobi_b08"),
        ("reversed sweep (last block first)", "band/spd_reverse_b08"), ("8 curvature seq. per rank", "band/spd_top_c8_b08"),
        ("layer sweep, all directions", "band/spd_full_b08"), ("sublayer sweep, all directions", "band/spd_sublayer_full_b08"),
        ("above-mean band (Gauss-Seidel)", "band/spd_top_b08"),
        ("sublayer sweep (q/k/v, o, up, down)", "band/spd_sublayer_b08"), ("per-matrix sweep", "band/spd_matrix_b08")]),
    ("16M PD β 0.9, 2× horizon", "band/control_2x", [("above-mean band", "band/band_top_2x")]),
    ("16M S∘PD β 0.8, 2× horizon", "band/spd_control_2x_b08", [("above-mean band", "band/spd_top_2x_b08"),
                                                              ("sublayer sweep", "band/spd_sublayer_2x_b08")]),
    ("4M PD β 0.9", "band/control_4m", [("above-mean band", "band/band_top_4m"), ("8 largest directions", "band/band_top8_4m"),
                                        ("above-mean band, 16M dose", "band/band_top_4m_half"),
                                        ("sublayer sweep, 16M dose", "band/band_sublayer_4m_half")]),
    ("1M PD β 0.95", "band/control_1m", [("above-mean band, 16M dose", "band/band_top_1m_half")]),
    ("16M S∘PD β 0.8 from step 0", "band/spd_control_from0", [("sublayer sweep, 8 seq.", "band/spd_sublayer_c8_from0"),
                                                               ("sublayer sweep, 8 seq. + EMA of C", "band/spd_sublayer_c8ema_from0"),
                                                               ("sublayer sweep", "band/spd_sublayer_from0")]),
    ("16M S∘PD β 0.8 from step 0, matched EMA-root control", "band/spd_control_c8ema_from0",
     [("sublayer sweep, 8 seq. + EMA of C", "band/spd_sublayer_c8ema_from0"),
      ("symmetric sweep (forward + backward)", "band/spd_sym_from0")]),
    ("16M S∘PD β 0.8 from step 0, LR ×1.4, matched control", "band/spd_control_c8ema_lr14_from0",
     [("sublayer sweep, 8 seq. + EMA of C", "band/spd_cheap_lr14_from0")]),
    ("32M S∘PD β 0.8 from step 0, matched control", "band/spd_control_c8ema_32m_from0",
     [("sublayer sweep, 8 seq. + EMA of C", "band/spd_cheap_32m_from0")]),
    ("16M S∘PD β 0.8 from step 0, 2×", "band/spd_control_2x_from0", [("sublayer sweep", "band/spd_sublayer_2x_from0"),
                                                                      ("sublayer sweep, 8 seq. + EMA of C", "band/spd_sublayer_c8ema_2x_from0")]),
    ("16M S∘PD β 0.8 from step 0, 2×, matched EMA-root control", "band/spd_control_c8ema_2x_from0",
     [("sublayer sweep, 8 seq. + EMA of C", "band/spd_sublayer_c8ema_2x_from0")]),
]


def final(path):
    rows = {}
    for f in (HERE / path / "steps").glob("step*.json"):
        r = json.loads(f.read_text())
        rows[r["step"]] = r
    done = (HERE / path / "done.json").exists()
    if not done or not rows:
        return None, None
    last = max(rows)
    return last, rows[last].get("validation_nll")


def main():
    out = sys.argv[1]
    labels, values, colors = [], [], []
    palette = ["#2e86c1", "#8e44ad", "#c0392b", "#16a085", "#7f8c8d", "#b8860b", "#27ae60", "#e0843a"]
    for g_i, (group, control, arms) in enumerate(GROUPS):
        c_last, c_val = final(control)
        if c_val is None:
            continue
        for label, path in arms:
            last, val = final(path)
            if val is None or last != c_last:
                continue
            labels.append(f"{group}: {label}")
            values.append(val - c_val)
            colors.append(palette[g_i % len(palette)])
    fig, ax = plt.subplots(figsize=(10, 0.36 * len(labels) + 1.5))
    y = range(len(labels))
    ax.barh(list(y), values, color=colors)
    for yi, v in zip(y, values):
        ax.text(v + (0.004 if v >= 0 else -0.004), yi, f"{v:+.3f}", va="center", ha="left" if v >= 0 else "right", fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_yticks(list(y), labels, fontsize=7)
    ax.invert_yaxis()
    ax.set(xlabel="final validation − own harness control (negative: better)",
           title="Cross-layer coordination variants: what pays and what does not")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print("wrote", out, len(labels), "arms")


if __name__ == "__main__":
    main()

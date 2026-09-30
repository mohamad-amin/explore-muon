"""How far is large-batch training from perfect batch scaling, and how much of that gap does cross-layer coordination
close? (2026-09-30, second-order audit; the GN paper's large-batch claim in tokens.)

Every batch is new data (one pass), so the training loss of a step before its update estimates the population loss. It
is smoothed with a centered running mean over ~3% of each run (at least one step on each side). For a target loss L*,
T(L*) is the number of tokens an arm has consumed when its smoothed loss first reaches L*. Below the critical batch,
T(L*) does not depend on the batch size; above it, a larger batch needs more tokens. The ratio T_16M / T_4M at the
same method measures the 16M run's token inefficiency; coordination's share of the gap is how much it lowers that ratio.

usage: batch_efficiency.py OUT_JSON OUT_PNG
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

# label: (path, kind, batch in M tokens, color, style); kind "real" = trainer run, "harness" = newton_train.py arm
ARMS = {
    "Muon 1M": ("soaudit_traj_20260926/M_lr0.007_s260925_l40s", "real", 1, "#7f8c8d", ":"),
    "Muon 4M": ("soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s", "real", 4, "#7f8c8d", "--"),
    "Muon 16M": ("soaudit_mom16m_20260928/M_b16M_lr0.02_mom0.8_s260925_l40s", "real", 16, "#7f8c8d", "-"),
    "S∘PD 1M": ("soaudit_traj_20260926/SPD_a0.25_lr0.01_s260925_ada", "real", 1, "#2e86c1", ":"),
    "S∘PD 4M": ("soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s", "real", 4, "#2e86c1", "--"),
    "S∘PD 16M": ("soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada", "real", 16, "#2e86c1", "-"),
    "S∘PD 16M harness control (EMA root)": ("second_order_audit_20260926/band/spd_control_c8ema_from0", "harness", 16,
                                            "#85c1e9", "-"),
    "coordinated S∘PD 16M (cheap)": ("second_order_audit_20260926/band/spd_sublayer_c8ema_from0", "harness", 16,
                                     "#c0392b", "-"),
    "S∘PD 32M harness control (EMA root)": ("second_order_audit_20260926/band/spd_control_c8ema_32m_from0", "harness", 32,
                                            "#85c1e9", "-."),
    "coordinated S∘PD 32M (cheap)": ("second_order_audit_20260926/band/spd_cheap_32m_from0", "harness", 32, "#c0392b", "-."),
    # aux LR scaled with batch (sqrt rule from 1M; MUON_CASE 2026-09-30 15:18 / 18:3x CDT)
    "S∘PD 4M harness control, aux ×2": ("second_order_audit_20260926/band/spd_control_4m_aux2_from0", "harness", 4,
                                         "#1a5276", "--"),
    "S∘PD 16M harness control, aux ×4": ("second_order_audit_20260926/band/spd_control_c8ema_auxlr4_from0", "harness", 16,
                                          "#1a5276", "-"),
    "coordinated S∘PD 16M (cheap), aux ×4": ("second_order_audit_20260926/band/spd_cheap_auxlr4_from0", "harness", 16,
                                              "#922b21", "-"),
}
TARGETS = [5.8, 5.6, 5.4, 5.2, 5.0, 4.9, 4.8, 4.7, 4.6, 4.5, 4.4]


def curve(path, kind, batch):
    folder = ROOT / path / ("scientific/steps" if kind == "real" else "steps")
    if not folder.exists():
        return None
    rows = {}
    for f in folder.glob("step*.json"):
        r = json.loads(f.read_text())
        if r.get("train_nll") is not None and r["step"] >= 1:
            rows[r["step"]] = r["train_nll"]
    if len(rows) < 5:
        return None
    steps = np.array(sorted(rows))
    loss = np.array([rows[s] for s in steps])
    half = max(1, int(round(0.015 * len(steps))))
    smooth = np.array([loss[max(0, i - half):i + half + 1].mean() for i in range(len(steps))])
    tokens = (steps - 1) * batch * 2 ** 20      # tokens consumed before the step's update
    total = len(steps)
    return {"steps": steps, "tokens": tokens, "loss": smooth, "total": total}


def first_reach(c, target):
    below = np.nonzero(c["loss"] <= target)[0]
    if len(below) == 0:
        return None
    i = below[0]
    if i == 0:
        return float(c["tokens"][0])
    # linear interpolation between the last point above and the first below
    t0, t1, l0, l1 = c["tokens"][i - 1], c["tokens"][i], c["loss"][i - 1], c["loss"][i]
    return float(t0 + (t1 - t0) * (l0 - target) / (l0 - l1))


def main():
    out_json, out_png = sys.argv[1], sys.argv[2]
    curves = {k: curve(p, kind, b) for k, (p, kind, b, _, _) in ARMS.items()}
    reach = {k: {str(t): first_reach(c, t) for t in TARGETS} for k, c in curves.items() if c is not None}
    ratios = {}
    for method, pairs in {"Muon": [("Muon 16M", "Muon 4M"), ("Muon 16M", "Muon 1M"), ("Muon 4M", "Muon 1M")],
                          "S∘PD": [("S∘PD 16M", "S∘PD 4M"), ("S∘PD 16M", "S∘PD 1M"), ("S∘PD 4M", "S∘PD 1M"),
                                   ("S∘PD 32M harness control (EMA root)", "S∘PD 16M harness control (EMA root)"),
                                   ("coordinated S∘PD 32M (cheap)", "coordinated S∘PD 16M (cheap)"),
                                   ("S∘PD 16M harness control, aux ×4", "S∘PD 4M harness control, aux ×2"),
                                   ("coordinated S∘PD 16M (cheap), aux ×4", "S∘PD 4M harness control, aux ×2"),
                                   ("S∘PD 4M harness control, aux ×2", "S∘PD 1M"),
                                   ("S∘PD 16M harness control (EMA root)", "S∘PD 4M"),
                                   ("coordinated S∘PD 16M (cheap)", "S∘PD 4M"),
                                   ("S∘PD 32M harness control (EMA root)", "S∘PD 4M"),
                                   ("coordinated S∘PD 32M (cheap)", "S∘PD 4M")]}.items():
        for big, small in pairs:
            if big not in reach or small not in reach:
                continue
            ratios[f"{big} / {small}"] = {str(t): (reach[big][str(t)] / reach[small][str(t)]
                                                   if reach[big][str(t)] and reach[small][str(t)] else None)
                                          for t in TARGETS}
    Path(out_json).write_text(json.dumps({"reach_tokens": reach, "token_ratio": ratios}, indent=1))
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    for k, c in curves.items():
        if c is None:
            continue
        _, _, _, color, style = ARMS[k]
        ax[0].plot(c["tokens"] / 1e9, c["loss"], style, color=color, lw=1.6 if "coordinated" in k else 1.2, label=k)
    ax[0].set(xlabel="tokens consumed (B)", ylabel="training loss before the step (one pass; smoothed)",
              ylim=(3.6, 6.0), title="Loss against tokens: perfect batch scaling would overlay the curves")
    ax[0].legend(fontsize=7)
    shown = [("Muon 16M / Muon 4M", "#7f8c8d", "-"), ("S∘PD 16M / S∘PD 4M", "#2e86c1", "-"),
             ("S∘PD 16M harness control (EMA root) / S∘PD 4M", "#85c1e9", "-"),
             ("coordinated S∘PD 16M (cheap) / S∘PD 4M", "#c0392b", "-"),
             ("S∘PD 32M harness control (EMA root) / S∘PD 4M", "#85c1e9", "-."),
             ("coordinated S∘PD 32M (cheap) / S∘PD 4M", "#c0392b", "-."),
             ("Muon 4M / Muon 1M", "#7f8c8d", ":"), ("S∘PD 4M / S∘PD 1M", "#2e86c1", ":"),
             ("S∘PD 16M harness control, aux ×4 / S∘PD 4M harness control, aux ×2", "#1a5276", "-"),
             ("coordinated S∘PD 16M (cheap), aux ×4 / S∘PD 4M harness control, aux ×2", "#922b21", "-")]
    for label, color, style in shown:
        if label not in ratios:
            continue
        xs = [t for t in TARGETS if ratios[label][str(t)] is not None]
        ax[1].plot(xs, [ratios[label][str(t)] for t in xs], style, marker="o", ms=3, color=color, label=label)
    ax[1].axhline(1, color="k", lw=0.8)
    ax[1].invert_xaxis()
    ax[1].set(xlabel="target loss", ylabel="tokens to reach it, relative to the smaller batch",
              title="Token cost of the larger batch (1 = perfect scaling)")
    ax[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    for label, r in ratios.items():
        print(f"{label:55s}", " ".join(f"{t}:{v:.2f}" if v else f"{t}:  - " for t, v in r.items()))


if __name__ == "__main__":
    main()

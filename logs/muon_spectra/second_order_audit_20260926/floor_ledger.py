"""Floor ledger: which mid-run leads are progress along the valley (MUON_CASE 2026-09-29 07:19 CDT).

For each arm, a 16-step linear anneal from a kept mid-run state: the held-out loss before it (raw) and after it (the floor
under the edge oscillation), and the released excess raw - floor. Real-trainer arms come from anneal_branch.py
(anneal.json: EVAL sequences); harness arms from newton_train.py --anneal-from S --anneal-steps K (validation_nll at S
and at the end, the run's validation tokens). For each pair (A vs B at the same step): the raw gap, the floor gap and
the kept fraction F = floor gap / raw gap.

usage: floor_ledger.py OUT_PREFIX
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
LEDGER = HERE / "floor_ledger"
VALLEY = HERE / "valley"

# arm label -> (source kind, path relative to LEDGER or VALLEY)
ARMS = {
    "Muon 16M @46": ("anneal", LEDGER / "M16M_46"),
    "PD 16M @46": ("anneal", LEDGER / "PD16M_46"),
    "S∘PD 16M @46": ("anneal", LEDGER / "SPD16M_46"),
    "PD β0.8 16M @46": ("anneal", LEDGER / "PDb08_16M_46"),
    "PD 16M 2× @37": ("anneal", LEDGER / "PD2x_37"),
    "PD warmup 16M 2× @37": ("anneal", LEDGER / "PDwarm2x_37"),
    "PD 16M 2× @92": ("anneal", LEDGER / "PD2x_92"),
    "PD warmup 16M 2× @92": ("anneal", LEDGER / "PDwarm2x_92"),
    "PD 4M @37": ("anneal", VALLEY / "PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada_a37"),
    "PD 4M @183": ("anneal", VALLEY / "PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada_a183"),
    "PD 4M @330": ("anneal", VALLEY / "PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada_a330"),
    "PD 4M @183 (repeat)": ("anneal", LEDGER / "PD4M_183_repeat"),
    "PD warmup 4M @37": ("anneal", LEDGER / "PDwarm4M_37"),
    "PD warmup 4M @183": ("anneal", LEDGER / "PDwarm4M_183"),
    "PD warmup 4M @330": ("anneal", LEDGER / "PDwarm4M_330"),
    "control (harness) 16M @46": ("harness", LEDGER / "control_46"),
    "staged (harness) 16M @46": ("harness", LEDGER / "staged_46"),
}
# pair label -> (A, B): A is the arm whose lead is asked about
PAIRS = {
    "PD vs Muon (16M, 46)": ("PD 16M @46", "Muon 16M @46"),
    "S∘PD vs PD (16M, 46)": ("S∘PD 16M @46", "PD 16M @46"),
    "β0.8 vs β0.9 (PD 16M, 46)": ("PD β0.8 16M @46", "PD 16M @46"),
    "warmup vs β0.9 (16M 2×, 37)": ("PD warmup 16M 2× @37", "PD 16M 2× @37"),
    "warmup vs β0.9 (16M 2×, 92)": ("PD warmup 16M 2× @92", "PD 16M 2× @92"),
    "warmup vs β0.9 (4M, 37)": ("PD warmup 4M @37", "PD 4M @37"),
    "warmup vs β0.9 (4M, 183)": ("PD warmup 4M @183", "PD 4M @183"),
    "warmup vs β0.9 (4M, 330)": ("PD warmup 4M @330", "PD 4M @330"),
    "staged vs control (16M, 46)": ("staged (harness) 16M @46", "control (harness) 16M @46"),
}


def load(kind, path):
    if kind == "anneal":
        f = path / "anneal.json"
        if not f.exists():
            return None
        r = json.loads(f.read_text())
        return {"raw": r["loss"][0], "floor": r["loss"][-1], "curve": r["loss"], "start": r["start_step"]}
    rows = {}
    for f in sorted((path / "steps").glob("step*.json")):
        r = json.loads(f.read_text())
        rows[r["step"]] = r
    run = json.loads((path / "run.json").read_text()) if (path / "run.json").exists() else None
    if run is None:
        return None
    start = int(run["args"]["anneal_from"])
    steps = int(run["args"]["anneal_steps"])
    end = start + steps
    if start not in rows or end not in rows or "validation_nll" not in rows[end]:
        return None
    curve = [rows[s]["validation_nll"] for s in range(start, end + 1) if s in rows and "validation_nll" in rows[s]]
    return {"raw": rows[start]["validation_nll"], "floor": rows[end]["validation_nll"], "curve": curve, "start": start}


def main():
    prefix = sys.argv[1]
    data = {name: load(*spec) for name, spec in ARMS.items()}
    table = {"arms": {}, "pairs": {}}
    for name, d in data.items():
        if d:
            table["arms"][name] = {"raw": d["raw"], "floor": d["floor"], "excess": d["raw"] - d["floor"]}
            print(f"{name:32s} raw {d['raw']:.4f}  floor {d['floor']:.4f}  excess {d['raw'] - d['floor']:.4f}")
    for label, (a, b) in PAIRS.items():
        if data.get(a) and data.get(b):
            raw = data[a]["raw"] - data[b]["raw"]
            floor = data[a]["floor"] - data[b]["floor"]
            kept = floor / raw if abs(raw) > 1e-9 else float("nan")
            table["pairs"][label] = {"raw_gap": raw, "floor_gap": floor, "kept_fraction": kept,
                                     "excess_difference": raw - floor}
            print(f"{label:32s} raw gap {raw:+.4f}  floor gap {floor:+.4f}  F {kept:+.2f}")
    Path(prefix + ".json").write_text(json.dumps(table, indent=1) + "\n")
    pairs = list(table["pairs"])
    if not pairs:
        return
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})
    xs = range(len(pairs))
    ax[0].bar([x - 0.2 for x in xs], [table["pairs"][p]["raw_gap"] for p in pairs], width=0.4, color="#9aa7b8",
              label="raw gap (loss the run is at)")
    ax[0].bar([x + 0.2 for x in xs], [table["pairs"][p]["floor_gap"] for p in pairs], width=0.4, color="#2e86c1",
              label="floor gap (after a 16-step anneal)")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(list(xs), pairs, rotation=35, ha="right", fontsize=8)
    ax[0].set(ylabel="A − B (negative: A ahead)", title="Which leads survive an anneal?")
    ax[0].legend(fontsize=8)
    for label in pairs:
        a, b = PAIRS[label]
        for name, style in ((a, "-"), (b, "--")):
            curve = data[name]["curve"]
            ax[1].plot(range(len(curve)), [v - data[name]["curve"][0] for v in curve], style, lw=1.2,
                       label=name if style == "-" else None)
    ax[1].set(xlabel="anneal step", ylabel="held-out loss − value before the anneal",
              title="Released excess along each anneal (solid: A, dashed: B)")
    ax[1].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(prefix + ".png", dpi=130)
    print("wrote", prefix + ".png")


if __name__ == "__main__":
    main()

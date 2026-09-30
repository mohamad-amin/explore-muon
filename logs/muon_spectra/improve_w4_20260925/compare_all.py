"""Improvement-search summary across waves 1-5 (reads step rows only; safe to rerun anytime).

Pairs each arm with Muon at the same hardware and seed: at the same LR when that run exists,
and always with Muon@0.01 (the best of 0.01/0.014/0.02). Run from the project root:
.venv/bin/python logs/muon_spectra/improve_w4_20260925/compare_all.py
"""
import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent
MARKS = (100, 200, 300, 500, 1000, 1250, 1469)
FIXED = {  # label: (scientific dir, hardware)
    "M_s260924_ref": (LOGS / "depth8_w512_nobias_rms_qk_20260925/muon/scientific", "L40S"),
    "S2_baseline_s260925": (LOGS / "improve_w1_20260925/S2_baseline_s260925/scientific", "L40S"),
    "C_track1_drop": (LOGS / "improve_w1_20260925/C_track1_drop/scientific", "RTX6000Ada"),
}


def method(config):
    if config.get("soap_precondition") and not config.get("data_norm_alpha", 0.0) > 0:
        suffix = {"both": "", "right": "_right", "left": "_left", "none": "_none",
                  "right_act": "_rightact"}[config.get("soap_basis", "both")]
        layers = {"all": "", "down": "_downonly", "not_down": "_notdown"}[config.get("soap_layers", "all")]
        return ("S" + suffix + layers + ("_col" if config.get("soap_norm", "entry") == "column" else "")
                + ("_gradsq" if config.get("soap_second_moment", "update") == "gradient" else ""))
    if config.get("data_norm_alpha", 0.0) > 0:
        if config.get("soap_precondition"):
            side = {"both": "", "left": "_left", "right": "_right", "none": "_none"}[config.get("soap_basis", "both")]
            return f"S{side}∘PD{config['data_norm_alpha']:g}"
        if config.get("data_norm_mode", "sandwich") == "magnitude_matched":
            return f"MM{config['data_norm_alpha']:g}"
        name = ("PD" if config.get("data_norm_post", True) else "PDin") + ("k" if config.get("data_norm_alpha_by_kind") else "")
        if not config.get("data_norm_pre", True):
            name = "PDout"
        name += "c" if config.get("data_norm_center") else ""
        name += "rows" if config.get("data_norm_rows") else ""
        if config.get("data_norm_damping", 1e-3) != 1e-3:
            name = "DDN" + f"d{config['data_norm_damping']:g}_"
        return name + f"{config['data_norm_alpha']:g}"
    if config.get("ns_polynomial") == "svd":
        return "X"
    if config.get("mean_whitening"):
        beta = config.get("mean_whitening_beta", -1.0)
        return "E*" if beta < 0 else f"E{beta:g}"
    if config.get("deflation"):
        return "C" if config.get("deflation_head_weight", 1.0) == 0 else "B"
    if config.get("muon_nesterov"):
        return "N"
    return "M_instr" if config.get("model", {}).get("track_input_stats") else "M"


def hardware(name):
    return "A6000" if "a6000" in name else "RTX6000Ada" if name.endswith("_g20") else "L40S"


def runs():
    out = {}
    for label, (run, hw) in FIXED.items():
        config = json.loads((run.parent / "config.json").read_text()) if (run.parent / "config.json").exists() else \
            json.loads((run / "metadata.json").read_text()).get("config", {})
        out[label] = dict(run=run, hw=hw, seed=config.get("seed", 260924), lr=config.get("learning_rate", 0.01),
                          method="C" if label == "C_track1_drop" else "M", tokens=1539870720)
    for wave in sorted(LOGS.glob("improve_w*_20260925"), key=lambda w: int(w.name.split("_")[1][1:])):
        if wave.name == "improve_w1_20260925":   # its completed arms are listed in FIXED
            continue
        for arm in sorted(p for p in wave.iterdir() if (p / "config.json").exists()):
            config = json.loads((arm / "config.json").read_text())
            width = config["model"].get("n_embd", 512)
            out[arm.name] = dict(run=arm / "scientific", hw=hardware(arm.name), seed=config["seed"],
                                 lr=config["learning_rate"],
                                 method=method(config) + ("" if width == 512 else f" [w{width}]"),
                                 tokens=config.get("total_tokens") or 1539870720)
    return out


def curve(run):
    val = {}
    for path in glob.glob(str(run / "steps" / "step*.json")):
        row = json.loads(Path(path).read_text())
        if "validation_nll" in row:
            val[row["step"]] = row["validation_nll"]
    return val


def main():
    table = runs()
    for info in table.values():
        info["val"] = curve(info["run"])
    table = {k: v for k, v in table.items() if v["val"]}
    for t in table.values():
        t["last"] = -(-t["tokens"] // 1048576)          # the run's final step
    lines = ["| arm | hw | seed | method | LR | final | " + " | ".join(f"@{m}" for m in MARKS) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(MARKS)]
    for name, t in table.items():
        final = f"{t['val'][t['last']]:.5f}" if t["last"] in t["val"] else f"(at {max(t['val'])})"
        if t["last"] != 1469:
            final += f" [{t['last']} steps]"
        lines.append(f"| {name} | {t['hw']} | {t['seed']} | {t['method']} | {t['lr']:g} | {final} | " +
                     " | ".join(f"{t['val'][m]:.4f}" if m in t["val"] else "–" for m in MARKS) + " |")
    lines += ["", "Paired differences (arm − Muon, same hardware and seed; '@0.01' pairs with Muon@0.01):"]
    finals = {}
    for name, t in table.items():
        if t["method"].split(" ")[0] == "M":
            continue
        for lr_key in sorted({t["lr"], 0.01, 0.007}):
            partners = [p for p, u in table.items() if u["method"].split(" ")[0] == "M" and u["hw"] == t["hw"]
                        and u["seed"] == t["seed"] and u["lr"] == lr_key and u["tokens"] == t["tokens"]]
            for partner in partners[:1]:
                marks = MARKS if t["last"] == 1469 else tuple(m for m in (100, 200, 500, 1000, 1469, 2000, 2500) if m < t["last"]) + (t["last"],)
                common = [m for m in marks if m in t["val"] and m in table[partner]["val"]]
                if not common:
                    continue
                tag = "" if lr_key == t["lr"] else f" [vs Muon@{lr_key:g}]"
                lines.append(f"- {name} − {partner}{tag}: " + ", ".join(
                    f"@{m} {t['val'][m] - table[partner]['val'][m]:+.4f}" for m in common))
                if t["last"] == 1469 and 1469 in common and lr_key == 0.01:
                    finals.setdefault((t["method"], t["lr"]), []).append(
                        (name, t["val"][1469] - table[partner]["val"][1469]))
    lines += ["", "Final paired Δ vs Muon@0.01 by method (screening; claims need the MUON_CASE rules):"]
    for (m, lr), items in sorted(finals.items()):
        deltas = np.array([d for _, d in items])
        extra = f", mean {deltas.mean():+.4f}, sd {deltas.std(ddof=1):.4f}" if len(deltas) > 1 else ""
        lines.append(f"- {m}@{lr:g}: n={len(deltas)} " + ", ".join(f"{d:+.4f}" for d in deltas) + extra)
    # Retention of S's gain by dissection arms at the same seed and hardware.
    s_gap = {(t["hw"], t["seed"]): t["val"][1469] for n, t in table.items()
             if t["method"] == "S" and t["lr"] == 0.01 and t["last"] == 1469 and 1469 in t["val"]}
    m_ref = {(t["hw"], t["seed"]): t["val"][1469] for n, t in table.items()
             if t["method"] == "M" and t["lr"] == 0.01 and t["last"] == 1469 and 1469 in t["val"]}
    rows = []
    for name, t in table.items():
        key = (t["hw"], t["seed"])
        if (t["method"] not in ("M", "S") and t["lr"] == 0.01 and t["last"] == 1469 and key in s_gap
                and key in m_ref and 1469 in t["val"]):
            gap = s_gap[key] - m_ref[key]
            rows.append(f"- {name}: retains {(t['val'][1469] - m_ref[key]) / gap:.2f} of S's gap ({gap:+.4f})")
    if rows:
        lines += ["", "Share of S's final gain (same seed and hardware):"] + rows
    # Best-vs-best: each method at a fixed LR against Muon@0.007 (Muon's bracketed best), same seed and hardware.
    best_m = {(t["hw"], t["seed"], t["tokens"]): t["val"][t["last"]] for n, t in table.items()
              if t["method"].split(" ")[0] == "M" and t["lr"] == 0.007 and t["last"] in t["val"]}
    groups = {}
    for name, t in table.items():
        key = (t["hw"], t["seed"], t["tokens"])
        if t["method"].split(" ")[0] != "M" and key in best_m and t["last"] in t["val"]:
            horizon = "" if t["last"] == 1469 or "[w" in t["method"] else f" [{t['tokens'] / 1539870720:g}x horizon]"
            groups.setdefault((t["method"] + horizon, t["lr"]), []).append((name, t["val"][t["last"]] - best_m[key]))
    # Token efficiency: shorter-horizon arms against Muon@0.007 at the full 1x horizon.
    efficiency = []
    for name, t in table.items():
        full = best_m.get((t["hw"], t["seed"], 1539870720))
        if t["method"] != "M" and t["tokens"] < 1539870720 and full is not None and t["last"] in t["val"]:
            efficiency.append(f"- {name}: {t['val'][t['last']]:.5f} at {t['tokens'] / 1539870720:.2f}x tokens "
                              f"vs Muon@0.007 at 1x {full:.5f} ({t['val'][t['last']] - full:+.4f})")
    if groups:
        from math import erf, sqrt
        lines += ["", "Best-vs-best: arm − Muon@0.007 (same seed and hardware):"]
        for (m, lr), items in sorted(groups.items()):
            d = np.array([x for _, x in items])
            extra = ""
            if len(d) > 1:
                tstat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
                extra = f", mean {d.mean():+.4f}, sd {d.std(ddof=1):.4f}, t {tstat:.1f} (df {len(d) - 1})"
            lines.append(f"- {m}@{lr:g}: n={len(d)} " + ", ".join(f"{x:+.4f} [{n}]" for n, x in items) + extra)
    if efficiency:
        lines += ["", "Token efficiency (shorter budgets, each with its own cooldown):"] + efficiency
    text = "\n".join(lines) + "\n"
    (HERE / "summary_all.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

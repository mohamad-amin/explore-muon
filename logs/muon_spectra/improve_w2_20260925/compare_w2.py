"""Wave-1/2 improvement-search summary (reads step rows only; safe to rerun anytime).

Run from the project root:
.venv/bin/python logs/muon_spectra/improve_w2_20260925/compare_w2.py
"""

import glob
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent
RUNS = {
    # label: (run directory, hardware, seed, method, learning rate)
    "M s1 L40S (reference)": (LOGS / "depth8_w512_nobias_rms_qk_20260925/muon/scientific", "L40S", 260924, "M", 0.01),
    "M s2 L40S": (LOGS / "improve_w1_20260925/S2_baseline_s260925/scientific", "L40S", 260925, "M", 0.01),
    "C drop-spike g20": (LOGS / "improve_w1_20260925/C_track1_drop/scientific", "RTX6000Ada", 260924, "C", 0.01),
}
for wave in (HERE, LOGS / "improve_w3_20260925"):
    if not wave.exists():
        continue
    for arm in sorted(p.name for p in wave.iterdir() if (p / "config.json").exists()):
        config = json.loads((wave / arm / "config.json").read_text())
        beta = config.get("mean_whitening_beta", -1.0)
        method = ("SOAP" if config.get("soap_precondition") else
                  "E0" if config["mean_whitening"] and beta == 0 else
                  f"E{beta:g}" if config["mean_whitening"] and beta > 0 else
                  "E*" if config["mean_whitening"] else "B" if config["deflation"] else
                  "N" if config["muon_nesterov"] else "M")
        hardware = "A6000" if "a6000" in arm else "RTX6000Ada" if arm.endswith("_g20") else "L40S"
        RUNS[arm] = (wave / arm / "scientific", hardware, config["seed"], method, config["learning_rate"])
MARKS = (100, 200, 500, 1000, 1250, 1469)


def rows(run):
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(run / "steps" / "step*.json")))]


def main():
    table, curves, extras = {}, {}, {}
    for label, (run, hardware, seed, method, lr) in RUNS.items():
        rs = rows(run)
        val = {r["step"]: r["validation_nll"] for r in rs if "validation_nll" in r}
        if not val:
            continue
        curves[label] = val
        table[label] = dict(hardware=hardware, seed=seed, method=method, lr=lr,
                            final=val.get(1469), last_step=max(val))
        whitening = [r["whitening"]["mean_beta"] for r in rs if "whitening" in r]
        deflation = [r["deflation"] for r in rs if "deflation" in r and "mean_u_autocos" in r["deflation"]]
        if whitening:
            extras[label] = {"mean_beta (steps 1-50, 51-500, 501-end)": [
                round(float(np.mean(whitening[a:b])), 4) for a, b in ((0, 50), (50, 500), (500, None))]}
        if deflation:
            extras[label] = {key: round(float(np.mean([d[key] for d in deflation[50:]])), 4)
                             for key in ("mean_head_energy", "mean_v_autocos", "mean_u_autocos")}
    lines = ["| run | hw | seed | method | LR | final val NLL | " + " | ".join(f"@{m}" for m in MARKS) + " |",
             "|---|---|---|---|---|---|" + "---|" * len(MARKS)]
    for label, t in table.items():
        final = f"{t['final']:.5f}" if t["final"] else f"(at {t['last_step']})"
        lines.append(f"| {label} | {t['hardware']} | {t['seed']} | {t['method']} | {t['lr']} | {final} | " +
                     " | ".join(f"{curves[label][m]:.4f}" if m in curves[label] else "–" for m in MARKS) + " |")
    lines += ["", "Paired differences (same hardware, seed and LR; arm − M):"]
    for label, t in table.items():
        if t["method"] == "M":
            continue
        partners = [l for l, u in table.items() if u["method"] == "M" and
                    (u["hardware"], u["seed"], u["lr"]) == (t["hardware"], t["seed"], t["lr"])]
        for partner in partners:
            common = [m for m in MARKS if m in curves[label] and m in curves[partner]]
            lines.append(f"- {label} − {partner}: " + ", ".join(
                f"@{m} {curves[label][m] - curves[partner][m]:+.4f}" for m in common))
    if extras:
        lines += ["", "Optimizer diagnostics:"] + [f"- {k}: {v}" for k, v in extras.items()]
    text = "\n".join(lines) + "\n"
    (HERE / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

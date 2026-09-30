"""Plain versus deflated 3-step NS, with the 5-step frontier-norm run as reference.

Reads saved step rows and momentum spectra only. Run from the project root:
.venv/bin/python logs/muon_spectra/depth8_nobias_rms_qk_ns3_deflation_20260925/compare_deflation.py
"""

import glob
import json
import re
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = {"5-step reference": HERE.parent / "depth8_w512_nobias_rms_qk_20260925/muon/scientific",
        "3-step plain": HERE / "plain/scientific",
        "3-step deflated": HERE / "deflated/scientific"}
KINDS = ("q", "k", "v", "o", "up", "down")
ROLES = {2: "early", 4: "mid", 6: "late", 8: "final"}


def rows(run):
    return [json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(run / "steps" / "step*.json")))]


def momentum(run):
    out = {}
    for path in sorted(glob.glob(str(run / "spectra_momentum" / "step*.npz"))):
        z = np.load(path)
        out[int(re.search(r"step(\d+)", path).group(1))] = {k: np.sort(z[k].astype(float))[::-1] for k in z.files}
    return out


def window(spectra, name, lo, hi):
    rho, med = [], []
    for step, mats in spectra.items():
        if lo <= step <= hi:
            s = mats[name] / np.sqrt(np.sum(mats[name] ** 2))
            rho.append(s[0] ** 2)
            med.append(s[int(np.ceil(0.5 * len(s))) - 1])
    return np.mean(rho), np.mean(med)


def reading(delta):
    if delta <= -0.02:
        return "clearly confirms at our scale"
    if delta < -0.005:
        return "consistent but small"
    if abs(delta) <= 0.005:
        return "no detectable effect"
    return "against"


def main():
    data = {tag: rows(run) for tag, run in RUNS.items()}
    final = {tag: next(r["validation_nll"] for r in rs if r.get("step") == 1469) for tag, rs in data.items()}
    delta = final["3-step deflated"] - final["3-step plain"]
    lines = ["# NS-budget deflation replication: results", "",
             "One seed per arm; readings as predeclared in research/adamw_spectra/MUON_CASE.md.", "",
             "| run | final val NLL | median step seconds |", "|---|---|---|"]
    for tag, rs in data.items():
        steady = [r["training_seconds"] for r in rs if r.get("step", 0) > 10 and "training_seconds" in r]
        lines.append(f"| {tag} | {final[tag]:.5f} | {np.median(steady):.3f} |")
    lines += ["", f"Δ = deflated − plain = {delta:+.5f} nats/token → **{reading(delta)}**.",
              f"Plain 3-step − 5-step reference = {final['3-step plain'] - final['5-step reference']:+.5f}; "
              f"deflated 3-step − 5-step reference = {final['3-step deflated'] - final['5-step reference']:+.5f}.", ""]
    val = {tag: {r["step"]: r["validation_nll"] for r in rs if "validation_nll" in r} for tag, rs in data.items()}
    common = sorted(set.intersection(*(set(v) for v in val.values())))
    marks = [s for s in common if s in (50, 100, 200, 300, 500, 750, 1000, 1250, 1469)]
    lines += ["Validation NLL along training:", "", "| step | " + " | ".join(RUNS) + " | deflated − plain |",
              "|---|" + "---|" * (len(RUNS) + 1)]
    for s in marks:
        lines.append(f"| {s} | " + " | ".join(f"{val[t][s]:.4f}" for t in RUNS) +
                     f" | {val['3-step deflated'][s] - val['3-step plain'][s]:+.4f} |")
    gate = [r["deflation"] for r in data["3-step deflated"] if "deflation" in r]
    if gate:
        def phase(lo, hi):
            sel = [g for r, g in zip([r for r in data["3-step deflated"] if "deflation" in r], gate)
                   if lo <= r["step"] <= hi]
            m, f, k = (sum(g[key] for g in sel) for key in ("matrices", "fired", "deflated_pairs"))
            return f"{f / m:.2f} of matrices fired, mean {k / max(f, 1):.1f} pairs"
        lines += ["", "Deflation gate (all 48 body matrices per step):",
                  f"- steps 1–100: {phase(1, 100)}", f"- steps 101–700: {phase(101, 700)}",
                  f"- steps 701–1469: {phase(701, 1469)}"]
    spectra = {tag: momentum(run) for tag, run in RUNS.items()}
    lines += ["", "Momentum ρ1 and median (Frobenius-normalized), mean over steps 1300–1469:", "",
              "| depth | kind | " + " | ".join(f"{t} ρ1 / median" for t in RUNS) + " |",
              "|---|---|" + "---|" * len(RUNS)]
    for block, role in ROLES.items():
        for kind in KINDS:
            name = f"block{block:02d}.{kind}"
            cells = [window(spectra[t], name, 1300, 1469) for t in RUNS]
            lines.append(f"| {role} | {kind} | " + " | ".join(f"{r:.2f} / {m:.2e}" for r, m in cells) + " |")
    text = "\n".join(lines) + "\n"
    (HERE / "comparison.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()

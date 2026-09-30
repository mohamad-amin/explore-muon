"""Old (biases, LayerNorm) versus frontier-norm variant at depth 8, for one optimizer arm.

Reads saved spectra and spike-diagnostic summaries only. Run from the project root:
.venv/bin/python logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925/compare_variant.py muon
"""

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent.parent
OLD = {"muon": (LOGS / "muon_spectra/depth8_w512_20260925_r2/scientific",
                LOGS / "muon_spectra/spike_diagnostics_20260925/science/muon_d8"),
       "adamw": (LOGS / "adamw_spectra/g20_20260924_212246_r2/77m_seed260924",
                 LOGS / "muon_spectra/spike_diagnostics_20260925/science/adamw_d8")}
KINDS = ("q", "k", "v", "o", "up", "down")
ROLES = {2: "early", 4: "mid", 6: "late", 8: "final"}


def final_nll(run):
    return json.loads((run / "steps" / "step001469.json").read_text())["validation_nll"]


def spectra(run, folder):
    out = {}
    for path in sorted(glob.glob(str(run / folder / "step*.npz"))):
        step = int(re.search(r"step(\d+)", path).group(1))
        z = np.load(path)
        out[step] = {k: np.sort(z[k].astype(float))[::-1] for k in z.files}
    return out


def window(data, name, key, lo=1300, hi=1469):
    vals = []
    for step, mats in data.items():
        if lo <= step <= hi:
            s = mats[name] / np.sqrt(np.sum(mats[name] ** 2))
            vals.append(s[0] ** 2 if key == "rho1" else s[int(np.ceil(0.5 * len(s))) - 1])
    return float(np.mean(vals))


def reading(m):
    if not m["fresh_mean_gradient"]["spike_reproduced"]:
        return "not reproduced"
    mean, sink = m["mean_product"]["of_u1v1"] >= 0.5, m["position0"]["of_u1v1"] >= 0.5
    return "mean+sink" if mean and sink else "mean" if mean else "sink" if sink else "distributed"


def main(arm):
    old_run, old_diag = OLD[arm]
    new_run, new_diag = HERE / arm / "scientific", HERE / arm / "diagnostic"
    folder = "spectra_momentum" if arm == "muon" else "spectra"
    label = "momentum" if arm == "muon" else "AdamW update"
    old_s, new_s = spectra(old_run, folder), spectra(new_run, folder)
    lines = [f"# {arm}: old architecture vs frontier-norm variant (depth 8, one seed each)", "",
             f"Final validation NLL: old {final_nll(old_run):.5f}, variant {final_nll(new_run):.5f} "
             f"(difference {final_nll(new_run) - final_nll(old_run):+.5f} nats/token).", ""]
    have = (new_diag / "summary.json").exists()
    od = json.loads((old_diag / "summary.json").read_text())["matrices"]
    nd = json.loads((new_diag / "summary.json").read_text())["matrices"] if have else {}
    lines += [f"{label} spectra, mean over steps 1300–1469; readings from the spike diagnostic.", "",
              "| depth | kind | ρ1 old → new | median old → new | reading old → new | mean share old → new | pos0 share old → new |",
              "|---|---|---|---|---|---|---|"]
    for block, role in ROLES.items():
        for kind in KINDS:
            name = f"block{block:02d}.{kind}"
            row = [f"{window(old_s, name, 'rho1'):.2f} → {window(new_s, name, 'rho1'):.2f}",
                   f"{window(old_s, name, 'median'):.2e} → {window(new_s, name, 'median'):.2e}"]
            if have:
                o, n = od[name], nd[name]
                row += [f"{reading(o)} → {reading(n)}",
                        f"{o['mean_product']['of_u1v1']:.2f} → {n['mean_product']['of_u1v1']:.2f}",
                        f"{o['position0']['of_u1v1']:.2f} → {n['position0']['of_u1v1']:.2f}"]
            else:
                row += ["–", "–", "–"]
            lines.append(f"| {role} | {kind} | " + " | ".join(row) + " |")
    if have:
        for tag, diag in (("old", od), ("variant", nd)):
            counts = {}
            for m in diag.values():
                counts[reading(m)] = counts.get(reading(m), 0) + 1
            bands = {b["band"]: [] for b in next(iter(diag.values()))["bands"]}
            for m in diag.values():
                for b in m["bands"]:
                    if b["ratio"] is not None:
                        bands[b["band"]].append(b["ratio"])
            lines += ["", f"{tag}: readings {counts}; cross-fitted/in-sample medians " +
                      ", ".join(f"{k} {np.median(v):.2f}" for k, v in bands.items())]
    text = "\n".join(lines) + "\n"
    (HERE / f"comparison_{arm}.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "muon")

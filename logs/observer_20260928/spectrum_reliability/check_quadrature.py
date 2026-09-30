"""CPU-only sensitivity audit of existing step-spectrum quadrature artifacts.

Re-Lanczos each saved 48-node measure as a tiny diagonal matrix. Truncations
recover the original Lanczos prefix rules (up to rounding), including the
mixed slope measure. This is NOT a higher-resolution spectral measurement.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
AUDIT = ROOT / "muon_spectra/second_order_audit_20260926"
OUT = Path(__file__).resolve().parent


def rule_prefixes(d):
    theta = np.array(d["theta"])
    e = np.array(d["energy"])
    slope = np.array(d["slope"])
    norm = np.sqrt(e.sum())
    v = np.sqrt(e) / norm
    mixed = slope / np.sqrt(e)
    basis, alpha, beta, outputs = [], [], [], {}
    for j in range(len(theta)):
        basis.append(v.copy())
        w = theta * v
        alpha.append(v @ w)
        V = np.array(basis)
        for _ in range(2):
            w -= V.T @ (V @ w)
        k = j + 1
        if k in (8, 16, 24, 32, 40, 48):
            T = np.diag(alpha) + np.diag(beta, 1) + np.diag(beta, -1)
            nodes, Y = np.linalg.eigh(T)
            dp = norm * Y[0]
            sp = dp * (Y.T @ (V @ mixed))
            outputs[k] = {"theta": nodes, "energy": dp**2, "slope": sp,
                          "curvature": nodes * dp**2}
        if k < len(theta):
            b = np.linalg.norm(w)
            beta.append(b)
            v = w / b
    return outputs


def band(d, cutoff):
    t, e, s, c = (np.asarray(d[key]) for key in ("theta", "energy", "slope", "curvature"))
    low = t < cutoff
    slope = s[low].sum()
    curv = c[low].sum()
    return {"nodes": int(low.sum()), "energy_share": float(e[low].sum()/e.sum()),
            "slope_share": float(slope/s.sum()), "curvature_share": float(curv/c.sum()),
            "slope": float(slope), "curvature": float(curv),
            "cstar": float(-slope/curv) if curv > 0 else None}


def main():
    rows = []
    for f in sorted((AUDIT / "step_spectrum").glob("*.json")):
        r = json.loads(f.read_text())
        if "directions" not in r:
            continue
        d = r["directions"]["actual"]
        top = max(max(v["theta"]) for v in r["directions"].values())
        cuts = {"own_relative": top * 1e-4, "absolute_1e-4": 1e-4, "absolute_1e-3": 1e-3, "absolute_1e-2": 1e-2}
        prefixes = rule_prefixes(d)
        row = {"file": str(f.relative_to(ROOT)), "item": r["item"], "batch_tokens": r["batch_tokens"],
               "step": r["step"], "top": top, "cutoff": cuts["own_relative"],
               "first_nodes": d["theta"][:3], "flat": band(d, cuts["own_relative"]),
               "norm": d["norm"], "curvature_total": d["curvature_total"], "slope_total": d["slope_total"],
               "prefixes": {k: {name: band(x, cut) for name, cut in cuts.items()} for k, x in prefixes.items()},
               "absolute": {name: band(d, cut) for name, cut in cuts.items()}}
        row["sum_curvature_rel_error"] = float((sum(d["curvature"]) - d["curvature_total"])/d["curvature_total"])
        row["sum_slope_rel_error"] = float((sum(d["slope"]) - d["slope_total"])/d["slope_total"])
        # Independent profile run has random-start top Ritz nodes; same G set.
        profile_file = AUDIT / "step_profile" / f.name
        if profile_file.exists():
            p = json.loads(profile_file.read_text())
            row["profile_q_same_set"] = p["directions"]["actual"]["curvature_curv_set"]
            row["profile_q_other_set"] = p["directions"]["actual"]["curvature"]
            row["sample_q_ratio"] = row["profile_q_other_set"] / row["profile_q_same_set"]
        rows.append(row)
    (OUT / "quadrature_sensitivity.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("states", len(rows), "flat node counts", sorted({r["flat"]["nodes"] for r in rows}))
    for r in rows:
        if r["batch_tokens"] == 16777216 and r["step"] in (46,83):
            arm = r["item"].split("/")[-1]
            vals = [r["prefixes"][k]["own_relative"]["cstar"] for k in (8,16,24,32,40,48)]
            print(arm, "top",round(r["top"],3),"nodes",r["flat"]["nodes"],"c*",[None if x is None else round(x,2) for x in vals],
                  "slope%",[round(r["prefixes"][k]["own_relative"]["slope_share"]*100,2) for k in (8,16,24,32,40,48)],
                  "qsample", round(r.get("sample_q_ratio",0),3))


if __name__ == "__main__":
    main()

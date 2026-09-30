"""Reanalyze saved Lanczos measures; no new GN products or training.

All prefixes use the same full48 top Ritz value for relative cutoffs. This
isolates resolution sensitivity from movement of the normalization itself.
The reconstructed diagonal 48-node measure is not the original GN operator.
"""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
AUDIT = ROOT / "logs/muon_spectra/second_order_audit_20260926"
OLD = ROOT / "logs/observer_20260928/spectrum_reliability/check_quadrature.py"
spec = importlib.util.spec_from_file_location("existing_readonly_audit", OLD)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)


def arm_label(item):
    arm = item.rsplit(":", 1)[0].split("/")[-1]
    return arm.split("_")[0] + ("_b08" if "mom0.8" in arm else "")


def metrics(d, lo, hi):
    t, e, s, c = (np.asarray(d[k]) for k in ("theta", "energy", "slope", "curvature"))
    keep = (t >= lo) & (t < hi)
    slope, q = float(s[keep].sum()), float(c[keep].sum())
    return {"nodes": int(keep.sum()), "energy_share": float(e[keep].sum() / e.sum()),
            "slope_share": float(slope / s.sum()), "curvature_share": float(q / c.sum()),
            "slope": slope, "curvature": q, "cstar": -slope / q if q > 0 else None}


def main():
    inputs = [OLD, OLD.with_name("quadrature_sensitivity.json"), AUDIT / "step_spectrum_probe.py",
              AUDIT / "plot_step_spectrum.py", AUDIT / "one_step_gn.py"]
    results, long = [], []
    raw = []
    for f in sorted((AUDIT / "step_spectrum").glob("*.json")):
        r = json.loads(f.read_text())
        if "directions" in r:
            raw.append((f, r))
    # Shared threshold within each matched batch/step, using Muon's top Ritz value.
    muon_top = {(r["batch_tokens"], r["step"]): max(max(d["theta"]) for d in r["directions"].values())
                for _, r in raw if arm_label(r["item"]) == "M"}
    for f, r in raw:
        inputs.append(f)
        d = r["directions"]["actual"]
        top = max(max(v["theta"]) for v in r["directions"].values())
        prefixes = old.rule_prefixes(d)
        cutoffs = {"relative_1e-4": 1e-4 * top, "relative_1e-3": 1e-3 * top,
                   "relative_1e-2": 1e-2 * top, "absolute_1e-4": 1e-4,
                   "absolute_1e-3": 1e-3, "absolute_1e-2": 1e-2}
        if (r["batch_tokens"], r["step"]) in muon_top:
            cutoffs["shared_muon_relative_1e-4"] = 1e-4 * muon_top[(r["batch_tokens"], r["step"])]
        bands = {key: (0.0, value) for key, value in cutoffs.items()}
        bands.update({f"band_{lo:g}_{hi:g}": (lo * top, hi * top)
                      for lo, hi in ((1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 1e-1))})
        bands["band_0.1_inf"] = (0.1 * top, float("inf"))
        row = {"file": str(f.relative_to(ROOT)), "batch": r["batch_tokens"] // 1048576,
               "step": r["step"], "arm": arm_label(r["item"]), "top": top, "cutoffs": cutoffs,
               "norm": d["norm"], "slope_total": d["slope_total"], "curvature_total": d["curvature_total"],
               "cstar_total": -d["slope_total"] / d["curvature_total"],
               "quality_total": d["slope_total"] ** 2 / (2 * d["curvature_total"]),
               "first_nodes": d["theta"][:4], "prefixes": {}, "checks": {}, "energy_lower_bounds": {}}
        for k, rule in prefixes.items():
            row["prefixes"][k] = {}
            for label, (lo, hi) in bands.items():
                met = metrics(rule, lo, hi)
                row["prefixes"][k][label] = met
                long.append({"file": row["file"], "batch": row["batch"], "step": row["step"],
                             "arm": row["arm"], "prefix": k, "band": label, "lower": lo,
                             "upper": hi, **met})
        for quantity, direct in (("energy", d["norm"] ** 2), ("slope", d["slope_total"]),
                                 ("curvature", d["curvature_total"])):
            row["checks"][quantity + "_direct_relerror"] = abs(sum(d[quantity]) / direct - 1)
            row["checks"][quantity + "_prefix_max_relerror"] = max(abs(sum(rule[quantity]) / sum(d[quantity]) - 1)
                                                                       for rule in prefixes.values())
        row["checks"]["full48_node_max_abs_error"] = float(np.max(np.abs(prefixes[48]["theta"] - d["theta"])))
        for label, cutoff in cutoffs.items():
            # For PSD G, q >= cutoff * ||P_{>=cutoff} D||^2. Thus this
            # lower bound uses the direct first moment, not Ritz projectors.
            row["energy_lower_bounds"][label] = max(0.0, 1 - d["curvature_total"] / (cutoff * d["norm"] ** 2))
        profile = AUDIT / "step_profile" / f.name
        if profile.exists():
            inputs.append(profile)
            p = json.loads(profile.read_text())["directions"]["actual"]
            row["q_other_sample_ratio"] = p["curvature"] / p["curvature_curv_set"]
        results.append(row)
    (OUT / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    with (OUT / "bands.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(long[0]))
        writer.writeheader()
        writer.writerows(long)
    provenance = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    (OUT / "source_hashes.json").write_text(json.dumps(provenance, indent=2) + "\n")
    summary = {"states": len(results), "checks": {key: max(r["checks"][key] for r in results)
                                                  for key in results[0]["checks"]},
               "flat_nodes48": sorted({r["prefixes"][48]["relative_1e-4"]["nodes"] for r in results})}
    for low in (8, 24, 32, 40, 48):
        vals = [r["prefixes"][k]["relative_1e-4"] for r in results for k in r["prefixes"] if k >= low]
        summary[f"prefix{low}plus"] = {field: [min(v[field] for v in vals if v[field] is not None),
                                               max(v[field] for v in vals if v[field] is not None)]
                                         for field in ("energy_share", "slope_share", "cstar")}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("\nEvidence table: own-relative flat band")
    print("batch step arm top | energy48 | slope32/40/48 % | cstar32/40/48 | slope abs1e-3 % | PSD lower bound")
    for r in results:
        if (r["batch"], r["step"]) not in ((16, 46), (16, 83), (4, 183), (1, 500)):
            continue
        v = r["prefixes"][48]["relative_1e-4"]
        slopes = "/".join(f'{100*r["prefixes"][k]["relative_1e-4"]["slope_share"]:.2f}' for k in (32,40,48))
        cs = "/".join(f'{r["prefixes"][k]["relative_1e-4"]["cstar"]:.2f}' for k in (32,40,48))
        print(r["batch"], r["step"], r["arm"], f'{r["top"]:.3f}', f'{v["energy_share"]*100:.2f}', slopes, cs,
              f'{100*r["prefixes"][48]["absolute_1e-3"]["slope_share"]:.2f}',
              f'{100*r["energy_lower_bounds"]["relative_1e-4"]:.1f}')


if __name__ == "__main__":
    main()

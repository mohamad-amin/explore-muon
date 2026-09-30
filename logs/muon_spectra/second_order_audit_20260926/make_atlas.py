"""Build the interactive atlas page (HTML with embedded data) from the audit's measurements.

usage: make_atlas.py OUT_HTML
Reads marginals/ (one-sided GN marginals), figures_gap/frame_summary.pt (gap-map bins), the batch-size studies' curves,
gn2/ and gn/ (exact one-step GN comparisons), persistence_lags/lags.json and split_vs_diag.json, and fills
atlas_template.html next to this script.
"""
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
KINDS = ("q", "k", "v", "o", "up", "down")
RUNS = ("M", "PD", "S", "SPD")
EDGES = np.logspace(-12, 6, 73)
BATCH = 2048


def r4(values):
    return [None if not np.isfinite(v) else float(f"{v:.4g}") for v in np.asarray(values, float)]


def log_ranks(d, points=44):
    return sorted({int(round(v)) for v in np.logspace(0, np.log10(d), points)})


def segment_median(values, ranks):
    out = []
    edges = [0] + [int(np.sqrt(ranks[i] * ranks[i + 1])) for i in range(len(ranks) - 1)] + [len(values)]
    for i in range(len(ranks)):
        lo, hi = max(edges[i], ranks[i] - 1 if i == 0 else edges[i]), max(edges[i + 1], edges[i] + 1)
        out.append(float(np.median(values[lo:hi])))
    return out


def marginals():
    out = {}
    for path in sorted((HERE / "marginals").glob("*.pt")):
        m = re.match(r"(M|PD|S|SPD)_.*_step(\d+)\.pt", path.name)
        if not m:
            continue
        run, step = m.group(1), int(m.group(2))
        arrays = torch.load(path, weights_only=False)["arrays"]
        report = json.loads(path.with_suffix(".json").read_text())["matrices"]
        entry = {}
        for name, a in arrays.items():
            lam = a["lam_C"].double().numpy()
            ratio = (a["in_exact_in"].double().numpy() / (a["trB"] * np.clip(lam, 1e-30, None)))
            ranks = log_ranks(len(lam))
            lb = a["lam_B"].double().numpy()
            rb = log_ranks(len(lb))
            rep = report[name]
            entry[name] = {
                "ri": ranks, "r": r4(segment_median(ratio, ranks)),
                "lc": r4(np.clip(lam[np.array(ranks) - 1], 1e-30, None) / lam.mean()),
                "rb": rb, "lb": r4(np.clip(lb[np.array(rb) - 1], 1e-30, None) / lb.mean()),
                "md": r4([rep["in"]["mean_direction"]["exact_over_kfac"]])[0],
                "sc": r4([rep["in"]["mean_direction"]["share_of_C"]])[0],
                "a": r4([rep["fit_exact"]["a_within"]])[0], "b": r4([rep["fit_exact"]["b_between"]])[0],
                "ns": r4([rep["gradient"]["noise_scale_sequences"]])[0],
            }
        out.setdefault(run, {})[step] = entry
    return out


def gap():
    path = HERE / "figures_gap" / "frame_summary.pt"
    if not path.exists():
        return {}
    summary = torch.load(path, weights_only=False)
    centers = np.concatenate([[EDGES[0] / 2], np.sqrt(EDGES[1:] * EDGES[:-1]), [EDGES[-1] * 2]])
    out = {}
    for run, steps in summary.items():
        for step, entry in steps.items():
            for kind, b in entry["bins"].items():
                b = {q: v.numpy() for q, v in b.items()}
                ok = b["count"] > 50
                dec = np.clip(b["dec"], 0, None)
                with np.errstate(divide="ignore", invalid="ignore"):
                    share = dec / max(dec.sum(), 1e-300)
                    bstar = np.where(b["s"] > 0, b["n"] / b["s"], np.nan)
                    ratio = (b["gD"] / b["s"]) / (b["eta_s"] / b["s"])
                    good = ok & (b["s"] > 0) & np.isfinite(ratio) & (ratio > 0)
                    if good.any() and dec[good].sum() > 0:
                        order = np.argsort(ratio[good])
                        cum = np.cumsum(dec[good][order]) / dec[good].sum()
                        ratio = ratio / ratio[good][order][np.searchsorted(cum, 0.5)]
                    over = np.where(b["gD"] > 0, b["hDD"] / (2 * b["gD"]), np.nan)
                    cos = b["Mg"] / np.sqrt(b["MM"] * b["gg"])
                    reach = {B: float(dec[(b["s"] > 0) & (b["n"] / np.where(b["s"] > 0, b["s"], 1) < B)].sum() /
                                      max(dec.sum(), 1e-300)) for B in (512, 2048, 8192, 32768, 131072)}
                idx = np.where(ok)[0]
                out.setdefault(run, {}).setdefault(step, {})[kind] = {
                    "x": r4(centers[idx]), "dec": r4(share[idx]), "bstar": r4(bstar[idx]),
                    "eff": r4(np.where(good, ratio, np.nan)[idx]), "over": r4(over[idx]), "cos": r4(cos[idx]),
                    "reach": {str(k): round(v, 4) for k, v in reach.items()}}
    return out


def batch_study():
    out = {}
    for arm in sorted(glob.glob(str(REPO / "logs/muon_spectra/soaudit_batch4m_20260927/*_b4M_*"))) + \
            sorted(glob.glob(str(REPO / "logs/muon_spectra/soaudit_alpha4m_20260927/*_b4M_*"))) + \
            sorted(glob.glob(str(REPO / "logs/muon_spectra/soaudit_ts4m_20260927/*_b4M_*"))) + \
            sorted(glob.glob(str(REPO / "logs/muon_spectra/soaudit_mom4m_20260927/*_b4M_*"))) + \
            sorted(glob.glob(str(REPO / "logs/muon_spectra/soaudit_mom4m2_20260927/*_b4M_*"))) + \
            sorted(glob.glob(str(REPO / "logs/muon_spectra/soaudit_traj_20260926/*"))):
        steps = sorted(glob.glob(arm + "/scientific/steps/step*.json"))
        curve = [(json.load(open(p))["tokens"], json.load(open(p)).get("validation_nll")) for p in steps]
        curve = [(t, v) for t, v in curve if v]
        if curve:
            out[Path(arm).name] = {"tokens": [t for t, _ in curve], "val": r4([v for _, v in curve])}
    return out


def gn_states():
    """Exact-GN one-step comparisons (one_step_gn.py): newest folder first, one entry per state."""
    label = {"M": "Muon", "PD": "PD α¼", "S": "SOAP-Muon", "SPD": "S∘PD"}
    out = {}
    for folder in ("gn2", "gn"):
        for path in sorted((HERE / folder).glob("*.json")):
            m = re.match(r"(M|PD|S|SPD)_.*_step(\d+)\.json", path.name)
            if not m:
                continue
            r = json.loads(path.read_text())
            if not r.get("inputs"):
                continue
            key = f"{label[m.group(1)]}, {'4M' if '_b4M_' in path.name else '1M'} run, step {int(m.group(2))}"
            if key in out:
                continue
            out[key] = {source: {"best": e["best"], "rho": e.get("rho"),
                                 "scores": {k: float(f"{v['best_decrease'] * 1e3:.5g}") for k, v in e["scores"].items()}}
                        for source, e in r["inputs"].items()}
    return out


def block_gn():
    """Cross-fitted one-step decreases of exact block-diagonal GN vs full GN (one_step_blockgn.py)."""
    out = {}
    for path in sorted((HERE / "blockgn").glob("*_g4M_*.json")):
        r = json.loads(path.read_text())
        state = path.name.split("_g4M")[0]
        entry = out.setdefault(state, {})
        for family, v in r.get("families", {}).items():
            entry[family] = float(f"{v['cross_fit']['cross_fitted'] * 1e3:.5g}")
    staged = {}
    names = {"M1Mg": "Muon 1M @500, fresh 4M gradient", "M1Mm": "Muon 1M @500, momentum (c = 0.95)",
             "PD1Mm": "PD 1M @500, momentum (c = 0.95)", "M1Mc0.25": "Muon 1M @500, g + 0.25 M",
             "M1Mc0.5": "Muon 1M @500, g + 0.5 M"}
    for path in sorted((HERE / "blockgn").glob("gsmap_*.json")):
        r = json.loads(path.read_text())
        tag = path.stem.replace("gsmap_", "")
        staged[names.get(tag, tag)] = {family: float(f"{v['cross_fit']['cross_fitted'] * 1e3:.5g}")
                                       for family, v in r.get("families", {}).items()}
    return {"blocks": out, "staged": staged}


def transport_study(folder="transport"):
    """transport_test.py results: staleness vs curvature transport, signal/noise and the momentum filter by curvature
    bin (pooled per kind), exact stiff-direction projections, and the one-step table."""
    import numpy as np
    out = {}
    for path in sorted((HERE / folder).glob("*_step*.json")):
        r = json.loads(path.read_text())
        if "frame_bins" not in r:
            continue
        names, K, beta = r["names"], r["replay"], r["beta"]
        geometric = (1 - beta ** K) / (1 - beta)
        edges = np.array(r["bins"]["edges"])
        centers = np.concatenate([[edges[0] / 1.5], np.sqrt(edges[1:] * edges[:-1]), [edges[-1] * 1.5]])
        kinds = {}
        for kind in KINDS + ("all",):
            members = [n for n in names if kind == "all" or n.endswith("." + kind)]
            acc = {}
            for n in members:
                for key, values in r["frame_bins"][n].items():
                    acc[key] = acc.get(key, 0) + np.array(values, dtype=float)
                acc["sumsq"] = acc.get("sumsq", 0) + np.array(r["noise"]["sum_sq_batches"][n], dtype=float)
            count = np.maximum(acc["count"], 1)
            noise = (acc["sumsq"] - K * acc["gbar*gbar"]) / (K - 1)
            ok = acc["count"] > 50
            clean = lambda a: [float(f"{x:.4g}") if (o and np.isfinite(x)) else None for x, o in zip(a, ok)]
            with np.errstate(divide="ignore", invalid="ignore"):
                kinds[kind] = {
                    "signal": clean(acc["gbar*gbar"] / count), "noise": clean(noise / count),
                    "snr": clean(acc["gbar*gbar"] / noise),
                    "gain_M": clean(acc["M*gbar"] / acc["gbar*gbar"] / geometric),
                    "gain_Mstar": clean(acc["Mstar*gbar"] / acc["gbar*gbar"] / geometric),
                    "stale": clean(acc["bS*bS"] / (beta ** 2 * acc["M*M"])),
                    "cos_G": clean(-acc["bS*GQ"] / np.sqrt(acc["bS*bS"] * acc["GQ*GQ"])),
                    "cos_H": clean(-acc["bS*HQ"] / np.sqrt(acc["bS*bS"] * acc["HQ*HQ"])),
                    "share": clean(acc["count"] / acc["count"].sum())}
        probes = np.array(r["replay_check"].get("probe_projections") or [[]])
        ritz = {}
        if probes.size:
            ritz = {"values": [float(f"{x:.4g}") for x in r["ritz"]["values"]],
                    "gbar": [float(f"{x:.4g}") for x in probes.mean(0) * geometric],
                    "noise_sd": [float(f"{x:.4g}") for x in probes.std(0, ddof=1)],
                    **{k: [float(f"{x:.4g}") for x in v] for k, v in r["ritz"].get("projections", {}).items()
                       if k in ("M", "Mstar", "bS", "GQ", "HQ")}}
        stale = r["staleness"]
        totals = {label: {g: {k: float(f"{d[g][k]:.4g}") for k in ("cos", "s_star", "residual_s1")}
                          for g in (*KINDS, "total")} for label, d in stale["predictions"].items()}
        one_step = {inp: {fam: float(f"{v['cross_fit']['cross_fitted'] * 1e3:.5g}") for fam, v in e["families"].items()}
                    for inp, e in r.get("one_step", {}).items()}
        out[f"{Path(r['arm']).name}:{r['step']}"] = {
            "beta": beta, "replay": K, "centers": [float(f"{x:.4g}") for x in centers], "kinds": kinds, "ritz": ritz,
            "norm_ratio": stale["norm_S"] / stale["norm_M"], "cos_M_Mstar": stale["cos_M_Mstar"], "totals": totals,
            "one_step": one_step, "check": [r["replay_check"]["check_loss"], r["replay_check"]["logged_loss"]]}
    return out


def gap_to_gn():
    """Valley-floor premise checks and GN-trainer runs: anneal curves, one-step shares of GN, trainer validation."""
    out = {"anneal": {}, "rescore": {}, "trainer": {}, "baseline": {}}
    for path in sorted((HERE / "valley").glob("*/anneal.json")):
        r = json.loads(path.read_text())
        out["anneal"][path.parent.name] = {"loss": [float(f"{x:.5g}") for x in r["loss"]], "lr": r["lr"],
                                           "start": r["start_step"], "batch": r["batch_tokens"]}
    for path in sorted(list((HERE / "valley").glob("rescore_*.json")) + list((HERE / "valley").glob("mom_*.json"))):
        r = json.loads(path.read_text())
        for inp, e in r.get("inputs", {}).items():
            if "gn" not in e.get("families", {}):
                continue
            cf = {f: v["cross_fit"]["cross_fitted"] * 1e3 for f, v in e["families"].items()}
            out["rescore"][f"{path.stem.replace('rescore_', '').replace('mom_', '')} {inp}"] = {k: float(f"{v:.4g}") for k, v in cf.items()} | {
                "stiff16": float(f"{e['stiff16_decrement'] * 1e3:.4g}")}
    for root in ("gntrain_sweep", "gntrain_pair", "gntrain_ref", "newton_ref"):
        for run in sorted((HERE / root).glob("*")) if (HERE / root).exists() else []:
            rows = []
            for f in sorted((run / "steps").glob("step*.json")):
                r = json.loads(f.read_text())
                if "validation_nll" in r:
                    rows.append([r["step"], r["validation_nll"], r.get("alpha")])
            if rows:
                out["trainer"][f"{root}/{run.name}"] = rows
    for name, arm in (("Muon 1M", "soaudit_traj_20260926/M_lr0.007_s260925_l40s"), ("PD 1M", "soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada")):
        rows = []
        for f in sorted((HERE.parent / arm / "scientific" / "steps").glob("step*.json")):
            r = json.loads(f.read_text())
            if "validation_nll" in r:
                rows.append([r["step"], r["validation_nll"]])
        out["baseline"][name] = rows
    return out


def step_rule_study():
    """Greedy step rule vs normalized steps, floors under the oscillation, step-equivalent speedups, normalized GN."""
    root = HERE.parent

    def val_curve(arm):
        out = {}
        for f in (Path(arm) / "scientific" / "steps").glob("step*.json"):
            r = json.loads(f.read_text())
            if "validation_nll" in r:
                out[r["step"]] = r["validation_nll"]
        return out

    def steps_to(c, target):
        s = sorted(c)
        for a, b in zip(s, s[1:]):
            if c[b] <= target < c[a]:
                f = (c[a] - target) / (c[a] - c[b])
                return math.exp(math.log(max(a, 1)) + f * (math.log(b) - math.log(max(a, 1))))
        return None

    out = {"floors": {}, "speedup": {}, "greedy": {}, "normgn": {}, "probe": {}}
    for label, pattern in (("Muon 1M @0.007", "M_lr0.007_s260925_l40s_a*"), ("PD α¼ 1M @0.01", "PD_a0.25_lr0.01_s260925_ada_a*"),
                           ("Muon 4M β0.9 @0.014", "M_b4M_lr0.014_mom0.9_s260925_l40s_a*"),
                           ("PD α½ 4M β0.9 @0.02", "PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada_a*")):
        pts = []
        for d in (HERE / "valley").glob(pattern):
            f = d / "anneal.json"
            if f.exists():
                r = json.loads(f.read_text())
                pts.append([r["start_step"], round(r["loss"][0], 5), round(r["loss"][-1], 5)])
        if pts:
            out["floors"][label] = sorted(pts)
    ref4 = val_curve(root / "soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s")
    for label, arm in (("PD α¼ @0.02 (4M)", "soaudit_mom4m3_20260927/PD_b4M_lr0.02_mom0.9_s260925_ada"),
                       ("PD α½ @0.02 (4M)", "soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada"),
                       ("TS ½/½ @0.02 (4M)", "soaudit_strength4m_20260927/TS_a0.5_b0.5gn_b4M_lr0.02_mom0.9_s260925_ada"),
                       ("S∘PD α½ @0.02 (4M)", "soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s")):
        c = val_curve(root / arm)
        pts = [[t, round(t / n, 4)] for t in sorted(ref4) if 50 <= t <= 330 and (n := steps_to(c, ref4[t]))]
        if pts:
            out["speedup"][label] = pts
    m1 = val_curve(root / "soaudit_traj_20260926/M_lr0.007_s260925_l40s")
    p1 = val_curve(root / "soaudit_traj_20260926/PD_a0.25_lr0.01_s260925_ada")
    out["speedup"]["PD α¼ @0.01 (1M)"] = [[t, round(t / n, 4)] for t in sorted(m1) if 50 <= t <= 1300 and (n := steps_to(p1, m1[t]))]
    for state, arm in (("Muon state", "M_lr0.007_s260925_l40s"), ("PD state", "PD_a0.25_lr0.01_s260925_ada")):
        base = val_curve(root / "soaudit_traj_20260926" / arm)
        entry = {"baseline": [[s, round(v, 5)] for s, v in sorted(base.items()) if 500 <= s <= 700]}
        prefix = "M1M" if state.startswith("Muon") else "PD1M"
        for label, d in (("greedy Muon", f"{prefix}_greedymuon_from500"), ("greedy Newton, momentum 0.95", f"{prefix}_mom0.95_from500"),
                         ("greedy Newton, momentum 0.9", f"{prefix}_mom0.9_from500"), ("greedy Newton, per batch", f"{prefix}_from500_v2"),
                         ("normalized GN (Krylov 64, d 1e-3)", f"{prefix}_normgn_k64_d1e-3_from500"),
                         ("Muon in the Newton harness (control)", f"{prefix}_polarcontrol_from500")):
            rows = []
            for f in sorted((HERE / "newton_ref" / d / "steps").glob("step*.json")):
                r = json.loads(f.read_text())
                if "validation_nll" in r and r["step"] <= 700:
                    rows.append([r["step"], round(r["validation_nll"], 5)])
            if rows:
                entry[label] = [[500, round(base[500], 5)]] + rows
        out["greedy"][state] = entry
    gn_dir = HERE / "newton_ref" / "M1M_normgn_k64_d1e-3_from500" / "steps"
    base_steps = root / "soaudit_traj_20260926/M_lr0.007_s260925_l40s/scientific/steps"
    rows = []
    for f in sorted(gn_dir.glob("step*.json")):
        r = json.loads(f.read_text())
        b = json.loads((base_steps / f.name).read_text())
        rows.append([r["step"], round(r["train_nll"] - b["train_nll"], 5), round(r["ritz_top"], 3), round(r["rho"], 3), round(r.get("cos_muon_mean", float("nan")), 4)])
    out["normgn"] = {"rows": rows}
    for state, name in (("Muon state @500", "M1M_500"), ("PD α¼ state @500", "PD1M_500")):
        main = json_if(f"normgn_probe/{name}.json")
        refs = json_if(f"normgn_probe/{name}_refs.json")
        if not main:
            continue
        base = main["base_loss"]
        curves = {}
        for label, key in (("the run's actual step", "actual"), ("Muon's direction", "muon"), ("momentum at the same norms", "momentum_at_muon_norms")):
            src = refs.get("references", {}).get(key) or main.get("references", {}).get(key)
            if src:
                curves[label] = [round(v - base, 5) for v in src["losses"]]
        for set_label, entry in main["sets"].items():
            v = entry["directions"].get("k64_d0.001")
            if v:
                curves[f"GN normalized, {set_label}"] = [round(x - base, 5) for x in v["s"]["losses"]]
        out["probe"][state] = {"scales": [0.25, 0.5, 1.0, 2.0], "curves": curves}
    sp = json_if("speedup_by_batch.json")
    out["matched"] = {f"{batch}: {label}": [[round(loss, 3), v] for _, v, loss in a["speedup"]]
                      for batch, entry in sp.items() for label, a in entry["arms"].items()}
    maps = {}
    for state, name in (("oscillating @500", "M1M_500"), ("floor @516", "M1M_a516")):
        gp = json_if(f"gnpd_probe/{name}.json").get("inputs", {}).get("momentum_g1M", {}).get("families", {})
        wp = json_if(f"weighted_probe/{name}.json").get("inputs", {}).get("momentum_g1M", {})
        row = {k: round(gp[k]["cross_fit"]["cross_fitted"] * 1e3, 3) for k in ("muon", "pd_a0.25", "pd_a0.5", "gn", "gnpd_p0.25", "gnpd_p0.5") if k in gp}
        if "pdw_a0.5" in wp:
            row["pdw_a0.5"] = round(wp["pdw_a0.5"] * 1e3, 3)
        if row:
            maps[state] = row
    out["maps"] = maps
    finals = {}
    for arm in sorted([a for c in ("soaudit_batch16m_20260927", "soaudit_sts_20260928", "soaudit_soapside16m_20260928", "soaudit_b16mlong_20260928", "soaudit_soapmode_20260928", "soaudit_seed16m_20260928", "soaudit_b16mlong_lr_20260928")
                       for a in (root / c).glob("*_b16M_*") if a.is_dir()]):
        best = None
        for f in (arm / "scientific" / "steps").glob("step*.json"):
            r = json.loads(f.read_text())
            if "validation_nll" in r and (best is None or r["step"] > best[0]):
                best = (r["step"], r["validation_nll"])
        if best and best[0] > 0:
            finals[arm.name] = round(best[1], 4)
    out["finals16m"] = finals
    cool = {}
    base_steps = root / "soaudit_traj_20260926/M_lr0.007_s260925_l40s/scientific/steps"
    cool["Muon baseline"] = [[s_, round(json.loads((base_steps / f"step{s_:06d}.json").read_text())["validation_nll"], 5)]
                             for s_ in (1300, 1350, 1400, 1450, 1469) if (base_steps / f"step{s_:06d}.json").exists()]
    for label, d in (("A: Muon in the harness", "A_polar"), ("B: PD in GN geometry, 1× LR", "B_gnpd_p0.25"),
                     ("B3: PD in GN geometry, 3× LR", "B3_gnpd_p0.25_lr3x"), ("C: Muon, then GN-PD for the last 19 steps", "C_polar_then_gnpd_lr3x")):
        rows = []
        for f in sorted((HERE / "cooldown_gnpd" / d / "steps").glob("step*.json")):
            r = json.loads(f.read_text())
            if "validation_nll" in r:
                rows.append([r["step"], round(r["validation_nll"], 5)])
        if rows:
            cool[label] = [[1300, 3.79478]] + rows if rows[0][0] != 1300 else rows
    out["cooldown"] = cool
    decomp = {}
    for state, name in (("oscillating @500", "M1M_500"), ("floor @516", "M1M_a516")):
        gp = json_if(f"gnpd_probe/{name}.json").get("inputs", {}).get("momentum_g1M", {}).get("families", {})
        bp = json_if(f"blockgnpd_probe/{name}.json").get("inputs", {}).get("momentum_g1M", {}).get("families", {})
        row = {}
        if gp:
            row["PD α½"] = gp["pd_a0.5"]["cross_fit"]["cross_fitted"] * 1e3
            row["damped GN"] = gp["gn"]["cross_fit"]["cross_fitted"] * 1e3
            row["GN-PD, full GN"] = max(gp[k]["cross_fit"]["cross_fitted"] for k in ("gnpd_p0.25", "gnpd_p0.5")) * 1e3
        if bp:
            row["GN-PD, Kronecker B⊗C"] = max(bp[k]["cross_fit"]["cross_fitted"] for k in ("kronpd_p0.25", "kronpd_p0.5")) * 1e3
        for label, fname in (("GN-PD, exact per-layer blocks", f"exactblock_probe/{name}_layer.json"), ("GN-PD, exact per-matrix blocks", f"exactblock_probe/{name}_matrix.json")):
            e = json_if(fname).get("inputs", {}).get("momentum_g1M")
            if e:
                row[label] = e["cross_fit"]["cross_fitted"] * 1e3
        if row:
            decomp[state] = {k: round(v, 3) for k, v in row.items()}
    out["decomp"] = decomp
    fs = {}
    for f in sorted((HERE / "floor_steps").glob("[at]516_*.json")):
        r = json.loads(f.read_text())
        rows = [[x["k"], round(x["eval_loss"], 5)] for x in r["rows"] if "eval_loss" in x]
        if rows:
            stem = f.stem
            rule = "smoothed rule" if (stem.startswith("a516") or "ema0" not in stem) else "instantaneous rule"
            name = stem[5:].replace("_ema0", "").replace("_transport", ", transported").replace("gnpd", "PD in GN geometry")
            name = name.replace("pd", "PD α½").replace("muon", "Muon").replace("_k", ", κ ")
            fs[f"{name} ({rule})"] = rows
    out["floorsteps"] = fs
    snr = {}
    for name, label in (("SPD16M_46", "16M S∘PD @46 (loss ~5.4)"), ("PD4M_183", "4M PD α½ @183 (~4.1)"), ("PD1M_900", "1M PD α¼ @900 (~3.8)")):
        r = json_if(f"frame_snr/{name}.json")
        if r:
            snr[label] = {f: [round(r["aggregate"][f][b]["entry_fraction_snr_gt_1"], 4) for b in ("1M", "4M", "16M")] for f in ("soap", "gn_out", "raw")}
    out["framesnr"] = snr
    return out


def edge_study():
    """Step profiles (each optimizer's own step at its own state) and the GN-rate pre-flight (2026-09-28)."""
    out = {"profile": [], "preflight": {}}
    for r in json_if("step_profile_summary.json") or []:
        flips = [x for x in r["flips_top4"] if x == x]
        out["profile"].append({"batch": r["batch"], "opt": r["optimizer"], "step": r["step"], "frac": round(r["fraction"], 4),
                               "lmax": round(r["lambda_max"], 3), "cstar": round(r["c_star_actual"], 4),
                               "energy16": r["energy_top16_actual"], "curv16": round(r["curvature_top16_actual"], 4),
                               "flip": round(sorted(flips)[len(flips) // 2], 3) if flips else None,
                               "grad16": round(r["grad_top_energy_now"], 4), "quality": r["quality_actual"]})
    edges = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.01]
    batch_name = {16777216: "16M", 4194304: "4M", 1048576: "1M"}
    out["bands"] = []
    for path in sorted((HERE / "step_spectrum").glob("*.json")):
        r = json.loads(path.read_text())
        arm, step = r["item"].rsplit(":", 1)
        name = arm.split("/")[-1]
        opt = next((label for prefix, label in (("SleftPD", "S_left∘PD"), ("SrightPD", "S_right∘PD"), ("SPD", "S∘PD"), ("TS", "TS"),
                                                ("PD", "PD"), ("M_", "Muon"), ("S_", "SOAP∘Muon")) if name.startswith(prefix)), name)
        if "_mom0.8_" in name:
            opt += " β0.8"
        d = r["directions"]
        top = max(max(v["theta"]) for v in d.values())
        a = d["actual"]
        theta = [t / top for t in a["theta"]]
        cstar, share = [], []
        total_slope = sum(a["slope"])
        for lo, hi in zip(edges[:-1], edges[1:]):
            idx = [i for i, t in enumerate(theta) if lo <= t < hi]
            sl, cu = sum(a["slope"][i] for i in idx), sum(a["curvature"][i] for i in idx)
            cstar.append(round(-sl / cu, 4) if cu > 0 else None)
            share.append(round(sl / total_slope, 4) if total_slope else None)
        out["bands"].append({"batch": batch_name[r["batch_tokens"]], "opt": opt, "step": int(step), "cstar": cstar, "share": share,
                             "cstar_all": round(-total_slope / sum(a["curvature"]), 4)})
    root = HERE.parent

    def final(arm):
        """The run's last validation (step 92 at 1x, 184 at 2x); None if the run is not complete."""
        status = root / arm / "scientific" / "status.json"
        if not status.exists() or json.loads(status.read_text()).get("status") != "complete":
            return None
        best = None
        for f in (root / arm / "scientific" / "steps").glob("step*.json"):
            r = json.loads(f.read_text())
            if "validation_nll" in r and (best is None or r["step"] > best[0]):
                best = (r["step"], r["validation_nll"])
        return best[1] if best else None
    refs = {"Muon": "soaudit_batch16m_20260927/M_b16M_lr0.02_mom0.9_s260925_l40s", "PD": "soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s",
            "S∘PD": "soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada"}
    interventions = [
        ("two-tap filter, β 0.9", "Muon", "soaudit_prefilter16m_20260928/M2tap_b16M_lr0.02_mom0.9_s260925_l40s"),
        ("two-tap filter, β 0.9", "PD", "soaudit_prefilter16m_20260928/PD2tap_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
        ("two-tap filter, β 0.9", "S∘PD", "soaudit_prefilter16m_20260928/SPD2tap_a0.5_b16M_lr0.028_mom0.9_s260925_ada"),
        ("two-tap filter, β 0.8", "S∘PD", "soaudit_prefilter16m_20260928/SPD2tap_a0.5_b16M_lr0.028_mom0.8_s260925_ada"),
        ("β 0.8", "S∘PD", "soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada"),
        ("β 0.7", "S∘PD", "soaudit_mom16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.7_s260925_ada"),
        ("β 0.6", "S∘PD", "soaudit_mom16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.6_s260925_ada"),
        ("β 0.8", "PD", "soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s"),
        ("β 0.8", "Muon", "soaudit_mom16m_20260928/M_b16M_lr0.02_mom0.8_s260925_l40s"),
        ("whitened head α ½", "PD", "soaudit_headwhite16m_20260928/PDhw0.5_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
        ("whitened head α ¼", "PD", "soaudit_headwhite16m_20260928/PDhw0.25_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
        ("centered whitened head α ½", "PD", "soaudit_headcenter16m_20260928/PDhwc0.5_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
        ("centered whitened head α ¼", "PD", "soaudit_headcenter16m_20260928/PDhwc0.25_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
        ("clip 0.1, β 0.9", "Muon", "soaudit_clip16m_20260928/Mclip0.1_b16M_lr0.02_mom0.9_s260925_l40s"),
        ("clip 0.1, β 0.8 (vs clip 0.1, β 0.9)", "Muon", "soaudit_clip16m_20260928/Mclip0.1_b16M_lr0.02_mom0.8_s260925_l40s",
         "soaudit_clip16m_20260928/Mclip0.1_b16M_lr0.02_mom0.9_s260925_l40s"),
        ("β 0.8, LR 0.014 (vs β 0.8 @0.02)", "Muon", "soaudit_mlr16m_20260928/M_b16M_lr0.014_mom0.8_s260925_l40s",
         "soaudit_mom16m_20260928/M_b16M_lr0.02_mom0.8_s260925_l40s"),
        ("β 0.8, LR 0.028 (vs β 0.8 @0.02)", "Muon", "soaudit_mlr16m_20260928/M_b16M_lr0.028_mom0.8_s260925_l40s",
         "soaudit_mom16m_20260928/M_b16M_lr0.02_mom0.8_s260925_l40s"),
        ("β 0.8, seed 260926 (vs its β 0.9)", "S∘PD", "soaudit_mom16mseed_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260926_ada",
         "soaudit_seed16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.9_s260926_ada"),
        ("β 0.8, LR 0.04 (vs β 0.8 @0.028)", "S∘PD", "soaudit_strength16m_20260928/SPD_a0.5_b16M_lr0.04_mom0.8_s260925_ada",
         "soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada"),
        ("α ¼, β 0.9 (vs PD α ½ β 0.9)", "PD", "soaudit_dose16m_20260928/PD_a0.25_b16M_lr0.028_mom0.9_s260925_l40s"),
        ("α ¼, β 0.8 (vs α ¼ β 0.9)", "PD", "soaudit_dose16m_20260928/PD_a0.25_b16M_lr0.028_mom0.8_s260925_l40s",
         "soaudit_dose16m_20260928/PD_a0.25_b16M_lr0.028_mom0.9_s260925_l40s"),
        ("warmup β 0.8→0.9, 1× (vs β 0.9)", "S∘PD", "soaudit_momwarm16m_20260928/SPD_a0.5_b16M_lr0.028_momwarm0.8to0.9_s260925_ada"),
        ("warmup β 0.8→0.9, 2× (vs 2× β 0.9)", "S∘PD", "soaudit_momwarm16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada",
         "soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada"),
        ("warmup β 0.8→0.9, 2×, seed 260926 (vs its 2× β 0.9)", "S∘PD", "soaudit_warmrep_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260926_ada",
         "soaudit_warmrep_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260926_ada"),
        ("warmup β 0.8→0.9, 2× (vs 2× β 0.9)", "PD", "soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_l40s",
         "soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_l40s"),
        ("PD-top α ½, β 0.9 (vs Muon β 0.9)", "Muon", "soaudit_pdtop_20260928/PDtop_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
        ("PD-top α ½, β 0.9 (vs PD α ½ β 0.9)", "PD", "soaudit_pdtop_20260928/PDtop_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
        ("PD-top α ½, β 0.8 (vs PD-top β 0.9)", "PD", "soaudit_pdtop_20260928/PDtop_a0.5_b16M_lr0.028_mom0.8_s260925_l40s",
         "soaudit_pdtop_20260928/PDtop_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
        ("PD-top α ½, 4M (vs PD α ½ 4M)", "PD", "soaudit_pdtop_20260928/PDtop_a0.5_b4M_lr0.02_mom0.9_s260925_ada",
         "soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada"),
        ("warmup β 0.8→0.9 in 120 steps, 4M (vs 4M β 0.9)", "PD", "soaudit_warm4m_20260928/PD_a0.5_b4M_lr0.02_momwarm0.8to0.9in120_s260925_ada",
         "soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada"),
        ("warmup β 0.8→0.9 in 120 steps, 4M (vs 4M β 0.9)", "S∘PD", "soaudit_warm4m_20260928/SPD_a0.5_b4M_lr0.02_momwarm0.8to0.9in120_s260925_l40s",
         "soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s"),
        ("β 0.9, 4M (vs 4M β 0.95)", "PD", "soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada",
         "soaudit_alpha4m_20260927/PD_a0.5_b4M_lr0.02_s260925_ada"),
        ("β 0.9, 4M (vs 4M β 0.95)", "S∘PD", "soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s",
         "soaudit_alpha4m_20260927/SPD_a0.5_b4M_lr0.02_s260925_l40s"),
        ("constant β 0.85, 4M (vs 4M warmup)", "PD", "soaudit_mlaw_20260928/PD_a0.5_b4M_lr0.02_mom0.85_s260925_ada",
         "soaudit_warm4m_20260928/PD_a0.5_b4M_lr0.02_momwarm0.8to0.9in120_s260925_ada"),
        ("constant β 0.85, 4M (vs 4M warmup)", "S∘PD", "soaudit_mlaw_20260928/SPD_a0.5_b4M_lr0.02_mom0.85_s260925_l40s",
         "soaudit_warm4m_20260928/SPD_a0.5_b4M_lr0.02_momwarm0.8to0.9in120_s260925_l40s"),
        ("law 0.45→0.8, 2× (vs 2× warmup)", "S∘PD", "soaudit_mlaw_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.45to0.8in55_s260925_ada",
         "soaudit_momwarm16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada"),
        ("law 0.45→0.8, 2× (vs 2× warmup)", "PD", "soaudit_mlaw_20260928/PD_a0.5_b16M_T2x_lr0.028_momwarm0.45to0.8in55_s260925_l40s",
         "soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_l40s"),
        ("constant β 0.7, 2× (vs 2× warmup)", "S∘PD", "soaudit_mlaw_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.7_s260925_ada",
         "soaudit_momwarm16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada"),
        ("constant β 0.7, 2× (vs 2× warmup)", "PD", "soaudit_mlaw_20260928/PD_a0.5_b16M_T2x_lr0.028_mom0.7_s260925_l40s",
         "soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_l40s"),
        ("1M warmup 0.8→0.95 in 150 (vs 1M β 0.95, α ¼)", "PD", "soaudit_warm1m_20260928/PD_a0.25_lr0.01_momwarm0.8to0.95in150_s260925_a6000",
         "soaudit_warm1m_20260928/PD_a0.25_lr0.01_mom0.95_s260925_a6000"),
        ("1M PD-top α ½ (vs PD α ½, Ada)", "PD", "soaudit_pdtop1m_20260928/PDtop_a0.5_lr0.01_s260925_ada",
         "soaudit_pdtop1m_20260928/PD_a0.5_lr0.01_s260925_ada"),
        ("1M PD-top α ½ (vs PD α ½, L40S)", "PD", "soaudit_pdtop1m_20260928/PDtop_a0.5_lr0.01_s260925_l40s",
         "soaudit_pdtop1m_20260928/PD_a0.5_lr0.01_s260925_l40s"),
        ("1M PD-top α ¼ (vs PD α ¼, A6000)", "PD", "soaudit_pdtop1m_a025_20260928/PDtop_a0.25_lr0.01_mom0.95_s260925_a6000",
         "soaudit_warm1m_20260928/PD_a0.25_lr0.01_mom0.95_s260925_a6000"),
        ("1M SNR gate α ½ (vs PD α ½, A6000)", "PD", "soaudit_snrgate1m_20260928/PDgate_a0.5_lr0.01_s260925_a6000",
         "improve_w10_20260925/PD_a0.5_lr0.01_s260925_a6000"),
        ("1M SNR gate α ½ (vs PD-top α ½, A6000)", "PD", "soaudit_snrgate1m_20260928/PDgate_a0.5_lr0.01_s260925_a6000",
         "soaudit_snrgate1m_20260928/PDtop_a0.5_lr0.01_s260925_a6000"),
        ("1M SNR gate α ¼ (vs PD α ¼, A6000)", "PD", "soaudit_snrgate1m_20260928/PDgate_a0.25_lr0.01_s260925_a6000",
         "soaudit_warm1m_20260928/PD_a0.25_lr0.01_mom0.95_s260925_a6000"),
        ("16M SNR gate α ½, β 0.9 (vs PD)", "PD", "soaudit_snrgate_large_20260928/PDgate_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
    ]
    out["interventions"] = []
    for label, base, arm, *custom in interventions:
        v, ref = final(arm), final(custom[0] if custom else refs[base])
        if v is not None and ref is not None:
            out["interventions"].append({"label": f"{base}: {label}", "final": round(v, 4), "reference": round(ref, 4), "change": round(v - ref, 4)})
    def smoothed(arm, half=3):
        rows = {}
        for f in (root / arm / "scientific" / "steps").glob("step*.json"):
            r = json.loads(f.read_text())
            if "train_nll" in r:
                rows[r["step"]] = r["train_nll"]
        return {t: sum(rows[x] for x in range(t - half, t + half + 1) if x in rows) /
                max(1, sum(1 for x in range(t - half, t + half + 1) if x in rows)) for t in rows}
    phase_pairs = {
        "S∘PD 1×, seed 260925": ("soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada",
                                 "soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada"),
        "S∘PD 1×, seed 260926": ("soaudit_mom16mseed_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260926_ada",
                                 "soaudit_seed16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.9_s260926_ada"),
        "PD 1×": ("soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s",
                  "soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
        "Muon 1×": ("soaudit_mom16m_20260928/M_b16M_lr0.02_mom0.8_s260925_l40s",
                    "soaudit_batch16m_20260927/M_b16M_lr0.02_mom0.9_s260925_l40s"),
        "Muon 1×, clip 0.1": ("soaudit_clip16m_20260928/Mclip0.1_b16M_lr0.02_mom0.8_s260925_l40s",
                              "soaudit_clip16m_20260928/Mclip0.1_b16M_lr0.02_mom0.9_s260925_l40s"),
        "S∘PD 2×": ("soaudit_horizon16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.8_s260925_ada",
                    "soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada"),
        "S∘PD 2×, warmup 0.8→0.9": ("soaudit_momwarm16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada",
                                    "soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada"),
        "S∘PD 1×, warmup 0.8→0.9": ("soaudit_momwarm16m_20260928/SPD_a0.5_b16M_lr0.028_momwarm0.8to0.9_s260925_ada",
                                    "soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada"),
        "S∘PD 4M": ("soaudit_mom4mb08_20260928/SPD_a0.5_b4M_lr0.02_mom0.8_s260925_l40s",
                    "soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s"),
        "PD 4M": ("soaudit_mom4mb08_20260928/PD_a0.5_b4M_lr0.02_mom0.8_s260925_ada",
                  "soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada"),
        "PD α¼ 1×": ("soaudit_dose16m_20260928/PD_a0.25_b16M_lr0.028_mom0.8_s260925_l40s",
                     "soaudit_dose16m_20260928/PD_a0.25_b16M_lr0.028_mom0.9_s260925_l40s"),
        "S∘PD 2×, warmup 0.8→0.9, seed 260926": ("soaudit_warmrep_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260926_ada",
                                                 "soaudit_warmrep_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260926_ada"),
        "PD 2×, warmup 0.8→0.9": ("soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_l40s",
                                  "soaudit_warmrep_20260928/PD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_l40s"),
        "PD-top 1×": ("soaudit_pdtop_20260928/PDtop_a0.5_b16M_lr0.028_mom0.8_s260925_l40s",
                      "soaudit_pdtop_20260928/PDtop_a0.5_b16M_lr0.028_mom0.9_s260925_l40s"),
        "PD 4M, warmup 0.8→0.9 in 120": ("soaudit_warm4m_20260928/PD_a0.5_b4M_lr0.02_momwarm0.8to0.9in120_s260925_ada",
                                         "soaudit_mom4m4_20260927/PD_a0.5_b4M_lr0.02_mom0.9_s260925_ada"),
        "S∘PD 4M, warmup 0.8→0.9 in 120": ("soaudit_warm4m_20260928/SPD_a0.5_b4M_lr0.02_momwarm0.8to0.9in120_s260925_l40s",
                                           "soaudit_strength4m_20260927/SPD_a0.5_b4M_lr0.02_mom0.9_s260925_l40s"),
        "Muon 4M, β 0.81": ("soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.81_s260925_l40s",
                            "soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s"),
    }
    out["phase"] = {}
    for label, (arm, ref) in phase_pairs.items():
        a, b = smoothed(arm), smoothed(ref)
        steps = sorted(set(a) & set(b))
        if len(steps) > 5:
            out["phase"][label] = [[t, round(a[t] - b[t], 4)] for t in steps]
    for tag in ("pd9", "pd46"):
        r = json_if(f"gnrate/preflight_{tag}.json")
        if r:
            out["preflight"][tag] = {"mean_eig": r["log"]["mean_eig_G_A"],
                                     "rows": {k: {"quality": v["slope"] ** 2 / (2 * v["curvature"]), "cstar": v["c_star"],
                                                  "true1": v["true_change"]["1.0"]} for k, v in r["scores"].items()},
                                     "cos_AB": r["cos"]["gnpd_A_bf16_k64_p0.5_d1e-3|gnpd_B_bf16_k64_p0.5_d1e-3"]["mean"],
                                     "cos_kron": r["cos"]["kron|gnpd_A_bf16_k64_p0.5_d1e-3"]["mean"]}
    out["momentum"] = momentum_study()
    out["secant"] = secant_study()
    out["coupling"] = coupling_study()
    out["staged"] = staged_study()
    out["coordination"] = coordination_study()
    return out


def staged_study():
    """Layer-staged PD branches in newton_train.py's harness vs their control: per-step train loss, validation, pre-clip
    gradient norm and the top GN eigenvalue (2026-09-29)."""
    runs = {}
    for name in ("control_9_sharp", "staged_9_sharp", "control_46", "staged_lr1_46", "staged_lr1.5_46", "switch_50", "switch_30"):
        folder = HERE / "staged_branch" / name / "steps"
        if not folder.exists():
            continue
        rows = []
        for f in sorted(folder.glob("step*.json")):
            r = json.loads(f.read_text())
            rows.append({"step": r["step"], "train": round(r["train_nll"], 5), "val": round(r["validation_nll"], 5) if "validation_nll" in r else None,
                         "gnorm": round(r.get("grad_norm_before_clip", float("nan")), 4), "sharp": round(r["gn_top"], 2) if "gn_top" in r else None})
        runs[name] = rows
    return runs


def coordination_study():
    """Layer-staged coordination variants in newton_train.py's harness against their same-harness controls (band_coordination
    and the robustness pairs, 2026-09-29 08:45 CDT on): step leads (plot_step_leads.leads, +-3 smoothing at 16M, +-5 at
    4M), top GN eigenvalue, mean cosine with plain PD's direction, final validation."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("psl", HERE / "plot_step_leads.py")
    psl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(psl)
    groups = {
        "16M": ("staged_branch/control_9_sharp", 3, [
            ("full correction", "staged_branch/staged_9_sharp"), ("full, 256 curvature sequences", "band/staged_big_256"),
            ("full, half strength", "band/staged_half"),
            ("above-mean input band", "band/band_top"), ("bulk input band", "band/band_bulk"),
            ("input-mean direction", "band/band_mean"), ("8 largest input directions", "band/band_top8")]),
        "16M 2× horizon": ("band/control_2x", 3, [("above-mean input band", "band/band_top_2x")]),
        "4M": ("band/control_4m", 5, [("above-mean input band", "band/band_top_4m"), ("8 largest input directions", "band/band_top8_4m"),
                                      ("above-mean band, half dose", "band/band_top_4m_half"),
                                      ("8 largest directions, half dose", "band/band_top8_4m_half"),
                                      ("sublayer sweep, half dose", "band/band_sublayer_4m_half")]),
        "16M, second initialization": ("band/control_s2", 3, [("above-mean input band", "band/band_top_s2")]),
        "16M β 0.8": ("band/control_b08", 3, [("above-mean input band", "band/band_top_b08"),
                                              ("8 largest input directions", "band/band_top8_b08")]),
        "16M S∘PD β 0.8": ("band/spd_control_b08", 3, [("above-mean input band", "band/spd_top_b08"),
                                                    ("oracle per-matrix reversal", "band/spd_reversal_b08"),
                                                    ("Jacobi (plain-step corrections)", "band/spd_jacobi_b08"),
                                                    ("reversed sweep (last block first)", "band/spd_reverse_b08"),
                                                    ("sublayer sweep (q/k/v, o, up, down)", "band/spd_sublayer_b08"),
                                                    ("per-matrix sweep", "band/spd_matrix_b08"),
                                                    ("sublayer sweep, all directions", "band/spd_sublayer_full_b08"),
                                                    ("layer sweep, all directions", "band/spd_full_b08"),
                                                    ("8 curvature sequences per rank", "band/spd_top_c8_b08")]),
        "16M S∘PD β 0.8, 2× horizon": ("band/spd_control_2x_b08", 3, [("above-mean input band", "band/spd_top_2x_b08"),
                                                                   ("sublayer sweep", "band/spd_sublayer_2x_b08")]),
        "1M PD β 0.95": ("band/control_1m", 10, [("above-mean input band, 16M dose", "band/band_top_1m_half")]),
        "16M S∘PD β 0.8 from step 0": ("band/spd_control_from0", 3, [("sublayer sweep, above-mean band", "band/spd_sublayer_from0"),
                                                                ("sublayer sweep, 8 curvature seq.", "band/spd_sublayer_c8_from0"),
                                                                ("sublayer sweep, 8 seq. + EMA of C", "band/spd_sublayer_c8ema_from0")]),
        "16M S∘PD β 0.8 from step 0, control with the EMA root": ("band/spd_control_c8ema_from0", 3,
                                                                  [("sublayer sweep, 8 seq. + EMA of C", "band/spd_sublayer_c8ema_from0")]),
        "16M S∘PD β 0.8 from step 0, 2× horizon": ("band/spd_control_2x_from0", 3,
                                                   [("sublayer sweep, above-mean band", "band/spd_sublayer_2x_from0"),
                                                    ("sublayer sweep, 8 seq. + EMA of C", "band/spd_sublayer_c8ema_2x_from0")]),
    }

    def rows(path):
        out = {}
        for f in sorted((HERE / path / "steps").glob("step*.json")):
            r = json.loads(f.read_text())
            out[r["step"]] = r
        return out
    out = {}
    for group, (control, half, arms) in groups.items():
        if not (HERE / control / "steps").exists():
            continue
        c = rows(control)
        entry = {"control": {"sharp": [[t, round(c[t]["gn_top"], 1)] for t in sorted(c) if "gn_top" in c[t]],
                             "final": c[max(c)].get("validation_nll")}, "arms": {}}
        for label, path in arms:
            if not (HERE / path / "steps").exists():
                continue
            r = rows(path)
            if len(set(r) & set(c)) < 2 * half + 2:     # just started: nothing to compare yet
                continue
            d = psl.leads(HERE / path / "steps", HERE / control / "steps", half)
            last = max(set(r) & set(c))
            entry["arms"][label] = {
                "lead": [[int(t), None if v != v else round(float(v), 2)] for t, v in d],
                "sharp": [[t, round(r[t]["gn_top"], 1)] for t in sorted(r) if "gn_top" in r[t]],
                "cos": [[t, round(r[t]["staged_cos_plain_mean"], 3)] for t in sorted(r) if "staged_cos_plain_mean" in r[t]],
                "final_gap": (round(r[last]["validation_nll"] - c[last]["validation_nll"], 4)
                              if "validation_nll" in r[last] and "validation_nll" in c[last] else None),
                "last_step": last,
                "done": (HERE / path / "done.json").exists() and (HERE / control / "done.json").exists()}
        out[group] = entry
    return out


def coupling_study():
    """GN curvature of each optimizer's actual step split into pairs of hidden matrices (step_coupling_probe.py, 2026-09-29)."""
    import numpy as np
    rows = {}
    for path in sorted((HERE / "step_coupling").glob("*.json")):
        r = json.loads(path.read_text())
        Q = np.array(r["Q"])
        ev = np.linalg.eigvalsh(Q)[::-1]
        names = r["names"]
        order = sorted(range(len(names)), key=lambda i: (names[i].split(".")[0], r["kinds"].index(names[i].split(".")[1])))
        rows[path.stem] = {"coherence": round(r["coherence"], 3), "parts": {k: round(v / r["total_q"], 4) for k, v in r["parts"].items()},
                           "top_mode": round(float(ev[0] / ev.sum()), 4), "pr": round(float(ev.sum() ** 2 / (ev ** 2).sum()), 3),
                           "corr": [[round(r["corr"][i][j], 3) for j in order] for i in order],
                           "labels": [names[i].replace("block0", "L") for i in order], "c_star_gn": round(r["c_star_gn"], 4)}
    return rows


def secant_study():
    """Collective secant step multiplier c* = -<mu_N, D> / <mu_N+1 - mu_N, D> per input-eigenvalue bin (secant_probe.py),
    its 1M time-lapse, and the two-sided input x output maps (twosided_secant_probe.py) (2026-09-29)."""
    def one(path):
        r = json.loads(path.read_text())
        rows = [b for b in r["bins"] if b["count"] >= 20 and b["c_star"] is not None and b["step_share"] > 2e-3]
        return {"u": [round((b["u_lo"] * b["u_hi"]) ** 0.5, 6) for b in rows], "c": [round(b["c_star"], 4) for b in rows],
                "energy": [round(b["step_share"], 4) for b in rows], "gain": [round(b["slope_share"], 4) for b in rows],
                "all": round(r["all"]["c_star"], 4)}
    out = {"states": {}, "timelapse": {}, "twosided": {}}
    for path in sorted((HERE / "secant").glob("*.json")):
        out["states"][path.stem] = one(path)
    for path in sorted((HERE / "secant_timelapse").glob("*.json")):
        out["timelapse"][path.stem] = one(path)
    for path in sorted((HERE / "twosided_secant").glob("*.json")):
        r = json.loads(path.read_text())
        g = r["grids"]["all"]
        out["twosided"][path.stem] = {"c": g["c_star"], "energy": [[round(x, 4) for x in row] for row in g["energy_share"]],
                                      "gain": [[round(x, 4) for x in row] for row in g["slope_share"]], "labels": r["labels"],
                                      "all": round(g["c_star_total"], 4)}
    return out


def momentum_study():
    """Momentum-buffer noise fraction (clip-corrected) and one-step autocorrelation of the mean gradient per input
    eigendirection bin, from direction_drift_probe.py (2026-09-28 21:2x CDT)."""
    from momentum_noise_clipfix import factor
    total = {1048576: 1468, 4194304: 368, 16777216: 92}
    name = {1048576: "1M", 4194304: "4M", 16777216: "16M"}
    rows = []
    for path in sorted((HERE / "direction_drift").glob("*.json")):
        r = json.loads(path.read_text())
        config = json.loads((Path(r["arm"]) / "config.json").read_text())
        f = factor(r["arm"], r["step"], r["beta"], config["grad_clip"])
        bins = [b for b in r["bins"] if b["count"] >= 20]
        tail = [b for b in bins if b["u_hi"] <= 1]
        energy = lambda bs: sum(b["momentum_noise_fraction"] * 1.0 for b in bs) / max(len(bs), 1)  # noqa: E731
        rows.append({"tag": path.stem, "opt": "Muon" if path.stem.startswith("M") else "PD", "batch": name[r["batch_train"]],
                     "beta": r["beta"], "step": r["step"], "frac": round(r["step"] / total[r["batch_train"]], 4),
                     "noise_all": round(r["all"]["momentum_noise_fraction"] * f, 4), "noise_tail_mean": round(energy(tail) * f, 4),
                     "clip_factor": round(f, 4),
                     "u": [round((b["u_lo"] * b["u_hi"]) ** 0.5, 6) for b in bins],
                     "autocorr": [round(b["autocorr_one_step"], 4) for b in bins],
                     "cos_m": [round(b["cos_momentum_mean"], 4) for b in bins],
                     "noise_bins": [round(b["momentum_noise_fraction"] * f, 4) for b in bins]})
    return rows


def json_if(name):
    path = HERE / name
    return json.loads(path.read_text()) if path.exists() else {}


def largebatch_study():
    """The large-batch gap (2026-09-30, MUON_CASE 11:40 → 18:29 CDT): tokens-to-loss across batch sizes, one step vs
    sequential steps on one 16M batch, the rate test of frozen-model inner loops, the LR and aux-LR brackets, and where the
    top of the whole model's GN spectrum lives."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("batch_efficiency", HERE / "batch_efficiency.py")
    be = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(be)
    curves = {}
    for label, (path, kind, batch, _, style) in be.ARMS.items():
        c = be.curve(path, kind, batch)
        if c is None:
            continue
        curves[label] = {"tokens": [round(float(t) / 1e9, 4) for t in c["tokens"]], "loss": [round(float(v), 4) for v in c["loss"]],
                         "batch": batch, "dash": style != "-"}
    ratios = json_if("figures_momentum/batch_efficiency.json").get("token_ratio", {})
    path = {}
    for tag in ("path9", "path46", "path46b48", "path83"):
        r = json_if(f"{tag}/probe.json")
        if r:
            path[tag] = {"base": r["base_validation"],
                         "lines": {k: {"c": v["c"], "norm": v["norm"], "true": v["true"], "model": v["model_val"]}
                                   for k, v in r["lines"].items()}}

    def steps_of(folder):
        rows = {}
        for f in (HERE / folder / "steps").glob("step*.json"):
            r = json.loads(f.read_text())
            rows[r["step"]] = r
        return rows
    rate = {}
    for name, folder, k in (("A: one 16M step", "path46/diag2/A", 1), ("B: four true 4M steps", "path46/diag2/B", 4),
                            ("C: four steps on the GN model", "path46/diag2/C", 4),
                            ("diff: small-batch gradient difference", "path46/diag2/L_diff", 4),
                            ("gnsmall: 32-sequence GN", "path46/diag2/L_gnsmall", 4),
                            ("full Hessian (finite differences)", "path46/diag2/L_hessfd", 4)):
        rows = steps_of(folder)
        if not rows:
            continue
        first = min(rows)
        val, lam = [], []
        for s_, r in sorted(rows.items()):
            batch = 47 + (s_ - first) // k
            if (s_ - first) % k == k - 1 and r.get("validation_nll") is not None:
                val.append([batch, round(r["validation_nll"], 4)])
            if (s_ - first) % k == 0 and r.get("gn_top") is not None:
                lam.append([batch, round(r["gn_top"], 1)])
        rate[name] = {"val": val, "lam": lam}

    def final(folder):
        rows = steps_of(folder)
        if not rows or not (HERE / folder / "done.json").exists():
            return None
        return round(rows[max(rows)].get("validation_nll", float("nan")), 4)
    brackets = {"lr": {"control": [[m, final(f)] for m, f in ((0.7, "band/spd_control_c8ema_lr07_from0"), (1.0, "band/spd_control_c8ema_from0"),
                                                               (1.4, "band/spd_control_c8ema_lr14_from0"))],
                       "coordinated": [[m, final(f)] for m, f in ((1.0, "band/spd_sublayer_c8ema_from0"), (1.4, "band/spd_cheap_lr14_from0"))]},
                "aux": {"control": [[m, final(f)] for m, f in ((1, "band/spd_control_c8ema_from0"), (2, "band/spd_control_c8ema_auxlr2_from0"),
                                                               (4, "band/spd_control_c8ema_auxlr4_from0"), (8, "band/spd_control_c8ema_auxlr8_from0"))],
                        "coordinated": [[m, final(f)] for m, f in ((1, "band/spd_sublayer_c8ema_from0"), (4, "band/spd_cheap_auxlr4_from0"))]}}
    spectrum = {}
    for item, e in json_if("figures_momentum/full_spectrum.json").items():
        tag = item.split("soaudit_")[1].split("/")[1].split("_s2609")[0] + " @" + item.rsplit(":", 1)[1]
        w = e["all"]["group_share_weighted"]
        spectrum[tag] = {"all": [round(v, 1) for v in e["all"]["top_ritz"][:3]], "body": [round(v, 1) for v in e["body"]["top_ritz"][:3]],
                         "head": round(w.get("head", 0), 3), "embed": round(w.get("embed", 0) + w.get("position", 0), 3)}
    return {"curves": curves, "ratios": ratios, "path": path, "rate": rate, "brackets": brackets, "spectrum": spectrum}


def main(out_path):
    data = {"runs": list(RUNS), "kinds": list(KINDS),
            "labels": {"M": "Muon", "PD": "PD α¼", "S": "SOAP-Muon", "SPD": "S∘PD"},
            "marg": marginals(), "gap": gap(), "batch": batch_study(), "gn": gn_states(),
            "lags": json_if("persistence_lags/lags.json"), "split": json_if("split_vs_diag.json"),
            "blockgn": block_gn(), "transport": transport_study(), "gap2gn": gap_to_gn(),
            "eos": {k: v for path in sorted(HERE.glob("eos_invariant*.json")) for k, v in json.loads(path.read_text()).items()},
            "midpoint": json_if("midpoint_loss.json"), "steprule": step_rule_study(), "edge": edge_study(),
            "eoslin": {k: v for path in sorted(HERE.glob("eos_linearized_[ab].json")) for k, v in json.loads(path.read_text()).items()},
            "largebatch": largebatch_study()}
    template = (HERE / "atlas_template.html").read_text()
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    Path(out_path).write_text(template.replace("__ATLAS_DATA__", payload))
    print(f"wrote {out_path}: {len(payload) / 1e6:.2f} MB of data")


if __name__ == "__main__":
    main(sys.argv[1])

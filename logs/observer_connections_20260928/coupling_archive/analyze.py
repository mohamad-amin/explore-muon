"""Small-matrix, CPU-only reanalysis of archived exact predictive-GN Grams."""
import os
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
ARCHIVE = ROOT / "logs/muon_spectra/second_order_audit_20260926"
NAMES = [f"block{layer:02d}.{kind}" for layer in range(1, 9) for kind in ("q", "k", "v", "o", "up", "down")]
KINDS = np.array([name.split(".")[1] for name in NAMES])
LAYERS = np.array([int(name[5:7]) for name in NAMES])
UPPER = np.triu(np.ones((48, 48), dtype=bool), 1)
OFF = ~np.eye(48, dtype=bool)
sources = {}


def read(path):
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    return json.loads(raw)


def eigen(matrix):
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2)
    order = np.arange(len(values) - 1, -1, -1)
    values, vectors = values[order], vectors[:, order]
    for i in range(len(values)):
        if vectors[:, i].sum() < 0:
            vectors[:, i] *= -1
    return values, vectors


def matrix_summary(matrix, slope, actual_coefficients):
    values, vectors = eigen(matrix)
    top = vectors[:, 0]
    residual = matrix - values[0] * np.outer(top, top)
    off = matrix[UPPER]
    return {
        "eigenvalues": values.tolist(),
        "top1_trace_share": float(values[0] / values.sum()),
        "top4_trace_share": float(values[:4].sum() / values.sum()),
        "effective_rank_trace_squared_over_frobenius_squared": float(values.sum() ** 2 / (values @ values)),
        "top1_actual_selected_step_curvature_share": float(values[0] * (top @ actual_coefficients) ** 2 / (actual_coefficients @ matrix @ actual_coefficients)),
        "top1_actual_selected_step_slope_share": float((top @ actual_coefficients) * (top @ slope) / (actual_coefficients @ slope)),
        "top1_loading_participation": float(1 / (top ** 4).sum()),
        "top1_same_sign_loading_count": int((top > 0).sum()),
        "top1_loadings": dict(zip(NAMES, top.tolist())),
        "top1_slope_alignment_squared": float((top @ slope) ** 2 / (slope @ slope)),
        "positive_offdiagonal_fraction": float((off > 0).mean()),
        "negative_offdiagonal_abs_share": float(np.abs(off[off < 0]).sum() / np.abs(off).sum()),
        "offdiagonal_quantiles": dict(zip(("min", "q10", "median", "q90", "max"), np.quantile(off, [0, .1, .5, .9, 1]).tolist())),
        "top1_residual_offdiagonal_frobenius_ratio": float(np.linalg.norm(residual[UPPER]) / np.linalg.norm(off)),
        "top1_residual_positive_offdiagonal_fraction": float((residual[UPPER] > 0).mean()),
        "top1_residual_offdiagonal_sum_over_original": float(residual[UPPER].sum() / off.sum()),
    }


def analyze(a, q):
    q = np.asarray(q, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    assert q.shape == (48, 48) and a.shape == (48,)
    symmetry = float(np.max(np.abs(q - q.T)))
    assert symmetry < 1e-8 and np.isfinite(q).all() and np.isfinite(a).all()
    assert np.diag(q).min() > 0
    scale = np.sqrt(np.diag(q))
    corr = q / np.outer(scale, scale)
    val, _ = eigen(q)
    assert val[-1] >= -1e-6 * val[0]
    raw = matrix_summary(q, a, np.ones(48))
    normalized = matrix_summary(corr, a / scale, scale)
    pair_stats = {}
    for label, mask in {
        "same_kind_cross_layer": (KINDS[:, None] == KINDS[None, :]) & (LAYERS[:, None] != LAYERS[None, :]),
        "same_layer_cross_kind": (KINDS[:, None] != KINDS[None, :]) & (LAYERS[:, None] == LAYERS[None, :]),
        "different_layer_and_kind": (KINDS[:, None] != KINDS[None, :]) & (LAYERS[:, None] != LAYERS[None, :]),
    }.items():
        mask &= UPPER
        pair_stats[label] = {
            "pairs": int(mask.sum()),
            "mean_correlation": float(corr[mask].mean()),
            "positive_fraction": float((q[mask] > 0).mean()),
            "signed_cross_curvature_share": float(q[mask].sum() / q[UPPER].sum()),
        }
    kind_stats = {}
    layer_stats = {}
    for values, target in ((KINDS, kind_stats), (LAYERS, layer_stats)):
        for item in np.unique(values):
            mask = values == item
            target[str(item)] = {
                "diagonal_share": float(np.diag(q)[mask].sum() / np.trace(q)),
                "signed_row_sum_share": float(q[mask].sum() / q.sum()),
                "slope_share": float(a[mask].sum() / a.sum()),
                "raw_top1_loading_squared_share": float(sum(raw["top1_loadings"][n] ** 2 for n, m in zip(NAMES, mask) if m)),
                "normalized_top1_loading_squared_share": float(sum(normalized["top1_loadings"][n] ** 2 for n, m in zip(NAMES, mask) if m)),
            }
    normalized_slope = -a / scale
    normalized_slope /= np.linalg.norm(normalized_slope)
    slope_rank_one = np.outer(normalized_slope, normalized_slope)
    fit_scale = float((corr[UPPER] * slope_rank_one[UPPER]).sum() / (slope_rank_one[UPPER] ** 2).sum())
    return {
        "symmetry_max_abs": symmetry,
        "a": a.tolist(),
        "Q": q.tolist(),
        "correlation": corr.tolist(),
        "coherence_sum_over_diagonal": float(q.sum() / np.trace(q)),
        "descent_piece_count": int((a < 0).sum()),
        "total_slope": float(a.sum()),
        "total_quadratic": float(q.sum()),
        "raw": raw,
        "normalized": normalized,
        "pairs": pair_stats,
        "kinds": kind_stats,
        "layers": layer_stats,
        "normalized_slope_rank_one_offdiagonal_fit": {
            "scale": fit_scale,
            "relative_residual_frobenius": float(np.linalg.norm((corr - fit_scale * slope_rank_one)[UPPER]) / np.linalg.norm(corr[UPPER])),
            "note": "Descriptive same-archive fit; no causal attribution or held-out intervention.",
        },
    }


def compare_halves(aa, qa, ab, qb):
    result = {}
    for label in ("raw", "normalized"):
        q0, q1 = np.asarray(qa), np.asarray(qb)
        if label == "normalized":
            q0 = q0 / np.sqrt(np.outer(np.diag(q0), np.diag(q0)))
            q1 = q1 / np.sqrt(np.outer(np.diag(q1), np.diag(q1)))
        e0, v0 = eigen(q0)
        e1, v1 = eigen(q1)
        result[label] = {
            "top1_loading_cosine_abs": float(abs(v0[:, 0] @ v1[:, 0])),
            "top4_mean_squared_principal_cosine": float(np.linalg.norm(v0[:, :4].T @ v1[:, :4]) ** 2 / 4),
            "offdiagonal_pearson_correlation": float(np.corrcoef(q0[UPPER], q1[UPPER])[0, 1]),
            "top1_A_rayleigh_B_over_B_optimum": float(v0[:, 0] @ q1 @ v0[:, 0] / e1[0]),
            "top1_B_rayleigh_A_over_A_optimum": float(v1[:, 0] @ q0 @ v1[:, 0] / e0[0]),
        }
    b0 = np.asarray(aa) / np.sqrt(np.diag(qa))
    b1 = np.asarray(ab) / np.sqrt(np.diag(qb))
    transfers = {}
    for label, cf, uf, bf, cs, bs in (
        ("A_to_B", q0, v0[:, 0], b0, q1, b1),
        ("B_to_A", q1, v1[:, 0], b1, q0, b0),
    ):
        outer = np.outer(uf, uf)
        alpha = float((cf[UPPER] * outer[UPPER]).sum() / (outer[UPPER] ** 2).sum())
        residual_diagonal = np.diag(cf) - alpha * uf ** 2
        model = np.diag(residual_diagonal) + alpha * outer
        c_fit = float(uf @ bf)
        c_score = float(uf @ bs)
        transfers[label] = {
            "fit_alpha": alpha,
            "fit_residual_diagonal_min": float(residual_diagonal.min()),
            "fit_residual_diagonal_max": float(residual_diagonal.max()),
            "fit_offdiagonal_relative_residual": float(np.linalg.norm((cf - model)[UPPER]) / np.linalg.norm(cf[UPPER])),
            "score_offdiagonal_relative_residual": float(np.linalg.norm((cs - model)[UPPER]) / np.linalg.norm(cs[UPPER])),
            "score_slope_relative_residual_fixed_orientation_and_amplitude": float(np.linalg.norm(bs - c_fit * uf) / np.linalg.norm(bs)),
            "score_slope_relative_residual_fixed_orientation_only": float(np.linalg.norm(bs - c_score * uf) / np.linalg.norm(bs)),
            "note": "Correlation matrices and slopes use each half's own measured diagonal for normalization. Only the fit-half mode and alpha specify the correlation prediction; the score-half slope amplitude is fit only for the explicitly orientation-only summary.",
        }
    result["rank_one_plus_residual_diagonal_transfer"] = transfers
    return result


out = {"names": NAMES, "entries": {}, "split_half": {}}
for path in sorted((ARCHIVE / "gn").glob("*_gram.json")):
    data = read(path)
    assert data["names"] == NAMES
    for label, entry in data.items():
        if label == "names":
            continue
        key = f"{path.stem.removesuffix('_gram')}:{label}"
        out["entries"][key] = {
            "source": str(path.relative_to(ROOT)), "source_key": label,
            "score_bank": "all128", **analyze(entry["a"], entry["Q"]),
        }
for path in sorted((ARCHIVE / "gn").glob("*.json")):
    if path.name.endswith("_gram.json"):
        continue
    data = read(path)
    for label, halves in data["inputs"]["momentum"]["gram"].items():
        key = f"{path.stem}:momentum:{label}"
        a = (np.asarray(halves["A"][0]) + np.asarray(halves["B"][0])) / 2
        q = (np.asarray(halves["A"][1]) + np.asarray(halves["B"][1])) / 2
        out["entries"][key] = {
            "source": str(path.relative_to(ROOT)), "source_key": f"inputs.momentum.gram.{label}",
            "score_bank": "mean_of_two64", **analyze(a, q),
            "halves": {bank: analyze(*values) for bank, values in halves.items()},
        }
        out["split_half"][key] = compare_halves(*halves["A"], *halves["B"])

out["prior_interventions"] = {}
for name in ("M1M_g4M_a", "M1M_g4M_gs", "M1M_g4M_gsrev", "M1M_g4M_gslay", "gsmap_M1Mg", "gsmap_M1Mm"):
    data = read(ARCHIVE / "blockgn" / f"{name}.json")
    out["prior_interventions"][name] = {label: entry["cross_fit"] for label, entry in data["families"].items()}
data = read(ARCHIVE / "gn3/PD_a0.25_lr0.01_s260925_ada_step000500.json")["inputs"]["g4M"]
out["prior_interventions"]["global_deflation_PD1M_g4M"] = {
    "best": data["best"],
    "ritz_values": data["deflation_ritz"],
    "scores": {k: v for k, v in data["scores"].items() if k in ("muon", "pd_a0.5", data["best"]["gnall"]) or "top" in k},
}
for name in ("gram_saved.py", "one_step_gn.py", "one_step_blockgn.py", "job_gram_saved.sbatch", "OBSERVATIONS.md"):
    path = ARCHIVE / name
    raw = path.read_bytes()
    sources[str(path.relative_to(ROOT))] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
out["provenance"] = sources
out["script_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(HERE / "analysis.json").write_text(json.dumps(out, indent=2) + "\n")
rows = []
for name, e in out["entries"].items():
    row = {"entry": name, "coherence": e["coherence_sum_over_diagonal"],
           "positive_pair_fraction": e["raw"]["positive_offdiagonal_fraction"],
           "negative_pair_abs_share": e["raw"]["negative_offdiagonal_abs_share"]}
    for mode in ("raw", "normalized"):
        for metric in ("top1_trace_share", "top4_trace_share", "top1_loading_participation",
                       "top1_same_sign_loading_count", "top1_slope_alignment_squared",
                       "top1_residual_offdiagonal_frobenius_ratio"):
            row[f"{mode}_{metric}"] = e[mode][metric]
    rows.append(row)
with (HERE / "summary.csv").open("w") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for r in rows:
    print(r["entry"], "coh", round(r["coherence"], 2), "positive", round(r["positive_pair_fraction"], 3),
          "top1 raw/norm", round(r["raw_top1_trace_share"], 3), round(r["normalized_top1_trace_share"], 3),
          "PR", round(r["normalized_top1_loading_participation"], 1))
print("analyzed", len(rows), "complete 48-by-48 Grams plus 16 halves")

"""CPU-only observer audit of existing logs and scalar linear stability.

No torch/NumPy, checkpoint loads, network, training, or modifications to sources.
Run from project root; writes only its own audit.json.
"""
import cmath
import hashlib
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
COHORTS = ["soaudit_batch16m_20260927", "soaudit_prefilter16m_20260928",
           "soaudit_mom16m_20260928", "soaudit_mom16mseed_20260928"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def roots(beta, a, two_tap=False):
    c = 1 + beta - (a / 2 if two_tap else a)
    d = beta + (a / 2 if two_tap else 0)
    delta = cmath.sqrt(c*c - 4*d)
    return [(c + delta)/2, (c - delta)/2]


def stable(beta, a, two_tap=False):
    return max(map(abs, roots(beta, a, two_tap))) < 1


runs = []
for cohort in COHORTS:
    for arm in sorted((ROOT / "logs/muon_spectra" / cohort).glob("*/config.json")):
        scientific = arm.parent / "scientific"
        metadata_path = scientific / "metadata.json"
        if not metadata_path.exists():
            continue
        metadata = read(metadata_path)
        config = metadata["config"]
        rows = sorted((read(p) for p in (scientific / "steps").glob("step*.json")), key=lambda x: x["step"])
        positive = [r for r in rows if r["step"] > 0]
        if not positive:
            continue
        gradients = [r["gradient_norm_before_clip"] for r in positive]
        windows = {}
        for lo, hi in ((1, 9), (10, 46), (47, 83), (84, 92)):
            selected = [r for r in positive if lo <= r["step"] <= hi]
            windows[f"{lo}-{hi}"] = ({
                "n": len(selected),
                "gradient_norm_median": median(r["gradient_norm_before_clip"] for r in selected),
                "clipped_fraction": sum(r["gradient_clipped"] for r in selected)/len(selected),
                "train_nll_mean": sum(r["train_nll"] for r in selected)/len(selected),
            } if selected else None)
        runs.append({
            "cohort": cohort, "arm": arm.parent.name, "path": str(scientific.relative_to(ROOT)),
            "config": config, "device": metadata["device_name"],
            "initial_model_sha256": metadata["initial_model_sha256"],
            "data_fingerprints": {k: hashlib.sha256(json.dumps(metadata[k], sort_keys=True).encode()).hexdigest()
                                  for k in ("train_manifest", "validation_manifest")},
            "source_sha256": metadata["source_sha256"],
            "input_file_sha256": {str(p.relative_to(ROOT)): sha(p) for p in (arm, metadata_path)},
            "status": read(scientific / "status.json") if (scientific / "status.json").exists() else None,
            "last_step": positive[-1]["step"], "n_steps": len(positive),
            "clipped_fraction": sum(r["gradient_clipped"] for r in positive)/len(positive),
            "gradient_norm_median": median(gradients), "gradient_norm_max": max(gradients),
            "gradient_norm_min": min(gradients), "windows": windows,
            "validation": {str(r["step"]): r["validation_nll"] for r in rows if "validation_nll" in r},
            "lr": {str(r["step"]): r["lr"] for r in positive},
            "matrix_norms_at_50": {k: v["frobenius_norm"] for r in positive if r["step"] == 50
                                   for k,v in r.get("matrices", {}).items()},
        })

stability = []
for beta in (0, .6, .7, .8, .9, .95):
    for two_tap in (False, True):
        bound = 2 * (1-beta if two_tap else 1+beta)
        assert stable(beta, bound*.99, two_tap)
        assert not stable(beta, bound*1.01, two_tap)
        stability.append({"beta": beta, "two_tap": two_tap, "a_stable_upper_bound": bound,
                          "boundary_period_steps": 2*math.pi/math.acos(beta) if two_tap else 2,
                          "root_radius_99pct_bound": max(map(abs, roots(beta,bound*.99,two_tap))),
                          "root_radius_101pct_bound": max(map(abs, roots(beta,bound*1.01,two_tap)))})

pairs = []
for treatment in runs:
    c = treatment["config"]
    if treatment["cohort"] == COHORTS[0]:
        continue
    candidates = [r for r in runs if r["cohort"] == COHORTS[0]
                  and r["config"]["seed"] == c["seed"]
                  and r["config"]["learning_rate"] == c["learning_rate"]
                  and r["config"]["data_norm_alpha"] == c["data_norm_alpha"]
                  and r["config"]["soap_precondition"] == c["soap_precondition"]
                  and r["config"]["data_norm_out_beta"] == c["data_norm_out_beta"]
                  and r["device"] == treatment["device"]]
    for reference in candidates:
        r = reference["config"]
        keys = sorted(set(r)|set(c))
        pairs.append({"reference": reference["path"], "treatment": treatment["path"],
                      "config_changes": {k: [r.get(k,"<missing>"),c.get(k,"<missing>")] for k in keys if r.get(k,"<missing>") != c.get(k,"<missing>")},
                      "same_init": reference["initial_model_sha256"] == treatment["initial_model_sha256"],
                      "same_data": reference["data_fingerprints"] == treatment["data_fingerprints"],
                      "same_lr_at_shared_steps": all(reference["lr"].get(k) == v for k,v in treatment["lr"].items()),
                      "frozen_source_changes": [k for k in sorted(set(reference["source_sha256"]) | set(treatment["source_sha256"]))
                                                if reference["source_sha256"].get(k) != treatment["source_sha256"].get(k)]})

output = {"runs": runs, "pairs": pairs, "scalar_quadratic_stability": stability}
(OUT / "audit.json").write_text(json.dumps(output, indent=2) + "\n")
for r in runs:
    c=r["config"]
    if r["cohort"]==COHORTS[0] and (c["learning_rate"] not in (.02,.028) or c["data_norm_out_beta"] > 0):
        continue
    vals=r["validation"]
    print(r["arm"], "step",r["last_step"], "val",round(vals[str(max(map(int,vals)))],6),
          "clip",round(r["clipped_fraction"],3), "grad_med",round(r["gradient_norm_median"],3),
          "grad_max",round(r["gradient_norm_max"],3))
for p in pairs:
    print("PAIR",p["treatment"], "changes", p["config_changes"], "init/data/lr",p["same_init"],p["same_data"],p["same_lr_at_shared_steps"])
print("Scalar stability checked against exact roots for", len(stability), "settings.")

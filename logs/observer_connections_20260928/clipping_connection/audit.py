"""Reconstruct scalar momentum-history kernels from saved norms only.

No training, torch, numpy, model loads, cluster submissions or outside writes.
"""
from pathlib import Path
import csv
import datetime
import hashlib
import json
import math
import statistics

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
COHORTS = [
    "soaudit_batch16m_20260927", "soaudit_seed16m_20260928",
    "soaudit_mom16m_20260928", "soaudit_mom16mseed_20260928",
    "soaudit_prefilter16m_20260928", "soaudit_clip16m_20260928",
    "soaudit_b16mlong_20260928", "soaudit_horizon16m_20260928",
    "soaudit_mom4m2_20260927", "soaudit_mom4m3_20260927",
    "soaudit_mom4m4_20260927", "soaudit_mom4m_20260927",
]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def moments(weights, steps, token_times, step, token_time):
    total = sum(weights)
    probs = [w / total for w in weights]
    return {
        "mass": total,
        "mean_age_steps": sum(p * (step - s) for p, s in zip(probs, steps)),
        "mean_age_tokens": sum(p * (token_time - x) for p, x in zip(probs, token_times)),
        "latest_fraction": probs[-1],
        "coefficient_ess": 1 / sum(p * p for p in probs),
        "period2_gain_abs": abs(sum(p * (-1) ** (step-s) for p, s in zip(probs, steps))),
    }


def kernel_update(weights, beta, coefficient, previous, two_tap):
    result = [beta * x for x in weights] + [0.0]
    if not two_tap or previous is None:
        result[-1] = coefficient
    else:
        result[-1] = coefficient / 2
        result[-2] += previous / 2
    return result


def correlation(xs, ys):
    if len(xs) < 2:
        return None
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x-mx) * (y-my) for x, y in zip(xs, ys))
    den = math.sqrt(sum((x-mx)**2 for x in xs) * sum((y-my)**2 for y in ys))
    return num / den if den else None


def summarize_window(rows):
    if not rows:
        return None
    fields = ["clip_coefficient", "gradient_norm_before_clip", "raw_mean_age_steps",
              "raw_mean_age_tokens", "unit_mean_age_steps", "unclipped_raw_mean_age_steps",
              "unclipped_unit_mean_age_steps", "raw_effective_history_decay",
              "raw_latest_fraction", "unit_latest_fraction", "aux_first_raw_mean_age_steps",
              "aux_second_raw_mean_age_steps", "train_nll"]
    result = {key: {"mean": statistics.mean(r[key] for r in rows),
                    "min": min(r[key] for r in rows), "max": max(r[key] for r in rows)}
              for key in fields}
    cs = [r["clip_coefficient"] for r in rows]
    result.update(n=len(rows), clipped=sum(r["gradient_clipped"] for r in rows),
                  coefficient_cv=statistics.pstdev(cs)/statistics.mean(cs),
                  coefficient_lag1_correlation=correlation(cs[:-1], cs[1:]))
    return result


def validate_formula():
    for beta in (.6, .8, .9):
        for two_tap in (False, True):
            scalar = 0.0
            weights, prev_c, prev_g = [], None, None
            for t in range(1, 41):
                c = 0.2 + 0.1 * (t % 5)
                g = math.sin(t * .37) + math.cos(t * .91)
                scalar = beta*scalar + (c*g if not two_tap or prev_g is None else (c*g+prev_c*prev_g)/2)
                weights = kernel_update(weights, beta, c, prev_c, two_tap)
                direct = sum(w*(math.sin(s*.37)+math.cos(s*.91)) for s,w in enumerate(weights,1))
                assert abs(direct - scalar) < 1e-12
                prev_c, prev_g = c, g
        weights = [beta**k for k in range(40)]
        direct_age = sum(k*w for k,w in enumerate(weights)) / sum(weights)
        analytic = beta/(1-beta)-40*beta**40/(1-beta**40)
        assert abs(direct_age-analytic) < 1e-12


validate_formula()
runs, all_rows, inputs = [], [], {}
for cohort in COHORTS:
    for arm in sorted((ROOT / "logs/muon_spectra" / cohort).glob("*/config.json")):
        scientific = arm.parent / "scientific"
        metadata_path = scientific / "metadata.json"
        if not metadata_path.exists():
            runs.append({"cohort": cohort, "arm": arm.parent.name,
                         "availability": "no scientific metadata at audit time"})
            continue
        metadata = read(metadata_path)
        config = metadata["config"]
        # These pre-momentum transforms need vector data, and are not included.
        assert not config.get("mean_whitening", False)
        assert not config.get("pmuon", False)
        nesterov = config.get("muon_nesterov", False)
        beta = config["muon_momentum"]
        two_tap = config.get("muon_prefilter", "none") == "two_tap"
        paths = sorted((scientific / "steps").glob("step*.json"))
        entries = [read(p) for p in paths]
        original = [r for r in entries if r["step"] > 0]
        if not original:
            runs.append({"cohort": cohort, "arm": arm.parent.name,
                         "availability": "no positive steps at audit time"})
            continue
        assert [r["step"] for r in original] == list(range(1, len(original)+1))
        for p in [arm, metadata_path, *paths]:
            inputs[str(p.relative_to(ROOT))] = sha(p)
        rows = []
        wr, wn, wu, wnu, wa1, wa2 = [], [], [], [], [], []
        steps, token_times = [], []
        previous_c, previous_norm, previous_mass = None, None, None
        for r in original:
            step, norm = r["step"], r["gradient_norm_before_clip"]
            c = min(1.0, config["grad_clip"] / (norm + 1e-6))
            assert r["gradient_clipped"] == (norm > config["grad_clip"])
            steps.append(step)
            token_times.append(r["tokens"])
            wr = kernel_update(wr,beta,c,previous_c,two_tap)
            wn = kernel_update(wn,beta,c*norm,None if previous_c is None else previous_c*previous_norm,two_tap)
            wu = kernel_update(wu,beta,1.0,None if previous_c is None else 1.0,two_tap)
            wnu = kernel_update(wnu,beta,norm,previous_norm,two_tap)
            # Auxiliary Adam sees clipped gradients, without the body two-tap filter.
            # Factors (1-beta) and bias correction cancel in normalized age summaries.
            wa1 = kernel_update(wa1,config["betas"][0],c,previous_c,False)
            wa2 = kernel_update(wa2,config["betas"][1],c*c,previous_c,False)
            row = {"cohort": cohort, "arm": arm.parent.name, "step": step,
                   "tokens": r["tokens"], "batch_tokens": r["batch_tokens"],
                   "lr": r["lr"], "aux_lr": r.get("aux_lr"),
                   "train_nll": r["train_nll"], "gradient_norm_before_clip": norm,
                   "gradient_clipped": r["gradient_clipped"], "clip_coefficient": c,
                   "validation_nll": r.get("validation_nll"),
                   "raw_effective_history_decay": beta * previous_mass/sum(wr) if previous_mass is not None else 0.0}
            for name, weights in (("raw",wr),("unit",wn),("unclipped_raw",wu),("unclipped_unit",wnu),
                                  ("aux_first_raw",wa1),("aux_second_raw",wa2)):
                row.update({name+"_"+k:v for k,v in moments(weights,steps,token_times,step,r["tokens"]).items()})
            # Effective history decay only has the stated recurrence meaning for plain momentum.
            if two_tap:
                row["raw_effective_history_decay_is_plain_ema"] = False
            rows.append(row)
            previous_c, previous_norm, previous_mass = c, norm, sum(wr)
        cooldown_steps = [r["step"] for r in rows if r["step"] > config["warmup_steps"] and r["lr"] < config["learning_rate"] * (1-1e-12)]
        first_cooldown = min(cooldown_steps) if cooldown_steps else None
        nfull = math.ceil(config["total_tokens"]/config["batch_tokens"])
        windows = {"all": summarize_window(rows),
                   "middle_10_to_half": summarize_window([r for r in rows if 10 <= r["step"] <= nfull//2]),
                   "late_constant_lr": summarize_window([r for r in rows if nfull//2 < r["step"] and r["lr"] == config["learning_rate"]]),
                   "cooldown": summarize_window([r for r in rows if first_cooldown is not None and r["step"] >= first_cooldown])}
        status = read(scientific/"status.json") if (scientific/"status.json").exists() else None
        if status:
            inputs[str((scientific/"status.json").relative_to(ROOT))] = sha(scientific/"status.json")
        run = {"cohort":cohort,"arm":arm.parent.name,"availability":"logged",
               "status_record":status,"n_logged_steps":len(rows),"expected_steps":nfull,
               "complete_verified":bool(status and status["status"]=="complete" and len(rows)==nfull and rows[-1]["tokens"]==config["total_tokens"]),
               "config":config,"nesterov_buffer_only":nesterov,
               "source_sha256":metadata["source_sha256"],
               "initial_model_sha256":metadata["initial_model_sha256"],
               "data_fingerprints":{k:hashlib.sha256(json.dumps(metadata[k],sort_keys=True).encode()).hexdigest() for k in ("train_manifest","validation_manifest")},
               "device":metadata["device_name"],"first_cooldown_step":first_cooldown,
               "validation":{str(r["step"]):r["validation_nll"] for r in entries if "validation_nll" in r},
               "windows":windows,
               "snapshots":{str(r["step"]):r for r in rows if r["step"] in (9,46,50,83,92,166,184) or r["step"]==len(rows)}}
        runs.append(run)
        all_rows.extend(rows)

pairs=[]
for i,a in enumerate(runs):
    if a["availability"] != "logged":
        continue
    for b in runs[i+1:]:
        if b["availability"] != "logged": continue
        ac,bc=a["config"],b["config"]
        fields=["seed","learning_rate","batch_tokens","total_tokens","data_norm_alpha","data_norm_out_beta","soap_precondition"]
        if any(ac[k]!=bc[k] for k in fields): continue
        if ac["muon_momentum"]==bc["muon_momentum"] and ac["grad_clip"]==bc["grad_clip"] and ac.get("muon_prefilter","none")==bc.get("muon_prefilter","none"): continue
        ra=[r for r in all_rows if r["cohort"]==a["cohort"] and r["arm"]==a["arm"]]
        rb=[r for r in all_rows if r["cohort"]==b["cohort"] and r["arm"]==b["arm"]]
        shared=min(len(ra),len(rb))
        diffs=[{ "step":x["step"],"train_nll_b_minus_a":y["train_nll"]-x["train_nll"],
                 "raw_age_b_minus_a":y["raw_mean_age_steps"]-x["raw_mean_age_steps"],
                 "unit_age_b_minus_a":y["unit_mean_age_steps"]-x["unit_mean_age_steps"]}
               for x,y in zip(ra[:shared],rb[:shared])]
        pairs.append({"a":a["arm"],"b":b["arm"],
                      "same_init":a["initial_model_sha256"]==b["initial_model_sha256"],
                      "same_data":a["data_fingerprints"]==b["data_fingerprints"],
                      "same_device":a["device"]==b["device"],
                      "same_lrs":all(x["lr"]==y["lr"] and x["aux_lr"]==y["aux_lr"] for x,y in zip(ra[:shared],rb[:shared])),
                      "config_diff":{k:[ac.get(k,"<missing>"),bc.get(k,"<missing>")] for k in sorted(set(ac)|set(bc)) if ac.get(k,"<missing>")!=bc.get(k,"<missing>")},
                      "frozen_source_changes":[k for k in set(a["source_sha256"])|set(b["source_sha256"]) if a["source_sha256"].get(k)!=b["source_sha256"].get(k)],
                      "validation_b_minus_a":{k:b["validation"][k]-v for k,v in a["validation"].items() if k in b["validation"]},
                      "late_constant_lr_train_nll_b_minus_a":statistics.mean(d["train_nll_b_minus_a"] for d,x,y in zip(diffs,ra[:shared],rb[:shared]) if x["step"]>a["expected_steps"]//2 and x["lr"]==ac["learning_rate"] and y["lr"]==bc["learning_rate"]) if shared>a["expected_steps"]//2 else None,
                      "step_differences":diffs})

output={"audit_time":datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "coefficient_formula":"min(1, grad_clip/(gradient_norm_before_clip+1e-6)); Python double approximation to logged FP32 norm",
        "qualifications":["direct kernel expansion agrees with scalar recurrence for six beta/filter settings to1e-12", "finite EMA age matches analytic formula to1e-12", "contiguous logged steps", "clip bool agrees with saved norm/threshold"],
        "runs":runs,"pairs":pairs,"inputs_sha256":inputs}
(OUT/"audit.json").write_text(json.dumps(output,indent=2)+"\n")
with (OUT/"step_weights.csv").open("w",newline="") as f:
    fields=sorted(set().union(*(r.keys() for r in all_rows)))
    writer=csv.DictWriter(f,fieldnames=fields)
    writer.writeheader(); writer.writerows(all_rows)
for run in runs:
    if run["availability"] != "logged":
        print(run["arm"],run["availability"]); continue
    w=run["windows"]["late_constant_lr"]
    print(run["arm"],"complete",run["complete_verified"],"steps",run["n_logged_steps"],
          "val",list(run["validation"].values())[-1],
          "late raw/unit/unclippedunit age",None if w is None else [round(w[k]["mean"],3) for k in ("raw_mean_age_steps","unit_mean_age_steps","unclipped_unit_mean_age_steps")],
          "late effective beta",None if w is None else round(w["raw_effective_history_decay"]["mean"],4))
print("Saved",len(runs),"run records",len(all_rows),"step rows",len(inputs),"input hashes")

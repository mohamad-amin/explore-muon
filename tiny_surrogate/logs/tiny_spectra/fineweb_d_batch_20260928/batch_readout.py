"""Prospective fixed-recipe batch comparison; keeps D's failed ordering closed.

Only the planner and readout are new. Training uses the ordering cohort's
unchanged frozen numerical sources and single-arm execution controller.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import statistics
import time

SEEDS = (20261001, 20261002)
BATCHES = (65536, 262144, 1048576)
RECIPES = (("muon", .01), ("muon", .02), ("spd", .02))
THRESHOLDS = tuple(i / 10 for i in range(40, 96))
TOTAL = 96242176


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def driver(cohort):
    spec = importlib.util.spec_from_file_location("frozen_ordering_driver", cohort / "ordering_driver.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evaluation_steps(batch, every):
    n = math.ceil(TOTAL / batch)
    return sorted({0, 1, n, *range(every, n + 1, every)})


def crossing(evaluations, threshold):
    for i, right in enumerate(evaluations):
        if right["validation_nll"] <= threshold:
            if i == 0:
                return dict(status="initial", left=0, right=0, step=0.)
            left = evaluations[i - 1]
            fraction = ((left["validation_nll"] - threshold) /
                        (left["validation_nll"] - right["validation_nll"]))
            return dict(status="reached", left=left["step"], right=right["step"],
                        step=left["step"] + fraction * (right["step"] - left["step"]))
    return dict(status="unreached", left=None, right=None, step=None)


def ratio(a, b):
    if a["status"] != "reached" or b["status"] != "reached" or min(a["left"], b["left"]) <= 0:
        return dict(value=None, lower=None, upper=None)
    return dict(value=a["step"] / b["step"], lower=a["left"] / b["right"],
                upper=a["right"] / b["left"])


def analyse(rows):
    expected = {(b, s, m, lr) for b in BATCHES for s in SEEDS for m, lr in RECIPES}
    keyed = {(r["batch_tokens"], r["seed"], r["method"], r["lr"]): r for r in rows}
    if set(keyed) != expected or len(rows) != len(expected):
        raise ValueError("Exactly all 18 fixed trajectories are required")
    crossings, common, excluded = {}, [], []
    for threshold in THRESHOLDS:
        reasons = []
        for key, row in keyed.items():
            value = crossing(row["evaluations"], threshold) if row["status"] == "complete" else dict(
                status="unstable", left=None, right=None, step=None)
            crossings[key, threshold] = value
            warmup = math.ceil(.034 * math.ceil(TOTAL / key[0]))
            if value["status"] != "reached" or value["left"] < warmup:
                reasons.append(dict(batch=key[0], seed=key[1], method=key[2], lr=key[3],
                                    status=value["status"], left=value["left"], warmup_end=warmup))
        if reasons:
            excluded.append(dict(threshold=threshold, reasons=reasons))
        else:
            common.append(threshold)
    coverage = (len(common) >= 6 and round(common[-1] - common[0], 8) >= .5
                and all(round(b - a, 8) == .1 for a, b in zip(common, common[1:])))
    all_ratios, summaries, endpoints = [], [], []
    for seed in SEEDS:
        for lr in (.01, .02):
            per_batch = {}
            for batch in BATCHES:
                selected = []
                a, b = keyed[batch, seed, "muon", lr], keyed[batch, seed, "spd", .02]
                endpoints.append(dict(seed=seed, muon_lr=lr, batch_tokens=batch,
                    muon_nll=a["full_development_nll"], spd_nll=b["full_development_nll"],
                    spd_minus_muon=(b["full_development_nll"] - a["full_development_nll"]
                                   if a["status"] == b["status"] == "complete" else None)))
                for threshold in THRESHOLDS:
                    ca = crossings[(batch, seed, "muon", lr), threshold]
                    cb = crossings[(batch, seed, "spd", .02), threshold]
                    r = ratio(ca, cb)
                    all_ratios.append(dict(seed=seed, muon_lr=lr, batch_tokens=batch,
                        threshold=threshold, common_range=threshold in common,
                        muon_crossing=ca, spd_crossing=cb, **r))
                    if threshold in common:
                        selected.append(r)
                per_batch[str(batch)] = {k: statistics.mean(math.log(r[k]) for r in selected)
                                         if selected else None for k in ("value", "lower", "upper")}
            lo, mid, hi = (per_batch[str(b)] for b in BATCHES)
            defined = bool(common)
            growth = {"value": hi["value"] - lo["value"],
                      "lower": hi["lower"] - lo["upper"],
                      "upper": hi["upper"] - lo["lower"]} if defined else None
            gates = dict(common_coverage=coverage,
                growth_resolved=bool(defined and growth["lower"] > 0),
                large_batch_advantage_resolved=bool(defined and hi["lower"] > 0),
                midpoint_monotone=bool(defined and lo["value"] <= mid["value"] <= hi["value"]))
            summaries.append(dict(seed=seed, muon_lr=lr, mean_log_ratios=per_batch,
                                  large_minus_small=growth, gates=gates, passed=all(gates.values())))
    return dict(common_thresholds=common, excluded_thresholds=excluded, coverage_passed=coverage,
        all_ratios=all_ratios, comparisons=summaries, endpoint_gaps=endpoints,
        fixed_recipe_batch_component_passed=all(s["passed"] for s in summaries),
        uncertainty="Observed crossing-resolution bounds, not statistical confidence intervals",
        five_method_ordering_still_failed=True, full_goal_qualified=False, sealed_test_scored=False)


def verify(cohort):
    module = driver(cohort)
    plan = module.verify(cohort, "initial")
    preflight = read(cohort / "PREFLIGHT.json")
    for name, key in (("batch_readout.py", "readout_sha256"), ("PEER_DISCUSSION.json", "peer_sha256"),
                      ("job_initial.sh", "job_script_sha256")):
        if sha(cohort / name) != preflight[key]:
            raise ValueError("Batch analysis/decision/execution identity changed: " + name)
    reference = Path(plan["reference_cohort"])
    if sha(reference / "report_final/results.json") != plan["reference_report_sha256"]:
        raise ValueError("Reused midpoint readout changed")
    for reused in plan["reused"]:
        root = reference / "runs" / reused["run_id"]
        for name, digest in reused["artifact_sha256"].items():
            if sha(root / name) != digest:
                raise ValueError("Reused midpoint artifact changed")
    return plan


def create(cohort, reference, study):
    if not cohort.is_relative_to(study) or (cohort / "PLAN.json").exists():
        raise ValueError("Use one new isolated cohort; never overwrite a frozen plan")
    old = driver(reference)
    old.verify(reference, "initial")
    old.verify(reference, "edges")
    previous = read(reference / "report_final/results.json")
    chosen = [r for r in previous["rows"] if (r["method"], r["lr"]) in RECIPES]
    if len(chosen) != 6 or any(r["status"] != "complete" for r in chosen):
        raise ValueError("All six fixed midpoint trajectories must already be complete")
    for directory in ("configs", "runs", "console", "workers"):
        (cohort / directory).mkdir()
    shutil.copytree(reference / "frozen", cohort / "frozen")
    shutil.copy2(reference / "source_manifest.json", cohort / "source_manifest.json")
    shutil.copy2(reference / "ordering_driver.py", cohort / "ordering_driver.py")
    shutil.copy2(__file__, cohort / "batch_readout.py")
    tasks, configurations = [], []
    for batch in (65536, 1048576):
        for row in chosen:
            cfg = read(reference / "configs" / (row["run_id"] + ".json"))
            cfg.update(batch_tokens=batch, root_refresh=10 if batch == 65536 else 2,
                       eval_every=8 if batch == 65536 else 1)
            cfg["run_id"] = f"{cfg['method']}_Dbatch_b{batch}_lr{cfg['lr']:g}_s{cfg['seed']}"
            path = cohort / "configs" / (cfg["run_id"] + ".json")
            write(path, cfg)
            configurations.append(cfg)
            tasks.append([dict(config=str(path), out=str(cohort / "runs" / cfg["run_id"]))])
    random.Random(20260928).shuffle(tasks)
    base = read(reference / "PLAN.json")
    plan = dict(stage="separate_fixed_recipe_batch_component", new_runs=12, reused_runs=6,
        batches=list(BATCHES), seeds=list(SEEDS), recipes=RECIPES, total_tokens=TOTAL,
        reference_cohort=str(reference), reference_report_sha256=sha(reference / "report_final/results.json"),
        original_candidate_plan_sha256=base["original_candidate_plan_sha256"],
        training_manifest_sha256=base["training_manifest_sha256"],
        full_development_targets=base["full_development_targets"],
        full_development_windows=base["full_development_windows"],
        reused=[dict(run_id=r["run_id"], artifact_sha256=r["artifact_sha256"]) for r in chosen],
        evaluation_steps={str(b): evaluation_steps(b, 1 if b == 1048576 else 8) for b in BATCHES},
        thresholds=list(THRESHOLDS), common_range="All 18 first crossings reached, crossing left >= own warmup end",
        minimum_consecutive_thresholds=6, minimum_threshold_span=.5,
        statistic="Equal-threshold mean log(steps_Muon/steps_SPD); each seed and control separately",
        gates=["common coverage", "positive growth lower resolution bound", "positive high-batch benefit lower bound",
               "midpoint interpolated mean between endpoints"],
        selection="No LR selection, no envelope, no checkpoint/threshold selection",
        gpu=base["gpu"], array_throttle=None, maximum_concurrent_gpus=None, maximum_gpu_hours=None,
        forecast_gpu_hours=[5, 12], scheduler_job_walltime_seconds=7200,
        worker_process_seconds=7140, per_arm_timeout_seconds=7080, automatic_requeue=False,
        automatic_retries=False, sealed_test_scoring=False, ordering_failure_preserved=True,
        decision_protocol=str(study / "PROTOCOL.md"), protocol_sha256_at_freeze=sha(study / "PROTOCOL.md"),
        no_automatic_followups=True)
    write(cohort / "PLAN.json", plan)
    write(cohort / "tasks_initial.json", tasks)
    old.make_script(cohort, study, "initial", 7140)
    write(cohort / "PREFLIGHT.json", dict(plan_sha256=sha(cohort / "PLAN.json"),
        source_manifest_sha256=sha(cohort / "source_manifest.json"), controller_sha256=sha(cohort / "ordering_driver.py"),
        tasks_initial_sha256=sha(cohort / "tasks_initial.json"), readout_sha256=sha(cohort / "batch_readout.py"),
        peer_sha256=sha(cohort / "PEER_DISCUSSION.json"), job_script_sha256=sha(cohort / "job_initial.sh"),
        initial_config_sha256={p.name: sha(p) for p in sorted((cohort / "configs").glob("*.json"))}))
    verify(cohort)
    return plan


def tensors(obj):
    import torch
    if torch.is_tensor(obj):
        yield obj
    elif isinstance(obj, dict):
        for x in obj.values():
            yield from tensors(x)
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            yield from tensors(x)


def report(cohort):
    import numpy as np
    import torch
    from torch.torch_version import TorchVersion
    plan = verify(cohort)
    out = cohort / "report"
    if out.exists():
        raise FileExistsError("Preserve existing batch readout")
    tasks = read(cohort / "tasks_initial.json")
    configs = [read(g[0]["config"]) for g in tasks]
    expected = {(b, s, m, lr) for b in (65536, 1048576) for s in SEEDS for m, lr in RECIPES}
    if len(configs) != 12 or {(c["batch_tokens"], c["seed"], c["method"], c["lr"]) for c in configs} != expected:
        raise ValueError("Incomplete declared batch family")
    if (cohort / "HALT.json").exists():
        raise ValueError("Preserved operational failure prevents readout")
    if any(not (cohort / "runs" / c["run_id"] / "ARM_EXECUTION.json").exists() for c in configs):
        raise ValueError("Wait for all 12 declared new arms before reading any batch scores")
    reference = Path(plan["reference_cohort"])
    previous = read(reference / "report_final/results.json")
    ids = {r["run_id"] for r in plan["reused"]}
    rows = [dict(r, batch_tokens=262144, reused=True) for r in previous["rows"] if r["run_id"] in ids]
    identity_keys = ("initial_parameter_sha256", "train_window_sha256", "validation_bank_sha256",
                     "validation_bank_lengths_sha256", "data_manifest_sha256")
    paired = {}
    for r in rows:
        meta = read(reference / "runs" / r["run_id"] / "metadata.json")
        identity = tuple(meta[k] for k in identity_keys)
        if r["seed"] in paired and paired[r["seed"]] != identity:
            raise ValueError("Reused midpoint pairing changed")
        paired[r["seed"]] = identity
    manifest_path = Path(configs[0]["data_path"])
    if sha(manifest_path) != plan["training_manifest_sha256"]:
        raise ValueError("Data manifest changed")
    manifest, population = read(manifest_path), {}
    for key, name in (("starts", "dev_starts"), ("target_counts", "dev_lengths"), ("document_indices", "dev_document_indices")):
        spec = manifest[name]
        path = (manifest_path.parent / spec["path"]).resolve()
        if not path.is_relative_to(manifest_path.parent) or sha(path) != spec["sha256"] or path.stat().st_size != spec["bytes"]:
            raise ValueError("Development population identity changed")
        population[key] = torch.from_numpy(np.fromfile(path, dtype="<i8").copy())
    if (len(population["starts"]) != plan["full_development_windows"] or
            int(population["target_counts"].sum()) != plan["full_development_targets"]):
        raise ValueError("Development population denominator changed")
    doc_spec = manifest["splits"]["val"]["documents"]
    doc_path = manifest_path.parent / doc_spec["path"]
    if sha(doc_path) != doc_spec["sha256"]:
        raise ValueError("Development documents changed")
    docs = read(doc_path)
    weights = {}
    for key, retention in (("input_cov_weight", .998), ("input_weight", .99)):
        w = torch.tensor(0., dtype=torch.float32)
        for _ in range(46994):
            w.mul_(retention).add_(1 - retention)
        weights[key] = float(w)
    for cfg in configs:
        root = cohort / "runs" / cfg["run_id"]
        execution, meta = read(root / "ARM_EXECUTION.json"), read(root / "metadata.json")
        steps = math.ceil(TOTAL / cfg["batch_tokens"])
        if (execution["config_sha256"] != sha(cohort / "configs" / (cfg["run_id"] + ".json")) or
                execution["source_manifest_sha256"] != sha(cohort / "source_manifest.json") or
                read(root / "config.json") != cfg or
                driver(cohort).classify_exit(root, execution["returncode"]) != execution["status"]):
            raise ValueError("Execution/config identity mismatch")
        if (tuple(meta[k] for k in identity_keys) != paired[cfg["seed"]] or
                meta["total_steps"] != steps or meta["n_parameters"] != 4861056 or
                meta["model_config"]["stats_clock"] != "microforward" or
                meta["covariance_gram_precision"] != "FP32; autocast disabled" or
                "A4000" not in meta["hardware"]["name"] or
                not 0 < meta["hardware"]["total_memory_bytes"] < 45 * 1024**3 or
                Path(meta["source_file"]).resolve() != cohort / "frozen/research/tiny_spectra/train.py"):
            raise ValueError("Hardware, pairing, architecture, source or clock mismatch")
        row = dict(run_id=cfg["run_id"], method=cfg["method"], lr=cfg["lr"], seed=cfg["seed"],
                   batch_tokens=cfg["batch_tokens"], status=execution["status"], reused=False,
                   full_development_nll=None, evaluations=[], hardware=meta["hardware"])
        names = ["ARM_EXECUTION.json", "config.json", "metadata.json", "metrics.jsonl"]
        if execution["status"] == "numerical_instability":
            row["failure"] = read(root / "failure.json")
            names.append("failure.json")
        elif execution["status"] == "complete":
            summary = read(root / "summary.json")
            metrics = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
            if (summary["steps"] != steps or summary["tokens"] != TOTAL or summary["status"] != "complete" or
                    [r["step"] for r in metrics] != list(range(steps + 1)) or
                    [r["tokens"] for r in metrics] != [min(i * cfg["batch_tokens"], TOTAL) for i in range(steps + 1)] or
                    [r["step"] for r in metrics if "validation_nll" in r] != plan["evaluation_steps"][str(cfg["batch_tokens"])]):
                raise ValueError("Incomplete horizon or evaluation cadence")
            for record in metrics:
                for key in ("train_nll", "validation_nll", "gradient_norm_before_clip", "train_probe_nll", "step_seconds"):
                    if key in record and not math.isfinite(record[key]):
                        raise ValueError("Nonfinite completed trajectory")
            saved = torch.load(root / "final_validation.pt", map_location="cpu", weights_only=True)
            if any(saved[k].dtype != torch.long or not torch.equal(saved[k], v) for k, v in population.items()):
                raise ValueError("Saved evaluation targets differ")
            losses = saved["sequence_nll"]
            if losses.shape != population["target_counts"].shape or not torch.isfinite(losses).all():
                raise ValueError("Invalid saved per-window losses")
            mean = float((losses * population["target_counts"]).sum() / population["target_counts"].sum())
            if any(abs(mean - value) > 1e-12 for value in
                   (summary["full_validation_nll"], summary["final_validation_nll"], metrics[-1]["validation_nll"])):
                raise ValueError("Per-window endpoint reconstruction failed")
            with torch.serialization.safe_globals([TorchVersion]):
                snapshot = torch.load(root / "final.pt", map_location="cpu", weights_only=True)
            if snapshot["config"] != cfg or snapshot["metadata"] != meta or not all(torch.isfinite(t).all() for t in tensors(snapshot)):
                raise ValueError("Checkpoint identity/finiteness failure")
            if cfg["method"] == "spd":
                states = snapshot["optimizer"]["model_statistics"]
                roots = snapshot["optimizer"]["external"]["data_norm"]["roots"]
                last_root = 1 + (steps - 1) // cfg["root_refresh"] * cfg["root_refresh"]
                final_forwards = math.ceil((TOTAL - (steps - 1) * cfg["batch_tokens"]) / 2048)
                if len(states) != 48 or len(roots) != 48 or any(r[1] != last_root for r in roots.values()):
                    raise ValueError("Root inventory/clock mismatch")
                if any(int(s["_total_forwards"]) != 46994 or int(s["_step_forwards"]) != final_forwards or
                       any(abs(float(s[k]) - v) > 1e-7 for k, v in weights.items()) for s in states.values()):
                    raise ValueError("EMA/microforward clock mismatch")
            del snapshot
            sums = torch.zeros(len(docs), dtype=torch.float64).scatter_add_(0, population["document_indices"], losses.double() * population["target_counts"])
            counts = torch.zeros(len(docs), dtype=torch.long).scatter_add_(0, population["document_indices"], population["target_counts"])
            groups = []
            for group in range(4):
                indices = torch.tensor([i for i, d in enumerate(docs) if int(d["identity"][:2], 16) % 4 == group])
                groups.append(dict(group=group, targets=int(counts[indices].sum()), nll=float(sums[indices].sum() / counts[indices].sum())))
            row.update(full_development_nll=mean, evaluations=[r for r in metrics if "validation_nll" in r],
                development_groups=groups, training_seconds=summary["training_seconds"], total_seconds=summary["total_seconds"])
            names += ["summary.json", "final.pt", "final_validation.pt"]
        else:
            raise ValueError("Operational failure invalidates readout")
        row["artifact_sha256"] = {name: sha(root / name) for name in names}
        rows.append(row)
    result = dict(analyse(rows), rows=rows, plan_sha256=sha(cohort / "PLAN.json"),
                  analysis_sha256=sha(cohort / "batch_readout.py"), completed_unix=time.time())
    out.mkdir()
    write(out / "results.json", result)
    with (out / "all_crossings.csv").open("x", newline="") as stream:
        flat = [dict((k, v) for k, v in r.items() if not isinstance(v, dict)) |
                {prefix + "_" + k: v for prefix in ("muon", "spd") for k, v in r[prefix + "_crossing"].items()}
                for r in result["all_ratios"]]
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    lines = ["# Fixed-recipe batch comparison", "", f"Development batch-component gate passed: **{result['fixed_recipe_batch_component_passed']}**.",
        "D's five-method ordering remains failed. No test panel was scored. This result does not complete the surrogate goal.", "",
        f"Common post-warmup threshold range: {result['common_thresholds']}. Bounds reflect crossing resolution, not statistical confidence.", "",
        "| Seed | Muon LR | Growth in mean log step ratio | Resolution bounds | All gates pass |", "|---|---:|---:|---|---|"]
    for r in result["comparisons"]:
        g = r["large_minus_small"]
        lines.append(f"| {r['seed']} | {r['muon_lr']} | {g['value'] if g else None} | {[g['lower'], g['upper']] if g else None} | {r['passed']} |")
    lines += ["", "Both fixed Muon controls, both seeds, all thresholds, censored targets and all endpoint document groups are retained in results.json. No envelope or favorable checkpoint/threshold selection is used. No automatic follow-up sweep is authorized."]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return {k: v for k, v in result.items() if k not in ("rows", "all_ratios", "excluded_thresholds")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("create", "verify", "submit", "report"))
    parser.add_argument("cohort", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--study", type=Path)
    args = parser.parse_args()
    cohort = args.cohort.resolve()
    if args.operation == "create":
        plan = create(cohort, args.reference.resolve(), args.study.resolve())
        print(json.dumps(dict(created=str(cohort), new_runs=plan["new_runs"])))
    elif args.operation == "verify":
        print(json.dumps(dict(verified=verify(cohort)["stage"])))
    elif args.operation == "submit":
        verify(cohort)
        driver(cohort).submit(cohort, "initial")
    else:
        print(json.dumps(report(cohort), indent=2))


if __name__ == "__main__":
    main()

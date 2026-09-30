"""Create/read Candidate-D numerical qualification; never submit jobs or score tests."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from torch.torch_version import TorchVersion

from .cohort import create, write_json
from .fineweb import KIND, load_training_corpus
from .scale_diagnostic_report import CORE


METHODS = ("adamw", "muon", "pd", "ts", "spd")
REQUIRED_SOURCES = set(CORE) | {"research/tiny_spectra/fineweb.py", "research/tiny_spectra/stories.py"}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(8 << 20), b""):
            h.update(part)
    return h.hexdigest()


def verify_sources(directory):
    manifest = read(directory / "source_manifest.json")
    if not REQUIRED_SOURCES <= set(manifest):
        raise ValueError("Qualification is missing a required execution source")
    for relative, expected in manifest.items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or digest(directory / "frozen" / path) != expected:
            raise ValueError("Frozen qualification source changed")
    return manifest


def finite_numbers(value):
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_numbers(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(map(finite_numbers, value))
    return True


def fp32_ema_weight(decay, forwards):
    weight = torch.zeros((), dtype=torch.float32)
    for _ in range(forwards):
        weight.mul_(decay).add_(1-decay)
    return float(weight)


def prepared_data(data_root, plan_path, preparation_log):
    """Require successful data execution before constructing any GPU cohort."""
    data_root, plan_path, preparation_log = map(Path, (data_root, plan_path, preparation_log))
    if (data_root / "PREPARATION_FAILED.json").exists() or (preparation_log / "EXECUTION_FAILURE.json").exists():
        raise ValueError("Preserved preparation failure requires a separate decision")
    needed = (data_root / "PREPARATION_COMPLETE.json", data_root / "MANIFESTS_PREPARED.json",
              preparation_log / "EXECUTION_COMPLETE.json", preparation_log / "PROCESS_EXIT.json")
    if not all(p.is_file() for p in needed):
        raise ValueError("Data preparation must finish successfully before GPU qualification")
    completion, manifests, execution, exit_record = map(read, needed)
    if (completion != manifests or execution["result"] != completion or exit_record["exit_code"] != 0
            or completion["no_model_inference"] is not True
            or execution["model_inference"] is not False or execution["gpu_requested"] is not False):
        raise ValueError("Preparation receipts disagree or execution did not succeed")
    sources = verify_sources(preparation_log)
    started = read(preparation_log / "EXECUTION_STARTED.json")
    preparation_started = read(data_root / "PREPARATION_STARTED.json")
    config = read(data_root / "preparation_config.json")
    if (started["source_manifest_sha256"] != digest(preparation_log / "source_manifest.json")
            or started["config_sha256"] != digest(data_root / "preparation_config.json")
            or preparation_started["config_sha256"] != started["config_sha256"]
            or preparation_started["source_sha256"] != sources["research/tiny_spectra/fineweb.py"]
            or config != read(data_root / "PREPARATION_CONFIG_COPY.json")
            or config["plan"]["sha256"] != digest(plan_path)):
        raise ValueError("Frozen preparation configuration/source identity differs")
    plan = read(plan_path)
    manifest_path = data_root / "training_manifest.json"
    if digest(manifest_path) != completion["training_manifest_sha256"]:
        raise ValueError("Completed training manifest changed")
    manifest = read(manifest_path)
    if (manifest["corpus_kind"] != KIND or manifest["role"] != "training_and_development_only"
            or manifest["vocab_size"] != 12576 or manifest["seq_len"] != 512
            or manifest["fresh_target_capacity"] < plan["minimum_fresh_target_capacity"]
            or manifest["preparation"]["plan_sha256"] != digest(plan_path)
            or manifest["roles_frozen_sha256"] != digest(data_root / "ROLES_FROZEN.json")
            or manifest["tokenizer_fit"] != read(data_root / "tokenizer_fit.json")):
        raise ValueError("Prepared data do not meet the fixed Candidate-D contract")
    for field, name in (("source_verification_sha256", "SOURCE_VERIFIED.json"),
                        ("decoder_qualification_sha256", "DECODER_QUALIFIED.json")):
        if manifest["preparation"][field] != digest(data_root / name):
            raise ValueError("Data qualification receipt changed: " + name)
    # Hash only the test manifest; no test tokens, model or scores are loaded.
    if digest(data_root / "sealed_test/manifest.json") != completion["test_manifest_sha256"]:
        raise ValueError("Prepared test manifest changed")
    corpus = load_training_corpus(manifest_path, expected_manifest_sha256=completion["training_manifest_sha256"])
    try:
        count = len(corpus.evaluation_starts("val", 512))
        targets = int(corpus.evaluation_lengths("val", 512).sum())
        if count != manifest["dev_full_windows"] or targets != manifest["dev_target_count"]:
            raise ValueError("Development coverage does not reconstruct")
    finally:
        corpus.close()
    return plan, manifest, completion


def make_smoke(data_root, plan_path, preparation_log, out):
    data_root, plan_path, preparation_log = (Path(p).resolve() for p in (data_root, plan_path, preparation_log))
    plan, manifest, completion = prepared_data(data_root, plan_path, preparation_log)
    if (plan["candidate"] != "reference_directed_D" or plan["qualification"]["updates_each"] != 21
            or plan["qualification"]["maximum_minutes"] != 20 or plan["screen_gpu_hour_cap"] != 8):
        raise ValueError("Unexpected qualification/resource plan")
    configs = []
    for method in METHODS:
        configs.append(dict(corpus_kind=KIND, data_path=str(data_root / "training_manifest.json"),
            data_sha256=completion["training_manifest_sha256"], seed=plan["development_seeds"][0],
            method=method, lr=plan["screen_lr_centers"][method], aux_lr=plan["aux_lr"],
            momentum=plan["midpoint_momentum"], alpha=.25, out_beta=.25,
            decay=.01, soap_beta2=.9, root_refresh=10, cov_ema=.998, stats_ema=.99,
            stats_clock="microforward", cov_stride=32, out_sequences=2, out_ema=.8,
            n_layer=8, n_embd=128, n_head=2, seq_len=512,
            batch_tokens=262144, total_tokens=21*262144, microbatch_sequences=4,
            evaluation_microbatch_sequences=32, warmup_fraction=.034, cooldown_fraction=.1,
            grad_clip=1., validation_tokens=manifest["dev_full_windows"]*512,
            eval_every=21, selection_metric="full", precision="bf16", device="cuda",
            cpu_threads=2, keep_model_every=0, hardware_family="rtxa4000",
            run_id=f"{method}_d_smoke_s{plan['development_seeds'][0]}"))
    cohort = create(out, "smoke", configs)
    # Submission is separate; its validated explicit walltime supersedes the
    # standard helper's 10-minute default for legacy qualification cohorts.
    settings = read(cohort / "cohort.json")
    settings.update(stage="candidate_d_numerical_qualification", walltime_minutes=20)
    write_json(cohort / "cohort.json", settings)
    write_json(cohort / "QUALIFICATION_PLAN.json", dict(candidate_plan=plan,
        candidate_plan_path=str(plan_path), candidate_plan_sha256=digest(plan_path),
        data_completion_sha256=digest(data_root / "PREPARATION_COMPLETE.json"),
        training_manifest_sha256=completion["training_manifest_sha256"],
        expected_parameters=4861056, expected_steps=21, expected_training_forwards=2688,
        expected_evaluation_steps=[0, 1, 21], evaluation_microbatch_sequences=32,
        full_development_targets=manifest["dev_target_count"],
        full_development_windows=manifest["dev_full_windows"],
        maximum_walltime_minutes=20, scientific_ranking_claim=False, test_scored=False))
    write_json(cohort / "QUALIFICATION_PREFLIGHT.json", dict(
        qualification_plan_sha256=digest(cohort / "QUALIFICATION_PLAN.json"),
        source_manifest_sha256=digest(cohort / "source_manifest.json"),
        configs={p.name:digest(p) for p in sorted((cohort / "configs").glob("*.json"))}))
    (cohort / "README.md").write_text(
        "# Candidate D numerical qualification\n\n"
        "Five sequential 21-update arms on one 16 GB RTX A4000; 20-minute job limit.\n"
        "Use QUALIFICATION_PLAN.json and the study protocol. No ranking is claimed.\n"
        "Training uses four-sequence microbatches for the declared statistic clock;\n"
        "evaluation uses 32 and every development target. No test tokens are loaded.\n"
        "This cohort is prepared, not submitted.\n")
    return cohort


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def evaluate_qualification(cohort, out):
    cohort, out = Path(cohort).resolve(), Path(out).resolve()
    if out.exists():
        raise FileExistsError("Preserve previous qualification readouts")
    preflight = read(cohort / "QUALIFICATION_PREFLIGHT.json")
    if (preflight["qualification_plan_sha256"] != digest(cohort / "QUALIFICATION_PLAN.json")
            or preflight["source_manifest_sha256"] != digest(cohort / "source_manifest.json")
            or preflight["configs"] != {p.name:digest(p) for p in sorted((cohort / "configs").glob("*.json"))}):
        raise ValueError("Frozen qualification plan/configuration/source identity changed")
    plan = read(cohort / "QUALIFICATION_PLAN.json")
    configs = [read(p) for p in sorted((cohort / "configs").glob("*.json"))]
    if sorted(c["method"] for c in configs) != sorted(METHODS):
        raise ValueError("Exactly the five planned qualification methods are required")
    roots = [cohort / "runs" / c["run_id"] for c in configs]
    if any(not (r / "status.json").is_file() or read(r / "status.json").get("status") != "complete" for r in roots):
        raise ValueError("All five qualification arms must finish successfully")
    verify_sources(cohort)
    corpus = load_training_corpus(configs[0]["data_path"], expected_manifest_sha256=plan["training_manifest_sha256"])
    try:
        population = dict(starts=corpus.evaluation_starts("val", 512).clone(),
                          target_counts=corpus.evaluation_lengths("val", 512).clone(),
                          document_indices=corpus.evaluation_document_indices("val", 512).clone())
    finally:
        corpus.close()
    expected_weights = {"input_cov_weight": fp32_ema_weight(.998, 2688),
                        "input_weight": fp32_ema_weight(.99, 2688)}
    records, paired, reference_windows = [], None, None
    for cfg, root in zip(configs, roots):
        summary, metadata = read(root / "summary.json"), read(root / "metadata.json")
        if (not finite_numbers(summary) or summary.get("status") != "complete"
                or not 0 < summary["training_seconds"] <= summary["total_seconds"]):
            raise ValueError("Invalid or nonfinite qualification summary/timing")
        if (read(root / "config.json") != cfg or summary["steps"] != 21
                or summary["tokens"] != 21*262144 or summary["n_parameters"] != 4861056
                or metadata["model_config"]["stats_clock"] != "microforward"
                or metadata["covariance_gram_precision"] != "FP32; autocast disabled"
                or cfg["data_sha256"] != plan["training_manifest_sha256"]
                or cfg["data_path"] != configs[0]["data_path"]
                or Path(metadata["source_file"]).resolve() != cohort / "frozen/research/tiny_spectra/train.py"):
            raise ValueError("Qualification recipe, horizon, architecture or precision mismatch")
        hardware = summary["hardware"]
        if ("A4000" not in hardware["name"] or not 0 < hardware["total_memory_bytes"] < 45*1024**3
                or hardware != metadata["hardware"]):
            raise ValueError("Qualification hardware is not the declared sub-48GB family")
        identity = [metadata[k] for k in ("initial_parameter_sha256", "train_window_sha256",
                    "validation_bank_sha256", "validation_bank_lengths_sha256", "data_manifest_sha256")]
        identity.append(hardware)
        if paired is None:
            paired = identity
        elif identity != paired:
            raise ValueError("Qualification methods do not have paired data/initialization/hardware")
        saved = torch.load(root / "final_validation.pt", map_location="cpu", weights_only=True)
        if (int(saved["target_counts"].sum()) != plan["full_development_targets"]
                or len(saved["starts"]) != plan["full_development_windows"]
                or any(not torch.equal(saved[key], value) for key, value in population.items())):
            raise ValueError("Qualification did not score all development targets")
        if reference_windows is None:
            reference_windows = saved
        elif any(not torch.equal(saved[k], reference_windows[k]) for k in ("starts", "target_counts", "document_indices")):
            raise ValueError("Qualification evaluation populations differ")
        reconstructed = float((saved["sequence_nll"] * saved["target_counts"]).sum() / saved["target_counts"].sum())
        if (not math.isfinite(reconstructed) or abs(reconstructed-summary["full_validation_nll"]) > 1e-12
                or abs(reconstructed-summary["final_validation_nll"]) > 1e-12):
            raise ValueError("Saved qualification losses do not reconstruct")
        with torch.serialization.safe_globals([TorchVersion]):
            snapshot = torch.load(root / "final.pt", map_location="cpu", weights_only=True)
        if (snapshot["config"] != cfg or snapshot["metadata"] != metadata
                or not all(torch.isfinite(value).all() for value in tensors(snapshot))):
            raise ValueError("Nonfinite qualification checkpoint tensor")
        statistics = snapshot["optimizer"]["model_statistics"]
        if cfg["method"] in ("pd", "ts", "spd"):
            if len(statistics) != 48:
                raise ValueError("All 48 body statistic modules must be present")
            for state in statistics.values():
                if (int(state["_total_forwards"]) != 2688 or int(state["_step_forwards"]) != 128
                        or any(abs(float(state[key])-weight) > 1e-7 for key, weight in expected_weights.items())):
                    raise ValueError("Per-microforward clock or evaluation exclusion failed")
            roots_state = snapshot["optimizer"]["external"]["data_norm"]["roots"]
            if len(roots_state) != 48 or any(entry[1] != 21 for entry in roots_state.values()):
                raise ValueError("Third input-root refresh was not exercised")
        if cfg["method"] == "ts":
            if len(snapshot["optimizer"]["external"]["data_norm"]["out_stats"]) != 48:
                raise ValueError("TS output statistics are incomplete")
        if cfg["method"] == "spd":
            if len(snapshot["optimizer"]["external"]["soap"]["states"]) != 48:
                raise ValueError("SOAP statistics are incomplete")
        rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines()]
        if (not finite_numbers(rows) or [row["step"] for row in rows] != list(range(22))
                or [row["tokens"] for row in rows] != [step*262144 for step in range(22)]):
            raise ValueError("Incomplete, reordered or nonfinite qualification trajectory")
        evaluations = [row for row in rows if "validation_nll" in row]
        if [row["step"] for row in evaluations] != [0, 1, 21]:
            raise ValueError("Qualification observation cadence differs from plan")
        windows = plan["full_development_windows"]
        probe = min(128, windows)
        smoke_eval_work = 4*windows + 2*probe
        full_eval_work = 49*windows + 47*probe
        projected = (summary["training_seconds"]*368/21
                     + (summary["total_seconds"]-summary["training_seconds"])*full_eval_work/smoke_eval_work)
        records.append(dict(method=cfg["method"], run_id=cfg["run_id"], total_seconds=summary["total_seconds"],
            training_seconds=summary["training_seconds"], projected_full_run_seconds=projected,
            peak_cuda_allocated_bytes=summary["peak_cuda_allocated_bytes"],
            full_development_nll=reconstructed, hardware=hardware,
            artifact_sha256={name:digest(root/name) for name in ("metadata.json", "summary.json", "final_validation.pt", "final.pt")}))
    # At most eight arms per method: 3 rates x 2 seeds + one edge x 2 seeds.
    forecast = 8*sum(r["projected_full_run_seconds"] for r in records)/3600
    result = dict(numerical_qualification_passed=True, scientific_ranking_claim=False,
        test_scored=False, runs=records, maximum_40_arm_forecast_gpu_hours=forecast,
        forecast_within_eight_gpu_hours=forecast <= 8,
        source_manifest_sha256=digest(cohort / "source_manifest.json"),
        qualification_plan_sha256=digest(cohort / "QUALIFICATION_PLAN.json"),
        fp32_ema_reference_weights=expected_weights,
        decision=("Numerical checks pass; freeze the scientific screen/resource details before any launch."
                  if forecast <= 8 else "Numerical checks pass but forecast exceeds cap; stop for an execution decision."))
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    p = commands.add_parser("create")
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--preparation-log", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("report")
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "create":
        print(make_smoke(args.data, args.plan, args.preparation_log, args.out))
    else:
        result = evaluate_qualification(args.cohort, args.out)
        print(json.dumps({k:result[k] for k in ("numerical_qualification_passed", "maximum_40_arm_forecast_gpu_hours", "decision")}, indent=2))

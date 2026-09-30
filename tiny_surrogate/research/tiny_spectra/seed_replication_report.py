"""Read the complete fixed-recipe seed replication; never performs inference.

Until all four new arms are complete, only status fields are used, with no loss
calculation or scientific-summary access. Completed status files also contain
loss fields, which the barrier ignores. The selection seed is context only.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import numpy as np
import torch

from .scale_diagnostic_report import CORE, digest, read


TARGETS = 1_018_977
DOCUMENTS = 4_957
WINDOWS = 10_610
TRAIN_BLOCKS = 257_343
DATA_SHA256 = "15a4968d198013e014c8666d602fcdb3e1e1a66f01c3a2ed98ea5ebab2660b1d"
ARTIFACTS = ("config.json", "metadata.json", "summary.json", "status.json", "windows.pt",
             "final_validation.pt", "development_documents.json", "metrics.jsonl")


class IncompleteCohort(ValueError):
    pass


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def require_complete(roots):
    """Use only status fields until every arm completes; do not interpret losses."""
    incomplete = []
    for root in roots:
        path = root / "status.json"
        if not path.is_file() or read(path).get("status") != "complete":
            incomplete.append(root.name)
    if incomplete:
        raise IncompleteCohort("All four fresh arms must complete before readout; incomplete: "
                               + ", ".join(incomplete))
    # Even a completed arm's summary stays unopened until every status passes.
    for root in roots:
        path = root / "summary.json"
        if not path.is_file() or read(path).get("status") != "complete":
            raise IncompleteCohort("Completion summary missing: " + root.name)


def validate_plan(cohort):
    plan, preflight = read(cohort / "PLAN.json"), read(cohort / "PREFLIGHT.json")
    if digest(cohort / "PLAN.json") != preflight["plan_sha256"]:
        raise ValueError("Frozen replication plan changed")
    expected = dict(role="fixed_recipe_development_seed_replication", methods=["ts", "spd"],
                    seeds=[20260929, 20260930], selection_seed_excluded_from_primary=20260928,
                    learning_rate=.016, batch_tokens=100992, total_tokens=32939904, steps=327,
                    late_steps=list(range(232, 289, 8)), primary_threshold=-.005,
                    required_fresh_pairs=2, original_scale_gate_remains_failed=True,
                    qualification=False, sealed_test_scoring=False,
                    additional_runs_or_repair_if_failure=False)
    if any(plan.get(key) != value for key, value in expected.items()):
        raise ValueError("Unexpected fixed-recipe replication plan")
    if (set(plan["training_stream_sha256"]) != {str(seed) for seed in plan["seeds"]}
            or len(set(plan["training_stream_sha256"].values())) != 2
            or set(preflight["numerical_sources_unchanged"]) != set(CORE)
            or preflight["only_config_changes"] != ["seed", "run_id"] or preflight["test_scored"]):
        raise ValueError("Invalid preflight or frozen fresh-stream identities")
    return plan


def config_inventory(cohort, plan, *, original=False):
    rows = {}
    seeds = [plan["selection_seed_excluded_from_primary"]] if original else plan["seeds"]
    expected = {(method, seed) for method in plan["methods"] for seed in seeds}
    for path in sorted((cohort / "configs").glob("*.json")):
        cfg = read(path)
        if original and cfg.get("lr") != plan["learning_rate"]:
            continue
        key = (cfg["method"], cfg["seed"])
        if (key not in expected or key in rows or cfg["lr"] != plan["learning_rate"]
                or Path(cfg["run_id"]).name != cfg["run_id"] or cfg["run_id"] in (".", "..")):
            raise ValueError("Unexpected or duplicate replication arm")
        rows[key] = (cfg, cohort / "runs" / cfg["run_id"])
    if set(rows) != expected:
        raise ValueError("Every declared method/seed pair must have exactly one configuration")
    return rows


def verify_sources(cohort, original):
    manifests = []
    for directory in (cohort, original):
        manifest = read(directory / "source_manifest.json")
        for relative, expected in manifest.items():
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts or digest(directory / "frozen" / path) != expected:
                raise ValueError("Frozen source changed: " + relative)
        if not set(CORE) <= set(manifest):
            raise ValueError("Missing numerical source in manifest")
        manifests.append(manifest)
    if any(manifests[0][name] != manifests[1][name] for name in CORE):
        raise ValueError("Numerical sources differ from selected scale recipes")
    return {name: manifests[0][name] for name in CORE}


def verified_asset(root, spec):
    path = (root / spec["path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Data artifact escaped its directory")
    raw = path.read_bytes()
    if len(raw) != spec["bytes"] or hashlib.sha256(raw).hexdigest() != spec["sha256"]:
        raise ValueError("Development artifact integrity failure")
    return raw


def load_population(cfg):
    """Read only development bank/membership, never any sealed-test artifact."""
    path = Path(cfg["data_path"])
    if cfg["data_sha256"] != DATA_SHA256 or digest(path) != DATA_SHA256:
        raise ValueError("Training/development manifest differs from the frozen recipe")
    manifest = read(path)
    if (manifest["role"] != "training_and_development_only" or manifest["evaluation_policy_version"] != 2
            or manifest["seq_len"] != 128 or manifest["dev_target_count"] != TARGETS):
        raise ValueError("Wrong data role or evaluation policy")
    tensors = {key: torch.from_numpy(np.frombuffer(verified_asset(path.parent, manifest[field]), dtype="<i8").copy())
               for key, field in (("starts", "dev_starts"), ("target_counts", "dev_lengths"),
                                  ("document_indices", "dev_document_indices"))}
    documents = json.loads(verified_asset(path.parent, manifest["splits"]["val"]["documents"]))
    if (len(documents) != DOCUMENTS or len({d["identity"] for d in documents}) != DOCUMENTS
            or any(value.shape != (WINDOWS,) for value in tensors.values())
            or int(tensors["target_counts"].sum()) != TARGETS):
        raise ValueError("Wrong full-development population")
    starts, counts, indices = [], [], []
    for i, doc in enumerate(documents):
        for start in range(doc["start"], doc["end"] - 1, 128):
            starts.append(start)
            counts.append(min(128, doc["end"] - start - 1))
            indices.append(i)
    for key, values in (("starts", starts), ("target_counts", counts), ("document_indices", indices)):
        if not torch.equal(tensors[key], torch.tensor(values, dtype=torch.long)):
            raise ValueError("Development bank does not cover each document target exactly once")
    return dict(manifest=manifest, documents=documents, targets=TARGETS, windows=WINDOWS, **tensors)


def reconstruct_evaluation(validation, document_record, population):
    for key in ("starts", "target_counts", "document_indices"):
        if validation[key].dtype != torch.int64 or not torch.equal(validation[key], population[key]):
            raise ValueError("Saved development windows/counts/membership differ from frozen population")
    losses = validation["sequence_nll"]
    weights, owners = validation["target_counts"], validation["document_indices"]
    if (losses.shape != weights.shape or not torch.isfinite(losses).all()
            or (weights <= 0).any() or int(weights.sum()) != population["targets"]):
        raise ValueError("Invalid saved development losses or target counts")
    nll = float((weights * losses.double()).sum() / weights.sum())
    documents = document_record["documents"]
    if (document_record["total_targets"] != population["targets"]
            or [d["identity"] for d in documents] != [d["identity"] for d in population["documents"]]):
        raise ValueError("Saved document population differs")
    sums = torch.zeros(len(documents), dtype=torch.float64).scatter_add_(0, owners, losses.double() * weights)
    totals = torch.zeros(len(documents), dtype=torch.long).scatter_add_(0, owners, weights)
    for i, doc in enumerate(documents):
        if (doc["targets"] != int(totals[i]) or totals[i] <= 0 or not math.isfinite(doc["nll"])
                or abs(doc["nll"] - float(sums[i] / totals[i])) > 1e-12):
            raise ValueError("Saved per-document losses do not reconstruct from their windows")
    if abs(sum(d["nll"] * d["targets"] for d in documents) / population["targets"] - nll) > 1e-12:
        raise ValueError("Document means do not reconstruct full endpoint")
    groups = {}
    for group in range(4):
        subset = [d for d in documents if int(d["identity"][:2], 16) % 4 == group]
        if not subset:
            raise ValueError("A fixed document group is missing")
        count = sum(d["targets"] for d in subset)
        groups[group] = dict(nll=sum(d["nll"] * d["targets"] for d in subset) / count,
                             targets=count, documents=len(subset))
    return nll, groups


def validate_trajectory(rows, plan):
    if [row["step"] for row in rows] != list(range(plan["steps"] + 1)):
        raise ValueError("Missing or duplicate training steps")
    if any(row["tokens"] != min(row["step"] * plan["batch_tokens"], plan["total_tokens"]) for row in rows):
        raise ValueError("Trajectory token clock differs from the fixed recipe")
    evaluations = [row for row in rows if "validation_nll" in row]
    expected = sorted({0, 1, plan["steps"], *range(8, plan["steps"] + 1, 8)})
    if ([row["step"] for row in evaluations] != expected
            or not all(math.isfinite(row["validation_nll"]) for row in evaluations)):
        raise ValueError("Full-development evaluation trajectory is incomplete or nonfinite")
    return {row["step"]: row["validation_nll"] for row in evaluations}


def load_run(cohort, cfg, root, plan, population, expected_stream=None):
    summary, status, metadata = (read(root / name) for name in ("summary.json", "status.json", "metadata.json"))
    if read(root / "config.json") != cfg:
        raise ValueError("Executed configuration differs from frozen arm")
    if (any(row.get("status") != "complete" or row["steps"] != plan["steps"]
            or row["tokens"] != plan["total_tokens"] for row in (summary, status))
            or metadata["total_steps"] != plan["steps"]):
        raise ValueError("Incomplete locked horizon")
    hardware = summary["hardware"]
    if (hardware != metadata["hardware"] or not 0 < hardware["total_memory_bytes"] < 45 * 1024**3
            or "2080 Ti" not in hardware["name"]):
        raise ValueError("Ineligible or inconsistent hardware")
    if (metadata["source_file"] != str((cohort / "frozen/research/tiny_spectra/train.py").resolve())
            or metadata["corpus"] != population["manifest"] or metadata["data_manifest_sha256"] != DATA_SHA256
            or metadata["validation_bank_target_count"] != TARGETS
            or metadata["full_development_target_count"] != TARGETS):
        raise ValueError("Source, data identity, or full-development coverage mismatch")
    windows = torch.load(root / "windows.pt", map_location="cpu", weights_only=True)
    train = windows["train"]
    if (train.dtype != torch.int64 or train.shape != (TRAIN_BLOCKS,)
            or not torch.equal(train.sort().values, torch.arange(TRAIN_BLOCKS) * cfg["seq_len"])
            or not torch.equal(windows["train_probe"], train[:128])
            or not torch.equal(windows["validation"], population["starts"])):
        raise ValueError("Training stream is not bijective or evaluation/probe windows changed")
    stream_sha = tensor_digest(train)
    if (stream_sha != metadata["train_window_sha256"] or stream_sha != summary["train_window_sha256"]
            or (expected_stream is not None and stream_sha != expected_stream)
            or tensor_digest(population["starts"]) != metadata["validation_bank_sha256"]
            or tensor_digest(population["target_counts"]) != metadata["validation_bank_lengths_sha256"]
            or summary["initial_parameter_sha256"] != metadata["initial_parameter_sha256"]):
        raise ValueError("Raw tensor hashes or initialization metadata differ from frozen identities")
    validation = torch.load(root / "final_validation.pt", map_location="cpu", weights_only=True)
    nll, groups = reconstruct_evaluation(validation, read(root / "development_documents.json"), population)
    if any(abs(nll - row[key]) > 1e-12 for row in (summary, status)
           for key in ("full_validation_nll", "final_validation_nll")):
        raise ValueError("Saved windows do not reconstruct summary endpoints")
    rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line.strip()]
    curve = validate_trajectory(rows, plan)
    if len(curve) != 43 or abs(curve[plan["steps"]] - nll) > 1e-12:
        raise ValueError("Final curve point differs from full endpoint or not all43 evaluations exist")
    return dict(run_id=cfg["run_id"], method=cfg["method"], seed=cfg["seed"], lr=cfg["lr"],
                root=str(root.resolve()), full_development_nll=nll, groups=groups, curve=curve, trajectory=rows,
                hardware=hardware, initial_parameter_sha256=metadata["initial_parameter_sha256"],
                train_window_sha256=stream_sha, validation_bank_sha256=metadata["validation_bank_sha256"],
                validation_bank_lengths_sha256=metadata["validation_bank_lengths_sha256"],
                data_manifest_sha256=metadata["data_manifest_sha256"], total_seconds=summary["total_seconds"],
                peak_cuda_allocated_bytes=summary["peak_cuda_allocated_bytes"],
                artifacts={name: digest(root / name) for name in ARTIFACTS})


def load_runs(cohort, original):
    plan = validate_plan(cohort)
    inventory = config_inventory(cohort, plan)
    # This barrier precedes even reading original/selected scientific results.
    require_complete([root for _, root in inventory.values()])
    controls = config_inventory(original, plan, original=True)
    require_complete([root for _, root in controls.values()])
    numerical_sources = verify_sources(cohort, original)
    for (method, _), (cfg, _) in inventory.items():
        template = controls[method, plan["selection_seed_excluded_from_primary"]][0]
        if ({k: v for k, v in cfg.items() if k not in ("seed", "run_id")}
                != {k: v for k, v in template.items() if k not in ("seed", "run_id")}):
            raise ValueError("A setting other than seed/run_id changed from selected scale recipe")
    population = load_population(next(iter(inventory.values()))[0])
    runs = {key: load_run(cohort, cfg, root, plan, population, plan["training_stream_sha256"][str(key[1])])
            for key, (cfg, root) in inventory.items()}
    original_runs = {key: load_run(original, cfg, root, plan, population) for key, (cfg, root) in controls.items()}
    all_runs = {**runs, **original_runs}
    if len({json.dumps(row["hardware"], sort_keys=True) for row in all_runs.values()}) != 1:
        raise ValueError("Fresh and original arms do not share matched hardware")
    pair_fields = ("initial_parameter_sha256", "train_window_sha256", "validation_bank_sha256",
                   "validation_bank_lengths_sha256", "data_manifest_sha256")
    seeds = [*plan["seeds"], plan["selection_seed_excluded_from_primary"]]
    for seed in seeds:
        if any(all_runs["ts", seed][name] != all_runs["spd", seed][name] for name in pair_fields):
            raise ValueError("Within-seed initialization/data/window pairing mismatch")
    for name in ("initial_parameter_sha256", "train_window_sha256"):
        if len({all_runs["ts", seed][name] for seed in seeds}) != len(seeds):
            raise ValueError("Fresh seeds do not have distinct initializations/training streams")
    old_result = read(original / "report/results.json")
    if old_result["diagnostic_gate_passed"] or old_result["qualified_surrogate"]:
        raise ValueError("Expected preserved failed scale gate")
    return plan, runs, original_runs, numerical_sources, old_result


def paired_results(plan, runs):
    pairs = {}
    for seed in plan["seeds"]:
        ts, spd = runs["ts", seed], runs["spd", seed]
        gap = spd["full_development_nll"] - ts["full_development_nll"]
        pairs[seed] = dict(ts_nll=ts["full_development_nll"], spd_nll=spd["full_development_nll"],
                           spd_minus_ts=gap, primary_pass=gap <= plan["primary_threshold"],
                           late_fixed_recipe_spd_minus_ts={step: spd["curve"][step] - ts["curve"][step]
                                                           for step in plan["late_steps"]},
                           document_group_spd_minus_ts={group: spd["groups"][group]["nll"] - ts["groups"][group]["nll"]
                                                        for group in range(4)})
    return pairs


def report(cohort, original, out):
    cohort, original, out = cohort.resolve(), original.resolve(), out.resolve()
    if out.exists():
        raise FileExistsError("Preserve existing readouts; choose a new output directory")
    plan, runs, originals, numerical_sources, old = load_runs(cohort, original)
    pairs = paired_results(plan, runs)
    differences = [row["spd_minus_ts"] for row in pairs.values()]
    passed = all(row["primary_pass"] for row in pairs.values())
    old_seed = plan["selection_seed_excluded_from_primary"]
    original_gap = originals["spd", old_seed]["full_development_nll"] - originals["ts", old_seed]["full_development_nll"]
    result = dict(role="fixed_recipe_development_seed_replication_only", plan=plan,
                  plan_sha256=digest(cohort / "PLAN.json"), preflight_sha256=digest(cohort / "PREFLIGHT.json"),
                  analysis_source_sha256={str(path.resolve()): digest(path) for path in
                                          (Path(__file__), Path(__file__).with_name("scale_diagnostic_report.py"))},
                  numerical_source_sha256=numerical_sources, fresh_pairs=pairs,
                  primary=dict(each_pair_threshold=plan["primary_threshold"], both_fresh_pairs_pass=passed,
                               fresh_pair_count=len(pairs), mean_fresh_spd_minus_ts=statistics.mean(differences),
                               observed_fresh_gap_range=[min(differences), max(differences)],
                               selection_seed_excluded=True),
                  selected_original_context=dict(seed=old_seed, spd_minus_ts=original_gap,
                                                  excluded_from_primary=True, runs=list(originals.values())),
                  preserved_scale_diagnostic=dict(path=str(original / "report/results.json"),
                        sha256=digest(original / "report/results.json"), diagnostic_gate_passed=False, gates=old["gates"]),
                  original_scale_gate_remains_failed=True, qualified_surrogate=False, sealed_test_scored=False,
                  full_development_targets=TARGETS, full_development_documents=DOCUMENTS,
                  full_development_windows=WINDOWS, evaluation_steps=sorted(next(iter(runs.values()))["curve"]),
                  final_update_targets=plan["total_tokens"] - (plan["steps"] - 1) * plan["batch_tokens"],
                  total_gpu_hours_from_run_seconds=sum(row["total_seconds"] for row in runs.values()) / 3600,
                  runs=list(runs.values()),
                  decision=("Both fresh pairs reproduce the fixed development contrast; the original scale diagnostic remains failed and surrogate qualification remains false."
                            if passed else "The fixed-recipe lead does not meet replication: close this comparison without adding seeds or repairing the recipe."),
                  limitations=["Development documents were used in recipe selection; fresh seeds are not independent test confirmation.",
                               "The selected original seed is context only and excluded from every primary aggregate.",
                               "Late trajectories and all four fixed document groups are reported descriptively, without new gates or LR envelopes.",
                               "This comparison does not establish five-method ordering, batch/momentum effects, horizon robustness, or a qualified surrogate.",
                               "Both independent confirmation populations remain unscored; no test artifact is loaded by this readout."])
    out.mkdir(parents=True, exist_ok=False)
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    write_plots(out, plan, runs)
    lines = ["# Fixed-recipe fresh-seed replication", "", result["decision"], "",
             "All results are development evidence. The original scale diagnostic remains failed; surrogate qualification remains false.", "",
             "| Fresh seed | TS NLL | SOAP-PD NLL | SOAP-PD minus TS | At most −0.005 |",
             "|---:|---:|---:|---:|:---:|"]
    lines += [f"| {seed} | {row['ts_nll']:.8f} | {row['spd_nll']:.8f} | {row['spd_minus_ts']:+.8f} | {row['primary_pass']} |"
              for seed, row in pairs.items()]
    lines += ["", f"Fresh-pair mean: {statistics.mean(differences):+.8f}; observed range: {min(differences):+.8f} to {max(differences):+.8f}.",
              f"Original selection seed {old_seed}: {original_gap:+.8f}, excluded from the primary estimate.", "",
              "| Late step | Seed 20260929 gap | Seed 20260930 gap |", "|---:|---:|---:|"]
    for step in plan["late_steps"]:
        lines.append(f"| {step} | " + " | ".join(f"{pairs[seed]['late_fixed_recipe_spd_minus_ts'][step]:+.8f}" for seed in plan["seeds"]) + " |")
    lines += ["", "| Fixed document group | Seed 20260929 gap | Seed 20260930 gap |", "|---:|---:|---:|"]
    for group in range(4):
        lines.append(f"| {group} | " + " | ".join(f"{pairs[seed]['document_group_spd_minus_ts'][group]:+.8f}" for seed in plan["seeds"]) + " |")
    lines += ["", "Groups use the first byte of normalized document SHA-256 modulo four. Late steps and groups are descriptive; no additional pass criteria are applied.", "",
              "All 43 evaluations use every 1,018,977 target from 4,957 development documents. Saved window and document losses reconstruct each endpoint. Plan/config/source hashes, raw stream hashes, nonrecycling, eligible matched hardware, and seed pairing were checked.",
              "", "![Four fixed trajectories](trajectories.png)", "", "![Paired differences](paired_differences.png)", "",
              "The dashed −0.005 line is the endpoint replication criterion; it is not a new trajectory gate.", ""]
    lines.extend(result["limitations"])
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return {key: result[key] for key in ("primary", "decision", "qualified_surrogate", "sealed_test_scored")}


def write_plots(out, plan, runs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = dict(zip(plan["seeds"], ("tab:blue", "tab:orange")))
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for (method, seed), row in runs.items():
        steps = sorted(row["curve"])
        ax.plot([min(step * plan["batch_tokens"], plan["total_tokens"]) / 1e6 for step in steps],
                [row["curve"][step] for step in steps], color=colors[seed],
                ls="-" if method == "spd" else "--", label=f"{'SOAP-PD' if method == 'spd' else 'TS'}, seed {seed}")
    ax.set(xlabel="Training targets (millions)", ylabel="Full-development NLL per BPE token",
           title="Four fixed-recipe fresh-seed trajectories")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out / "trajectories.png", dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for seed in plan["seeds"]:
        steps = sorted(runs["ts", seed]["curve"])
        ax.plot([min(step * plan["batch_tokens"], plan["total_tokens"]) / 1e6 for step in steps],
                [runs["spd", seed]["curve"][step] - runs["ts", seed]["curve"][step] for step in steps],
                color=colors[seed], label=f"Seed {seed}")
    ax.axhline(0, color="black", lw=.7)
    ax.axhline(plan["primary_threshold"], color="gray", ls="--", label="Endpoint criterion −0.005")
    ax.axvspan(.7 * plan["total_tokens"] / 1e6, .9 * plan["total_tokens"] / 1e6, color="gray", alpha=.15)
    ax.set(xlabel="Training targets (millions)", ylabel="SOAP-PD minus TS NLL",
           title="Actual paired differences at fixed LR 0.016")
    ax.grid(alpha=.2); ax.legend(fontsize=8)
    fig.savefig(out / "paired_differences.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "original", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.cohort, args.original, args.out), indent=2))

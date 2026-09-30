"""Read-only cohort accounting and explicitly exploratory surrogate reports.

Select learning rates using the declared endpoint development metric:
``selection_metric=bank`` (the historical default) or ``selection_metric=full``.
Always retain both bank and full-split results and label the selection basis.
Run with ``python -m research.tiny_spectra.analyze COHORT``.
"""
import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import time


ARCH_KEYS = ("n_layer", "n_embd", "n_head", "seq_len")
FIELDS = ("run_id", "status", "method", "lr", "batch_tokens", "seed", "momentum", "alpha",
          "total_tokens", "last_step", "last_tokens", "endpoint_bank_nll", "full_validation_nll",
          "selection_metric", "train_probe_nll", "pre_cooldown_step", "pre_cooldown_tokens", "pre_cooldown_nll",
          "best_observed_bank_nll", "endpoint_minus_best", "late_train_probe_change",
          "overfit_sign", "training_seconds", "peak_cuda_allocated_bytes", "issues")


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def selection_policy(config):
    policy = config.get("selection_metric", "bank")
    if policy not in ("bank", "full"):
        raise ValueError(f"Unknown selection_metric {policy!r}; expected 'bank' or 'full'")
    return policy


def selection_basis(policy):
    return "final fixed-bank validation NLL" if policy == "bank" else "final full-development-split validation NLL"


def data_identity_sha256(metadata):
    """Frozen tokenized-data manifest identity, with the legacy text fallback."""
    return metadata.get("data_manifest_sha256") or metadata.get("corpus", {}).get("raw_utf8_sha256")


def corpus_execution_sources(config):
    kind = config.get("corpus_kind")
    if kind == "tiny_stories_byte_bpe_v1":
        return ("research/tiny_spectra/stories.py",)
    if kind == "fineweb_byte_bpe_v1":
        return ("research/tiny_spectra/fineweb.py", "research/tiny_spectra/stories.py")
    return ()


def frozen_source_provenance(cohort, relative_paths):
    manifest_path = Path(cohort) / "source_manifest.json"
    issues, observed = [], {}
    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        for relative in relative_paths:
            path = Path(cohort) / "frozen" / relative
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            observed[relative] = digest
            if not digest or manifest.get(relative) != digest:
                issues.append(f"frozen execution source mismatch/missing: {relative}")
    except (OSError, ValueError) as error:
        manifest_hash = None
        issues.append(f"missing/unreadable source manifest or execution source: {error}")
    return dict(path=str(manifest_path), sha256=manifest_hash, execution_sha256=observed, issues=issues)


def atomic_text(path, text):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_json(path, issues, required=False):
    if not path.exists():
        if required:
            issues.append(f"missing {path.name}")
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (ValueError, OSError) as error:
        issues.append(f"unreadable {path.name}: {error}")
        return {}


def read_run(cohort, config_path):
    issues = []
    cfg = read_json(config_path, issues, required=True)
    run_id = str(cfg.get("run_id", config_path.stem))
    if Path(run_id).name != run_id or run_id in (".", ".."):
        raise ValueError(f"Unsafe run_id in {config_path}: {run_id!r}")
    run_path = cohort / "runs" / run_id
    meta = read_json(run_path / "metadata.json", issues)
    status_record = read_json(run_path / "status.json", issues)
    summary = read_json(run_path / "summary.json", issues)
    status = status_record.get("status", summary.get("status", "missing" if not run_path.exists() else "unknown"))
    if status == "complete" and summary.get("status") != "complete":
        issues.append("complete status has no complete summary")
        status = "inconsistent"
    elif summary.get("status") == "complete" and status != "complete":
        issues.append("summary and status disagree")
        status = "inconsistent"
    if status == "failed":
        failure = read_json(run_path / "failure.json", issues)
        if failure.get("error") or status_record.get("error"):
            issues.append("recorded failure: " + str(failure.get("error", status_record.get("error"))))
    saved_cfg = read_json(run_path / "config.json", issues)
    if saved_cfg and saved_cfg != cfg:
        issues.append("saved run config differs from cohort config")
    records = []
    metrics_path = run_path / "metrics.jsonl"
    if metrics_path.exists():
        try:
            metric_lines = metrics_path.read_text().splitlines()
        except OSError as error:
            issues.append(f"unreadable metrics.jsonl: {error}")
            metric_lines = []
        for line_number, line in enumerate(metric_lines, 1):
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or not finite(record.get("step")) or not finite(record.get("tokens")):
                    raise ValueError("record needs finite step and tokens")
                records.append(record)
            except ValueError as error:
                issues.append(f"metrics line {line_number}: {error}")
    if any(b["step"] <= a["step"] or b["tokens"] < a["tokens"] for a, b in zip(records, records[1:])):
        issues.append("metrics steps/tokens are not strictly increasing/nondecreasing")
    evaluations = [r for r in records if finite(r.get("validation_nll"))]
    endpoint = summary.get("final_validation_nll") if status == "complete" else None
    if endpoint is not None and not finite(endpoint):
        issues.append("nonfinite endpoint")
        endpoint = None
    if status == "complete":
        for field in ("initial_parameter_sha256", "train_window_sha256", "validation_bank_sha256"):
            if not meta.get(field):
                issues.append(f"complete run lacks metadata {field}")
        if not data_identity_sha256(meta):
            issues.append("complete run lacks corpus/data identity checksum")
        elif cfg.get("data_sha256") and data_identity_sha256(meta) != cfg["data_sha256"]:
            issues.append("corpus/data identity differs from configured data hash")
        if summary.get("tokens") != cfg.get("total_tokens"):
            issues.append("complete summary does not reach configured token budget")
        if not evaluations or evaluations[-1]["tokens"] != cfg.get("total_tokens"):
            issues.append("complete run lacks endpoint validation metric")
        elif finite(endpoint) and abs(endpoint - evaluations[-1]["validation_nll"]) > 1e-6:
            issues.append("summary endpoint disagrees with fixed-bank metric")
    # Last evaluation at or before the declared cooldown start. This avoids
    # silently calling an already cooled-down point 'pre-cooldown'.
    target = cfg.get("total_tokens", 0) * (1 - cfg.get("cooldown_fraction", 0.1))
    earlier = [r for r in evaluations if r["tokens"] <= target]
    pre = earlier[-1] if earlier else {}
    best = min((r["validation_nll"] for r in evaluations), default=None)
    probes = [r for r in evaluations if finite(r.get("train_probe_nll"))]
    late_change = probes[-1]["train_probe_nll"] - pre["train_probe_nll"] if probes and finite(pre.get("train_probe_nll")) else None
    overfit = bool(finite(endpoint) and finite(pre.get("validation_nll")) and finite(late_change)
                  and endpoint > pre["validation_nll"] and late_change < 0)
    row = {key: cfg.get(key) for key in FIELDS[:9]}
    row.update(run_id=run_id, status=status, total_tokens=cfg.get("total_tokens"),
               selection_metric=selection_policy(cfg),
               last_step=records[-1]["step"] if records else None,
               last_tokens=records[-1]["tokens"] if records else None,
               endpoint_bank_nll=endpoint,
               full_validation_nll=summary.get("full_validation_nll"), train_probe_nll=summary.get("train_probe_nll"),
               pre_cooldown_step=pre.get("step"), pre_cooldown_tokens=pre.get("tokens"),
               pre_cooldown_nll=pre.get("validation_nll"), best_observed_bank_nll=best,
               endpoint_minus_best=endpoint-best if finite(endpoint) and finite(best) else None,
               late_train_probe_change=late_change, overfit_sign=overfit,
               training_seconds=summary.get("training_seconds"),
               peak_cuda_allocated_bytes=summary.get("peak_cuda_allocated_bytes"), issues=issues)
    result = dict(config=cfg, config_path=str(config_path), path=str(run_path), metadata=meta,
                  summary=summary, row=row, evaluations=evaluations)
    if corpus_execution_sources(cfg):
        source = frozen_source_provenance(run_path.resolve().parent.parent, corpus_execution_sources(cfg))
        row["issues"].extend(source["issues"])
        result["conditional_source_provenance"] = source
    return result


def provenance_groups(runs):
    groups = {}
    for run in runs:
        cfg, meta = run["config"], run["metadata"]
        # Tracking flags differ by optimizer and are not architecture changes.
        key = dict(seed=cfg.get("seed"), total_tokens=cfg.get("total_tokens"), selection_metric=selection_policy(cfg),
                   architecture={k: cfg.get(k) for k in ARCH_KEYS})
        groups.setdefault(canonical(key), []).append(run)
    result = []
    for key, members in groups.items():
        checks = {}
        getters = [
            ("initial_parameter_sha256", lambda m: m.get("initial_parameter_sha256")),
            ("train_window_sha256", lambda m: m.get("train_window_sha256")),
            ("validation_bank_sha256", lambda m: m.get("validation_bank_sha256")),
            ("data_identity_sha256", data_identity_sha256),
            ("corpus_splits", lambda m: canonical(m["corpus"]["splits"]) if m.get("corpus", {}).get("splits") else None),
        ]
        # Character cohorts predate masked evaluation windows. All-absent is
        # legacy-compatible; once a member records lengths, compare/require
        # the hash in its completed peers too. Pending metadata remains pending.
        if any("validation_bank_lengths_sha256" in r["metadata"] for r in members):
            getters.append(("validation_bank_lengths_sha256", lambda m: m.get("validation_bank_lengths_sha256")))
        for label, getter in getters:
            values = {r["row"]["run_id"]: getter(r["metadata"]) for r in members}
            observed = set(v for v in values.values() if v is not None)
            missing = [k for k, v in values.items() if v is None]
            checks[label] = dict(status="mismatch" if len(observed) > 1 else "incomplete" if missing else "matched",
                                 missing_run_ids=missing, values_by_run=values)
        if any("conditional_source_provenance" in r for r in members):
            values = {r["row"]["run_id"]: r.get("conditional_source_provenance", {}).get("execution_sha256") for r in members}
            observed = {canonical(value) for value in values.values() if value}
            missing = [run_id for run_id, value in values.items() if not value]
            checks["conditional_execution_sources"] = dict(
                status="mismatch" if len(observed) > 1 else "incomplete" if missing else "matched",
                missing_run_ids=missing, values_by_run=values)
        result.append(dict(group=json.loads(key), run_ids=[r["row"]["run_id"] for r in members], checks=checks))
    return result


def apply_provenance_vetoes(runs, paired):
    mismatched_ids = {run_id for group in paired
                      if any(check["status"] == "mismatch" for check in group["checks"].values())
                      for run_id in group["run_ids"]}
    missing_lengths_ids = {run_id for group in paired
                           for run_id in group["checks"].get("validation_bank_lengths_sha256", {}).get("missing_run_ids", [])}
    missing_source_ids = {run_id for group in paired
                         for run_id in group["checks"].get("conditional_execution_sources", {}).get("missing_run_ids", [])}
    for run in runs:
        run_id = run["row"]["run_id"]
        if run_id in mismatched_ids:
            run["row"]["issues"].append("paired provenance group has mismatched hashes; selection withheld")
        if run["row"]["status"] == "complete" and run_id in missing_lengths_ids:
            run["row"]["issues"].append("complete paired run lacks validation_bank_lengths_sha256; selection withheld")
        if run["row"]["status"] == "complete" and run_id in missing_source_ids:
            run["row"]["issues"].append("complete paired run lacks conditional execution-source hashes; selection withheld")


def candidate_selections(runs):
    groups = {}
    # Hold every recipe choice except seed and LR fixed. This prevents accidental
    # comparisons across token budgets, architectures, alpha, or momentum.
    ignored = {"run_id", "seed", "lr"}
    for run in runs:
        key = {k: v for k, v in run["config"].items() if k not in ignored}
        key["selection_metric"] = selection_policy(run["config"])
        groups.setdefault(canonical(key), []).append(run)
    selected = []
    for key, members in groups.items():
        policy = selection_policy(members[0]["config"])
        score_field = "endpoint_bank_nll" if policy == "bank" else "full_validation_nll"
        rates = sorted({r["config"].get("lr") for r in members if finite(r["config"].get("lr"))})
        expected_seeds = sorted({r["config"].get("seed") for r in members if r["config"].get("seed") is not None})
        candidates = []
        for rate in rates:
            planned = [r for r in members if r["config"].get("lr") == rate]
            valid = [r for r in planned if r["row"]["status"] == "complete"
                     and finite(r["row"]["endpoint_bank_nll"]) and finite(r["row"].get(score_field)) and not r["row"]["issues"]]
            seeds = [r["config"].get("seed") for r in valid]
            eligible = len(valid) == len(planned) and sorted(seeds) == expected_seeds
            candidates.append(dict(lr=rate, planned_run_ids=[r["row"]["run_id"] for r in planned],
                                   completed_run_ids=[r["row"]["run_id"] for r in valid], seeds=seeds,
                                   eligible=eligible,
                                   mean_selection_nll=statistics.mean(r["row"][score_field] for r in valid) if valid else None,
                                   mean_endpoint_bank_nll=statistics.mean(r["row"]["endpoint_bank_nll"] for r in valid) if valid else None,
                                   mean_full_validation_nll=statistics.mean(r["row"]["full_validation_nll"] for r in valid)
                                   if valid and all(finite(r["row"]["full_validation_nll"]) for r in valid) else None))
        eligible = [c for c in candidates if c["eligible"]]
        winner = min(eligible, key=lambda c: (c["mean_selection_nll"], c["lr"])) if eligible else None
        minimizing_rates = ([c["lr"] for c in eligible if c["mean_selection_nll"] == winner["mean_selection_nll"]]
                            if winner else [])
        touches_boundary = bool(minimizing_rates and any(rate in (rates[0], rates[-1]) for rate in minimizing_rates))
        selected.append(dict(recipe=json.loads(key), expected_seeds=expected_seeds, candidates=candidates,
                             selected=winner, selection_metric="mean " + selection_basis(policy), selection_policy=policy,
                             discovery_only=True, grid_complete=all(c["eligible"] for c in candidates) and bool(candidates),
                             minimizing_rates=minimizing_rates,
                             boundary_lr=bool(len(rates) > 1 and touches_boundary),
                             bracket_qualified=bool(winner and len(rates) >= 3 and not touches_boundary)))
    return selected


def threshold_rows(runs, thresholds):
    results = []
    for run in runs:
        points = run["evaluations"]
        for threshold in thresholds:
            reached = next((i for i, point in enumerate(points) if point["validation_nll"] <= threshold), None)
            result = dict(run_id=run["row"]["run_id"], status=run["row"]["status"], threshold=threshold,
                          reached=reached is not None, observed_step=None, observed_tokens=None,
                          interpolated_step=None, interpolated_tokens=None)
            if reached is not None:
                right = points[reached]
                result.update(observed_step=right["step"], observed_tokens=right["tokens"])
                if reached > 0:
                    left = points[reached - 1]
                    fraction = (left["validation_nll"] - threshold) / (left["validation_nll"] - right["validation_nll"])
                    for axis in ("step", "tokens"):
                        result["interpolated_" + axis] = left[axis] + fraction * (right[axis] - left[axis])
                else:
                    result.update(interpolated_step=right["step"], interpolated_tokens=right["tokens"])
            results.append(result)
    return results


def write_csv(path, rows, fields):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: canonical(v) if isinstance(v, (list, dict)) else v for k, v in row.items()})
    atomic_text(path, stream.getvalue())


def plot_runs(runs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    groups = {}
    for run in runs:
        cfg = run["config"]
        key = (cfg.get("batch_tokens"), cfg.get("total_tokens"), tuple(cfg.get(k) for k in ARCH_KEYS), selection_policy(cfg))
        groups.setdefault(key, []).append(run)
    paths = []
    for index, (key, members) in enumerate(groups.items()):
        for axis in ("tokens", "step"):
            fig, ax = plt.subplots(figsize=(10, 6))
            for run in members:
                cfg, points = run["config"], run["evaluations"]
                if not points:
                    continue
                label = f"{cfg.get('method')} lr={cfg.get('lr')} β={cfg.get('momentum')} seed={cfg.get('seed')} [{run['row']['status']}]"
                ax.plot([p[axis] for p in points], [p["validation_nll"] for p in points], label=label, linewidth=1.2, alpha=0.8)
            ax.set(xlabel="Training tokens" if axis == "tokens" else "Optimizer steps", ylabel="Fixed-bank validation NLL",
                   title=f"Development bank trajectories · batch {key[0]} · token budget {key[1]} · selection={key[3]}")
            ax.grid(alpha=0.2)
            if ax.lines:
                ax.legend(fontsize=6, loc="upper right")
            fig.tight_layout()
            path = out / f"validation_{axis}_{index:02d}.png"
            temporary = path.with_suffix(".tmp")
            fig.savefig(temporary, format="png", dpi=140)
            temporary.replace(path)
            plt.close(fig)
            paths.append(path.name)
    return paths


def parse_thresholds(value):
    if value is None:
        return []
    if value.startswith("["):
        parsed = json.loads(value)
    elif Path(value).is_file():
        parsed = json.loads(Path(value).read_text())
    else:
        parsed = [float(part) for part in value.split(",")]
    if not isinstance(parsed, list) or not parsed or not all(finite(x) for x in parsed):
        raise ValueError("thresholds must be a nonempty JSON list or comma-separated finite numbers")
    return sorted(set(float(x) for x in parsed), reverse=True)


def analyze(cohort, out=None, thresholds=None):
    cohort = Path(cohort).resolve()
    out = Path(out).resolve() if out else cohort / "report"
    forbidden = (cohort / "runs", cohort / "configs")
    if out == cohort or any(out == path or path in out.parents for path in forbidden):
        raise ValueError("Report output must not overlap cohort inputs or experiment artifacts")
    marker = out / "report_manifest.json"
    if out.exists() and any(out.iterdir()) and not marker.exists():
        raise FileExistsError("Refusing to overwrite an existing directory without a report manifest")
    previous_manifest = json.loads(marker.read_text()) if marker.exists() else {}
    if previous_manifest and previous_manifest.get("generator") != "research.tiny_spectra.analyze":
        raise ValueError("Output manifest belongs to a different generator")
    config_paths = sorted((cohort / "configs").glob("*.json"))
    if not config_paths:
        raise ValueError(f"No config files in {cohort / 'configs'}")
    runs = [read_run(cohort, path) for path in config_paths]
    ids = [run["row"]["run_id"] for run in runs]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate run_id values in cohort configs")
    unconfigured = sorted(path.name for path in (cohort / "runs").iterdir() if path.is_dir() and path.name not in ids) if (cohort / "runs").exists() else []
    paired = provenance_groups(runs)
    apply_provenance_vetoes(runs, paired)
    selections = candidate_selections(runs)
    requested_thresholds = thresholds or []
    crossings = threshold_rows(runs, requested_thresholds)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", [r["row"] for r in runs], FIELDS)
    atomic_json(out / "pairing.json", paired)
    atomic_json(out / "selection.json", dict(discovery_only=True, groups=selections,
                rule="Minimum mean declared endpoint development NLL across identical planned seeds: selection_metric='bank' (default) or 'full'. Policies are never pooled.",
                scope="Separate complete recipes; incomplete LR grids yield provisional candidates only."))
    atomic_json(out / "thresholds.json", dict(thresholds=requested_thresholds, records=crossings,
                provenance="Caller-supplied; this report does not establish predeclaration.",
                rule="First observed crossing, plus linear interpolation between adjacent evaluations; no extrapolation."))
    if crossings:
        write_csv(out / "thresholds.csv", crossings, list(crossings[0]))
    plots = plot_runs(runs, out)
    counts = {state: sum(r["row"]["status"] == state for r in runs) for state in sorted({r["row"]["status"] for r in runs})}
    lines = ["# Tiny surrogate discovery report", "", "Exploratory candidate screening only. Optimizer ordering, batch effects, and momentum optima require fresh-seed confirmation; this report makes no statistical ranking claim.", "",
             f"Accounted for all {len(runs)} configured runs: " + ", ".join(f"{k}={v}" for k, v in counts.items()) + ".",
             "Status is the saved artifact status, not a live scheduler/process check.", "",
             "Selection uses the declared mean endpoint development metric, with the same planned seeds per learning rate: `selection_metric=bank` (default) selects the fixed bank; `selection_metric=full` selects the full development split. Both scores are retained; incompatible policies form separate groups. These are development scores, not sealed-test results. Incomplete grids remain provisional.", "",
             "| Method | Batch | Momentum | α | Selection basis | Selected LR | Fixed-bank NLL | Full NLL | Seeds | Bracket |",
             "|---|---:|---:|---:|---|---:|---:|---:|---:|---|"]
    for group in selections:
        cfg, winner = group["recipe"], group["selected"]
        if winner:
            bracket = "boundary" if group["boundary_lr"] else "interior" if group["bracket_qualified"] else "unqualified"
            if not group["grid_complete"]:
                bracket += "; incomplete grid"
            full = winner["mean_full_validation_nll"]
            lines.append(f"| {cfg.get('method')} | {cfg.get('batch_tokens')} | {cfg.get('momentum')} | {cfg.get('alpha')} | {group['selection_policy']} | {winner['lr']:g} | {winner['mean_endpoint_bank_nll']:.6f} | {full if full is not None else '—'} | {len(winner['seeds'])} | {bracket} |")
        else:
            lines.append(f"| {cfg.get('method')} | {cfg.get('batch_tokens')} | {cfg.get('momentum')} | {cfg.get('alpha')} | {group['selection_policy']} | — | — | — | — | no eligible candidate |")
    lines += ["", "See `selection.json` for complete recipes, all candidate scores, and selected run IDs; `runs.csv` includes every configured status and issue.", "",
              "The pre-cooldown value is the final evaluation at or before the configured cooldown start (90% by default). An overfit sign means validation worsened from that point while the fixed training probe improved; it is a descriptive flag, not a significance test.", ""]
    overfit = [r["row"]["run_id"] for r in runs if r["row"]["overfit_sign"]]
    lines.append("Overfit signs: " + (", ".join(overfit) if overfit else "none observed in available paired probe points") + ".")
    mismatches = [p for p in paired if any(check["status"] == "mismatch" for check in p["checks"].values())]
    incomplete = [p for p in paired if any(check["status"] == "incomplete" for check in p["checks"].values())]
    lines += ["", f"Pairing audit: {len(mismatches)} groups with mismatched hashes; {len(incomplete)} groups with missing hashes. Details and missing run IDs are in `pairing.json`. Hash matching checks initial weights, the training-window stream, the fixed validation bank, corpus/data identity (tokenized manifest or legacy text hash), and split manifests within seed/budget/architecture groups. Validation-bank length hashes are checked when any peer records them; historical groups that all omit them remain compatible."]
    if unconfigured:
        lines += ["", "Unconfigured run directories (not silently included in selection): " + ", ".join(unconfigured) + "."]
    issues = [(r["row"]["run_id"], issue) for r in runs for issue in r["row"]["issues"]]
    if issues:
        lines += ["", "Artifact issues:", ""] + [f"- {run_id}: {issue}" for run_id, issue in issues]
    lines += ["", ("Requested common-threshold crossings are in `thresholds.csv`; interpolation is exploratory, and missing crossings are censored rather than extrapolated." if requested_thresholds else "No common loss thresholds were supplied. No success thresholds or step-equivalent speedups were selected after seeing the trajectories."), ""]
    for plot in plots:
        lines += [f"![Discovery validation trajectory]({plot})", ""]
    atomic_text(out / "README.md", "\n".join(lines))
    generated_files = ["README.md", "runs.csv", "pairing.json", "selection.json", "thresholds.json", *plots]
    if crossings:
        generated_files.append("thresholds.csv")
    # Delete only explicitly tracked generated files from the prior report.
    # Experiment directories and unknown user files are never touched.
    for name in previous_manifest.get("generated_files", []):
        if name not in generated_files and Path(name).name == name:
            stale = out / name
            if stale.is_file():
                stale.unlink()
    atomic_json(marker, dict(schema_version=1, generator="research.tiny_spectra.analyze", cohort=str(cohort),
                generated_unix=time.time(), counts=counts, configured_run_ids=ids, unconfigured_run_directories=unconfigured,
                config_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in config_paths},
                thresholds=requested_thresholds, plots=plots, generated_files=generated_files))
    return dict(report=str(out), counts=counts, pairing_mismatch_groups=len(mismatches), unconfigured_run_directories=unconfigured)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--thresholds", help="JSON list, JSON file, or comma-separated loss thresholds; provenance is caller supplied")
    args = parser.parse_args()
    print(json.dumps(analyze(args.cohort, args.out, parse_thresholds(args.thresholds)), indent=2))


if __name__ == "__main__":
    main()

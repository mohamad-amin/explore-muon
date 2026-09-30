"""Exploratory batch × momentum report from immutable tiny-surrogate cohorts.

python -m research.tiny_spectra.dynamics_report --cohorts COHORT ... --out NEW_DIR
Learning rates and best tested momenta use the declared FINAL development
metric: selection_metric='bank' (historical default) or 'full'. Both scores are
retained, and incompatible selection policies are never pooled.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

from .analyze import (atomic_json, atomic_text, canonical, corpus_execution_sources, data_identity_sha256, finite, frozen_source_provenance, read_run,
                      selection_basis, selection_policy, write_csv)


GENERATOR = "research.tiny_spectra.dynamics_report"
THRESHOLDS = (2.4, 2.2, 2.0)
EXECUTION_SOURCES = ("research/tiny_spectra/train.py", "research/tiny_spectra/model.py",
                     "research/tiny_spectra/data.py", "research/tiny_spectra/optim.py",
                     "research/adamw_spectra/model.py", "research/adamw_spectra/muon.py",
                     "research/adamw_spectra/data_norm_muon.py")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def recipe(config, omit):
    # Preserve seed, precision, architecture, token budget, auxiliary LR, decay,
    # all statistic clocks, etc. No silent recipe pooling across these choices.
    # eval_every is an observation cadence, not a training choice in this
    # dropout-free runner. Actual crossings retain each recorded cadence.
    result = {k: v for k, v in config.items() if k not in {"run_id", "eval_every", *omit}}
    result["selection_metric"] = selection_policy(config)
    return result


def load_runs(cohorts):
    by_path, aliases, sources = {}, [], {}
    for cohort in cohorts:
        configs = sorted((cohort / "configs").glob("*.json"))
        if not configs:
            raise ValueError(f"No configs in {cohort}")
        for path in configs:
            run = read_run(cohort, path)
            resolved = str(Path(run["path"]).resolve())
            origin = dict(path=str(path), sha256=sha256(path), cohort=str(cohort))
            if resolved in by_path:
                old = by_path[resolved]
                if run["config"] != old["config"]:
                    old["row"]["issues"].append("aliased configs disagree")
                old["config_origins"].append(origin)
                aliases.append(dict(alias=run["path"], resolved=resolved))
                continue
            run.update(uid=resolved, config_origins=[origin])
            actual_cohort = Path(resolved).parent.parent
            manifest_path = actual_cohort / "source_manifest.json"
            required_sources = EXECUTION_SOURCES + corpus_execution_sources(run["config"])
            source_key = (str(manifest_path), required_sources)
            if source_key not in sources:
                sources[source_key] = frozen_source_provenance(actual_cohort, required_sources)
            source = sources[source_key]
            run["row"]["issues"].extend(issue for issue in source["issues"] if issue not in run["row"]["issues"])
            run["provenance"] = dict(source=source, artifact_sha256={name: sha256(Path(resolved) / name)
                for name in ("config.json", "metadata.json", "summary.json", "status.json", "metrics.jsonl")})
            if run["row"]["status"] == "complete":
                if not finite(run["row"].get("full_validation_nll")):
                    run["row"]["issues"].append("complete run has no finite full-validation result")
                if not run["evaluations"] or run["evaluations"][0]["step"] != 0:
                    run["row"]["issues"].append("complete run lacks step-zero validation")
                cfg, meta = run["config"], run["metadata"]
                expected_steps = math.ceil(cfg["total_tokens"] / cfg["batch_tokens"])
                if run["summary"].get("steps") != expected_steps:
                    run["row"]["issues"].append("complete summary has incorrect step budget")
                if data_identity_sha256(meta) != cfg.get("data_sha256"):
                    run["row"]["issues"].append("corpus/data identity differs from configured data hash")
                memory = meta.get("hardware", {}).get("total_memory_bytes")
                if cfg.get("device") == "cuda" and (not finite(memory) or memory >= 45 * 1024**3):
                    run["row"]["issues"].append("missing or ineligible sub-48GB GPU evidence")
            by_path[resolved] = run
    return list(by_path.values()), aliases


def audit_pairing(runs):
    groups = {}
    for run in runs:
        key = canonical(recipe(run["config"], {"method", "lr", "batch_tokens", "momentum"}))
        groups.setdefault(key, []).append(run)
    result = []
    for index, (key, members) in enumerate(sorted(groups.items())):
        checks = {}
        getters = [
            ("initial_parameter_sha256", lambda r: r["metadata"].get("initial_parameter_sha256")),
            ("train_window_sha256", lambda r: r["metadata"].get("train_window_sha256")),
            ("validation_bank_sha256", lambda r: r["metadata"].get("validation_bank_sha256")),
            ("data_identity_sha256", lambda r: data_identity_sha256(r["metadata"])),
            ("corpus_splits", lambda r: r["metadata"].get("corpus", {}).get("splits")),
            ("execution_sources", lambda r: r["provenance"]["source"]["execution_sha256"]),
        ]
        if any("validation_bank_lengths_sha256" in r["metadata"] for r in members):
            getters.append(("validation_bank_lengths_sha256", lambda r: r["metadata"].get("validation_bank_lengths_sha256")))
        for name, getter in getters:
            observed = {r["uid"]: getter(r) for r in members}
            values = {canonical(value) for value in observed.values() if value}
            missing = [uid for uid, value in observed.items() if not value]
            mismatch = len(values) > 1
            checks[name] = dict(mismatch=mismatch, missing=missing, values=observed)
            if mismatch:
                for run in members:
                    run["row"]["issues"].append(f"comparison family mismatches {name}; selection vetoed")
            for run in members:
                if run["uid"] in missing and run["row"]["status"] == "complete":
                    run["row"]["issues"].append(f"complete run lacks pairing evidence: {name}")
        family = f"family_{index:02d}"
        for run in members:
            run["family"] = family
        result.append(dict(family=family, recipe=json.loads(key), checks=checks))
    return result


def select_lrs(runs):
    groups = {}
    for run in runs:
        key = canonical(recipe(run["config"], {"lr"}))
        groups.setdefault(key, []).append(run)
    selections = []
    for index, (key, members) in enumerate(sorted(groups.items())):
        policy = selection_policy(members[0]["config"])
        score_field = "endpoint_bank_nll" if policy == "bank" else "full_validation_nll"
        counts = Counter(r["config"].get("lr") for r in members)
        candidates = []
        for run in sorted(members, key=lambda r: (r["config"].get("lr", 0), r["uid"])):
            row = run["row"]
            issues = list(row["issues"])
            if counts[run["config"].get("lr")] > 1:
                issues.append("duplicate independent runs for same recipe/seed/LR; no pooling")
            eligible = (row["status"] == "complete" and not issues
                        and finite(row["endpoint_bank_nll"]) and finite(row.get(score_field)))
            candidates.append(dict(uid=run["uid"], lr=run["config"].get("lr"), status=row["status"],
                eligible=eligible, issues=issues, bank_nll=row["endpoint_bank_nll"],
                full_nll=row["full_validation_nll"], selection_nll=row.get(score_field)))
        valid = [c for c in candidates if c["eligible"]]
        winner = min(valid, key=lambda c: (c["selection_nll"], c["lr"])) if valid else None
        rates = sorted({c["lr"] for c in candidates if finite(c["lr"])})
        minimizing_rates = ([c["lr"] for c in valid if c["selection_nll"] == winner["selection_nll"]]
                            if winner else [])
        touches_boundary = bool(minimizing_rates and any(rate in (rates[0], rates[-1]) for rate in minimizing_rates))
        complete = bool(candidates) and all(c["eligible"] for c in candidates)
        selections.append(dict(selection_id=f"lr_{index:03d}", recipe=json.loads(key), family=members[0]["family"],
            candidates=candidates, selected=winner, grid_complete=complete, provisional=not complete,
            selection_metric=policy, selection_basis=selection_basis(policy),
            minimizing_rates=minimizing_rates, boundary_lr=touches_boundary,
            bracketed=bool(winner and len(rates) >= 3 and not touches_boundary)))
    return selections


def select_momenta(selections):
    groups = {}
    for selection in selections:
        key = canonical(recipe(selection["recipe"], {"momentum"}))
        groups.setdefault(key, []).append(selection)
    result = []
    for key, members in sorted(groups.items()):
        valid = [s for s in members if s["selected"]]
        winner = min(valid, key=lambda s: (s["selected"]["selection_nll"], s["recipe"]["momentum"])) if valid else None
        momenta = sorted({s["recipe"]["momentum"] for s in members})
        result.append(dict(recipe=json.loads(key), family=members[0]["family"],
            selection_metric=selection_policy(members[0]["recipe"]),
            selection_basis=selection_basis(selection_policy(members[0]["recipe"])),
            tested_momenta=momenta, selection_ids=[s["selection_id"] for s in members],
            selected=winner, provisional=any(not s["grid_complete"] for s in members),
            momentum_boundary=bool(winner and winner["recipe"]["momentum"] in (momenta[0], momenta[-1])),
            only_one_momentum=len(momenta) == 1))
    return result


def crossing(points, threshold):
    """First passage interpolated only inside an observed adjacent interval.

    Left-censor thresholds already beaten at the first observation. Right-censor
    thresholds never reached. Equality at the first observation is observed;
    its zero-step ratio is still undefined. Exact hits at later observations
    retain the preceding bracket: the unseen crossing time is not exact.
    Brackets describe observed transitions, not confidence intervals; an
    earlier unobserved dip cannot be excluded. No smoothing is applied.
    """
    if not points:
        return dict(status="missing", step=None, tokens=None, bracket_steps=None, bracket_tokens=None)
    if threshold > points[0]["validation_nll"]:
        return dict(status="left_censored", step=None, tokens=None,
                    bracket_steps=[None, points[0]["step"]], bracket_tokens=[None, points[0]["tokens"]])
    for i, right in enumerate(points):
        if right["validation_nll"] <= threshold:
            if i == 0:
                return dict(status="observed", step=right["step"], tokens=right["tokens"],
                            bracket_steps=[right["step"], right["step"]],
                            bracket_tokens=[right["tokens"], right["tokens"]],
                            bracket_nll=[right["validation_nll"], right["validation_nll"]])
            left = points[i - 1]
            fraction = (left["validation_nll"] - threshold) / (left["validation_nll"] - right["validation_nll"])
            return dict(status="interpolated" if 0 < fraction < 1 else "observed",
                        **{axis: left[axis] + fraction * (right[axis] - left[axis]) for axis in ("step", "tokens")},
                        bracket_steps=[left["step"], right["step"]],
                        bracket_tokens=[left["tokens"], right["tokens"]],
                        bracket_nll=[left["validation_nll"], right["validation_nll"]])
    return dict(status="right_censored", step=None, tokens=None,
                last_observed_step=points[-1]["step"], last_observed_tokens=points[-1]["tokens"],
                bracket_steps=[points[-1]["step"], None], bracket_tokens=[points[-1]["tokens"], None])


def ratio_interval(numerator, denominator):
    """Observation-resolution bounds; never extrapolate a censored crossing."""
    result = dict(lower=None, upper=None, upper_unbounded=False, status="censored_or_missing")
    a, b = numerator.get("bracket_steps"), denominator.get("bracket_steps")
    if not a or not b or not all(finite(value) for value in [*a, *b]):
        return result
    if b[1] <= 0:
        return dict(result, status="undefined_zero_denominator")
    result["lower"] = a[0] / b[1]
    if b[0] == 0:
        if a[1] == 0:
            return dict(result, status="undefined_zero_over_zero")
        return dict(result, upper_unbounded=True, status="upper_unbounded")
    return dict(result, upper=a[1] / b[0], status="bounded")


def ratio_record(muon, spd, loss):
    left, right = crossing(muon["evaluations"], loss), crossing(spd["evaluations"], loss)
    m, s = left["step"], right["step"]
    valid = finite(m) and finite(s) and m > 0 and s > 0
    return dict(loss=loss, muon_crossing=left, spd_crossing=right,
                muon_steps_per_spd_step=m / s if valid else None,
                spd_steps_per_muon_step=s / m if valid else None,
                muon_steps_per_spd_step_interval=ratio_interval(left, right),
                spd_steps_per_muon_step_interval=ratio_interval(right, left))


def ratio_text(record):
    value = record["muon_steps_per_spd_step"]
    interval = record["muon_steps_per_spd_step_interval"]
    estimate = f"{value:.4f}" if value is not None else "undefined"
    if interval["status"] == "bounded":
        return estimate + f" [{interval['lower']:.4f}, {interval['upper']:.4f}]"
    if interval["upper_unbounded"]:
        return estimate + f" [{interval['lower']:.4f}, ∞)"
    return estimate + f" ({interval['status']})"


def bracket_text(record):
    bracket = record.get("bracket_steps")
    if bracket is None:
        return "missing"
    return "[" + ", ".join(f"{value:g}" if finite(value) else "?" for value in bracket) + "]"


def compare_curves(runs, selections, momentum_selections):
    by_uid = {r["uid"]: r for r in runs}
    momentum_audit = {s["selected"]["selection_id"]: s for s in momentum_selections if s["selected"]}
    modes = {"fixed_momentum_0.9": [(s, s["provisional"]) for s in selections
        if s["recipe"].get("momentum") == .9],
        "best_tested_momentum": [(s["selected"], s["provisional"]) for s in momentum_selections if s["selected"]]}
    comparisons, missing = [], []
    for mode, selected in modes.items():
        groups = {}
        for choice, provisional in selected:
            cfg = choice["recipe"]
            if cfg["method"] not in ("muon", "spd"):
                continue
            key = canonical(recipe(cfg, {"method", "momentum"}))
            groups.setdefault(key, {})[cfg["method"]] = (choice, provisional)
        for key, methods in sorted(groups.items()):
            if set(methods) != {"muon", "spd"} or any(not s["selected"] for s, _ in methods.values()):
                missing.append(dict(mode=mode, recipe=json.loads(key), reason="missing eligible complete Muon/SOAP-PD pair"))
                continue
            m_choice, s_choice = methods["muon"][0], methods["spd"][0]
            muon, spd = (by_uid[c["selected"]["uid"]] for c in (m_choice, s_choice))
            lo = max(min(p["validation_nll"] for p in r["evaluations"]) for r in (muon, spd))
            hi = min(r["evaluations"][0]["validation_nll"] for r in (muon, spd))
            grid = {lo + (hi-lo)*i/200 for i in range(201)} if hi >= lo else set()
            grid.update(p["validation_nll"] for r in (muon, spd) for p in r["evaluations"] if lo <= p["validation_nll"] <= hi)
            grid.update(loss for loss in THRESHOLDS if lo <= loss <= hi)
            comparisons.append(dict(mode=mode, family=muon["family"], recipe=json.loads(key),
                selection_metric=m_choice["selection_metric"], selection_basis=m_choice["selection_basis"],
                trajectory_metric="fixed-bank validation NLL",
                provisional=any(p for _, p in methods.values()), muon_uid=muon["uid"], spd_uid=spd["uid"],
                muon_lr=m_choice["selected"]["lr"], spd_lr=s_choice["selected"]["lr"],
                muon_momentum=m_choice["recipe"]["momentum"], spd_momentum=s_choice["recipe"]["momentum"],
                muon_tested_momenta=momentum_audit.get(m_choice["selection_id"], {}).get("tested_momenta") if mode.startswith("best") else [.9],
                spd_tested_momenta=momentum_audit.get(s_choice["selection_id"], {}).get("tested_momenta") if mode.startswith("best") else [.9],
                endpoint_bank_spd_minus_muon=spd["row"]["endpoint_bank_nll"]-muon["row"]["endpoint_bank_nll"],
                endpoint_full_spd_minus_muon=spd["row"]["full_validation_nll"]-muon["row"]["full_validation_nll"],
                shared_loss_span=[lo, hi] if hi >= lo else None,
                threshold_results=[ratio_record(muon, spd, loss) for loss in THRESHOLDS],
                whole_shared_span=[ratio_record(muon, spd, loss) for loss in sorted(grid, reverse=True)]))
    comparisons.sort(key=lambda c: (c["family"], c["mode"], c["recipe"]["batch_tokens"]))
    return comparisons, missing


def plot_report(selections, comparisons, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = []
    for family in sorted({s["family"] for s in selections}):
        members = [s for s in selections if s["family"] == family]
        policies = {s["selection_metric"] for s in members}
        if len(policies) != 1:
            raise ValueError("Plot family mixes incompatible selection policies")
        policy = next(iter(policies))
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for method in sorted({s["recipe"]["method"] for s in members}):
            for batch in sorted({s["recipe"]["batch_tokens"] for s in members}):
                line = sorted([s for s in members if s["recipe"]["method"] == method
                               and s["recipe"]["batch_tokens"] == batch and s["selected"]],
                              key=lambda s: s["recipe"]["momentum"])
                if not line:
                    continue
                provisional = any(s["provisional"] for s in line)
                label = f"{method} b={batch}" + (" (provisional)" if provisional else "")
                for axis, field in zip(axes, ("bank_nll", "full_nll")):
                    axis.plot([s["recipe"]["momentum"] for s in line], [s["selected"][field] for s in line],
                              marker="o", linestyle="--" if provisional else "-", label=label)
        titles = ("Final fixed bank" + (" (LR selection)" if policy == "bank" else " (reported diagnostic)"),
                  "Full development split" + (" (LR selection)" if policy == "full" else " (reported diagnostic)"))
        for ax, title in zip(axes, titles):
            ax.set(xlabel="Momentum β", ylabel="Development validation NLL", title=title)
            ax.grid(alpha=.2)
        if axes[0].lines:
            axes[0].legend(fontsize=7)
        fig.suptitle(f"{family}: best tested LR per β by {policy}; one seed, development evidence")
        fig.tight_layout()
        name = f"momentum_{family}.png"
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        paths.append(name)
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, mode in zip(axes, ("fixed_momentum_0.9", "best_tested_momentum")):
            for comparison in comparisons:
                if comparison["family"] != family or comparison["mode"] != mode:
                    continue
                rows = comparison["whole_shared_span"]
                label = (f"b={comparison['recipe']['batch_tokens']}, "
                         f"β M/S={comparison['muon_momentum']:g}/{comparison['spd_momentum']:g}")
                if mode.startswith("best") and any(len(comparison[k]) == 1 for k in ("muon_tested_momenta", "spd_tested_momenta")):
                    label += " (single β available)"
                if comparison["provisional"]:
                    label += " (provisional)"
                ax.plot([r["loss"] for r in rows], [r["muon_steps_per_spd_step"] for r in rows], label=label)
            for threshold in THRESHOLDS:
                ax.axvline(threshold, color="gray", alpha=.25, linewidth=.8)
            ax.axhline(1, color="black", linewidth=.8)
            ax.invert_xaxis()
            ax.set(xlabel="Common fixed-bank validation loss (first passage)", ylabel="Muon steps / SOAP-PD steps",
                   title=("β = 0.9; LR selected by " if mode.startswith("fixed") else "Best tested β and LR selected by ") + policy)
            ax.grid(alpha=.2)
            if any(line.get_label()[0] != "_" for line in ax.lines):
                ax.legend(fontsize=7)
        fig.suptitle(f"{family}: development curves; >1 favors SOAP-PD; interpolation only (resolution bounds in report)")
        fig.tight_layout()
        name = f"step_equivalent_{family}.png"
        fig.savefig(out / name, dpi=150)
        plt.close(fig)
        paths.append(name)
    return paths


def analyze(cohorts, out):
    cohorts, out = [Path(c).resolve() for c in cohorts], Path(out).resolve()
    for cohort in cohorts:
        relative = out.relative_to(cohort).parts if out.is_relative_to(cohort) else ()
        if out == cohort or out in cohort.parents or any(part in {"configs", "runs", "frozen"} for part in relative):
            raise ValueError("Output must not overlap cohort inputs or artifacts")
    marker = out / "dynamics_manifest.json"
    if marker.is_symlink():
        raise ValueError("Refusing a symlink report-ownership manifest")
    if out.exists() and any(out.iterdir()) and not marker.is_file():
        raise FileExistsError("Nonempty output lacks this report's ownership manifest")
    previous = json.loads(marker.read_text()) if marker.exists() else {}
    if previous and previous.get("generator") != GENERATOR:
        raise ValueError("Output belongs to another generator")
    runs, aliases = load_runs(cohorts)
    # Resolved symlink destinations need the same protection as input paths.
    for run in runs:
        actual_cohort = Path(run["uid"]).parent.parent
        relative = out.relative_to(actual_cohort).parts if out.is_relative_to(actual_cohort) else ()
        if out == actual_cohort or out in actual_cohort.parents or any(part in {"runs", "configs", "frozen"} for part in relative):
            raise ValueError("Output overlaps resolved cohort inputs or artifacts")
    pairing = audit_pairing(runs)
    selections = select_lrs(runs)
    momenta = select_momenta(selections)
    comparisons, missing = compare_curves(runs, selections, momenta)
    out.mkdir(parents=True, exist_ok=True)
    # Refuse collisions with unknown files, including plots that a prior report
    # did not generate. Refresh only files claimed by our previous manifest.
    anticipated = {"README.md", "runs.json", "runs.csv", "pairing.json", "selection.json", "ratios.json", "thresholds.csv"}
    for family in {s["family"] for s in selections}:
        anticipated.update({f"momentum_{family}.png", f"step_equivalent_{family}.png"})
    owned = set(previous.get("generated_files", []))
    for name in anticipated | {marker.name}:
        path = out / name
        if path.is_symlink() or (path.exists() and name not in owned and name != marker.name):
            raise FileExistsError(f"Refusing to replace unowned output: {path}")
    atomic_json(out / "runs.json", runs)
    rows = [dict(uid=r["uid"], family=r["family"], **r["row"]) for r in runs]
    write_csv(out / "runs.csv", rows, list(rows[0]))
    atomic_json(out / "pairing.json", pairing)
    atomic_json(out / "selection.json", dict(rule="Minimize declared endpoint development NLL, per seed and exact recipe: selection_metric='bank' (default) or 'full'. LR and best momentum use the same policy; policies are never pooled.",
        learning_rates=selections, best_tested_momenta=momenta))
    atomic_json(out / "ratios.json", dict(rule="First observed passage; adjacent linear interpolation only. Zero-step and censored estimates are null. "
        "Observation-resolution bounds are [M_left/S_right, M_right/S_left]; a zero lower denominator yields an unbounded/null upper bound. "
        "These are not statistical confidence intervals and do not exclude an earlier unobserved crossing/rebound; no smoothing or monotonicity assumption.",
        evaluation_role="development (including both fixed bank and full split)",
        fixed_thresholds=THRESHOLDS, comparisons=comparisons, unavailable_comparisons=missing))
    thresholds = [dict(family=c["family"], mode=c["mode"], batch_tokens=c["recipe"]["batch_tokens"],
        selection_metric=c["selection_metric"],
        provisional=c["provisional"], muon_uid=c["muon_uid"], spd_uid=c["spd_uid"], **r)
        for c in comparisons for r in c["threshold_results"]]
    write_csv(out / "thresholds.csv", thresholds, list(thresholds[0]) if thresholds else ["family", "mode", "batch_tokens", "loss"])
    plots = plot_report(selections, comparisons, out)
    counts = dict(Counter(r["row"]["status"] for r in runs))
    lines = ["# Tiny surrogate: batch and momentum evidence", "",
        f"All {len(runs)} unique configured runs are retained: {canonical(counts)}. "
        "Saved artifact statuses are a snapshot, not a live scheduler check.", "",
        "**All current scores are development evidence, including the full split; neither is an untouched evaluation panel.** "
        "Exploratory discovery only. No seed pooling: different seeds, architectures, budgets, and other recipe choices form separate families. "
        "Observation cadence (eval_every) may differ and is retained per run; it is not a training recipe change in this dropout-free runner. "
        "Every LR is selected by the declared final development metric: `selection_metric=bank` (default) uses the fixed bank; "
        "`selection_metric=full` uses the full development split. Both scores are displayed, with explicit selection basis. "
        "Different selection policies form separate families. "
        "Failed, unfinished, inconsistent, or provenance-mismatched runs are ineligible. An unfinished/failed LR grid leaves its winner provisional. "
        "Best tested momentum uses that same declared endpoint metric and is separately labeled; this is not independent confirmation.", "",
        "| Family | Method | Batch | β | Selection basis | Selected LR | Bank NLL | Full NLL | Grid / bracket |",
        "|---|---|---:|---:|---|---:|---:|---:|---|"]
    for s in selections:
        c, w = s["recipe"], s["selected"]
        status = ("provisional" if s["provisional"] else "complete") + ("; interior" if s["bracketed"] else "; unbracketed")
        prefix = f"| {s['family']} | {c['method']} | {c['batch_tokens']} | {c['momentum']:g} | {s['selection_metric']} |"
        lines.append(prefix + (f" {w['lr']:g} | {w['bank_nll']:.6f} | {w['full_nll']:.6f} | {status} |" if w else " — | — | — | no eligible candidate |"))
    lines += ["", "## Fixed common thresholds", "",
        "The original thresholds 2.4, 2.2, and 2.0 remain in this table and `thresholds.csv`, including censoring. "
        "Ratios are Muon steps / SOAP-PD steps (>1 favors SOAP-PD); `ratios.json` also stores the reciprocal. "
        "Trajectories and common-loss crossings use the recorded fixed-bank validation NLL, even when endpoint recipe selection uses the full development split. "
        "Curves show the whole shared observed loss span, including all evaluation loss knots and a 201-point grid. "
        "No extrapolation, best-threshold selection, or monotonic smoothing is used. Sparse-evaluation interpolation and first crossings are descriptive. "
        "Endpoint-tuned recipes are not reselected per threshold. Each table cell reports the interpolation followed by the observation-resolution interval "
        "[M_left/S_right, M_right/S_left]. An upper bound of ∞ means SOAP-PD's lower step bound is zero. "
        "These are bounds for the observed crossing at the available evaluation resolution, **not statistical confidence intervals** or guarantees against an earlier "
        "unobserved crossing/rebound. Exact threshold hits at later evaluations retain the preceding bracket. Curves may be nonmonotone. "
        "Different evaluation cadences (for example, every 128 versus 8 steps) leave different step resolution; "
        "**coarse or overlapping intervals do not prove batch monotonicity**. Plot lines retain the interpolated values; intervals and observed brackets are shown below.", "",
        "| Family | Selection | Batch | β Muon / SOAP-PD | Full NLL Δ SOAP-PD−Muon | Loss 2.4 | Loss 2.2 | Loss 2.0 |",
        "|---|---|---:|---|---:|---:|---:|---:|"]
    for c in comparisons:
        values = [ratio_text(r) for r in c["threshold_results"]]
        label = c["mode"] + f" (selected by {c['selection_metric']})" + (" (provisional)" if c["provisional"] else "")
        if c["mode"].startswith("best") and any(len(c[k]) == 1 for k in ("muon_tested_momenta", "spd_tested_momenta")):
            label += " (single β available)"
        lines.append(f"| {c['family']} | {label} | {c['recipe']['batch_tokens']} | {c['muon_momentum']:g} / {c['spd_momentum']:g} | {c['endpoint_full_spd_minus_muon']:+.6f} | " + " | ".join(values) + " |")
    lines += ["", "Observed crossing step brackets (unknown censored limits are `?`; token/loss brackets are retained in JSON/CSV):", "",
              "| Family | Selection | Batch | Loss | Muon bracket | SOAP-PD bracket |",
              "|---|---|---:|---:|---|---|"]
    for c in comparisons:
        for r in c["threshold_results"]:
            lines.append(f"| {c['family']} | {c['mode']} | {c['recipe']['batch_tokens']} | {r['loss']:g} | "
                         f"{bracket_text(r['muon_crossing'])} | {bracket_text(r['spd_crossing'])} |")
    lines += ["", "`runs.json` preserves every config origin, resolved run path, metadata, file hashes, frozen execution source hashes, issues, and validation curve. "
              "`selection.json` preserves every candidate and boundary/incomplete flags. `pairing.json` records the hash vetoes. "
              "Corpus/data identity uses the tokenized-data manifest hash when present, otherwise the legacy text hash. "
              "Bank-length hashes are checked if any peer records them; all-absent historical character groups remain compatible. "
              "Different family plots cannot be read as controlled batch comparisons. More favorable single-seed endpoints do not qualify the surrogate.", ""]
    lines += [f"![{name}]({name})\n" for name in plots]
    atomic_text(out / "README.md", "\n".join(lines))
    generated = sorted(anticipated)
    for name in owned - set(generated):
        if Path(name).name == name and (out / name).is_file() and not (out / name).is_symlink():
            (out / name).unlink()
    atomic_json(marker, dict(generator=GENERATOR, generated_unix=time.time(), cohorts=list(map(str, cohorts)),
        generated_files=generated, counts=counts, aliases=aliases, script_sha256=sha256(Path(__file__))))
    return dict(out=str(out), counts=counts, families=len(pairing), comparisons=len(comparisons))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", nargs="+", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.cohorts, args.out), indent=2))


if __name__ == "__main__":
    main()

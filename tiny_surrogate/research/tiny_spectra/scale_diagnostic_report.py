"""Read every arm of the frozen exposure/batch diagnostic; no model inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch


CORE = ["research/adamw_spectra/" + name + ".py" for name in
        ("model", "muon", "data_norm_muon")]
CORE += ["research/tiny_spectra/" + name + ".py" for name in
         ("train", "model", "optim", "stories", "data")]


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def normalize_config(cfg):
    return {key: str(Path(value).resolve()) if key == "data_path" else value
            for key, value in cfg.items() if key not in
            ("method", "lr", "run_id", "batch_tokens", "total_tokens", "validation_tokens")}


def load_runs(cohort, original):
    plan = read(cohort / "PLAN.json")
    if digest(cohort / "PLAN.json") != read(cohort / "PREFLIGHT.json")["plan_sha256"]:
        raise ValueError("Frozen diagnostic plan changed")
    if (plan["methods"] != ["ts", "spd"] or plan["learning_rates"] != [.004, .008, .016]
            or plan["qualification"] or plan["sealed_test_scoring"]):
        raise ValueError("Unexpected diagnostic plan")
    sources, control_sources = (read(root / "source_manifest.json") for root in (cohort, original))
    for relative, expected in sources.items():
        if digest(cohort / "frozen" / relative) != expected:
            raise ValueError("Frozen source changed: " + relative)
    if any(sources[k] != control_sources[k] for k in CORE):
        raise ValueError("Numerical sources differ from C")
    template_path = original / "runs/ts_b16384_lr0.008_m0.8_s20260928"
    template = read(template_path / "config.json")
    control_summary = read(template_path / "summary.json")
    old_windows = torch.load(template_path / "windows.pt", map_location="cpu", weights_only=True)
    expected_pairs = {(method, lr) for method in plan["methods"] for lr in plan["learning_rates"]}
    runs, pairing, population = {}, None, None
    expected_steps = sorted({0, 1, plan["steps"], *range(8, plan["steps"] + 1, 8)})
    for config_path in sorted((cohort / "configs").glob("*.json")):
        cfg = read(config_path)
        key = (cfg["method"], cfg["lr"])
        if key not in expected_pairs or key in runs:
            raise ValueError("Unexpected or duplicate diagnostic arm")
        if (normalize_config(cfg) != normalize_config(template)
                or cfg["batch_tokens"] != plan["batch_tokens"]
                or cfg["total_tokens"] != plan["total_tokens"]
                or cfg["validation_tokens"] != 10610 * 128):
            raise ValueError("Unplanned configuration change")
        root = cohort / "runs" / cfg["run_id"]
        summary, status, metadata = (read(root / filename) for filename in
                                    ("summary.json", "status.json", "metadata.json"))
        if summary["status"] != "complete" or status["status"] != "complete":
            raise ValueError("All six arms must complete before the diagnostic readout")
        if read(root / "config.json") != cfg:
            raise ValueError("Executed configuration differs from frozen arm")
        if (summary["steps"] != plan["steps"] or summary["tokens"] != plan["total_tokens"]
                or summary["initial_parameter_sha256"] != control_summary["initial_parameter_sha256"]
                or summary["hardware"] != control_summary["hardware"]
                or not 0 < summary["hardware"]["total_memory_bytes"] < 45 * 1024**3):
            raise ValueError("Horizon/initialization/hardware mismatch")
        if (metadata["corpus"]["role"] != "training_and_development_only"
                or metadata["validation_bank_target_count"] != 1018977
                or metadata["full_development_target_count"] != 1018977
                or metadata["source_file"] != str((cohort / "frozen/research/tiny_spectra/train.py").resolve())):
            raise ValueError("Wrong data role, evaluation coverage, or training source")
        current_pairing = [metadata[k] for k in ("initial_parameter_sha256", "train_window_sha256",
                           "validation_bank_sha256", "validation_bank_lengths_sha256", "data_manifest_sha256")]
        if pairing is None:
            pairing = current_pairing
        elif current_pairing != pairing:
            raise ValueError("Cross-arm pairing mismatch")
        windows = torch.load(root / "windows.pt", map_location="cpu", weights_only=True)
        validation = torch.load(root / "final_validation.pt", map_location="cpu", weights_only=True)
        if (not torch.equal(windows["validation"], validation["starts"])
                or len(windows["train"]) != 257343 or len(windows["train"].unique()) != 257343
                or not torch.equal(windows["train"][:len(old_windows["train"])], old_windows["train"])):
            raise ValueError("Full evaluation, fresh training, or paired prefix mismatch")
        weights, losses = validation["target_counts"], validation["sequence_nll"]
        if int(weights.sum()) != 1018977 or not torch.isfinite(losses).all() or (weights <= 0).any():
            raise ValueError("Invalid saved full-development targets or losses")
        nll = float((weights * losses).sum() / weights.sum())
        if any(abs(nll - summary[k]) > 1e-12 for k in ("full_validation_nll", "final_validation_nll")):
            raise ValueError("Saved windows do not reconstruct full endpoint")
        documents = read(root / "development_documents.json")["documents"]
        identities = [(doc["identity"], doc["targets"]) for doc in documents]
        if population is None:
            population = identities
        elif population != identities:
            raise ValueError("Document populations differ")
        if len(documents) != 4957 or abs(sum(d["nll"] * d["targets"] for d in documents) / 1018977 - nll) > 1e-12:
            raise ValueError("Documents do not reconstruct the full population mean")
        groups = {}
        for group_id in range(4):
            subset = [d for d in documents if int(d["identity"][:2], 16) % 4 == group_id]
            groups[group_id] = sum(d["nll"] * d["targets"] for d in subset) / sum(d["targets"] for d in subset)
        rows = [json.loads(line) for line in (root / "metrics.jsonl").read_text().splitlines() if line.strip()]
        curve = {row["step"]: row["validation_nll"] for row in rows if "validation_nll" in row}
        if sorted(curve) != expected_steps or not all(math.isfinite(x) for x in curve.values()):
            raise ValueError("Full-development trajectory is incomplete or nonfinite")
        if abs(curve[plan["steps"]] - nll) > 1e-12:
            raise ValueError("Final curve point differs from reconstructed full endpoint")
        runs[key] = dict(run_id=cfg["run_id"], method=key[0], lr=key[1], full_development_nll=nll,
            groups=groups, curve=curve, total_seconds=summary["total_seconds"],
            peak_cuda_allocated_bytes=summary["peak_cuda_allocated_bytes"],
            root=str(root.resolve()), artifacts={name: digest(root / name) for name in
                ("config.json", "metadata.json", "summary.json", "windows.pt", "final_validation.pt",
                 "development_documents.json", "metrics.jsonl")}, trajectory=rows)
    if set(runs) != expected_pairs:
        raise ValueError("Every configured arm must be present")
    return plan, runs


def report(cohort, original, out):
    cohort, original = cohort.resolve(), original.resolve()
    plan, runs = load_runs(cohort, original)
    methods, rates = plan["methods"], plan["learning_rates"]
    best = {m: min((row for (method, _), row in runs.items() if method == m),
                   key=lambda row: row["full_development_nll"]) for m in methods}
    prior = {m: read(original / "runs" / f"{m}_b16384_lr0.008_m0.8_s20260928/summary.json")["full_validation_nll"]
             for m in methods}
    absolute = {m: best[m]["full_development_nll"] - prior[m] for m in methods}
    matched = {lr: runs["spd", lr]["full_development_nll"] - runs["ts", lr]["full_development_nll"] for lr in rates}
    endpoint = best["spd"]["full_development_nll"] - best["ts"]["full_development_nll"]
    late = {step: min(runs["spd", lr]["curve"][step] for lr in rates)
                  - min(runs["ts", lr]["curve"][step] for lr in rates) for step in plan["late_steps"]}
    groups = {group: best["spd"]["groups"][group] - best["ts"]["groups"][group] for group in range(4)}
    gates = dict(absolute_progress_both=all(value <= -.005 for value in absolute.values()),
                 two_material_matched_lrs=sum(value <= -.005 for value in matched.values()) >= 2,
                 both_minima_interior=all(
                     runs[m, .008]["full_development_nll"] < runs[m, edge]["full_development_nll"]
                     for m in methods for edge in (.004, .016)),
                 material_endpoint=endpoint <= -.005,
                 every_document_group_favors_spd=all(value < 0 for value in groups.values()),
                 late_direction_consistent=all(value < 0 for value in late.values()),
                 late_median_material=statistics.median(late.values()) <= -.005)
    passed = all(gates.values())
    result = dict(role="development_diagnostic_only", plan=plan,
        plan_sha256=digest(cohort / "PLAN.json"), gates=gates, diagnostic_gate_passed=passed,
        qualified_surrogate=False, sealed_test_scored=False,
        final_batch_targets=plan["total_tokens"] - (plan["steps"] - 1) * plan["batch_tokens"],
        selected={m: {k: best[m][k] for k in ("run_id", "lr", "full_development_nll")} for m in methods},
        absolute_changes_vs_original_c=absolute, matched_lr_spd_minus_ts=matched,
        endpoint_envelope_spd_minus_ts=endpoint, selected_recipe_document_group_differences=groups,
        late_envelope_spd_minus_ts=late,
        total_gpu_hours_from_run_seconds=sum(row["total_seconds"] for row in runs.values()) / 3600,
        runs=list(runs.values()), decision=(
            "Diagnostic passes only; require a prospective fresh training-data/horizon-control plan and fresh development-seed replication."
            if passed else "Diagnostic does not pass; no edge expansion or followup optimizer/architecture sweep under this diagnostic."),
        limitations=["One development seed and an adaptively selected regime, not independent confirmation.",
                     "Joint batch and exposure change cannot identify their separate effects.",
                     "LR envelopes switch recipes and are not realizable training trajectories or speedups.",
                     "Current fresh-data capacity cannot provide a doubled horizon from this base.",
                     "Five-method, batch, momentum, and sealed confirmation gates remain unsatisfied."])
    out.mkdir(parents=True, exist_ok=False)
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    colors = {lr: color for lr, color in zip(rates, ("tab:blue", "tab:orange", "tab:green"))}
    for (method, lr), row in runs.items():
        x = sorted(row["curve"])
        axes[0].plot([min(step * plan["batch_tokens"], plan["total_tokens"]) / 1e6 for step in x],
                     [row["curve"][step] for step in x], color=colors[lr],
                     ls="-" if method == "spd" else "--", label=f"{method}, LR {lr:g}")
    common_steps = sorted(next(iter(runs.values()))["curve"])
    for lr in rates:
        axes[1].plot([min(step * plan["batch_tokens"], plan["total_tokens"]) / 1e6 for step in common_steps],
                     [runs["spd", lr]["curve"][step] - runs["ts", lr]["curve"][step] for step in common_steps],
                     color=colors[lr], label=f"Matched LR {lr:g}")
    axes[1].axhline(0, color="black", lw=.7)
    axes[1].axvspan(.7 * plan["total_tokens"] / 1e6, .9 * plan["total_tokens"] / 1e6, color="gray", alpha=.15)
    for ax, title, ylabel in zip(axes, ("All full-development trajectories", "SOAP-PD minus TS at matched LR"),
                                ("NLL per BPE token", "NLL difference")):
        ax.set(title=title, xlabel="Training targets (millions)", ylabel=ylabel)
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.savefig(out / "trajectories.png", dpi=160)
    plt.close(fig)
    lines = ["# Exposure-and-batch diagnostic", "", result["decision"], "",
             "Development evidence only. The two original failed candidate verdicts remain unchanged.", "",
             "| LR | TS full NLL | SOAP-PD full NLL | SOAP-PD minus TS |", "|---:|---:|---:|---:|"]
    for lr in rates:
        lines.append(f"| {lr:g} | {runs['ts', lr]['full_development_nll']:.8f} | {runs['spd', lr]['full_development_nll']:.8f} | {matched[lr]:+.8f} |")
    lines += ["", "| Predeclared gate | Pass |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in gates.items()]
    lines += ["", f"Endpoint LR-envelope gap: {endpoint:+.8f}. Absolute changes versus original C: {absolute}.",
              "", f"The final update uses an underfilled batch of {result['final_batch_targets']} targets (129 contexts); the first326 updates use100992 targets each.",
              "", f"Selected-recipe document-group differences: {groups}.",
              "", f"Fixed late-step LR-envelope differences: {late}.",
              "", "Every curve and endpoint includes all1,018,977 real development targets. Saved window/document means, source hashes, stream prefixes and pairing were verified. Complete raw trajectories and every gate are retained in results.json.",
              "", "![All six trajectories and matched-LR differences](trajectories.png)", ""]
    lines += result["limitations"]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return {key: result[key] for key in ("gates", "diagnostic_gate_passed", "selected", "decision")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cohort", "original", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.cohort, args.original, args.out), indent=2))

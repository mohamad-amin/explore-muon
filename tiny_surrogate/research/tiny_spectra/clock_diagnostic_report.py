"""Read the preregistered two-arm C-clock diagnostic; no model inference."""
import argparse
import hashlib
import json
from pathlib import Path

import torch


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def report(control, diagnostic, out):
    roots = {}
    for label, cohort in (("control", control), ("diagnostic", diagnostic)):
        matches = []
        for config in (cohort / "configs").glob("*.json"):
            cfg = read(config)
            if cfg["method"] in ("ts", "spd") and cfg["lr"] == .008:
                matches.append((cfg["method"], cohort / "runs" / cfg["run_id"]))
        if sorted(m for m, _ in matches) != ["spd", "ts"]:
            raise ValueError("Exactly one TS and one SPD arm at the declared LR required")
        roots.update({(label, method): root for method, root in matches})
    records = {}
    reference_windows = reference_pairing = reference_documents = None
    for key, root in roots.items():
        cfg, summary, metadata, status = (read(root / name) for name in
            ("config.json", "summary.json", "metadata.json", "status.json"))
        if summary["status"] != "complete" or status["status"] != "complete":
            raise ValueError("Every diagnostic arm and preserved control must complete")
        if (summary["steps"] != 512 or summary["tokens"] != 8388608
                or cfg["selection_metric"] != "full" or cfg["seed"] != 20260928):
            raise ValueError("Run is outside the preregistered diagnostic")
        if metadata["corpus"]["role"] != "training_and_development_only":
            raise ValueError("Only development evidence may enter this report")
        if not 0 < summary["hardware"]["total_memory_bytes"] < 45 * 1024**3:
            raise ValueError("GPU restriction violated")
        pairing = [summary[k] for k in ("initial_parameter_sha256", "train_window_sha256")]
        pairing += [summary["hardware"], metadata["corpus"]]
        if reference_pairing is None:
            reference_pairing = pairing
        elif pairing != reference_pairing:
            raise ValueError("Initialization/data/stream/hardware do not match")
        windows = torch.load(root / "final_validation.pt", map_location="cpu", weights_only=True)
        denominator = int(windows["target_counts"].sum())
        if denominator != 1018977 or not torch.isfinite(windows["sequence_nll"]).all():
            raise ValueError("Invalid full-development losses/denominator")
        if reference_windows is None:
            reference_windows = windows
        elif any(not torch.equal(windows[k], reference_windows[k]) for k in
                 ("starts", "target_counts", "document_indices")):
            raise ValueError("Evaluation populations differ")
        nll = float((windows["sequence_nll"] * windows["target_counts"]).sum() / denominator)
        if abs(nll - summary["full_validation_nll"]) > 1e-12:
            raise ValueError("Saved per-window losses do not reconstruct full NLL")
        docs = read(root / "development_documents.json")["documents"]
        identities = [(doc["identity"], doc["targets"]) for doc in docs]
        if reference_documents is None:
            reference_documents = identities
        elif identities != reference_documents:
            raise ValueError("Document membership differs")
        doc_nll = sum(doc["nll"] * doc["targets"] for doc in docs) / denominator
        if abs(nll - doc_nll) > 1e-12:
            raise ValueError("Document losses do not reconstruct full NLL")
        sources = read(root.parents[1] / "source_manifest.json")
        core = {k: v for k, v in sources.items() if k in (
            "research/adamw_spectra/model.py", "research/adamw_spectra/muon.py",
            "research/adamw_spectra/data_norm_muon.py", "research/tiny_spectra/train.py",
            "research/tiny_spectra/model.py", "research/tiny_spectra/optim.py",
            "research/tiny_spectra/data.py", "research/tiny_spectra/stories.py")}
        if len(core) != 8:
            raise ValueError("Incomplete numerical source coverage")
        for relative, expected in core.items():
            if digest(root.parents[1] / "frozen" / relative) != expected:
                raise ValueError("Frozen numerical source changed")
        records[key] = dict(config=cfg, full_development_nll=nll,
            macro_document_nll=sum(doc["nll"] for doc in docs) / len(docs),
            groups=summary["development_groups"], core_sources=core,
            root=str(root.resolve()), artifacts={name: digest(root / name) for name in
            ("config.json", "summary.json", "metadata.json", "final_validation.pt",
             "development_documents.json", "metrics.jsonl")})
    for method in ("ts", "spd"):
        old, new = records["control", method], records["diagnostic", method]
        normalize = lambda cfg: {k: str(Path(v).resolve()) if k == "data_path" else v
                                for k, v in cfg.items() if k not in ("run_id", "cov_ema")}
        if (normalize(old["config"]) != normalize(new["config"])
                or old["config"]["cov_ema"] != .9
                or new["config"]["cov_ema"] != .998**128
                or old["core_sources"] != new["core_sources"]):
            raise ValueError("More than the declared covariance clock changed")
    losses = {k: row["full_development_nll"] for k, row in records.items()}
    absolute = {m: losses["diagnostic", m] - losses["control", m] for m in ("ts", "spd")}
    gaps = {l: losses[l, "spd"] - losses[l, "ts"] for l in ("control", "diagnostic")}
    difference = gaps["diagnostic"] - gaps["control"]
    material = difference <= -.005 and absolute["spd"] <= -.005
    group_deltas = []
    for i in range(4):
        group = {k: row["groups"][i] for k, row in records.items()}
        if any(g["group_id"] != i or g["targets"] != group["control", "ts"]["targets"]
               for g in group.values()):
            raise ValueError("Document group population mismatch")
        d = {m: group["diagnostic", m]["nll"] - group["control", m]["nll"] for m in ("ts", "spd")}
        group_deltas.append(dict(group_id=i, absolute_changes=d,
                                 difference_in_differences=d["spd"] - d["ts"]))
    result = dict(role="development_diagnostic_only", qualified_surrogate=False,
        complete_full_development_targets=1018977, documents=len(reference_documents),
        absolute_nll_changes=absolute, spd_minus_ts=gaps,
        difference_in_differences=difference, material_clock_gate_passed=material,
        decision=("Replicate this contrast on a fresh development seed before further claims."
                  if material else "Close the input-clock hypothesis at this setting; no nearby EMA sweep."),
        groups=group_deltas, runs={"/".join(k): v for k, v in records.items()},
        caveats=["Same development text and one seed; no independent generalization claim.",
                 "Document groups diagnose heterogeneity and are not training-seed replications.",
                 "The five-method ordering, batch and momentum gates remain unsatisfied."])
    out.mkdir(parents=True, exist_ok=False)
    (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    trajectories = {"/".join(key): [json.loads(line) for line in
        (root / "metrics.jsonl").read_text().splitlines() if line.strip()]
        for key, root in roots.items()}
    (out / "trajectories.json").write_text(json.dumps(trajectories, indent=2, allow_nan=False) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for label, rows in trajectories.items():
        color = "tab:blue" if label.endswith("/ts") else "tab:orange"
        style = "--" if label.startswith("control/") else "-"
        for ax, metric in zip(axes, ("validation_nll", "train_probe_nll")):
            observed = [row for row in rows if metric in row]
            ax.plot([row["tokens"] / 1e6 for row in observed],
                    [row[metric] for row in observed], color=color, ls=style, label=label)
    for ax, title in zip(axes, ("Fixed development bank (diagnostic)", "Fixed training probe")):
        ax.set(title=title, xlabel="Training targets (millions)", ylabel="NLL per BPE token")
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(out / "trajectories.png", dpi=160)
    plt.close(fig)
    lines = ["# Input covariance clock diagnostic", "", "Development evidence only; both failed candidate verdicts remain unchanged.", "",
             "| Method | Original C | Reference-derived clock | Change |",
             "|---|---:|---:|---:|"]
    for m in ("ts", "spd"):
        lines.append(f"| {m} | {losses['control', m]:.9f} | {losses['diagnostic', m]:.9f} | {absolute[m]:+.9f} |")
    lines += ["", f"Change in SOAP-PD minus TS: **{difference:+.9f}** (negative favors SOAP-PD).",
              "", result["decision"], "", "Both the relative gain and SOAP-PD's absolute gain must reach0.005 NLL to pass the declared diagnostic gate.",
              "", "All1,018,977 development targets from4,957 documents enter the endpoint. Saved windows and documents reproduce each mean; initialization, stream, hardware, population and numerical-source matching are checked.",
              "", "See results.json for every fixed document group and artifact hashes. No sealed test was scored.",
              "", "The complete saved trajectories are in [trajectories.json](trajectories.json). The plotted development curve uses the fixed bank; the decision above uses the full endpoint population. No checkpoint is selected from these curves.",
              "", "![All four development and training-probe trajectories](trajectories.png)"]
    (out / "README.md").write_text("\n".join(lines) + "\n")
    return {k: result[k] for k in ("absolute_nll_changes", "difference_in_differences", "material_clock_gate_passed", "decision")}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "diagnostic", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(report(args.control, args.diagnostic, args.out), indent=2))

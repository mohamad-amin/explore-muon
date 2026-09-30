"""Render reproducible static figures and descriptive summaries from a run."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .measure import QUANTILES
from .train import atomic_json


KINDS = ("q", "k", "v", "o", "up", "down")


def load_run(out, quantity="update"):
    out = Path(out)
    metadata = json.loads((out / "metadata.json").read_text())
    rows = [json.loads(p.read_text()) for p in sorted((out / "steps").glob("step*.json"))]
    if not rows or [r["step"] for r in rows] != list(range(rows[-1]["step"] + 1)):
        raise ValueError("Run has missing or out-of-order step records")
    if quantity not in ("update", "momentum"):
        raise ValueError("Unknown spectral quantity")
    folder = "spectra" if quantity == "update" else "spectra_momentum"
    if quantity == "momentum":
        for row in rows:
            row.pop("matrices", None)
            if "momentum_matrices" in row:
                row["matrices"] = row["momentum_matrices"]
    names = set(metadata["measured_parameters"])
    for row in rows:
        if "matrices" in row:
            if set(row["matrices"]) != names:
                raise ValueError(f"Missing matrix measurements at step {row['step']}")
            with np.load(out / folder / f"step{row['step']:06d}.npz") as data:
                if set(data.files) != names:
                    raise ValueError("Spectrum archive does not match matrix records")
                for name in names:
                    v = data[name]
                    if len(v) != min(metadata["measured_parameters"][name]) or not np.isfinite(v).all():
                        raise ValueError("Invalid stored singular-value spectrum")
                    if np.any(v < 0) or np.any(np.diff(v) > 0):
                        raise ValueError("Singular values must be nonnegative and descending")
    return metadata, rows


def analyze(out, snapshot_step=None, quantity="update"):
    out = Path(out)
    metadata, rows = load_run(out, quantity)
    folder = "spectra" if quantity == "update" else "spectra_momentum"
    label = ("Muon pre-NS momentum" if quantity == "momentum" else
             "Muon post-NS update" if metadata["config"].get("optimizer") == "muon" else
             "AdamW adaptive update")
    cfg = metadata["config"]
    measured = [r for r in rows if "matrices" in r]
    if not measured:
        raise ValueError("No spectra in this run (LR pilots intentionally disable them)")
    plots = out / ("plots" if quantity == "update" else "plots_momentum")
    plots.mkdir(exist_ok=True)
    names = list(metadata["measured_parameters"])
    depths = sorted({name.split(".")[0] for name in names})
    cooldown_tokens = metadata["budget_tokens"] * (1 - cfg["cooldown_fraction"])
    cooldown_step = cooldown_tokens / cfg["batch_tokens"] + 1
    colors = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, len(QUANTILES)))

    def panels():
        return plt.subplots(6, len(depths), figsize=(4 * len(depths), 16), squeeze=False)

    def finish(fig, filename, title):
        fig.suptitle(title)
        fig.tight_layout(rect=(0, 0, 1, .975))
        fig.savefig(plots / f"{filename}.png", dpi=150)
        fig.savefig(plots / f"{filename}.pdf")
        plt.close(fig)

    for axis_name in ("step", "tokens"):
        fig, axes = panels()
        for i, kind in enumerate(KINDS):
            for j, depth in enumerate(depths):
                name, ax = f"{depth}.{kind}", axes[i, j]
                for q, color in zip(QUANTILES, colors):
                    points = [(r[axis_name], r["matrices"][name]["quantiles"][str(q)]) for r in measured
                              if not r["matrices"][name]["zero_matrix"]]
                    if points:
                        x, y = zip(*points)
                        ax.plot(x, y, label=f"q={q}", color=color, linewidth=1)
                if axis_name == "step":
                    boundaries = (cfg["warmup_steps"], cooldown_step)
                else:
                    boundaries = (cfg["warmup_steps"] * cfg["batch_tokens"], cooldown_tokens)
                for boundary in boundaries:
                    if 0 < boundary <= rows[-1][axis_name]:
                        ax.axvline(boundary, color="gray", linestyle=":", linewidth=.7)
                ax.set(title=name, yscale="log", xlabel=axis_name)
                ax.grid(alpha=.15)
        axes[0, 0].legend(fontsize=7)
        finish(fig, f"quantiles_{axis_name}", f"{label}: normalized descending-rank quantiles")

    # Match the paper's median-over-time panel, retaining the other quantiles above.
    fig, axes = panels()
    for i, kind in enumerate(KINDS):
        for j, depth in enumerate(depths):
            name, ax = f"{depth}.{kind}", axes[i, j]
            points = [(r["step"], r["matrices"][name]["quantiles"]["0.5"]) for r in measured
                      if not r["matrices"][name]["zero_matrix"]]
            if points:
                ax.plot(*zip(*points), linewidth=1)
            ax.set(title=name, yscale="log", xlabel="step")
            for boundary in (cfg["warmup_steps"], cooldown_step):
                if 0 < boundary <= rows[-1]["step"]:
                    ax.axvline(boundary, color="gray", linestyle=":", linewidth=.7)
    finish(fig, "median", f"{label}: median singular value")

    selected = snapshot_step if snapshot_step is not None else measured[-1]["step"]
    selected_row = next((r for r in measured if r["step"] == selected), None)
    if selected_row is None:
        raise ValueError(f"No spectrum was recorded at step {selected}")
    with np.load(out / folder / f"step{selected:06d}.npz") as archive:
        spectra = {name: archive[name].copy() for name in names
                   if not selected_row["matrices"][name]["zero_matrix"]}
    for remove_top in (False, True):
        values = {name: v[1:] if remove_top else v for name, v in spectra.items()}
        maximum = max((v.max() for v in values.values() if len(v)), default=1.0)
        bins = np.linspace(0, maximum if maximum > 0 else 1.0, 51)
        fig, axes = panels()
        for i, kind in enumerate(KINDS):
            for j, depth in enumerate(depths):
                name, ax = f"{depth}.{kind}", axes[i, j]
                if name in values:
                    ax.hist(values[name], bins=bins, log=True)
                else:
                    ax.text(.5, .5, "zero update", ha="center", transform=ax.transAxes)
                ax.set(title=name, xlabel="normalized singular value", ylabel="count")
        suffix = "bulk" if remove_top else "full"
        finish(fig, f"spectrum_{suffix}_step{selected:06d}",
               f"{label} step {selected}: {suffix} spectrum (common bins; no bulk renormalization)")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    training = [r for r in rows if "train_nll" in r]
    validation = [r for r in rows if "validation_nll" in r]
    axes[0].plot([r["tokens"] for r in training], [r["train_nll"] for r in training], label="training")
    axes[0].plot([r["tokens"] for r in validation], [r["validation_nll"] for r in validation], label="validation")
    axes[0].set(xlabel="tokens", ylabel="NLL")
    axes[0].legend()
    axes[1].plot([r["step"] for r in training], [r["lr"] for r in training])
    axes[1].set(xlabel="step", ylabel="learning rate")
    axes[2].plot([r["step"] for r in training], [r["gradient_norm_before_clip"] for r in training])
    axes[2].axhline(cfg["grad_clip"], color="gray", linestyle=":")
    axes[2].set(xlabel="step", ylabel="gradient norm before clipping")
    finish(fig, "training", "Training health and schedule")

    windows = {}
    for lower, upper in ((1100, 1300), (1300, 1500)):
        records = [r for r in measured if lower <= r["step"] <= upper]
        windows[f"{lower}-{upper}"] = {
            "observed_steps": [r["step"] for r in records], "count": len(records),
            "complete_every_step_coverage": len(records) == upper - lower + 1,
            "cooldown_count": sum(r["tokens"] - r["batch_tokens"] >= cooldown_tokens for r in records),
            "mean_quantiles": {name: {str(q): float(np.mean(v)) if (v := [
                r["matrices"][name]["quantiles"][str(q)] for r in records
                if not r["matrices"][name]["zero_matrix"]]) else None for q in QUANTILES}
                for name in names}}
    status = json.loads((out / "status.json").read_text()) if (out / "status.json").exists() else {}
    checkpoint = (json.loads((out / "checkpoint.json").read_text())
                  if (out / "checkpoint.json").exists() else {})
    summary = {"step": rows[-1]["step"], "tokens": rows[-1]["tokens"],
               "run_status": status.get("status", "unknown"),
               "durable_checkpoint": checkpoint,
               "budget_complete": (status.get("status") == "complete" and
                                   status.get("step") == rows[-1]["step"] and
                                   rows[-1]["tokens"] == metadata["budget_tokens"]),
               "spectra_samples": len(measured), "snapshot_step": selected,
               "clipping_fraction": float(np.mean([r["gradient_clipped"] for r in training])),
               "training_seconds": sum(r["training_seconds"] for r in training),
               "measurement_seconds": sum(r["measurement_seconds"] for r in training),
               "validation_seconds": sum(r["validation_seconds"] for r in training),
               "windows": windows,
               "interpretation": "Descriptive windows only; no stabilization or cross-size scaling claim."}
    summary["spectral_quantity"] = quantity
    atomic_json(out / ("summary.json" if quantity == "update" else "summary_momentum.json"), summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path)
    parser.add_argument("--snapshot-step", type=int)
    parser.add_argument("--quantity", choices=("update", "momentum"), default="update")
    args = parser.parse_args()
    summary = analyze(args.out, args.snapshot_step, args.quantity)
    print(json.dumps({k: v for k, v in summary.items() if k != "windows"}, indent=2))


if __name__ == "__main__":
    main()

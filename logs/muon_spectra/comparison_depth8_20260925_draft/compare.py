"""Read-only, token-matched AdamW/Muon comparison with reproducible figures."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

from .analyze import KINDS, load_run
from .measure import QUANTILES
from .train import atomic_json


COLORS = {"AdamW update": "#2964B4", "Muon update": "#D66A16", "Muon momentum": "#64746B"}
LABELS = {"AdamW update": "AdamW adaptive update", "Muon update": "Muon post-NS update",
          "Muon momentum": "Muon pre-NS momentum"}
FOCUS = ("block04.o", "block06.q", "block08.up")
FOCUS_TITLES = ("Block 4 · attention O", "Block 6 · attention Q", "Block 8 · MLP up")


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(adam_path, muon_path, out):
    adam_path, muon_path, out = map(lambda p: Path(p).resolve(), (adam_path, muon_path, out))
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite comparison: {out}")
    a_meta, a_rows = load_run(adam_path)
    m_meta, m_rows = load_run(muon_path)
    _, momentum_rows = load_run(muon_path, "momentum")
    roots = {"AdamW": adam_path, "Muon": muon_path}
    metas = {"AdamW": a_meta, "Muon": m_meta}
    rows = {"AdamW": a_rows, "Muon": m_rows}
    controls = {}
    for field in ("initial_model_sha256", "parameter_count", "budget_tokens", "total_steps",
                  "train_manifest", "validation_manifest", "measured_parameters"):
        controls[field] = a_meta[field] == m_meta[field]
    for field in ("model", "seed", "batch_tokens", "microbatch_sequences", "validation_tokens",
                  "validation_every", "spectra_every", "warmup_steps", "cooldown_fraction",
                  "grad_clip", "weight_decay", "betas", "epsilon", "precision"):
        controls[f"config.{field}"] = a_meta["config"][field] == m_meta["config"][field]
    controls["frozen_model_source"] = a_meta["source_sha256"]["model.py"] == m_meta["source_sha256"]["model.py"]
    if not all(controls.values()):
        raise ValueError(f"Mismatched controls: {[k for k, v in controls.items() if not v]}")
    if a_meta["config"].get("optimizer", "adamw") != "adamw" or m_meta["config"].get("optimizer") != "muon":
        raise ValueError("Expected AdamW and Muon runs in that order")
    if a_meta["config"]["model"]["n_layer"] != 8:
        raise ValueError("This report's fixed representative panels are for eight layers")
    names = list(a_meta["measured_parameters"])
    total_steps = a_meta["total_steps"]
    budget = a_meta["budget_tokens"]
    cfg = a_meta["config"]
    cooldown_step = math.ceil(budget * (1 - cfg["cooldown_fraction"]) / cfg["batch_tokens"]) + 1
    expected = sorted({1, total_steps, *range(cfg["spectra_every"], total_steps + 1, cfg["spectra_every"])})
    statuses = {}
    for label, root in roots.items():
        status = json.loads((root / "status.json").read_text())
        checkpoint = json.loads((root / "checkpoint.json").read_text())
        if (status["status"] != "complete" or status["tokens"] != budget or status["step"] != total_steps
                or checkpoint["step"] != total_steps or rows[label][-1]["tokens"] != budget):
            raise ValueError(f"Run not complete: {label}")
        statuses[label] = status
    if [(r["step"], r["tokens"]) for r in a_rows] != [(r["step"], r["tokens"]) for r in m_rows]:
        raise ValueError("Step/token trajectories differ")
    measured = {"AdamW update": [r for r in a_rows if "matrices" in r],
                "Muon update": [r for r in m_rows if "matrices" in r],
                "Muon momentum": [r for r in momentum_rows if "matrices" in r]}
    spectral_csv = []
    energy_error = 0.0
    for label, records in measured.items():
        if [r["step"] for r in records] != expected:
            raise ValueError(f"Missing spectral samples: {label}")
        root = adam_path if label == "AdamW update" else muon_path
        folder = "spectra_momentum" if label == "Muon momentum" else "spectra"
        for row in records:
            with np.load(root / folder / f"step{row['step']:06d}.npz") as archive:
                for name in names:
                    stats, singular = row["matrices"][name], archive[name]
                    if stats["zero_matrix"]:
                        raise ValueError(f"Zero spectral object needs explicit plotting treatment: {label}/{name}")
                    energy = float(np.square(singular.astype(np.float64)).sum())
                    energy_error = max(energy_error, abs(energy - 1))
                    if abs(energy - 1) > 2e-4:
                        raise ValueError("Invalid normalized spectral energy")
                    for q in QUANTILES:
                        if not np.isclose(singular[math.ceil(q * len(singular)) - 1],
                                          stats["quantiles"][str(q)], rtol=1e-6, atol=1e-10):
                            raise ValueError("Stored quantile differs from archived singular value")
                    spectral_csv.append({"object": label, "step": row["step"], "tokens": row["tokens"],
                        "matrix": name, **{f"q{q}": stats["quantiles"][str(q)] for q in QUANTILES},
                        **{key: stats[key] for key in ("frobenius_norm", "adaptive_step_norm", "weight_decay_step_norm",
                                                       "top_one_energy", "stable_rank")}})
    val = {label: {r["step"]: r for r in records if "validation_nll" in r} for label, records in rows.items()}
    if set(val["AdamW"]) != set(val["Muon"]):
        raise ValueError("Validation steps differ")
    validation = [{"step": step, "tokens": val["AdamW"][step]["tokens"],
                   "adamw_nll": val["AdamW"][step]["validation_nll"],
                   "muon_nll": val["Muon"][step]["validation_nll"],
                   "muon_minus_adamw": val["Muon"][step]["validation_nll"] - val["AdamW"][step]["validation_nll"]}
                  for step in sorted(val["AdamW"])]
    final_values = {}
    final_csv = []
    for label in measured:
        root = adam_path if label == "AdamW update" else muon_path
        folder = "spectra_momentum" if label == "Muon momentum" else "spectra"
        with np.load(root / folder / f"step{total_steps:06d}.npz") as archive:
            final_values[label] = {name: archive[name].copy() for name in names}
        for name, values in final_values[label].items():
            for rank, value in enumerate(values, 1):
                final_csv.append({"object": label, "matrix": name, "step": total_steps,
                                  "descending_rank": rank, "normalized_singular_value": float(value)})
    out.mkdir(parents=True)
    write_csv(out / "validation.csv", validation)
    write_csv(out / "spectral_metrics.csv", spectral_csv)
    write_csv(out / "final_spectra.csv", final_csv)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "medium", "axes.labelcolor": "#30343B",
                         "text.color": "#202832", "grid.alpha": .16,
                         "savefig.facecolor": "white", "pdf.fonttype": 42})
    book = PdfPages(out / "comparison.pdf")
    handles = [Line2D([0], [0], color=COLORS[label], lw=2,
                     linestyle="--" if label == "Muon momentum" else "-", label=LABELS[label])
               for label in measured]

    def finish(fig, name, title, caption, legend=True, large=False):
        fig.suptitle(title, fontsize=16 if not large else 20, x=.06, ha="left", y=.995)
        if legend:
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.54, .952 if not large else .97),
                       ncol=3, frameon=False, fontsize=10 if not large else 12)
        fig.text(.06, .014, caption, fontsize=9 if not large else 11, color="#505763", va="bottom")
        fig.tight_layout(rect=(.02, .09 if not large else .045, .99, .87 if not large else .942))
        fig.savefig(out / f"{name}.png", dpi=155)
        fig.savefig(out / f"{name}.pdf")
        book.savefig(fig)
        plt.close(fig)

    def timeline(ax):
        ax.axvspan(cooldown_step, total_steps, color="#C7CED7", alpha=.22, lw=0, zorder=0)
        ax.axvline(cfg["warmup_steps"], color="#A6ACB5", lw=.8, linestyle=":")
        ax.set_xlim(0, total_steps)
        ax.set_xlabel("Optimizer step")
        ax.grid(True, which="major")

    def trajectory(ax, name, field="median", labels=None):
        labels = labels or list(measured)
        for label in labels:
            records = measured[label]
            values = [(r["matrices"][name]["quantiles"]["0.5"] if field == "median"
                       else r["matrices"][name][field]) for r in records]
            if field == "top_one_energy":
                values = np.array(values) * 100
            ax.plot([r["step"] for r in records], values, color=COLORS[label], lw=1.8,
                    linestyle="--" if label == "Muon momentum" else "-")
        ax.set_yscale("log")
        timeline(ax)

    fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.8))
    xs = np.array([r["step"] for r in validation])
    for label, key in (("AdamW", "adamw_nll"), ("Muon", "muon_nll")):
        ys = np.array([r[key] for r in validation])
        for ax in axes[:2]:
            ax.plot(xs, ys, label=label, color=COLORS[label + " update"], lw=2)
    axes[0].set(title="Complete validation trajectory", ylabel="Validation NLL (nats/token)")
    axes[1].set(title="Late training · same data exposure", ylabel="Validation NLL (nats/token)")
    axes[2].plot(xs, [r["muon_minus_adamw"] for r in validation], color=COLORS["Muon update"], lw=2)
    axes[2].axhline(0, color="#737B88", lw=.9)
    axes[2].set(title="Muon − AdamW", ylabel="NLL difference · negative favors Muon")
    for ax in axes:
        timeline(ax)
    axes[1].set_xlim(700, total_steps)
    late_values = [r[key] for r in validation if r["step"] >= 700 for key in ("adamw_nll", "muon_nll")]
    axes[1].set_ylim(min(late_values) - .04, max(late_values) + .04)
    axes[0].legend(frameon=False)
    for label, key in (("AdamW", "adamw_nll"), ("Muon", "muon_nll")):
        axes[1].annotate(f"{label} {validation[-1][key]:.4f}",
                         (total_steps, validation[-1][key]), xytext=(-8, 9 if label == "AdamW" else -16),
                         textcoords="offset points", ha="right", color=COLORS[label + " update"], fontsize=10)
    finish(fig, "validation", "8 layers · Muon and AdamW validation loss",
           "Same initialization, width 512, global batch and 1.540B tokens. One seed. Shading marks final-10% cooldown.\n"
           "Recipes differ: AdamW LR 0.0012 throughout; Muon body LR 0.01 and auxiliary AdamW LR 0.002.", legend=False)

    fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.8))
    for ax, name, title in zip(axes, FOCUS, FOCUS_TITLES):
        trajectory(ax, name)
        ax.axhline(1 / math.sqrt(512), color="#B5BAC2", linestyle=":", lw=1, zorder=0)
        ax.set(title=title, ylabel="Median normalized singular value")
    finish(fig, "median_focus", "8 layers · spectral evolution at three fixed locations",
           "Each matrix is divided by its own Frobenius norm. Dashed green is momentum, before NS; solid curves are updates.\n"
           "Dotted horizontal line: equal singular values, 1/√512 ≈ 0.0442. LR and weight decay are excluded.")

    depths = (2, 4, 6, 8)
    for field, filename, title, caption in (
        ("median", "median_all24", "Median normalized singular value · all 24 matrices",
         "60 snapshots per object. Each matrix normalized separately; LR/decay excluded. Shading: final-10% cooldown."),
        ("top_one_energy", "top_energy_all24", "Energy in the leading singular direction · all 24 matrices",
         "100 × σ₁² / ||matrix||²_F. Lower values mean less concentration in one direction, not necessarily better learning."),
        ("adaptive_step_norm", "step_norm_all24", "Magnitude of the adaptive parameter step · all 24 matrices",
         "||LR × adaptive direction||_F, before weight decay and parameter-write rounding. Group learning rates differ."),
    ):
        fig, axes = plt.subplots(6, 4, figsize=(14, 17), sharex=True)
        for i, kind in enumerate(KINDS):
            for j, depth in enumerate(depths):
                name = f"block{depth:02d}.{kind}"
                labels = ("AdamW update", "Muon update") if field == "adaptive_step_norm" else None
                trajectory(axes[i, j], name, field, labels)
                axes[i, j].set_title(f"Block {depth} · {kind.upper()}")
                if i < 5:
                    axes[i, j].set_xlabel("")
                if field == "median":
                    axes[i, j].axhline(1 / math.sqrt(512), color="#B5BAC2", linestyle=":", lw=.8)
                if field == "top_one_energy":
                    axes[i, j].set_ylim(.1, 100)
        # Norm panels have no momentum curve; keep the legend faithful.
        previous_handles = handles
        if field == "adaptive_step_norm":
            handles = handles[:2]
        finish(fig, filename, title, caption, large=True)
        handles = previous_handles

    for focus in (True, False):
        if focus:
            fig, axes = plt.subplots(1, 3, figsize=(13.7, 4.8))
            items = list(zip(axes, FOCUS, FOCUS_TITLES))
        else:
            fig, axes = plt.subplots(6, 4, figsize=(14, 17), sharex=True, sharey=True)
            items = [(axes[i, j], f"block{depth:02d}.{kind}", f"Block {depth} · {kind.upper()}")
                     for i, kind in enumerate(KINDS) for j, depth in enumerate(depths)]
        for ax, name, title in items:
            for label in measured:
                values = final_values[label][name]
                ax.plot(np.arange(1, len(values) + 1), values, color=COLORS[label], lw=1.7,
                        linestyle="--" if label == "Muon momentum" else "-")
            ax.axhline(1 / math.sqrt(512), color="#B5BAC2", linestyle=":", lw=.8)
            ax.set(title=title, xlabel="Descending singular-value rank", yscale="log", xlim=(1, 512),
                   ylabel="Normalized singular value" if focus else None)
            ax.grid(True, which="major")
        finish(fig, "final_spectra_focus" if focus else "final_spectra_all24",
               f"Full spectra at the final update · step {total_steps}",
               "All 512 singular values are shown. Each matrix has unit Frobenius norm; leading singular values are retained.",
               large=not focus)
    book.close()

    endpoint = validation[-1]
    delta = endpoint["muon_minus_adamw"]
    ranges = {}
    late_window = {}
    for label, records in measured.items():
        final = records[-1]["matrices"]
        ranges[label] = {key: [min(final[n][key] for n in names), max(final[n][key] for n in names)]
                         for key in ("top_one_energy", "stable_rank")}
        window = [r for r in records if 1100 <= r["step"] <= 1300]
        late_window[label] = {"steps": [r["step"] for r in window], "cooldown_overlap": False,
            "mean_quantiles": {name: {str(q): float(np.mean([r["matrices"][name]["quantiles"][str(q)]
                                        for r in window])) for q in QUANTILES} for name in names}}
    provenance = {}
    for label, root in roots.items():
        digest = hashlib.sha256()
        files = [root / "metadata.json", root / "status.json", root / "checkpoint.json"]
        files += sorted((root / "steps").glob("step*.json"))
        for folder in ("spectra", "spectra_momentum"):
            files += sorted((root / folder).glob("step*.npz"))
        for path in files:
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
        provenance[label] = {"root": str(root), "files": len(files), "artifact_sha256": digest.hexdigest(),
                             "training_source_sha256": metas[label]["source_sha256"]}
    summary = {"controls": controls, "parameter_count": a_meta["parameter_count"], "tokens": budget,
               "steps": total_steps, "spectral_samples_per_object": len(expected), "matrices": names,
               "maximum_normalized_energy_error": energy_error, "endpoint": endpoint,
               "perplexity": {"AdamW": math.exp(endpoint["adamw_nll"]), "Muon": math.exp(endpoint["muon_nll"]),
                              "relative_reduction": 1 - math.exp(delta)},
               "final_ranges_across_24_matrices": ranges, "pre_cooldown_window": late_window,
               "execution": {label: {"world_size": metas[label].get("world_size", 1),
                              "device_name": metas[label]["device_name"],
                              "invocation_seconds": statuses[label]["invocation_seconds"]} for label in roots},
               "provenance": provenance, "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "limits": ["One seed; no uncertainty estimate or statistical-significance claim.",
                          "Optimizer recipes include different auxiliary rates and LR-dependent weight decay.",
                          "Different GPU counts and FP32 reduction orders; no wall-clock optimizer-speed claim.",
                          "Momentum and post-preconditioning updates are distinct observables.",
                          "Shared local initialization is not verified paper initialization.",
                          "No stabilization test or spectral cause of loss improvement is established."]}
    atomic_json(out / "summary.json", summary)
    report = f"""# Eight-layer Muon / AdamW comparison

Both runs completed all {total_steps:,} updates and {budget:,} tokens, with
{len(expected)} snapshots of all24 matrices. Initial weights, frozen model
code, data manifests, token exposure, batch and validation bank agree.

Final validation NLL: **Muon {endpoint['muon_nll']:.6f}**, **AdamW {endpoint['adamw_nll']:.6f}**;
Muon minus AdamW is **{delta:.6f} nats/token**. Perplexities are
{math.exp(endpoint['muon_nll']):.3f} and {math.exp(endpoint['adamw_nll']):.3f}
({100 * (1-math.exp(delta)):.2f}% lower for the Muon recipe in this one run).

At the endpoint, the leading singular direction contains
{100*ranges['AdamW update']['top_one_energy'][0]:.2f}–{100*ranges['AdamW update']['top_one_energy'][1]:.2f}%
of AdamW update energy across the24 matrices, versus
{100*ranges['Muon update']['top_one_energy'][0]:.3f}–{100*ranges['Muon update']['top_one_energy'][1]:.3f}%
for Muon's post-NS update. Muon's momentum remains concentrated before NS:
{100*ranges['Muon momentum']['top_one_energy'][0]:.2f}–{100*ranges['Muon momentum']['top_one_energy'][1]:.2f}%.
The post-NS flattening is expected from orthogonalization; it does not by itself
establish why validation loss improved. The shared normalized reference is
1/sqrt(512)=0.04419 for a matrix with512 equal singular values.

This is a comparison of optimizer recipes at one seed. AdamW LR is0.0012;
Muon body LR is0.01 with auxiliary AdamW LR0.002. The auxiliary rate and
effective decay therefore also change. AdamW used one RTX6000 Ada GPU and
Muon used four; elapsed-time differences do not isolate optimizer efficiency.
Both use our same local GPT initialization, not a verified paper initializer.

Figures (PNG/PDF): [validation](validation.png), [median trajectories](median_focus.png),
[all24 medians](median_all24.png), [final spectra](final_spectra_focus.png),
[all24 final spectra](final_spectra_all24.png), [leading energy](top_energy_all24.png),
[adaptive step magnitude](step_norm_all24.png). [Complete PDF](comparison.pdf).

Raw exports: [validation](validation.csv), [all sampled spectral metrics](spectral_metrics.csv),
[all final singular values](final_spectra.csv), [summary and provenance](summary.json).
The summary preserves the predeclared1100–1300 pre-cooldown window. No formal
stabilization criterion or uncertainty band is introduced after seeing the curves.

Sources: `{adam_path}` and `{muon_path}`. Figures reuse saved exact singular
values; no training or new SVD computation is performed. `compare.py` is copied
here with its source hash and can be rerun through the research module.
"""
    (out / "README.md").write_text(report)
    (out / "compare.py").write_bytes(Path(__file__).read_bytes())
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adamw", required=True, type=Path)
    parser.add_argument("--muon", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = compare(args.adamw, args.muon, args.out)
    print(json.dumps({key: result[key] for key in ("endpoint", "perplexity", "final_ranges_across_24_matrices")}, indent=2))


if __name__ == "__main__":
    main()

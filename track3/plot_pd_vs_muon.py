"""Figure: Track 3 tuned Muon (#36) vs partial data-norm Muon + geometry decay (alpha 1/8) at 3150 steps.

Validation loss vs step. Runs are grouped by identical logged code (as in earliest_step.py), so no run is hand-picked:
  - #36 tuned Muon, 3250 steps: our runs of the unchanged script (plus the analysis-only save variant) on the
    same GPU types as the PD runs, and the mean of #36's 10 published H100 logs;
  - PD + geometry decay, #36's schedule compressed to 3150 steps (the Track 3 result, n = 6);
  - the same PD configuration on #36's full 3250-step schedule (context).
Bands span min..max over runs; lines are means. Writes track3/figures/pd_vs_muon_3150.png.

With --variants, also the 3150-step variants of the same configuration that have finished runs: diagonal-input PD
(only the diagonal of C), PD plus IsoMuon's diagonal output-side factor, both diagonal, and TS (PD plus the full-matrix
output factor with GN labels, beta 1/4 and 1/8; its statistics pass is an extra forward-backward, so it is a research
comparison, not a Track 3-legal run). A third panel shows each
variant's difference from the full-PD mean at matching steps. Writes track3/figures/pd_vs_muon_3150_variants.png.
"""
import re
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PUBLISHED = Path("/home-nfs/mohamadamin/.tmp/claude-1457/-share-data-dl-theory-amin-projects-explore-muon/"
                 "740185e2-c249-4cf1-b369-ce504ecea942/scratchpad/mng/records/track_3_optimization/results/"
                 "20260610_tuned_baseline_3250")
LINE = re.compile(r"^step:(\d+)/(\d+) val_loss:([\d.]+)", re.M)
SEP = "\n" + "=" * 100 + "\n"
VARIANTS = [  # (label, scripts, color, final-marker x offset)
    ("PD + diagonal output-side factor (IsoMuon's), 3150 steps", ["train_gpt_pd_pdwd_a0.125_s3150_row.py"],
     "#2ca02c", -8),
    ("Diagonal PD (only the diagonal of C), 3150 steps", ["train_gpt_pd_pdwd_a0.125_s3150_diag.py"], "#9467bd", 8),
    ("Diagonal PD + diagonal output-side factor, 3150 steps", ["train_gpt_pd_pdwd_a0.125_s3150_diagrow.py"],
     "#17becf", 16),
    ("TS (PD + full output factor, β=¼; extra stats pass, not Track 3-legal), 3150 steps",
     ["train_gpt_pd_pdwd_a0.125_s3150_ts_b0.25.py"], "#ff7f0e", -16),
    ("TS (PD + full output factor, β=⅛; extra stats pass, not Track 3-legal), 3150 steps",
     ["train_gpt_pd_pdwd_a0.125_s3150_ts_b0.125.py"], "#8c564b", -24),
    ("Legal TS (β=⅛, output statistic from the training backward, data labels), 3150 steps",
     ["train_gpt_pd_pdwd_a0.125_s3150_tsef_b0.125.py"], "#e377c2", 24),
    ("Legal TS + two-sided geometry decay (β=⅛), 3150 steps",
     ["train_gpt_pd_pdwd_a0.125_s3150_tsef_b0.125_geo2.py"], "#bcbd22", 32),
]


def curves_for(scripts, folder=HERE / "logs"):
    sources = [(HERE / s).read_text().rstrip("\n") for s in scripts]
    out = []
    for path in sorted(folder.glob("*.txt")):
        if path.name.startswith(("console", "queue")):
            continue
        text = path.read_text(errors="replace")
        if text.split(SEP, 1)[0].rstrip("\n") not in sources:
            continue
        points = {int(m[1]): float(m[3]) for m in LINE.finditer(text)}
        first = next(LINE.finditer(text), None)
        if first is None:                        # just started, no validation yet
            continue
        total = int(first[2])
        if total in points:                      # finished runs only
            out.append(points)
    return out


def published():
    out = []
    for path in sorted(PUBLISHED.glob("*.txt")):
        points = {int(m[1]): float(m[3]) for m in LINE.finditer(path.read_text(errors="replace"))}
        if 3250 in points:
            out.append(points)
    return out


def band(runs):
    steps = sorted(set.intersection(*(set(r) for r in runs)) - {0})
    mean = [sum(r[s] for r in runs) / len(runs) for s in steps]
    return steps, mean, [min(r[s] for r in runs) for s in steps], [max(r[s] for r in runs) for s in steps]


def main():
    variants = "--variants" in sys.argv[1:]
    groups = [
        ("Tuned Muon (#36), 3250 steps, our GPUs", curves_for(["baseline/train_gpt_simple.py", "train_gpt_simple_save.py"]),
         "#4a4a4a", "-"),
        ("PD + geometry decay (α=⅛), 3150 steps", curves_for(["train_gpt_pd_pdwd_a0.125_s3150.py"]), "#d62728", "-"),
        ("PD + geometry decay (α=⅛), 3250 steps", curves_for(["train_gpt_pd_pdwd_a0.125.py"]), "#ff9896", "--"),
    ]
    extra = [(label, curves_for(scripts), color, "-", dx) for label, scripts, color, dx in VARIANTS] if variants else []
    extra = [e for e in extra if e[1]]           # variants with at least one finished run
    groups += [e[:4] for e in extra]
    h100 = published()
    if variants:
        fig = plt.figure(figsize=(14, 10.8))
        grid = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], height_ratios=[1, 0.85])
        ax1, ax2, ax3 = fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[1, :])
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.6), gridspec_kw={"width_ratios": [1, 1.25]})
    zoom_top = 3.336 if variants else 3.330
    for ax, (x0, x1, y0, y1) in ((ax1, (0, 3300, 3.25, 4.2)), (ax2, (2750, 3265, 3.263, zoom_top))):
        for label, runs, color, style in groups:
            steps, mean, lo, hi = band(runs)
            ax.fill_between(steps, lo, hi, color=color, alpha=0.18, linewidth=0)
            ax.plot(steps, mean, style, color=color, lw=2, label=f"{label} (n={len(runs)}): final {mean[-1]:.4f}")
        steps, mean, _, _ = band(h100)
        ax.plot(steps, mean, ":", color="#1f77b4", lw=2,
                label=f"Tuned Muon (#36), published H100 mean (n={len(h100)}): final {mean[-1]:.4f}")
        ax.axhline(3.28, color="k", lw=1, alpha=0.6)
        ax.set_xlim(x0, x1)
        ax.set_ylim(y0, y1)
        ax.set_xlabel("training step")
        ax.set_ylabel("validation loss")
        ax.grid(alpha=0.3)
    ax1.set_title("Full run")
    ax1.text(150, 3.284, "target 3.28", fontsize=9, va="bottom")
    # Zoom panel: finals and the Track 3 criterion.
    pd3150 = [r[3150] for r in groups[1][1]]
    muon = [r[3250] for r in groups[0][1]]
    mean_pd = sum(pd3150) / len(pd3150)
    ax2.scatter([3150] * len(pd3150), pd3150, color="#d62728", s=22, zorder=5)
    ax2.scatter([3250] * len(muon), muon, color="#4a4a4a", s=22, zorder=5)
    for _, runs, color, _, dx in extra:
        ax2.scatter([3150 + dx] * len(runs), [r[3150] for r in runs], color=color, s=22, zorder=5)
    ax2.axvline(3150, color="#d62728", lw=1, alpha=0.4)
    ax2.axvline(3250, color="#4a4a4a", lw=1, alpha=0.4)
    score = (3.28 - mean_pd) * len(pd3150) ** 0.5
    ax2.annotate(f"PD at 3150 steps: mean {mean_pd:.5f} over n={len(pd3150)} runs\n"
                 f"Track 3 score (3.28−mean)·√n = {score:.4f} ≥ 0.004 ✓",
                 xy=(3150, mean_pd), xytext=(2765, 3.2665), fontsize=9, color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))
    ax2.annotate("", xy=(3150, 3.2652), xytext=(3250, 3.2652),
                 arrowprops=dict(arrowstyle="<->", color="k", lw=1.2))
    ax2.text(3200, 3.2657, "100 fewer steps", ha="center", va="bottom", fontsize=9)
    ax2.set_title("End of training (zoom): markers are individual runs' finals")
    ax2.legend(loc="upper right", fontsize=8 if variants else 8.5, framealpha=0.95)
    if variants:
        # Difference from the full-PD mean at matching steps, same compressed schedule.
        bsteps, bmean, blo, bhi = band(groups[1][1])
        base = dict(zip(bsteps, bmean))
        ax3.fill_between(bsteps, [lo - m for lo, m in zip(blo, bmean)], [hi - m for hi, m in zip(bhi, bmean)],
                         color="#d62728", alpha=0.15, linewidth=0,
                         label=f"full PD + geometry decay: run-to-run range around its mean (n={len(groups[1][1])})")
        ax3.axhline(0, color="#d62728", lw=1.6)
        first, off_scale = 375, []  # the first validations differ by far more; noted instead of plotted
        for label, runs, color, _, _ in extra:
            steps, mean, lo, hi = band(runs)
            keep = [i for i, s in enumerate(steps) if s in base]
            xs = [steps[i] for i in keep]
            ax3.fill_between(xs, [lo[i] - base[steps[i]] for i in keep], [hi[i] - base[steps[i]] for i in keep],
                             color=color, alpha=0.2, linewidth=0)
            ax3.plot(xs, [mean[i] - base[steps[i]] for i in keep], "-o", color=color, lw=2, ms=3,
                     label=f"{label} (n={len(runs)}): {mean[-1] - base[3150]:+.4f} at step 3150")
            short = label.split(" (")[0].split(",")[0]
            if "β=" in label:
                short += " " + label[label.index("β="):label.index("β=") + 3]
            off_scale.append(short + ": " + ", ".join(
                f"{mean[i] - base[steps[i]]:+.3f}" for i in keep if steps[i] < first))
        off_note = f"Off scale before step {first} (steps 125, 250): " + "; ".join(off_scale) + "."
        ax3.set_xlim(first - 25, 3200)
        ax3.set_ylim(-0.012, 0.028)
        ax3.set_xlabel("training step (all on #36's schedule compressed to 3150 steps)")
        ax3.set_ylabel("val loss − full-PD mean")
        ax3.set_title("Difference from full PD at matching steps (below 0 = better than full PD)")
        ax3.grid(alpha=0.3)
        ax3.legend(loc="upper right", fontsize=8.5, framealpha=0.95)
        fig.suptitle("Track 3 (124M GPT, 524,288-token batch): PD + geometry decay and its preconditioner variants "
                     "(diagonal, output-side factor, TS)", fontsize=12)
        fig.text(0.01, 0.004, "Bands: min–max over runs; lines: means. All PD variants use α=⅛ and PD-geometry "
                 "weight decay; only the preconditioner differs. Our runs: L40S, RTX 6000 Ada and A6000 (4 GPUs "
                 "each); #36's published runs: 8×H100.\n" + "\n".join(textwrap.wrap(off_note, 250)),
                 fontsize=7.5, color="#555555")
        fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    else:
        fig.suptitle("Track 3 (124M GPT, 524,288-token batch): PD + geometry decay reaches 3.28 in 3150 steps; "
                     "tuned Muon (#36) needs 3250", fontsize=12)
        fig.text(0.01, 0.008, "Bands: min–max over runs; lines: means. The 3150-step PD runs use #36's schedule "
                 "compressed to 3150 steps, so their LR decays earlier and mid-run they sit below the 3250-step "
                 "curves; compare end points.\nOur runs: L40S, RTX 6000 Ada and A6000 (4 GPUs each); #36's "
                 "published runs: 8×H100.", fontsize=8, color="#555555")
        fig.tight_layout(rect=(0, 0.05, 1, 1))
    out = HERE / "figures" / ("pd_vs_muon_3150_variants.png" if variants else "pd_vs_muon_3150.png")
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=160)
    print(out, {label: len(runs) for label, runs, _, _ in groups}, "h100", len(h100))


if __name__ == "__main__":
    main()

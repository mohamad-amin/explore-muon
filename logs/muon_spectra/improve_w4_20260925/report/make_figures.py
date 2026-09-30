"""Figures for the draft report (reads step rows, momentum spectra and probe JSON only; writes report/figures/).

Adapted from ../plot_curves.py, ../plot_summary.py and ../plot_best.py, which are left unchanged.
Run from the project root:
    .venv/bin/python logs/muon_spectra/improve_w4_20260925/report/make_figures.py
A run counts only when its final step row exists; an unfinished arm is left out and reported.
"""
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
LOGS = HERE.parent.parent            # logs/muon_spectra
OUT = HERE / "figures"

# Categorical slots in fixed order (validated: dataviz validate_palette.js, light mode, all pairs).
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
COLOR = {"M": INK, "PD": "#2a78d6", "S": "#eb6834", "SPD": "#1baf7a"}
NAME = {"M": "Muon (tuned)", "PD": "data-norm Muon", "S": "SOAP-Muon", "SPD": "SOAP-Muon + data norm"}
GPU_MARK = {"A6000": "o", "L40S": "s", "RTX 6000 Ada": "^"}

plt.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7.8, "axes.labelsize": 7.5, "legend.fontsize": 6.6,
    "xtick.labelsize": 6.8, "ytick.labelsize": 6.8, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK2, "axes.labelcolor": INK, "text.color": INK,
    "axes.titlecolor": INK, "grid.color": GRID, "grid.linewidth": 0.5, "grid.linestyle": "-",
    "lines.linewidth": 1.3, "lines.solid_capstyle": "round", "legend.frameon": False,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})


def find(name):
    hits = glob.glob(str(LOGS / "improve_w*_20260925" / name))
    if len(hits) != 1:
        raise FileNotFoundError(f"{name}: {hits}")
    return Path(hits[0])


def rows(name):
    return [json.loads(Path(p).read_text()) for p in glob.glob(str(find(name) / "scientific/steps/step*.json"))]


def val(name):
    return dict(sorted((r["step"], r["validation_nll"]) for r in rows(name) if "validation_nll" in r and r["step"] > 0))


def gpu(name):
    return "A6000" if "a6000" in name else "RTX 6000 Ada" if name.endswith("_g20") else "L40S"


def seed(name):
    return int(name.split("_s")[-1][:6])


def finished(name, last):
    return last in val(name)


def delta(arm, base, start=300):
    a, b = val(arm), val(base)
    steps = [s for s in sorted(a) if s in b and s >= start]
    return np.array(steps), np.array([a[s] - b[s] for s in steps])


# Paired arms: method arm -> tuned Muon@0.007 on the same seed, GPU type, width and horizon.
W512 = {
    "PD": [("PD_a0.25_lr0.01_s260924", "M_lr0.007_s260924"), ("PD_a0.25_lr0.01_s260924_g20", "M_lr0.007_s260924_g20"),
           ("PD_a0.25_lr0.01_s260925_a6000", "M_lr0.007_s260925_a6000"), ("PD_a0.25_lr0.01_s260926_a6000", "M_lr0.007_s260926_a6000"),
           ("PD_a0.25_lr0.01_s260927_a6000", "M_lr0.007_s260927_a6000"), ("PD_a0.25_lr0.01_s260928_a6000", "M_lr0.007_s260928_a6000")],
    "S": [("S_lr0.007_s260924", "M_lr0.007_s260924"), ("S_lr0.007_s260925_a6000", "M_lr0.007_s260925_a6000"),
          ("S_lr0.007_s260926_a6000", "M_lr0.007_s260926_a6000"), ("S_lr0.007_s260927_a6000", "M_lr0.007_s260927_a6000"),
          ("S_lr0.007_s260928_a6000", "M_lr0.007_s260928_a6000")],
    "SPD": [("SPD_a0.25_lr0.01_s260924", "M_lr0.007_s260924"), ("SPD_a0.25_lr0.01_s260924_g20", "M_lr0.007_s260924_g20"),
            ("SPD_a0.25_lr0.01_s260925_a6000", "M_lr0.007_s260925_a6000"), ("SPD_a0.25_lr0.01_s260926_a6000", "M_lr0.007_s260926_a6000"),
            ("SPD_a0.25_lr0.01_s260927_a6000", "M_lr0.007_s260927_a6000"), ("SPD_a0.25_lr0.01_s260928_a6000", "M_lr0.007_s260928_a6000")],
}
LONG = [("PD_a0.25_lr0.01_2x_s260924", "M_lr0.007_2x_s260924"), ("PD_a0.25_lr0.01_2x_s260925_g20", "M_lr0.007_2x_s260925_g20")]
W768 = {
    "PD": [("PD_w768_a0.25_lr0.01_s260925_a6000", "M_w768_lr0.007_s260925_a6000"), ("PD_w768_a0.25_lr0.01_s260924", "M_w768_lr0.007_s260924"),
           ("PD_w768_a0.25_lr0.01_s260926_g20", "M_w768_lr0.007_s260926_g20"), ("PD_w768_a0.25_lr0.01_s260927_a6000", "M_w768_lr0.007_s260927_a6000")],
    "SPD": [("SPD_w768_a0.25_lr0.01_s260924", "M_w768_lr0.007_s260924"), ("SPD_w768_a0.25_lr0.01_s260926_g20", "M_w768_lr0.007_s260926_g20")],
    "S": [("S_w768_lr0.007_s260924", "M_w768_lr0.007_s260924")],
}
S768_DONE = finished("S_w768_lr0.007_s260924", 2562)
if not S768_DONE:
    W768["S"] = []
SELECTION = {"w512": (260925, 260926), "w768": (260925,)}


def resolution_unit(curvatures):
    """Smallest step between distinct non-negative curvature values (the finite-difference quantum)."""
    pos = np.unique(np.round(curvatures[curvatures > 0], 12))
    steps = np.diff(np.concatenate([[0.0], pos]))
    return float(np.min(steps[steps > 1e-9]))


def style(ax, grid_y=True):
    ax.grid(axis="y" if grid_y else "both", alpha=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


FINALS = []


def absolute(ax, arms, title, xlim, ylim, last):
    for key, name in arms:
        v = val(name)
        x = [s for s in v if xlim[0] <= s <= xlim[1]]
        ax.plot(x, [v[s] for s in x], color=COLOR[key], label=NAME[key])
        FINALS.append((title, NAME[key], v.get(last)))
    ax.set(title=title, xlim=xlim, ylim=ylim, xlabel="step", ylabel="validation loss (nats/token)")
    style(ax)


def band(ax, pairs, key, label=None):
    """Mean paired difference over seeds with its min-max range."""
    curves = [delta(a, b) for a, b in pairs]
    steps = curves[0][0]
    for s, _ in curves:
        steps = np.intersect1d(steps, s)
    d = np.array([np.interp(steps, s, y) for s, y in curves])
    ax.fill_between(steps, d.min(0), d.max(0), color=COLOR[key], alpha=0.14, lw=0)
    ax.plot(steps, d.mean(0), color=COLOR[key],
            label=(label or NAME[key]) + f" (n = {len(pairs)}): {d.mean(0)[-1]:+.4f}")


def method_legend(fig, keys=("M", "S", "PD", "SPD"), y=1.0, size=7.0):
    handles = [Line2D([], [], color=COLOR[k], lw=1.6, label=NAME[k]) for k in keys]
    fig.legend(handles=handles, loc="upper center", ncol=len(keys), fontsize=size, bbox_to_anchor=(0.5, y),
               handlelength=2.0, columnspacing=1.6)


def figure_overview():
    """Figure 1: width 512 at a glance."""
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35))
    w512 = [("M", "M_lr0.007_s260927_a6000"), ("S", "S_lr0.007_s260927_a6000"),
            ("PD", "PD_a0.25_lr0.01_s260927_a6000"), ("SPD", "SPD_a0.25_lr0.01_s260927_a6000")]
    absolute(axes[0], w512, "(a) validation loss, last 700 steps (one seed)", (770, 1469), (3.66, 3.95), 1469)
    ax = axes[1]
    for key in ("S", "PD", "SPD"):
        band(ax, W512[key], key)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set(title="(b) difference from tuned Muon (all seeds)", xlabel="step", ylabel="Δ validation loss",
           xlim=(300, 1469), ylim=(-0.12, 0.005))
    style(ax)
    ax.legend(loc="lower right", handlelength=1.6, fontsize=6.3)
    fig.tight_layout(w_pad=1.2, rect=(0, 0, 1, 0.9))
    method_legend(fig)
    fig.savefig(OUT / "overview.pdf")
    plt.close(fig)


# Matrix kinds: four validated hues (violet, yellow, magenta, green; all-pairs pass) plus two neutrals.
KIND_ORDER = ["q", "k", "v", "o", "up", "down"]
KIND_COLOR = {"v": "#4a3aa7", "up": "#eda100", "q": "#e87ba4", "o": "#008300", "down": "#52514e", "k": "#a9a7a0"}
KIND_LABEL = {"q": "query q", "k": "key k", "v": "value v", "o": "attention output o", "up": "MLP up", "down": "MLP down"}
SPIKE_RUN = LOGS / "depth8_w512_nobias_rms_qk_20260925/muon"
NS5 = ((4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037),
       (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012))      # research/adamw_spectra/muon.py NS_COEFFICIENTS
NS3 = ((3.4445, -4.7750, 2.0315),) * 3                            # JORDAN_QUINTIC, three steps
BIG = {"font.size": 8.0, "axes.titlesize": 8.4, "axes.labelsize": 8.0, "xtick.labelsize": 7.4,
       "ytick.labelsize": 7.4, "legend.fontsize": 7.2}


def ns_map(x, coeffs):
    x = np.asarray(x, dtype=np.float64)
    for a, b, c in coeffs:
        x = a * x + b * x ** 3 + c * x ** 5
    return x


def spectra(folder):
    out = {}
    for f in sorted((SPIKE_RUN / "scientific" / folder).glob("step*.npz")):
        z = np.load(f)
        out[int(f.stem[4:])] = {k: z[k].astype(np.float64) for k in z.keys()}
    return out


def kind_legend(fig, y=1.0):
    handles = [Line2D([], [], color=KIND_COLOR[k], lw=2.0, label=KIND_LABEL[k]) for k in KIND_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=6, bbox_to_anchor=(0.5, y), handlelength=1.6,
               columnspacing=1.2, fontsize=7.4)


def figure_outlier_observation():
    """Figure 2: the outlier, its persistence, and the deflation argument (paper's model, Muon at LR 0.01)."""
    mom, upd = spectra("spectra_momentum"), spectra("spectra")
    with plt.rc_context(BIG):
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.55))
        ax = axes[0]
        for kind in KIND_ORDER:
            sv = mom[1300][f"block04.{kind}"]
            ax.loglog(np.arange(1, len(sv) + 1), sv, color=KIND_COLOR[kind], lw=1.8 if kind == "v" else 1.1,
                      zorder=3 if kind == "v" else 2)
        u = upd[1300]["block04.v"]
        ax.loglog(np.arange(1, len(u) + 1), u, color=KIND_COLOR["v"], lw=1.2, ls=(0, (3, 1.5)), zorder=3)
        ax.text(1.05, 0.0016, "dashed: v after\northogonalization", fontsize=6.6, color=KIND_COLOR["v"], va="bottom")
        ax.set(xlabel="rank i", ylabel=r"$\sigma_i\,/\,\|M\|_F$", ylim=(3e-4, 1.4), xlim=(0.8, 700),
               title="(a) one outlier per matrix")
        style(ax, grid_y=False)
        ax.grid(which="major", alpha=1.0)
        ax = axes[1]
        steps = [t for t in sorted(mom) if t >= 25]
        ax.axvspan(1323, 1469, color=GRID, alpha=0.8, lw=0)
        ax.text(1310, 0.93, "decay", fontsize=6.6, color=INK2, ha="right", va="center")
        for kind in KIND_ORDER:
            rho = [np.median([mom[t][f"block{b:02d}.{kind}"][0] ** 2 / np.sum(mom[t][f"block{b:02d}.{kind}"] ** 2)
                              for b in (2, 4, 6, 8)]) for t in steps]
            ax.plot(steps, rho, color=KIND_COLOR[kind], lw=1.8 if kind == "v" else 1.1, zorder=3 if kind == "v" else 2)
        ax.set(xlabel="step", ylabel=r"top-mode share $\sigma_1^2/\|M\|_F^2$", ylim=(0, 1.0), xlim=(0, 1469),
               title="(b) it persists through training")
        style(ax)
        ax = axes[2]
        x = np.logspace(-4, np.log10(1.3), 400)
        ax.semilogx(x, ns_map(x, NS5), color=INK, lw=1.4)
        ax.semilogx(x, ns_map(x, NS3), color=INK2, lw=1.2, ls=(0, (4, 2)))
        ax.legend(handles=[Line2D([], [], color=INK, lw=1.4, label="5 steps (used here)"),
                           Line2D([], [], color=INK2, lw=1.2, ls=(0, (4, 2)), label="3 classic steps")],
                  loc="lower right", bbox_to_anchor=(1.0, 0.27), fontsize=6.4, handlelength=2.0,
                  frameon=True, facecolor="white", edgecolor="none", framealpha=0.95)
        sv = mom[1469]["block04.v"]
        sv = sv / np.sqrt(np.sum(sv ** 2))
        rest = sv[1:] / np.sqrt(np.sum(sv[1:] ** 2))
        keep = lambda a: a[(a >= 1e-4) & (a <= 1.3)]
        ax.vlines(keep(sv[1:]), -0.19, -0.07, color=KIND_COLOR["v"], lw=0.4, alpha=0.35)
        ax.vlines(keep(rest), -0.36, -0.24, color="#8f84d6", lw=0.4, alpha=0.35)
        ax.scatter([sv[0]], [-0.13], marker="v", s=18, color=KIND_COLOR["v"], zorder=4)
        ax.axhline(1.0, color=GRID, lw=0.8, zorder=0)
        ax.axhline(-0.03, color=AXIS, lw=0.5, zorder=0)
        ax.set(xlabel=r"input $\sigma_i/\|M\|_F$", ylabel="output singular value",
               ylim=(-0.4, 1.3), xlim=(1e-4, 1.3), title="(c) why deflation was proposed")
        ax.set_yticks([-0.30, -0.13, 0, 0.5, 1.0], ["v, head out", "v, head in", "0", "0.5", "1"])
        for lab in ax.get_yticklabels()[:2]:
            lab.set_color(KIND_COLOR["v"]); lab.set_fontsize(6.4)
        style(ax, grid_y=False)
        fig.tight_layout(w_pad=1.0, rect=(0, 0, 1, 0.88))
        kind_legend(fig)
        fig.savefig(OUT / "outlier_observation.pdf")
        plt.close(fig)
    late = [np.mean([mom[t][n][0] ** 2 / np.sum(mom[t][n] ** 2) for t in mom if 1300 <= t <= 1469]) for n in mom[1300]]
    print(f"outlier figure 1: late rho1 {min(late):.2f}-{max(late):.2f}; v block 4 final sigma1 {sv[0]:.3f}, "
          f"bulk median {np.median(sv[1:]):.4f} (head removed {np.median(rest):.4f})")


def figure_outlier_origin():
    """Figure 3: what the outlier is made of (fresh gradients at the final checkpoint of the paper's model)."""
    diag = json.loads((SPIKE_RUN / "diagnostic/summary.json").read_text())["matrices"]

    def reading(m):
        if not m["fresh_mean_gradient"]["spike_reproduced"]:
            return "not reproduced"
        mean, sink = m["mean_product"]["of_u1v1"] >= 0.5, m["position0"]["of_u1v1"] >= 0.5
        return "mean+sink" if mean and sink else "mean" if mean else "sink" if sink else "distributed"

    shade = {"mean": ("#52514e", "white"), "sink": ("#b9b7af", INK), "distributed": ("#ebeae4", INK),
             "not reproduced": ("white", INK2)}
    with plt.rc_context(BIG):
        fig, axes = plt.subplots(1, 3, figsize=(6.6, 2.75), gridspec_kw={"width_ratios": [1.25, 1.0, 1.0]})
        ax = axes[0]
        counts = {}
        for i, kind in enumerate(KIND_ORDER):
            for b in range(1, 9):
                m = diag[f"block{b:02d}.{kind}"]
                r = reading(m)
                counts[r] = counts.get(r, 0) + 1
                face, ink = shade[r]
                ax.add_patch(plt.Rectangle((b - 0.5, i - 0.5), 1, 1, facecolor=face, edgecolor="white", lw=1.2))
                if r == "not reproduced":
                    ax.add_patch(plt.Rectangle((b - 0.46, i - 0.46), 0.92, 0.92, facecolor="white",
                                               edgecolor=AXIS, lw=0.8))
                    txt = "n.r."
                else:
                    txt = f"{m['mean_product']['of_u1v1']:.2f}".replace("-0.00", "0.00")
                ax.text(b, i, txt, ha="center", va="center", fontsize=5.6, color=ink)
        ax.set_xlim(0.5, 8.5)
        ax.set_ylim(len(KIND_ORDER) - 0.5, -0.5)
        ax.set_xticks(range(1, 9))
        ax.set_yticks(range(len(KIND_ORDER)), KIND_ORDER)
        ax.set_xlabel("block")
        ax.set_title("(a) what the outlier is made of")
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        handles = [plt.Rectangle((0, 0), 1, 1, facecolor=shade[r][0], edgecolor=AXIS, lw=0.5,
                                 label=f"{lab} ({counts.get(r, 0)})")
                   for r, lab in (("mean", "token-mean product"), ("sink", "position 0 (sink)"),
                                  ("distributed", "distributed"), ("not reproduced", "not reproduced"))]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6.4,
                  handlelength=1.0, columnspacing=0.8, handletextpad=0.4)
        ax = axes[1]
        labels = ["1", "2–5", "6–52", "53–256", "257–461", "462–512"]
        rng = np.random.default_rng(1)
        for j in range(6):
            vals = []
            for kind in KIND_ORDER:
                for b in range(1, 9):
                    r = diag[f"block{b:02d}.{kind}"]["bands"][j]["ratio"]
                    vals.append(r)
                    ax.scatter(j + rng.uniform(-0.22, 0.22), r, s=7, color=KIND_COLOR[kind], lw=0, alpha=0.85, zorder=3)
            ax.plot([j - 0.3, j + 0.3], [np.median(vals)] * 2, color=INK, lw=1.3, zorder=4)
        ax.axhline(1.0, color=INK2, lw=0.7)
        ax.axhline(0.2, color=INK2, lw=0.7, ls=(0, (3, 2)))
        ax.text(1.5, 0.07, "noise level", fontsize=6.3, color=INK2, ha="center")
        ax.set_xticks(range(6), labels, rotation=35, ha="right")
        ax.set(ylim=(-0.15, 1.8), xlabel="singular-value rank band", ylabel="fresh descent / in-sample",
               title="(b) the outlier is signal")
        style(ax)
        ax = axes[2]
        tokens = np.array([r["tokens"] for r in diag["block04.v"]["scaling"]]) / 1e6
        for kind in KIND_ORDER:
            y = [r["sigma1_over_median"] for r in diag[f"block04.{kind}"]["scaling"]]
            ax.loglog(tokens, y, color=KIND_COLOR[kind], lw=1.8 if kind == "v" else 1.1, marker="o", ms=2.2,
                      zorder=3 if kind == "v" else 2)
        y0 = diag["block04.v"]["scaling"][0]["sigma1_over_median"]
        ax.loglog(tokens, y0 * np.sqrt(tokens / tokens[0]), color=INK2, lw=0.9, ls=(0, (3, 2)))
        ax.text(tokens[-2] * 0.8, y0 * np.sqrt(tokens[-2] / tokens[0]) * 1.25, r"$\propto\sqrt{N}$", fontsize=7.0,
                color=INK2, ha="right")
        ax.set_xticks([0.5, 2, 8, 32], ["0.5M", "2M", "8M", "32M"])
        ax.set(xlabel="tokens averaged, N", ylabel=r"$\sigma_1\,/\,$median singular value",
               title="(c) it grows with averaging")
        style(ax, grid_y=False)
        ax.grid(which="major", alpha=1.0)
        fig.tight_layout(w_pad=0.9, rect=(0, 0, 1, 0.9))
        kind_legend(fig)
        fig.savefig(OUT / "outlier_origin.pdf")
        plt.close(fig)
    print("outlier figure 2 readings:", counts)


def figure_results():
    """Figure 3: every paired final difference (curves over training are in the appendix figure)."""
    fig = plt.figure(figsize=(6.6, 2.35))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.4, 1.0], wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])
    lax = fig.add_subplot(gs[0, 1])
    lax.axis("off")
    groups = [(0.0, "PD", W512["PD"], 1469, "w512"), (1.0, "S", W512["S"], 1469, "w512"),
              (2.0, "SPD", W512["SPD"], 1469, "w512"), (3.5, "PD", LONG, 2938, "w512"),
              (5.0, "PD", W768["PD"], 2562, "w768")]
    if S768_DONE:
        groups.append((6.0, "S", W768["S"], 2562, "w768"))
    groups.append((7.0, "SPD", W768["SPD"], 2562, "w768"))
    for x0, key, pairs, last, width in groups:
        ds = []
        offsets = np.linspace(-0.14, 0.14, len(pairs)) if len(pairs) > 1 else [0.0]
        for (arm, base), off in zip(pairs, offsets):
            d = val(arm)[last] - val(base)[last]
            ds.append(d)
            selection = seed(arm) in SELECTION[width]
            ax.scatter(x0 + off, d, s=24, marker=GPU_MARK[gpu(arm)], facecolor="white" if selection else COLOR[key],
                       edgecolor=COLOR[key], linewidth=1.1, zorder=3)
        m = float(np.mean(ds))
        ax.plot([x0 - 0.24, x0 + 0.24], [m, m], color=INK, lw=1.0, zorder=2)
        ax.text(x0, min(ds) - 0.0011, f"{m:+.4f}", va="top", ha="center", fontsize=6.0, color=INK2)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set_xticks([1.0, 3.5, 6.0], ["width 512 (77M)", "2× tokens (77M)", "width 768 (134M)"])
    ax.set_xlim(-0.5, 7.75)
    ax.set_ylim(-0.037, 0.002)
    ax.set_ylabel("final Δ loss vs tuned Muon")
    style(ax)
    handles = [Line2D([], [], marker="s", ls="", color=COLOR[k], markersize=5, label=NAME[k]) for k in ("PD", "S", "SPD")]
    handles += [Line2D([], [], marker=GPU_MARK[g], ls="", color=INK2, markerfacecolor=INK2, markersize=4.5, label=g)
                for g in GPU_MARK]
    handles += [Line2D([], [], marker="o", ls="", markerfacecolor="white", markeredgecolor=INK2, markersize=4.5,
                       label="selection seed")]
    lax.legend(handles=handles, loc="center left", fontsize=6.2, handletextpad=0.4, borderaxespad=0.0)
    fig.savefig(OUT / "results.pdf")
    plt.close(fig)


def figure_alpha():
    """Figure: the exponent sweep and the exact (Hessian-vector-product) curvature measurement."""
    hvp = json.loads((LOGS / "gamma_probe_20260925/gamma_hvp.json").read_text())
    g512 = hvp["M_w512_lr0.007_s260925"]["median_gamma"]
    with plt.rc_context(BIG):
        fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
        ax = axes[0]
        m7 = val("M_lr0.007_s260925_a6000")[1469]
        m10 = val("M_lr0.01_s260925_a6000_gpusvd")[1469]
        sweep = {0.125: "PD_a0.125_lr0.01_s260925_a6000", 0.25: "PD_a0.25_lr0.01_s260925_a6000",
                 0.375: "PD_a0.375_lr0.01_s260925_a6000", 0.5: "PD_a0.5_lr0.01_s260925_a6000"}
        xs = [0.0] + sorted(sweep)
        ys = [m10 - m7] + [val(sweep[a])[1469] - m7 for a in sorted(sweep)]
        ax.axvline(g512 / 2, color=INK2, lw=0.9, ls=(0, (3, 2)), zorder=1,
                   label=f"curvature-matched, γ/2 = {g512 / 2:.2f}")
        ax.plot(xs, ys, color=COLOR["PD"], marker="o", ms=4.5, mec="white", mew=0.8, zorder=3,
                label="data-norm Muon (α = 0 is Muon)")
        for d, mk in ((0.1, "D"), (0.3, "s")):
            y = val(f"DDN_a0.5_d{d:g}_lr0.01_s260925_a6000")[1469] - m7
            ax.scatter(0.5, y, marker=mk, s=20, facecolor="white", edgecolor=COLOR["PD"], lw=1.0, zorder=4,
                       label=f"α = ½, damping {d:g}")
        ax.axhline(0, color=INK, lw=0.7)
        ax.set(xlabel="exponent α", ylabel="final Δ loss vs tuned Muon",
               title="(a) best at α = ¼", xlim=(-0.03, 0.55), ylim=(-0.021, 0.014))
        ax.set_xticks([0, 0.125, 0.25, 0.375, 0.5], ["0", "⅛", "¼", "⅜", "½"])
        style(ax)
        ax.legend(loc="upper center", fontsize=6.4, handletextpad=0.4, frameon=True, facecolor="white",
                  edgecolor="none", framealpha=0.9, bbox_to_anchor=(0.46, 1.0))
        ax = axes[1]
        mats = hvp["M_w512_lr0.007_s260925"]["matrices"]
        for name, info in sorted(mats.items()):
            kind = name.split(".")[1]
            lam = np.array([x["eigenvalue"] for x in info["rows"]]); cur = np.array([x["curvature"] for x in info["rows"]])
            gl, gc = np.exp(np.log(lam).mean()), np.exp(np.log(cur).mean())
            ax.loglog(lam / gl, cur / gc, color=KIND_COLOR[kind], lw=0.9, marker="o", ms=1.8, alpha=0.85,
                      zorder=3 if kind == "k" else 2)
        x = np.logspace(-3.3, 2.2, 20)
        ax.loglog(x, x, color=INK, lw=1.1, ls=(0, (1.2, 1.4)), zorder=4)
        ax.text(2.2e-3, 5.5e-4, "slope 1 (dotted)", fontsize=6.6, color=INK, ha="left", va="center")
        ax.set_ylim(3e-4, 5e2)
        ax.set(xlabel=r"input eigenvalue $\lambda_j$ / geometric mean", ylabel="curvature / geometric mean",
               title=f"(b) curvature ∝ input variance (median γ = {g512:.2f})")
        style(ax, grid_y=False)
        ax.grid(which="major", alpha=1.0)
        handles = [Line2D([], [], color=KIND_COLOR[k], lw=1.6, label=KIND_LABEL[k]) for k in KIND_ORDER]
        ax.legend(handles=handles, loc="lower right", fontsize=6.0, ncol=2, handlelength=1.2, columnspacing=0.8)
        fig.tight_layout(w_pad=1.5)
        fig.savefig(OUT / "alpha.pdf")
        plt.close(fig)
    print(f"alpha figure: HVP median gamma w512 {g512:.3f}; labels {list(hvp)}")


def figure_curves_full():
    """Appendix: absolute and paired curves for every setting."""
    fig, axes = plt.subplots(2, 3, figsize=(6.6, 4.0))
    w512 = [("M", "M_lr0.007_s260927_a6000"), ("S", "S_lr0.007_s260927_a6000"),
            ("PD", "PD_a0.25_lr0.01_s260927_a6000"), ("SPD", "SPD_a0.25_lr0.01_s260927_a6000")]
    absolute(axes[0, 0], w512, "(a) width 512, one seed", (50, 1469), (3.6, 6.3), 1469)
    absolute(axes[0, 1], w512, "(b) width 512, last 700 steps", (770, 1469), (3.66, 3.95), 1469)
    ax = axes[0, 2]
    for key in ("S", "PD", "SPD"):
        band(ax, W512[key], key)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set(title="(c) width 512: Δ vs tuned Muon", xlabel="step", ylabel="Δ validation loss", xlim=(300, 1469),
           ylim=(-0.12, 0.005))
    style(ax)
    w768 = [("M", "M_w768_lr0.007_s260924")] + ([("S", "S_w768_lr0.007_s260924")] if S768_DONE else []) + \
           [("PD", "PD_w768_a0.25_lr0.01_s260924"), ("SPD", "SPD_w768_a0.25_lr0.01_s260924")]
    absolute(axes[1, 0], w768, "(d) width 768, one seed", (1260, 2562), (3.43, 3.75), 2562)
    ax = axes[1, 1]
    for key in ("S", "PD", "SPD"):
        if W768[key]:
            band(ax, W768[key], key)
    ax.axhline(0, color=INK, lw=0.7)
    ax.set(title="(e) width 768: Δ vs tuned Muon", xlabel="step", ylabel="Δ validation loss", xlim=(300, 2562),
           ylim=(-0.12, 0.005))
    style(ax)
    long = [("M", "M_lr0.007_2x_s260924"), ("PD", "PD_a0.25_lr0.01_2x_s260924")]
    absolute(axes[1, 2], long, "(f) 2× tokens, one seed", (1240, 2938), (3.55, 3.85), 2938)
    fig.tight_layout(h_pad=1.0, w_pad=0.8, rect=(0, 0, 1, 0.95))
    method_legend(fig)
    fig.savefig(OUT / "training_curves.pdf")
    plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    for old in OUT.glob("*.pdf"):
        old.unlink()
    figure_overview()
    figure_outlier_observation()
    figure_outlier_origin()
    figure_results()
    figure_alpha()
    figure_curves_full()
    print("width-768 SOAP-Muon core finished:", S768_DONE)
    for row in FINALS:
        print("final", row)
    print("written:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()

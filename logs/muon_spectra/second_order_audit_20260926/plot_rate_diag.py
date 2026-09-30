"""Rate test with the aux frozen (2026-09-30, MUON_CASE 14:19 CDT): validation and top GN eigenvalue at every 16M batch start
for one 16M step per batch (A), four true 4M steps (B), four steps on each batch's frozen GN model (C) and its stiff/bulk
split (C split: the top GN eigenvectors projected out of inner steps 2-4).

usage: plot_rate_diag.py OUT_PNG
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
ARMS = [("A: one 16M PD step per batch", "path46/diag2/A", 1, "#2e86c1"),
        ("B: four true 4M steps per batch", "path46/diag2/B", 4, "#27ae60"),
        ("C: four steps on the batch's frozen GN model", "path46/diag2/C", 4, "#c0392b"),
        ("C split: stiff subspace kept on the plain step", "path46/diag2/Csplit", 4, "#e0843a")]


def rows(path):
    out = {}
    for f in (HERE / path / "steps").glob("step*.json"):
        r = json.loads(f.read_text())
        out[r["step"]] = r
    return out


def per_batch(r, k):
    """(batch, validation after the batch, lambda at the batch start) for an arm with k steps per 16M batch."""
    if not r:
        return [], [], []
    first = min(r)
    val, lam = {}, {}
    for s, row in r.items():
        batch = 47 + (s - first) // k
        if (s - first) % k == k - 1 and row.get("validation_nll") is not None:
            val[batch] = row["validation_nll"]
        if (s - first) % k == 0 and row.get("gn_top") is not None:
            lam[batch] = row["gn_top"]
    return val, lam


def main():
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    curves = {}
    for label, path, k, color in ARMS:
        val, lam = per_batch(rows(path), k)
        if not val:
            continue
        curves[label] = val
        ax[0].plot(sorted(val), [val[b] for b in sorted(val)], "-o", ms=3, color=color, label=label)
        ax[1].semilogy(sorted(lam), [lam[b] for b in sorted(lam)], "-o", ms=3, color=color, label=label)
    a_label, b_label = ARMS[0][0], ARMS[1][0]
    for label, _, _, color in ARMS[2:]:
        if label not in curves or a_label not in curves or b_label not in curves:
            continue
        a, b, c = curves[a_label], curves[b_label], curves[label]
        common = sorted(set(a) & set(b) & set(c))
        ratio = [(a[t] - c[t]) / (a[t] - b[t]) for t in common if a[t] != b[t]]
        ax[2].plot(common[:len(ratio)], ratio, "-o", ms=3, color=color, label=label)
    ax[0].set(xlabel="16M batch", ylabel="validation NLL", ylim=(4.8, 5.8), title="From PD @46, aux frozen")
    ax[0].legend(fontsize=7.5)
    ax[1].set(xlabel="16M batch", ylabel="top GN eigenvalue at the batch start",
              title="The frozen model loses the edge's plateau")
    ax[2].axhline(0, color="k", lw=0.7)
    ax[2].axhline(1, color="#27ae60", lw=0.7, ls="--")
    ax[2].set(xlabel="16M batch", ylabel="(L_A − L) / (L_A − L_B)", ylim=(-0.5, 1.1),
              title="Share of the true inner loop's gain kept")
    ax[2].legend(fontsize=7.5)
    fig.tight_layout()
    fig.savefig(sys.argv[1], dpi=130)
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()

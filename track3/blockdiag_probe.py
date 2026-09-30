"""Block-diagonal input statistics for PD's MLP-down layers: what changes relative to the full statistic?

PD preconditions each MLP-down matrix with R = (C / mean eig C + 1e-3 I)^(-alpha), where C is the EMA of the
3072-dim input second moment (ReLU^2 activations, all >= 0, so C has one dominant mean-like direction). The
block-diagonal idea keeps only C's four 768 x 768 diagonal blocks. This probe, on the final statistics saved by a
Track 3 PD run, asks *what* that changes; whether the change helps or hurts is a question for training.
  1. cos between PD steps polar(M R) R with full and approximate R (random and data-like momentum M) for:
     block-diagonal R; block-diagonal R with the top-k eigen-directions of C kept exactly; Muon (R = I).
  2. the spectrum of C vs that of its block-diagonal part;
  3. how block-diagonal R rescales each eigen-direction of C relative to full R, and the three block-contrast
     directions (the top eigenvector's pieces in the four blocks, recombined orthogonally to it).
All approximations normalize by the full C's mean eigenvalue (its mean diagonal, available from the blocks).
Writes track3/figures/blockdiag_mlp_down.png.

Usage: python track3/blockdiag_probe.py [RUN_ID]   (default: 2124499a, PD + geometry decay, alpha 1/8, 3150 steps)
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = Path(__file__).resolve().parent
DEFAULT_RUN = "2124499a-f8e9-4a2d-ad7c-b0874a698ebb"
RUN = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN
ALPHA, DAMPING, BLOCKS, LAYERS, TOPK = 0.125, 1e-3, 4, (1, 4, 7, 10), (1, 4, 16)
torch.set_num_threads(16)


def power(c, mean):
    """(c / mean + damping I)^(-alpha) for symmetric PSD c."""
    ev, vec = torch.linalg.eigh(0.5 * (c + c.T))
    return (vec * (ev.clamp_min(0) / mean + DAMPING).pow(-ALPHA)) @ vec.T


def block_power(c, mean=None):
    """power() of c's block-diagonal part, block by block; mean=None uses each block's own mean eigenvalue."""
    d, out = c.shape[0] // BLOCKS, torch.zeros_like(c)
    for i in range(BLOCKS):
        s = slice(i * d, (i + 1) * d)
        out[s, s] = power(c[s, s], c[s, s].diagonal().mean() if mean is None else mean)
    return out


def blockdiag(c):
    d, out = c.shape[0] // BLOCKS, torch.zeros_like(c)
    for i in range(BLOCKS):
        s = slice(i * d, (i + 1) * d)
        out[s, s] = c[s, s]
    return out


def step(M, r):
    u, _, vh = torch.linalg.svd(M @ r, full_matrices=False)
    d = (u @ vh) @ r
    return d / d.norm()


def unit_trace(r):
    return r / r.diagonal().mean()


def main():
    saved = torch.load(HERE / "logs" / f"{RUN}_final.pt", map_location="cpu", weights_only=False)["pd_stats"]
    torch.manual_seed(0)
    fig, axes = plt.subplots(2, len(LAYERS), figsize=(4.2 * len(LAYERS), 8.2))
    print(f"run {RUN[:8]}, MLP-down, alpha {ALPHA}: cos(step with full R, step with approximate R)")
    for col, layer in enumerate(LAYERS):
        s = saved[f"blocks.{layer}.mlp.proj"]
        C = (s["cov"] / s["weight"].clamp_min(1e-12)).double()
        C = 0.5 * (C + C.T)
        ev, U = torch.linalg.eigh(C)
        ev, U = ev.flip(0).clamp_min(0), U.flip(1)
        mean = ev.mean()
        R = {"full": power(C, mean), "block-diagonal": block_power(C, mean),
             "block-diagonal, own block means": block_power(C)}
        for k in TOPK:
            top = (U[:, :k] * ev[:k]) @ U[:, :k].T
            R[f"block-diagonal + top-{k} exact"] = power(top + blockdiag(C - top), mean)
        R["Muon (R = I)"] = torch.eye(C.shape[0], dtype=C.dtype)
        momenta = {"random M": torch.randn(768, C.shape[0], dtype=C.dtype),
                   "data-like M": torch.randn(768, C.shape[0], dtype=C.dtype) @ ((U * ev.sqrt()) @ U.T)}
        print(f" layer {layer:2d}: lambda_1 / mean = {float(ev[0] / mean):.0f}, "
              f"top-16 share of trace = {float(ev[:16].sum() / ev.sum()):.2f}")
        for mname, M in momenta.items():
            ref = step(M, R["full"])
            cells = [f"{name} {float((ref * step(M, r)).sum()):.4f}" for name, r in R.items() if name != "full"]
            print(f"   {mname:12s} " + " | ".join(cells))

        # Which full-R exponent does the block-diagonal step resemble most? (full R at exponent a, a <= alpha)
        grid = [a / 200 for a in range(8, 26)]
        for mname, M in momenta.items():
            bd = step(M, R["block-diagonal"])
            fits = [(float((bd * step(M, (U * (ev / mean + DAMPING).pow(-a)) @ U.T)).sum()), a) for a in grid]
            best = max(fits)
            print(f"   {mname:12s} block-diagonal step: closest full-R exponent {best[1]:.3f} (cos {best[0]:.4f}); "
                  f"cos with Muon {float((bd * step(M, R['Muon (R = I)'])).sum()):.4f}")

        # Per-direction scaling in C's eigenbasis, each R normalized to unit mean eigenvalue.
        diag = {name: ((unit_trace(r) @ U) * U).sum(0) for name, r in R.items()}
        x = (ev / mean).numpy()
        # Block-contrast directions: the top eigenvector's pieces in each block, orthogonal to the top eigenvector.
        d = C.shape[0] // BLOCKS
        pieces = torch.zeros(C.shape[0], BLOCKS, dtype=C.dtype)
        for i in range(BLOCKS):
            pieces[i * d:(i + 1) * d, i] = U[i * d:(i + 1) * d, 0]
        pieces -= U[:, :1] @ (U[:, :1].T @ pieces)
        Q, _ = torch.linalg.qr(pieces)
        Q = Q[:, :BLOCKS - 1]
        w, V = torch.linalg.eigh(Q.T @ C @ Q)
        contrast = Q @ V
        quad = lambda r, v: ((unit_trace(r) @ v) * v).sum(0)
        c_var = (((C @ contrast) * contrast).sum(0) / mean).numpy()
        c_ratio = (quad(R["block-diagonal"], contrast) / quad(R["full"], contrast)).numpy()
        top_ratio = float(quad(R["block-diagonal"], U[:, :1]) / quad(R["full"], U[:, :1]))
        print(f"   top direction: block-diagonal/full scaling {top_ratio:.2f}; block-contrast directions: "
              f"true variance/mean {', '.join(f'{v:.2f}' for v in c_var)}, "
              f"block-diagonal/full scaling {', '.join(f'{v:.2f}' for v in c_ratio)}")

        ax = axes[0, col]
        bd_ev = torch.cat([torch.linalg.eigvalsh(C[i * d:(i + 1) * d, i * d:(i + 1) * d]) for i in range(BLOCKS)])
        bd_ev = bd_ev.clamp_min(0).sort(descending=True).values
        rank = range(1, len(ev) + 1)
        ax.loglog(rank, x, color="k", lw=1.6, label="full C")
        ax.loglog(rank, (bd_ev / mean).numpy(), color="#e6550d", lw=1.6, label="4 diagonal blocks of C")
        ax.set_title(f"layer {layer} MLP-down: input spectrum")
        ax.set_xlabel("rank")
        ax.set_ylabel("eigenvalue / mean eigenvalue")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=8)

        ax = axes[1, col]
        ax.scatter(x, (diag["block-diagonal"] / diag["full"]).numpy(), s=3, color="#e6550d", alpha=0.5,
                   label="block-diagonal R", rasterized=True)
        ax.scatter(x, (diag["block-diagonal + top-1 exact"] / diag["full"]).numpy(), s=3, color="#3182bd",
                   alpha=0.5, label="block-diagonal + top-1 exact", rasterized=True)
        order = x.argsort()
        ax.plot(x[order], (diag["Muon (R = I)"] / diag["full"]).numpy()[order], color="#636363", lw=1.4,
                label="Muon (R = I)")
        ax.scatter(c_var, c_ratio, s=60, marker="*", color="#a50f15", zorder=5,
                   label="block-contrast directions (block-diag.)")
        ax.axhline(1, color="k", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(f"layer {layer}: scaling of C's eigen-directions / full PD")
        ax.set_xlabel("eigenvalue of C / mean (direction's input energy)")
        ax.set_ylabel("step scaling relative to full-R PD")
        ax.grid(alpha=0.3, which="both")
        if col == 0:
            ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle(f"PD's MLP-down preconditioner (α = {ALPHA}): full input statistic vs its four 768×768 diagonal "
                 f"blocks. Track 3 run {RUN[:8]} (PD + geometry decay, 3150 steps), statistics at the end of training",
                 fontsize=11)
    fig.text(0.01, 0.005, "Bottom: diagonal of each R in C's eigenbasis, each R normalized to unit mean eigenvalue, "
             "divided by full R's. Above 1: that direction gets a larger step than full PD gives it (relative to the "
             "other directions); below 1: smaller. Muon's line is PD's reweighting undone.", fontsize=8,
             color="#555555")
    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    out = HERE / "figures" / ("blockdiag_mlp_down.png" if RUN == DEFAULT_RUN else f"blockdiag_mlp_down_{RUN[:8]}.png")
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()

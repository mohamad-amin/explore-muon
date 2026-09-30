"""Diagonal PD: how far is R = (diag(C) / mean + 1e-3)^(-alpha) from full PD, per input type?

On the final PD statistics of a Track 3 run, for each hidden-matrix input (attention q/k/v input, attention-proj input,
MLP-fc input, MLP-down input) in every layer:
  - cos between the PD steps polar(M R) R with full and diagonal R, and with Muon (R = I), for data-like momentum
    (rows ~ N(0, C), shape d_out x d_in);
  - the full-R exponent whose step the diagonal step resembles most;
  - how coordinate-aligned C's top eigenvector is (participation ratio 1 / sum u_j^4: ~1 means one channel, d means spread).
A step probe describes what the change is; whether it helps is a training question.

Usage: python track3/diag_probe.py [RUN_ID]   (default: 2124499a, PD + geometry decay, alpha 1/8, 3150 steps)
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
RUN = sys.argv[1] if len(sys.argv) > 1 else "2124499a-f8e9-4a2d-ad7c-b0874a698ebb"
ALPHA, DAMPING = 0.125, 1e-3
KINDS = {"attn.q": ("q/k/v input", 768), "attn.proj": ("attn-proj input", 768), "mlp.fc": ("MLP-fc input", 3072),
         "mlp.proj": ("MLP-down input", 768)}   # kind -> (label, d_out)
torch.set_num_threads(16)


def step(M, r):
    u, _, vh = torch.linalg.svd(M @ r, full_matrices=False)
    d = (u @ vh) @ r
    return d / d.norm()


def main():
    stats = torch.load(HERE / "logs" / f"{RUN}_final.pt", map_location="cpu", weights_only=False)["pd_stats"]
    torch.manual_seed(0)
    grid = [a / 200 for a in range(0, 26)]            # full-R exponents 0 .. 0.125
    print(f"run {RUN[:8]}, alpha {ALPHA}; per input type, range over the 12 layers (layer of min in brackets)")
    for kind, (label, d_out) in KINDS.items():
        rows = []
        for layer in range(12):
            s = stats[f"blocks.{layer}.{kind}"]
            C = (s["cov"] / s["weight"].clamp_min(1e-12)).double()
            C = 0.5 * (C + C.T)
            ev, U = torch.linalg.eigh(C)
            ev, U = ev.flip(0).clamp_min(0), U.flip(1)
            mean = ev.mean()
            power = lambda a: (U * (ev / mean + DAMPING).pow(-a)) @ U.T
            full = power(ALPHA)
            diag = torch.diag((C.diagonal().clamp_min(0) / mean + DAMPING).pow(-ALPHA))
            M = torch.randn(d_out, C.shape[0], dtype=C.dtype) @ ((U * ev.sqrt()) @ U.T)
            ref, dg, mu = step(M, full), step(M, diag), step(M, torch.eye(C.shape[0], dtype=C.dtype))
            fits = max((float((dg * step(M, power(a))).sum()), a) for a in grid)
            rows.append(dict(layer=layer, diag=float((ref * dg).sum()), muon=float((ref * mu).sum()),
                             fit_a=fits[1], fit_cos=fits[0], top=float(ev[0] / mean),
                             pr=float(1 / U[:, 0].pow(4).sum()), dmax=float(C.diagonal().max() / mean)))
        def rng(key, fmt):
            v = [r[key] for r in rows]
            lo = min(rows, key=lambda r: r[key])
            return f"{fmt.format(min(v))}–{fmt.format(max(v))} [{lo['layer']}]"
        print(f"\n{label} (d_in {C.shape[0]}):")
        print(f"  cos(full, diagonal) {rng('diag', '{:.4f}')}   cos(full, Muon) {rng('muon', '{:.4f}')}")
        print(f"  diagonal step closest to full R at exponent {rng('fit_a', '{:.3f}')} (cos {rng('fit_cos', '{:.4f}')})")
        print(f"  top eigenvalue / mean {rng('top', '{:.0f}')}; top eigenvector participation ratio "
              f"{rng('pr', '{:.1f}')}; largest diagonal entry / mean {rng('dmax', '{:.0f}')}")
        for r in rows:
            print(f"    layer {r['layer']:2d}: diag {r['diag']:.4f}  Muon {r['muon']:.4f}  fit a {r['fit_a']:.3f} "
                  f"({r['fit_cos']:.4f})  top/mean {r['top']:6.0f}  PR {r['pr']:7.1f}  max diag/mean {r['dmax']:6.0f}")


if __name__ == "__main__":
    main()

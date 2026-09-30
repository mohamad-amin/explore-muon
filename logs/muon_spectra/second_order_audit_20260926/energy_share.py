import sys, torch, numpy as np
path = sys.argv[1]
d = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
stats = d["pd_stats"]
alphas = [0, 0.125, 0.25, 0.375, 0.5]
rows = []
for name, s in stats.items():
    cov = (s["cov"] / s["weight"].clamp_min(1e-12)).double()
    ev = torch.linalg.eigvalsh(0.5 * (cov + cov.T)).clamp_min(0).flip(0).numpy()
    unit = ev / ev.mean()
    cum = np.cumsum(ev) / ev.sum()
    k99 = int(np.searchsorted(cum, 0.99)) + 1
    k90 = int(np.searchsorted(cum, 0.90)) + 1
    pr = ev.sum() ** 2 / (ev ** 2).sum()
    out = {"name": name, "n": len(ev), "k90": k90, "k99": k99, "pr": pr, "top/med": ev[0] / np.median(ev)}
    for a in alphas:
        r2 = (unit + 1e-3) ** (-2 * a)
        share = r2 / r2.sum()
        out[f"E99_{a}"] = share[:k99].sum()      # update energy on the 99%-variance span (isotropic polar output)
        out[f"E90_{a}"] = share[:k90].sum()
        # variance-weighted output energy E||dW x||^2 relative to Muon (alpha 0), same Frobenius norm
        out[f"out_{a}"] = (share * ev).sum() / (ev.mean())
    rows.append(out)
for r in rows:
    print(f"{r['name']:<22} n={r['n']:<5} k90={r['k90']:<5} k99={r['k99']:<5} pr={r['pr']:6.1f} top/med={r['top/med']:9.0f} | "
          + " ".join(f"E99[{a}]={r[f'E99_{a}']:.2f}" for a in alphas) + " | "
          + " ".join(f"out[{a}]={r[f'out_{a}']:.3f}" for a in alphas))
print("median over modules:")
for a in alphas:
    print(a, "E99", np.median([r[f'E99_{a}'] for r in rows]), "E90", np.median([r[f'E90_{a}'] for r in rows]), "outputE/Muon", np.median([r[f'out_{a}'] for r in rows]))

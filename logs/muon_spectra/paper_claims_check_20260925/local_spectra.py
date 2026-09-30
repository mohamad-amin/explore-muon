"""Read-only checks on saved local Muon pre-NS momentum spectra (8- and 12-layer, width 512).

Uses archived exact singular values; no training and no new SVD of model tensors.
Run from the project root with the shared .venv.
"""
import glob, os, re
import numpy as np

ROOT = 'logs/muon_spectra'
RUNS = {8: f'{ROOT}/depth8_w512_20260925_r2/scientific/spectra_momentum',
        12: f'{ROOT}/depth12_w512_20260925_r2/scientific/spectra_momentum'}
NS = [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037),
      (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)]


def f_ns(x):
    x = np.asarray(x, float)
    for a, b, c in NS:
        x = a * x + b * x**3 + c * x**5
    return x


def load(depth):
    out = {}
    for path in sorted(glob.glob(os.path.join(RUNS[depth], 'step*.npz'))):
        z = np.load(path)
        out[int(re.search(r'step(\d+)', path).group(1))] = {k: np.sort(z[k].astype(float))[::-1] for k in z.files}
    return out


def stats(s):
    s = s / np.sqrt(np.sum(s**2)); r = len(s)
    q = {p: s[int(np.ceil(p * r)) - 1] for p in (0.1, 0.5, 0.9)}   # paper convention, descending
    return dict(rho1=s[0]**2, med=q[0.5], r10=q[0.1] / q[0.5], r90=q[0.9] / q[0.5], bulk=1 - s[0]**2,
                bulk_exp=r * 2 * (q[0.5] / np.log(2))**2, bulk_nuc=(s.sum() - s[0]) / s.sum(),
                ns=np.sum(s * f_ns(s)) / s.sum())


def window(data, m, lo, hi, key):
    return np.mean([stats(data[t][m])[key] for t in sorted(data) if lo <= t <= hi])


def main():
    D = {d: load(d) for d in (8, 12)}
    for d in (8, 12):
        print(f'== depth {d}: paper window (steps 1300-1469) means; LR cooldown starts near step 1322')
        print('   matrix        rho1   median   q.1/q.5  q.9/q.5  bulkE  bulkE(exp.assump.)  bulk nuclear  NS retained')
        for m in sorted(D[d][1469]):
            a = {k: window(D[d], m, 1300, 1469, k) for k in ('rho1', 'med', 'r10', 'r90', 'bulk', 'bulk_exp', 'bulk_nuc', 'ns')}
            print(f"   {m:12s} {a['rho1']:.3f}  {a['med']:.2e}  {a['r10']:6.2f}  {a['r90']:6.3f}   {a['bulk']:.3f}   {a['bulk_exp']:.3f}"
                  f"              {a['bulk_nuc']:.3f}         {a['ns']:.3f}")
        ratios = [window(D[d], m, 1300, 1469, 'med') / window(D[d], m, 1100, 1299, 'med') for m in sorted(D[d][1469])]
        print(f'   paper-window / pre-cooldown (1100-1299) median ratio: {min(ratios):.3f}-{max(ratios):.3f}, mean {np.mean(ratios):.3f}\n')
    print('== fixed width 512: 12-layer / 8-layer median ratio at matched relative depth (rho1 8 -> 12)')
    for kind in ('q', 'k', 'v', 'o', 'up', 'down'):
        cells = []
        for b8, b12, name in ((2, 3, 'early'), (4, 6, 'mid'), (6, 9, 'late'), (8, 12, 'final')):
            m8, m12 = f'block{b8:02d}.{kind}', f'block{b12:02d}.{kind}'
            cells.append(f"{name} x{window(D[12], m12, 1300, 1469, 'med') / window(D[8], m8, 1300, 1469, 'med'):.2f} "
                         f"({window(D[8], m8, 1300, 1469, 'rho1'):.2f}->{window(D[12], m12, 1300, 1469, 'rho1'):.2f})")
        print(f'   {kind:5s}' + ' | '.join(cells))


if __name__ == '__main__':
    main()

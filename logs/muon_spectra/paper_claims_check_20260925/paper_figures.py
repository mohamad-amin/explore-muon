"""Extract numbers from the vector figures of arXiv 2606.04058v2 (read-only).

Fig. 14 and Fig. 6: histogram bar geometry -> per-matrix bulk medians and energy
(validated: every panel sums to the expected 2559/2560 singular values).
Fig. 16: stabilization values (7 sizes x 5 quantiles x 24 panels).
Requires PyMuPDF (run from a separate environment; not installed in the shared .venv).
"""
import os, json
import numpy as np
import pymupdf

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, 'sources')
NS = [(4.0848, -6.8946, 2.9270), (3.9505, -6.3029, 2.6377), (3.7418, -5.5913, 2.3037),
      (2.8769, -3.1427, 1.2046), (2.8366, -3.0525, 1.2012)]   # paper A.2.1, f = b5 o ... o b1
SIZES = np.array([77e6, 160e6, 354e6, 600e6, 1.2e9, 1.6e9, 2.8e9])


def f_ns(x):
    x = np.asarray(x, float)
    for a, b, c in NS:
        x = a * x + b * x**3 + c * x**5
    return x


def drawings(name):
    return pymupdf.open(os.path.join(SRC, name))[0].get_drawings()


def axes_of(dr, lo, hi):
    return sorted([x['rect'] for x in dr if x.get('fill') == (1.0, 1.0, 1.0) and len(x['items']) == 1
                   and x['items'][0][0] == 're' and lo < x['rect'].width < hi], key=lambda r: (round(r.y0), r.x0))


def histograms(dr, axes, spacing):
    """Per panel: (left, right, count) of every bar; x from the first major tick (=0) and label spacing."""
    lines = [x for x in dr if x.get('type') == 'fs' and len(x['items']) == 1 and x['items'][0][0] == 'l']
    bars = [x['rect'] for x in dr if x.get('fill') and x.get('fill_opacity', 1) < 0.95
            and len(x['items']) == 1 and x['items'][0][0] == 're']
    out = []
    for ax, sp in zip(axes, spacing):
        yt = sorted({round(l['rect'].y0, 2) for l in lines if abs(l['rect'].x0 - ax.x0) < 0.5
                     and abs(l['rect'].width - 5) < 0.3 and ax.y0 - 1 <= l['rect'].y0 <= ax.y1 + 1}, reverse=True)
        xt = sorted({round(l['rect'].x0, 2) for l in lines if abs(l['rect'].height - 5) < 0.3
                     and abs(l['rect'].y1 - ax.y1) < 0.6 and ax.x0 - 1 <= l['rect'].x0 <= ax.x1 + 1})
        ppd, y0, ppu, x0 = yt[0] - yt[1], yt[0], (xt[1] - xt[0]) / sp, xt[0]
        bs = [b for b in bars if ax.x0 - 0.5 <= b.x0 and b.x1 <= ax.x1 + 0.5 and ax.y0 - 1 <= b.y0 < ax.y1]
        L = np.array([(b.x0 - x0) / ppu for b in bs]); R = np.array([(b.x1 - x0) / ppu for b in bs])
        raw = 10 ** ((y0 - np.array([b.y0 for b in bs])) / ppd); C = np.round(raw)
        o = np.argsort(L)
        out.append((L[o], R[o], C[o], np.abs(raw - C).max()))
    return out


def values(L, R, C):  # spread each bin's count uniformly inside the bin
    return np.clip(np.concatenate([np.linspace(l, r, int(c) + 2)[1:-1] for l, r, c in zip(L, R, C) if c > 0]), 0, None)


def median(L, R, C):
    cum = np.cumsum(C); n = cum[-1]; j = np.searchsorted(cum, n / 2)
    return L[j] + (n / 2 - (cum[j - 1] if j else 0)) / C[j] * (R[j] - L[j])


def main():
    # ---- Fig. 14: 2.8B, step 1450, largest singular value removed; labels as printed in Fig. 14
    rows, cols = ['Q', 'K', 'V', 'O', 'MLP Down', 'MLP Up'], ['L31', 'L23', 'L15', 'L7']
    spacing = [.025, .02, .02, .02, .05, .025, .025, .05, .01, .025, .02, .02,
               .01, .02, .02, .05, .01, .02, .02, .02, .01, .02, .02, .02]   # read from tick labels
    dr = drawings('svalue_hist_2.8B_1450_all_layers.svg')
    h14 = dict(zip([(r, c) for r in rows for c in cols], histograms(dr, axes_of(dr, 300, 600), spacing)))
    # ---- Fig. 16 stabilization values
    dr = drawings('scaling_laws_all_quantiles.svg')
    ax16 = axes_of(dr, 200, 800)
    names16 = [(r, c) for r in rows for c in ['Final', 'Mid-late', 'Mid', 'Mid-early']]
    colors = {(0.13, 0.4, 0.67): 'q0.1', (0.3, 0.67, 0.15): 'q0.25', (0.84, 0.38, 0.3): 'q0.5',
              (0.56, 0.27, 0.68): 'q0.75', (0.9, 0.49, 0.13): 'q0.9'}
    grid = [x for x in dr if x.get('type') == 's' and tuple(round(v, 2) for v in (x.get('color') or ())) == (0.83, 0.83, 0.83)]
    polys = [x for x in dr if x.get('type') == 's' and len(x['items']) == 6]
    f16 = {}
    for (r, c), ax in zip(names16, ax16):
        gy = sorted({round(g['items'][0][1].y, 3) for g in grid if ax.x0 - 1 <= g['rect'].x0 <= ax.x1 + 1
                     and ax.y0 - 1 <= g['rect'].y0 <= ax.y1 + 1}, reverse=True)
        A = np.polyfit(gy, np.log10([1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]), 1)
        for p in polys:
            if not (ax.x0 - 1 <= p['rect'].x0 and p['rect'].x1 <= ax.x1 + 1 and ax.y0 - 1 <= p['rect'].y0 and p['rect'].y1 <= ax.y1 + 1):
                continue
            pts = [p['items'][0][1]] + [it[2] for it in p['items']]
            ys = np.polyval(A, [q.y for q in pts]); xs = np.array([q.x for q in pts])
            if np.abs(np.polyval(np.polyfit(xs, ys, 1), xs) - ys).max() > 1e-3:      # data line, not the fit
                f16[f'{c}|{r}|{colors[tuple(round(v, 2) for v in p["color"])]}'] = (10 ** ys).tolist()
    json.dump(f16, open(os.path.join(HERE, 'fig16_points.json'), 'w'), indent=0)

    print('== Fig. 14 (2.8B, step 1450) bulk medians vs Fig. 16 2.8B q=0.5 (labels as printed in each figure)')
    depth = {'L31': 'Final', 'L23': 'Mid-late', 'L15': 'Mid', 'L7': 'Mid-early'}
    for r in rows:
        for c in cols:
            L, R, C, err = h14[(r, c)]
            same = f16[f'{depth[c]}|{r}|q0.5'][-1]
            other = {'MLP Down': 'MLP Up', 'MLP Up': 'MLP Down'}.get(r)
            txt = f'  {c} {r:9s} Fig14 median {median(L, R, C):.2e} (n={C.sum():.0f}, max int err {err:.2f}) | Fig16 same label {same:.2e}'
            if other:
                txt += f' | Fig16 other MLP label {f16[f"{depth[c]}|{other}|q0.5"][-1]:.2e}'
            print(txt)

    print('\n== Fig. 6 (caption: 2.8B; panel titles: Layer 20 Attn Q, Layer 27 Attn O, Layer 27 MLP proj)')
    dr = drawings('svalue_hist_2.8B_1450_combined.svg')
    labels = ['L20 Attn Q full', 'L27 Attn O full', 'L27 MLP proj full', 'L20 Attn Q bulk', 'L27 Attn O bulk', 'L27 MLP proj bulk']
    for lab, (L, R, C, err) in zip(labels, histograms(dr, axes_of(dr, 150, 600), [0.1, 0.2, 0.2, 0.025, 0.01, 0.02])):
        txt = f'  {lab:18s} n={C.sum():.0f} median {median(L, R, C):.2e}'
        if 'full' in lab:
            top = (L[-1] + R[-1]) / 2
            txt += f' | outlier sigma1/||M||_F ~ {top:.3f}, rho1 ~ {top**2:.3f} (+- one bin)'
        print(txt)

    print('\n== 2.8B first-order descent retained by 5-step NS, relative to exact polar (Fig. 14 spectra; sigma1 from unit Frobenius norm)')
    print('   panel              E_bulk  rho1   bulk nuclear  plain NS  rank-1 deflated  Schatten-4 norm  frac f<0.5')
    for (r, c), (L, R, C, _) in h14.items():
        b = values(L, R, C); s1 = np.sqrt(max(1 - np.sum(b**2), 0)); s = np.concatenate([[s1], b]); nuc = s.sum()
        plain = np.sum(s * f_ns(s)) / nuc
        defl = (s1 + np.sum(b * f_ns(b / np.sqrt(np.sum(b**2))))) / nuc
        s4 = np.sum(s * f_ns(s / np.sum(s**4) ** 0.25)) / nuc
        print(f'   {c} {r:9s}      {np.sum(b**2):.3f}  {s1**2:.3f}   {(nuc - s1) / nuc:.3f}       {plain:.3f}     {defl:.3f}            {s4:.3f}           {np.mean(f_ns(b) < 0.5):.3f}')

    print('\n== Fig. 16: exponents (q=0.5, all sizes) and local slopes between consecutive sizes')
    lx = np.log(SIZES)
    for r in rows:
        for c in ['Final', 'Mid-late', 'Mid', 'Mid-early']:
            v = np.array(f16[f'{c}|{r}|q0.5']); loc = np.diff(np.log(v)) / np.diff(lx)
            qs = [np.polyfit(lx, np.log(f16[f'{c}|{r}|{q}']), 1)[0] for q in ('q0.1', 'q0.25', 'q0.5', 'q0.75', 'q0.9')]
            print(f'   {c + " " + r:20s} slope {np.polyfit(lx, np.log(v), 1)[0]:6.2f} | local ' + ' '.join(f'{x:6.2f}' for x in loc)
                  + ' | per-quantile ' + ' '.join(f'{x:5.2f}' for x in qs))
    w = np.sqrt([512, 768, 1024, 1280, 1792, 2048, 2560])
    print('   width-only 1/sqrt(d)            | local ' + ' '.join(f'{x:6.2f}' for x in -np.diff(np.log(w)) / np.diff(lx)))
    print(f'   d ~ M^{np.polyfit(lx, np.log([512, 768, 1024, 1280, 1792, 2048, 2560]), 1)[0]:.3f}; '
          f'slope change from a 5% bias at 77M: {((lx - lx.mean()) / np.sum((lx - lx.mean())**2))[0] * np.log(0.95):.4f}')
    print('   NS slope at 0:', round(float(np.prod([a for a, _, _ in NS])), 1), '| f(1e-3) =', round(float(f_ns(1e-3)), 3))


if __name__ == '__main__':
    main()

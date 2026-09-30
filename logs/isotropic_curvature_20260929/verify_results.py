"""Independent scalar/NPZ verification; never imports a model or atlas analyzer.

Default: require all 18 panels, all six joint states, and completed analyzer CSVs.
--partial: verify only finalized raw files, explicitly without CSV/completion claims.
Print JSON to stdout; --out additionally writes a NEW JSON file, never overwriting.
All numerical libraries are limited to one thread before NumPy is imported.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import numpy as np

HERE = Path(__file__).resolve().parent
METHODS, STEPS, BLOCKS = ('Muon', 'PD'), (10, 500, 1300), (1, 4, 8)
LABELS = ('actual', 'left0', 'left1', 'right_raw', 'right_white')
SCALES = np.array([-16, -8, -4, -2, -1, -.5, -.25, 0, .25, .5, 1, 2, 4, 8, 16.])
GROUPS = {**{f'context{i}': [i] for i in range(4)},
          'bank0': [0, 1], 'bank1': [2, 3], 'all4': [0, 1, 2, 3]}
EXPECTED = {(m, s, b) for m in METHODS for s in STEPS for b in BLOCKS}


def demand(condition, label):
    if not condition:
        raise AssertionError(label)


def read_json(path):
    return json.loads(path.read_text())


def record(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    after = path.stat()
    demand((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
           f'File changed while hashing: {path}')
    return {'path': str(path), 'bytes': after.st_size, 'mtime_ns': after.st_mtime_ns,
            'sha256': digest.hexdigest()}


class Check:
    def __init__(self):
        self.stats = {}

    def close(self, category, actual, expected, label, atol=2e-12, rtol=2e-8):
        a, b = np.broadcast_arrays(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float))
        demand(np.isfinite(a).all() and np.isfinite(b).all(), label + ': nonfinite comparison')
        err = np.abs(a - b)
        limit = atol + rtol * np.abs(b)
        maximum = float(err.max(initial=0))
        stat = self.stats.setdefault(category, {'comparisons': 0, 'elements': 0,
                                               'max_absolute_error': 0., 'max_tolerance_fraction': 0.})
        stat['comparisons'] += 1
        stat['elements'] += int(a.size)
        stat['max_absolute_error'] = max(stat['max_absolute_error'], maximum)
        stat['max_tolerance_fraction'] = max(stat['max_tolerance_fraction'],
                                             float((err / np.maximum(limit, 1e-300)).max(initial=0)))
        demand(bool(np.all(err <= limit)), f'{label}: max absolute error {maximum:.5g}')

    def fields(self, row, expected, label):
        for key, value in expected.items():
            demand(key in row, f'{label}: missing CSV field {key}')
            if value is None:
                demand(row[key] == '', f'{label}/{key}: expected empty unqualified ratio')
            elif isinstance(value, (bool, np.bool_)):
                demand(row[key] == str(bool(value)), f'{label}/{key}: Boolean mismatch')
            else:
                # Ratios of near-zero remainders amplify float64 summation order.
                # The numerator/denominator are separately checked at 2e-12.
                atol = 5e-5 if ('relative_' in key or key == 'odd_signed_fraction') else 2e-12
                self.close('csv/' + key, float(row[key]), value, f'{label}/{key}', atol=atol)


def arrays(path, shapes):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    demand(set(shapes) <= set(data), f'{path}: missing arrays')
    for key, value in data.items():
        if np.issubdtype(value.dtype, np.number):
            demand(np.isfinite(value).all(), f'{path}/{key}: nonfinite raw array')
    for key, shape in shapes.items():
        demand(data[key].shape == shape, f'{path}/{key}: shape {data[key].shape} != {shape}')
    demand(np.array_equal(data['scales'], SCALES), f'{path}: incorrect radius grid')
    return data


def table(path, keys, count):
    with path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    demand(len(rows) == count, f'{path}: row count {len(rows)} != {count}')
    index = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        demand(key not in index, f'{path}: duplicate row {key}')
        index[key] = row
    return index


def curve_fields(losses, slopes, h, g):
    """Form per-token remainders first; independent of analyzer's reduction order."""
    rem = (losses - losses[:, 7:8, :] - slopes[:, None, :] * SCALES[None, :, None]).mean(axis=(0, 2))
    means = losses.mean(axis=(0, 2))
    base = float(losses[:, 7].mean())
    linear = SCALES * slopes.mean()
    qh, qg = SCALES**2 * h / 2, SCALES**2 * g / 2
    dh, dg = np.maximum.reduce([np.abs(rem), np.abs(qh), np.full(15, 1e-10)]), \
             np.maximum.reduce([np.abs(rem), np.abs(qg), np.full(15, 1e-10)])
    return {'loss': means, 'loss_change': means - base, 'true_linear_term': linear,
            'linear_prediction': base + linear, 'remainder': rem,
            'hessian_quadratic': qh, 'gn_quadratic': qg,
            'hessian_departure': rem - qh, 'gn_departure': rem - qg,
            'hessian_departure_denominator': dh, 'gn_departure_denominator': dg,
            'signed_relative_hessian_departure': (rem - qh) / dh,
            'signed_relative_gn_departure': (rem - qg) / dg}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE / 'run1')
    parser.add_argument('--analysis', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    run = args.run.resolve()
    analysis = (args.analysis or run / 'analysis').resolve()
    if args.out:
        demand(not args.out.exists(), f'Output already exists: {args.out}')
    checks = Check()
    source = [record(Path(__file__))]
    tables = {}
    if not args.partial:
        for path in (run / 'status.json', run / 'result.json', analysis / 'status.json', analysis / 'result.json'):
            demand(read_json(path)['status'] == 'complete', f'Not complete: {path}')
            source.append(record(path))
        demand(not (run / 'failure.json').exists(), 'Producer failure.json exists')
        for name, keys, count in (
            ('curvature', ('method', 'step', 'block', 'aggregate', 'direction'), 18 * 7 * 5),
            ('curves', ('method', 'step', 'block', 'aggregate', 'direction', 'signed_scale'), 18 * 7 * 5 * 15),
            ('radius_symmetry', ('method', 'step', 'block', 'aggregate', 'direction', 'abs_scale'), 18 * 7 * 5 * 7),
            ('joint', ('method', 'step', 'aggregate', 'signed_scale'), 6 * 7 * 15),
            ('projected_curvature', ('method', 'step', 'block', 'aggregate', 'direction_i', 'direction_j'),
             18 * 7 * 25 + 6 * 7 * 9)):
            path = analysis / f'{name}.csv'
            tables[name] = table(path, keys, count)
            source.append(record(path))
    panel_shapes = {'loss': (4, 5, 15, 512), 'slope': (4, 5, 512), 'hessian': (4, 5, 512),
                    'gn': (4, 5, 512), 'activation_radius': (4, 5, 512),
                    'projected_hessian': (4, 5, 5), 'projected_gn': (4, 512, 5, 5)}
    joint_shapes = {'loss': (4, 15, 512), 'slope': (4, 512), 'hessian': (4, 512), 'gn': (4, 512),
                    'projected_hessian': (4, 3, 3), 'projected_gn': (4, 512, 3, 3),
                    'individual_loss': (3, 4, 15, 512), 'individual_slope': (3, 4, 512),
                    'individual_hessian': (3, 4, 512), 'individual_gn': (3, 4, 512)}
    panels, actuals, panel_results, joint_results = set(), {}, [], []
    for method, step, block in sorted(EXPECTED):
        directory = run / f'{method}_{step:06d}' / f'block{block:02d}_up'
        path = directory / 'per_token.npz'
        if args.partial and not (path.exists() and (directory / 'summary.json').exists()):
            continue
        data = arrays(path, panel_shapes)
        source.extend([record(path), record(directory / 'summary.json')])
        summary = read_json(directory / 'summary.json')
        demand((summary['method'], summary['step'], summary['block']) == (method, step, block), f'{path}: identity')
        demand(tuple(data['labels']) == LABELS, f'{path}: direction labels')
        panels.add((method, step, block))
        h, g = data['projected_hessian'], data['projected_gn']
        for key, value in [('H', h), ('GN', g)]:
            checks.close('raw/symmetry', value, value.swapaxes(-1, -2), f'{path}/{key} symmetry', atol=1e-10)
        checks.close('raw/H_diagonal', np.diagonal(h, axis1=-2, axis2=-1), data['hessian'].mean(-1),
                     str(path), atol=1e-10, rtol=1e-7)
        checks.close('raw/GN_diagonal', np.diagonal(g, axis1=-2, axis2=-1).transpose(0, 2, 1), data['gn'],
                     str(path), atol=1e-10, rtol=1e-7)
        checks.close('raw/zero_baseline', data['loss'][:, :, 7], data['loss'][:, 0:1, 7], str(path))
        demand(np.min(data['activation_radius']) >= 0, str(path) + ': negative norm')
        demand(np.min(data['gn']) >= -1e-12, str(path) + ': negative predictive GN')
        for i in (1, 2):
            checks.close('raw/left_norm', data['activation_radius'][:, i], data['activation_radius'][:, 0],
                         str(path), atol=1e-13, rtol=1e-12)
        actuals[(method, step, block)] = {key: data[key][:, 0].copy() for key in ('loss', 'slope', 'hessian', 'gn')}
        for group, indices in GROUPS.items():
            prefix = (method, str(step), str(block), group)
            mh = data['hessian'][indices].mean(axis=(0, 2))
            mg = data['gn'][indices].mean(axis=(0, 2))
            panel_row = {'method': method, 'step': step, 'block': block, 'aggregate': group,
                         'actual_hessian': float(mh[0]), 'actual_gn': float(mg[0]),
                         'actual_over_mean_left_hessian': float(mh[0] / mh[1:3].mean()) if mh[1:3].mean() != 0 else None,
                         'actual_over_mean_left_gn': float(mg[0] / mg[1:3].mean()) if mg[1:3].mean() != 0 else None}
            for di, direction in enumerate(LABELS):
                ht, gt = data['hessian'][indices, di], data['gn'][indices, di]
                hmean, gmean = float(ht.mean()), float(gt.mean())
                rms = float(np.sqrt(np.square(data['activation_radius'][indices, di]).mean()))
                qualified = gmean > 1e-12
                static = {'slope': float(data['slope'][indices, di].mean()), 'hessian': hmean, 'gn': gmean,
                          'baseline_loss': float(data['loss'][indices, di, 7].mean()),
                          'hessian_minus_gn': hmean - gmean,
                          'hessian_over_gn': hmean / gmean if qualified else None,
                          'h_over_gn_qualified': qualified,
                          'activation_rms_per_position': rms, 'activation_sequence_norm_rms': np.sqrt(512) * rms,
                          'parameter_direction_norm': summary['direction_norms'][direction],
                          'token_distribution_count': ht.size,
                          'hessian_token_negative_fraction': float((ht < 0).mean()),
                          'mean_abs_token_hessian_minus_gn': float(np.abs(ht - gt).mean()),
                          'mean_abs_token_hessian_minus_gn_over_mean_gn': float(np.abs(ht - gt).mean() / gmean) if qualified else None,
                          'abs_token_h_minus_gn_over_mean_gn_qualified': qualified}
                for name, values in [('hessian', ht), ('gn', gt)]:
                    for q, v in zip((10, 50, 90), np.percentile(values, (10, 50, 90))):
                        static[f'{name}_token_q{q}'] = float(v)
                fields = curve_fields(data['loss'][indices, di], data['slope'][indices, di], hmean, gmean)
                key = prefix + (direction,)
                if tables:
                    checks.fields(tables['curvature'][key], static, str(key))
                    for si, scale in enumerate(SCALES):
                        checks.fields(tables['curves'][key + (str(float(scale)),)],
                                      {**static, **{k: v[si] for k, v in fields.items()}}, str(key) + f'/{scale}')
                for si in range(8, 15):
                    magnitude = SCALES[si]
                    rp, rm = fields['remainder'][si], fields['remainder'][14 - si]
                    even, odd = (rp + rm) / 2, (rp - rm) / 2
                    qh = fields['hessian_quadratic'][si]
                    den_even, den_odd = max(abs(even), abs(qh), 1e-10), max(abs(rp) + abs(rm), 1e-10)
                    sf = {'R_plus': rp, 'R_minus': rm, 'R_even': even, 'R_odd': odd,
                          'hessian_quadratic': qh, 'even_hessian_departure': even - qh,
                          'even_hessian_departure_denominator': den_even,
                          'signed_relative_even_hessian_departure': (even - qh) / den_even,
                          'odd_fraction_denominator': den_odd, 'odd_signed_fraction': (rp - rm) / den_odd}
                    if tables:
                        checks.fields(tables['radius_symmetry'][key + (str(float(magnitude)),)],
                                      {**static, **sf}, str(key) + f'/paired{magnitude}')
                    if di == 0 and magnitude in (1, 16):
                        panel_row[f'actual_s{int(magnitude)}'] = {k: float(v) for k, v in sf.items()}
                if di == 0:
                    panel_row.update({k: static[k] for k in ('hessian_over_gn', 'mean_abs_token_hessian_minus_gn_over_mean_gn',
                                                            'hessian_token_negative_fraction', 'activation_rms_per_position')})
            if tables:
                for i, di in enumerate(LABELS):
                    for j, dj in enumerate(LABELS):
                        checks.fields(tables['projected_curvature'][prefix + (di, dj)],
                                      {'hessian': h[indices, i, j].mean(), 'gn': g[indices, :, i, j].mean()}, str(prefix))
            panel_results.append(panel_row)
    if not args.partial:
        demand(panels == EXPECTED, 'Incomplete 18-panel coverage')
    states = []
    for method in METHODS:
        for step in STEPS:
            directory = run / f'{method}_{step:06d}'
            path = directory / 'joint.npz'
            if args.partial and not (path.exists() and (directory / 'status.json').exists()):
                continue
            demand(read_json(directory / 'status.json')['status'] == 'complete', f'{directory}: incomplete state')
            demand(all((method, step, b) in actuals for b in BLOCKS), f'{directory}: missing individual panels')
            data = arrays(path, joint_shapes)
            source.extend([record(path), record(directory / 'status.json')])
            demand(tuple(data['blocks']) == (0, 3, 7), f'{path}: block order')
            states.append((method, step))
            h, g = data['projected_hessian'], data['projected_gn']
            for key, value in [('H', h), ('GN', g)]:
                checks.close('joint/symmetry', value, value.swapaxes(-1, -2), f'{path}/{key}', atol=1e-10)
            for bi, block in enumerate(BLOCKS):
                a = actuals[(method, step, block)]
                for field in ('loss', 'slope', 'hessian', 'gn'):
                    checks.close('joint/individual_archive', data['individual_' + field][bi], a[field], str(path))
                checks.close('joint/baseline', data['loss'][:, 7], a['loss'][:, 7], str(path))
            demand(data['gn'].min() >= -1e-12, f'{path}: negative joint GN')
            checks.close('joint/slope_additivity', data['slope'], data['individual_slope'].sum(0), str(path), atol=1e-10)
            checks.close('joint/H_sum', h.sum(axis=(1, 2)), data['hessian'].mean(-1), str(path), atol=1e-10, rtol=1e-7)
            checks.close('joint/GN_sum', g.sum(axis=(2, 3)), data['gn'], str(path), atol=1e-10, rtol=1e-7)
            checks.close('joint/H_diagonal', np.diagonal(h, axis1=-2, axis2=-1), data['individual_hessian'].mean(-1).T,
                         str(path), atol=1e-10, rtol=1e-7)
            checks.close('joint/GN_diagonal', np.diagonal(g, axis1=-2, axis2=-1), data['individual_gn'].transpose(1, 2, 0),
                         str(path), atol=1e-10, rtol=1e-7)
            for group, indices in GROUPS.items():
                prefix = (method, str(step), group)
                hc, gc = h[indices].mean(0), g[indices].mean(axis=(0, 1))
                offdiag = ~np.eye(3, dtype=bool)
                crossh, crossg = hc[offdiag].sum(), gc[offdiag].sum()
                hm, gm = data['hessian'][indices].mean(), data['gn'][indices].mean()
                cf = curve_fields(data['loss'][indices], data['slope'][indices], hm, gm)
                # Individual remainders are reconstructed from independently loaded panel files.
                summed = np.zeros(15)
                for b in BLOCKS:
                    a = actuals[(method, step, b)]
                    summed += curve_fields(a['loss'][indices], a['slope'][indices],
                                           a['hessian'][indices].mean(), a['gn'][indices].mean())['remainder']
                interaction = cf['remainder'] - summed
                qcross = SCALES**2 * crossh / 2
                den = np.maximum.reduce([np.abs(interaction), np.abs(qcross), np.full(15, 1e-10)])
                jf = {**cf, 'sum_individual_remainders': summed, 'interaction': interaction,
                      'hessian_cross_prediction': qcross, 'gn_cross_prediction': SCALES**2 * crossg / 2,
                      'interaction_hessian_departure': interaction - qcross,
                      'interaction_departure_denominator': den,
                      'signed_relative_interaction_departure': (interaction - qcross) / den}
                if tables:
                    for si, scale in enumerate(SCALES):
                        checks.fields(tables['joint'][prefix + (str(float(scale)),)],
                                      {'hessian': hm, 'gn': gm, 'hessian_cross': crossh, 'gn_cross': crossg,
                                       **{k: v[si] for k, v in jf.items()}}, str(prefix) + f'/{scale}')
                    for i, bi in enumerate(BLOCKS):
                        for j, bj in enumerate(BLOCKS):
                            key = (method, str(step), 'joint', group, f'block{bi:02d}_actual', f'block{bj:02d}_actual')
                            checks.fields(tables['projected_curvature'][key], {'hessian': hc[i, j], 'gn': gc[i, j]}, str(key))
                joint_results.append({'method': method, 'step': step, 'aggregate': group,
                                      'hessian_cross': float(crossh), 'gn_cross': float(crossg),
                                      'joint_H_over_diagonal_sum': float(hc.sum() / np.trace(hc)) if np.trace(hc) != 0 else None,
                                      'joint_GN_over_diagonal_sum': float(gc.sum() / np.trace(gc)) if np.trace(gc) != 0 else None,
                                      'interaction_by_signed_scale': {str(float(s)): float(v) for s, v in zip(SCALES, interaction)},
                                      'quadratic_cross_prediction_by_signed_scale': {str(float(s)): float(v) for s, v in zip(SCALES, qcross)}})
    if not args.partial:
        demand(len(states) == 6, 'Incomplete six-state joint coverage')
    for entry in source:
        stat = Path(entry['path']).stat()
        demand((stat.st_size, stat.st_mtime_ns) == (entry['bytes'], entry['mtime_ns']),
               'Input changed during verification: ' + entry['path'])
    result = {'status': 'partial_raw_checks_passed' if args.partial else 'complete_verification_passed',
              'partial': args.partial, 'panels_checked': len(panels), 'joint_states_checked': len(states),
              'csv_tables_compared': sorted(tables), 'checks': checks.stats,
              'panel_aggregates': panel_results, 'joint_aggregates': joint_results,
              'input_records': source, 'seconds': time.monotonic() - start,
              'scope': 'Archived scalar/NPZ identities and independent numerical reductions only; no rerun of AD or model.',
              'ratio_roundoff_rule': 'Finite-loss ratios allow 5e-5 absolute error because a 1e-10 denominator amplifies float64 reduction-order error; raw numerators and denominators independently require 2e-12 + 2e-8 relative.',
              'limits': 'Four contexts; restricted spans; actual/left ratios are descriptive, not optimizer-quality claims.'}
    rendered = json.dumps(result, indent=2, allow_nan=False) + '\n'
    if args.out:
        with args.out.open('x') as stream:
            stream.write(rendered)
    print(rendered, end='')


if __name__ == '__main__':
    main()

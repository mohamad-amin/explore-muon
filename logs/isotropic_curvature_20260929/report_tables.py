"""Compact complete-atlas report tables, derived solely from retained reductions.

No model, statistical test, fit, or cell selection. All 18 matrix panels and
six joint states are represented; these tables are retrospective summaries.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import csv
import hashlib
import json
import math

HERE = Path(__file__).resolve().parent
RUN = HERE / 'run1'
OUT = RUN / 'report_tables'


def read_csv(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def write_csv(name, rows):
    with (OUT / name).open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def val(row, key):
    value = float(row[key])
    assert math.isfinite(value), (row, key)
    return value


def divide(num, den):
    return num / den if den != 0 else None


def range_of(rows, field):
    values = [r[field] for r in rows if r[field] is not None]
    return dict(min=min(values), max=max(values), count=len(values))


def main():
    for directory in ('analysis', 'geometry_analysis', 'transmission_analysis'):
        assert json.loads((RUN / directory / 'result.json').read_text())['status'] == 'complete'
    OUT.mkdir(exist_ok=False)
    curvature = read_csv(RUN / 'analysis/curvature.csv')
    symmetry = read_csv(RUN / 'analysis/radius_symmetry.csv')
    curves = read_csv(RUN / 'analysis/curves.csv')
    joint = read_csv(RUN / 'analysis/joint.csv')
    transmission = read_csv(RUN / 'transmission_analysis/metrics.csv')
    geometry = read_csv(RUN / 'geometry_analysis/radius_moments.csv')
    ck = {(r['method'], int(r['step']), int(r['block']), r['aggregate'], r['direction']): r for r in curvature}
    sk = {(r['method'], int(r['step']), int(r['block']), r['aggregate'], r['direction'], float(r['abs_scale'])): r for r in symmetry}
    fk = {(r['method'], int(r['step']), int(r['block']), r['aggregate'], r['direction'], float(r['signed_scale'])): r for r in curves}
    tk = {(r['method'], int(r['step']), int(r['block']), r['aggregate'], r['direction']): r for r in transmission}
    assert len(ck) == 630 and len(sk) == 4410 and len(fk) == 9450 and len(tk) == 270
    actual_rows, orientation_rows, right_rows = [], [], []
    for method in ('Muon', 'PD'):
        for step in (10, 500, 1300):
            for block in (1, 4, 8):
                for aggregate in ('bank0', 'bank1', 'all4'):
                    prefix = method, step, block, aggregate
                    c = ck[prefix + ('actual',)]
                    common = dict(method=method, step=step, block=block, aggregate=aggregate)
                    left_h = sum(val(ck[prefix + (d,)], 'hessian') for d in ('left0', 'left1')) / 2
                    left_gn = sum(val(ck[prefix + (d,)], 'gn') for d in ('left0', 'left1')) / 2
                    row = dict(**common, slope=val(c, 'slope'), H=val(c, 'hessian'), GN=val(c, 'gn'),
                               H_over_GN=val(c, 'hessian_over_gn'), actual_over_mean_left_H=divide(val(c, 'hessian'), left_h),
                               actual_over_mean_left_GN=divide(val(c, 'gn'), left_gn),
                               mean_abs_token_H_minus_GN_over_mean_GN=val(c, 'mean_abs_token_hessian_minus_gn_over_mean_gn'),
                               negative_token_H_fraction=val(c, 'hessian_token_negative_fraction'),
                               activation_RMS=val(c, 'activation_rms_per_position'), parameter_norm=val(c, 'parameter_direction_norm'))
                    for scale in (1, 16):
                        s = sk[prefix + ('actual', float(scale))]
                        q = val(s, 'hessian_quadratic')
                        row.update({f'even_over_quadratic_s{scale}': divide(val(s, 'R_even'), q),
                                    f'odd_over_even_s{scale}': divide(val(s, 'R_odd'), val(s, 'R_even')),
                                    f'Rplus_over_quadratic_s{scale}': divide(val(s, 'R_plus'), q),
                                    f'Rminus_over_quadratic_s{scale}': divide(val(s, 'R_minus'), q)})
                    actual_rows.append(row)
                    ta = tk[prefix + ('actual',)]
                    for control in ('left0', 'left1'):
                        tc = tk[prefix + (control,)]
                        before = val(ta, 'GN_per_preactivation_energy') / val(tc, 'GN_per_preactivation_energy')
                        trans = val(ta, 'transmission_gain') / val(tc, 'transmission_gain')
                        after = val(ta, 'GN_per_residual_energy') / val(tc, 'GN_per_residual_energy')
                        assert math.isclose(before, trans * after, rel_tol=1e-11, abs_tol=1e-14)
                        orientation_rows.append(dict(**common, control=control,
                            actual_over_control_H=divide(val(c, 'hessian'), val(ck[prefix + (control,)], 'hessian')),
                            GN_per_pre_contrast=before, transmission_contrast=trans, GN_per_residual_contrast=after))
                    raw = tk[prefix + ('right_raw',)]
                    white = tk[prefix + ('right_white',)]
                    right_rows.append(dict(**common,
                        raw_over_white_pre_energy=val(raw, 'preactivation_energy') / val(white, 'preactivation_energy'),
                        raw_over_white_GN=val(raw, 'GN') / val(white, 'GN'),
                        raw_over_white_GN_per_pre=val(raw, 'GN_per_preactivation_energy') / val(white, 'GN_per_preactivation_energy'),
                        raw_over_white_transmission=val(raw, 'transmission_gain') / val(white, 'transmission_gain'),
                        raw_over_white_GN_per_residual=val(raw, 'GN_per_residual_energy') / val(white, 'GN_per_residual_energy')))
    joint_rows = []
    for r in joint:
        if r['aggregate'] not in ('bank0', 'bank1', 'all4') or float(r['signed_scale']) not in (-16, -1, 1, 16):
            continue
        h, g = val(r, 'hessian'), val(r, 'gn')
        joint_rows.append(dict(method=r['method'], step=int(r['step']), aggregate=r['aggregate'],
            signed_scale=val(r, 'signed_scale'), H_joint_over_individual_sum=divide(h, h-val(r, 'hessian_cross')),
            GN_joint_over_individual_sum=divide(g, g-val(r, 'gn_cross')),
            interaction=val(r, 'interaction'), quadratic_interaction=val(r, 'hessian_cross_prediction'),
            interaction_over_quadratic=divide(val(r, 'interaction'), val(r, 'hessian_cross_prediction'))))
    assert len(actual_rows) == 54 and len(orientation_rows) == 108 and len(right_rows) == 54 and len(joint_rows) == 72
    write_csv('actual_direction.csv', actual_rows)
    write_csv('left_orientation_decomposition.csv', orientation_rows)
    write_csv('raw_vs_white.csv', right_rows)
    write_csv('joint_summary.csv', joint_rows)
    ranges = {}
    for step in (10, 500, 1300):
        cells = [r for r in actual_rows if r['step'] == step and r['aggregate'] == 'all4']
        ranges[str(step)] = {field: range_of(cells, field) for field in
            ('H_over_GN', 'actual_over_mean_left_H', 'mean_abs_token_H_minus_GN_over_mean_GN',
             'negative_token_H_fraction', 'even_over_quadratic_s1', 'even_over_quadratic_s16',
             'odd_over_even_s16', 'Rplus_over_quadratic_s1', 'Rminus_over_quadratic_s1')}
    ranges['transmission_all4'] = {field: range_of([r for r in orientation_rows if r['aggregate'] == 'all4'], field)
        for field in ('GN_per_pre_contrast', 'transmission_contrast', 'GN_per_residual_contrast')}
    ranges['geometry_actual'] = {}
    for frame in ('raw', 'white'):
        cells = [r for r in geometry if r['frame'] == frame and r['bank'] == 'all' and r['direction'] == 'actual']
        ranges['geometry_actual'][frame] = {field: range_of([{field: val(r, field)} for r in cells], field)
            for field in ('effective_rank', 'observed_relative_variance', 'spherical_relative_variance',
                          'observed_to_spherical_mean', 'observed_to_spherical_relative_variance')}
    (OUT / 'ranges.json').write_text(json.dumps(ranges, indent=2, allow_nan=False) + '\n')
    lines = ['# Complete atlas: compact numerical tables', '', 'Retrospective summaries; all four score contexts pooled. Both banks remain in the adjacent CSVs.', '',
             '| Method | Step | Block | H/GN | H / mean left H | mean abs(H−GN) / mean GN | R_even(16) / quadratic | R_odd(16) / R_even(16) |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in actual_rows:
        if r['aggregate'] == 'all4':
            lines.append(f"| {r['method']} | {r['step']} | {r['block']} | {r['H_over_GN']:.3f} | {r['actual_over_mean_left_H']:.3f} | {r['mean_abs_token_H_minus_GN_over_mean_GN']:.3f} | {r['even_over_quadratic_s16']:.3f} | {r['odd_over_even_s16']:.3f} |")
    (OUT / 'TABLES.md').write_text('\n'.join(lines) + '\n')
    result = dict(status='complete', source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=dict(actual=len(actual_rows), orientation=len(orientation_rows), right=len(right_rows), joint=len(joint_rows)))
    (OUT / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

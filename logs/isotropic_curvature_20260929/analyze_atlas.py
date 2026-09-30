"""Analyze a COMPLETE atlas; no torch, model, optimizer, or tensor-archive reads.

Run only after the producer is terminal and analysis has been authorized.
Example: .venv/bin/python logs/isotropic_curvature_20260929/analyze_atlas.py
Existing output directories are refused so that earlier analyses are preserved.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplcache_analysis')

import argparse
import csv
import hashlib
import json
import math
import time
import traceback
import numpy as np

METHODS = ('Muon', 'PD')
STEPS = (10, 500, 1300)
BLOCKS = (1, 4, 8)  # displayed one-based; joint.npz uses zero-based blocks
LABELS = ('actual', 'left0', 'left1', 'right_raw', 'right_white')
SCALES = np.array([-16., -8., -4., -2., -1., -.5, -.25, 0., .25, .5, 1., 2., 4., 8., 16.])
ZERO = 7
DEPARTURE_FLOOR = 1e-10
GN_POSITIVE_FLOOR = 1e-12  # numerical denominator rule, not a scientific gate
RADIUS_SYMMETRY_NOTE = ('Retrospective exact even/odd summary added after root inspected an early panel; '
                        'producer observations, signed-radius grid, and existing readouts are unchanged. '
                        'All seven paired magnitudes are retained without fitting or a selection threshold.')
COLORS = dict(zip(LABELS, ('#202124', '#0072B2', '#56B4E9', '#D55E00', '#009E73')))
AGGREGATES = ([('context', f'context{j}', (j,), j, j // 2) for j in range(4)] +
              [('bank', f'bank{b}', (2*b, 2*b+1), None, b) for b in range(2)] +
              [('all_contexts', 'all4', (0, 1, 2, 3), None, None)])


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, data):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, name, atol=1e-9, rtol=1e-8):
    require(np.allclose(actual, expected, atol=atol, rtol=rtol),
            f'{name}: disagreement; max absolute difference {np.max(np.abs(np.asarray(actual)-expected))}')


def file_record(path, hashed=True):
    stat = path.stat()
    record = dict(path=str(path), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    if hashed:
        digest = hashlib.sha256()
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1 << 20), b''):
                digest.update(block)
        record['sha256'] = digest.hexdigest()
    return record


def load_npz(path, shapes):
    with np.load(path, allow_pickle=False) as archive:
        require(set(shapes) <= set(archive.files), f'{path}: missing array keys')
        arrays = {name: archive[name] for name in archive.files}
    for name, shape in shapes.items():
        value = arrays[name]
        require(value.shape == shape, f'{path}:{name}: {value.shape} != {shape}')
        require(np.isfinite(value).all(), f'{path}:{name}: nonfinite data; no partial analysis')
    close(arrays['scales'], SCALES, f'{path}:scales', atol=0, rtol=0)
    return arrays


def validate_and_load(run):
    """Read all small scalar/NPZ artifacts before producing any final analysis."""
    require(run.is_dir(), f'Run directory missing: {run}')
    for name in ('result.json', 'status.json', 'prepared.json', 'input_manifest.json'):
        require((run / name).is_file(), f'Incomplete producer: missing {name}')
    require(not (run / 'failure.json').exists(), 'Producer failure.json exists; refusing final analysis')
    result, status, prepared = (read_json(run / name) for name in ('result.json', 'status.json', 'prepared.json'))
    for name, obj in [('result', result), ('status', status)]:
        require(obj.get('status') == 'complete', f'Producer {name} is not complete')
        for key, value in [('panels', 18), ('states', 6), ('directions_per_panel', 5),
                           ('score_contexts', 4), ('calibration_contexts', 8)]:
            require(obj.get(key) == value, f'Producer {name}: {key} != {value}')
        require(tuple(obj['labels']) == LABELS, f'{name}: unexpected direction labels')
        close(obj['scales'], SCALES, name + ':scales', atol=0, rtol=0)
    require(tuple(prepared['methods']) == METHODS, 'Unexpected method set/order')
    require(tuple(prepared['steps']) == STEPS, 'Unexpected step set/order')
    require(tuple(prepared['blocks']) == (0, 3, 7), 'Unexpected selected blocks')
    require(tuple(prepared['labels']) == LABELS, 'Unexpected prepared labels')
    close(prepared['scales'], SCALES, 'prepared:scales', atol=0, rtol=0)
    require(len(result['summaries']) == 18, 'Producer summary count is not 18')
    expected_keys = {(m, s, b) for m in METHODS for s in STEPS for b in BLOCKS}
    require({(r['method'], r['step'], r['block']) for r in result['summaries']} == expected_keys,
            'Producer summaries do not cover the exact 18 panels')
    inputs = [file_record(run / name) for name in ('result.json', 'status.json', 'prepared.json', 'input_manifest.json')]
    for name, key in [('executed_instrument.py', 'executed_instrument_sha256'),
                      ('executed_PROTOCOL.md', 'executed_protocol_sha256')]:
        record = file_record(run / name)
        require(record['sha256'] == result[key], f'{name}: producer source hash mismatch')
        inputs.append(record)
    panels, states = {}, {}
    panel_shapes = dict(loss=(4, 5, 15, 512), slope=(4, 5, 512), hessian=(4, 5, 512),
                        gn=(4, 5, 512), activation_radius=(4, 5, 512),
                        projected_hessian=(4, 5, 5), projected_gn=(4, 512, 5, 5))
    joint_shapes = dict(loss=(4, 15, 512), slope=(4, 512), hessian=(4, 512), gn=(4, 512),
                        projected_hessian=(4, 3, 3), projected_gn=(4, 512, 3, 3),
                        individual_loss=(3, 4, 15, 512), individual_slope=(3, 4, 512),
                        individual_hessian=(3, 4, 512), individual_gn=(3, 4, 512))
    for method in METHODS:
        for step in STEPS:
            state_path = run / f'{method}_{step:06d}'
            state_status = read_json(state_path / 'status.json')
            require(state_status.get('status') == 'complete' and state_status.get('panels') == 3,
                    f'{state_path}: state incomplete')
            metadata = read_json(state_path / 'metadata.json')
            require(metadata['method'] == method and metadata['step'] == step, f'{state_path}: identity mismatch')
            baseline = None
            for block in BLOCKS:
                path = state_path / f'block{block:02d}_up'
                for name in ('summary.json', 'per_token.npz', 'tensors.pt'):
                    require((path / name).is_file(), f'{path}: missing completed {name}')
                arrays = load_npz(path / 'per_token.npz', panel_shapes)
                require(tuple(arrays['labels'].tolist()) == LABELS, f'{path}: unexpected labels')
                summary = read_json(path / 'summary.json')
                require((summary['method'], summary['step'], summary['block']) == (method, step, block),
                        f'{path}: summary identity mismatch')
                require(set(summary['direction_norms']) == set(LABELS), f'{path}: direction norm keys')
                for label in LABELS:
                    norm = summary['direction_norms'][label]
                    require(math.isfinite(norm) and norm >= 0, f'{path}: invalid direction norm')
                zero_losses = arrays['loss'][:, :, ZERO]
                close(zero_losses, zero_losses[:, :1], f'{path}: zero baseline across directions', atol=1e-10)
                if baseline is None:
                    baseline = zero_losses[:, 0]
                close(zero_losses[:, 0], baseline, f'{path}: zero baseline across blocks', atol=1e-10)
                close(baseline.mean(-1), metadata['baseline_nll_by_sequence'], f'{path}: metadata baseline')
                require(arrays['activation_radius'].min() >= 0, f'{path}: negative activation radius')
                require(arrays['gn'].min() >= -1e-12, f'{path}: invalid negative predictive GN')
                for di in (1, 2):
                    close(arrays['activation_radius'][:, di], arrays['activation_radius'][:, 0],
                          f'{path}: left rotation activation norm', atol=1e-10)
                h, g = arrays['projected_hessian'], arrays['projected_gn']
                close(h, h.swapaxes(-1, -2), f'{path}: finite-span H symmetry')
                close(g, g.swapaxes(-1, -2), f'{path}: finite-span GN symmetry')
                close(np.diagonal(h, axis1=-2, axis2=-1), arrays['hessian'].mean(-1), f'{path}: H diagonal')
                close(np.diagonal(g, axis1=-2, axis2=-1).swapaxes(1, 2), arrays['gn'], f'{path}: GN diagonal')
                for di, label in enumerate(LABELS):
                    values = dict(slope=arrays['slope'][:, di].mean(), hessian=arrays['hessian'][:, di].mean(),
                                  gn=arrays['gn'][:, di].mean(),
                                  activation_rms=np.sqrt(np.mean(arrays['activation_radius'][:, di] ** 2)))
                    for name, value in values.items():
                        close(value, summary['curvature_mean'][label][name], f'{path}: summary {label}/{name}')
                panels[(method, step, block)] = dict(arrays=arrays, summary=summary)
                inputs.extend(file_record(path / name) for name in ('per_token.npz', 'summary.json'))
                # Large tensors are retained by the producer but never loaded by this analyzer.
                inputs.append(file_record(path / 'tensors.pt', hashed=False))
            joint = load_npz(state_path / 'joint.npz', joint_shapes)
            require(tuple(joint['blocks'].tolist()) == (0, 3, 7), f'{state_path}: joint block order')
            close(joint['loss'][:, ZERO], baseline, f'{state_path}: joint zero baseline', atol=1e-10)
            require(joint['gn'].min() >= -1e-12, f'{state_path}: negative joint predictive GN')
            for bi, block in enumerate(BLOCKS):
                p = panels[(method, step, block)]['arrays']
                for field in ('loss', 'slope', 'hessian', 'gn'):
                    close(joint['individual_' + field][bi], p[field][:, 0], f'{state_path}: archived individual {field}')
            close(joint['slope'], joint['individual_slope'].sum(0), f'{state_path}: slope additivity', atol=1e-8)
            h, g = joint['projected_hessian'], joint['projected_gn']
            close(h, h.swapaxes(-1, -2), f'{state_path}: joint H symmetry')
            close(g, g.swapaxes(-1, -2), f'{state_path}: joint GN symmetry')
            close(h.sum((-1, -2)), joint['hessian'].mean(-1), f'{state_path}: joint H sum', atol=1e-8)
            close(g.sum((-1, -2)), joint['gn'], f'{state_path}: joint GN sum', atol=1e-8)
            close(np.diagonal(h, axis1=-2, axis2=-1), joint['individual_hessian'].mean(-1).T,
                  f'{state_path}: joint H diagonal vs independent rays', atol=1e-8)
            close(np.diagonal(g, axis1=-2, axis2=-1), joint['individual_gn'].transpose(1, 2, 0),
                  f'{state_path}: joint GN diagonal vs independent rays', atol=1e-8)
            states[(method, step)] = dict(arrays=joint, metadata=metadata)
            inputs.extend(file_record(state_path / name) for name in ('joint.npz', 'metadata.json', 'status.json'))
    return dict(panels=panels, states=states, prepared=prepared, result=result, inputs=inputs)


def aggregation_fields(spec):
    level, name, members, context, bank = spec
    return dict(aggregation=level, aggregate=name, context=context, bank=bank,
                context_members=','.join(map(str, members)), context_count=len(members))


def sign_name(s):
    return 'negative' if s < 0 else 'positive' if s > 0 else 'zero'


def ratio_fields(h, g):
    qualified = bool(g > GN_POSITIVE_FLOOR)
    return dict(hessian_minus_gn=float(h-g), hessian_over_gn=float(h/g) if qualified else None,
                h_over_gn_qualified=qualified,
                h_over_gn_reason='positive_GN_above_numerical_floor' if qualified else 'GN_at_or_below_numerical_floor')


def token_distribution_fields(hessian, gn, mean_gn):
    """Descriptive pooled-token distributions; tokens are not independent replicates.

    In bank/all-four rows, pool the member contexts' equally sized token arrays
    before quantiles and absolute differences. Do not average context quantiles
    or token ratios. Early-block per-token losses include cross-position effects.
    """
    h = np.asarray(hessian).reshape(-1)
    g = np.asarray(gn).reshape(-1)
    require(h.shape == g.shape and h.size > 0, 'Mismatched token curvature distributions')
    hq = np.quantile(h, [.1, .5, .9], method='linear')
    gq = np.quantile(g, [.1, .5, .9], method='linear')
    mean_absolute_difference = float(np.mean(np.abs(h-g)))
    qualified = bool(mean_gn > GN_POSITIVE_FLOOR)
    return dict(token_distribution_count=int(h.size),
                hessian_token_q10=float(hq[0]), hessian_token_q50=float(hq[1]), hessian_token_q90=float(hq[2]),
                gn_token_q10=float(gq[0]), gn_token_q50=float(gq[1]), gn_token_q90=float(gq[2]),
                hessian_token_negative_fraction=float(np.mean(h < 0)),
                mean_abs_token_hessian_minus_gn=mean_absolute_difference,
                mean_abs_token_hessian_minus_gn_over_mean_gn=(mean_absolute_difference/mean_gn if qualified else None),
                abs_token_h_minus_gn_over_mean_gn_qualified=qualified)


def curve_terms(loss, base, slope, hessian, gn):
    linear = SCALES * slope
    remainder = loss - base - linear
    qh, qg = .5*SCALES**2*hessian, .5*SCALES**2*gn
    denh = np.maximum(np.maximum(np.abs(remainder), np.abs(qh)), DEPARTURE_FLOOR)
    deng = np.maximum(np.maximum(np.abs(remainder), np.abs(qg)), DEPARTURE_FLOOR)
    return dict(loss=loss, loss_change=loss-base, true_linear_term=linear, linear_prediction=base+linear,
                remainder=remainder, hessian_quadratic=qh, gn_quadratic=qg,
                hessian_departure=remainder-qh, gn_departure=remainder-qg,
                hessian_departure_denominator=denh, gn_departure_denominator=deng,
                signed_relative_hessian_departure=(remainder-qh)/denh,
                signed_relative_gn_departure=(remainder-qg)/deng)


def build_tables(atlas):
    curves, curvature, projected, joint_rows, symmetry_rows = [], [], [], [], []
    for (method, step, block), panel in atlas['panels'].items():
        a, summary = panel['arrays'], panel['summary']
        for spec in AGGREGATES:
            members = list(spec[2]); common = dict(method=method, step=step, block=block, **aggregation_fields(spec))
            hm = a['projected_hessian'][members].mean(0)
            gm = a['projected_gn'][members].mean((0, 1))
            for i, label in enumerate(LABELS):
                loss = a['loss'][members, i].mean((0, 2)); base = float(loss[ZERO])
                slope = float(a['slope'][members, i].mean())
                h, g = float(a['hessian'][members, i].mean()), float(a['gn'][members, i].mean())
                rms = float(np.sqrt(np.mean(a['activation_radius'][members, i] ** 2)))
                weight_norm = float(summary['direction_norms'][label])
                static = dict(**common, direction=label, baseline_loss=base, slope=slope, hessian=h, gn=g,
                              parameter_direction_norm=weight_norm, activation_rms_per_position=rms,
                              activation_sequence_norm_rms=math.sqrt(512)*rms, **ratio_fields(h, g),
                              **token_distribution_fields(a['hessian'][members, i], a['gn'][members, i], g))
                curvature.append(static)
                terms = curve_terms(loss, base, slope, h, g)
                for si, s in enumerate(SCALES):
                    curves.append(dict(**static, signed_scale=float(s), side=sign_name(s), abs_scale=float(abs(s)),
                                       parameter_displacement_norm=abs(float(s))*weight_norm,
                                       activation_displacement_rms=abs(float(s))*rms,
                                       activation_sequence_displacement_norm_rms=abs(float(s))*math.sqrt(512)*rms,
                                       **{name: float(value[si]) for name, value in terms.items()}))
                for plus_index in np.flatnonzero(SCALES > 0):
                    magnitude = float(SCALES[plus_index])
                    minus_indices = np.flatnonzero(SCALES == -magnitude)
                    require(len(minus_indices) == 1, 'Each positive radius must have one negative counterpart')
                    minus_index = int(minus_indices[0])
                    r_plus = float(terms['remainder'][plus_index])
                    r_minus = float(terms['remainder'][minus_index])
                    r_even, r_odd = .5*(r_plus+r_minus), .5*(r_plus-r_minus)
                    quadratic = float(terms['hessian_quadratic'][plus_index])
                    even_denominator = max(abs(r_even), abs(quadratic), DEPARTURE_FLOOR)
                    odd_denominator = max(abs(r_plus)+abs(r_minus), DEPARTURE_FLOOR)
                    symmetry_rows.append(dict(**static, abs_scale=magnitude,
                                              parameter_displacement_norm=magnitude*weight_norm,
                                              activation_displacement_rms=magnitude*rms,
                                              activation_sequence_displacement_norm_rms=magnitude*math.sqrt(512)*rms,
                                              R_plus=r_plus, R_minus=r_minus, R_even=r_even, R_odd=r_odd,
                                              hessian_quadratic=quadratic,
                                              even_hessian_departure=r_even-quadratic,
                                              even_hessian_departure_denominator=even_denominator,
                                              signed_relative_even_hessian_departure=(r_even-quadratic)/even_denominator,
                                              odd_fraction_denominator=odd_denominator,
                                              odd_signed_fraction=(r_plus-r_minus)/odd_denominator))
            for i, label_i in enumerate(LABELS):
                for j, label_j in enumerate(LABELS):
                    projected.append(dict(**common, span='five_selected_direction_coefficients',
                                          direction_i=label_i, direction_j=label_j,
                                          hessian=float(hm[i, j]), gn=float(gm[i, j])))
    for (method, step), state in atlas['states'].items():
        a = state['arrays']
        for spec in AGGREGATES:
            members = list(spec[2]); common = dict(method=method, step=step, **aggregation_fields(spec))
            loss = a['loss'][members].mean((0, 2)); base = float(loss[ZERO])
            slope, h, g = (float(a[key][members].mean()) for key in ('slope', 'hessian', 'gn'))
            terms = curve_terms(loss, base, slope, h, g)
            iloss = a['individual_loss'][:, members].mean((1, 3))
            islope = a['individual_slope'][:, members].mean((1, 2))
            ih, ig = (a[key][:, members].mean((1, 2)) for key in ('individual_hessian', 'individual_gn'))
            individual_remainders = iloss - iloss[:, ZERO, None] - islope[:, None]*SCALES
            remainder_sum = individual_remainders.sum(0)
            interaction = terms['remainder'] - remainder_sum
            hm = a['projected_hessian'][members].mean(0)
            gm = a['projected_gn'][members].mean((0, 1))
            hcross, gcross = float(hm.sum()-np.trace(hm)), float(gm.sum()-np.trace(gm))
            close(hcross, h-ih.sum(), f'{method}/{step}/{spec[1]}: H cross', atol=2e-8)
            close(gcross, g-ig.sum(), f'{method}/{step}/{spec[1]}: GN cross', atol=2e-8)
            qcross, qgcross = .5*SCALES**2*hcross, .5*SCALES**2*gcross
            denominator = np.maximum(np.maximum(abs(interaction), abs(qcross)), DEPARTURE_FLOOR)
            for si, s in enumerate(SCALES):
                joint_rows.append(dict(**common, selected_blocks='1,4,8', signed_scale=float(s), side=sign_name(s),
                                       abs_scale=float(abs(s)), baseline_loss=base, slope=slope, hessian=h, gn=g,
                                       **ratio_fields(h, g), **{name: float(v[si]) for name, v in terms.items()},
                                       sum_individual_remainders=float(remainder_sum[si]),
                                       interaction=float(interaction[si]), hessian_cross=hcross, gn_cross=gcross,
                                       hessian_cross_prediction=float(qcross[si]), gn_cross_prediction=float(qgcross[si]),
                                       interaction_hessian_departure=float(interaction[si]-qcross[si]),
                                       interaction_departure_denominator=float(denominator[si]),
                                       signed_relative_interaction_departure=float((interaction[si]-qcross[si])/denominator[si])))
            for i, block_i in enumerate(BLOCKS):
                for j, block_j in enumerate(BLOCKS):
                    projected.append(dict(**common, block='joint', span='three_selected_up_write_coefficients',
                                          direction_i=f'block{block_i:02d}_actual', direction_j=f'block{block_j:02d}_actual',
                                          hessian=float(hm[i, j]), gn=float(gm[i, j])))
    require(len(curves) == 18*7*5*15 and len(curvature) == 18*7*5, 'Unexpected panel table coverage')
    require(len(joint_rows) == 6*7*15, 'Unexpected joint table coverage')
    require(len(symmetry_rows) == 18*7*5*7, 'Unexpected paired-radius table coverage')
    return dict(curves=curves, curvature=curvature, projected_curvature=projected,
                joint=joint_rows, radius_symmetry=symmetry_rows)


def csv_write(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def signed_axis(ax, values):
    """Symmetric-log display keeps negative values and zero; raw tables are unchanged."""
    magnitude = max(float(np.max(np.abs(np.asarray(v)))) for v in values)
    ax.set_yscale('symlog', linthresh=max(1e-8, magnitude*1e-4), linscale=.8)
    ax.axhline(0, color='#bbbbbb', lw=.6)
    ax.grid(True, alpha=.18)


def save_figure(fig, path, plt):
    fig.savefig(path.with_suffix('.png'), dpi=145, bbox_inches='tight')
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight')
    plt.close(fig)


def figures(atlas, tables, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
                         'legend.fontsize': 8, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fdir = out / 'figures'; fdir.mkdir()
    lookup = {(r['method'], r['step'], r['block'], r['aggregate'], r['direction']): r for r in tables['curvature']}
    for group, labels in [('left_vs_actual', ('actual', 'left0', 'left1')),
                          ('right_vs_actual', ('actual', 'right_raw', 'right_white'))]:
        fig, axes = plt.subplots(3, 4, figsize=(15, 10), squeeze=False)
        for bi, block in enumerate(BLOCKS):
            for mi, method in enumerate(METHODS):
                for ci, quantity in enumerate(('hessian', 'gn')):
                    ax = axes[bi, 2*mi+ci]; values = []
                    for label in labels:
                        for aggregate in ('bank0', 'bank1', 'all4'):
                            y = [lookup[(method, step, block, aggregate, label)][quantity] for step in STEPS]
                            values.append(y)
                            ax.plot(STEPS, y, color=COLORS[label], marker='o' if aggregate == 'all4' else None,
                                    lw=1.8 if aggregate == 'all4' else .7,
                                    alpha=1 if aggregate == 'all4' else .3,
                                    label=label if aggregate == 'all4' else None)
                    signed_axis(ax, values)
                    ax.set_xticks(STEPS); ax.set_xlabel('Training update')
                    ax.set_title(f'{method} · block {block} · {"true H" if quantity == "hessian" else "predictive GN"}')
                    if mi == 0 and ci == 0: ax.set_ylabel('Directional curvature\n(nats/token per s²; signed symlog)')
        axes[0, 0].legend(loc='best')
        fig.suptitle(f'Directional curvature through training: {group.replace("_", " ")}\n'
                     'Own saved-write directions at each state; bold = four-context mean, thin = two-context bank means', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .94)); save_figure(fig, fdir / f'curvature_training_{group}', plt)
    fig, axes = plt.subplots(3, 4, figsize=(15, 10), squeeze=False)
    for bi, block in enumerate(BLOCKS):
        for mi, method in enumerate(METHODS):
            for ci, quantity in enumerate(('parameter_direction_norm', 'activation_rms_per_position')):
                ax = axes[bi, 2*mi+ci]
                for label in LABELS:
                    ax.plot(STEPS, [lookup[(method, step, block, 'all4', label)][quantity] for step in STEPS],
                            color=COLORS[label], marker='o', label=label)
                ax.set_xticks(STEPS); ax.set_xlabel('Training update'); ax.grid(alpha=.2)
                ax.set_title(f'{method} · block {block} · {"parameter norm" if ci == 0 else "sequence activation RMS"}')
    axes[0, 0].legend()
    fig.suptitle('Unit-multiplier direction scales; right-white is not parameter-norm matched\n'
                 'Activation RMS = √mean(context,position) ||D x||²; a summary of all injected positions', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, .94)); save_figure(fig, fdir / 'direction_scales_training', plt)
    style_handles = [Line2D([], [], color='#0072B2', lw=1.8, label='finite remainder R'),
                     Line2D([], [], color='#333333', ls='--', label='½s² true H'),
                     Line2D([], [], color='#AA4499', ls=':', label='½s² predictive GN')]
    with PdfPages(fdir / 'signed_radius_profiles_all_contexts.pdf') as pdf, \
         PdfPages(fdir / 'signed_departure_all_contexts.pdf') as departures:
        for (method, step, block), panel in atlas['panels'].items():
            a = panel['arrays']
            for side in ('negative', 'positive'):
                selected = np.where(SCALES < 0 if side == 'negative' else SCALES > 0)[0]
                selected = selected[np.argsort(abs(SCALES[selected]))]
                fig, axes = plt.subplots(5, 4, figsize=(18, 17), squeeze=False, sharey='col')
                column_values = [[] for _ in range(4)]
                for di, label in enumerate(LABELS):
                    for j in range(4):
                        ax = axes[di, j]; row = lookup[(method, step, block, f'context{j}', label)]
                        terms = curve_terms(a['loss'][j, di].mean(-1), row['baseline_loss'], row['slope'], row['hessian'], row['gn'])
                        rms = row['activation_rms_per_position']
                        # A zero kick has no physical log-radius axis; keep the multiplier axis and say so.
                        factor = rms if rms > 0 else 1.
                        x = abs(SCALES[selected])*factor
                        ax.plot(x, terms['remainder'][selected], color='#0072B2', marker='o', ms=3, lw=1.7)
                        ax.plot(x, terms['hessian_quadratic'][selected], color='#333333', ls='--', lw=1.2)
                        ax.plot(x, terms['gn_quadratic'][selected], color='#AA4499', ls=':', lw=1.5)
                        values = [terms[k][selected] for k in ('remainder', 'hessian_quadratic', 'gn_quadratic')]
                        column_values[j].extend(values)
                        signed_axis(ax, values)
                        ax.set_xscale('log', base=2); ax.axvline(factor, color='#888888', ls=':', lw=.7)
                        ax.set_title(f'{label} · context {j} · bank {j//2}')
                        ax.set_xlabel('Sequence RMS activation displacement' if rms > 0 else '|s| (unit kick is zero)')
                        if j == 0: ax.set_ylabel('R and quadratic terms\n(nats/token; signed symlog)')
                        top = ax.secondary_xaxis('top', functions=(lambda x, f=factor: x/f, lambda s, f=factor: s*f))
                        top.set_xticks([.25, 1, 4, 16]); top.set_xticklabels(['.25', '1', '4', '16'])
                        top.set_xlabel('|s|', fontsize=8)
                # Common y scale across the five directions in each context exposes magnitude differences.
                for j in range(4):
                    magnitude = max(float(np.max(np.abs(v))) for v in column_values[j])
                    axes[0, j].set_yscale('symlog', linthresh=max(1e-8, magnitude*1e-4), linscale=.8)
                fig.suptitle(f'{method} · state {step} · block {block} up · {side} multipliers\n'
                             'Full-context loss; shared y scale within each context; radius summarizes all injected positions', fontsize=13)
                fig.legend(handles=style_handles, loc='lower center', ncol=3)
                fig.tight_layout(rect=(0, .03, 1, .95)); pdf.savefig(fig)
                fig.savefig(fdir / f'profile_{method}_{step:06d}_block{block:02d}_{side}.png', dpi=115, bbox_inches='tight')
                plt.close(fig)
            fig, axes = plt.subplots(4, 2, figsize=(11, 12), squeeze=False)
            for j in range(4):
                for si, side in enumerate(('negative', 'positive')):
                    ax = axes[j, si]
                    selected = np.where(SCALES < 0 if side == 'negative' else SCALES > 0)[0]
                    selected = selected[np.argsort(abs(SCALES[selected]))]
                    for di, label in enumerate(LABELS):
                        row = lookup[(method, step, block, f'context{j}', label)]
                        terms = curve_terms(a['loss'][j, di].mean(-1), row['baseline_loss'], row['slope'], row['hessian'], row['gn'])
                        ax.plot(abs(SCALES[selected]), terms['signed_relative_hessian_departure'][selected],
                                color=COLORS[label], marker='o', ms=3, label=label)
                    ax.set_xscale('log', base=2); ax.set_ylim(-2.05, 2.05); ax.axhline(0, color='gray', lw=.6)
                    ax.grid(alpha=.2); ax.set_title(f'context {j} · bank {j//2} · {side} s')
                    ax.set_xlabel('|s|, actual-write multiplier'); ax.set_ylabel('(R − ½s²H) / max(|R|, |½s²H|, 1e−10)')
            axes[0, 0].legend()
            fig.suptitle(f'{method} · state {step} · block {block}: signed quadratic departure\n'
                         'Ratios computed after whole-context aggregation; negative H and R retained', fontsize=12)
            fig.tight_layout(rect=(0, 0, 1, .94)); departures.savefig(fig); plt.close(fig)
    joint_lookup = {}
    for row in tables['joint']:
        joint_lookup.setdefault((row['method'], row['step'], row['aggregate'], row['side']), []).append(row)
    def joint_plot(ax, method, step, aggregate, side):
        rows = sorted(joint_lookup[(method, step, aggregate, side)], key=lambda r: r['abs_scale'])
        x = [r['abs_scale'] for r in rows]
        for name, color, style, label in [('interaction', '#0072B2', '-', 'joint R − Σ individual R'),
                                          ('hessian_cross_prediction', '#333333', '--', '½s² H cross'),
                                          ('gn_cross_prediction', '#AA4499', ':', '½s² GN cross')]:
            ax.plot(x, [r[name] for r in rows], color=color, ls=style, marker='o' if style == '-' else None,
                    ms=3, label=label)
        signed_axis(ax, [[r[k] for r in rows] for k in ('interaction', 'hessian_cross_prediction', 'gn_cross_prediction')])
        ax.set_xscale('log', base=2); ax.set_xlabel('|s|, joint actual-write multiplier')
        ax.set_ylabel('Selected-up interaction\n(nats/token; signed symlog)')
    with PdfPages(fdir / 'joint_interaction_all_contexts.pdf') as pdf:
        for method in METHODS:
            for step in STEPS:
                fig, axes = plt.subplots(4, 2, figsize=(11, 12), squeeze=False)
                for j in range(4):
                    for si, side in enumerate(('negative', 'positive')):
                        joint_plot(axes[j, si], method, step, f'context{j}', side)
                        axes[j, si].set_title(f'context {j} · bank {j//2} · {side} s')
                axes[0, 0].legend()
                fig.suptitle(f'{method} · state {step}: joint blocks 1/4/8 up\n'
                             'Three selected parameter writes; current later inputs; H cross = ΣHᵢⱼ − tr(H)', fontsize=12)
                fig.tight_layout(rect=(0, 0, 1, .94)); pdf.savefig(fig); plt.close(fig)
    for side in ('negative', 'positive'):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
        for mi, method in enumerate(METHODS):
            for si, step in enumerate(STEPS):
                joint_plot(axes[mi, si], method, step, 'all4', side)
                axes[mi, si].set_title(f'{method} · state {step}')
        axes[0, 0].legend()
        fig.suptitle(f'Selected blocks 1/4/8 joint interaction: {side} s, four-context means\n'
                     'Finite three-write span; this is not full-body coupling or a full Hessian spectrum', fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, .92)); save_figure(fig, fdir / f'joint_interaction_{side}', plt)
    time_colors = {10: '#0072B2', 500: '#D55E00', 1300: '#009E73'}
    symmetry = [r for r in tables['radius_symmetry'] if r['direction'] == 'actual' and r['aggregate'] == 'all4']
    for quantity, title, ylabel, limit in [
            ('signed_relative_even_hessian_departure', 'Even remainder: departure from the true-H quadratic',
             '(R_even − ½s²H) / max(|R_even|, |½s²H|, 1e−10)', 2.05),
            ('odd_signed_fraction', 'Signed odd component of the remainder',
             '(R_plus − R_minus) / max(|R_plus| + |R_minus|, 1e−10)', 1.05)]:
        fig, axes = plt.subplots(3, 2, figsize=(12, 11), squeeze=False)
        for bi, block in enumerate(BLOCKS):
            for mi, method in enumerate(METHODS):
                ax = axes[bi, mi]
                for step in STEPS:
                    rows = sorted((r for r in symmetry if r['method'] == method and r['step'] == step and r['block'] == block),
                                  key=lambda r: r['abs_scale'])
                    require(len(rows) == 7, 'Symmetry overview must retain all seven paired magnitudes')
                    ax.plot([r['abs_scale'] for r in rows], [r[quantity] for r in rows],
                            color=time_colors[step], marker='o', ms=4, label=f'state {step}')
                ax.set_xscale('log', base=2); ax.set_ylim(-limit, limit)
                ax.axhline(0, color='gray', lw=.7); ax.grid(alpha=.2)
                ax.set_title(f'{method} · block {block} up · actual direction')
                ax.set_xlabel('|s|, paired actual-write multipliers'); ax.set_ylabel(ylabel)
        axes[0, 0].legend()
        fig.suptitle(title + '\nFour-context means; retrospective exact summary after the early-panel reading', fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, .94)); save_figure(fig, fdir / f'radius_symmetry_{quantity}', plt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=HERE / 'run1')
    parser.add_argument('--out', type=Path, help='New output directory; default RUN/analysis')
    parser.add_argument('--tables-only', action='store_true', help='Explicitly omit figures; output states this limitation')
    args = parser.parse_args()
    run = args.run.resolve(); out = (args.out or run / 'analysis').resolve()
    require(out != run, 'Output directory must differ from the producer directory')
    require(not out.exists(), f'Output already exists; use a fresh --out directory: {out}')
    started = time.monotonic()
    atlas = validate_and_load(run)  # refuses partial/failed producers before creating output
    tables = build_tables(atlas)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'status.json', dict(status='running', stage='validated_complete_inputs'))
    try:
        for name, rows in tables.items():
            csv_write(out / f'{name}.csv', rows)
        if not args.tables_only:
            figures(atlas, tables, out)
        # Producer artifacts must remain unchanged throughout analysis.
        for record in atlas['inputs']:
            stat = Path(record['path']).stat()
            require(stat.st_size == record['bytes'] and stat.st_mtime_ns == record['mtime_ns'],
                    f'Producer artifact changed during analysis: {record["path"]}')
        manifest = dict(analyzer=file_record(Path(__file__)), inputs=atlas['inputs'],
                        large_tensor_archive_policy='Existence/size/mtime only; tensors.pt never loaded or hashed',
                        numerical_threads=1, model_or_optimizer_imports=False)
        write_json(out / 'input_manifest.json', manifest)
        report = dict(status='complete', seconds=time.monotonic()-started, producer=str(run),
                      panels=18, states=6, score_contexts=4, banks=2, directions=5, signed_scales=SCALES.tolist(),
                      aggregation='Mean over 512 tokens, then equal mean over declared contexts. Ratios after aggregation.',
                      token_distributions='Pool all token values within the declared context/bank/all-four row; linear quantiles at .1/.5/.9, H<0 fraction, mean abs(H-GN). Descriptive, not independent token replicates.',
                      radius_symmetry_note=RADIUS_SYMMETRY_NOTE,
                      table_rows={name:len(rows) for name, rows in tables.items()}, figures_complete=not args.tables_only,
                      departure_formula='(R-Q)/max(abs(R),abs(Q),1e-10); signed raw R and Q retained',
                      gn_ratio_numerical_floor=GN_POSITIVE_FLOOR,
                      gn_ratio_qualified_rows=sum(r['h_over_gn_qualified'] for r in tables['curvature']),
                      zero_and_negative_values_preserved=True,
                      scope='Five-direction panel spans and three-selected-up-write joint span; no full-spectrum, rate, causal or fitted-power claim.')
        write_json(out / 'result.json', report)
        (out / 'README.md').write_text(
            '# Complete-atlas reductions and figures\n\n'
            'All 18 panels, six joint states, five directions, four contexts, two banks, and 15 signed scales passed structural and numerical consistency checks. '
            'The analyzer reads saved NPZ/JSON observations only; it does not load tensors.pt or construct a model.\n\n'
            '- `curves.csv`: context/bank/all-four loss, exact AD linear term, signed remainder, true-H/GN quadratics, absolute and normalized departures, parameter and sequence activation radii.\n'
            '- `curvature.csv`: the corresponding H, GN, H−GN, qualified H/GN, parameter direction norm and activation RMS; also pooled-token H/GN 10/50/90% quantiles, fraction H<0, mean|H−GN|, and mean|H−GN|/mean(GN). Blank ratios have mean GN ≤ 1e−12; negative H remains valid. These distribution columns also accompany the corresponding `curves.csv` rows.\n'
            '- `joint.csv`: selected-up joint remainder minus the three separate remainders and ½s² times the full off-diagonal H/GN sums.\n'
            '- `radius_symmetry.csv`: every direction/context/bank/all-four row at all seven paired |s| values; R_plus, R_minus, R_even, R_odd, the true-H quadratic, signed even-quadratic departure, and signed odd fraction with explicit denominators.\n'
            '- `projected_curvature.csv`: signed coefficient-space entries in the finite five-direction or three-write span, not the full parameter Hessian.\n'
            '- `figures/`: training summaries (PNG/PDF), every context and sign of every radius panel (36-page PDF plus PNG pages), signed departure profiles (18-page PDF), and joint profiles (six-page PDF plus mean summaries).\n\n'
            'All loss summaries are nats/token. Sequence activation RMS is √mean_position ||D x||²; the sequence Euclidean norm is √512 times this. '
            'At signed multiplier s, both radii scale by |s|. For blocks 1 and 4, this radius summarizes all injected positions and is not a token-local loss-response coordinate. '
            'Joint plots use actual-write multiplier only: the saved joint archive contains no current-input joint activation-radius trajectory.\n\n'
            'Token-distribution columns pool the 512, 1024, or 2048 token values belonging to each context, bank, or all-four row, with equal token weights. '
            'Quantiles use linear interpolation on that pool, rather than averaging per-context quantiles. The H<0 fraction uses the signed stored values without thresholding. '
            'The absolute difference is formed token by token before averaging; its ratio divides by the matching aggregate mean GN, using the same positive-GN numerical floor as H/GN. '
            'Agreement H≈GN in the mean can reflect cancellation of the signed model-curvature term H−GN, so it does not imply small |H−GN| at individual tokens. '
            'These are descriptive distributions, not additional independent replicates; for earlier blocks the per-token loss derivative includes effects from kicks at other positions.\n\n'
            'Retrospective addition after an early-panel reading: the exact paired-radius decomposition is R_even=(R_plus+R_minus)/2 and R_odd=(R_plus−R_minus)/2. '
            'The even-quadratic departure is (R_even−½s²H)/max(|R_even|,|½s²H|,1e−10); the signed odd fraction is (R_plus−R_minus)/max(|R_plus|+|R_minus|,1e−10). '
            'A remainder depending only on the kick norm is even in s; this decomposition describes the measured asymmetry without fitting a radial model or defining a success threshold. '
            'The table retains all seven magnitudes for every panel and aggregate; the two compact PNG/PDF overviews show actual directions and four-context means only. '
            'Producer observations, the signed-radius grid, and existing readouts are unchanged.\n\n'
            'Ratios are formed after token/context aggregation, never averaged from token ratios. Signed relative quadratic departure is (R−Q)/max(|R|,|Q|,1e−10). '
            'Zero is retained in the tables; log-radius figures display the nonzero positive and negative sides separately. Signed-symlog y axes retain zero and negative curvature/remainders. '
            'The numerical floors are denominator conventions, not usefulness gates. No fitted exponent, p-value, confidence interval, optimizer ranking or population-isotropy claim is produced.\n\n'
            + ('Figures were explicitly omitted with --tables-only.\n' if args.tables_only else 'Figures and numeric tables are complete.\n'))
        write_json(out / 'status.json', report)
        print(json.dumps(report, indent=2, allow_nan=False))
    except BaseException as error:
        write_json(out / 'failure.json', dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc()))
        write_json(out / 'status.json', dict(status='failed', seconds=time.monotonic()-started, message=str(error)))
        raise


if __name__ == '__main__':
    main()

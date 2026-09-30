"""Fixed Q/K factorial readout; --self-test never reads scientific outputs."""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplconfig')
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_key] = '2'

import argparse
import csv
import hashlib
import json
import time
import numpy as np

LABELS = ['e028_b09', 'e028_b08', 'e040_b09', 'e040_b08']
PARTS = ['interaction', 'overlap', 'mixed_loss', 'J_direct', 'KL_add', 'base_entropy']
BANKS = {'all': slice(None), 'bank0': slice(0, 8), 'bank1': slice(8, 16)}
LR_BOUND = (.04 / .028) ** 2
MATERIAL = .005
TOL = 1e-10
FACTORIAL = [
    ('beta08_over09', 'LR .028', 'e028_b08', 'e028_b09', '>', 1.),
    ('beta08_over09', 'LR .04', 'e040_b08', 'e040_b09', '>', 1.),
    ('lr040_over028', 'beta .9', 'e040_b09', 'e028_b09', '<', LR_BOUND),
    ('lr040_over028', 'beta .8', 'e040_b08', 'e028_b08', '<', LR_BOUND),
]
SCOPE = ('Four selected own states at 46->47, one seed, two fixed contiguous '
         'eight-sequence banks. Finite KL and centered logit-response covariance '
         'are not tangent GN, common-state treatment effects, training-rate '
         'mediation, or independent seed replications. No inferential statistics.')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def describe(x):
    """Dispersion of paired sequence summaries, never a confidence interval."""
    x = np.asarray(x, dtype=np.float64)
    assert x.ndim == 1 and len(x) > 1 and np.isfinite(x).all()
    return dict(mean=float(x.mean()), sd_across_sequences=float(x.std(ddof=1)),
                min=float(x.min()), max=float(x.max()), sequences=len(x),
                negative_sequences=int(np.sum(x < 0)), positive_sequences=int(np.sum(x > 0)),
                zero_sequences=int(np.sum(x == 0)))


def ratio_value(numerator, denominator):
    """No outcome-dependent epsilon or clipping of an undefined ratio."""
    a, b = float(numerator), float(denominator)
    assert np.isfinite(a) and np.isfinite(b)
    defined = a >= 0 and b > 0
    return dict(numerator_mean=a, denominator_mean=b,
                ratio=a / b if defined else None,
                defined=defined,
                reason=None if defined else 'nonpositive_denominator_or_negative_numerator',
                denominator_at_or_below_numerical_tolerance=abs(b) <= TOL)


def derive(L, K, conditional, G, parts, linear):
    """Validate per-token identities and derive fixed finite-response quantities."""
    assert L.ndim == 4 and L.shape[0] == 4 and L.shape[2] == 4
    m, n, _, t = L.shape
    assert K.shape == (m, n, 3, t) and conditional.shape == (m, n, 2, t)
    assert G.shape == (m, n, t, 3, 3) and parts.shape == (m, n, 6, t)
    assert linear.shape == (m, n, 3, t)
    assert all(np.isfinite(a).all() for a in (L, K, conditional, G, parts, linear))
    minimum_kl = min(float(K.min()), float(conditional.min()), float(parts[:, :, 4].min()))
    assert minimum_kl >= -TOL
    delta = L[:, :, 1:] - L[:, :, :1]
    interaction = L[:, :, 3] - L[:, :, 1] - L[:, :, 2] + L[:, :, 0]
    checks = {
        'CE_KL_linear_max_abs': float(np.max(np.abs(delta - K - linear))),
        'interaction_CE_max_abs': float(np.max(np.abs(interaction - parts[:, :, 0]))),
        'interaction_split_max_abs': float(np.max(np.abs(interaction - parts[:, :, 1] - parts[:, :, 2]))),
        'mixed_identity_max_abs': float(np.max(np.abs(parts[:, :, 2] - parts[:, :, 3] - K[:, :, 2] + parts[:, :, 4]))),
        'J_identity_max_abs': float(np.max(np.abs(parts[:, :, 3] - interaction + K[:, :, 2] - K[:, :, 0] - K[:, :, 1]))),
    }
    assert max(checks.values()) <= TOL, checks
    diagonal = np.diagonal(G, axis1=-2, axis2=-1)
    scale = np.maximum(1., np.max(np.abs(diagonal), axis=-1))
    symmetry = np.max(np.abs(G - np.swapaxes(G, -1, -2)), axis=(-1, -2))
    eig = np.linalg.eigvalsh((G + np.swapaxes(G, -1, -2)) / 2.)
    assert np.all(symmetry <= TOL * scale)
    assert np.all(eig[..., 0] >= -TOL * scale)
    q, r, full = diagonal[..., 0], diagonal[..., 1], diagonal[..., 2]
    # Average the two stored cross entries within the qualified symmetry
    # tolerance; retain the full original covariance in exported state records.
    cov = (G[..., 0, 1] + G[..., 1, 0]) / 2
    cov_q_full = (G[..., 0, 2] + G[..., 2, 0]) / 2
    cov_r_full = (G[..., 1, 2] + G[..., 2, 1]) / 2
    additive = q + r + 2 * cov
    coefficients = np.array([-1., -1., 1.])
    mixed = np.einsum('i,...ij,j->...', coefficients, G, coefficients)
    add_mixed = cov_q_full + cov_r_full - additive
    expansion = np.max(np.abs(full - additive - mixed - 2 * add_mixed))
    assert expansion <= TOL * float(np.max(scale))
    views = {
        'nll_base': L[:, :, 0], 'nll_Q': L[:, :, 1], 'nll_R': L[:, :, 2], 'nll_full': L[:, :, 3],
        'delta_nll_Q_alone': delta[:, :, 0], 'delta_nll_R_alone': delta[:, :, 1], 'delta_nll_full': delta[:, :, 2],
        'delta_nll_Q_last': L[:, :, 3] - L[:, :, 2], 'delta_nll_R_last': L[:, :, 3] - L[:, :, 1],
        'forward_kl_Q': K[:, :, 0], 'forward_kl_R': K[:, :, 1], 'forward_kl_full': K[:, :, 2],
        'conditional_kl_Q_last': conditional[:, :, 0], 'conditional_kl_R_last': conditional[:, :, 1],
        'finite_label_linear_Q': linear[:, :, 0], 'finite_label_linear_R': linear[:, :, 1],
        'finite_label_linear_full': linear[:, :, 2],
        **{name: parts[:, :, i] for i, name in enumerate(PARTS)},
        'response_variance_Q': q, 'response_variance_R': r, 'response_variance_full': full,
        'response_covariance_Q_R': cov, 'response_twice_covariance_Q_R': 2 * cov,
        'response_covariance_Q_full': cov_q_full, 'response_covariance_R_full': cov_r_full,
        'response_variance_additive': additive, 'response_variance_mixed': mixed,
        'response_covariance_additive_mixed': add_mixed,
        'response_twice_covariance_additive_mixed': 2 * add_mixed,
    }
    # Positive symmetric credit means improvement; this convention is descriptive only.
    views['symmetric_improvement_credit_Q'] = -(views['delta_nll_Q_alone'] + views['delta_nll_Q_last']) / 2
    views['symmetric_improvement_credit_R'] = -(views['delta_nll_R_alone'] + views['delta_nll_R_last']) / 2
    credit_error = np.max(np.abs(views['symmetric_improvement_credit_Q'] + views['symmetric_improvement_credit_R'] + delta[:, :, 2]))
    assert credit_error <= TOL
    checks.update(minimum_KL=minimum_kl, covariance_max_scaled_asymmetry=float(np.max(symmetry / scale)),
                  covariance_min_scaled_eigenvalue=float(np.min(eig[..., 0] / scale)),
                  covariance_expansion_max_abs=float(expansion), symmetric_credit_identity_max_abs=float(credit_error),
                  covariance_expansion_scope='Algebra from retained 3x3 covariance; direct mixed-response qualification belongs to producer.')
    return views, checks


def summarize_states(views, G):
    states = {}
    for i, label in enumerate(LABELS):
        sequences = {name: value[i].mean(axis=-1) for name, value in views.items()}
        groups = {}
        for bank, sl in BANKS.items():
            metrics = {name: describe(value[sl]) for name, value in sequences.items()}
            rms = {}
            for name in ('Q', 'R', 'full', 'additive', 'mixed'):
                variance = metrics['response_variance_' + name]['mean']
                # Only a display root; original signed variance always remains in metrics.
                rms[name] = dict(value=float(np.sqrt(max(0., variance))),
                                 tiny_negative_clamped_for_display=variance < 0)
            groups[bank] = dict(metrics=metrics, mean_finite_gram=G[i, sl].mean(axis=(0, 1)).tolist(),
                                finite_response_rms=rms, sequences=len(sequences['nll_base'][sl]),
                                tokens=len(sequences['nll_base'][sl]) * G.shape[2])
        states[label] = dict(groups=groups, per_sequence={name: value.tolist() for name, value in sequences.items()},
                             per_sequence_finite_gram=G[i].mean(axis=1).tolist())
    return states


def build_ratios(states):
    ratios = []
    metrics = ['forward_kl_Q', 'conditional_kl_Q_last', 'forward_kl_R', 'forward_kl_full', 'conditional_kl_R_last']
    for family, setting, numerator, denominator, relation, bound in FACTORIAL:
        for bank in BANKS:
            for metric in metrics + ['finite_response_rms_Q', 'finite_response_rms_R', 'finite_response_rms_full']:
                is_rms = metric.startswith('finite_response_rms_')
                source_metric = 'response_variance_' + metric.removeprefix('finite_response_rms_') if is_rms else metric
                a = states[numerator]['groups'][bank]['metrics'][source_metric]['mean']
                b = states[denominator]['groups'][bank]['metrics'][source_metric]['mean']
                value = ratio_value(a, b)
                if is_rms:
                    if value['ratio'] is not None:
                        value['ratio'] = float(np.sqrt(value['ratio']))
                    value['numerator_mean'] = float(np.sqrt(a)) if a >= 0 else None
                    value['denominator_mean'] = float(np.sqrt(b)) if b >= 0 else None
                primary_metric = metric in ('forward_kl_Q', 'conditional_kl_Q_last')
                resolved = value['defined'] and (not primary_metric or (a > TOL and b > TOL))
                resolution_reason = (None if resolved else value['reason'] or
                                     'primary_KL_mean_at_or_below_1e-10')
                distance = None if value['ratio'] is None or not primary_metric else (
                    value['ratio'] - bound if relation == '>' else bound - value['ratio'])
                ratios.append(dict(family=family, setting=setting, numerator=numerator, denominator=denominator,
                                   bank=bank, metric=metric, **value,
                                   input_units='RMS=sqrt(mean finite-response variance)' if is_rms else 'KL',
                                   source_numerator_variance=float(a) if is_rms else None,
                                   source_denominator_variance=float(b) if is_rms else None,
                                   numerically_resolved=bool(resolved), resolution_reason=resolution_reason,
                                   criterion_relation=relation if primary_metric else None,
                                   criterion_bound=bound if primary_metric else None,
                                   signed_distance_to_bound=distance,
                                   condition_met=None if distance is None or not resolved else bool(distance > 0)))
    return ratios


def decision(states, ratios):
    predictions = {}
    details = {}
    for name, metric in [('P1_isolated_factorial_transmission', 'forward_kl_Q'),
                         ('P2_conditional_factorial_transmission', 'conditional_kl_Q_last')]:
        cells = [r for r in ratios if r['metric'] == metric and r['bank'] != 'all']
        assert len(cells) == 8
        details[name] = cells
        predictions[name] = all(r['condition_met'] is True for r in cells)
    useful = []
    for label in ('e028_b08', 'e040_b08'):
        for bank in ('bank0', 'bank1', 'all'):
            v = states[label]['groups'][bank]['metrics']['delta_nll_Q_last']['mean']
            passed = v <= -MATERIAL if bank == 'all' else v < 0
            useful.append(dict(label=label, bank=bank, mean_QK_last_loss=v,
                               relation='<=' if bank == 'all' else '<',
                               threshold=-MATERIAL if bank == 'all' else 0., condition_met=bool(passed)))
    name = 'P3_short_momentum_local_usefulness'
    details[name] = useful
    predictions[name] = all(r['condition_met'] for r in useful)
    supported = all(predictions.values())
    undefined = [r for r in ratios if r['metric'] in ('forward_kl_Q', 'conditional_kl_Q_last')
                 and r['bank'] != 'all' and not r['defined']]
    unresolved = [r for r in ratios if r['metric'] in ('forward_kl_Q', 'conditional_kl_Q_last')
                  and r['bank'] != 'all' and not r['numerically_resolved']]
    return dict(predictions=predictions, checks=details, all_three_supported=supported,
                classification='selected_state_functional_and_local_premise_supported' if supported else
                               'fixed_bridge_predictions_not_all_met',
                undefined_required_ratios=undefined,
                unresolved_required_ratios=unresolved,
                next_decision='A later separately designed causal question may be considered; no automatic intervention.' if supported else
                              'Close this fixed bridge without more states, heads, inputs or fitted rescaling.',
                no_inferential_test=True)


def synthetic_qualification():
    """Finite-logit identities plus isolated/conditional/cell mapping, without file reads."""
    rng = np.random.default_rng(3917)
    z = rng.normal(size=(4, 16, 4, 3, 7))
    target = rng.integers(0, 7, size=(4, 16, 3))
    def logsumexp(x):
        top = x.max(axis=-1, keepdims=True)
        return (top + np.log(np.exp(x - top).sum(axis=-1, keepdims=True))).squeeze(-1)
    lp = z - logsumexp(z)[..., None]
    prob = np.exp(lp)
    gather = np.broadcast_to(target[:, :, None, :, None], (4, 16, 4, 3, 1))
    L = -np.take_along_axis(lp, gather, axis=-1)[..., 0]
    p0, lp0 = prob[:, :, 0], lp[:, :, 0]
    K = np.sum(p0[:, :, None] * (lp0[:, :, None] - lp[:, :, 1:]), axis=-1)
    C = np.stack([np.sum(prob[:, :, 2] * (lp[:, :, 2] - lp[:, :, 3]), axis=-1),
                  np.sum(prob[:, :, 1] * (lp[:, :, 1] - lp[:, :, 3]), axis=-1)], axis=2)
    raw = lp[:, :, 1] + lp[:, :, 2] - lp0
    overlap = logsumexp(raw)
    lpadd = raw - overlap[..., None]
    addloss = -np.take_along_axis(lpadd, target[..., None], axis=-1)[..., 0]
    d = lp[:, :, 1:] - lp0[:, :, None]
    centered = d - np.sum(p0[:, :, None] * d, axis=-1, keepdims=True)
    G = np.einsum('mntv,mnitv,mnjtv->mntij', p0, centered, centered)
    linear = np.sum(p0[:, :, None] * d, axis=-1) - np.take_along_axis(d, gather[:, :, 1:], axis=-1)[..., 0]
    interaction = L[:, :, 3] - L[:, :, 1] - L[:, :, 2] + L[:, :, 0]
    J = linear[:, :, 2] - linear[:, :, 0] - linear[:, :, 1]
    parts = np.stack([interaction, overlap, L[:, :, 3] - addloss, J,
                      np.sum(p0 * (lp0 - lpadd), axis=-1), -np.sum(p0 * lp0, axis=-1)], axis=2)
    views, checks = derive(L, K, C, G, parts, linear)
    assert np.array_equal(views['conditional_kl_Q_last'], C[:, :, 0])
    assert np.array_equal(views['conditional_kl_R_last'], C[:, :, 1])
    assert np.array_equal(views['delta_nll_Q_last'], L[:, :, 3] - L[:, :, 2])
    states = summarize_states(views, G)
    assert states['e040_b08']['groups']['bank1']['sequences'] == 8
    assert abs(states['e028_b09']['groups']['bank0']['metrics']['nll_base']['mean'] - L[0, :8, 0].mean()) < 1e-14
    # Exact mapping/decision fixtures; these are not scientifically scored data.
    movement = [1., 1.5, 1.3, 1.8]
    conditional = [1., 1.6, 1.2, 1.9]
    for i, label in enumerate(LABELS):
        for bank in BANKS:
            metrics = states[label]['groups'][bank]['metrics']
            metrics['forward_kl_Q']['mean'] = movement[i]
            metrics['conditional_kl_Q_last']['mean'] = conditional[i]
            metrics['delta_nll_Q_last']['mean'] = -.01
    assert decision(states, build_ratios(states))['all_three_supported']
    import copy
    bad = copy.deepcopy(states)
    bad['e040_b08']['groups']['bank1']['metrics']['forward_kl_Q']['mean'] = .9
    d1 = decision(bad, build_ratios(bad))
    assert not d1['predictions']['P1_isolated_factorial_transmission'] and d1['predictions']['P2_conditional_factorial_transmission']
    bad = copy.deepcopy(states)
    bad['e040_b08']['groups']['bank0']['metrics']['conditional_kl_Q_last']['mean'] = .5
    assert not decision(bad, build_ratios(bad))['predictions']['P2_conditional_factorial_transmission']
    bad = copy.deepcopy(states)
    bad['e028_b08']['groups']['all']['metrics']['delta_nll_Q_last']['mean'] = -.001
    assert not decision(bad, build_ratios(bad))['predictions']['P3_short_momentum_local_usefulness']
    bad = copy.deepcopy(states)
    bad['e040_b08']['groups']['bank0']['metrics']['delta_nll_Q_last']['mean'] = 0.
    assert not decision(bad, build_ratios(bad))['predictions']['P3_short_momentum_local_usefulness']
    bad = copy.deepcopy(states)
    bad['e040_b09']['groups']['bank0']['metrics']['forward_kl_Q']['mean'] = LR_BOUND
    assert not decision(bad, build_ratios(bad))['predictions']['P1_isolated_factorial_transmission']
    bad = copy.deepcopy(states)
    bad['e028_b09']['groups']['bank0']['metrics']['forward_kl_Q']['mean'] = 1e-12
    bad['e028_b08']['groups']['bank0']['metrics']['forward_kl_Q']['mean'] = 1.5e-12
    tiny_ratios = build_ratios(bad)
    tiny = next(r for r in tiny_ratios if r['family'] == 'beta08_over09' and r['setting'] == 'LR .028'
                and r['bank'] == 'bank0' and r['metric'] == 'forward_kl_Q')
    assert tiny['ratio'] == 1.5 and tiny['defined'] and tiny['condition_met'] is None
    assert tiny['resolution_reason'] == 'primary_KL_mean_at_or_below_1e-10'
    assert decision(bad, tiny_ratios)['unresolved_required_ratios']
    assert not decision(bad, tiny_ratios)['predictions']['P1_isolated_factorial_transmission']
    assert ratio_value(1., 0.)['ratio'] is None and ratio_value(-1e-12, 1.)['ratio'] is None
    return dict(status='passed', scientific_outputs_read=False, corner_and_bank_mapping=True,
                direct_finite_logit_identities=checks, factorial_orientation_and_strict_bounds=True,
                conditional_not_substituted_for_isolated=True, pooled_materiality_and_bank_signs=True,
                undefined_denominators_not_regularized=True,
                sub_tolerance_KL_retained_but_cannot_pass=True)


def make_figure(states, ratios, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), constrained_layout=True)
    ticklabels = ['beta @.028\n>1', 'beta @.04\n>1', 'LR @.9\n<2.041', 'LR @.8\n<2.041']
    for ax, metric, title in zip(axes[:2], ['forward_kl_Q', 'conditional_kl_Q_last'],
                                 ['Q/K alone: KL(base || Q)', 'Q/K last: KL(rest || full)']):
        for bank, offset, marker in [('all', 0., 'o'), ('bank0', -.1, 's'), ('bank1', .1, '^')]:
            rows = [r for r in ratios if r['metric'] == metric and r['bank'] == bank]
            values = [np.nan if r['ratio'] is None else r['ratio'] for r in rows]
            ax.plot(np.arange(4) + offset, values, marker=marker, linewidth=1 if bank == 'all' else 0,
                    alpha=1 if bank == 'all' else .7, label=bank)
        ax.plot(np.arange(4), [1., 1., LR_BOUND, LR_BOUND], 'k_', markersize=14, label='Declared bounds')
        ax.set_xticks(np.arange(4), ticklabels, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Treatment-cell ratio of mean KL')
        ax.grid(alpha=.15)
    ax = axes[2]
    for bank, offset, marker in [('all', 0., 'o'), ('bank0', -.1, 's'), ('bank1', .1, '^')]:
        values = [states[k]['groups'][bank]['metrics']['delta_nll_Q_last']['mean'] for k in LABELS]
        ax.plot(np.arange(4) + offset, values, marker=marker, linewidth=1 if bank == 'all' else 0,
                alpha=1 if bank == 'all' else .7, label=bank)
    ax.axhline(0, color='gray', linewidth=.8)
    ax.axhline(-MATERIAL, color='black', linestyle=':', linewidth=.8, label='Pooled beta-.8 floor')
    ax.set_xticks(np.arange(4), ['.028/.9', '.028/.8', '.04/.9', '.04/.8'], fontsize=8)
    ax.set_xlabel('LR / momentum beta')
    ax.set_ylabel('NLL(full) - NLL(rest)')
    ax.set_title('Signed Q/K-last loss effect', fontsize=10)
    ax.grid(alpha=.15)
    axes[0].legend(frameon=False, fontsize=7)
    fig.suptitle('Actual 46→47 writes at four own states; no causal rate attribution', fontsize=11)
    fig.supxlabel('Two contiguous eight-sequence banks; points are descriptive, not confidence intervals.', fontsize=8)
    fig.savefig(out / 'qk_function.png', dpi=170)
    fig.savefig(out / 'qk_function.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    qualification = synthetic_qualification()
    if args.self_test:
        print(json.dumps(qualification, indent=2, allow_nan=False))
        return
    started = time.monotonic()
    out = HERE / 'run1'
    report_path = out / 'result.json'
    report = json.loads(report_path.read_text())
    assert report['status'] == 'complete', 'Do not read scientific arrays before producer completion.'
    if (out / 'status.json').exists():
        assert json.loads((out / 'status.json').read_text())['status'] == 'complete'
    assert report['labels'] == LABELS and report['parts'] == PARTS
    clarification_path = HERE / 'readout_clarification.json'
    annotated_protocol = HERE / 'PROTOCOL_with_readout_clarification.md'
    clarification = json.loads(clarification_path.read_text())
    assert sha(HERE / 'PROTOCOL.md') == clarification['original_protocol_sha256']
    assert sha(annotated_protocol) == clarification['annotated_copy_sha256']
    paths = [report_path, out / 'scalars.npz', HERE / 'PROTOCOL.md', Path(__file__),
             clarification_path, annotated_protocol]
    if (out / 'executed_protocol.md').exists():
        assert sha(out / 'executed_protocol.md') == clarification['original_protocol_sha256']
        paths.append(out / 'executed_protocol.md')
    manifest = {str(p.relative_to(HERE)): dict(bytes=p.stat().st_size, sha256=sha(p)) for p in paths}
    with np.load(out / 'scalars.npz', allow_pickle=False) as data:
        arrays = [np.array(data[k], dtype=np.float64, copy=True) for k in
                  ('corner_nll', 'forward_kl', 'conditional_kl', 'finite_gram', 'parts', 'linear')]
    L, K, C, G, P, A = arrays
    assert L.shape == (4, 16, 4, 512)
    views, checks = derive(L, K, C, G, P, A)
    states = summarize_states(views, G)
    ratios = build_ratios(states)
    verdict = decision(states, ratios)
    means = []
    for label, entry in states.items():
        for bank, group in entry['groups'].items():
            for metric, summary in group['metrics'].items():
                means.append(dict(label=label, bank=bank, metric=metric, tokens=group['tokens'], **summary))
    for p in paths:
        assert sha(p) == manifest[str(p.relative_to(HERE))]['sha256'], ('input_changed', str(p))
    result = dict(status='complete', scope=SCOPE, labels=LABELS, banks={'bank0': [0, 8], 'bank1': [8, 16]},
                  covariance_order=['Q', 'R', 'full'], lr_KL_bound=LR_BOUND, local_usefulness_floor=-MATERIAL,
                  synthetic_qualification=qualification, producer_qualification=report.get('qualification'),
                  numerical_checks=checks, input_manifest=manifest, readout_clarification=clarification,
                  primary_KL_resolution_tolerance=TOL, states=states, ratios=ratios, **verdict,
                  interpretation_limits=[
                      'All ratios compare treatment cells, never a component fraction of near-zero full movement.',
                      'Finite-response covariance and RMS are descriptive, not tangent GN or causal attribution.',
                      'Symmetric improvement credit is a two-order averaging convention; it is not the P3 criterion.',
                      'Positive interaction can reflect useful overlap; negative interaction need not be nonlinear benefit.',
                      'Tiny differences from declared bounds do not establish a large compensation effect.',
                      'Primary ratios retain their raw value but cannot pass when either mean KL is at or below 1e-10.',
                      'A failed local premise does not prove Q/K was useless throughout training.',
                      'No state, head, bank, metric or threshold may be substituted after seeing these outcomes.'],
                  seconds=time.monotonic() - started)
    write_json(out / 'analysis.json', result)
    write_csv(out / 'means.csv', means)
    write_csv(out / 'ratios.csv', ratios)
    make_figure(states, ratios, out)
    print(json.dumps(dict(predictions=verdict['predictions'], classification=verdict['classification'],
                          numerical_checks=checks, seconds=time.monotonic() - started), indent=2))


if __name__ == '__main__':
    main()

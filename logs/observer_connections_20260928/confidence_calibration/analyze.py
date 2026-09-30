"""Fixed opposite-bank logit-scale analysis. No model imports or model calls.

Run after probe completion, or run --self-test for outcome-free qualification.
"""
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

LABELS = ['h1_b08', 'h1_b09', 'h1_warm', 'h2_b08', 'h2_b09', 'h2_warm']
SCALES = np.array([.8, .9, 1., 1.1, 1.2], dtype=np.float64)
TENTHS = (8, 9, 10, 11, 12)
BASE = 2
BANKS = (slice(0, 16), slice(16, 32))
MATERIAL = .005
SCOPE = (
    'Six fixed SOAP-PD schedule endpoints, one seed, two fixed contiguous '
    '16-sequence banks. Select logit scale on one bank and score the other. '
    'The finite-grid materiality readings are neither significance nor '
    'equivalence tests, and identify no causal training mechanism or phase clock.'
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(path, data):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def choose_scale(mean_losses):
    """Exact loss tie: nearest one in integer tenths, then lower scale."""
    values = np.asarray(mean_losses, dtype=np.float64)
    assert values.shape == (5,) and np.isfinite(values).all()
    return min(range(5), key=lambda j: (float(values[j]), abs(TENTHS[j] - 10), TENTHS[j]))


def crossfit_one(sequence_nll, label):
    """Input is [32 sequences, 5 scales], each sequence already token-averaged."""
    sequence_nll = np.asarray(sequence_nll, dtype=np.float64)
    assert sequence_nll.shape == (32, 5) and np.isfinite(sequence_nll).all()
    bank_means = np.stack([sequence_nll[bank].mean(axis=0) for bank in BANKS])
    folds = []
    per_sequence_calibrated = np.empty(32, dtype=np.float64)
    for train_bank, score_bank in ((0, 1), (1, 0)):
        selected = choose_scale(bank_means[train_bank])
        scored = sequence_nll[BANKS[score_bank], selected]
        per_sequence_calibrated[BANKS[score_bank]] = scored
        # Match the selected-score reduction path exactly at scale one.
        raw = float(sequence_nll[BANKS[score_bank], BASE].mean())
        calibrated = float(scored.mean())
        folds.append(dict(label=label, fold=f'fit_{"AB"[train_bank]}_score_{"AB"[score_bank]}',
                          fit_bank='AB'[train_bank], score_bank='AB'[score_bank],
                          fit_bank_index=train_bank, score_bank_index=score_bank,
                          fit_sequences=16, score_sequences=16, score_tokens=8192,
                          selected_index=selected, selected_scale=float(SCALES[selected]),
                          boundary=selected in (0, 4),
                          selected_fit_nll=float(bank_means[train_bank, selected]),
                          fit_raw_nll=float(bank_means[train_bank, BASE]),
                          raw_nll=raw, calibrated_nll=calibrated,
                          correction=calibrated - raw))
    pooled_raw = float(np.mean([f['raw_nll'] for f in folds]))
    pooled_calibrated = float(np.mean([f['calibrated_nll'] for f in folds]))
    assert abs(pooled_calibrated - float(per_sequence_calibrated.mean())) < 1e-12
    return dict(label=label, bank_grid_means=bank_means.tolist(), folds=folds,
                pooled_raw_nll=pooled_raw, pooled_calibrated_nll=pooled_calibrated,
                pooled_correction=pooled_calibrated - pooled_raw,
                all_selections_one=all(f['selected_index'] == BASE for f in folds),
                per_sequence_calibrated_nll=per_sequence_calibrated.tolist())


def comparison(warm, constant, horizon, primary):
    """The primary original order flips by horizon; beta-.9 is secondary."""
    # Original validation: warm worse than beta .8 at 1x; warm better in
    # the other three fixed comparisons. Never choose order from new scores.
    expected_raw_sign = 1 if primary and horizon == 1 else -1
    expected_change_sign = -expected_raw_sign
    folds = []
    for w, c in zip(warm['folds'], constant['folds']):
        assert (w['fit_bank'], w['score_bank']) == (c['fit_bank'], c['score_bank'])
        raw = w['raw_nll'] - c['raw_nll']
        calibrated = w['calibrated_nll'] - c['calibrated_nll']
        folds.append(dict(fold=w['fold'], fit_bank=w['fit_bank'], score_bank=w['score_bank'],
                          warm_scale=w['selected_scale'], constant_scale=c['selected_scale'],
                          warm_boundary=w['boundary'], constant_boundary=c['boundary'],
                          raw_gap=raw, calibrated_gap=calibrated, gap_change=calibrated - raw,
                          warm_correction=w['correction'], constant_correction=c['correction'],
                          original_loser_correction=w['correction'] if expected_raw_sign > 0 else c['correction'],
                          absolute_gap_shrinks=abs(calibrated) < abs(raw),
                          reversal=raw * calibrated < 0))
    changes = np.array([f['gap_change'] for f in folds])
    pooled = {key: float(np.mean([f[key] for f in folds]))
              for key in ('raw_gap', 'calibrated_gap', 'gap_change', 'warm_correction', 'constant_correction')}
    all_one = warm['all_selections_one'] and constant['all_selections_one']
    gates = dict(
        raw_order_reproduced_both_banks=all(expected_raw_sign * f['raw_gap'] > 0 for f in folds),
        all_compared_selections_interior=all(not f['warm_boundary'] and not f['constant_boundary'] for f in folds),
        differential_signs_agree=bool(np.sign(changes[0]) == np.sign(changes[1])),
        correction_toward_original_zero_both_folds=all(expected_change_sign * f['gap_change'] > 0 for f in folds),
        absolute_gap_shrinks_both_folds=all(f['absolute_gap_shrinks'] for f in folds),
        original_loser_improves_both_folds=all(f['original_loser_correction'] < 0 for f in folds),
        calibrated_original_order_both_folds=all(expected_raw_sign * f['calibrated_gap'] > 0 for f in folds),
        pooled_material_correction=expected_change_sign * pooled['gap_change'] >= MATERIAL,
        pooled_differential_below_material=abs(pooled['gap_change']) < MATERIAL,
        all_compared_selections_one=all_one,
    )
    base_ok = (gates['raw_order_reproduced_both_banks'] and gates['all_compared_selections_interior']
               and gates['differential_signs_agree'])
    material = (base_ok and gates['correction_toward_original_zero_both_folds']
                and gates['absolute_gap_shrinks_both_folds'] and gates['original_loser_improves_both_folds']
                and gates['pooled_material_correction'])
    survives = (base_ok and gates['calibrated_original_order_both_folds']
                and gates['pooled_differential_below_material'] and not all_one)
    if not primary:
        classification = 'secondary_descriptive_only'
    elif all_one:
        classification = 'coarse_grid_resolves_no_finite_correction' if gates['raw_order_reproduced_both_banks'] else 'unresolved'
    elif material:
        classification = 'material_confidence_component_in_this_panel'
    elif survives:
        classification = 'ranking_survives_this_modest_family'
    else:
        classification = 'unresolved'
    reasons = []
    if not gates['raw_order_reproduced_both_banks']: reasons.append('original_raw_order_not_reproduced_in_both_banks')
    if not gates['all_compared_selections_interior']: reasons.append('selected_scale_hits_fixed_grid_boundary')
    if not gates['differential_signs_agree']: reasons.append('differential_corrections_disagree_in_sign')
    if all_one: reasons.append('both_models_choose_one_in_both_folds; finer_scales_not_resolved')
    if gates['absolute_gap_shrinks_both_folds'] and not gates['original_loser_improves_both_folds']:
        reasons.append('gap_attenuation_does_not_improve_original_loser_in_both_folds')
    if primary and classification == 'unresolved' and not reasons:
        reasons.append('fixed_material_or_ranking_survival_conditions_not_all_met')
    return dict(horizon=horizon, primary=primary, warm_label=warm['label'], constant_label=constant['label'],
                expected_raw_gap_sign=expected_raw_sign, materiality_threshold=MATERIAL,
                folds=folds, pooled=pooled, gates=gates, classification=classification, reasons=reasons)


def synthetic_qualification():
    """Outcome-free tests of selection, opposite-bank scores, and interpretation."""
    assert choose_scale([1, 1, 1, 1, 1]) == 2
    assert choose_scale([3, 1, 2, 1, 3]) == 1
    assert choose_scale([1, 3, 4, 3, 1]) == 0
    assert choose_scale([2, 1, 1 + 1e-12, 2, 3]) == 1  # not a tolerance tie
    seq = np.vstack([np.tile([5, 1, 2, 4, 6], (16, 1)), np.tile([6, 4, 3, 1, 5], (16, 1))])
    model = crossfit_one(seq, 'synthetic')
    f, g = model['folds']
    assert f['fit_bank'] == 'A' and f['score_bank'] == 'B' and f['selected_scale'] == .9
    assert g['fit_bank'] == 'B' and g['score_bank'] == 'A' and g['selected_scale'] == 1.1
    assert (f['raw_nll'], f['calibrated_nll'], f['correction']) == (3., 4., 1.)
    assert (g['raw_nll'], g['calibrated_nll'], g['correction']) == (2., 4., 2.)
    assert model['pooled_calibrated_nll'] == 4. and model['pooled_correction'] == 1.5
    assert model['per_sequence_calibrated_nll'] == [4.] * 32

    # Nonexact decimals expose the old axis-0 versus vector-mean rounding
    # difference (~1 ulp) even when selected and raw scales are identical.
    decimals = np.random.default_rng(19).uniform(1.25, 6.75, 32)
    one_grid = decimals[:, None] + .05 * (np.asarray(TENTHS) - 10)**2
    identical = crossfit_one(one_grid, 'synthetic_all_one_decimals')
    assert identical['all_selections_one']
    for fold in identical['folds']:
        assert fold['raw_nll'] == fold['calibrated_nll']
        assert fold['correction'] == 0.
    assert identical['pooled_raw_nll'] == identical['pooled_calibrated_nll']
    assert identical['pooled_correction'] == 0.

    def made(label, raw, calibrated, scales=(.9, .9)):
        fs = []
        for i, (fit, score) in enumerate((('A', 'B'), ('B', 'A'))):
            a = float(np.atleast_1d(raw)[i if np.ndim(raw) else 0])
            b = float(np.atleast_1d(calibrated)[i if np.ndim(calibrated) else 0])
            s = scales[i]
            fs.append(dict(label=label, fold=f'fit_{fit}_score_{score}', fit_bank=fit, score_bank=score,
                           raw_nll=a, calibrated_nll=b, correction=b-a, selected_scale=s,
                           selected_index=TENTHS.index(round(s*10)), boundary=s in (.8, 1.2)))
        return dict(label=label, folds=fs, all_selections_one=all(s == 1 for s in scales))

    warm = made('h1_warm', 2.04, 2.)
    ref = made('h1_b08', 2., 1.99)
    assert comparison(warm, ref, 1, True)['classification'] == 'material_confidence_component_in_this_panel'
    assert comparison(made('h2_warm', 1.98, 1.98, (1, 1)), made('h2_b08', 2., 1.99), 2, True)['classification'] == 'material_confidence_component_in_this_panel'
    # Gap shrinks only by harming winner: this must not become confidence evidence.
    harm = comparison(made('h1_warm', 2.04, 2.04, (1, 1)), made('h1_b08', 2., 2.03), 1, True)
    assert harm['classification'] == 'unresolved' and not harm['gates']['original_loser_improves_both_folds']
    assert comparison(made('h1_warm', 2.04, 2., (.8, .8)), ref, 1, True)['classification'] == 'unresolved'
    assert comparison(made('h1_warm', 2.04, 2.04, (1, 1)), made('h1_b08', 2., 2., (1, 1)), 1, True)['classification'] == 'coarse_grid_resolves_no_finite_correction'
    assert comparison(made('h1_warm', [2.04, 1.99], [2., 1.96]), ref, 1, True)['classification'] == 'unresolved'
    assert comparison(made('h1_warm', 2.04, [2., 2.06]), made('h1_b08', 2., 2., (1, 1)), 1, True)['classification'] == 'unresolved'
    # A surviving small differential uses nontrivial interior selections.
    assert comparison(made('h1_warm', 2.04, 2.03), made('h1_b08', 2., 1.99), 1, True)['classification'] == 'ranking_survives_this_modest_family'
    # Secondary h1 warm is originally BETTER than beta .9, unlike the primary.
    sec = comparison(made('h1_warm', 1.99, 1.98), made('h1_b09', 2., 1.99), 1, False)
    assert sec['expected_raw_gap_sign'] == -1 and sec['classification'] == 'secondary_descriptive_only'
    return dict(status='passed', exact_tie_rules=True, opposite_bank_mapping=True,
                exact_scale_one_reduction=True,
                correction_signs=True, original_loser_gate=True, boundary_and_all_one_gates=True,
                original_raw_order_and_fold_disagreement_gates=True,
                secondary_original_order=True, scientific_outputs_read=False)


def figure(comparisons, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True, constrained_layout=True)
    colors = ('#287c8e', '#9961a8')
    for ax, c, color in zip(axes, [c for c in comparisons if c['primary']], colors):
        for i, fold in enumerate(c['folds']):
            ax.plot([0, 1], [fold['raw_gap'], fold['calibrated_gap']], marker=('o', 's')[i],
                    color=color, alpha=.5, linewidth=1.2,
                    label=f"Fit {fold['fit_bank']} / score {fold['score_bank']}")
        ax.plot([0, 1], [c['pooled']['raw_gap'], c['pooled']['calibrated_gap']], 'D-',
                color='#222222', linewidth=2, markersize=5, label='Pooled held-out score')
        ax.axhline(0, color='#777777', linewidth=.8)
        ax.set_xticks([0, 1], ['Raw', 'Cross-fitted scales'])
        ax.set_title(f"{c['horizon']}× horizon")
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('NLL gap: warmup − constant beta .8')
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle('Global confidence correction at fixed schedule endpoints', fontsize=12)
    fig.supxlabel('One seed; two contiguous banks. Fold markers are not confidence intervals.', fontsize=8)
    fig.savefig(out/'confidence_gaps.png', dpi=180)
    fig.savefig(out/'confidence_gaps.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    check = synthetic_qualification()
    if args.self_test:
        print(json.dumps(check, indent=2))
        return
    started = time.monotonic()
    out = HERE/'run1'
    result_path = out/'result.json'
    result = json.loads(result_path.read_text())
    assert result['status'] == 'complete', 'Never analyze an unfinished scientific run.'
    assert result['labels'] == LABELS and np.array_equal(np.asarray(result['scales']), SCALES)
    assert isinstance(result['qualification'], dict)
    if (out/'status.json').exists():
        assert json.loads((out/'status.json').read_text())['status'] == 'complete'
    input_paths = [result_path, out/'per_token.npz', HERE/'PROTOCOL.md', Path(__file__)]
    inputs = {str(p.relative_to(HERE)): dict(bytes=p.stat().st_size, sha256=sha256(p)) for p in input_paths}
    with np.load(out/'per_token.npz', allow_pickle=False) as data:
        nll = np.array(data['nll'], dtype=np.float64, copy=True)
        entropy = np.array(data['entropy'], dtype=np.float64, copy=True)
        derivative = np.array(data['derivative'], dtype=np.float64, copy=True)
        curvature = np.array(data['curvature'], dtype=np.float64, copy=True)
    assert nll.shape == (6, 32, 5, 512)
    assert entropy.shape == derivative.shape == curvature.shape == (6, 32, 512)
    assert all(np.isfinite(a).all() for a in (nll, entropy, derivative, curvature))
    assert float(curvature.min()) >= 0., 'Centered-second-moment curvature must be nonnegative.'
    identity = float(np.max(np.abs(nll[:, :, BASE, :] - entropy - derivative)))
    assert identity <= 1e-10, ('scale derivative identity', identity)
    seq_nll = nll.mean(axis=-1)
    models = [crossfit_one(seq_nll[i], label) for i, label in enumerate(LABELS)]
    by_label = {m['label']: m for m in models}
    comparisons = [comparison(by_label[f'h{h}_warm'], by_label[f'h{h}_{ref}'], h, ref == 'b08')
                   for h in (1, 2) for ref in ('b08', 'b09')]
    model_rows = []
    for m in models:
        for f in m['folds']:
            model_rows.append({**f, 'pooled_raw_nll': m['pooled_raw_nll'],
                               'pooled_calibrated_nll': m['pooled_calibrated_nll'],
                               'pooled_correction': m['pooled_correction']})
    contrast_rows = []
    for c in comparisons:
        for f in c['folds']:
            contrast_rows.append(dict(horizon=c['horizon'], primary=c['primary'],
                                      warm_label=c['warm_label'], constant_label=c['constant_label'],
                                      classification=c['classification'], **f,
                                      **{'pooled_'+k:v for k,v in c['pooled'].items()}))
    grid_rows = []
    sequence_records = []
    for i, label in enumerate(LABELS):
        for bank, sl in enumerate(BANKS):
            for k, scale in enumerate(SCALES):
                grid_rows.append(dict(label=label, bank='AB'[bank], scale=float(scale),
                                      sequences=16, tokens=8192,
                                      mean_nll=float(seq_nll[i, sl, k].mean()),
                                      delta_from_one=float((seq_nll[i, sl, k]-seq_nll[i, sl, BASE]).mean())))
        for sequence in range(32):
            sequence_records.append(dict(label=label, sequence=sequence,
                bank='AB'[int(sequence >= 16)], sequence_in_bank=sequence%16,
                tokens=512, nll_by_scale=seq_nll[i, sequence].tolist(),
                raw_nll=float(seq_nll[i, sequence, BASE]),
                cross_fitted_nll=models[i]['per_sequence_calibrated_nll'][sequence],
                entropy=float(entropy[i, sequence].mean()),
                derivative=float(derivative[i, sequence].mean()),
                curvature=float(curvature[i, sequence].mean())))
    for p in input_paths:
        assert inputs[str(p.relative_to(HERE))]['sha256'] == sha256(p), ('input changed during analysis', str(p))
    analysis = dict(status='complete', scope=SCOPE, labels=LABELS, scales=SCALES.tolist(),
                    materiality_threshold=MATERIAL, input_manifest=inputs,
                    synthetic_qualification=check,
                    producer_qualification=result['qualification'],
                    numerical_checks=dict(max_scale_derivative_identity_error=identity,
                                          minimum_scale_curvature=float(curvature.min()),
                                          scientific_sequences=192, prediction_token_positions=98304,
                                          grid_token_nlls=491520,
                                          scientific_shape=list(nll.shape)),
                    models=models, comparisons=comparisons, per_sequence_scores=sequence_records,
                    all_six_models_select_one=all(m['all_selections_one'] for m in models),
                    seconds=time.monotonic()-started,
                    interpretation_limits=[
                        'No confidence intervals or equivalence claim from one seed and two contiguous banks.',
                        'Selected endpoints do not establish a training-rate mechanism or universal phase clock.',
                        'Boundary winners and disagreeing folds are unresolved; do not expand this fixed panel.',
                        'All scale-one selections resolve no finite correction in this coarse family.',
                        'Secondary comparisons cannot replace a failed primary.',
                        'Positive logit scaling preserves token ranking and is an output-only intervention.'
                    ])
    save_csv(out/'model_scores.csv', model_rows)
    save_csv(out/'contrasts.csv', contrast_rows)
    save_csv(out/'grid_curves.csv', grid_rows)
    figure(comparisons, out)
    analysis['seconds'] = time.monotonic() - started
    save_json(out/'analysis.json', analysis)
    print(json.dumps(dict(status='complete', seconds=analysis['seconds'],
                         primary=[dict(horizon=c['horizon'], classification=c['classification'],
                                       pooled=c['pooled'], gates=c['gates'], reasons=c['reasons'])
                                  for c in comparisons if c['primary']]), indent=2))


if __name__ == '__main__':
    main()

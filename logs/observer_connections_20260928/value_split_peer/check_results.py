"""Independent scalar-only check of the frozen value-split result; no torch/model."""
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
RESULT = HERE.parent / 'value_split/result.json'
ANALYSIS = HERE.parent / 'value_split/analysis.json'
r = json.loads(RESULT.read_text())
a = json.loads(ANALYSIS.read_text())
assert r['status'] == 'complete'
checks = {'inputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in (RESULT, ANALYSIS)}, 'methods': {}}
pairs = {'constant': (1, 0), 'centered': (0, 1),
         'combined': (1, 1), 'half_combined': (.5, .5)}
for method, m in r['methods'].items():
    rows = [row for bank in m['banks'] for row in bank]
    inter, gp, ratios, errs = [], [], [], []
    predictionerr = {key: [] for key in pairs}
    for row in rows:
        loss, slope, g = row['losses'], row['slopes'], row['gn']
        mixed = loss['combined'] - loss['constant'] - loss['centered'] + loss['base']
        inter.append(mixed)
        gp.append(g[0][1])
        ratios.append(mixed / g[0][1])
        errs.append(mixed - g[0][1])
        for key, v in pairs.items():
            pred = sum(x*y for x, y in zip(slope, v)) + .5*sum(
                v[i]*g[i][j]*v[j] for i in range(2) for j in range(2))
            predictionerr[key].append(loss[key] - loss['base'] - pred)
    pooled = a['methods'][method]['pooled']
    g, eff = pooled['gn_mean'], pooled['effects']
    aggregate_errors = {key: eff[key]['mean'] - statistics.mean(
        row['losses'][key] - row['losses']['base'] for row in rows) for key in pairs}
    assert max(map(abs, aggregate_errors.values())) < 1e-15
    checks['methods'][method] = {
        'n': len(rows),
        'qualification': m['qualification'],
        'cross_gn_positive_count': sum(x > 0 for x in gp),
        'finite_interaction_positive_count': sum(x > 0 for x in inter),
        'finite_interaction_over_G12_range': [min(ratios), max(ratios)],
        'finite_interaction_over_G12_pooled': statistics.mean(inter)/statistics.mean(gp),
        'interaction_prediction_abs_error_max': max(map(abs, errs)),
        'mean_actual_minus_gn_prediction': {k: statistics.mean(v) for k, v in predictionerr.items()},
        'max_absolute_actual_minus_gn_prediction': {k: max(map(abs, v)) for k, v in predictionerr.items()},
        'constant_gain_lost_to_interaction_pooled': eff['interaction']['mean']/(-eff['constant']['mean']),
        'gn_cross_fraction': 2*g[0][1]/sum(map(sum, g)),
        'coupling_vs_block_diagonal': sum(map(sum, g))/(g[0][0]+g[1][1]),
        'aggregation_effect_error': aggregate_errors,
    }
(HERE / 'RESULT_CHECKS.json').write_text(json.dumps(checks, indent=2) + '\n')
print(json.dumps(checks, indent=2))

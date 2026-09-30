"""Read exactly 18 declared saved profiles; no torch or model execution."""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
AUDIT = ROOT / 'logs/muon_spectra/second_order_audit_20260926'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def preserve(source, target):
    data = source.read_bytes()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != data:
        raise RuntimeError(f'Refusing to replace changed source snapshot: {target}')
    if not target.exists():
        target.write_bytes(data)
    return {'source': str(source.relative_to(ROOT)), 'snapshot': str(target.relative_to(HERE)),
            'sha256': sha(data), 'bytes': len(data)}


def span(values):
    return {'min': min(values), 'median': statistics.median(values), 'max': max(values)}


def main():
    source = preserve(AUDIT / 'step_profile_probe.py', HERE / 'source/step_profile_probe.py')
    rows, snapshots = [], []
    for beta in (.9, .8):
        directory = AUDIT / ('step_profile' if beta == .9 else 'step_profile_mom')
        for method in ('M', 'PD', 'SPD'):
            for step in (9, 46, 83):
                matches = list(directory.glob(f'*__{method}_*b16M_*mom{beta}_*__{step}.json'))
                assert len(matches) == 1, matches
                f = matches[0]
                snapshots.append(preserve(f, HERE / 'inputs' / f.name))
                d = json.loads(f.read_text())
                assert d['step'] == step and d['batch_tokens'] == 16777216
                dirs = d['directions']
                r = {'method_state': method, 'beta': beta, 'step': step,
                     'item': d['item'], 'source': str(f.relative_to(ROOT)),
                     'top_ritz': d['ritz'][0], 'directions': dirs,
                     'gradient_pair_body_only': d['gradient_pair']}
                for label, v in dirs.items():
                    for key in ('norm', 'slope', 'curvature', 'curvature_curv_set'):
                        assert math.isfinite(v[key])
                    assert v['norm'] > 0 and v['curvature'] > 0
                    v['slope_per_norm'] = v['slope'] / v['norm']
                    v['curvature_per_norm2_held'] = v['curvature'] / v['norm'] ** 2
                    v['curvature_per_norm2_curv'] = v['curvature_curv_set'] / v['norm'] ** 2
                    v['curv_to_held_ratio'] = v['curvature_curv_set'] / v['curvature']
                    v['quality_nonnegative_step'] = max(-v['slope'], 0) ** 2 / (2*v['curvature'])
                    v['top16_norm2'] = v['energy_top16'] * v['norm'] ** 2
                    assert math.isclose(v['quality'], v['slope']**2/(2*v['curvature']), rel_tol=2e-6)
                    assert math.isclose(v['c_star'], -v['slope']/v['curvature'], rel_tol=2e-6)
                m, p, raw = dirs['muon'], dirs['pd'], dirs['momentum']
                ratios = {f'pd_muon_energy_top{k}': p[f'energy_top{k}']/m[f'energy_top{k}'] for k in (1,4,16)}
                ratios.update({
                    'pd_muon_top16_absolute_norm2': p['top16_norm2']/m['top16_norm2'],
                    'pd_muon_norm': p['norm']/m['norm'],
                    'pd_muon_curvature_per_norm2_held': p['curvature_per_norm2_held']/m['curvature_per_norm2_held'],
                    'pd_muon_curvature_per_norm2_curv': p['curvature_per_norm2_curv']/m['curvature_per_norm2_curv'],
                    'muon_momentum_energy_top16': m['energy_top16']/raw['energy_top16'],
                    'pd_momentum_energy_top16': p['energy_top16']/raw['energy_top16'],
                })
                r['same_state_ratios'] = ratios
                rows.append(r)
    assert len(rows) == 18
    ratios = {k: span([r['same_state_ratios'][k] for r in rows]) for k in rows[0]['same_state_ratios']}
    signs = {label: {'uphill': sum(r['directions'][label]['slope']>0 for r in rows),
                     'downhill': sum(r['directions'][label]['slope']<0 for r in rows)}
             for label in rows[0]['directions']}
    beta_pairs = []
    for method in ('M','PD','SPD'):
        for step in (9,46,83):
            pair = {r['beta']:r for r in rows if r['method_state']==method and r['step']==step}
            beta_pairs.append({'method_state':method,'step':step,
                'top_ritz_ratio_08_over_09':pair[.8]['top_ritz']/pair[.9]['top_ritz'],
                'energy_top16_ratio_08_over_09':{label: pair[.8]['directions'][label]['energy_top16']/pair[.9]['directions'][label]['energy_top16'] for label in pair[.8]['directions']}})
    result = {'scope':'Frozen ideal maps of lagged M_s, fresh full alpha-.5 root, no per-matrix norm matching. Actual is next body displacement; gradient pair is body-only. No causal stability or rate estimate.',
              'n':len(rows), 'source':source,'input_manifest':snapshots,
              'same_state_ratio_summary':ratios,'slope_sign_counts':signs,
              'different_state_beta_pairs':beta_pairs,'rows':rows}
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    with (HERE/'table.csv').open('w') as out:
        writer=csv.writer(out)
        writer.writerow(['method_state','beta','step','top_ritz','momentum_energy_top16','muon_energy_top16','pd_energy_top16','actual_energy_top16','pd_muon_held_rayleigh_ratio','muon_slope_per_norm','pd_slope_per_norm'])
        for r in rows:
            d=r['directions']; writer.writerow([r['method_state'],r['beta'],r['step'],r['top_ritz'],*[d[x]['energy_top16'] for x in ('momentum','muon','pd','actual')],r['same_state_ratios']['pd_muon_curvature_per_norm2_held'],d['muon']['slope_per_norm'],d['pd']['slope_per_norm']])
    lines=['| State | β | Step | M_s stiff fraction | polar(M_s) | PD(M_s) | Actual next body step |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        d=r['directions']; lines.append(f"| {r['method_state']} | {r['beta']} | {r['step']} | "+' | '.join(f"{d[x]['energy_top16']:.4g}" for x in ('momentum','muon','pd','actual'))+' |')
    (HERE/'TABLE.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'n':len(rows),'ratios':ratios,'slope_signs':signs,'different_state_beta_pairs':beta_pairs},indent=2))


if __name__=='__main__':
    main()

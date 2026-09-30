"""Independent saved-artifact checks only; no model, eigensolver, or GPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
from pathlib import Path
import json
import math
import statistics
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
here = Path(__file__).resolve().parent
run = here.parent / 'ts_repeatability/run2'
r = json.loads((run / 'result.json').read_text())
assert r['status'] == 'complete'
out = {'scope': 'Saved JSON and one 512x512 tensor per pair; no model or GPU.',
       'aggregation_checks': {}, 'stored_tensor_checks': {}}

for label, pair in r['direction_pairs'].items():
    for scope, record in pair['scopes'].items():
        mats = [v for k, v in pair['matrices'].items()
                if scope == 'all' or k.split('.')[-1] == scope]
        assert len(mats) == (48 if scope == 'all' else 8)
        sums = {k: sum(v[k] for v in mats) for k in
                ['norm2_a', 'norm2_b', 'dot', 'distance2', 'added_geometry_denominator']}
        cosine = sums['dot'] / math.sqrt(sums['norm2_a'] * sums['norm2_b'])
        ratio = sums['distance2'] / sums['added_geometry_denominator']
        assert abs(cosine - record['cosine']) < 1e-12
        assert abs(ratio - record['noise_to_added_geometry']) < 1e-12
    out['aggregation_checks'][label] = 'all 48 and all six kinds agree'

pd = torch.load(run / 'D_PD.pt', map_location='cpu', mmap=True,
                weights_only=True)['block01.q'].double()
for a, b in [('A0', 'B0'), ('AB', 'CD'), ('ABCD', 'ABCD_bf16')]:
    x = torch.load(run / f'D_{a}.pt', map_location='cpu', mmap=True,
                   weights_only=True)['block01.q'].double()
    y = torch.load(run / f'D_{b}.pt', map_location='cpu', mmap=True,
                   weights_only=True)['block01.q'].double()
    assert x.shape == y.shape == (512, 512)
    cosine = float((x*y).sum() / (x.norm()*y.norm()))
    distance = float((x-y).square().sum())
    ratio = distance / float((x-pd).square().sum() + (y-pd).square().sum())
    ref = r['direction_pairs'][a + '__' + b]['matrices']['block01.q']
    assert abs(cosine-ref['cosine']) < 1e-12
    assert abs(distance-ref['distance2']) < 1e-10
    assert abs(ratio-ref['noise_to_added_geometry']) < 1e-12
    out['stored_tensor_checks'][a + '__' + b] = {
        'matrix': 'block01.q', 'cosine': cosine, 'distance2': distance,
        'noise_to_added_geometry': ratio, 'matches_report': True}

out['damping_fraction_max_all_factor_sets'] = max(
    v['damping_fraction'] for factor in r['map_info'].values() for v in factor.values())
out['negative_eigenvalue_fraction_max_all_factor_sets'] = max(
    v['negative_eigenvalue_fraction'] for factor in r['map_info'].values() for v in factor.values())
out['per_kind_mean_cosines'] = {
    kind: {t: statistics.mean(pair['scopes'][kind]['cosine']
                             for pair in r['direction_pairs'].values()
                             if pair['comparison_type'] == t)
           for t in ['independent8', 'same_sequences_new_labels',
                     'independent16', 'ema_refresh_proxy']}
    for kind in ['q', 'k', 'v', 'o', 'up', 'down']}
e8 = [v['scopes']['all']['cosine'] for v in r['direction_pairs'].values()
      if v['comparison_type'] == 'independent8']
e16 = r['direction_pairs']['AB__CD']['scopes']['all']['cosine']
out['strict_flag'] = ('stable' if min(e8) >= .99 and e16 >= .995 else
                     'material_sample_sensitivity' if min(e8) <= .90
                     and e16 > statistics.mean(e8) else 'mixed')
assert out['strict_flag'] == json.loads((run/'analysis.json').read_text())['strict_descriptive_flag']
(here/'validation.json').write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))

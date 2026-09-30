"""CPU-only signed temporal moments in fixed, independently chosen rank groups."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time
import torch

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
AUDIT = ROOT / 'logs/muon_spectra/second_order_audit_20260926'
LABELS = ['rank1', 'ranks2_8', 'ranks9_64', 'ranks65_end']


def rank_index(n):
    v = torch.arange(n)
    return (v >= 1).long() + (v >= 8).long() + (v >= 64).long()


def derived(sums, n, batch, global_energy, global_count):
    a, b, cross = sums['mean_a2'], sums['mean_b2'], sums['cross']
    ca, cb = sums['noise_a'] / n, sums['noise_b'] / n
    corrected_a, corrected_b = a - ca, b - cb
    result = dict(sums)
    result.update({
        'signal_a2': corrected_a, 'signal_b2': corrected_b,
        'mean_noise_correction_a': ca, 'mean_noise_correction_b': cb,
        'noise_correction_fraction_a': ca / a if a else None,
        'noise_correction_fraction_b': cb / b if b else None,
        'cosine_raw': cross / math.sqrt(a * b) if a * b > 0 else None,
        'cosine_corrected': cross / math.sqrt(corrected_a * corrected_b)
            if min(corrected_a, corrected_b) > 0 else None,
        'pair_average_signal_energy': (corrected_a + corrected_b + 2 * cross) / 4,
        'pair_difference_signal_energy': (corrected_a + corrected_b - 2 * cross) / 4,
        'group_trace_snr_at_training_batch': corrected_a * batch / sums['noise_a'] if sums['noise_a'] else None,
        'update_energy_share': sums['update_energy'] / global_energy,
        'coordinate_share': sums['count'] / global_count,
        'update_energy_enrichment': (sums['update_energy'] / global_energy) / (sums['count'] / global_count),
        'noise_sensitivity': {},
    })
    for factor in [1.0, 1.1, 1.25]:
        sa, sb = a - factor * ca, b - factor * cb
        result['noise_sensitivity'][str(factor)] = {
            'signal_a2': sa, 'signal_b2': sb,
            'cosine': cross / math.sqrt(sa * sb) if min(sa, sb) > 0 else None,
        }
    return result


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def analyze(path):
    start = time.time()
    data = torch.load(path, weights_only=False, map_location='cpu', mmap=True)
    meta = data['meta']
    frame_path = AUDIT / 'frame' / (path.stem.replace('_t', '_step') + '.pt')
    frame = torch.load(frame_path, weights_only=False, map_location='cpu', mmap=True)
    assert Path(meta['arm']).resolve() == Path(frame['meta']['arm']).resolve()
    assert meta['t'] == frame['meta']['step']
    cfg_path = Path(meta['arm']) / 'config.json'
    cfg = json.loads(cfg_path.read_text())
    n = meta['sequences']
    batch = cfg['batch_tokens'] / cfg['model']['seq_len']
    labels = [k[:-7] for k in next(iter(data['matrices'].values())) if k.endswith(':signal') and k != 't:signal']
    grouped = {label: defaultdict(lambda: defaultdict(lambda: torch.zeros(16, dtype=torch.float64))) for label in labels}
    spectra = []
    for name, m in data['matrices'].items():
        fm = frame['matrices'][name]
        a, na = m['t:signal'].double(), m['t:noise'].double()
        update = fm['update'].double()
        ds = update - fm['decay'].double()
        index = (4 * rank_index(a.shape[0])[:, None] + rank_index(a.shape[1])[None, :]).reshape(-1)
        common = {'count': torch.ones_like(a), 'mean_a2': a.square(), 'noise_a': na,
                  'update_energy': update.square(), 'decay_subtracted_energy': ds.square()}
        kind = name.split('.')[-1]
        for label in labels:
            b, nb = m[label + ':signal'].double(), m[label + ':noise'].double()
            values = {**common, 'mean_b2': b.square(), 'noise_b': nb, 'cross': a * b}
            for key, v in values.items():
                accum = torch.bincount(index, weights=v.reshape(-1), minlength=16)
                for group in ['all', kind]: grouped[label][group][key] += accum
        for side in ['lam_B', 'lam_C']:
            lam, other = m[side].double(), fm[side].double()
            spectra.append({'matrix': name, 'side': side,
                'bitwise_equal': torch.equal(lam, other),
                'max_difference_over_top': float((lam - other).abs().max() / lam.abs().max()),
                'boundary_gap_over_local_eigenvalue': {str(i): float((lam[i-1] - lam[i]) / lam[i-1]) for i in [1, 8, 64] if i < len(lam)}})
    output = {}
    for label, kinds in grouped.items():
        global_energy = float(kinds['all']['update_energy'].sum())
        global_count = float(kinds['all']['count'].sum())
        output[label] = {}
        for kind, arrays in kinds.items():
            subsets = {'all': list(range(16))}
            subsets.update({f'out:{LABELS[i]}': list(range(i * 4, i * 4 + 4)) for i in range(4)})
            subsets.update({f'in:{LABELS[j]}': [4*i+j for i in range(4)] for j in range(4)})
            subsets.update({f'out:{LABELS[i]}/in:{LABELS[j]}': [4*i+j] for i in range(4) for j in range(4)})
            output[label][kind] = {}
            for group, idx in subsets.items():
                sums = {key: float(arr[idx].sum()) for key, arr in arrays.items()}
                output[label][kind][group] = derived(sums, n, batch, global_energy, global_count)
    return {
        'meta': meta, 'frame_meta': frame['meta'], 'batch_sequences': batch,
        'groups': output, 'spectral_alignment_checks': spectra,
        'source_files': {str(p.relative_to(ROOT)): {'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns, 'sha256': digest(p)} for p in [path, frame_path, cfg_path]},
        'seconds': time.time() - start,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', nargs='+', type=int, default=[500])
    parser.add_argument('--out', default='persistence_groups.json')
    args = parser.parse_args()
    paths = sorted(p for step in args.steps for p in (AUDIT / 'persistence').glob(f'*_t{step:06d}.pt'))
    records = []
    for path in paths:
        d = analyze(path)
        records.append(d)
        keys = ['signal_a2', 'signal_b2', 'cross', 'cosine_corrected', 'noise_correction_fraction_a', 'update_energy_share', 'coordinate_share', 'update_energy_enrichment', 'group_trace_snr_at_training_batch']
        concise = {label: {group: {k: vals[k] for k in keys} for group, vals in kinds['all'].items() if group in ['all', 'in:rank1', 'in:ranks65_end', 'out:ranks65_end', 'out:ranks65_end/in:ranks65_end']} for label, kinds in d['groups'].items()}
        print(json.dumps({'file': path.name, 'seconds': d['seconds'], 'groups': concise}), flush=True)
    sources = [AUDIT / 'measure_persistence.py', AUDIT / 'measure_frame.py', ROOT / 'research/adamw_spectra/gn_probe.py']
    out = {'records': records, 'rank_groups': LABELS,
           'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
           'scope': 'Independent-sample lag products within each saved frame; rank-group energy association across distinct probes. No confidence intervals or eigenbasis identity assumed. Noise multipliers are sensitivity checks, not estimated corrections.'}
    (HERE / args.out).write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()

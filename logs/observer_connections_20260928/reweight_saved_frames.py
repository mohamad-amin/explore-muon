"""Compare gradient-signal and actual-update weights in independently fitted frames.

No forward/backward/optimizer call; reads trusted saved tensors on CPU.
Estimated per-coordinate masks are descriptive, not calibrated significance tests.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
from collections import defaultdict
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch

torch.set_num_threads(2)
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FRAME = ROOT / 'logs/muon_spectra/second_order_audit_20260926/frame'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, nargs='+', default=[500])
    parser.add_argument('--out', default='saved_frame_reweighting.json')
    args = parser.parse_args()
    started = time.time()
    records = []
    paths = sorted(p for step in args.steps for p in FRAME.glob(f'*step{step:06d}.pt'))
    for path in paths:
        t0 = time.time()
        d = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        meta = d['meta']
        cfg_path = Path(meta['arm']) / 'config.json'
        if not cfg_path.is_absolute(): cfg_path = ROOT / cfg_path
        cfg = json.loads(cfg_path.read_text())
        batch = cfg['batch_tokens'] / meta['T']
        n = meta['sequences']
        groups = defaultdict(lambda: defaultdict(float))
        block_ratios = []
        for name, m in d['matrices'].items():
            mean, noise = m['mean'].double(), m['noise'].double()
            signal = mean.square() - noise / n
            positive_signal = signal.clamp_min(0)
            actual = m['update'].double()
            optimizer = actual - m['decay'].double()
            t2 = n * mean.square() / noise.clamp_min(1e-300)
            masks = {
                'estimated_snr_gt1': positive_signal * batch > noise,
                'absolute_t_gt3': t2 > 9,
                'all': torch.ones_like(mean, dtype=torch.bool),
            }
            measures = {
                'count': torch.ones_like(mean),
                'signed_signal': signal,
                'positive_signal': positive_signal,
                'mean_energy': mean.square(),
                'noise_trace': noise,
                'actual_update_energy': actual.square(),
                'optimizer_update_energy': optimizer.square(),
                'actual_descent': -mean * actual,
                'optimizer_descent': -mean * optimizer,
            }
            kinds = ['all', name.split('.')[-1]]
            for mask_name, mask in masks.items():
                sums = {q: float(v[mask].sum()) for q, v in measures.items()}
                for kind in kinds:
                    for q, value in sums.items(): groups[kind][mask_name + '/' + q] += value
            if m.get('block_var_ratio_median') is not None:
                block_ratios.append(m['block_var_ratio_median'])
        summary = {}
        for kind, g in groups.items():
            row = {'totals': {q.split('/')[1]: v for q, v in g.items() if q.startswith('all/')}}
            for mask in ['estimated_snr_gt1', 'absolute_t_gt3']:
                row[mask] = {
                    q: g[mask + '/' + q] / g['all/' + q]
                    if abs(g['all/' + q]) > 1e-30 else None
                    for q in measures
                }
            summary[kind] = row
        stat = path.stat()
        out = {
            'file': str(path.relative_to(ROOT)), 'bytes': stat.st_size,
            'mtime_ns': stat.st_mtime_ns, 'meta': meta, 'batch_sequences': batch,
            'config_path': str(cfg_path.relative_to(ROOT)),
            'config_sha256': hashlib.sha256(cfg_path.read_bytes()).hexdigest(),
            'block_var_ratio_median_range': [min(block_ratios), max(block_ratios)],
            'groups': summary,
            'seconds': time.time() - t0,
        }
        records.append(out)
        print(json.dumps({'arm': path.stem, 'all': summary['all'],
                          'block_var_ratio_range': out['block_var_ratio_median_range'],
                          'seconds': out['seconds']}), flush=True)
        del d
    output = {'records': records, 'seconds': time.time() - started,
              'scope': 'Kronecker coordinate masks at each method own state, not exact GN spectral bands. Actual updates are independent of held-out gradient estimates; mask selection is not. Signed descent shares may exceed [0,1].'}
    (HERE / args.out).write_text(json.dumps(output, indent=2) + '\n')


if __name__ == '__main__':
    main()

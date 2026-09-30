"""CPU-only null calibration of the saved frame-SNR statistic; no model loads."""
from pathlib import Path
import csv
import hashlib
import json
import math
import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import numpy as np
from scipy.special import betaincc
from scipy.stats import t

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'logs/muon_spectra/second_order_audit_20260926'


def null_exact(k, budget_ratio):
    # Let Z^2~chi2_1 and V~chi2_(k-1), independent. The unclipped
    # bias-corrected estimate in sigma^2/k units is Z^2 - V/(k-1).
    # SNR at B / (K*m) = budget_ratio exceeds 1 iff t_(k-1)^2 > 1+1/ratio.
    nu = k - 1
    a = 1 + 1 / budget_ratio
    threshold = a / (nu + a)

    def truncated_signal(cut):
        return (nu + 1) / nu * (
            betaincc(1.5, nu / 2, cut) - betaincc(0.5, nu / 2, cut))

    return {
        'entry_fraction_snr_gt_1': float(2 * t.sf(math.sqrt(a), nu)),
        'signal_fraction_snr_gt_1': float(
            truncated_signal(threshold) / truncated_signal(1 / (nu + 1))),
        'mean_clipped_squared_signal_over_true_noise_variance': float(
            truncated_signal(1 / (nu + 1)) / k),
    }


def main():
    k, micro_tokens, seed, n = 64, 16384, 260928, 2_000_000
    rng = np.random.default_rng(seed)
    z2 = rng.normal(size=n) ** 2
    variance = rng.chisquare(k - 1, size=n) / (k - 1)
    estimated_signal = np.maximum(z2 - variance, 0)
    nulls = {}
    for label, ratio in [('1M', 1), ('4M', 4), ('16M', 16)]:
        high = ratio * estimated_signal / variance > 1
        exact = null_exact(k, ratio)
        measured = {
            'entry_fraction_snr_gt_1': float(high.mean()),
            'signal_fraction_snr_gt_1': float(estimated_signal[high].sum() / estimated_signal.sum()),
        }
        nulls[label] = {'exact': exact, 'monte_carlo': measured}
        for key in measured:
            assert abs(exact[key] - measured[key]) < 0.002, (label, key)

    # A counterexample to identifying adaptive-step energy from raw signal energy.
    # Each coordinate has batch noise variance 1; just one has signal mean 100.
    mu = np.r_[100.0, np.full(999, 0.1)]
    noise = np.ones_like(mu)
    high = mu**2 > noise
    adaptive_mean = mu / np.sqrt(mu**2 + noise)
    toy = {
        'description': '1000 independent coordinates; means [100, .1 x 999], batch noise variance 1',
        'raw_true_signal_energy_in_high_snr': float((mu[high]**2).sum() / (mu**2).sum()),
        'normalized_mean_direction_energy_in_high_snr': float((adaptive_mean[high]**2).sum() / (adaptive_mean**2).sum()),
        'expected_stochastic_normalized_energy_in_high_snr': float(high.mean()),
        'scope': 'Entrywise normalization before any polar map; mathematical counterexample, not real SOAP measurement',
    }
    rows, hashes = [], {}
    for file in sorted((SOURCE / 'frame_snr').glob('*.json')):
        d = json.loads(file.read_text())
        hashes[str(file.relative_to(ROOT))] = hashlib.sha256(file.read_bytes()).hexdigest()
        assert d['micro_batches'] == k and d['micro_tokens'] == micro_tokens
        for frame, batches in d['aggregate'].items():
            for label, values in batches.items():
                rows.append({
                    'state': file.stem, 'frame': frame, 'batch': label,
                    **values,
                    'fixed_frame_gaussian_null_entry_fraction': nulls[label]['exact']['entry_fraction_snr_gt_1'],
                    'fixed_frame_gaussian_null_signal_fraction': nulls[label]['exact']['signal_fraction_snr_gt_1'],
                })
    probe = SOURCE / 'frame_snr_probe.py'
    hashes[str(probe.relative_to(ROOT))] = hashlib.sha256(probe.read_bytes()).hexdigest()
    output = {
        'sample_microbatches': k, 'micro_tokens': micro_tokens,
        'seed': seed, 'monte_carlo_entries': n,
        'null_assumptions': 'Independent Gaussian microbatch noise, zero mean, fixed externally chosen basis, homogeneous variances; not an empirical null fitted to the network',
        'null': nulls, 'weighting_counterexample': toy,
        'saved_aggregates': rows, 'source_sha256': hashes,
    }
    (HERE / 'signal_noise_audit.json').write_text(json.dumps(output, indent=2) + '\n')
    with (HERE / 'signal_noise_aggregates.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps({'null': nulls, 'weighting_counterexample': toy}, indent=2))

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['1M', '4M', '16M']
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, field, title in zip(axs,
            ['entry_fraction_snr_gt_1', 'signal_fraction_snr_gt_1'],
            ['Fraction of entries above SNR = 1', 'Fraction of estimated signal energy above SNR = 1']):
        ax.plot(labels, [nulls[b]['exact'][field] for b in labels], '--o', color='black', label='Pure Gaussian noise: exact null')
        for name in ['PD1M_900', 'PD4M_183', 'SPD16M_46']:
            vals = [next(r[field] for r in rows if r['state'] == name and r['frame'] == 'raw' and r['batch'] == b) for b in labels]
            ax.plot(labels, vals, '-o', label=name + ' (raw frame)')
        ax.set(title=title, xlabel='Extrapolated batch (measurement uses 1M tokens)', ylim=(0, 1.04))
        ax.grid(alpha=.2)
    axs[0].legend(fontsize=8)
    fig.suptitle('Estimator calibration: a high weighted SNR fraction also occurs with zero signal', fontsize=12)
    fig.savefig(HERE / 'signal_noise_null.png', dpi=160)
    fig.savefig(HERE / 'signal_noise_null.pdf')


if __name__ == '__main__':
    main()

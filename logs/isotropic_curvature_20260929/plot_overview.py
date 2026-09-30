"""Compact scientific overview from the complete report tables."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import csv
import json
HERE = Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR'] = str(HERE / '.mplconfig_overview')
OUT = HERE / 'run1/report_tables'
assert json.loads((OUT / 'result.json').read_text())['status'] == 'complete'
with (OUT / 'actual_direction.csv').open() as f:
    rows = list(csv.DictReader(f))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42})
fig, axes = plt.subplots(2, 3, figsize=(11, 6.7), sharex=True, sharey='row')
for bi, block in enumerate((1, 4, 8)):
    for ri, (field, ylabel) in enumerate((('H_over_GN', 'True Hessian / Gauss–Newton'),
                                       ('actual_over_mean_left_H', 'Actual H / mean rotated H'))):
        ax = axes[ri, bi]
        for method, color in (('Muon', '#0072B2'), ('PD', '#D55E00')):
            for aggregate in ('bank0', 'bank1', 'all4'):
                selected = sorted((r for r in rows if r['method'] == method and int(r['block']) == block and r['aggregate'] == aggregate), key=lambda r: int(r['step']))
                assert len(selected) == 3
                mean = aggregate == 'all4'
                ax.plot([int(r['step']) for r in selected], [float(r[field]) for r in selected],
                        color=color, marker='o' if mean else None, lw=2 if mean else .7,
                        alpha=1 if mean else .28, label=method if mean else None)
        ax.axhline(1, color='#777777', ls=':', lw=1)
        ax.grid(alpha=.18)
        ax.set_xticks([10, 500, 1300])
        if ri == 0:
            ax.set_title(f'MLP-up, block {block}')
        else:
            ax.set_xlabel('Checkpoint update')
        if bi == 0:
            ax.set_ylabel(ylabel)
axes[0, 0].legend(frameon=False)
fig.suptitle('Curvature along saved updates evolves; orientation dependence persists', fontsize=14)
fig.text(.5, .015, 'Bold: four-context mean. Faint: two separate banks. Left rotations preserve every activation-kick norm.\nDirections change with checkpoints; these are directional measurements, not global Hessian spectra.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, .085, 1, .945))
fig.savefig(OUT / 'trajectory_overview.png', dpi=170)
fig.savefig(OUT / 'trajectory_overview.pdf')
plt.close(fig)
with (HERE / 'run1/analysis/radius_symmetry.csv').open() as f:
    symmetry = [r for r in csv.DictReader(f) if r['direction'] == 'actual' and r['aggregate'] == 'all4']
fig, axes = plt.subplots(2, 3, figsize=(11, 6.7), sharex=True, sharey='row')
for bi, block in enumerate((1, 4, 8)):
    for method, style in (('Muon', '-'), ('PD', '--')):
        for step, color in ((10, '#0072B2'), (500, '#D55E00'), (1300, '#009E73')):
            selected = sorted((r for r in symmetry if r['method'] == method and int(r['block']) == block and int(r['step']) == step), key=lambda r: float(r['abs_scale']))
            assert len(selected) == 7
            xx = [float(r['abs_scale']) for r in selected]
            even = [float(r['R_even']) / float(r['hessian_quadratic']) if float(r['hessian_quadratic']) != 0 else None for r in selected]
            odd = [float(r['R_odd']) / float(r['R_even']) if float(r['R_even']) != 0 else None for r in selected]
            for ri, yy in enumerate((even, odd)):
                axes[ri, bi].plot(xx, yy, style, color=color, marker='o', ms=3, lw=1.6,
                                 label=f'{method}, {step}')
    for ri in (0, 1):
        ax = axes[ri, bi]
        ax.axhline(1 if ri == 0 else 0, color='#777777', lw=.8, ls=':')
        ax.set_xscale('log', base=2)
        ax.set_xticks([.25, 1, 4, 16], ['¼', '1', '4', '16'])
        ax.grid(alpha=.18)
        if ri == 0:
            ax.set_title(f'MLP-up, block {block}')
        else:
            ax.set_xlabel('Absolute saved-write multiplier')
axes[0, 0].set_ylabel('Even remainder / true quadratic')
axes[1, 0].set_ylabel('Odd remainder / even remainder')
axes[0, 0].legend(frameon=False, fontsize=8, ncol=2)
fig.suptitle('Finite loss remainder: symmetric growth and directional asymmetry', fontsize=14)
fig.text(.5, .015, 'All 18 actual-write panels; four-context means. Solid: Muon. Dashed: PD. Colors: checkpoint.\nEven and odd components use both signs after subtracting the exact first-order term.', ha='center', fontsize=9)
fig.tight_layout(rect=(0, .085, 1, .945))
fig.savefig(OUT / 'finite_remainders_overview.png', dpi=170)
fig.savefig(OUT / 'finite_remainders_overview.pdf')
plt.close(fig)
print('Complete trajectory overview written')

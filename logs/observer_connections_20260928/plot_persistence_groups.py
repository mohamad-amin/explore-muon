"""Render fixed-group persistence summaries; no model loading."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
records = sum([json.loads((HERE / f).read_text())['records'] for f in ['persistence_groups.json', 'persistence_groups_temporal.json']], [])
rows = []
for r in records:
    method = 'Muon' if Path(r['meta']['arm']).name.startswith('M_') else 'PD'
    g = r['groups']['t+1']['all']
    tail = g['in:ranks65_end']; head = g['in:rank1']
    rows.append({'method': method, 'step': r['meta']['t'],
                 'head_corr': head['cosine_corrected'], 'tail_corr': tail['cosine_corrected'],
                 'tail_corr_noise125': tail['noise_sensitivity']['1.25']['cosine'],
                 'tail_snr': tail['group_trace_snr_at_training_batch'],
                 'tail_update_energy_share': tail['update_energy_share'],
                 'tail_coordinate_share': tail['coordinate_share'],
                 'tail_signal_share': tail['signal_a2'] / g['all']['signal_a2'],
                 'head_signal_share': head['signal_a2'] / g['all']['signal_a2'],
                 'tail_cross': tail['cross'], 'head_cross': head['cross']})
rows.sort(key=lambda r: (r['method'],r['step']))
(HERE / 'persistence_compact.json').write_text(json.dumps(rows,indent=2)+'\n')
fig, axs = plt.subplots(1,3,figsize=(12,4),constrained_layout=True)
for method,color in [('Muon','tab:blue'),('PD','tab:red')]:
    rr = [r for r in rows if r['method']==method];x=[r['step'] for r in rr]
    axs[0].plot(x,[r['head_corr'] for r in rr],'--o',color=color,label=method+': input rank 1')
    axs[0].plot(x,[r['tail_corr'] for r in rr],'-o',color=color,label=method+': input ranks 65+')
    axs[0].fill_between(x,[r['tail_corr'] for r in rr],[r['tail_corr_noise125'] for r in rr],color=color,alpha=.15)
    axs[1].plot(x,[100*r['tail_update_energy_share'] for r in rr],'-o',color=color,label=method)
    axs[2].plot(x,[r['tail_snr'] for r in rr],'-o',color=color,label=method)
axs[0].axhline(0,color='gray',lw=.8)
axs[0].set(title='Next-step gradient alignment',ylabel='Noise-corrected cosine',ylim=(-.55,.5))
axs[0].legend(fontsize=8,loc='lower right')
axs[1].axhline(90.625,color='gray',ls=':',label='Coordinate share (90.6%)')
axs[1].set(title='Update energy in input ranks 65+',ylabel='Percent of actual body displacement',ylim=(80,100))
axs[1].legend(fontsize=8)
axs[2].axhline(1,color='gray',ls=':')
axs[2].set(title='Input-tail signal versus sampling noise',ylabel='Group trace SNR at training batch',ylim=(0,2.6))
for ax in axs:
    ax.set(xlabel='Training step',xticks=[200,500,900]); ax.grid(alpha=.15)
fig.suptitle('Independent-set gradients: dominant input modes and weaker input modes have different dynamics',fontsize=12)
fig.savefig(HERE/'persistence_groups.png',dpi=160)
fig.savefig(HERE/'persistence_groups.pdf')
print(json.dumps(rows,indent=2))

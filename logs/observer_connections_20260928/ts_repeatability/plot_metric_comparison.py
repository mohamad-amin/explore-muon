"""Show the metric and amplitude qualifications from saved results only."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;RUN=HERE/'run2'
p=json.loads((RUN/'analysis.json').read_text());f=json.loads((RUN/'functional/analysis.json').read_text())
q=np.array(f['gram_pooled']);ix={n:i for i,n in enumerate(f['labels'])}
r=q/np.sqrt(np.outer(np.diag(q),np.diag(q)))
pairs=['A0__B0','A0__A1','AB__CD'];labels=['Disjoint 8','New labels, same inputs','Disjoint 16']
unit={}
for pair in pairs:
    a,b=pair.split('__');i,j,k=ix[a],ix[b],ix['PD']
    unit[pair]=(2-2*r[i,j])/((2-2*r[i,k])+(2-2*r[j,k]))
source={row['pair']:row for row in p['rows']}
fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
x=np.arange(3)
axs[0].plot(x,[source[k]['direction_cosine'] for k in pairs],'o-',label='Parameter metric')
axs[0].plot(x,[f['pooled'][k]['gn_cosine'] for k in pairs],'o-',label='Predictive-GN metric')
axs[0].set(title='The same directions, two metrics',ylabel='Cosine',ylim=(.65,1.01))
axs[1].plot(x,[source[k]['noise_to_added_geometry'] for k in pairs],'o-',label='Parameter metric')
axs[1].plot(x,[f['pooled'][k]['noise_to_added_TS_geometry'] for k in pairs],'o-',label='GN, actual amplitudes')
axs[1].plot(x,[unit[k] for k in pairs],'^--',color='gray',label='GN, unit amplitudes (post hoc)')
axs[1].set(title='Amplitude qualifies the small GN ratio',ylabel='Variation / TS–PD difference energy',yscale='log')
for ax in axs:
    ax.set_xticks(x,labels,fontsize=8);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Conditional TS sampling sensitivity: fixed-state evidence, not a training-rate claim')
fig.savefig(RUN/'metric_comparison.png',dpi=160);fig.savefig(RUN/'metric_comparison.pdf')
(RUN/'posthoc_amplitude_check.json').write_text(json.dumps({'unit_functional_norm_ratios':unit,'scope':'Post-hoc interpretation only; no new normalization was applied in a training or finite-loss test.'},indent=2)+'\n')

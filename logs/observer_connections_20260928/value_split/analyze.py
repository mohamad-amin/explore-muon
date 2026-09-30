"""Summarize the predeclared functional split without fitting a new step."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'result.json').read_text())
assert d['status']=='complete'
coeff={'constant':np.array([1.,0.]),'centered':np.array([0.,1.]),'combined':np.array([1.,1.]),'half_combined':np.array([.5,.5])}


def aggregate(rows):
    slopes=np.array([r['slopes'] for r in rows]);q=np.array([r['gn'] for r in rows])
    losses={k:np.array([r['losses'][k] for r in rows]) for k in ['base',*coeff]}
    delta={k:v-losses['base'] for k,v in losses.items() if k!='base'}
    delta['constant_given_centered']=losses['combined']-losses['centered']
    delta['centered_given_constant']=losses['combined']-losses['constant']
    delta['interaction']=losses['combined']-losses['constant']-losses['centered']+losses['base']
    mean_q=q.mean(0)
    return {'n':len(rows),'slope_mean':slopes.mean(0).tolist(),'slope_sd':slopes.std(0,ddof=1).tolist(),
            'gn_mean':mean_q.tolist(),'gn_cross_correlation':float(mean_q[0,1]/np.sqrt(mean_q[0,0]*mean_q[1,1])),
            'cross_share_of_combined_gn':float(2*mean_q[0,1]/mean_q.sum()),
            'base_loss_mean':float(losses['base'].mean()),
            'effects':{k:{'mean':float(v.mean()),'sd':float(v.std(ddof=1)),'negative_sequences':int((v<0).sum()),'values':v.tolist()} for k,v in delta.items()},
            'gn_prediction':{k:float(slopes.mean(0)@c+.5*c@mean_q@c) for k,c in coeff.items()},
            'first_order':{k:float(slopes.mean(0)@c) for k,c in coeff.items()}}


out={'scope':'Own-state actual8V displacement, fixed independent mean, two banks4 sequences each; no rate, causal or formal-significance claim. All actual component scales retained. GN is predictive GN, not the full Hessian or finite interaction.','methods':{}}
for method,r in d['methods'].items():
    out['methods'][method]={'banks':[aggregate(b) for b in r['banks']], 'pooled':aggregate(sum(r['banks'],[]))}
(HERE/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True,sharey=True)
for ax,(method,r) in zip(axs,out['methods'].items()):
    names=['constant','centered','combined','constant_given_centered'];positions=np.arange(len(names))
    for j,bank in enumerate(r['banks']):
        ax.plot(positions+(-.06 if j==0 else .06),[1000*bank['effects'][k]['mean'] for k in names],'o',label=f'Bank {j+1} (4 sequences)',alpha=.9)
    ax.axhline(0,color='gray',lw=.8)
    ax.set(title=method,xticks=positions,xticklabels=['Constant\nonly','Centered\nonly','Combined','Constant given\ncentered'],ylabel='True loss change (millinats/token)')
    ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Actual value update at step500: true loss by functional component',fontsize=12)
fig.savefig(HERE/'effects.png',dpi=160);fig.savefig(HERE/'effects.pdf')
for method,r in out['methods'].items():
    a=r['pooled'];print(method,json.dumps({'slopes':a['slope_mean'],'gn':a['gn_mean'],'cross_share':a['cross_share_of_combined_gn'],
               'effects':{k:{'mean':v['mean'],'negative_sequences':v['negative_sequences']} for k,v in a['effects'].items()},'prediction':a['gn_prediction']}))

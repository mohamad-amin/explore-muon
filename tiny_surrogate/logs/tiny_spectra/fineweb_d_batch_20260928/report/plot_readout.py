"""Presentation of the completed frozen batch readout; no new model inference."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=Path(__file__).resolve().parent
source=out/'results.json'
r=json.loads(source.read_text())
batches=[65536,262144,1048576]
seeds=[20261001,20261002]
colors=['#377eb8','#ff7f00','#4daf4a']
fig,axes=plt.subplots(2,2,figsize=(12,8.5))
for lr,color,marker in [(0.01,'#5e3c99','o'),(0.02,'#e66101','s')]:
    values=[]
    for seed in seeds:
        vals=[next(x['spd_minus_muon'] for x in r['endpoint_gaps'] if x['seed']==seed and x['muon_lr']==lr and x['batch_tokens']==b) for b in batches]
        values.append(vals)
        axes[0,0].plot(batches,vals,':',color=color,alpha=.5)
    axes[0,0].plot(batches,np.mean(values,axis=0),marker=marker,color=color,label=f'Muon LR {lr:g}')
    for seed in seeds:
        item=next(x for x in r['comparisons'] if x['seed']==seed and x['muon_lr']==lr)
        values=np.array([math.exp(item['mean_log_ratios'][str(b)]['value']) for b in batches])
        lower=np.array([math.exp(item['mean_log_ratios'][str(b)]['lower']) for b in batches])
        upper=np.array([math.exp(item['mean_log_ratios'][str(b)]['upper']) for b in batches])
        shift=(.94 if lr==.01 else 1.06)*(.97 if seed==seeds[0] else 1.03)
        axes[0,1].errorbar(np.array(batches)*shift,values,yerr=[values-lower,upper-values],marker=marker,
            color=color,alpha=.8 if seed==seeds[0] else .45,capsize=3,linewidth=1,
            label=f'LR {lr:g}, seed {seed%100}' )
axes[0,0].axhline(0,color='black',lw=.7)
axes[0,0].set(title='Endpoint gap grows with batch',ylabel='SoapMuon+PD minus Muon NLL\nNegative favors geometry')
axes[0,0].legend(fontsize=8)
axes[0,1].axhline(1,color='black',lw=.7)
axes[0,1].set(title='Common-loss advantage: growth remains unresolved',ylabel='Geometric mean of Muon / SoapMuon+PD steps\nWhiskers: evaluation-resolution bounds')
axes[0,1].legend(fontsize=7,ncol=2)
for ax in axes[0]:
    ax.set_xscale('log',base=2)
    ax.set_xticks(batches,['65,536','262,144','1,048,576'])
    ax.set_xlabel('Batch size in targets')
for ax,lr in zip(axes[1],[.01,.02]):
    for batch,color in zip(batches,colors):
        for seed in seeds:
            values=[x for x in r['all_ratios'] if x['batch_tokens']==batch and x['seed']==seed and x['muon_lr']==lr and x['value'] is not None]
            ax.plot([x['threshold'] for x in values],[x['value'] for x in values],
                color=color,linestyle='-' if seed==seeds[0] else ':',alpha=.8,
                label=f'B={batch:,}' if seed==seeds[0] else None)
    ax.axvspan(min(r['common_thresholds']),max(r['common_thresholds']),color='grey',alpha=.12)
    ax.axhline(1,color='black',lw=.7)
    ax.set(xlim=(4,9.5),xlabel='Full-development target loss',ylabel='Muon / SoapMuon+PD steps',
           title=f'All defined thresholds, Muon LR {lr:g}')
    ax.legend(fontsize=8)
for ax in axes.flat:
    ax.grid(alpha=.15)
    ax.spines[['top','right']].set_visible(False)
fig.suptitle('FineWeb miniature: fixed-recipe batch comparison — development only',fontsize=14)
fig.text(.5,.012,'Two seeds; both Muon controls kept separate. Shading marks the prespecified common-range rule. No independent test data scored.',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.96))
for suffix in ['png','pdf']:
    fig.savefig(out/f'batch_comparison.{suffix}',dpi=170)
(out/'PLOTS.json').write_text(json.dumps({name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['results.json','plot_readout.py','batch_comparison.png','batch_comparison.pdf']},indent=2)+'\n')

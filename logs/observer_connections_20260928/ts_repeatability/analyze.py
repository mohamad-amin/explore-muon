"""Retrospective summaries with the prospective stricter pooling comparator."""
from pathlib import Path
import argparse
import csv
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--run',default='run2');args=parser.parse_args()
RUN=HERE/args.run
d=json.loads((RUN/'result.json').read_text());assert d['status']=='complete'
rows=[]
for pair,p in d['direction_pairs'].items():
    all_=p['scopes']['all'];cs=np.array([v['cosine'] for v in p['matrices'].values()])
    ratios=np.array([v['noise_to_added_geometry'] for v in p['matrices'].values()])
    row={'pair':pair,'type':p['comparison_type'],'direction_cosine':all_['cosine'],
         'relative_direction_distance':all_['relative_distance'],
         'noise_to_added_geometry':all_['noise_to_added_geometry'],
         'matrix_cosine_min':float(cs.min()),'matrix_cosine_median':float(np.median(cs)),
         'matrix_cosine_max':float(cs.max()),'matrix_noise_to_geometry_median':float(np.median(ratios))}
    for tag in ['trace_normalized_B','trace_normalized_L']:
        vals=p.get(tag)
        row[tag+'_cosine_median']=float(np.median([v['cosine'] for v in vals.values()])) if vals else None
        row[tag+'_relative_distance_median']=float(np.median([v['relative_distance'] for v in vals.values()])) if vals else None
    rows.append(row)
e8=[r['direction_cosine'] for r in rows if r['type']=='independent8']
e16=next(r['direction_cosine'] for r in rows if r['type']=='independent16')
flag='stable' if min(e8)>=.99 and e16>=.995 else ('material_sample_sensitivity' if min(e8)<=.90 and e16>np.mean(e8) else 'mixed')
summary={'strict_descriptive_flag':flag,'executed_provisional_flag':d['descriptive_flag'],
         'independent8_cosines':e8,'independent8_mean':float(np.mean(e8)),
         'independent16_cosine':e16,'pooling_exceeds_mean_independent8':bool(e16>np.mean(e8)),
         'rows':rows,'vs_PD':{k:v['scopes']['all'] for k,v in d['versus_PD'].items()},
         'scope':'Conditional fixed-M46 and rank0-C proxy at TS16M W46, CPU FP32; fresh B and EMA-refresh proxy are not online variance or a training replay. No loss scores or rate claims.'}
(RUN/'analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
with (RUN/'comparisons.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
names=['independent8','same_sequences_new_labels','independent16','ema_refresh_proxy','numerical_control']
labels=['Disjoint8\nsequences','Same sequences\nnew labels','Disjoint16\nsequences','One-refresh\nEMA proxy','CPU FP32 vs\nBF16 NS']
fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for i,name in enumerate(names):
    rr=[r for r in rows if r['type']==name]
    dx=np.linspace(-.12,.12,len(rr)) if len(rr)>1 else np.array([0.])
    axs[0].scatter(i+dx,[r['direction_cosine'] for r in rr],s=30)
    axs[1].scatter(i+dx,[r['noise_to_added_geometry'] for r in rr],s=30)
axs[0].axhline(.99,ls=':',color='gray',lw=.8);axs[0].set(ylabel='Whole-body direction cosine',title='Conditional direction repeatability')
axs[1].axhline(1,ls=':',color='gray',lw=.8);axs[1].set(ylabel='Squared pair distance / added-geometry energy',title='Variation relative to TS versus PD')
for ax in axs:
    ax.set_xticks(range(len(names)),labels,fontsize=8);ax.grid(axis='y',alpha=.2)
fig.suptitle('Fixed-state TS factor sampling: points are comparisons, not independent replicates')
fig.savefig(RUN/'repeatability.png',dpi=160);fig.savefig(RUN/'repeatability.pdf')
print(json.dumps({k:v for k,v in summary.items() if k not in ['rows','vs_PD']},indent=2))
for r in rows:print(r['pair'],r['type'],round(r['direction_cosine'],6),round(r['noise_to_added_geometry'],5),round(r['matrix_cosine_min'],5))

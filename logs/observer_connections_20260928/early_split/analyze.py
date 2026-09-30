"""Declared finite-effect predictions; no fresh fitting or sample selection."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig'))
import csv
import json
import numpy as np
OUT=HERE/'run1'


def stats(x):
    return {'mean':float(x.mean()),'sd_across_sequences':float(x.std(ddof=1)),
            'negative_sequences':int(np.sum(x<0)),'positive_sequences':int(np.sum(x>0)),'n':len(x),
            'min':float(x.min()),'max':float(x.max())}


def main():
    report=json.loads((OUT/'result.json').read_text());assert report['status']=='complete'
    data=np.load(OUT/'scalars.npz');p=data['parts'].mean(-1);L=data['corner_nll'].mean(-1);K=data['corner_forward_kl'].mean(-1)
    assert p.shape==(4,16,6) and np.isfinite(p).all()
    groups={'old':slice(0,8),'old_bank0':slice(0,4),'old_bank1':slice(4,8),
            'fresh':slice(8,16),'fresh_bank0':slice(8,12),'fresh_bank1':slice(12,16)}
    rows={};arrays={}
    for j,(label,step) in enumerate(report['states']):
        key=f'{label}_{step}';v={name:p[j,:,i] for i,name in enumerate(report['parts'])}
        v.update(KL_full=K[j,:,2],mixed_KL_correction=K[j,:,2]-p[j,:,4],base_nll=L[j,:,0])
        for n,k in [('body',1),('aux',2),('full',3)]:v['delta_nll_'+n]=L[j,:,k]-L[j,:,0]
        arrays[key]=v
        rows[key]={'groups':{g:{n:stats(a[ix]) for n,a in v.items()} for g,ix in groups.items()},
                   'per_sequence':{n:a.tolist() for n,a in v.items()}}
    contrast=arrays['h09_9']['mixed_loss']-arrays['q09_9']['mixed_loss']
    stage=arrays['h09_46']['mixed_loss']-arrays['h09_9']['mixed_loss']
    contrasts={'early_half_minus_quarter':{g:stats(contrast[ix]) for g,ix in groups.items()},
               'half_late_minus_early':{g:stats(stage[ix]) for g,ix in groups.items()}}
    m=rows['h09_9']['groups'];c=contrasts['early_half_minus_quarter'];s=contrasts['half_late_minus_early']
    predictions={
        'P1_early_half_material_help':m['fresh']['mixed_loss']['mean']<=-.01 and all(m[g]['mixed_loss']['mean']<0 for g in ['fresh_bank0','fresh_bank1']),
        'P2_half_more_helpful_than_quarter':c['fresh']['mean']<=-.01 and all(c[g]['mean']<0 for g in ['fresh_bank0','fresh_bank1']),
        'P3_half_less_helpful_later':all(s[g]['mean']>0 for g in ['fresh_bank0','fresh_bank1'])}
    result={'scope':'Fixed finite-logit path and fresh-input consistency at selected own states; no seed significance, training mediation or realizable additive-parameter update claim.',
            'states':rows,'contrasts':contrasts,'predictions':predictions,'qualification':report['qualification']}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'summary.csv').open('w') as out:
        w=csv.writer(out);w.writerow(['state','group','total_interaction','additive_overlap','finite_mixed_loss','base_linear_J','mixed_KL_correction'])
        for key,d in rows.items():
            for group,z in d['groups'].items():w.writerow([key,group,*[z[n]['mean'] for n in ['interaction','overlap','mixed_loss','J_direct','mixed_KL_correction']]])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(rows);x=np.arange(4);fig,ax=plt.subplots(figsize=(9,4.5))
    for offset,name,color,title in [(-.24,'interaction','#777777','Total interaction'),(0,'overlap','#0072B2','Additive overlap'),(.24,'mixed_loss','#D55E00','Finite mixed loss')]:
        vals=[rows[k]['groups']['fresh'][name]['mean'] for k in labels]
        ax.bar(x+offset,vals,.23,label=title,color=color,alpha=.8)
    ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(x,['Quarter, step 9','Half, step 9','Quarter, step 46','Half, step 46'])
    ax.set_ylabel('NLL interaction on 8 fresh sequences');ax.set_title('Exact finite-logit split; beta .9, actual saved displacements')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);fig.tight_layout()
    fig.savefig(OUT/'finite_split.png',dpi=170);fig.savefig(OUT/'finite_split.pdf');plt.close(fig)
    print(json.dumps({'predictions':predictions,'states':{k:{g:{n:d['groups'][g][n]['mean'] for n in ['interaction','overlap','mixed_loss','J_direct','mixed_KL_correction']} for g in ['old','fresh','fresh_bank0','fresh_bank1']} for k,d in rows.items()},'contrasts':contrasts},indent=2))


if __name__=='__main__':main()

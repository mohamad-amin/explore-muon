"""Fixed primary contrasts and interpolation controls; no fresh fitting."""
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE/'run1'


def summarize(values,bins):
    values=np.asarray(values,dtype=np.float64);n=values.shape[0];nt=values.size
    rare=(bins==1)|(bins==2);common=bins>=4
    sums_r=(values*rare).sum(1);nr=rare.sum(1)
    sums_c=(values*common).sum(1);nc=common.sum(1)
    r=float(sums_r.sum()/nr.sum());c=float(sums_c.sum()/nc.sum());contrast=r-c
    influence=(sums_r-r*nr)/nr.sum()-(sums_c-c*nc)/nc.sum()
    se=float(np.sqrt(n/(n-1)*np.sum(influence**2)))
    rows=[]
    for j in range(6):
        m=bins==j;count=int(m.sum());total=float(values[m].sum())
        rows.append({'bin':j,'tokens':count,'mean':total/count if count else None,'contribution_to_global':total/nt})
    global_mean=float(values.mean());assert abs(sum(x['contribution_to_global'] for x in rows)-global_mean)<1e-12
    perseq=[]
    for i in range(n):perseq.append({'sequence_index':i,'rare_tokens':int(nr[i]),'common_tokens':int(nc[i]),
        'rare_loss_sum':float(sums_r[i]),'common_loss_sum':float(sums_c[i]),
        'rare_common_contrast':float(sums_r[i]/nr[i]-sums_c[i]/nc[i]) if nr[i] and nc[i] else None})
    return {'sequences':n,'tokens':nt,'mean':global_mean,'bins':rows,
        'primary_rare_mean':r,'primary_common_mean':c,'primary_contrast':contrast,
        'nominal_sequence_cluster_se':se,'per_sequence':perseq,
        'scope':'Nominal SE treats sequences as independent; adjacent sequences may share documents. No seed/population significance claim.'}


def main():
    report=json.loads((OUT/'result.json').read_text());assert report['status']=='complete'
    nll=np.load(OUT/'per_token_nll.npz')['nll'];data=np.load(OUT/'tokens_frequency.npz');bins=data['bins']
    assert nll.shape==(10,32,512) and np.isfinite(nll).all() and bins.shape==(32,512)
    states={tuple(x):nll[i].astype(np.float64) for i,x in enumerate(report['states'])}
    groups={'all':slice(None),'bank0':slice(0,16),'bank1':slice(16,32)}
    raw={f'{m}{s}':{bank:summarize(v[ix],bins[ix]) for bank,ix in groups.items()} for (m,s),v in states.items()}
    residuals={};arrays={};raw_effects={}
    for cal in report['calibrations']:
        target=states[tuple(cal['target'])];left=states[tuple(cal['left'])];right=states[tuple(cal['right'])]
        reference=(1-cal['weight'])*left+cal['weight']*right;res=target-reference
        arrays[cal['label']]=res
        residuals[cal['label']]={'calibration':cal,'scores':{bank:summarize(res[ix],bins[ix]) for bank,ix in groups.items()}}
        if cal['role']=='method':
            baseline=states[('M',cal['target'][1])]
            raw_effects[cal['label']]={bank:summarize((target-baseline)[ix],bins[ix]) for bank,ix in groups.items()}
    direct=states[('S',500)]-states[('PD',500)]
    direct_scores={bank:summarize(direct[ix],bins[ix]) for bank,ix in groups.items()}
    result={'scope':'Fixed fresh-panel prediction profiles; old-bank progress weights, no refitting. Muon secants are references, not interpolated models.32sequences are descriptive, not training replications.',
            'raw_profiles':raw,'same_step_effects':raw_effects,'phase_residuals':residuals,'direct_S500_minus_PD500':direct_scores}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (OUT/'primary_contrasts.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['label','bank','role','global_residual','rare_common_contrast','nominal_sequence_cluster_se'])
        for label,d in residuals.items():
            for bank,s in d['scores'].items():w.writerow([label,bank,d['calibration']['role'],s['mean'],s['primary_contrast'],s['nominal_sequence_cluster_se']])
        for bank,s in direct_scores.items():w.writerow(['S500-PD500',bank,'direct nearly matched old-bank mean',s['mean'],s['primary_contrast'],s['nominal_sequence_cluster_se']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=list(residuals);x=np.arange(len(labels));colors=['#0072B2' if residuals[k]['calibration']['role']=='method' else '#777777' for k in labels]
    means=[residuals[k]['scores']['all']['primary_contrast'] for k in labels]
    errors=[residuals[k]['scores']['all']['nominal_sequence_cluster_se'] for k in labels]
    fig,ax=plt.subplots(figsize=(10,4.3))
    ax.bar(x,means,color=colors,alpha=.7)
    ax.errorbar(x,means,yerr=errors,fmt='none',ecolor='black',capsize=3,label='Nominal sequence SE')
    for bank,marker,offset in [('bank0','o',-.12),('bank1','s',.12)]:
        ax.scatter(x+offset,[residuals[k]['scores'][bank]['primary_contrast'] for k in labels],marker=marker,s=25,label=bank)
    ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(x,labels,rotation=25,ha='right')
    ax.set_ylabel('Mid-rare minus common NLL\n(after progress correction)')
    ax.set_title('Fixed32-sequence panel; gray bars are own-Muon interpolation controls')
    ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);fig.tight_layout()
    fig.savefig(OUT/'phase_contrasts.png',dpi=170);fig.savefig(OUT/'phase_contrasts.pdf');plt.close(fig)
    compact={k:{bank:{field:s[field] for field in ['mean','primary_contrast','nominal_sequence_cluster_se']} for bank,s in d['scores'].items()} for k,d in residuals.items()}
    print(json.dumps({'phase_residuals':compact,'direct_S500_PD500':{bank:{field:s[field] for field in ['mean','primary_contrast','nominal_sequence_cluster_se']} for bank,s in direct_scores.items()}},indent=2))


if __name__=='__main__':main()

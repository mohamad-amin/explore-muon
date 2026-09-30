"""Fixed own-state predictive-amplitude contrasts; no post-hoc scale fitting."""
import csv
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE/'run1'


def describe(v):
    return {'mean':float(v.mean()),'sd_across_sequences':float(v.std(ddof=1)),
            'min':float(v.min()),'max':float(v.max()),'positive_sequences':int(np.sum(v>0))}


def main():
    meta=json.loads((OUT/'result.json').read_text());assert meta['status']=='complete'
    data=np.load(OUT/'scalar_tokens.npz');L=data['loss64'];F=data['kl_forward'];R=data['kl_reverse'];E=data['base_entropy']
    assert L.shape==(8,8,4,512) and F.shape==R.shape==(8,8,3,512)
    assert all(np.isfinite(x).all() for x in [L,F,R,E])
    l=L.mean(-1);f=F.mean(-1);r=R.mean(-1);sym=(f+r)/2
    states={};raw={};groups={'all':slice(None),'bank0':slice(0,4),'bank1':slice(4,8)}
    for j,(label,step) in enumerate(meta['states']):
        vals={'base_nll':l[j,:,0],'base_entropy':E[j].mean(-1)}
        for n,k in [('body',1),('aux',2),('full',3)]:
            vals['delta_nll_'+n]=l[j,:,k]-l[j,:,0]
            vals['forward_kl_'+n]=f[j,:,k-1];vals['reverse_kl_'+n]=r[j,:,k-1];vals['symmetric_kl_'+n]=sym[j,:,k-1]
            vals['finite_label_linear_'+n]=vals['delta_nll_'+n]-vals['forward_kl_'+n]
        vals['interaction_nll']=l[j,:,3]-l[j,:,1]-l[j,:,2]+l[j,:,0]
        key=f'{label}_{step}';raw[key]=vals
        states[key]={'norms':meta['step_norms'][key],'banks':{bank:{k:describe(v[ix]) for k,v in vals.items()} for bank,ix in groups.items()},
                     'per_sequence':{k:v.tolist() for k,v in vals.items()}}
    ratios=[]
    pairs=[]
    for step in [9,46]:
        for beta in ['09','08']:pairs.append(('alpha_half_over_quarter',f'h{beta}_{step}',f'q{beta}_{step}',step,beta))
        for alpha in ['q','h']:pairs.append(('beta08_over09',f'{alpha}08_{step}',f'{alpha}09_{step}',step,alpha))
    for family,numerator,denominator,step,setting in pairs:
        bankrows={}
        for bank,ix in groups.items():
            bankrows[bank]={}
            for metric in ['forward_kl','reverse_kl','symmetric_kl']:
                for corner in ['body','aux','full']:
                    key=metric+'_'+corner;a=raw[numerator][key][ix];b=raw[denominator][key][ix]
                    assert float(b.mean())>0
                    bankrows[bank][key]={'ratio_of_means':float(a.mean()/b.mean()),'numerator_smaller_sequences':int(np.sum(a<b)),
                                         'sequences':len(a)}
        ratios.append({'family':family,'step':step,'setting':setting,'numerator':numerator,'denominator':denominator,'banks':bankrows,
                       'body_parameter_norm_ratio':states[numerator]['norms']['body']/states[denominator]['norms']['body']})
    result={'scope':'Own-state actual finite steps on shared8inputs, not a common-state mediation intervention or rate test. KL bases differ across states. Finite label-linear terms are identities ΔCE−KL, not parameter gradients.',
            'states':states,'ratios':ratios,'validation':meta['validation'],'training_windows':meta['training_windows']}
    (OUT/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    with (OUT/'ratios.csv').open('w') as out:
        w=csv.writer(out);w.writerow(['family','step','setting','bank','corner','forward_KL_ratio','reverse_KL_ratio','symmetric_KL_ratio','body_parameter_norm_ratio'])
        for row in ratios:
            for bank,d in row['banks'].items():
                for corner in ['body','aux','full']:w.writerow([row['family'],row['step'],row['setting'],bank,corner,*[d[k+'_'+corner]['ratio_of_means'] for k in ['forward_kl','reverse_kl','symmetric_kl']],row['body_parameter_norm_ratio']])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    selected=[r for r in ratios if r['family']=='alpha_half_over_quarter'];x=np.arange(4)
    fig,axes=plt.subplots(1,3,figsize=(11,3.7),sharey=True)
    for ax,corner in zip(axes,['body','aux','full']):
        key='forward_kl_'+corner
        ax.plot(x,[r['banks']['all'][key]['ratio_of_means'] for r in selected],color='#0072B2',marker='o',label='All8inputs')
        for bank,marker in [('bank0','s'),('bank1','^')]:
            ax.scatter(x,[r['banks'][bank][key]['ratio_of_means'] for r in selected],marker=marker,s=25,alpha=.8,label=bank)
        ax.axhline(1,color='black',linewidth=.8);ax.set_yscale('log');ax.set_title(corner.capitalize()+' update')
        ax.set_xticks(x,[f"s{r['step']} β.{r['setting'][-1]}" for r in selected],rotation=30,ha='right');ax.grid(alpha=.15)
    axes[0].set_ylabel('Predictive KL ratio: alpha½ / alpha¼');axes[-1].legend(frameon=False,fontsize=8)
    fig.suptitle('Actual steps at their own states; equal nominal body norms')
    fig.tight_layout();fig.savefig(OUT/'alpha_kl_ratios.png',dpi=170);fig.savefig(OUT/'alpha_kl_ratios.pdf');plt.close(fig)
    compact={k:{'norms':d['norms'],'all':{f:v['mean'] for f,v in d['banks']['all'].items()}} for k,d in states.items()}
    print(json.dumps({'validation':meta['validation'],'states':compact,'alpha_ratios':[{k:v for k,v in r.items() if k!='banks'}|
          {'forward_ratios':{bank:{c:d['forward_kl_'+c]['ratio_of_means'] for c in ['body','aux','full']} for bank,d in r['banks'].items()}} for r in selected]},indent=2))


if __name__=='__main__':main()

"""Summarize a fixed factorial, retaining mixed contrasts and input roles."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent;RUN=HERE/'run1'
d=json.loads((RUN/'result.json').read_text());assert d['status']=='complete'


def aggregate(rows):
    losses={k:float(np.mean([r['loss64'][k] for r in rows])) for k in rows[0]['loss64']}
    mob={k:float(np.mean([r['mobius'][k] for r in rows])) for k in rows[0]['mobius']}
    planes={}
    for k in rows[0]['planes']:
        vals={name:np.array([r['planes'][k][name] for r in rows]) for name in rows[0]['planes'][k]}
        planes[k]={'mean':{n:float(v.mean()) for n,v in vals.items()},
                   'sd':{n:float(v.std(ddof=1)) for n,v in vals.items()},
                   'positive_counts':{n:int((v>0).sum()) for n,v in vals.items()}}
    q=np.mean([np.array(r['finite_logit_gram']) for r in rows],axis=0)
    return {'n':len(rows),'loss64':losses,'mobius':mob,'planes':planes,'finite_logit_gram':q.tolist(),
            'finite_response_cosines':(q/np.sqrt(np.outer(np.diag(q),np.diag(q)))).tolist(),
            'predictive_kl':{k:float(np.mean([r['predictive_kl_from_base'][k] for r in rows])) for k in rows[0]['predictive_kl_from_base']}}


out={'banks':[{**{k:v for k,v in b.items() if k!='rows'},'summary':aggregate(b['rows'])} for b in d['banks']],
     'discovery':aggregate(sum([b['rows'] for b in d['banks'][:2]],[])),
     'fresh':aggregate(sum([b['rows'] for b in d['banks'][2:]],[])),
     'scope':'Old16 are reproduction/discovery, fresh8 consistency at selected state. Finite-logit overlap is label independent; mixed-logit loss is not optimizer-state transport. No rate/causal/independent-seed claim.'}
out['max_old_loss32_difference']=max(abs(v) for b in d['banks'][:2] for r in b['rows'] for v in r['old_loss32_differences'].values())
(RUN/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
fig,axs=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
terms=['BH','BE','BN','BHE','BHN','BEN','BHEN'];x=np.arange(len(terms))
for i,role in enumerate(['discovery','fresh']):
    axs[0].plot(x+(i-.5)*.12,[1000*out[role]['mobius'][k] for k in terms],'o',label=role)
axs[0].set_xticks(x,terms);axs[0].set(title='All Body-containing mixed contrasts',ylabel='Finite loss contrast (millinats/token)')
planes=['B:H|base','B:E|base','B:N|base','B:HEN|base'];x=np.arange(len(planes));w=.25
for i,term in enumerate(['interaction','overlap','mixed_logit_loss']):
    axs[1].bar(x+(i-1)*w,[1000*out['fresh']['planes'][p]['mean'][term] for p in planes],width=w,label=term)
axs[1].set_xticks(x,['Body/head','Body/embeddings','Body/norms','Body/all aux'],rotation=12)
axs[1].set(title='Exact split on fresh inputs',ylabel='Loss effect (millinats/token)')
for ax in axs:
    ax.axhline(0,color='gray',lw=.8);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=8)
fig.suptitle('One saved training step: attribution without fitting new step sizes')
fig.savefig(RUN/'attribution.png',dpi=160);fig.savefig(RUN/'attribution.pdf')
for role in ['discovery','fresh']:
    r=out[role];print(role,'mixed', {k:round(r['mobius'][k],8) for k in terms})
    for p in planes:print(p,r['planes'][p]['mean'])
    print('singletonKL',r['predictive_kl'],'responseGram',r['finite_logit_gram'])

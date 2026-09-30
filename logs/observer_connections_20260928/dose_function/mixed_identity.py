"""Exact scalar consequence of recorded CE/KL; no model or fresh endpoint."""
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;OUT=HERE/'run1'


def summary(a):return {'mean':float(a.mean()),'sd':float(a.std(ddof=1)),'positive':int(np.sum(a>0)),'negative':int(np.sum(a<0)),'n':len(a)}


def main():
    r=json.loads((OUT/'result.json').read_text());assert r['status']=='complete'
    data=np.load(OUT/'scalar_tokens.npz');L=data['loss64'];K=data['kl_forward']
    interaction=L[:,:,3]-L[:,:,1]-L[:,:,2]+L[:,:,0]
    kl_interaction=K[:,:,2]-K[:,:,0]-K[:,:,1]
    mixed=interaction-kl_interaction
    rows={}
    for i,(label,step) in enumerate(r['states']):
        arrays={'loss_interaction':interaction[i].mean(-1),'predictive_KL_interaction':kl_interaction[i].mean(-1),
                'base_residual_mixed_logit':mixed[i].mean(-1)}
        rows[f'{label}_{step}']={'all':{k:summary(a) for k,a in arrays.items()},
            'banks':{b:{k:summary(a[ix]) for k,a in arrays.items()} for b,ix in [('bank0',slice(0,4)),('bank1',slice(4,8))]},
            'per_sequence':{k:a.tolist() for k,a in arrays.items()}}
    result={'scope':'Post-hoc exact CE/KL identity. Base-residual mixed-logit term, not CE(full)−CE(additive logits), residual-Hessian measurement, overlap fraction, subgroup attribution or training mediator.',
            'formula':'J=I_CE−[KL(p0||pF)−KL(p0||pB)−KL(p0||pA)] = mean[(p0−e_y)·(zF−zB−zA+z0)]', 'states':rows}
    (OUT/'mixed_identity.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v['all'] for k,v in rows.items()},indent=2))


if __name__=='__main__':main()

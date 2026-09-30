"""Dense-lag signed gradient products by input eigenvalue rank. CPU only."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
from collections import defaultdict
import json
import math
from pathlib import Path
import torch
from persistence_groups import rank_index, digest, LABELS

torch.set_num_threads(2)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'logs/muon_spectra/second_order_audit_20260926/persistence_lags'


def summarize(values,n):
    a,b,c,va,vb,count=values
    sa,sb=a-va/n,b-vb/n
    return {'mean_a2':a,'mean_b2':b,'cross':c,'signal_a2':sa,'signal_b2':sb,
            'noise_a':va,'noise_b':vb,'count':count,
            'cosine_raw':c/math.sqrt(a*b) if a*b>0 else None,
            'cosine':c/math.sqrt(sa*sb) if min(sa,sb)>0 else None,
            'correction_fraction_a':va/n/a if a else None,
            'correction_fraction_b':vb/n/b if b else None}


results=[]
for path in sorted(SOURCE.glob('*.pt')):
    data=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
    n=data['meta']['sequences'];base=data['meta']['t']
    labels=[k[:-7] for k in next(iter(data['matrices'].values())) if k.endswith(':signal') and k!='t:signal']
    totals=defaultdict(lambda:defaultdict(lambda:torch.zeros(4,6,dtype=torch.float64)))
    per_layer={}
    for name,m in data['matrices'].items():
        a=m['t:signal'].double();va=m['t:noise'].double();index=rank_index(a.shape[1])
        kind=name.split('.')[-1]
        per_layer[name]={}
        for label in labels:
            lag=1 if label=='t+1' else int(label.split('=')[1])-base
            b=m[label+':signal'].double();vb=m[label+':noise'].double()
            cols=[a.square().sum(0),b.square().sum(0),(a*b).sum(0),va.sum(0),vb.sum(0),torch.full((a.shape[1],),a.shape[0],dtype=torch.float64)]
            bins=torch.stack([torch.bincount(index,weights=v,minlength=4) for v in cols],dim=1)
            for scope in ['all',kind]:totals[lag][scope]+=bins
            per_layer[name][str(lag)]={LABELS[i]:summarize(bins[i].tolist(),n) for i in range(4)}
    grouped={str(lag):{scope:{**{LABELS[i]:summarize(bins[i].tolist(),n) for i in range(4)},'all':summarize(bins.sum(0).tolist(),n)} for scope,bins in kinds.items()} for lag,kinds in sorted(totals.items())}
    result={'meta':data['meta'],'groups':grouped,'per_layer':per_layer,'input':{'path':str(path.relative_to(ROOT)),'bytes':path.stat().st_size,'sha256':digest(path)}}
    results.append(result)
    print(json.dumps({'arm':path.stem,'lags':{lag:{g:round(d['cosine'],4) if d['cosine'] is not None else None for g,d in kinds['all'].items()} for lag,kinds in grouped.items()}}),flush=True)
    del data
(HERE/'dense_input_persistence.json').write_text(json.dumps({'records':results,'scope':'Fixed independent-basis input ranks; sample A at base, shared B at all later states; no lag independence or displacement association implied.'},indent=2)+'\n')

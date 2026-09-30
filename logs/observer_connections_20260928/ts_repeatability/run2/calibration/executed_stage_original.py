"""Saved-tensor inverse-root and map-stage sensitivity, without model calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import json
import math
import time
import hashlib
import torch
from probe import inverse_root,mapped,muon

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];RUN=HERE/'run2';OUT=RUN/'calibration'
OUT.mkdir(exist_ok=False);start=time.time()
result=json.loads((RUN/'result.json').read_text());assert result['status']=='complete'
load=lambda p:torch.load(p,map_location='cpu',weights_only=False,mmap=True)
factors={n:load(RUN/f'B_{n}.pt') for n in ['A0','A1','B0','C0','D0']}
roots=load(RUN/'fixed_R.pt')['roots'];reference_D={n:load(RUN/f'D_{n}.pt') for n in ['A0','B0']}
cp=load(Path(result['checkpoint']));ids=cp['optimizer']['param_groups'][0]['params']
names=list(roots);momentum={n:cp['optimizer']['state'][i]['momentum_buffer'] for n,i in zip(names,ids)}

def normed(x,target):return x.float()*(target/x.float().norm().clamp_min(1e-30))
def metrics(x,y):
    xx=float(x.double().square().sum());yy=float(y.double().square().sum());xy=float((x.double()*y.double()).sum())
    return {'xx':xx,'yy':yy,'xy':xy,'cosine':xy/math.sqrt(xx*yy),'relative_distance':math.sqrt(max(0,xx+yy-2*xy)/xx)}

records=[];checks={}
for i,name in enumerate(names):
    m=momentum[name].float();r=roots[name];target=math.sqrt(m.shape[0])
    left={key:inverse_root(factors[key][name])[0] for key in ['A0','A1','B0']}
    cd=(factors['C0'][name].double()+factors['D0'][name].double())/2
    cdunit=cd/(cd.trace()/len(cd));ev,u=torch.linalg.eigh(.5*(cdunit+cdunit.T));ev=ev.flip(0).clamp_min(0);u=u.flip(1)
    whitening=(ev+.001).rsqrt();reference_scale=float((ev/(ev+.001)).norm())
    stages={}
    for key,l in left.items():
        z=l@m@r;p=muon.newton_schulz(z,muon.NS_COEFFICIENTS);d=normed(l@p@r,target)
        stages[key]={'entry':normed(z,target),'polar':normed(p,target),'final':d,'p_raw':p}
        if key in reference_D:
            err=float((d.double()-reference_D[key][name].double()).norm()/reference_D[key][name].double().norm())
            checks[name+':'+key]=err;assert err<1e-5
    for other in ['A1','B0']:
        a=factors['A0'][name].double();b=factors[other][name].double()
        a=a/(a.trace()/len(a));b=b/(b.trace()/len(b));delta=a-b
        rotated=u.T@delta@u;white=rotated*whitening[:,None]*whitening[None,:]
        cut=len(ev)//4
        raw_fraction=float(rotated[cut:,cut:].square().sum()/rotated.square().sum())
        white_fraction=float(white[cut:,cut:].square().sum()/white.square().sum())
        pre_only=normed(left['A0']@stages[other]['p_raw']@r,target)
        post_only=normed(left[other]@stages['A0']['p_raw']@r,target)
        records.append({'matrix':name,'pair':'A0__'+other,'dimension':len(ev),
          'factor_delta_relative_to_CD':float(delta.norm()/cdunit.norm()),
          'whitened_delta_relative_to_CD':float(white.norm()/reference_scale),
          'bottom_three_quarters_pair_dimension_share':((len(ev)-cut)/len(ev))**2,
          'raw_delta_bottom_block_share':raw_fraction,'whitened_delta_bottom_block_share':white_fraction,
          'reference_min_eigenvalue_over_mean':float(ev[-1]),'reference_max_eigenvalue_over_mean':float(ev[0]),
          'reference_condition':float(ev[0]/ev[-1].clamp_min(1e-30)),
          'stages':{stage:metrics(stages['A0'][stage],stages[other][stage]) for stage in ['entry','polar','final']},
          'pre_only_vs_A0':metrics(stages['A0']['final'],pre_only),
          'post_only_vs_A0':metrics(stages['A0']['final'],post_only)})
    if i%8==0:print(json.dumps({'matrix':i,'seconds':time.time()-start}),flush=True)
    if time.time()-start>180:raise TimeoutError('180-second saved-tensor calibration bound')
summary={}
for pair in ['A0__A1','A0__B0']:
    summary[pair]={}
    for kind in ['all','q','k','v','o','up','down']:
        rr=[r for r in records if r['pair']==pair and (kind=='all' or r['matrix'].endswith('.'+kind))]
        aggregate={}
        for stage in ['entry','polar','final']:
            sums={k:sum(r['stages'][stage][k] for r in rr) for k in ['xx','yy','xy']}
            aggregate[stage]=sums['xy']/math.sqrt(sums['xx']*sums['yy'])
        for tag in ['pre_only_vs_A0','post_only_vs_A0']:
            sums={k:sum(r[tag][k] for r in rr) for k in ['xx','yy','xy']}
            aggregate[tag]=sums['xy']/math.sqrt(sums['xx']*sums['yy'])
        aggregate['median_factor_delta']=float(torch.tensor([r['factor_delta_relative_to_CD'] for r in rr]).median())
        aggregate['median_whitened_delta']=float(torch.tensor([r['whitened_delta_relative_to_CD'] for r in rr]).median())
        summary[pair][kind]=aggregate
out={'records':records,'summary':summary,'saved_direction_relative_errors':checks,'seconds':time.time()-start,
     'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'scope':'Independent CD reference; finite-sample inverse metric, not true B. All stages normalized to identical per-matrix target solely for comparisons; hybrid maps are sensitivity diagnostics.'}
(OUT/'result.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'seconds':out['seconds'],'summary':summary},indent=2),flush=True)

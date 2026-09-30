"""Synthetic-only BF16 storage-interval checks. Never reads project tensors."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='2'
from pathlib import Path
import json
import numpy as np
import torch

torch.set_num_threads(2);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent
U=np.finfo(np.float64).eps/2

def outward(x,sign):
    return np.nextafter(x, np.inf if sign>0 else -np.inf)

def decode(bits):
    return (np.asarray(bits,dtype=np.uint32)<<16).view(np.float32).astype(np.float64)

def bins(x):
    """Closed RNE preimages; max finite values excluded to avoid overflow cells."""
    x=np.asarray(x,dtype=np.float32)
    q=torch.from_numpy(x.copy()).to(torch.bfloat16).float().numpy()
    bits=q.view(np.uint32)>>16
    assert np.isfinite(q).all() and not np.any((bits&0x7fff)==0x7f7f)
    negative=(bits&0x8000)!=0;zero=(bits&0x7fff)==0
    low=np.where(zero,0x8001,np.where(negative,bits+1,bits-1))
    high=np.where(zero,0x0001,np.where(negative,bits-1,bits+1))
    center=q.astype(np.float64)
    lo=(decode(low)+center)/2;hi=(decode(high)+center)/2
    assert np.all(x.astype(np.float64)>=lo) and np.all(x.astype(np.float64)<=hi)
    return center,lo,hi

def norm_box(lo,hi):
    near=np.where(lo>0,lo,np.where(hi<0,-hi,0.))
    far=np.maximum(np.abs(lo),np.abs(hi))
    n=lo.size;gamma=(2*n+8)*U/(1-(2*n+8)*U)
    lower=float(np.sqrt(max(0,float(np.sum(near*near))*(1-gamma))))
    upper=float(np.sqrt(float(np.sum(far*far))*(1+gamma)))
    return [max(0,float(outward(lower,-1))),float(outward(upper,+1))]

def dot_box(lx,ux,ly,uy):
    products=np.stack([lx*ly,lx*uy,ux*ly,ux*uy])
    low=products.min(0);high=products.max(0)
    n=lx.size;gamma=(2*n+8)*U/(1-(2*n+8)*U)
    error=gamma*np.sum(np.maximum(np.abs(low),np.abs(high)))
    return [float(outward(low.sum()-error,-1)),float(outward(high.sum()+error,+1))]

def cosine_box(lx,ux,ly,uy):
    nx=norm_box(lx,ux);ny=norm_box(ly,uy)
    if nx[0]==0 or ny[0]==0:return None
    d=dot_box(lx,ux,ly,uy)
    den=[nx[0]*ny[0],nx[1]*ny[1]]
    vals=[x/y for x in d for y in den]
    return [max(-1,float(outward(min(vals),-1))),min(1,float(outward(max(vals),+1)))]

def case(ga,g4,h):
    a,al,au=bins(ga);c,cl,cu=bins(g4);h=np.asarray(h,dtype=np.float32).astype(np.float64)
    b=(4*c-a)/3
    bl=outward(outward(4*cl-au,-1)/3,-1);bu=outward(outward(4*cu-al,+1)/3,+1)
    bal=outward(h+al,-1);bau=outward(h+au,+1);bbl=outward(h+bl,-1);bbu=outward(h+bu,+1)
    rawcos=cosine_box(al,au,bl,bu);conditionalcos=cosine_box(bal,bau,bbl,bbu)
    raw=[norm_box(al,au),norm_box(bl,bu)];conditional=[norm_box(bal,bau),norm_box(bbl,bbu)]
    sums_raw=np.sum(raw,axis=0);sums_b=np.sum(conditional,axis=0)
    amp=None if sums_b[0]<=0 else [float(sums_raw[0]/sums_b[1]),float(sums_raw[1]/sums_b[0])]
    # Conditional difference is evaluated through the shared gradient expression.
    d=4*(a-c)/3
    assert np.max(np.abs((h+a)-(h+b)-d))<1e-14
    rng=np.random.default_rng(18)
    for _ in range(2000):
        av=rng.uniform(al,au);cv=rng.uniform(cl,cu);bv=(4*cv-av)/3
        assert np.all(bv>=bl) and np.all(bv<=bu)
        for x,lx,ux,y,ly,uy,co in [(av,al,au,bv,bl,bu,rawcos),(h+av,bal,bau,h+bv,bbl,bbu,conditionalcos)]:
            n=norm_box(lx,ux);assert n[0]<=np.linalg.norm(x)<=n[1]
            z=dot_box(lx,ux,ly,uy);assert z[0]<=np.dot(x,y)<=z[1]
            if co is not None:
                val=float(np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y)))
                assert co[0]-1e-14<=val<=co[1]+1e-14
    return dict(raw_norm_intervals=raw,conditional_norm_intervals=conditional,
                raw_cosine_interval=rawcos,conditional_cosine_interval=conditionalcos,
                denominator_amplification_interval=amp,latent_interval_draws_checked=2000)

def main():
    # Covers signed zero, powers of two, neighbors, and BF16 subnormals.
    values=np.array([0.,-0.,1.,-1.,2.,-2.,.5,-.5,2**-130,-2**-130,
                     1.00390625,1.0078125,-1.00390625],dtype=np.float32)
    q,lo,hi=bins(values)
    results={
        'cancellation':case([1.,.01],[1.,-.005],[-1.,0.]),
        'reinforcement':case([1.,.01],[1.,-.005],[1.,0.]),
        'zero_predictor_unresolved':case([1.,0.],[1.,0.],[-1.,0.]),
    }
    assert results['cancellation']['raw_cosine_interval'][0]>.98
    assert results['cancellation']['conditional_cosine_interval'][1]<0
    assert results['reinforcement']['denominator_amplification_interval'][1]<1
    assert results['zero_predictor_unresolved']['conditional_cosine_interval'] is None
    result=dict(status='passed',scope='Synthetic arrays only; BF16 storage rounding, not earlier FP32 accumulation or historical execution provenance.',
                endpoint_examples=[dict(source=float(x),stored=float(y),lower=float(l),upper=float(u)) for x,y,l,u in zip(values,q,lo,hi)],
                cases=results,actual_scientific_tensors_read=False,model_calls=0,gpu_calls=0)
    (HERE/'precision_peer_synthetic.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()

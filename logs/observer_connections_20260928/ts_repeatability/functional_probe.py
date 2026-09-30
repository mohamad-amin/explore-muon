"""Exact predictive-GN Gram of already retained TS directions, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import json
import hashlib
import time
import numpy as np
import torch
from torch.func import functional_call,jvp
from probe import P,TokenStream,inverse_root,mapped,pair_metrics

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];RUN=HERE/'run2';OUT=RUN/'functional'
OUT.mkdir(exist_ok=False);start=time.time()
load=lambda p:torch.load(p,map_location='cpu',weights_only=False,mmap=True)
base_result=json.loads((RUN/'result.json').read_text());assert base_result['status']=='complete'
cp=load(Path(base_result['checkpoint']));model=P.build_model(cp['config']['model'],cp['model'],torch.device('cpu'))
for p in model.parameters():p.requires_grad_(False)
layers=P.hidden_linears(model);names=list(layers);keys=P.parameter_keys(model,names)
ids=cp['optimizer']['param_groups'][0]['params'];mom={n:cp['optimizer']['state'][i]['momentum_buffer'] for n,i in zip(names,ids)}
roots=load(RUN/'fixed_R.pt')['roots'];b1=load(RUN/'B_A1.pt')
directions={label:load(RUN/f'D_{label}.pt') for label in ['A0','B0','AB','CD','ABCD','PD','ABCD_bf16']}
directions['A1']={n:mapped(mom[n],roots[n],inverse_root(b1[n])[0])[0] for n in names}
match=pair_metrics(directions['A0'],directions['A1'],directions['PD'])['scopes']['all']['cosine']
assert abs(match-base_result['direction_pairs']['A0__A1']['scopes']['all']['cosine'])<1e-10
labels=list(directions);factor=-.028
dirs={k:{n:factor*v for n,v in d.items()} for k,d in directions.items()}
base={**dict(model.named_parameters()),**dict(model.named_buffers())};primals=tuple(base[k] for k in keys)
stream=TokenStream(str(ROOT/cp['config']['train_pattern']));offsets=[2_800_098_304,2_900_098_304]
tokens={};banks=[];checks={}

def save(name,obj):
    p=OUT/(name+'.tmp');p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');p.replace(OUT/name)

for bank,offset in enumerate(offsets):
    bx,by=stream.batch(offset,4,512,torch.device('cpu'));tokens[f'bank{bank}_x']=bx.numpy();tokens[f'bank{bank}_y']=by.numpy()
    records=[]
    for seq in range(4):
        t0=time.time();x=bx[seq:seq+1];ys=by[seq:seq+1];weighted=[]
        def f(*weights):return functional_call(model,{**base,**dict(zip(keys,weights))},(x,))
        with P.explicit_attention():
            p=None
            for label in labels:
                tangents=tuple(dirs[label][n] for n in names)
                logits,dz=jvp(f,primals,tangents)
                if p is None:p=logits.double().softmax(-1)
                dz=dz.double();mean=(p*dz).sum(-1,keepdim=True)
                weighted.append(((dz-mean)*p.sqrt()).reshape(-1))
                del logits,dz
        # Chunk the reduction to bound temporary memory; stored tangent arrays
        # are FP64 and the final Gram is a sum of Gram matrices.
        q=torch.zeros(len(labels),len(labels),dtype=torch.float64)
        for first in range(0,weighted[0].numel(),262144):
            chunk=torch.stack([w[first:first+262144] for w in weighted])
            q.add_(chunk@chunk.T)
        q/=x.numel()
        assert torch.isfinite(q).all() and float(torch.linalg.eigvalsh(q).min())>-1e-8
        del weighted,p,chunk
        if bank==0 and seq==0:
            direct=float(P.gn_quadratic(model,x,dirs['A0'])[0].mean())
            diff={n:dirs['A0'][n]-dirs['B0'][n] for n in names}
            direct_diff=float(P.gn_quadratic(model,x,diff)[0].mean())
            i,j=labels.index('A0'),labels.index('B0');via_diff=float(q[i,i]+q[j,j]-2*q[i,j])
            for name,a,b in [('diagonal',float(q[i,i]),direct),('difference',via_diff,direct_diff)]:
                err=abs(a-b);checks[name]={'gram':a,'direct':b,'absolute_error':err,'relative_error':err/max(abs(b),1e-30)}
                assert err<=max(1e-8,1e-4*abs(b)),checks[name]
            projection=(time.time()-t0)*8+(time.time()-start)
            checks['projected_seconds']=projection
            if projection>300:
                save('result.json',{'status':'cost_gate_stop','checks':checks});raise SystemExit(0)
        records.append({'gram':q.tolist(),'seconds':time.time()-t0})
        status={'status':'running','pid':os.getpid(),'bank':bank,'sequence':seq+1,'seconds':time.time()-start}
        save('status.json',status);print(json.dumps(status),flush=True)
        if status['seconds']>300:raise TimeoutError('300-second functional probe bound')
    banks.append(records)
    save('partial.json',{'labels':labels,'banks':banks,'checks':checks})
np.savez(OUT/'tokens.npz',**tokens)
out={'status':'complete','labels':labels,'step_factor':factor,'offsets':offsets,'sequences_per_bank':4,'banks':banks,'checks':checks,
     'A1_parameter_cosine_reconstruction':match,'seconds':time.time()-start,
     'token_hashes':{n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in tokens.items()},
     'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
     'scope':'Exact finite-input predictive-GN Gram, not population covariance or loss/rate scoring. Lagged stored M and rank0-C proxy; no online state replay.'}
save('result.json',out);save('status.json',{'status':'complete','pid':os.getpid(),'seconds':out['seconds']})
print(json.dumps({'status':'complete','seconds':out['seconds'],'checks':checks}),flush=True)

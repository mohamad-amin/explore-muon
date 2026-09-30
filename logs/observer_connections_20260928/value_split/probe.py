"""CPU functional decomposition of the actual saved value-matrix step."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE/'source'))
from adamw_spectra import gn_probe as P
from adamw_spectra.data import TokenStream

torch.set_num_threads(2);torch.set_num_interop_threads(1)
ARMS={'Muon':'M_lr0.007_s260925_l40s','PD':'PD_a0.25_lr0.01_s260925_ada'}
OFFSETS=[2_500_098_304,2_600_098_304]
N=4
PAIRS={'constant':[1.,0.],'centered':[0.,1.],'combined':[1.,1.],'half_combined':[.5,.5]}


def save(name,obj):
    tmp=HERE/(name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(HERE/name)


def thash(t):return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def relnorm(a,b):return float((a.double()-b.double()).norm()/b.double().norm().clamp_min(1e-30))


class Family:
    def __init__(self,arm_name):
        self.arm=ROOT/'logs/muon_spectra/soaudit_traj_20260926'/arm_name
        kept=self.arm/'scientific/kept'
        self.checkpoint=kept/'step000500.pt';self.nextfile=kept/'step000501_weights.pt'
        before=torch.load(self.checkpoint,map_location='cpu',weights_only=False,mmap=True)
        after=torch.load(self.nextfile,map_location='cpu',weights_only=False,mmap=True)
        assert before['step']==500 and after['step']==501
        self.config=before['config']
        self.model=P.build_model(self.config['model'],before['model'],torch.device('cpu'))
        for p in self.model.parameters():p.requires_grad_(False)
        mp=ROOT/'logs/muon_spectra/second_order_audit_20260926/marginals'/f'{arm_name}_step000500.pt'
        marginal=torch.load(mp,map_location='cpu',weights_only=False)
        assert marginal['step']==500 and Path(marginal['arm']).resolve()==self.arm
        self.state={'z':torch.zeros(2),'enabled':True}
        self.layers=[];self.handles=[];self.provenance={'means':{},'base_V':{},'next_V':{}}
        for index,block in enumerate(self.model.blocks):
            key=f'blocks.{index}.attn.v.weight';name=f'block{index+1:02d}.v'
            w=before['model'][key].float().clone();w1=after['model'][key].float().clone();delta=w1-w
            mu=marginal['arrays'][name]['x_mean'].float().clone();constant=delta@mu
            self.layers.append((block.attn.v,w,delta))
            self.provenance['means'][name]=thash(mu);self.provenance['base_V'][key]=thash(w);self.provenance['next_V'][key]=thash(w1)
            def make_hook(d,c,mean):
                def hook(module,inputs,output):
                    if not self.state['enabled']:return output
                    a,b=self.state['z'].unbind()
                    return output+a*c+b*F.linear(inputs[0]-mean,d)
                return hook
            self.handles.append(block.attn.v.register_forward_hook(make_hook(delta,constant,mu)))
        self.provenance['marginal']={'path':str(mp.relative_to(ROOT)),'sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'sequences':marginal['sequences']}
        self.provenance['checkpoints']={str(p.relative_to(ROOT)):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in [self.checkpoint,self.nextfile]}

    def logits(self,z,x):
        self.state['z']=z
        return self.model(x)

    def direct(self,c,x):
        self.state['enabled']=False
        try:
            with torch.no_grad():
                for layer,w,d in self.layers:layer.weight.copy_(w+c*d)
                value=self.model(x)
                for layer,w,d in self.layers:layer.weight.copy_(w)
            return value
        finally:self.state['enabled']=True

    def close(self):
        for h in self.handles:h.remove()


def terms(base,jc,jz,y):
    p=base.double().softmax(-1);a=jc.double();b=jz.double()
    ma=(p*a).sum(-1);mb=(p*b).sum(-1)
    sc=(ma-a.gather(-1,y[...,None]).squeeze(-1)).mean()
    sz=(mb-b.gather(-1,y[...,None]).squeeze(-1)).mean()
    qcc=((p*a.square()).sum(-1)-ma.square()).mean()
    qzz=((p*b.square()).sum(-1)-mb.square()).mean()
    qcz=((p*a*b).sum(-1)-ma*mb).mean()
    q=torch.stack([torch.stack([qcc,qcz]),torch.stack([qcz,qzz])])
    assert float(torch.linalg.eigvalsh(q).min())>=-1e-9
    return {'slopes':[float(sc),float(sz)],'gn':q.tolist(),
            'logit_rms':[float(a.square().mean().sqrt()),float(b.square().mean().sqrt())]}


def qualify(f,x,y):
    start=time.time();zero=torch.zeros(2)
    with P.explicit_attention():
        base,jc=jvp(lambda z:f.logits(z,x),(zero,),(torch.tensor([1.,0.]),))
        _,jz=jvp(lambda z:f.logits(z,x),(zero,),(torch.tensor([0.,1.]),))
        _,jj=jvp(lambda z:f.logits(z,x),(zero,),(torch.ones(2),))
        addition=relnorm(jj,jc+jz);assert addition<=1e-5
        finite={};diagonal={}
        with torch.no_grad():
            for c in [.5,1.]:
                hooked=f.logits(torch.full((2,),c),x);direct=f.direct(c,x)
                e=relnorm(hooked,direct);diagonal[str(c)]=e;assert e<=1e-5
            for label,v,tangent in [('constant',torch.tensor([1.,0.]),jc),('centered',torch.tensor([0.,1.]),jz)]:
                fd=(f.logits(.02*v,x)-f.logits(-.02*v,x))/.04
                e=relnorm(fd,tangent);finite[label]=e;assert e<=.02
        local=terms(base,jc,jz,y)
        coeff=zero.clone().requires_grad_()
        loss=P.token_losses(f.logits(coeff,x),y).mean()
        reverse=torch.autograd.grad(loss,coeff)[0].detach().double()
        error=(reverse-torch.tensor(local['slopes'],dtype=torch.float64)).abs()
        bound=torch.maximum(torch.full_like(error,1e-6),.002*reverse.abs())
        assert bool((error<=bound).all()),(reverse,local,error)
    return {'jvp_additivity_relative_error':addition,'direct_weight_diagonal_relative_error':diagonal,
            'central_difference_relative_error':finite,'reverse_slopes':reverse.tolist(),
            'jvp_slopes':local['slopes'],'slope_absolute_errors':error.tolist(),'seconds':time.time()-start}


def measure(f,x,y):
    zero=torch.zeros(2)
    with P.explicit_attention():
        base,jc=jvp(lambda z:f.logits(z,x),(zero,),(torch.tensor([1.,0.]),))
        _,jz=jvp(lambda z:f.logits(z,x),(zero,),(torch.tensor([0.,1.]),))
        result=terms(base,jc,jz,y)
        result['losses']={'base':float(P.token_losses(base,y).mean())}
        del base,jc,jz
        with torch.no_grad():
            for name,pair in PAIRS.items():result['losses'][name]=float(P.token_losses(f.logits(torch.tensor(pair),x),y).mean())
    return result


def main():
    start=time.time();report={'status':'running','step':500,'next_step':501,'offsets':OFFSETS,'sequences_per_bank':N,'methods':{}}
    stream=TokenStream(str(ROOT/'data/fineweb10B/fineweb_train_*.bin'))
    token_arrays={}
    for i,offset in enumerate(OFFSETS):
        x,y=stream.batch(offset,N,512,torch.device('cpu'));token_arrays[f'bank{i}_x']=x.numpy();token_arrays[f'bank{i}_y']=y.numpy()
    np.savez(HERE/'tokens.npz',**token_arrays)
    report['tokens_sha256']={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in token_arrays.items()}
    manifest=json.loads((HERE/'source/manifest.json').read_text())
    for name,row in manifest.items():assert hashlib.sha256((HERE/'source/adamw_spectra'/name).read_bytes()).hexdigest()==row['sha256']
    report['source_manifest']=manifest;report['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for method,arm in ARMS.items():
        f=Family(arm)
        x=torch.from_numpy(token_arrays['bank0_x'][:1]);y=torch.from_numpy(token_arrays['bank0_y'][:1])
        try:q=qualify(f,x,y)
        except Exception as exc:
            report['status']='qualification_failed';report['failure']={'method':method,'type':type(exc).__name__,'message':str(exc)}
            save('result.json',report);save('status.json',{'status':'qualification_failed','method':method,'seconds':time.time()-start});raise
        report['methods'][method]={'qualification':q,'provenance':f.provenance,'config':f.config,'banks':[]}
        print(json.dumps({'stage':'qualified','method':method,**q}),flush=True)
        if method=='Muon' and q['seconds']*(2*N*len(ARMS)+1)>900:
            report['status']='cost_gate_stop';save('result.json',report);save('status.json',{'status':'cost_gate_stop','seconds':time.time()-start});return
        for bank in range(2):
            bank_rows=[]
            for seq in range(N):
                x=torch.from_numpy(token_arrays[f'bank{bank}_x'][seq:seq+1]);y=torch.from_numpy(token_arrays[f'bank{bank}_y'][seq:seq+1])
                row=measure(f,x,y);bank_rows.append(row)
                save('status.json',{'status':'running','pid':os.getpid(),'method':method,'bank':bank,'completed_sequences':seq+1,'seconds':time.time()-start})
                print(json.dumps({'method':method,'bank':bank,'seq':seq,'slopes':row['slopes'],'gn':row['gn'],'losses':row['losses'],'seconds':time.time()-start}),flush=True)
            report['methods'][method]['banks'].append(bank_rows);save('result.json',report)
        for path,record in f.provenance['checkpoints'].items():
            stat=(ROOT/path).stat();assert stat.st_size==record['bytes'] and stat.st_mtime_ns==record['mtime_ns']
        f.close();del f
    report['status']='complete';report['seconds']=time.time()-start;save('result.json',report)
    save('status.json',{'status':'complete','pid':os.getpid(),'seconds':report['seconds']})
    print(json.dumps({'stage':'complete','seconds':report['seconds']}),flush=True)


if __name__=='__main__':main()

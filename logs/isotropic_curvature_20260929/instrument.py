"""FP64 finite-curvature instrumentation; no optimizer/training/CUDA calls."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(HERE/'source'))
from adamw_spectra.model import GPT,ModelConfig
from adamw_spectra.data import TokenStream

BLOCKS=(0,3,7)
STEPS=(10,500,1300)
SCALES=(-16.,-8.,-4.,-2.,-1.,-.5,-.25,0.,.25,.5,1.,2.,4.,8.,16.)
LABELS=('actual','left0','left1','right_raw','right_white')
ARMS={'Muon':'M_lr0.007_s260925_l40s','PD':'PD_a0.25_lr0.01_s260925_ada'}
BASE=ROOT/'logs/muon_spectra/soaudit_traj_20260926'
CAL_OFFSET=2_900_262_144
SCORE_OFFSETS=(2_910_262_144,2_920_262_144)


def sha(b): return hashlib.sha256(b).hexdigest()


def tensor_hash(t): return sha(t.detach().contiguous().numpy().tobytes())


def write_json(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(path)


def read_json(path,manifest):
    b=path.read_bytes();manifest[str(path.relative_to(ROOT))]={'bytes':len(b),'sha256':sha(b)}
    return json.loads(b)


def token_loss(z,y):
    assert z.dtype==torch.float64
    return torch.logsumexp(z,dim=-1)-z.gather(-1,y[...,None]).squeeze(-1)


def make_model(cfg,state):
    model=GPT(ModelConfig(**{**cfg,'track_input_stats':False,'track_input_cov':False})).double()
    model.load_state_dict(state,strict=True);model.eval()
    for p in model.parameters():p.requires_grad_(False)
    return model


@torch.no_grad()
def capture(model,x):
    h=model.embed(x)+model.position(model._positions[:x.shape[1]])
    caches={}
    for i,b in enumerate(model.blocks):
        h=h+b.attn(b.ln1(h));u=b.ln2(h);z=b.mlp.up(u)
        if i in BLOCKS:caches[i]={'residual':h.clone(),'input':u.clone(),'preactivation':z.clone()}
        h=h+b.mlp.down(F.gelu(z,approximate='tanh'))
    return caches,model.head(model.norm(h))


def suffix(model,block,cache,z):
    h=cache['residual']+model.blocks[block].mlp.down(F.gelu(z,approximate='tanh'))
    for b in model.blocks[block+1:]:h=b(h)
    return model.head(model.norm(h))


def ray_fn(model,block,cache,direction):
    dz=F.linear(cache['input'],direction)
    def fn(a):return suffix(model,block,cache,cache['preactivation']+a*dz)
    return fn,dz


def panel_fn(model,block,cache,directions):
    kicks=torch.stack([F.linear(cache['input'],d) for d in directions])
    def fn(a):
        z=cache['preactivation']+(a.reshape(-1,1,1,1)*kicks).sum(0)
        return suffix(model,block,cache,z)
    return fn


def joint_fn(model,cache0,directions):
    """Three simultaneous parameter changes, using each CURRENT layer input."""
    mapping={b:i for i,b in enumerate(BLOCKS)}
    def fn(coeff):
        h=cache0['residual']
        for i,b in enumerate(model.blocks):
            if i:h=h+b.attn(b.ln1(h))
            u=b.ln2(h);z=b.mlp.up(u)
            if i in mapping:z=z+coeff[mapping[i]]*F.linear(u,directions[i])
            h=h+b.mlp.down(F.gelu(z,approximate='tanh'))
        return model.head(model.norm(h))
    return fn


@torch.no_grad()
def directional(fn,y):
    zero=torch.zeros((),dtype=torch.float64);one=torch.ones_like(zero)
    def first(a):return jvp(fn,(a,),(one,))
    (z,dz),(dz_again,d2z)=jvp(first,(zero,),(one,))
    repeat=float((dz-dz_again).abs().max());assert repeat<1e-10
    lp=torch.log_softmax(z,-1);p=lp.exp();mean=(p*dz).sum(-1)
    slope=mean-dz.gather(-1,y[...,None]).squeeze(-1)
    gn=(p*(dz-mean[...,None]).square()).sum(-1)
    nonlinear=(p*d2z).sum(-1)-d2z.gather(-1,y[...,None]).squeeze(-1)
    hessian=gn+nonlinear
    loss=-lp.gather(-1,y[...,None]).squeeze(-1)
    assert all(torch.isfinite(t).all() for t in (loss,slope,gn,hessian,dz))
    assert float(gn.min())>=-1e-12
    return dict(loss=loss,slope=slope,gn=gn,hessian=hessian,nonlinear=nonlinear,
                tangent=dz,probability=p,repeat_error=repeat)


def projected_hessian(fn,y,n):
    a=torch.zeros(n,dtype=torch.float64,requires_grad=True)
    h=torch.autograd.functional.hessian(lambda v:token_loss(fn(v),y).mean(),a,vectorize=False)
    symmetry=float((h-h.T).abs().max());assert symmetry<1e-9*max(1.,float(h.abs().max()))
    return h.detach(),symmetry


@torch.no_grad()
def predictive_gram(tangents,p):
    out=torch.empty((*p.shape[:-1],len(tangents),len(tangents)),dtype=torch.float64)
    means=[(p*d).sum(-1) for d in tangents]
    for i in range(len(tangents)):
        for j in range(i,len(tangents)):
            value=(p*(tangents[i]-means[i][...,None])*(tangents[j]-means[j][...,None])).sum(-1)
            out[...,i,j]=out[...,j,i]=value
    return out


def gradient_at_site(model,block,cache,y):
    z=cache['preactivation'].detach().requires_grad_(True)
    loss=token_loss(suffix(model,block,cache,z),y).mean()
    e,=torch.autograd.grad(loss,z)
    g=e.flatten(0,1).T@cache['input'].flatten(0,1)
    return g.detach(),(e*y.numel()).detach()


def rotations(d,c,block):
    eigen,v=torch.linalg.eigh((c+c.T)/2)
    assert float(eigen.min())>-1e-10
    shift=.001*float(eigen.mean());reg=eigen.clamp_min(0)+shift
    s=(v*reg.sqrt())@v.T;si=(v*reg.rsqrt())@v.T
    rng=np.random.default_rng(29090000+block)
    operators={};dirs={'actual':d}
    for j in range(2):
        perm=rng.permutation(d.shape[0]);sign=rng.choice([-1.,1.],size=d.shape[0])
        operators[f'left{j}']={'permutation':perm.tolist(),'sign':sign.tolist()}
        dirs[f'left{j}']=d[torch.from_numpy(perm)]*torch.from_numpy(sign)[:,None]
    perm=rng.permutation(d.shape[1]);sign=rng.choice([-1.,1.],size=d.shape[1])
    operators['right']={'permutation':perm.tolist(),'sign':sign.tolist()}
    def right(x):return x[:,torch.from_numpy(perm)]*torch.from_numpy(sign)[None,:]
    dirs['right_raw']=right(d);p=d@s;dirs['right_white']=right(p)@si
    gram=d.T@d;checks={}
    for name in ['left0','left1']:
        checks[name+'_gram_error']=float((dirs[name].T@dirs[name]-gram).norm()/gram.norm())
    target=gram[torch.from_numpy(perm)][:,torch.from_numpy(perm)]*torch.from_numpy(sign)[:,None]*torch.from_numpy(sign)[None,:]
    checks['right_raw_gram_error']=float((dirs['right_raw'].T@dirs['right_raw']-target).norm()/gram.norm())
    checks['white_mapback_error']=float((dirs['right_white']@s-right(p)).norm()/p.norm())
    assert max(checks.values())<1e-10,checks
    return dirs,dict(C=c,C_eigenvalues=eigen,C_vectors=v,S=s,S_inverse=si,C_regularization=shift,
                     input_whitening_eigenvalues=eigen/(eigen+shift)),operators,checks


def load_pair(method,step,manifest):
    p=BASE/ARMS[method]/'scientific'
    meta=read_json(p/'metadata.json',manifest)
    assert read_json(p/'status.json',manifest)['status']=='complete'
    source={}
    for name,expected in meta['source_sha256'].items():
        b=(p/'source'/name).read_bytes();assert sha(b)==expected
        source[name]=expected
    paths=[p/'kept'/f'step{step:06d}.pt',p/'kept'/f'step{step+1:06d}_weights.pt']
    info=[dict(path=str(f.relative_to(ROOT)),bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns) for f in paths]
    a,b=[torch.load(f,map_location='cpu',weights_only=False,mmap=True) for f in paths]
    assert a['step']==step and b['step']==step+1
    assert a['tokens']==step*1048576 and b['tokens']==(step+1)*1048576
    assert a['config']==meta['config']
    assert set(a['model'])==set(b['model'])
    body=[name for name in a['model'] if meta['parameter_optimizers'].get(name)=='muon']
    ids=a['optimizer']['param_groups'][0]['params'];assert len(body)==len(ids)==48
    state=a['optimizer']['state'];mapping=dict(zip(body,ids))
    dirs={};mom={}
    for block,expected_position in zip(BLOCKS,[4,22,46]):
        name=f'blocks.{block}.mlp.up.weight';assert body[expected_position]==name
        w0,w1=a['model'][name],b['model'][name]
        assert w0.shape==w1.shape==(2048,512)
        dirs[block]=w1.double()-w0.double()
        mom[block]=state[mapping[name]]['momentum_buffer'].clone()
        assert mom[block].shape==w0.shape
    for rec,obj in zip(info,[a,b]):
        rec['model_tensor_sha256']={name:tensor_hash(t) for name,t in obj['model'].items()}
    next_row=read_json(p/'steps'/f'step{step+1:06d}.json',manifest)
    return a,b,meta,dirs,mom,dict(files=info,source_checks=source,next_step=next_row)


def data(meta):
    stream=TokenStream(str(ROOT/meta['config']['train_pattern']))
    assert stream.manifest==meta['train_manifest'] and stream.total==3200000000
    assert CAL_OFFSET>1539870720
    calx,caly=stream.batch(CAL_OFFSET,8,512,'cpu')
    banks=[stream.batch(offset,2,512,'cpu') for offset in SCORE_OFFSETS]
    return calx,caly,torch.cat([b[0] for b in banks]),torch.cat([b[1] for b in banks])

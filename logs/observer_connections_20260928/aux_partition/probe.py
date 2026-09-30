"""Fixed endpoint factorial and exact finite-logit interaction, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE/'source'))
from adamw_spectra import gn_probe as P
from adamw_spectra.data import TokenStream

torch.set_num_threads(2);torch.set_num_interop_threads(1)
GROUPS={'B':1,'H':2,'E':4,'N':8}
OLD=HERE.parent/'body_aux'
ARM=ROOT/'logs/muon_spectra/soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s'
START=time.time();OUT=HERE/'run1';OUT.mkdir(exist_ok=False)


def save(name,obj):
    p=OUT/(name+'.tmp');p.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');p.replace(OUT/name)


def progress(**kwargs):
    d={'status':'running','pid':os.getpid(),'seconds':time.time()-START,**kwargs}
    save('status.json',d);print(json.dumps(d),flush=True)
    if d['seconds']>600:raise TimeoutError('600-second factorial CPU wall bound')


def mask_name(mask):return ''.join(k for k,v in GROUPS.items() if mask&v) or 'base'


def mobius(loss):
    out=loss.copy()
    for bit in [1,2,4,8]:
        for mask in range(16):
            if mask&bit:out[mask]-=out[mask^bit]
    for mask in range(16):
        direct=sum((-1)**((mask.bit_count()-sub.bit_count()))*loss[sub] for sub in range(16) if sub&mask==sub)
        assert abs(direct-out[mask])<1e-10
        assert abs(sum(out[sub] for sub in range(16) if sub&mask==sub)-loss[mask])<1e-10
    total=loss[15]-loss[1]-loss[14]+loss[0]
    assert abs(sum(out[m] for m in range(16) if m&1 and m!=1)-total)<1e-10
    return out


def summaries(z,y):
    lse={};loss32={};loss64={}
    target=y.flatten()
    for mask,a in z.items():
        aa=a.double();lse[mask]=torch.logsumexp(aa,dim=-1)
        loss64[mask]=float((lse[mask]-aa.gather(-1,target[:,None]).squeeze(-1)).mean())
        loss32[mask]=float(F.cross_entropy(a,target))
    effects=mobius(np.array([loss64[i] for i in range(16)]))
    planes={}
    triples=[(bit,context) for bit in [2,4,8] for context in range(16) if context&(15^(14^bit))==0]
    assert len(triples)==12
    triples.append((14,0))
    for abit,context in triples:
        i,j,k,l=context,context|1,context|abit,context|1|abit
        zadd=z[j].double()+z[k].double()-z[i].double()
        lse_add=torch.logsumexp(zadd,dim=-1)
        ce_add=float((lse_add-zadd.gather(-1,target[:,None]).squeeze(-1)).mean())
        overlap=float((lse_add-lse[j]-lse[k]+lse[i]).mean())
        mixed=loss64[l]-ce_add
        interaction=loss64[l]-loss64[j]-loss64[k]+loss64[i]
        assert abs(overlap+mixed-interaction)<1e-10
        c=z[l].double()-zadd
        planes[f'B:{mask_name(abit)}|{mask_name(context)}']={
            'body_effect':loss64[j]-loss64[i],'aux_effect':loss64[k]-loss64[i],
            'aux_given_body':loss64[l]-loss64[j],'body_given_aux':loss64[l]-loss64[k],
            'interaction':interaction,'overlap':overlap,'mixed_logit_loss':mixed,
            'mixed_logit_rms':float(c.square().mean().sqrt()),
            'additive_logit_change_rms':float((zadd-z[i].double()).square().mean().sqrt())}
    # Finite-response geometry at base, distinct from an infinitesimal GN.
    base=z[0].double();p=base.softmax(-1)
    changes=[z[bit].double()-base for bit in [1,2,4,8]]
    means=[(p*v).sum(-1) for v in changes]
    q=torch.empty(4,4,dtype=torch.float64)
    for i in range(4):
        for j in range(4):q[i,j]=((p*changes[i]*changes[j]).sum(-1)-means[i]*means[j]).mean()
    assert float(torch.linalg.eigvalsh(q).min())>=-1e-9
    kl={mask_name(bit):float((lse[bit]-lse[0]-(p*(z[bit].double()-base)).sum(-1)).mean()) for bit in [1,2,4,8,14,15]}
    assert min(kl.values())>=-1e-10
    return {'loss32':{mask_name(i):loss32[i] for i in range(16)},
            'loss64':{mask_name(i):loss64[i] for i in range(16)},
            'mobius':{mask_name(i):float(effects[i]) for i in range(16)},
            'planes':planes,'finite_logit_gram':q.tolist(),'predictive_kl_from_base':kl}


def main():
    old=json.loads((OLD/'result.json').read_text());assert old['status']=='complete'
    kept=ARM/'scientific/kept'
    paths=[kept/'step000183.pt',kept/'step000184_weights.pt']
    before,after=[torch.load(p,map_location='cpu',weights_only=False,mmap=True) for p in paths]
    assert before['step']==183 and after['step']==184
    model=P.build_model(before['config']['model'],before['model'],torch.device('cpu'))
    for p in model.parameters():p.requires_grad_(False)
    named=dict(model.named_parameters());body_ids={id(l.weight) for l in P.hidden_linears(model).values()}
    partition={key:('B' if id(p) in body_ids else 'H' if key=='head.weight' else 'E' if key in ['embed.weight','position.weight'] else 'N') for key,p in named.items()}
    assert set(named)==set(before['model'])==set(after['model'])
    assert sum(g=='B' for g in partition.values())==48 and sum(g=='N' for g in partition.values())==33
    assert all(named[k].ndim==1 for k,g in partition.items() if g=='N')
    assert model.head.weight is not model.embed.weight
    def set_corner(mask):
        with torch.no_grad():
            for key,p in named.items():p.copy_((after if mask&GROUPS[partition[key]] else before)['model'][key])
    captured={}
    handle=model.head.register_forward_pre_hook(lambda mod,args:captured.__setitem__('h',args[0].detach()))
    tokens={};banks=[]
    oldtokens=np.load(OLD/'probe_tokens.npz')
    for i in range(2):
        tokens[f'bank{i}_x']=oldtokens[f'bank{i}_inputs'];tokens[f'bank{i}_y']=oldtokens[f'bank{i}_targets']
        banks.append({'role':'reproduction/discovery','offset':old['banks'][i]['offset'],'n':8})
    stream=TokenStream(str(ROOT/before['config']['train_pattern']))
    for i,offset in enumerate([2_550_131_072,2_650_131_072],2):
        x,y=stream.batch(offset,4,512,torch.device('cpu'));tokens[f'bank{i}_x']=x.numpy();tokens[f'bank{i}_y']=y.numpy()
        banks.append({'role':'fresh-input consistency','offset':offset,'n':4})
    np.savez(OUT/'tokens.npz',**tokens)
    report={'status':'running','groups':GROUPS,'partition':partition,'config':before['config'],
            'checkpoint_files':{str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in paths},
            'token_hashes':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in tokens.items()},'banks':[],'qualification':{}}
    manifest=json.loads((HERE/'source/manifest.json').read_text())
    for name,v in manifest.items():assert hashlib.sha256((HERE/'source/adamw_spectra'/name).read_bytes()).hexdigest()==v['sha256']
    report['source_manifest']=manifest;report['probe_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    total=sum(b['n'] for b in banks)
    for bank,bmeta in enumerate(banks):
        rows=[]
        for seq in range(bmeta['n']):
            t0=time.time();x=torch.from_numpy(tokens[f'bank{bank}_x'][seq:seq+1]);y=torch.from_numpy(tokens[f'bank{bank}_y'][seq:seq+1]);z={};hidden={}
            with torch.no_grad():
                for mask in range(16):
                    if mask&2:continue
                    set_corner(mask)
                    if bank==0 and seq==0 and mask==0:assert all(torch.equal(p,before['model'][k]) for k,p in named.items())
                    z[mask]=model(x).squeeze(0)
                    h=captured['h'];z[mask|2]=F.linear(h,after['model']['head.weight']).squeeze(0)
                    if bank==0 and seq==0 and mask in [0,1]:hidden[mask]=h.clone()
                if bank==0 and seq==0:
                    direct_checks={}
                    for mask in [2,15]:
                        set_corner(mask)
                        if mask==15:assert all(torch.equal(p,after['model'][k]) for k,p in named.items())
                        zz=model(x).squeeze(0);err=float((zz-z[mask]).abs().max());direct_checks[str(mask)]=err;assert err<=1e-5
                    delta=after['model']['head.weight'].double()-before['model']['head.weight'].double()
                    predicted=F.linear((hidden[1]-hidden[0]).double(),delta).squeeze(0)
                    measured=z[3].double()-z[1].double()-z[2].double()+z[0].double()
                    err=float((predicted-measured).square().mean().sqrt());scale=float(measured.square().mean().sqrt())
                    report['qualification']={'head_shortcut_max_logit_errors':direct_checks,'bilinear_rms_error':err,'mixed_logit_rms':scale,'bilinear_relative_error':err/max(scale,1e-30)}
                    assert err<=max(2e-6,.02*scale)
            row=summaries(z,y);del z,hidden
            if bank<2:
                differences={key:row['loss32'][key]-old['banks'][bank]['losses'][legacy][seq] for key,legacy in [('base','base'),('B','body'),('HEN','aux'),('BHEN','full')]}
                row['old_loss32_differences']=differences;assert max(abs(v) for v in differences.values())<=2e-6,differences
            rows.append(row)
            if bank==0 and seq==0:
                forecast=(time.time()-t0)*total+(time.time()-START)
                report['qualification']['projected_seconds']=forecast
                if forecast>600:
                    report['status']='cost_gate_stop';save('result.json',report);save('status.json',{'status':'cost_gate_stop','seconds':time.time()-START});return
            progress(bank=bank,sequence=seq+1)
        report['banks'].append({**bmeta,'rows':rows});save('result.json',report)
    handle.remove()
    for path,meta in report['checkpoint_files'].items():
        s=Path(path).stat();assert s.st_size==meta['bytes'] and s.st_mtime_ns==meta['mtime_ns']
    report['status']='complete';report['seconds']=time.time()-START;save('result.json',report)
    save('status.json',{'status':'complete','pid':os.getpid(),'seconds':report['seconds']})
    print(json.dumps({'status':'complete','seconds':report['seconds'],'qualification':report['qualification']}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        (OUT/'failure.txt').write_text(traceback.format_exc());save('status.json',{'status':'failed','type':type(exc).__name__,'message':str(exc),'seconds':time.time()-START});raise

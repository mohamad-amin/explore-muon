"""Exact four-corner logit-plane split, frozen inputs, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'2')
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OLD=HERE.parent/'dose_function/run1'
sys.path.insert(0,str(HERE/'source'))
from adamw_spectra import gn_probe as P
from adamw_spectra.data import TokenStream
torch.set_num_threads(2);torch.set_num_interop_threads(1)
OUT=HERE/'run1';OUT.mkdir(exist_ok=False);START=time.time();CAP=600.
STATES=[('q09',9),('h09',9),('q09',46),('h09',46)]
PARTS=['interaction','overlap','mixed_loss','J_direct','KL_add','base_entropy']


def save(name,data):
    tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(OUT/name)


def status(**kw):
    data={'status':'running','pid':os.getpid(),'seconds':time.time()-START,**kw};save('status.json',data);print(json.dumps(data),flush=True)
    if data['seconds']>CAP:raise TimeoutError('600-second checked CPU boundary')


def state_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()


def main():
    old=json.loads((OLD/'result.json').read_text());assert old['status']=='complete'
    olddata=np.load(OLD/'scalar_tokens.npz');oldtokens=np.load(OLD/'tokens.npz')
    old_index={tuple(v):i for i,v in enumerate(old['states'])}
    xs=[torch.from_numpy(oldtokens['inputs'])];ys=[torch.from_numpy(oldtokens['targets'])]
    for name,v in [('inputs',xs[0]),('targets',ys[0])]:assert hashlib.sha256(v.numpy().tobytes()).hexdigest()==old['token_hashes'][name]
    cfg=old['metadata']['q09']['config'];stream=TokenStream(str(ROOT/cfg['train_pattern']))
    new_offsets=[3_170_131_072,3_195_131_072]
    for offset in new_offsets:
        x,y=stream.batch(offset,4,512,'cpu');xs.append(x);ys.append(y)
    x=torch.cat(xs);y=torch.cat(ys);assert x.shape==y.shape==(16,512)
    np.savez(OUT/'tokens.npz',inputs=x.numpy(),targets=y.numpy())
    L=np.full((4,16,4,512),np.nan);KL=np.full((4,16,3,512),np.nan);parts=np.full((4,16,6,512),np.nan)
    report={'status':'prepared','states':STATES,'parts':PARTS,'old_offsets':old['score_offsets'],'fresh_offsets':new_offsets,
            'roles':['old_reproduction']*8+['fresh_bank0']*4+['fresh_bank1']*4,
            'token_hashes':{k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in [('inputs',x),('targets',y)]},
            'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (HERE/'source/adamw_spectra').glob('*.py')},
            'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checkpoint_provenance':{},
            'qualification':{'max_split_error':0.,'max_mixed_identity_error':0.,'max_J_identity_error':0.,
                             'max_old_token_nll_error':0.,'max_old_mean_nll_error':0.,'max_old_token_kl_error':0.,'max_old_mean_kl_error':0.}}
    save('prepared.json',report);model=None;loads=[];durations=[];done=0
    def arrays(name):np.savez(OUT/name,corner_nll=L,corner_forward_kl=KL,parts=parts)
    for j,(label,step) in enumerate(STATES):
        begin=time.time();prior=old['checkpoint_provenance'][f'{label}_{step}'];paths=[Path(d['path']) for d in prior]
        before,after=[torch.load(p,map_location='cpu',weights_only=False,mmap=True) for p in paths]
        assert before['step']==step and after['step']==step+1
        assert before['tokens']<min(new_offsets)
        current=[]
        for d,p,expected in zip([before,after],paths,prior):
            sha=state_hash(d['model']);assert sha==expected['model_tensor_sha256']
            current.append({'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,'model_tensor_sha256':sha})
        report['checkpoint_provenance'][f'{label}_{step}']=current
        if model is None:model=P.build_model(before['config']['model'],before['model'],torch.device('cpu'))
        else:model.load_state_dict(before['model'])
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        named=dict(model.named_parameters());body={k for k,p in named.items() if k.startswith('blocks.') and p.ndim==2}
        assert len(named)==84 and len(body)==48 and set(named)==set(before['model'])==set(after['model'])
        def set_corner(mask,verify=False):
            with torch.no_grad():
                for k,p in named.items():
                    target=(after if mask&(1 if k in body else 2) else before)['model'][k]
                    p.copy_(target)
                    if verify:assert torch.equal(p,target)
        loads.append(time.time()-begin)
        for i in range(16):
            started=time.time();logp=[]
            with torch.inference_mode():
                for mask in range(4):
                    set_corner(mask,verify=(j==0 and i==0))
                    lp=torch.log_softmax(model(x[i:i+1])[0].double(),-1)
                    assert torch.isfinite(lp).all();logp.append(lp)
                    L[j,i,mask]=(-lp.gather(-1,y[i,:,None]).squeeze(-1)).numpy()
                p0=logp[0].exp()
                for mask in [1,2,3]:
                    k=(p0*(logp[0]-logp[mask])).sum(-1)
                    assert torch.isfinite(k).all() and float(k.min())>=-1e-10
                    KL[j,i,mask-1]=k.numpy()
                raw=logp[1]+logp[2]-logp[0]
                overlap=torch.logsumexp(raw,-1);add=raw-overlap[:,None]
                loss_add=-add.gather(-1,y[i,:,None]).squeeze(-1)
                mixed=torch.from_numpy(L[j,i,3])-loss_add
                interaction=torch.from_numpy(L[j,i,3]-L[j,i,1]-L[j,i,2]+L[j,i,0])
                delta_mixed=logp[3]-logp[1]-logp[2]+logp[0]
                J=(p0*delta_mixed).sum(-1)-delta_mixed.gather(-1,y[i,:,None]).squeeze(-1)
                kl_add=(p0*(logp[0]-add)).sum(-1);assert float(kl_add.min())>=-1e-10
                ent=-(p0*logp[0]).sum(-1)
                for t,v in enumerate([interaction,overlap,mixed,J,kl_add,ent]):parts[j,i,t]=v.numpy()
                errors={'max_split_error':float((interaction-overlap-mixed).abs().max()),
                        'max_mixed_identity_error':float((mixed-J-torch.from_numpy(KL[j,i,2])+kl_add).abs().max()),
                        'max_J_identity_error':float((J-interaction+torch.from_numpy(KL[j,i,2]-KL[j,i,0]-KL[j,i,1])).abs().max())}
                assert max(errors.values())<=1e-10,errors
            if i<8:
                k=old_index[(label,step)];de=L[j,i]-olddata['loss64'][k,i];dk=KL[j,i]-olddata['kl_forward'][k,i]
                errors.update(max_old_token_nll_error=float(np.max(np.abs(de))),max_old_mean_nll_error=float(np.max(np.abs(de.mean(-1)))),
                              max_old_token_kl_error=float(np.max(np.abs(dk))),max_old_mean_kl_error=float(np.max(np.abs(dk.mean(-1)))))
                assert errors['max_old_token_nll_error']<=2e-5 and errors['max_old_token_kl_error']<=2e-5
                assert errors['max_old_mean_nll_error']<=2e-6 and errors['max_old_mean_kl_error']<=2e-6
            for k,v in errors.items():report['qualification'][k]=max(report['qualification'][k],v)
            durations.append(time.time()-started);done+=1
            if j==0 and i==0:
                forecast=time.time()-START+63*durations[0]*1.25+3*loads[0]*1.25+20
                report['qualification'].update(projected_seconds=forecast,first_sequence_seconds=durations[0],first_load_seconds=loads[0])
                save('qualification.json',report['qualification'])
                if forecast>CAP:
                    arrays('partial.npz');save('result.json',{**report,'status':'cost_gate_stop','seconds':time.time()-START})
                    save('status.json',{'status':'cost_gate_stop','seconds':time.time()-START,'projected_seconds':forecast});return
            status(stage='scoring',state=f'{label}_{step}',sequences=i+1,total_sequences=done,score_forwards=done*4)
            del logp,p0,raw,add,delta_mixed
        arrays('partial.npz');save('progress.json',report);del before,after
    assert all(np.isfinite(v).all() for v in [L,KL,parts]);arrays('scalars.npz')
    report.update(status='complete',seconds=time.time()-START,load_seconds=loads,sequence_seconds=durations,score_forwards=done*4)
    save('result.json',report);save('status.json',{'status':'complete','pid':os.getpid(),'seconds':report['seconds'],'score_forwards':done*4})
    print(json.dumps({'status':'complete','seconds':report['seconds'],'score_forwards':done*4}),flush=True)


try:main()
except BaseException as exc:
    save('failure.json',{'status':'failed','seconds':time.time()-START,'error':str(exc),'traceback':traceback.format_exc()})
    save('status.json',{'status':'failed','pid':os.getpid(),'seconds':time.time()-START,'error':str(exc)})
    raise

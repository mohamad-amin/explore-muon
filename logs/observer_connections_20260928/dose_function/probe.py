"""Own-state adjacent-step four-corner predictive amplitudes; CPU only."""
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
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE/'source'))
from adamw_spectra import gn_probe as P
from adamw_spectra.data import TokenStream
torch.set_num_threads(2);torch.set_num_interop_threads(1)
OUT=HERE/'run1';OUT.mkdir(exist_ok=False);START=time.time();CAP=600.
ARMS={
 'q09':'soaudit_dose16m_20260928/PD_a0.25_b16M_lr0.028_mom0.9_s260925_l40s',
 'q08':'soaudit_dose16m_20260928/PD_a0.25_b16M_lr0.028_mom0.8_s260925_l40s',
 'h09':'soaudit_batch16m_20260927/PD_a0.5_b16M_lr0.028_mom0.9_s260925_l40s',
 'h08':'soaudit_mom16m_20260928/PD_a0.5_b16M_lr0.028_mom0.8_s260925_l40s'}
STATES=[(k,s) for s in [9,46] for k in ARMS]
DEFAULTS={'muon_prefilter':'none','head_whitening_alpha':0.,'head_whitening_center':False,
          'head_whitening_norm':'match','muon_momentum_start':-1.,'muon_momentum_warmup':0.}


def save(name,d):
    p=OUT/(name+'.tmp');p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');p.replace(OUT/name)


def status(**kw):
    d={'status':'running','pid':os.getpid(),'seconds':time.time()-START,**kw};save('status.json',d);print(json.dumps(d),flush=True)
    if d['seconds']>CAP:raise TimeoutError('600-second checked CPU boundary')


def model_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()


def canonical(c):
    d=dict(c)
    for k,v in DEFAULTS.items():d.setdefault(k,v)
    d.pop('data_norm_alpha');d.pop('muon_momentum');return d


def main():
    base=ROOT/'logs/muon_spectra';metadata={};traces={};manifest={}
    def read(f):
        b=f.read_bytes();manifest[str(f)]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return json.loads(b)
    for label,arm in ARMS.items():
        p=base/arm/'scientific';metadata[label]=read(p/'metadata.json');assert read(p/'status.json')['status']=='complete'
        traces[label]={}
        for f in sorted((p/'steps').glob('step*.json')):
            d=read(f);traces[label][d['step']]=d
    ref=metadata['q09'];assert ref['config']['learning_rate']==.028 and ref['total_steps']==92
    for m in metadata.values():
        assert canonical(m['config'])==canonical(ref['config'])
        for k in ['initial_model_sha256','train_manifest','validation_manifest','parameter_count','budget_tokens','device_name','world_size']:
            assert m[k]==ref[k],k
    source_checks={}
    for label,arm in ARMS.items():
        p=base/arm/'scientific/source'
        checks={}
        for name,expected in metadata[label]['source_sha256'].items():
            f=p/name
            if f.exists():
                actual=hashlib.sha256(f.read_bytes()).hexdigest();assert actual==expected,(label,name)
                checks[name]=actual
        assert {'model.py','muon.py','train.py','distributed.py','data.py'}<=set(checks)
        source_checks[label]=checks
    validation=[]
    for s in sorted(set.intersection(*[{s for s,d in trace.items() if d.get('validation_nll') is not None} for trace in traces.values()])):
        vals={k:v[s]['validation_nll'] for k,v in traces.items()};dq=vals['q08']-vals['q09'];dh=vals['h08']-vals['h09']
        validation.append({'step':s,'loss':vals,'beta_effect_quarter':dq,'beta_effect_half':dh,'interaction':dh-dq})
    windows=[]
    for first,last in [(4,23),(24,43),(44,63),(64,83),(84,92)]:
        den=sum(traces['q09'][s]['batch_tokens'] for s in range(first,last+1))
        means={k:sum(v[s]['train_nll']*v[s]['batch_tokens'] for s in range(first,last+1))/den for k,v in traces.items()}
        windows.append({'first':first,'last':last,'loss':means,'interaction':means['h08']-means['h09']-means['q08']+means['q09']})
    report={'status':'prepared','arms':ARMS,'states':STATES,'metadata':metadata,'trace_manifest':manifest,'validation':validation,'training_windows':windows,
            'source_checks':source_checks,'checkpoint_provenance':{},'qualification':{},'step_norms':{},
            'helper_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (HERE/'source/adamw_spectra').glob('*.py')},
            'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    stream=TokenStream(str(ROOT/ref['config']['train_pattern']));xs=[];ys=[]
    for offset in [3_180_000_000,3_190_000_000]:
        x,y=stream.batch(offset,4,512,'cpu');xs.append(x);ys.append(y)
    x=torch.cat(xs);y=torch.cat(ys);np.savez(OUT/'tokens.npz',inputs=x.numpy(),targets=y.numpy())
    report['score_offsets']=[3_180_000_000,3_190_000_000]
    report['token_hashes']={k:hashlib.sha256(v.numpy().tobytes()).hexdigest() for k,v in [('inputs',x),('targets',y)]}
    save('prepared.json',report)
    loss64=np.full((8,8,4,512),np.nan);loss32=np.full_like(loss64,np.nan,dtype=np.float32)
    klf=np.full((8,8,3,512),np.nan);klr=np.full_like(klf,np.nan);entropy=np.full((8,8,512),np.nan)
    model=None;loads=[];sequence_seconds=[];done=0
    def arrays(name):np.savez(OUT/name,loss64=loss64,loss32=loss32,kl_forward=klf,kl_reverse=klr,base_entropy=entropy)
    for j,(label,step) in enumerate(STATES):
        load_start=time.time();kept=base/ARMS[label]/'scientific/kept'
        paths=[kept/f'step{step:06d}.pt',kept/f'step{step+1:06d}_weights.pt']
        before,after=[torch.load(p,map_location='cpu',weights_only=False,mmap=True) for p in paths]
        assert before['step']==step and after['step']==step+1
        assert before['tokens']<min(report['score_offsets'])
        if model is None:model=P.build_model(before['config']['model'],before['model'],torch.device('cpu'))
        else:model.load_state_dict(before['model'])
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        named=dict(model.named_parameters());body={k for k,p in named.items() if k.startswith('blocks.') and p.ndim==2}
        assert len(body)==48 and len(named)==84 and set(named)==set(before['model'])==set(after['model'])
        assert sum(p.numel() for p in named.values())==ref['parameter_count']
        body2=aux2=0.
        for k in named:
            q=float((after['model'][k].double()-before['model'][k].double()).square().sum())
            if k in body:body2+=q
            else:aux2+=q
        report['step_norms'][f'{label}_{step}']={'body':body2**.5,'aux':aux2**.5,'full':(body2+aux2)**.5,'nominal_body_before_decay':192*traces[label][step+1]['lr']}
        report['checkpoint_provenance'][f'{label}_{step}']=[{'path':str(p),'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns,
            'model_tensor_sha256':model_hash(d['model'])} for p,d in zip(paths,[before,after])]
        def corner(mask,verify=False):
            with torch.no_grad():
                for k,p in named.items():
                    use=bool(mask&(1 if k in body else 2));target=(after if use else before)['model'][k]
                    p.copy_(target)
                    if verify:assert torch.equal(p,target)
        loads.append(time.time()-load_start)
        for i in range(8):
            t=time.time();base_logp=base_p=base_logits=None
            for mask in range(4):
                corner(mask,verify=(j==0 and i==0))
                with torch.inference_mode():
                    z=model(x[i:i+1])[0]
                    assert torch.isfinite(z).all()
                    lp=torch.log_softmax(z.double(),-1)
                    nll=-lp.gather(-1,y[i,:,None]).squeeze(-1)
                    nll32=F.cross_entropy(z,y[i],reduction='none')
                    loss64[j,i,mask]=nll.numpy();loss32[j,i,mask]=nll32.numpy()
                    if mask==0:
                        base_logp=lp;base_p=lp.exp();entropy[j,i]=-(base_p*lp).sum(-1).numpy()
                        if j==0 and i==0:base_logits=z.clone()
                    else:
                        f=(base_p*(base_logp-lp)).sum(-1);r=(lp.exp()*(lp-base_logp)).sum(-1)
                        assert torch.isfinite(f).all() and torch.isfinite(r).all()
                        assert min(float(f.min()),float(r.min()))>=-1e-10
                        klf[j,i,mask-1]=f.numpy();klr[j,i,mask-1]=r.numpy()
            if j==0 and i==0:
                corner(0,verify=True)
                with torch.inference_mode():restored=model(x[i:i+1])[0]
                zerr=float((restored-base_logits).abs().max());cerr=float(np.max(np.abs(loss64[j,i]-loss32[j,i])))
                assert zerr<=2e-6 and cerr<=5e-5
                null=float((base_p*(base_logp-base_logp)).sum());assert null==0.
                report['qualification'].update(restored_logits_max_error=zerr,fp64_fp32_ce_max_error=cerr,null_kl=null,corner_tensor_equality=True)
            sequence_seconds.append(time.time()-t);done+=1
            if j==0 and i==0:
                forecast=time.time()-START+63*sequence_seconds[0]*1.25+7*loads[0]*1.25+20
                report['qualification'].update(projected_seconds=forecast,first_sequence_seconds=sequence_seconds[0],first_load_seconds=loads[0])
                save('qualification.json',report['qualification'])
                if forecast>CAP:
                    arrays('partial_scalar_tokens.npz');save('result.json',{**report,'status':'cost_gate_stop','seconds':time.time()-START})
                    save('status.json',{'status':'cost_gate_stop','seconds':time.time()-START,'projected_seconds':forecast});return
            status(stage='scoring',state=f'{label}_{step}',sequences=i+1,total_sequences=done,score_forwards=done*4)
        arrays('partial_scalar_tokens.npz');save('progress.json',report)
        del before,after
    assert all(np.isfinite(a).all() for a in [loss64,loss32,klf,klr,entropy])
    arrays('scalar_tokens.npz');report.update(status='complete',seconds=time.time()-START,load_seconds=loads,sequence_seconds=sequence_seconds,score_forwards=done*4)
    save('result.json',report);save('status.json',{'status':'complete','pid':os.getpid(),'seconds':report['seconds'],'score_forwards':done*4})
    print(json.dumps({'status':'complete','seconds':report['seconds'],'score_forwards':done*4}),flush=True)


try:main()
except BaseException as exc:
    save('failure.json',{'status':'failed','error':str(exc),'traceback':traceback.format_exc(),'seconds':time.time()-START})
    save('status.json',{'status':'failed','pid':os.getpid(),'seconds':time.time()-START,'error':str(exc)})
    raise

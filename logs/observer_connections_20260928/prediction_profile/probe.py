"""Frozen ten-checkpoint forward-only profile, CPU with a hard cost gate."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(name,'2')
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE/'source'))
from adamw_spectra import gn_probe as P
from adamw_spectra.data import TokenStream

torch.set_num_threads(2);torch.set_num_interop_threads(1)
OUT=HERE/'run1';OUT.mkdir(exist_ok=False)
START=time.time()
ARMS={'M':'M_lr0.007_s260925_l40s','PD':'PD_a0.25_lr0.01_s260925_ada',
      'S':'S_lr0.007_s260925_l40s','SPD':'SPD_a0.25_lr0.01_s260925_ada'}
STATES=[('M',200),('M',500),('M',900),('M',1300),('PD',500),('PD',900),('S',500),('S',900),('SPD',500),('SPD',900)]
OFFSETS=[3_120_000_000,3_160_000_000]
SEQ=512;N=16;CAP=600.


def save(name,data):
    tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(OUT/name)


def status(**kw):
    d={'status':'running','pid':os.getpid(),'seconds':time.time()-START,**kw};save('status.json',d);print(json.dumps(d),flush=True)
    if d['seconds']>CAP:raise TimeoutError('Declared600-second CPU wall boundary exceeded')


def tensor_hash(state):
    h=hashlib.sha256()
    for k,v in sorted(state.items()):
        h.update(k.encode());h.update(str(tuple(v.shape)).encode());h.update(v.numpy().tobytes())
    return h.hexdigest()


def main():
    base=ROOT/'logs/muon_spectra/soaudit_traj_20260926'
    metadata={};cal_losses={};source_records={}
    for method,arm in ARMS.items():
        p=base/arm/'scientific';metadata[method]=json.loads((p/'metadata.json').read_text());cal_losses[method]={}
        for step in sorted({s for m,s in STATES if m==method}):
            f=p/'steps'/f'step{step:06d}.json';raw=f.read_bytes();record=json.loads(raw)
            assert record.get('validation_nll') is not None
            cal_losses[method][step]=record['validation_nll']
            source_records[str(f)]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    ref=metadata['M']
    for method,meta in metadata.items():
        for key in ['initial_model_sha256','train_manifest','validation_manifest','parameter_count','budget_tokens']:
            assert meta[key]==ref[key],(method,key)
        # Statistic collection does not alter this architecture or its evaluation forward.
        a={k:v for k,v in meta['config']['model'].items() if not k.startswith('track_')}
        b={k:v for k,v in ref['config']['model'].items() if not k.startswith('track_')}
        assert a==b and a['seq_len']==SEQ
    calibrations=[]
    for method in ['PD','S','SPD']:
        for step,left,right in [(500,500,900),(900,900,1300)]:
            t=(cal_losses['M'][left]-cal_losses[method][step])/(cal_losses['M'][left]-cal_losses['M'][right]);assert 0<=t<=1
            calibrations.append({'label':f'{method}{step}','target':[method,step],'left':['M',left],'right':['M',right],'weight':t,'role':'method'})
    for step,left,right in [(500,200,900),(900,500,1300)]:
        t=(cal_losses['M'][left]-cal_losses['M'][step])/(cal_losses['M'][left]-cal_losses['M'][right]);assert 0<=t<=1
        calibrations.append({'label':f'M{step}_control','target':['M',step],'left':['M',left],'right':['M',right],'weight':t,'role':'interpolation_control'})
    report={'status':'prepared','states':STATES,'score_offsets':OFFSETS,'calibration_losses':cal_losses,'calibrations':calibrations,
            'calibration_sources':source_records,'metadata':metadata,
            'source_hashes':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (HERE/'source/adamw_spectra').glob('*.py')},
            'probe_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'checkpoint_provenance':{},'qualification':{}}
    save('prepared.json',report)
    stream=TokenStream(str(ROOT/ref['config']['train_pattern']))
    xs=[];ys=[]
    for offset in OFFSETS:
        x,y=stream.batch(offset,N,SEQ,'cpu');xs.append(x);ys.append(y)
    x=torch.cat(xs);y=torch.cat(ys)
    assert x.shape==y.shape==(32,512)
    counts=np.zeros(ref['config']['model']['vocab_size'],dtype=np.int64)
    for first in range(0,50_000_000,1_000_000):
        counts+=np.bincount(stream.read(2_500_000_000+first,1_000_000),minlength=len(counts))
    assert int(counts.sum())==50_000_000
    freq=counts/counts.sum();edges=np.array([0,1e-6,1e-5,1e-4,1e-3,1e-2,1.])
    bins=np.digitize(freq[y.numpy()],edges)-1;assert bins.min()>=0 and bins.max()<6
    np.savez(OUT/'tokens_frequency.npz',inputs=x.numpy(),targets=y.numpy(),frequency_counts=counts,edges=edges,bins=bins)
    report['data_hashes']={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in [('inputs',x.numpy()),('targets',y.numpy()),('frequency_counts',counts)]}
    report['score_bin_counts']=np.bincount(bins.reshape(-1),minlength=6).tolist()
    assert sum(report['score_bin_counts'])==x.numel()
    save('prepared.json',report)
    model=None;nll=np.full((len(STATES),32,SEQ),np.nan,dtype=np.float32)
    load_seconds=[];forward_seconds=[];evaluated=0
    def evaluate(i):
        start=time.time()
        with torch.inference_mode():loss=P.token_losses(model(x[i:i+1]),y[i:i+1])[0]
        assert loss.shape==(SEQ,) and torch.isfinite(loss).all()
        forward_seconds.append(time.time()-start)
        return loss.numpy().copy()
    for j,(method,step) in enumerate(STATES):
        start=time.time();arm=base/ARMS[method]
        path=arm/'scientific/kept'/f'step{step:06d}.pt'
        saved=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        assert saved['step']==step and saved['tokens']<min(OFFSETS)
        if model is None:model=P.build_model(saved['config']['model'],saved['model'],torch.device('cpu'))
        else:model.load_state_dict(saved['model'])
        model.eval()
        for p in model.parameters():p.requires_grad_(False)
        assert sum(p.numel() for p in model.parameters())==ref['parameter_count']
        report['checkpoint_provenance'][f'{method}{step}']={'path':str(path),'bytes':path.stat().st_size,'mtime_ns':path.stat().st_mtime_ns,
            'model_tensor_sha256':tensor_hash(saved['model']),'tokens':saved['tokens'],'step':saved['step']}
        del saved
        load_seconds.append(time.time()-start)
        for i in range(32):
            nll[j,i]=evaluate(i);evaluated+=1
            if j==0 and i==0:
                again=evaluate(i)
                with torch.inference_mode():scalar=float(model(x[i:i+1],y[i:i+1]))
                repeat_error=float(np.max(np.abs(again-nll[j,i])));scalar_error=abs(scalar-float(nll[j,i].mean()))
                assert repeat_error<=2e-6 and scalar_error<=2e-6
                report['qualification'].update(repeat_max_error=repeat_error,scalar_ce_error=scalar_error)
            if j==0 and i==1:
                forecast=time.time()-START+318*max(forward_seconds)*1.25+9*load_seconds[0]*1.25+20
                report['qualification'].update(projected_seconds=forecast,first_load_seconds=load_seconds[0],pilot_forward_seconds=forward_seconds.copy())
                save('qualification.json',report['qualification'])
                status(stage='cost_gate',projected_seconds=forecast,score_forwards=evaluated)
                if forecast>CAP:
                    np.savez(OUT/'partial_nll.npz',nll=nll)
                    save('result.json',{**report,'status':'cost_gate_stop','seconds':time.time()-START})
                    save('status.json',{'status':'cost_gate_stop','seconds':time.time()-START,'projected_seconds':forecast})
                    return
            if (i+1)%8==0:status(stage='scoring',state=f'{method}{step}',sequences=i+1,score_forwards=evaluated)
        np.savez(OUT/'partial_nll.npz',nll=nll)
        save('progress.json',report)
    assert np.isfinite(nll).all()
    np.savez(OUT/'per_token_nll.npz',nll=nll)
    report.update(status='complete',seconds=time.time()-START,load_seconds=load_seconds,forward_seconds=forward_seconds,score_forwards=evaluated)
    save('result.json',report);save('status.json',{'status':'complete','pid':os.getpid(),'seconds':report['seconds'],'score_forwards':evaluated})
    print(json.dumps({'status':'complete','seconds':report['seconds'],'score_forwards':evaluated}),flush=True)


try:main()
except BaseException as exc:
    save('failure.json',{'status':'failed','type':type(exc).__name__,'error':str(exc),'traceback':traceback.format_exc(),'seconds':time.time()-START})
    save('status.json',{'status':'failed','pid':os.getpid(),'seconds':time.time()-START,'error':str(exc)})
    raise

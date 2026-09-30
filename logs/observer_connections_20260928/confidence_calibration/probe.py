"""Six fixed endpoints and a fixed temperature family; CPU scoring only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='2'
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE/'source'))
from adamw_spectra.gn_probe import build_model
from adamw_spectra.data import TokenStream
torch.set_num_threads(2)
torch.set_num_interop_threads(1)
OUT=HERE/'run1'
CAP=600.
OFFSETS=[3_140_131_072,3_150_131_072]
SCALES=[.8,.9,1.,1.1,1.2]
ARMS={
 'h1_b08':'soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada',
 'h1_b09':'soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada',
 'h1_warm':'soaudit_momwarm16m_20260928/SPD_a0.5_b16M_lr0.028_momwarm0.8to0.9_s260925_ada',
 'h2_b08':'soaudit_horizon16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.8_s260925_ada',
 'h2_b09':'soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada',
 'h2_warm':'soaudit_momwarm16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada'}
DEFAULTS={'muon_prefilter':'none','head_whitening_alpha':0.,'head_whitening_center':False,
          'head_whitening_norm':'match','muon_momentum_start':-1.,'muon_momentum_warmup':0.}
_partial_writer=None
_started=None


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def save(name,data):
    tmp=OUT/(name+'.tmp')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
    tmp.replace(OUT/name)


def model_hash(state):
    h=hashlib.sha256()
    for key,tensor in sorted(state.items()):
        h.update(key.encode());h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.contiguous().numpy().tobytes())
    return h.hexdigest()


def canonical(config):
    config=dict(config)
    for key,value in DEFAULTS.items():config.setdefault(key,value)
    for key in ('muon_momentum','muon_momentum_start','muon_momentum_warmup',
                'keep_checkpoints','total_tokens'):
        config.pop(key)
    return config


def scores(logits,targets):
    z=logits.double()
    zy=z.gather(-1,targets[:,None]).squeeze(-1)
    lse=torch.logsumexp(z,-1)
    base=lse-zy
    lp=z-lse[:,None]
    p=lp.exp()
    entropy=-(p*lp).sum(-1)
    ez=(p*z).sum(-1)
    derivative=ez-zy
    curvature=(p*(z-ez[:,None]).square()).sum(-1)
    assert torch.isfinite(curvature).all() and float(curvature.min())>=0.
    identity=float((derivative-(base-entropy)).abs().max())
    assert identity<1e-10,identity
    ce32=F.cross_entropy(logits,targets,reduction='none')
    precision=float((base-ce32.double()).abs().max())
    assert precision<5e-5,precision
    losses=[]
    for scale in SCALES:
        loss=base if scale==1. else torch.logsumexp(scale*z,-1)-scale*zy
        assert torch.isfinite(loss).all() and float(loss.min())>=-1e-10
        losses.append(loss.numpy())
    return np.stack(losses),entropy.numpy(),derivative.numpy(),curvature.numpy(),identity,precision


def synthetic_checks():
    z=torch.tensor([[1.2,-.7,2.1,.3],[-1.,.1,.9,1.4]],dtype=torch.float64)
    y=torch.tensor([0,3]);zy=z.gather(-1,y[:,None]).squeeze(-1)
    def loss(a):return torch.logsumexp(a*z,-1)-a*zy
    p=torch.softmax(z,-1);g=(p*z).sum(-1)-zy
    eps=1e-5
    finite=(loss(1+eps)-loss(1-eps))/(2*eps)
    error=float((g-finite).abs().max());assert error<1e-8
    shifted=z+torch.tensor([[31.],[-17.]],dtype=torch.float64)
    shift_error=0.
    for a in SCALES:
        shifted_loss=torch.logsumexp(a*shifted,-1)-a*shifted.gather(-1,y[:,None]).squeeze(-1)
        shift_error=max(shift_error,float((loss(a)-shifted_loss).abs().max()))
    assert shift_error<1e-12
    return dict(derivative_finite_difference_error=error,common_shift_error=shift_error)


def main():
    global _partial_writer,_started
    OUT.mkdir(exist_ok=False)
    shutil.copyfile(__file__,OUT/'executed_probe.py')
    shutil.copyfile(HERE/'PROTOCOL.md',OUT/'executed_protocol.md')
    start=time.monotonic()
    _started=start
    manifest={};metadata={};original_scores={};source_checks={}
    def read(path):
        raw=path.read_bytes()
        manifest[str(path.relative_to(ROOT))]=dict(bytes=len(raw),sha256=digest(raw))
        return json.loads(raw)
    for label,arm in ARMS.items():
        p=ROOT/'logs/muon_spectra'/arm/'scientific'
        meta=read(p/'metadata.json');metadata[label]=meta
        status=read(p/'status.json');sidecar=read(p/'checkpoint.json')
        horizon=int(label[1]);step=92*horizon
        assert status['status']=='complete' and status['step']==sidecar['step']==step
        assert status['tokens']==sidecar['tokens']==meta['budget_tokens']==horizon*1539870720
        assert meta['total_steps']==step
        row=read(p/'steps'/f'step{step:06d}.json')
        assert row['step']==step and row['tokens']==meta['budget_tokens']
        original_scores[label]=row['validation_nll']
        c=meta['config']
        assert c['learning_rate']==.028 and c['data_norm_alpha']==.5 and c['soap_precondition']
        assert c['batch_tokens']==16777216 and c['seed']==260925
        if label.endswith('warm'):
            assert c['muon_momentum']==.9 and c['muon_momentum_start']==.8 and c['muon_momentum_warmup']==.5
        else:
            assert c['muon_momentum']==(.8 if label.endswith('b08') else .9)
            assert c.get('muon_momentum_start',-1.)==-1.
        checks={}
        for name,expected in meta['source_sha256'].items():
            source=p/'source'/name
            assert source.exists(),source
            actual=digest(source.read_bytes());assert actual==expected,(label,name)
            checks[name]=actual
        assert {'model.py','data.py','muon.py','data_norm_muon.py','distributed.py','train.py'}<=set(checks)
        source_checks[label]=checks
    ref=metadata['h1_b08']
    for m in metadata.values():
        assert canonical(m['config'])==canonical(ref['config'])
        for key in ('initial_model_sha256','train_manifest','validation_manifest','parameter_count',
                    'device_name','world_size','effective_precision','effective_compile','parameter_optimizers'):
            assert m[key]==ref[key],key
    helpers={}
    for p in (HERE/'source/adamw_spectra').glob('*.py'):
        assert p.read_bytes()==(HERE.parent/'dose_function/source/adamw_spectra'/p.name).read_bytes()
        helpers[p.name]=digest(p.read_bytes())
    stream=TokenStream(str(ROOT/ref['config']['train_pattern']))
    assert stream.manifest==ref['train_manifest']
    assert stream.total==3_200_000_000
    assert min(OFFSETS)>max(m['budget_tokens'] for m in metadata.values())
    assert OFFSETS[0]+8193<=OFFSETS[1] and OFFSETS[-1]+8193<=stream.total
    # Previous observer scoring extends at most 50M positions from 2.5B;
    # all later declared banks are explicitly checked here with a conservative span.
    previous=[(2_500_000_000,50_000_000),(2_600_000_000,200_000),
              (2_650_000_000,512),(2_700_000_000,4096),(2_710_000_000,4096),
              (2_720_000_000,4096),(2_730_000_000,4096),(2_800_098_304,2048),
              (2_900_098_304,2048),(3_120_000_000,8192),(3_160_000_000,8192),
              (3_170_131_072,2048),(3_180_000_000,2048),
              (3_190_000_000,2048),(3_195_131_072,2048)]
    for offset in OFFSETS:
        for old,count in previous:assert offset+8193<=old or old+count+1<=offset
    xs=[];ys=[]
    for offset in OFFSETS:
        x,y=stream.batch(offset,16,512,'cpu');xs.append(x);ys.append(y)
    x=torch.cat(xs);y=torch.cat(ys)
    assert x.shape==y.shape==(32,512)
    np.savez(OUT/'tokens.npz',inputs=x.numpy(),targets=y.numpy())
    labels=list(ARMS)
    report=dict(status='prepared',labels=labels,arms=ARMS,scales=SCALES,offsets=OFFSETS,
                metadata=metadata,original_validation_nll=original_scores,manifest=manifest,
                source_checks=source_checks,helper_hashes=helpers,
                token_hashes={k:digest(v.numpy().tobytes()) for k,v in [('inputs',x),('targets',y)]},
                qualification=synthetic_checks(),checkpoint_provenance={},load_seconds=[],
                probe_sha256=digest(Path(__file__).read_bytes()),
                protocol_sha256=digest((HERE/'PROTOCOL.md').read_bytes()))
    save('prepared.json',report)
    nll=np.full((6,32,5,512),np.nan)
    entropy=np.full((6,32,512),np.nan)
    derivative=np.full_like(entropy,np.nan);curvature=np.full_like(entropy,np.nan)
    def arrays(name):
        np.savez(OUT/name,nll=nll,entropy=entropy,derivative=derivative,curvature=curvature)
    _partial_writer=arrays
    model=None;done=0;extra=0;max_identity=0.;max_precision=0.;sequence_times=[]
    for j,label in enumerate(labels):
        load_start=time.monotonic()
        p=ROOT/'logs/muon_spectra'/ARMS[label]/'scientific/checkpoint.pt'
        fileinfo=(p.stat().st_size,p.stat().st_mtime_ns)
        cp=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
        m=metadata[label]
        assert cp['step']==m['total_steps'] and cp['tokens']==m['budget_tokens']
        assert cp['config']==m['config']
        assert len(cp['model'])==84
        h=model_hash(cp['model'])
        if model is None:model=build_model(cp['config']['model'],cp['model'],torch.device('cpu'))
        else:model.load_state_dict(cp['model'],strict=True)
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        assert len(dict(model.named_parameters()))==84
        assert sum(v.numel() for v in model.parameters())==m['parameter_count']
        assert model_hash(model.state_dict())==h
        report['checkpoint_provenance'][label]=dict(path=str(p.relative_to(ROOT)),bytes=fileinfo[0],
                mtime_ns=fileinfo[1],model_tensor_sha256=h,step=cp['step'],tokens=cp['tokens'])
        del cp
        report['load_seconds'].append(time.monotonic()-load_start)
        for i in range(32):
            tick=time.monotonic()
            with torch.inference_mode():
                logits=model(x[i:i+1])[0]
                assert logits.shape==(512,50304) and torch.isfinite(logits).all()
                loss,en,g,var,ie,pe=scores(logits,y[i])
                nll[j,i]=loss;entropy[j,i]=en;derivative[j,i]=g;curvature[j,i]=var
                max_identity=max(max_identity,ie);max_precision=max(max_precision,pe)
            seconds=time.monotonic()-tick
            sequence_times.append(seconds);done+=1
            if j==0 and i==0:
                repeat_start=time.monotonic()
                with torch.inference_mode():repeat=model(x[0:1])[0]
                error=float((repeat-logits).abs().max());assert error==0.,error
                extra+=1
                report['qualification'].update(repeat_logits_error=error,
                     repeat_forward_seconds=time.monotonic()-repeat_start,first_sequence_seconds=seconds)
                del repeat
                forecast=time.monotonic()-start+191*seconds*1.25+5*report['load_seconds'][0]*1.25+20.
                report['qualification']['forecast_seconds']=forecast
                save('qualification.json',report['qualification'])
                if forecast>CAP:
                    arrays('partial.npz');save('result.json',{**report,'status':'cost_gate_stop'})
                    raise TimeoutError(f'Forecast {forecast:.2f}s exceeds {CAP}s')
            del logits
            elapsed=time.monotonic()-start
            if elapsed>=CAP:
                raise TimeoutError(f'Wall boundary {elapsed:.2f}s exceeds {CAP}s')
            if done%8==0:
                arrays('partial.npz')
                save('status.json',dict(status='running',model=label,completed_forwards=done,
                     qualification_forwards=extra,seconds=elapsed,pid=os.getpid()))
                print(json.dumps(dict(model=label,completed_forwards=done,seconds=elapsed)),flush=True)
        assert (p.stat().st_size,p.stat().st_mtime_ns)==fileinfo
        arrays('partial.npz')
        save('progress.json',dict(status='running',completed_models=j+1,scientific_forwards=done,
                                 seconds=time.monotonic()-start))
    assert done==192 and extra==1
    assert all(np.isfinite(a).all() for a in (nll,entropy,derivative,curvature))
    arrays('per_token.npz')
    report['qualification'].update(max_scale_derivative_identity_error=max_identity,
                                  max_fp64_fp32_scale_one_nll_error=max_precision)
    report.update(status='complete',seconds=time.monotonic()-start,scientific_forwards=done,
                  qualification_forwards=extra,sequence_seconds=sequence_times)
    save('result.json',report)
    save('status.json',dict(status='complete',seconds=report['seconds'],scientific_forwards=done,
                           qualification_forwards=extra,pid=os.getpid()))
    print(json.dumps(dict(status='complete',seconds=report['seconds'],qualification=report['qualification'])),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            failure=traceback.format_exc()
            partial_error=None
            if _partial_writer is not None:
                try:_partial_writer('partial.npz')
                except Exception:partial_error=traceback.format_exc()
            terminal='cost_gate_stop' if isinstance(exc,TimeoutError) else 'failed'
            save('failure.json',dict(status=terminal,traceback=failure,partial_write_error=partial_error))
            save('status.json',dict(status=terminal,pid=os.getpid(),
                 seconds=time.monotonic()-_started if _started is not None else None,
                 reason=str(exc)))
        raise

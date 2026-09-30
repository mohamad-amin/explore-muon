"""Four fixed actual-update Q/K/rest corners; finite predictions, CPU only."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import ast
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

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE/'source'))
from adamw_spectra.gn_probe import build_model
from adamw_spectra.data import TokenStream
torch.set_num_threads(2);torch.set_num_interop_threads(1)
OUT=HERE/'run1';CAP=600.;STEP=46
OFFSETS=[3_100_131_072,3_110_131_072]
LABELS=['e028_b09','e028_b08','e040_b09','e040_b08']
PARTS=['interaction','overlap','mixed_loss','J_direct','KL_add','base_entropy']
_started=None;_partial_writer=None


def sha(raw):return hashlib.sha256(raw).hexdigest()


def save(name,data):
    p=OUT/(name+'.tmp');p.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');p.replace(OUT/name)


def state_hash(state):
    h=hashlib.sha256()
    for name,t in sorted(state.items()):
        h.update(name.encode());h.update(str(tuple(t.shape)).encode());h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


def readout(lp,y):
    loss=torch.stack([-v.gather(-1,y[:,None]).squeeze(-1) for v in lp])
    p0=lp[0].exp()
    responses=[lp[j]-lp[0] for j in (1,2,3)]
    means=[(p0*v).sum(-1) for v in responses]
    kl=torch.stack([-v for v in means])
    linear=torch.stack([m-v.gather(-1,y[:,None]).squeeze(-1) for m,v in zip(means,responses)])
    centered=[v-m[:,None] for v,m in zip(responses,means)]
    gram=torch.empty((y.numel(),3,3),dtype=torch.float64)
    for i in range(3):
        for j in range(i,3):
            value=(p0*centered[i]*centered[j]).sum(-1)
            gram[:,i,j]=value;gram[:,j,i]=value
    conditional=torch.stack([(lp[j].exp()*(lp[j]-lp[3])).sum(-1) for j in (2,1)])
    raw_add=lp[1]+lp[2]-lp[0]
    overlap=torch.logsumexp(raw_add,-1)
    lp_add=raw_add-overlap[:,None]
    loss_add=-lp_add.gather(-1,y[:,None]).squeeze(-1)
    interaction=loss[3]-loss[1]-loss[2]+loss[0]
    mixed_loss=loss[3]-loss_add
    mixed=responses[2]-responses[0]-responses[1]
    J=(p0*mixed).sum(-1)-mixed.gather(-1,y[:,None]).squeeze(-1)
    kl_add=(p0*(lp[0]-lp_add)).sum(-1)
    entropy=-(p0*lp[0]).sum(-1)
    parts=torch.stack([interaction,overlap,mixed_loss,J,kl_add,entropy])
    eigenvalues=torch.linalg.eigvalsh(gram)
    scale=gram.diagonal(dim1=1,dim2=2).max(-1).values.clamp_min(1.)
    direct_mixed=(p0*(centered[2]-centered[0]-centered[1]).square()).sum(-1)
    coeff=torch.tensor([-1.,-1.,1.],dtype=torch.float64)
    from_gram=torch.einsum('i,tij,j->t',coeff,gram,coeff)
    direct_add=(p0*(centered[0]+centered[1]).square()).sum(-1)
    add_gram=gram[:,0,0]+gram[:,1,1]+2*gram[:,0,1]
    checks=dict(max_ce_kl_error=float((loss[1:]-loss[0]-linear-kl).abs().max()),
                max_split_error=float((interaction-overlap-mixed_loss).abs().max()),
                max_mixed_identity_error=float((mixed_loss-J-kl[2]+kl_add).abs().max()),
                max_J_identity_error=float((J-interaction+kl[2]-kl[0]-kl[1]).abs().max()),
                max_mixed_covariance_error=float(((from_gram-direct_mixed).abs()/scale).max()),
                max_additive_covariance_error=float(((add_gram-direct_add).abs()/scale).max()),
                min_relative_gram_eigenvalue=float((eigenvalues[:,0]/scale).min()),
                min_kl=float(torch.cat([kl.flatten(),conditional.flatten(),kl_add.flatten()]).min()))
    assert all(torch.isfinite(v).all() for v in (loss,kl,conditional,gram,parts,linear))
    assert all(v<=1e-10 for k,v in checks.items() if k.startswith('max_')),checks
    assert checks['min_relative_gram_eigenvalue']>=-1e-10 and checks['min_kl']>=-1e-10,checks
    return loss,kl,conditional,gram,parts,linear,checks


def synthetic():
    z=torch.tensor([[[1.,-.4,2.,.2],[.5,1.2,-.2,-.1]],
                    [[1.1,-.7,2.2,.2],[.7,1.,-.2,-.3]],
                    [[.9,-.3,2.1,.5],[.3,1.3,.2,-.1]],
                    [[1.2,-.4,2.4,.5],[.6,1.2,.3,-.2]]],dtype=torch.float64)
    y=torch.tensor([0,1]);lp=[torch.log_softmax(v,-1) for v in z]
    out=readout(lp,y)
    shifts=torch.tensor([[13.,-21.],[-7.,18.],[22.,1.],[-19.,7.]])[:,:,None]
    other=readout([torch.log_softmax(v,-1) for v in z+shifts],y)
    error=max(float((a-b).abs().max()) for a,b in zip(out[:6],other[:6]))
    assert error<1e-12,error
    return dict(max_synthetic_shift_error=error,synthetic_identities=out[-1])


def main():
    global _started,_partial_writer
    OUT.mkdir(exist_ok=False);_started=time.monotonic()
    shutil.copyfile(__file__,OUT/'executed_probe.py');shutil.copyfile(HERE/'PROTOCOL.md',OUT/'executed_protocol.md')
    tree=ast.parse((HERE.parent/'momentum_lr/analyze.py').read_text())
    arms=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='ARMS' for t in n.targets))
    assert list(arms)==[(.028,.9),(.028,.8),(.04,.9),(.04,.8)]
    prior=json.loads((HERE.parent/'momentum_lr/result.json').read_text())
    angular=json.loads((HERE.parent/'angular_clock/run1/result.json').read_text())
    ang_by_key={(p['lr'],p['beta'],p['step']):p for p in angular['provenance']}
    metadata={};manifest={};sources={};traces={}
    def read(path):
        b=path.read_bytes();manifest[str(path.relative_to(ROOT))]=dict(bytes=len(b),sha256=sha(b));return json.loads(b)
    for label,(key,arm) in zip(LABELS,arms.items()):
        p=ROOT/'logs/muon_spectra'/arm/'scientific'
        meta=read(p/'metadata.json');assert meta==prior['metadata'][str(key)]
        assert read(p/'status.json')['status']=='complete'
        assert meta['total_steps']==92 and meta['config']['model']['qk_norm'] and not meta['config']['model']['bias']
        metadata[label]=meta;traces[label]=read(p/'steps'/f'step{STEP+1:06d}.json')
        source={}
        for name,expected in meta['source_sha256'].items():
            f=p/'source'/name;assert f.exists();actual=sha(f.read_bytes());assert actual==expected
            source[name]=actual
        assert {'model.py','muon.py','data_norm_muon.py','distributed.py','train.py','data.py'}<=set(source)
        sources[label]=source
    ref=metadata[LABELS[0]]
    helper_hashes={}
    for p in (HERE/'source/adamw_spectra').glob('*.py'):
        assert p.read_bytes()==(HERE.parent/'confidence_calibration/source/adamw_spectra'/p.name).read_bytes()
        helper_hashes[p.name]=sha(p.read_bytes())
    stream=TokenStream(str(ROOT/ref['config']['train_pattern']))
    assert stream.total==3200000000 and stream.manifest==ref['train_manifest']
    assert min(OFFSETS)>max(m['budget_tokens'] for m in metadata.values())
    assert OFFSETS[0]+4097<OFFSETS[1] and OFFSETS[-1]+4097<=stream.total
    # All prior observer score intervals above 3B; older panels lie below 3B.
    older=[(3120000000,8193),(3160000000,8193),(3170131072,2049),(3180000000,2049),
           (3190000000,2049),(3195131072,2049),(3140131072,8193),(3150131072,8193)]
    for offset in OFFSETS:
        for old,count in older:assert offset+4097<=old or old+count<=offset
    xs=[];ys=[]
    for offset in OFFSETS:
        x,y=stream.batch(offset,8,512,'cpu');xs.append(x);ys.append(y)
    x=torch.cat(xs);y=torch.cat(ys);assert x.shape==y.shape==(16,512)
    np.savez(OUT/'tokens.npz',inputs=x.numpy(),targets=y.numpy())
    arrays=dict(corner_nll=np.full((4,16,4,512),np.nan),forward_kl=np.full((4,16,3,512),np.nan),
                conditional_kl=np.full((4,16,2,512),np.nan),finite_gram=np.full((4,16,512,3,3),np.nan),
                parts=np.full((4,16,6,512),np.nan),linear=np.full((4,16,3,512),np.nan))
    def persist(name):np.savez(OUT/name,**arrays)
    _partial_writer=persist
    report=dict(status='prepared',labels=LABELS,parts=PARTS,arms={label:arm for label,arm in zip(LABELS,arms.values())},
                step=STEP,next_step=STEP+1,offsets=OFFSETS,metadata=metadata,source_checks=sources,manifest=manifest,
                helper_hashes=helper_hashes,token_hashes={k:sha(v.numpy().tobytes()) for k,v in [('inputs',x),('targets',y)]},
                probe_sha256=sha(Path(__file__).read_bytes()),protocol_sha256=sha((HERE/'PROTOCOL.md').read_bytes()),
                qualification=synthetic(),checkpoint_provenance={},parameter_norms={},load_seconds=[])
    save('prepared.json',report)
    model=None;done=0;extra=0;times=[];max_checks={};min_checks={}
    for j,(label,(key,arm)) in enumerate(zip(LABELS,arms.items())):
        tick=time.monotonic();root=ROOT/'logs/muon_spectra'/arm/'scientific/kept'
        paths=[root/f'step{STEP:06d}.pt',root/f'step{STEP+1:06d}_weights.pt']
        fileinfo=[(p.stat().st_size,p.stat().st_mtime_ns) for p in paths]
        before,after=[torch.load(p,map_location='cpu',weights_only=False,mmap=True) for p in paths]
        assert before['step']==STEP and after['step']==STEP+1
        assert before['tokens']==STEP*16777216 and after['tokens']==(STEP+1)*16777216
        assert before['config']==metadata[label]['config']
        if model is None:model=build_model(before['config']['model'],before['model'],torch.device('cpu'))
        else:model.load_state_dict(before['model'],strict=True)
        model.eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        named=dict(model.named_parameters());assert len(named)==84
        assert set(named)==set(before['model'])==set(after['model'])
        assert sum(v.numel() for v in named.values())==ref['parameter_count']
        qk={name for name in named if name.endswith(('.attn.q.weight','.attn.k.weight'))}
        body={name for name,t in named.items() if name.startswith('blocks.') and t.ndim==2}
        assert len(qk)==16 and len(body)==48 and len(set(named)-qk)==68
        prov=[]
        for p,state,info,old in zip(paths,[before,after],fileinfo,ang_by_key[(*key,STEP)]['files']):
            assert str(p.relative_to(ROOT))==old['path'] and info==(old['bytes'],old['mtime_ns'])
            for name in body:assert sha(state['model'][name].contiguous().numpy().tobytes())==old['consumed_body_tensor_sha256'][name]
            prov.append(dict(path=str(p.relative_to(ROOT)),bytes=info[0],mtime_ns=info[1],model_tensor_sha256=state_hash(state['model'])))
        assert state_hash(model.state_dict())==prov[0]['model_tensor_sha256']
        report['checkpoint_provenance'][label]=prov
        norms={'qk':0.,'rest':0.}
        for name in named:
            norms['qk' if name in qk else 'rest']+=float((after['model'][name].double()-before['model'][name].double()).square().sum())
        report['parameter_norms'][label]={k:v**.5 for k,v in norms.items()}
        def corner(mask,verify=False):
            with torch.no_grad():
                for name,parameter in named.items():
                    target=(after if mask&(1 if name in qk else 2) else before)['model'][name]
                    parameter.copy_(target)
                    if verify:assert torch.equal(parameter,target)
        report['load_seconds'].append(time.monotonic()-tick)
        for i in range(16):
            tick=time.monotonic();lp=[];precision=0.
            with torch.inference_mode():
                for mask in range(4):
                    corner(mask,verify=(i==0))
                    logits=model(x[i:i+1])[0];assert logits.shape==(512,50304) and torch.isfinite(logits).all()
                    logp=torch.log_softmax(logits.double(),-1);lp.append(logp)
                    exact=-logp.gather(-1,y[i,:,None]).squeeze(-1)
                    precision=max(precision,float((exact-F.cross_entropy(logits,y[i],reduction='none').double()).abs().max()))
                    if j==0 and i==0 and mask==0:first_logits=logits.clone()
                assert precision<=5e-5,precision
                values=readout(lp,y[i])
                for name,value in zip(arrays,values[:6]):arrays[name][j,i]=value.numpy()
                checks={**values[-1],'max_fp64_fp32_ce_error':precision}
            for name,value in checks.items():
                if name.startswith('max_'):max_checks[name]=max(max_checks.get(name,0.),value)
                else:min_checks[name]=min(min_checks.get(name,float('inf')),value)
            done+=1;times.append(time.monotonic()-tick)
            if j==0 and i==0:
                corner(0,verify=True)
                with torch.inference_mode():restored=model(x[0:1])[0]
                error=float((restored-first_logits).abs().max());assert error==0.,error
                extra+=1;del restored,first_logits
                forecast=time.monotonic()-_started+63*times[0]*1.25+3*report['load_seconds'][0]*1.25+20.
                report['qualification'].update(first_sequence_seconds=times[0],forecast_seconds=forecast,
                                               restored_base_logits_error=error)
                save('qualification.json',report['qualification'])
                if forecast>CAP:raise TimeoutError(f'Forecast {forecast:.2f} exceeds {CAP}')
            del lp,values,logits,logp
            elapsed=time.monotonic()-_started
            if elapsed>=CAP:raise TimeoutError(f'Wall boundary {elapsed:.2f} exceeds {CAP}')
            if done%4==0:
                persist('partial.npz')
                save('status.json',dict(status='running',state=label,completed_sequences=done,
                     scientific_forwards=done*4,qualification_forwards=extra,seconds=elapsed,pid=os.getpid()))
                print(json.dumps(dict(state=label,completed_sequences=done,seconds=elapsed)),flush=True)
        for p,info in zip(paths,fileinfo):assert (p.stat().st_size,p.stat().st_mtime_ns)==info
        save('progress.json',report);del before,after
    assert done==64 and extra==1 and all(np.isfinite(v).all() for v in arrays.values())
    persist('scalars.npz');report['qualification'].update(**max_checks,**min_checks)
    report.update(status='complete',seconds=time.monotonic()-_started,scientific_forwards=done*4,
                  qualification_forwards=extra,sequence_seconds=times)
    save('result.json',report)
    save('status.json',dict(status='complete',seconds=report['seconds'],scientific_forwards=256,qualification_forwards=extra,pid=os.getpid()))
    print(json.dumps(dict(status='complete',seconds=report['seconds'],qualification=report['qualification'])),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT.exists():
            failure=traceback.format_exc();partial_error=None
            if _partial_writer is not None:
                try:_partial_writer('partial.npz')
                except Exception:partial_error=traceback.format_exc()
            terminal='cost_gate_stop' if isinstance(exc,TimeoutError) else 'failed'
            save('failure.json',dict(status=terminal,traceback=failure,partial_write_error=partial_error))
            save('status.json',dict(status=terminal,seconds=time.monotonic()-_started if _started else None,
                                   pid=os.getpid(),reason=str(exc)))
        raise

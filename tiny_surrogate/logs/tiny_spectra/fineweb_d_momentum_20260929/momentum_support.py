"""Artifact validation for the frozen momentum and horizon comparison."""
import hashlib
import json
import math
from pathlib import Path


def read(p):
    return json.loads(Path(p).read_text())


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()


def write(p,x):
    with Path(p).open('x') as f:
        json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def tensors(x):
    import torch
    if torch.is_tensor(x):yield x
    elif isinstance(x,dict):
        for v in x.values():yield from tensors(v)
    elif isinstance(x,(tuple,list)):
        for v in x:yield from tensors(v)


class Validator:
    def __init__(self,cohort,plan,driver):
        import numpy as np
        import torch
        self.c,self.plan,self.driver=cohort,plan,driver
        path=Path(plan['data_path'])
        if sha(path)!=plan['training_manifest_sha256']:raise ValueError('Changed data manifest')
        m=read(path);self.pop={}
        for key,name in [('starts','dev_starts'),('target_counts','dev_lengths'),('document_indices','dev_document_indices')]:
            spec=m[name];p=(path.parent/spec['path']).resolve()
            if not p.is_relative_to(path.parent) or sha(p)!=spec['sha256'] or p.stat().st_size!=spec['bytes']:
                raise ValueError('Changed development population')
            self.pop[key]=torch.from_numpy(np.fromfile(p,dtype='<i8').copy())
        if len(self.pop['starts'])!=3082 or int(self.pop['target_counts'].sum())!=1168147:
            raise ValueError('Wrong full-development population')
        spec=m['splits']['val']['documents'];p=path.parent/spec['path']
        if sha(p)!=spec['sha256']:raise ValueError('Changed document identities')
        self.docs=read(p)
        self.streams={s:torch.randperm((m['splits']['train']['token_count']-1)//512,
                     generator=torch.Generator().manual_seed(s+1729))*512 for s in plan['seeds']}
        self.ema={}

    def expected_stream(self,seed,tokens):
        return self.streams[seed][:tokens//512]

    def row(self,cfg):
        import torch
        from torch.torch_version import TorchVersion
        c=self.c;root=c/'runs'/cfg['run_id'];p=self.plan
        exe=read(root/'ARM_EXECUTION.json');meta=read(root/'metadata.json')
        n=math.ceil(cfg['total_tokens']/cfg['batch_tokens'])
        if (read(root/'config.json')!=cfg or exe['config_sha256']!=sha(c/'configs'/(cfg['run_id']+'.json')) or
            exe['source_manifest_sha256']!=sha(c/'source_manifest.json') or
            self.driver.classify_exit(root,exe['returncode'])!=exe['status']):raise ValueError('Execution identity mismatch')
        expected=self.expected_stream(cfg['seed'],cfg['total_tokens'])
        identities=p['seed_identities'][str(cfg['seed'])]
        if (any(meta[k]!=v for k,v in identities.items()) or
            meta['train_window_sha256']!=hashlib.sha256(expected.numpy().tobytes()).hexdigest() or
            meta['n_parameters']!=4861056 or meta['total_steps']!=n or
            meta['model_config']['stats_clock']!='microforward' or
            meta['covariance_gram_precision']!='FP32; autocast disabled' or
            Path(meta['source_file']).resolve()!=c/'frozen/research/tiny_spectra/train.py' or
            'A4000' not in meta['hardware']['name'] or not 0<meta['hardware']['total_memory_bytes']<45*1024**3):
            raise ValueError('Data/initialization/stream/hardware/source mismatch')
        windows=torch.load(root/'windows.pt',map_location='cpu',weights_only=True)
        if not torch.equal(windows['train'],expected) or not torch.equal(windows['validation'],self.pop['starts']):
            raise ValueError('Shared training prefix or development windows changed')
        row=dict(run_id=cfg['run_id'],method=cfg['method'],lr=cfg['lr'],momentum=cfg['momentum'],seed=cfg['seed'],
                 batch_tokens=cfg['batch_tokens'],total_tokens=cfg['total_tokens'],status=exe['status'],
                 full_development_nll=None,evaluations=[],reused=False)
        names=['ARM_EXECUTION.json','config.json','metadata.json','metrics.jsonl','windows.pt']
        if exe['status']=='numerical_instability':
            row['failure']=read(root/'failure.json');names.append('failure.json')
        elif exe['status']=='complete':
            summary=read(root/'summary.json');metrics=[json.loads(x) for x in (root/'metrics.jsonl').read_text().splitlines()]
            steps=sorted({0,1,n,*range(cfg['eval_every'],n+1,cfg['eval_every'])})
            if (summary['status']!='complete' or summary['steps']!=n or summary['tokens']!=cfg['total_tokens'] or
                [x['step'] for x in metrics]!=list(range(n+1)) or
                [x['tokens'] for x in metrics]!=[min(i*cfg['batch_tokens'],cfg['total_tokens']) for i in range(n+1)] or
                [x['step'] for x in metrics if 'validation_nll' in x]!=steps):raise ValueError('Horizon/cadence mismatch')
            for x in metrics:
                for key in ['train_nll','validation_nll','gradient_norm_before_clip','train_probe_nll','step_seconds']:
                    if key in x and not math.isfinite(x[key]):raise ValueError('Nonfinite completed trajectory')
            saved=torch.load(root/'final_validation.pt',map_location='cpu',weights_only=True)
            if any(saved[k].dtype!=torch.long or not torch.equal(saved[k],v) for k,v in self.pop.items()):
                raise ValueError('Saved population mismatch')
            losses=saved['sequence_nll']
            if losses.shape!=self.pop['target_counts'].shape or not torch.isfinite(losses).all():raise ValueError('Invalid losses')
            mean=float((losses*self.pop['target_counts']).sum()/self.pop['target_counts'].sum())
            if any(not math.isfinite(v) or abs(mean-v)>1e-12 for v in
                [summary['full_validation_nll'],summary['final_validation_nll'],metrics[-1]['validation_nll']]):
                raise ValueError('Endpoint reconstruction failed')
            with torch.serialization.safe_globals([TorchVersion]):snap=torch.load(root/'final.pt',map_location='cpu',weights_only=True)
            if snap['config']!=cfg or snap['metadata']!=meta or not all(torch.isfinite(t).all() for t in tensors(snap)):
                raise ValueError('Snapshot identity/finiteness mismatch')
            if cfg['method']=='spd':
                total_forwards=math.ceil(cfg['total_tokens']/2048)
                last_forwards=math.ceil((cfg['total_tokens']-(n-1)*cfg['batch_tokens'])/2048)
                if total_forwards not in self.ema:
                    weights={}
                    for key,beta in [('input_cov_weight',.998),('input_weight',.99)]:
                        w=torch.tensor(0.,dtype=torch.float32)
                        for _ in range(total_forwards):w.mul_(beta).add_(1-beta)
                        weights[key]=float(w)
                    self.ema[total_forwards]=weights
                states=snap['optimizer']['model_statistics'];roots=snap['optimizer']['external']['data_norm']['roots']
                if len(states)!=48 or len(roots)!=48 or any(v[1]!=1+(n-1)//cfg['root_refresh']*cfg['root_refresh'] for v in roots.values()):
                    raise ValueError('Root clock mismatch')
                if any(int(s['_total_forwards'])!=total_forwards or int(s['_step_forwards'])!=last_forwards or
                       any(abs(float(s[k])-v)>1e-7 for k,v in self.ema[total_forwards].items()) for s in states.values()):
                    raise ValueError('EMA/microforward mismatch')
            del snap
            sums=torch.zeros(len(self.docs),dtype=torch.float64).scatter_add_(0,self.pop['document_indices'],losses.double()*self.pop['target_counts'])
            counts=torch.zeros(len(self.docs),dtype=torch.long).scatter_add_(0,self.pop['document_indices'],self.pop['target_counts'])
            groups=[]
            for group in range(4):
                ix=torch.tensor([i for i,d in enumerate(self.docs) if int(d['identity'][:2],16)%4==group])
                groups.append(dict(group=group,targets=int(counts[ix].sum()),nll=float(sums[ix].sum()/counts[ix].sum())))
            row.update(full_development_nll=mean,development_groups=groups,evaluations=[x for x in metrics if 'validation_nll' in x],
                clipping_fraction=sum(x.get('gradient_norm_before_clip',0)>cfg['grad_clip'] for x in metrics[1:])/n,
                training_seconds=summary['training_seconds'],total_seconds=summary['total_seconds'])
            names+=['summary.json','final.pt','final_validation.pt']
        else:raise ValueError('Operational failure blocks analysis')
        row['artifact_sha256']={name:sha(root/name) for name in names}
        return row

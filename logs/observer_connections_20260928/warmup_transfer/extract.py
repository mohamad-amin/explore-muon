"""Reproduce the completed main PD warmup transfer from scalar records only."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import csv
import hashlib
import json
import time

BASE=ROOT/'logs/muon_spectra/soaudit_warmrep_20260928'
ARMS={
 'warmup':'PD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_l40s',
 'constant':'PD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_l40s'}
WINDOWS=[(1,46),(47,92),(93,138),(139,166),(167,184)]


def main():
    start=time.monotonic();manifest={};meta={};rows={};source_checks={}
    def read(p):
        b=p.read_bytes();manifest[str(p.relative_to(ROOT))]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
        return json.loads(b)
    protocol=BASE/'README.md'
    manifest[str(protocol.relative_to(ROOT))]=dict(bytes=protocol.stat().st_size,sha256=hashlib.sha256(protocol.read_bytes()).hexdigest())
    assert 'beats β 0.9 by ≥ 0.01 for both' in protocol.read_text()
    for label,arm in ARMS.items():
        p=BASE/arm/'scientific';meta[label]=read(p/'metadata.json');status=read(p/'status.json');sidecar=read(p/'checkpoint.json')
        assert status['status']=='complete' and status['step']==sidecar['step']==184
        assert status['tokens']==sidecar['tokens']==3079741440
        assert protocol.stat().st_mtime_ns<(p/'status.json').stat().st_mtime_ns
        ds={}
        for f in sorted((p/'steps').glob('step*.json')):
            d=read(f);ds[d['step']]=d
        assert set(ds)==set(range(185));rows[label]=ds
        checks={}
        for name,expected in meta[label]['source_sha256'].items():
            f=p/'source'/name;actual=hashlib.sha256(f.read_bytes()).hexdigest();assert actual==expected
            checks[name]=actual
        source_checks[label]=checks
    assert source_checks['warmup']==source_checks['constant']
    ref=meta['constant'];w=meta['warmup']
    for field in ('initial_model_sha256','train_manifest','validation_manifest','parameter_count','world_size',
                  'device_name','effective_precision','effective_compile','parameter_optimizers','source_sha256'):
        assert w[field]==ref[field],field
    cfgs={k:dict(m['config']) for k,m in meta.items()}
    differences={k:[cfgs['warmup'].get(k),cfgs['constant'].get(k)]
                 for k in set(cfgs['warmup'])|set(cfgs['constant']) if cfgs['warmup'].get(k)!=cfgs['constant'].get(k)}
    assert differences=={'muon_momentum_start':[.8,-1.],'muon_momentum_warmup':[.5,0.]},differences
    c=ref['config'];assert c['muon_momentum']==.9 and c['learning_rate']==.028
    assert c['data_norm_alpha']==.5 and not c['soap_precondition'] and c['batch_tokens']==16777216
    validation=[];trajectory=[];consumed=0
    for step in range(185):
        a,b=rows['warmup'][step],rows['constant'][step]
        for key in ('tokens','batch_tokens','lr','aux_lr','world_size','dummy_sequences'):
            assert a.get(key)==b.get(key),(step,key)
        if 'validation_nll' in a:
            assert 'validation_nll' in b
            validation.append(dict(step=step,warmup=a['validation_nll'],constant=b['validation_nll'],
                                   difference=a['validation_nll']-b['validation_nll']))
        if step:
            beta=.8+.1*min(1.,consumed/(.5*3079741440))
            trajectory.append(dict(step=step,tokens=a['tokens'],batch_tokens=a['batch_tokens'],
                 lr=a['lr'],aux_lr=a['aux_lr'],beta_warmup=beta,beta_constant=.9,
                 train_warmup=a['train_nll'],train_constant=b['train_nll'],
                 train_difference=a['train_nll']-b['train_nll'],
                 warmup_clipped=a['gradient_clipped'],constant_clipped=b['gradient_clipped']))
            consumed=a['tokens']
    assert consumed==3079741440 and [r['step'] for r in validation]==[0,50,100,150,184]
    windows=[]
    for first,last in WINDOWS:
        rr=[r for r in trajectory if first<=r['step']<=last];count=sum(r['batch_tokens'] for r in rr)
        windows.append(dict(first=first,last=last,steps=len(rr),
             mean_difference=sum(r['train_difference'] for r in rr)/len(rr),
             token_weighted_difference=sum(r['train_difference']*r['batch_tokens'] for r in rr)/count,
             negative_steps=sum(r['train_difference']<0 for r in rr)))
    final=validation[-1]['difference'];passed=final<=-.01
    result=dict(scope='Existing main training pair; one seed, one L40S pair, 2x horizon. No new experiment or causal mediator claim.',
                arms=ARMS,config_differences=differences,source_checks=source_checks,input_manifest=manifest,
                initialization_sha256=ref['initial_model_sha256'],validation=validation,training_windows=windows,
                endpoint_difference=final,declared_gain_floor=.01,PD_pair_passes=passed,
                joint_two_pair_prediction_resolved=False,
                first_full_beta_update=next(r['step'] for r in trajectory if r['beta_warmup']==.9),
                first_cooldown_update=next(r['step'] for r in trajectory if r['step']>3 and r['lr']<.028),
                clip_counts={label:sum(r[label+'_clipped'] for r in trajectory) for label in ARMS},
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (HERE/'trajectory.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(trajectory[0]));writer.writeheader();writer.writerows(trajectory)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8.8,4.1))
    ax.plot([r['step'] for r in trajectory],[r['train_difference'] for r in trajectory],color='#0072B2',alpha=.7,linewidth=1,label='Paired incoming-batch training loss')
    ax.plot([r['step'] for r in validation],[r['difference'] for r in validation],'o-',color='#D55E00',label='Paired checkpoint validation loss')
    ax.axhline(0,color='gray',linewidth=.8);ax.axvline(93,color='black',linestyle=':',linewidth=.8,label='Both beta .9 from update 93')
    ax.axvspan(166.5,184,color='gray',alpha=.12,label='Common LR cooldown')
    ax.set(xlabel='Optimizer update',ylabel='NLL: warmup minus constant beta .9',title='PD warmup transfer at 16M batch, doubled horizon, one matched seed')
    ax.legend(frameon=False,fontsize=8);ax.grid(alpha=.15);fig.tight_layout()
    fig.savefig(HERE/'trajectory.png',dpi=170);fig.savefig(HERE/'trajectory.pdf');plt.close(fig)
    result['seconds']=time.monotonic()-start
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('endpoint_difference','PD_pair_passes','first_full_beta_update','first_cooldown_update','clip_counts','seconds')},indent=2))


if __name__=='__main__':main()

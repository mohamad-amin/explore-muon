"""Frozen retrospective four-arm contrast; no training or model imports."""
import csv
import difflib
import hashlib
import json
import math
import struct
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=ROOT/'logs/muon_spectra'
ARMS={
    (.028,.9):'soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada',
    (.028,.8):'soaudit_prefilter16m_20260928/SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada',
    (.04,.9):'soaudit_batch16m_20260927/SPD_a0.5_b16M_lr0.04_mom0.9_s260925_ada',
    (.04,.8):'soaudit_strength16m_20260928/SPD_a0.5_b16M_lr0.04_mom0.8_s260925_ada',
}
DEFAULTS={'muon_prefilter':'none','head_whitening_alpha':0.,'head_whitening_center':False}
WINDOWS=[(4,23),(24,43),(44,63),(64,83),(84,92)]


def norm_config(c):
    c=dict(c)
    for k,v in DEFAULTS.items():c.setdefault(k,v)
    for k in ['learning_rate','muon_momentum']:c.pop(k)
    return c


def main():
    datasets={};metadata={};manifest={};sources={}
    def read(f):
        b=f.read_bytes();manifest[str(f.relative_to(ROOT))]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        return json.loads(b)
    for key,arm in ARMS.items():
        p=BASE/arm/'scientific';meta=read(p/'metadata.json');status=read(p/'status.json')
        assert status['status']=='complete',arm
        assert meta['total_steps']==92 and meta['config']['learning_rate']==key[0] and meta['config']['muon_momentum']==key[1]
        metadata[str(key)]=meta
        rows={}
        for f in sorted((p/'steps').glob('step*.json')):
            d=read(f);rows[d['step']]=d
        assert all(s in rows and 'train_nll' in rows[s] for s in range(1,93))
        assert rows[92]['tokens']==meta['budget_tokens']
        datasets[key]=rows
        src=BASE/arm.split('/')[0]/'frozen/adamw_spectra'
        sources[arm]={n:(src/n).read_text() for n in ['model.py','muon.py','train.py','distributed.py','data.py']}
    ref=next(iter(metadata.values()))
    fields=['initial_model_sha256','train_manifest','validation_manifest','budget_tokens','world_size','device_name','effective_precision','effective_compile','nesterov','ns_coefficients','parameter_optimizers']
    for m in metadata.values():
        assert norm_config(m['config'])==norm_config(ref['config'])
        for k in fields:assert m[k]==ref[k],k
    baseline=ARMS[(.028,.9)];diffs=[]
    for arm,src in sources.items():
        for name,contents in src.items():
            diffs.extend(difflib.unified_diff(sources[baseline][name].splitlines(),contents.splitlines(),fromfile=baseline+'/'+name,tofile=arm+'/'+name,n=3))
    (HERE/'source_differences.patch').write_text('\n'.join(diffs)+'\n')
    # Nominal exact direction norm for the common architecture.
    c=ref['config']['model'];w=c['n_embd'];L=c['n_layer']
    direction_norm=math.sqrt(L*(5*w+4*w));assert direction_norm==192.
    clocks={}
    for key,rows in datasets.items():
        total=aux_total=decay=0.;shrink=1.;arr={}
        for step in range(1,93):
            d=rows[step];arr[step]={'train_body_lr_sum':total,'train_nominal_body_arc':direction_norm*total,
                                   'train_aux_lr_sum':aux_total,'train_decay_exposure':decay}
            total+=d['lr'];aux_total+=d['aux_lr'];decay+=d['lr']*ref['config']['weight_decay'];shrink*=1-d['lr']*ref['config']['weight_decay']
            arr[step].update(validation_body_lr_sum=total,validation_nominal_body_arc=direction_norm*total,
                             validation_aux_lr_sum=aux_total,validation_decay_exposure=decay,shrink_product=shrink)
        clocks[key]=arr
    for eta in [.028,.04]:
        assert clocks[(eta,.8)]==clocks[(eta,.9)]
    aux_clock_errors=[]
    for s in range(1,93):
        x=datasets[(.028,.8)][s]['aux_lr'];y=datasets[(.04,.8)][s]['aux_lr']
        assert abs(x-y)<=4*max(math.ulp(x),math.ulp(y))
        assert struct.pack('f',x)==struct.pack('f',y)
        aux_clock_errors.append(abs(x-y))
    rows=[]
    for step in range(1,93):
        r={'step':step,'tokens':datasets[(.028,.9)][step]['tokens'],'batch_tokens':datasets[(.028,.9)][step]['batch_tokens']}
        for eta in [.028,.04]:
            tag=str(eta);lo=datasets[(eta,.8)][step];hi=datasets[(eta,.9)][step]
            r['delta_'+tag]=lo['train_nll']-hi['train_nll']
            r['clock_'+tag]=clocks[(eta,.9)][step]['train_body_lr_sum']
        r['interaction']=r['delta_0.04']-r['delta_0.028'];rows.append(r)
    summaries=[]
    for first,last in WINDOWS:
        selected=[r for r in rows if first<=r['step']<=last];den=sum(r['batch_tokens'] for r in selected)
        summaries.append({'first_step':first,'last_step':last,'tokens':den,
                          **{k:sum(r[k]*r['batch_tokens'] for r in selected)/den for k in ['delta_0.028','delta_0.04','interaction']}})
    common=set.intersection(*[{s for s,d in ds.items() if d.get('validation_nll') is not None} for ds in datasets.values()])
    validation=[]
    for s in sorted(common):
        losses={str(k):datasets[k][s]['validation_nll'] for k in ARMS}
        d1=datasets[(.028,.8)][s]['validation_nll']-datasets[(.028,.9)][s]['validation_nll']
        d2=datasets[(.04,.8)][s]['validation_nll']-datasets[(.04,.9)][s]['validation_nll']
        validation.append({'step':s,'losses':losses,'delta_0.028':d1,'delta_0.04':d2,'interaction':d2-d1})
    result={'scope':'One-seed retrospective beta×body-LR contrast; steps/windows are not replications. Training is pre-update; validation is post-update.',
            'arms':{str(k):v for k,v in ARMS.items()},'pairing_checked_fields':fields,'initial_model_sha256':ref['initial_model_sha256'],
            'config_normalization_defaults':DEFAULTS,'direction_norm_before_lr_decay_rounding':direction_norm,
            'aux_clock_max_abs_difference':max(aux_clock_errors),'aux_clock_fp32_rates_identical':True,
            'validation':validation,'training_windows':summaries,'training_rows':rows,
            'end_clocks':{str(k):v[92] for k,v in clocks.items()},'source_hashes':{str(k):v['source_sha256'] for k,v in metadata.items()},
            'input_manifest':manifest,'metadata':metadata}
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    with (HERE/'training.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Static figure; smoothing is display-only and includes every observation.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(1,2,figsize=(10,3.9),sharey=True)
    for eta,color in [(.028,'#0072B2'),(.04,'#D55E00')]:
        ys=[r['delta_'+str(eta)] for r in rows]
        smooth=[sum(ys[max(0,i-4):min(len(ys),i+5)])/len(ys[max(0,i-4):min(len(ys),i+5)]) for i in range(len(ys))]
        for a,x in zip(ax,[[r['step'] for r in rows],[r['clock_'+str(eta)] for r in rows]]):
            a.plot(x,ys,color=color,alpha=.23,linewidth=.8)
            a.plot(x,smooth,color=color,label=f'Body LR {eta}')
    for a in ax:a.axhline(0,color='gray',linewidth=.8);a.grid(alpha=.15);a.legend(frameon=False)
    ax[0].set(xlabel='Training batch step (before update)',ylabel='Training NLL: beta .8 minus beta .9')
    ax[1].set(xlabel='Cumulative body LR before the batch')
    ax[0].axvspan(83.5,92,color='gray',alpha=.1)
    fig.suptitle('Same initialization, data, 16M batch and auxiliary schedule; one seed')
    fig.tight_layout();fig.savefig(HERE/'contrasts.png',dpi=170);fig.savefig(HERE/'contrasts.pdf');plt.close(fig)
    print(json.dumps({'validation':validation,'training_windows':summaries,'end_clocks':result['end_clocks']},indent=2))


if __name__=='__main__':main()

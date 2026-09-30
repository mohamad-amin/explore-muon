"""Read the newly completed SOAP-PD replication; preserve earlier PD extraction."""
import csv
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BASE=ROOT/'logs/muon_spectra/soaudit_warmrep_20260928'
ARMS={'warmup':'SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260926_ada',
      'constant':'SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260926_ada'}


def main():
    manifest={};meta={};records={};sources={}
    def read(path):
        b=path.read_bytes();manifest[str(path.relative_to(ROOT))]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());return json.loads(b)
    for label,arm in ARMS.items():
        p=BASE/arm/'scientific';meta[label]=read(p/'metadata.json')
        status=read(p/'status.json');cp=read(p/'checkpoint.json')
        assert status['status']=='complete' and status['step']==cp['step']==184
        assert status['tokens']==cp['tokens']==3079741440
        records[label]={}
        for file in sorted((p/'steps').glob('step*.json')):
            r=read(file);records[label][r['step']]=r
        assert set(records[label])==set(range(185))
        sources[label]={}
        for name,expected in meta[label]['source_sha256'].items():
            actual=hashlib.sha256((p/'source'/name).read_bytes()).hexdigest();assert actual==expected
            sources[label][name]=actual
    assert sources['warmup']==sources['constant']
    for field in ('initial_model_sha256','train_manifest','validation_manifest','parameter_count','world_size',
                  'device_name','effective_precision','effective_compile','parameter_optimizers','source_sha256'):
        assert meta['warmup'][field]==meta['constant'][field],field
    w=meta['warmup']['config'];c=meta['constant']['config']
    differences={k:[w.get(k),c.get(k)] for k in set(w)|set(c) if w.get(k)!=c.get(k)}
    assert differences=={'muon_momentum_start':[.8,-1.],'muon_momentum_warmup':[.5,0.]}
    assert c['seed']==260926 and c['soap_precondition'] and c['data_norm_alpha']==.5 and c['learning_rate']==.028
    validation=[];trajectory=[]
    for step in range(185):
        a,b=records['warmup'][step],records['constant'][step]
        for key in ('tokens','batch_tokens','lr','aux_lr','world_size','dummy_sequences'):
            assert a.get(key)==b.get(key),(step,key)
        if 'validation_nll' in a:
            validation.append(dict(step=step,warmup=a['validation_nll'],constant=b['validation_nll'],
                                   difference=a['validation_nll']-b['validation_nll']))
        if step:trajectory.append(dict(step=step,tokens=a['tokens'],batch_tokens=a['batch_tokens'],
             lr=a['lr'],aux_lr=a['aux_lr'],train_warmup=a['train_nll'],train_constant=b['train_nll'],
             train_difference=a['train_nll']-b['train_nll'],warmup_clipped=a['gradient_clipped'],constant_clipped=b['gradient_clipped']))
    assert [r['step'] for r in validation]==[0,50,100,150,184]
    original_arms={'warmup':'soaudit_momwarm16m_20260928/SPD_a0.5_b16M_T2x_lr0.028_momwarm0.8to0.9_s260925_ada',
                   'constant':'soaudit_b16mlong_20260928/SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada'}
    original={label:read(ROOT/'logs/muon_spectra'/arm/'scientific/steps/step000184.json')['validation_nll']
              for label,arm in original_arms.items()}
    pd=read(HERE/'result.json');assert pd['PD_pair_passes']
    # Recheck the PD endpoint values instead of treating a derived Boolean as proof.
    pd_raw={label:read(BASE/arm/'scientific/steps/step000184.json')['validation_nll'] for label,arm in pd['arms'].items()}
    assert pd_raw['warmup']-pd_raw['constant']==pd['endpoint_difference']
    gap=validation[-1]['difference'];passed=gap<=-.01
    result=dict(scope='Completed main-program replication; SOAP-PD across two seeds and PD at one seed, not a crossed two-method/two-seed study.',
       arms=ARMS,seed=260926,config_differences=differences,source_checks=sources,input_manifest=manifest,
       validation=validation,endpoint_difference=gap,SOAP_replication_passes=passed,
       PD_transfer_difference=pd['endpoint_difference'],original_SOAP_difference=original['warmup']-original['constant'],
       both_predeclared_cohort_pairs_pass=passed and pd['endpoint_difference']<=-.01,
       declared_gain_floor=.01,clip_counts={label:sum(r[label+'_clipped'] for r in trajectory) for label in ARMS},
       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with (HERE/'replication_trajectory.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(trajectory[0]));writer.writeheader();writer.writerows(trajectory)
    (HERE/'replication_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('endpoint_difference','SOAP_replication_passes','PD_transfer_difference','original_SOAP_difference','both_predeclared_cohort_pairs_pass','clip_counts')},indent=2))


if __name__=='__main__':main()

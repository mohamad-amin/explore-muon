"""Frozen two-momentum comparison and deterministic horizon controls."""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import random
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from momentum_support import read,sha,write,Validator
from momentum_analysis import SEEDS,BATCHES,METHODS,BETAS,LRS,TOTAL,analyse_base,analyse_controls


def driver(c):
    spec=importlib.util.spec_from_file_location('qualified_controller',c/'ordering_driver.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def verify(c):
    p=driver(c).verify(c,'initial');pre=read(c/'PREFLIGHT.json')
    for name,digest in pre['analysis_and_decision_sha256'].items():
        if sha(c/name)!=digest:raise ValueError('Changed frozen momentum identity: '+name)
    ref=Path(p['reference_cohort'])
    if sha(ref/'report/results.json')!=p['reference_report_sha256']:raise ValueError('Changed base reference')
    for row in p['reused']:
        for name,digest in row['artifact_sha256'].items():
            if sha(ref/'runs'/row['run_id']/name)!=digest:raise ValueError('Changed reused artifact')
    if (c/'EDGE_PREFLIGHT.json').exists():
        driver(c).verify(c,'edges');edge=read(c/'EDGE_PREFLIGHT.json')
        if sha(c/'report_base/results.json')!=edge['base_report_sha256']:raise ValueError('Changed control-selection base report')
        if sha(c/'job_edges.sh')!=edge['job_script_sha256']:raise ValueError('Changed control launcher')
    return p


def create(c,ref,study):
    if not c.is_relative_to(study) or (c/'PLAN.json').exists():raise ValueError('Use one new isolated cohort')
    driver(ref).verify(ref,'initial')
    previous=read(ref/'report/results.json')
    oldplan=read(ref/'PLAN.json')
    reused=[r for r in previous['rows'] if r['batch_tokens'] in BATCHES]
    if len(reused)!=12 or any(r['status']!='complete' for r in reused):raise ValueError('Need12 completed reference arms')
    for d in ['configs','runs','console','workers']:(c/d).mkdir()
    shutil.copytree(ref/'frozen',c/'frozen')
    for name in ['source_manifest.json','ordering_driver.py']:shutil.copy2(ref/name,c/name)
    for name in ['momentum_support.py','momentum_analysis.py']:shutil.copy2(Path(__file__).parent/name,c/name)
    shutil.copy2(__file__,c/'momentum_driver.py')
    templates={};existing={};identities={};reuse_records=[]
    keys=['initial_parameter_sha256','validation_bank_sha256','validation_bank_lengths_sha256','data_manifest_sha256']
    for row in reused:
        root=ref/'runs'/row['run_id'];cfg=read(root/'config.json');meta=read(root/'metadata.json')
        key=(cfg['method'],cfg['batch_tokens'],cfg['momentum'],cfg['lr'],cfg['seed'])
        existing[key]=row;templates[cfg['method'],cfg['batch_tokens'],cfg['seed']]=cfg
        identity={k:meta[k] for k in keys}
        if str(cfg['seed']) in identities and identities[str(cfg['seed'])]!=identity:raise ValueError('Reference pairing mismatch')
        identities[str(cfg['seed'])]=identity
        hashes=dict(row['artifact_sha256'],**{'windows.pt':sha(root/'windows.pt')})
        reuse_records.append(dict(run_id=cfg['run_id'],artifact_sha256=hashes))
    tasks=[]
    for m in METHODS:
        for b in BATCHES:
            for beta in BETAS:
                for lr in LRS:
                    for seed in SEEDS:
                        if (m,b,beta,lr,seed) in existing:continue
                        cfg=dict(templates[m,b,seed],momentum=beta,lr=lr)
                        cfg['run_id']=f'{m}_Dmomentum_base_b{b}_m{beta:g}_lr{lr:g}_s{seed}'
                        path=c/'configs'/(cfg['run_id']+'.json');write(path,cfg)
                        tasks.append([dict(config=str(path),out=str(c/'runs'/cfg['run_id']))])
    if len(tasks)!=20:raise ValueError('Expected20 new base arms')
    random.Random(20260929).shuffle(tasks)
    p=dict(stage='separate_momentum_component',base_new_runs=20,base_reused_runs=12,control_runs=24,
        reference_cohort=str(ref),reference_report_sha256=sha(ref/'report/results.json'),reused=reuse_records,
        methods=METHODS,batches=BATCHES,betas=BETAS,lrs=LRS,seeds=SEEDS,base_total_tokens=TOTAL,
        data_path=templates[METHODS[0],BATCHES[0],SEEDS[0]]['data_path'],training_manifest_sha256=oldplan['training_manifest_sha256'],
        seed_identities=identities,primary_rule='Per method: G_high>0 and I>0 at each LR and seed; within each seed means across rates >=0.005 for both',
        gain='G=L_beta0.9-L_beta0.8; I=G_high-G_low',secondary='Joint mean-seed LR per beta; cannot rescue fixed-LR primary failure',
        control_rule='One common LR per method/batch; minimize mean base endpoint across both betas and both seeds, ties smaller LR',
        controls_regardless_of_base_scientific_sign=True,matched_steps=128,extended_high_tokens=2*TOTAL,
        control_eval_every=1,forecast_gpu_hours=[20,40],gpu=oldplan['gpu'],scheduler_job_walltime_seconds=10800,
        worker_process_seconds=10740,per_arm_timeout_seconds=10680,array_throttle=None,maximum_gpu_hours=None,
        automatic_retries=False,automatic_requeue=False,full_goal_qualified=False,sealed_test_scoring=False,
        original_ordering_and_batch_rate_gates_remain_failed=True,protocol_sha256=sha(study/'PROTOCOL.md'))
    write(c/'PLAN.json',p);write(c/'tasks_initial.json',tasks)
    driver(c).make_script(c,study,'initial',10740)
    write(c/'PREFLIGHT.json',dict(plan_sha256=sha(c/'PLAN.json'),source_manifest_sha256=sha(c/'source_manifest.json'),
        controller_sha256=sha(c/'ordering_driver.py'),tasks_initial_sha256=sha(c/'tasks_initial.json'),
        initial_config_sha256={p.name:sha(p) for p in (c/'configs').glob('*.json')},
        analysis_and_decision_sha256={name:sha(c/name) for name in ['momentum_driver.py','momentum_support.py','momentum_analysis.py','PEER_DISCUSSION.json','job_initial.sh']}))
    verify(c)
    return dict(created=str(c),new_base_arms=20,reused=12,subsequent_controls=24)


def configs_ready(c,stage):
    if (c/'HALT.json').exists():raise ValueError('Preserved operational failure')
    cfgs=[read(a[0]['config']) for a in read(c/f'tasks_{stage}.json')]
    for cfg in cfgs:
        p=c/'runs'/cfg['run_id']/'ARM_EXECUTION.json'
        if not p.exists():raise ValueError('Wait for the complete declared stage before scientific readout')
        if read(p)['status'] not in ['complete','numerical_instability']:raise ValueError('Operational failure')
    return cfgs


def base_report(c):
    p=verify(c)
    if (c/'report_base').exists():raise FileExistsError('Preserve base report')
    cfgs=configs_ready(c,'initial');v=Validator(c,p,driver(c));ref=Path(p['reference_cohort'])
    old=read(ref/'report/results.json');ids={r['run_id'] for r in p['reused']};rows=[]
    for r in old['rows']:
        if r['run_id'] not in ids:continue
        root=ref/'runs'/r['run_id'];cfg=read(root/'config.json');metrics=[json.loads(x) for x in (root/'metrics.jsonl').read_text().splitlines()]
        windows=__import__('torch').load(root/'windows.pt',map_location='cpu',weights_only=True)
        if not __import__('torch').equal(windows['train'],v.expected_stream(cfg['seed'],TOTAL)):raise ValueError('Reference stream prefix mismatch')
        rows.append(dict(r,momentum=.9,total_tokens=TOTAL,reused=True,
            clipping_fraction=sum(x.get('gradient_norm_before_clip',0)>cfg['grad_clip'] for x in metrics[1:])/(len(metrics)-1)))
    rows.extend(v.row(cfg) for cfg in cfgs)
    result=dict(analyse_base(rows),rows=rows,plan_sha256=sha(c/'PLAN.json'),analysis_sha256=sha(c/'momentum_analysis.py'))
    (c/'report_base').mkdir();write(c/'report_base/results.json',result)
    return result


def prepare_controls(c,study):
    p=verify(c)
    if (c/'EDGE_PREFLIGHT.json').exists():raise FileExistsError('Controls already frozen')
    base=read(c/'report_base/results.json')
    if base['plan_sha256']!=sha(c/'PLAN.json'):raise ValueError('Base report plan mismatch')
    fresh=analyse_base(base['rows'])
    if fresh['control_rate_choices']!=base['control_rate_choices']:raise ValueError('Control selection mismatch')
    # Recheck every base artifact before carrying its selected rate forward.
    ref=Path(p['reference_cohort'])
    for row in base['rows']:
        root=(ref if row['reused'] else c)/'runs'/row['run_id']
        for name,digest in row['artifact_sha256'].items():
            if sha(root/name)!=digest:raise ValueError('Changed base selection artifact')
    choices=fresh['control_rate_choices']
    if any(x['selected_lr'] is None for x in choices.values()):raise ValueError('No finite balanced control LR; preserve failure without inventing rate')
    tasks=[];kinds={}
    for kind in ['matched','extended']:
        for m in METHODS:
            for b in (BATCHES if kind=='matched' else (BATCHES[-1],)):
                lr=choices[f'{m}:{b}']['selected_lr']
                for beta in BETAS:
                    for seed in SEEDS:
                        row=next(r for r in base['rows'] if (r['method'],r['batch_tokens'],r['momentum'],r['lr'],r['seed'])==(m,b,beta,lr,seed))
                        cfg=read((ref if row['reused'] else c)/'runs'/row['run_id']/'config.json')
                        cfg.update(total_tokens=b*128 if kind=='matched' else 2*TOTAL,eval_every=1)
                        cfg['run_id']=f'{m}_Dmomentum_{kind}_b{b}_m{beta:g}_lr{lr:g}_s{seed}'
                        path=c/'configs'/(cfg['run_id']+'.json');write(path,cfg);kinds[cfg['run_id']]=kind
                        tasks.append([dict(config=str(path),out=str(c/'runs'/cfg['run_id']))])
    random.Random(20260929).shuffle(tasks);write(c/'tasks_edges.json',tasks)
    driver(c).make_script(c,study,'edges',10740)
    write(c/'EDGE_PREFLIGHT.json',dict(plan_sha256=sha(c/'PLAN.json'),base_report_sha256=sha(c/'report_base/results.json'),
        tasks_edges_sha256=sha(c/'tasks_edges.json'),edges_config_sha256={Path(t[0]['config']).name:sha(t[0]['config']) for t in tasks},
        job_script_sha256=sha(c/'job_edges.sh'),control_kinds=kinds,control_rate_choices=choices))
    verify(c);return dict(frozen_controls=len(tasks),choices=choices)


def final_report(c):
    p=verify(c)
    if (c/'report_final').exists():raise FileExistsError('Preserve final report')
    cfgs=configs_ready(c,'edges');v=Validator(c,p,driver(c));edge=read(c/'EDGE_PREFLIGHT.json')
    rows=[dict(v.row(cfg),control_kind=edge['control_kinds'][cfg['run_id']]) for cfg in cfgs]
    base=read(c/'report_base/results.json')
    ref=Path(p['reference_cohort'])
    for row in base['rows']:
        root=(ref if row['reused'] else c)/'runs'/row['run_id']
        for name,digest in row['artifact_sha256'].items():
            if sha(root/name)!=digest:raise ValueError('Base artifact changed during controls')
    result=dict(phase_controls=analyse_controls(base['rows'],rows,edge['control_rate_choices']),control_rows=rows,
        base_primary=base['primary_by_method'],base_report_sha256=sha(c/'report_base/results.json'),
        plan_sha256=sha(c/'PLAN.json'),full_goal_qualified=False,sealed_test_scored=False)
    (c/'report_final').mkdir();write(c/'report_final/results.json',result);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('operation',choices=['create','verify','submit','base-report','prepare-controls','final-report'])
    ap.add_argument('cohort',type=Path);ap.add_argument('--reference',type=Path);ap.add_argument('--study',type=Path)
    ap.add_argument('--stage',choices=['initial','edges'],default='initial');a=ap.parse_args();c=a.cohort.resolve()
    if a.operation=='create':result=create(c,a.reference.resolve(),a.study.resolve())
    elif a.operation=='verify':result=dict(verified=verify(c)['stage'])
    elif a.operation=='submit':verify(c);driver(c).submit(c,a.stage);result={}
    elif a.operation=='base-report':result=base_report(c)
    elif a.operation=='prepare-controls':result=prepare_controls(c,a.study.resolve())
    else:result=final_report(c)
    print(json.dumps({k:v for k,v in result.items() if k not in ['rows','control_rows','all_fixed_rate_crossings']},indent=2))

"""Bounded same-node1/2/4-GPU execution and numerical qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

METHODS=('adamw','muon','pd','ts','soap','spd','sts')


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def prepare(out):
    from research.tiny_spectra.cohort import PROJECT,freeze
    study=PROJECT.resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    for name in ['configs','runs','console']:(out/name).mkdir()
    native=study/'logs/tiny_spectra/fineweb_d_smoke_20260928'
    base=read(study/'configs/multigpu/spd.json')
    for method in METHODS:
        cfg=dict(base,method=method,run_id='parallel_qualification_'+method,seed=20261001,momentum=.9,
            lr=.0006 if method=='adamw' else .01,batch_tokens=262144,total_tokens=21*262144+69*512,
            root_refresh=10,eval_every=21,parallel_audit_steps=[1,22])
        write(out/'configs'/(method+'.json'),cfg)
    empty=dict(base,method='spd',run_id='parallel_empty_ranks',momentum=.9,lr=.01,batch_tokens=512,total_tokens=512,
        root_refresh=10,eval_every=1,parallel_audit_steps=[1])
    write(out/'configs/empty.json',empty)
    freeze(out/'frozen');shutil.copy2(__file__,out/'qualification_driver.py')
    plan=dict(methods=METHODS,worlds=[1,2,4],native_reference=str(native),native_source_manifest_sha256=sha(native/'source_manifest.json'),
        source_manifest_sha256=sha(out/'source_manifest.json'),driver_sha256=sha(out/'qualification_driver.py'),
        config_sha256={p.name:sha(p) for p in (out/'configs').glob('*.json')},
        same_state_rmse_absolute=1e-8,same_state_rmse_relative=1e-5,write_relative_l2=.01,write_cosine=.9999,
        native_world1_tensors_exact=True,free_trajectory_drift='report separately; no silent tolerance relaxation',
        full_updates=21,tail_sequences=69,total_updates=22,expected_global_forwards=2706,
        includes_zero_work_ranks=True,includes_worker_failure=True,worker_failure_deadline_seconds=60,
        gpu='nvidia_rtx_a4000',allocated_gpus=4,walltime='01:30:00',scientific_ranking_claim=False,sealed_test_scored=False)
    write(out/'PLAN.json',plan)
    import shlex
    (out/'job.sh').write_text('#!/bin/bash\nset -euo pipefail\nexport OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1\nexec '+shlex.quote(str(study/'run'))+' '+shlex.quote(str(out/'qualification_driver.py'))+' worker '+shlex.quote(str(out))+'\n')
    return out


def verify(c):
    p=read(c/'PLAN.json')
    if sha(c/'qualification_driver.py')!=p['driver_sha256'] or sha(c/'source_manifest.json')!=p['source_manifest_sha256']:raise ValueError('Changed qualification sources')
    native=Path(p['native_reference'])
    if sha(native/'source_manifest.json')!=p['native_source_manifest_sha256']:raise ValueError('Changed native manifest')
    for root in [c,native]:
        for path,digest in read(root/'source_manifest.json').items():
            if sha(root/'frozen'/path)!=digest:raise ValueError('Changed numerical source')
    for name,digest in p['config_sha256'].items():
        if sha(c/'configs'/name)!=digest:raise ValueError('Changed qualification config')
    return p


def launch(c,label,method,world=None,fault=False):
    p=verify(c);source=Path(p['native_reference'])/'frozen' if world is None else c/'frozen'
    cfg=c/'configs'/(method+'.json');out=c/'runs'/label
    if out.exists():raise FileExistsError('Never overwrite qualification attempt')
    if world is None:command=[sys.executable,'-m','research.tiny_spectra.train','--config',str(cfg),'--out',str(out)]
    else:
        command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc-per-node={world}',
            '--max-restarts=0','--module','research.tiny_spectra.parallel_train','--config',str(cfg),'--out',str(out),
            '--expected-world-size',str(world)]
        if fault:command+=['--inject-failure-rank','1']
    start=time.time();env=dict(os.environ,PYTHONPATH=str(source),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONUNBUFFERED='1')
    process=None
    try:
        with (c/'console'/(label+'.log')).open('x') as log:
            process=subprocess.Popen(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=process.wait(timeout=60 if fault else 4500)
    except BaseException:
        if process is not None and process.poll() is None:os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise
    elapsed=time.time()-start
    record=dict(label=label,method=method,world_size=1 if world is None else world,legacy_native=world is None,
        returncode=code,wall_seconds=elapsed,active_worker_gpu_seconds=elapsed*(1 if world is None else world),
        allocation_gpu_seconds=elapsed*4,intentional_worker_failure=fault)
    write(c/(label+'_execution.json'),record);print(json.dumps(record),flush=True)
    if fault:
        if code==0 or not (out/'failure_rank1.json').exists():raise ValueError('Intentional worker failure did not propagate')
    elif code!=0 or not (out/'summary.json').exists():raise RuntimeError('Qualification execution failed: '+label)


def worker(c):
    verify(c);write(c/'EXECUTION_STARTED.json',dict(started_unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID')))
    try:
        for method in METHODS:
            launch(c,'native_'+method,method)
            for world in [1,2,4]:launch(c,f'w{world}_{method}',method,world)
        for world in [1,2,4]:launch(c,f'w{world}_empty','empty',world)
        launch(c,'worker_failure','empty',2,True)
        write(c/'EXECUTION_COMPLETE.json',dict(finished_unix=time.time(),normal_runs=31,expected_failure_runs=1))
    except BaseException as e:
        write(c/'EXECUTION_FAILURE.json',dict(error=repr(e),failed_unix=time.time()));raise


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('operation',choices=['prepare','worker','verify']);ap.add_argument('cohort',type=Path)
    a=ap.parse_args();c=a.cohort.resolve()
    if a.operation=='prepare':print(prepare(c))
    elif a.operation=='worker':worker(c)
    else:print(json.dumps(verify(c),indent=2))

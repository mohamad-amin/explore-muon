"""Freeze, run or submit one tiny-surrogate experiment on1,2,or4 A4000s."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time


def read(p):return json.loads(Path(p).read_text())

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for x in iter(lambda:f.read(8<<20),b''):h.update(x)
    return h.hexdigest()


def write(p,x):
    with Path(p).open('x') as f:json.dump(x,f,indent=2,sort_keys=True,allow_nan=False);f.write('\n')


def prepare(config,out,gpus,walltime='03:00:00',node=None,profile=None):
    from research.tiny_spectra.cohort import freeze,PROJECT
    study=PROJECT.resolve();out=Path(out).resolve();config=Path(config).resolve()
    if gpus not in (1,2,4):raise ValueError('Supported GPU counts are1,2,4')
    if not re.fullmatch(r'\d{1,3}:[0-5]\d:[0-5]\d',walltime):raise ValueError('Walltime must beHH:MM:SS')
    if not out.is_relative_to(study):raise ValueError('Keep every artifact inside tiny_surrogate')
    cfg=read(config)
    cfg['execution_profile']=profile or cfg.get('execution_profile','deterministic')
    if cfg['execution_profile'] not in ('native','deterministic'):raise ValueError('Unsupported execution profile')
    if cfg.get('precision') not in ('bf16','fp32'):raise ValueError('Precision must bebf16 orfp32')
    if cfg.get('device','cuda')!='cuda':raise ValueError('These launchers request CUDA GPUs')
    data=Path(cfg['data_path']);data=(data if data.is_absolute() else study/data).resolve()
    if not data.is_file() or sha(data)!=cfg['data_sha256']:raise ValueError('Data/configuration identity mismatch')
    if cfg.get('corpus_kind') in ('fineweb_byte_bpe_v1','tiny_stories_byte_bpe_v1') and read(data).get('role')!='training_and_development_only':
        raise ValueError('Only training/development manifests are accepted')
    cfg['data_path']=str(data)
    if cfg.get('execution_backend','eager')!='eager':raise ValueError('Only native eager kernels supported')
    if any(type(cfg[k]) is not int or cfg[k]<1 for k in ('total_tokens','batch_tokens','seq_len','microbatch_sequences')):
        raise ValueError('Positive integer global token and microbatch settings required')
    if cfg['total_tokens']%cfg['seq_len'] or cfg['batch_tokens']%cfg['seq_len']:raise ValueError('Token budgets must be multiples of context')
    out.mkdir(parents=True,exist_ok=False);(out/'console').mkdir()
    write(out/'config.json',cfg);shutil.copy2(config,out/'input_config.json');freeze(out/'frozen')
    shutil.copy2(__file__,out/'launcher.py')
    script='#!/bin/bash\nset -euo pipefail\nexec '+shlex.quote(str(study/'run'))+' '+shlex.quote(str(out/'launcher.py'))+' worker --out '+shlex.quote(str(out))+'\n'
    (out/'job.sh').write_text(script)
    plan=dict(gpus=gpus,global_batch_tokens=cfg['batch_tokens'],total_tokens=cfg['total_tokens'],microbatch_sequences=cfg['microbatch_sequences'],
        study=str(study),gpu_type='nvidia_rtx_a4000',walltime=walltime,node=node,cpus=cfg.get('cpu_threads',2)*gpus,memory_gib=8*gpus,
        execution_profile=cfg['execution_profile'],config_sha256=sha(out/'config.json'),source_manifest_sha256=sha(out/'source_manifest.json'),launcher_sha256=sha(out/'launcher.py'),
        job_script_sha256=sha(out/'job.sh'),created_unix=time.time(),automatic_retries=False,optimizer_state_owner=0,
        numeric_note='FP32 reduction/EMA reassociation; compare methods at the same GPU count; no bitwise-identity claim',
        sealed_test_scoring=False)
    write(out/'RUN_PLAN.json',plan)
    write(out/'PREFLIGHT.json',dict(plan_sha256=sha(out/'RUN_PLAN.json')))
    return plan


def verify(out):
    out=Path(out).resolve();p=read(out/'RUN_PLAN.json')
    if sha(out/'RUN_PLAN.json')!=read(out/'PREFLIGHT.json')['plan_sha256']:raise ValueError('Frozen run plan changed')
    for name,key in [('config.json','config_sha256'),('source_manifest.json','source_manifest_sha256'),('launcher.py','launcher_sha256'),('job.sh','job_script_sha256')]:
        if sha(out/name)!=p[key]:raise ValueError('Frozen launch identity changed: '+name)
    for name,value in read(out/'source_manifest.json').items():
        path=Path(name)
        if path.is_absolute() or '..' in path.parts or sha(out/'frozen'/path)!=value:raise ValueError('Frozen source changed')
    return p


def submission_command(out,p):
    command=['sbatch','--parsable','--partition=gpu','--nodes=1','--ntasks=1',f"--gres=gpu:{p['gpu_type']}:{p['gpus']}",
        f"--cpus-per-task={p['cpus']}",f"--mem={p['memory_gib']}G",'--time='+p['walltime'],'--no-requeue',
        f"--job-name=tiny-{p['gpus']}gpu",'--chdir='+str(out),'--output='+str(out/'console/slurm_%j.log')]
    if p.get('node'):command+=['--nodelist='+p['node']]
    return command+[str(out/'job.sh')]


def submit(out,dry_run=False):
    out=Path(out).resolve();p=verify(out);command=submission_command(out,p)
    if dry_run:print(shlex.join(command));return
    if (out/'run').exists():raise ValueError('An existing run cannot be resubmitted')
    write(out/'SUBMISSION_INTENT.json',dict(command=command,created_unix=time.time()))
    result=subprocess.run(command,capture_output=True,text=True,check=True)
    write(out/'submission.json',dict(command=command,job_id=result.stdout.strip(),submitted_unix=time.time()))
    print(result.stdout.strip(),flush=True)


def worker(out):
    out=Path(out).resolve();p=verify(out)
    if (out/'run').exists():raise FileExistsError('Existing run output is preserved; no automatic resume')
    write(out/'EXECUTION_STARTED.json',dict(started_unix=time.time(),job_id=os.environ.get('SLURM_JOB_ID'),world_size=p['gpus']))
    command=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1',f"--nproc-per-node={p['gpus']}",
        '--max-restarts=0','--module','research.tiny_spectra.parallel_train','--config',str(out/'config.json'),
        '--out',str(out/'run'),'--expected-world-size',str(p['gpus'])]
    env=dict(os.environ,PYTHONPATH=str(out/'frozen'),OMP_NUM_THREADS=str(read(out/'config.json').get('cpu_threads',2)),
             OPENBLAS_NUM_THREADS=str(read(out/'config.json').get('cpu_threads',2)),PYTHONUNBUFFERED='1')
    if p['execution_profile']=='deterministic':env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    process=None;started=time.time()
    def stop(signum,frame):
        if process is not None and process.poll() is None:
            os.killpg(process.pid,signal.SIGTERM)
            try:process.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
        raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        with (out/'console/training.log').open('x') as log:
            process=subprocess.Popen(command,cwd=out/'frozen',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            code=process.wait()
        if code!=0:raise RuntimeError(f'torchrun failed with exit{code}; see console/training.log')
        summary=read(out/'run/summary.json')
        if summary['status']!='complete' or summary['world_size']!=p['gpus']:raise RuntimeError('Missing or wrong completed distributed run')
        write(out/'EXECUTION_COMPLETE.json',dict(finished_unix=time.time(),elapsed_seconds=time.time()-started,world_size=p['gpus'],command=command))
    except BaseException as error:
        write(out/'EXECUTION_FAILURE.json',dict(error=repr(error),finished_unix=time.time(),returncode=process.returncode if process else None))
        raise


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('operation',choices=['prepare','run','submit','submit-prepared','worker','verify'])
    p.add_argument('--config',type=Path);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--gpus',type=int,choices=[1,2,4],default=1);p.add_argument('--time',default='03:00:00');p.add_argument('--node')
    p.add_argument('--profile',choices=['native','deterministic'])
    p.add_argument('--dry-run',action='store_true');a=p.parse_args();out=a.out.resolve()
    if a.operation in ('prepare','run','submit'):
        if a.config is None:p.error('--config is required')
        plan=prepare(a.config,out,a.gpus,a.time,a.node,a.profile)
        print(json.dumps(dict(prepared=str(out),gpus=plan['gpus'],global_batch_tokens=plan['global_batch_tokens'])),flush=True)
    if a.operation in ('submit','submit-prepared'):submit(out,a.dry_run)
    elif a.operation in ('run','worker'):worker(out)
    elif a.operation=='verify':print(json.dumps(dict(verified=verify(out)),indent=2))


if __name__=='__main__':main()

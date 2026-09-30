import json,os,subprocess,sys,time,hashlib
from pathlib import Path
c=Path(__file__).resolve().parent
for name,digest in json.loads((c/'source_manifest.json').read_text()).items():
    if hashlib.sha256((c/name).read_bytes()).hexdigest()!=digest:raise ValueError('Changed diagnostic source')
p=json.loads((c/'PLAN.json').read_text())
for label,engine,profile in p['cases']:
    source=p['native_source'] if engine=='native' else p['parallel_source']
    env=dict(os.environ,PYTHONPATH=source,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
    if profile=='deterministic':env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    with (c/'console'/(label+'.log')).open('x') as log:
        subprocess.run([sys.executable,str(c/'worker.py'),str(c),label,engine,profile],env=env,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=110)
(c/'COMPLETE.json').write_text(json.dumps(dict(completed_unix=time.time(),cases=5))+'\n')

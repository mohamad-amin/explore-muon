"""Wait for this specific producer, then analyze its complete fixed dataset."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
HERE=Path(__file__).resolve().parent
RUN=HERE/'run1'
def main():
    started=time.monotonic()
    while True:
        if (RUN/'failure.json').exists():raise RuntimeError('Producer reports failure; no analysis or restart is attempted.')
        if (RUN/'status.json').exists():
            status=json.loads((RUN/'status.json').read_text())
            if status['status']=='complete':break
        if time.monotonic()-started>14400:raise TimeoutError('Producer completion not observed within four hours; no restart.')
        time.sleep(5)
    env=os.environ.copy()
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:env[key]='1'
    env['PYTHONDONTWRITEBYTECODE']='1'
    for script,args in [('analyze_atlas.py',['--run',str(RUN)]),('analyze_geometry.py',[])]:
        print('Starting',script,flush=True)
        subprocess.run([sys.executable,str(HERE/script),*args],check=True,env=env,cwd=HERE.parents[1])
        print('Finished',script,flush=True)
    (HERE/'postprocess_status.json').write_text(json.dumps({'status':'complete','seconds_including_wait':time.monotonic()-started},indent=2)+'\n')
if __name__=='__main__':main()

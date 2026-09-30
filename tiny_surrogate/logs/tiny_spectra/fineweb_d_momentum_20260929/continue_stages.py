"""Execute exactly the two predeclared CPU transitions, guarded by Slurm dependencies."""
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from momentum_support import read,sha,write

c=Path(__file__).resolve().parent
phase=sys.argv[1]
if phase not in ('base','final'):raise ValueError('Unknown declared phase')
p=read(c/'AUTOMATION_PLAN.json')
for name,digest in p['source_sha256'].items():
    if sha(c/name)!=digest:raise ValueError('Changed continuation source: '+name)
if sha(c/'PREFLIGHT.json')!=p['experiment_preflight_sha256']:raise ValueError('Changed experiment identity')
write(c/f'AUTOMATION_{phase}_STARTED.json',dict(started_unix=time.time(),phase=phase))
try:
    command=[sys.executable,str(c/'momentum_driver.py')]
    if phase=='base':
        subprocess.run(command+['base-report',str(c)],check=True)
        subprocess.run(command+['prepare-controls',str(c),'--study',p['study']],check=True)
        subprocess.run(command+['submit',str(c),'--stage','edges'],check=True)
        control_id=read(c/'submission_edges.json')['job_id'].split(';')[0]
        if not re.fullmatch(r'[0-9]+',control_id):raise ValueError('Invalid control job ID')
        args=['sbatch','--parsable','--partition=cpu','--nodes=1','--ntasks=1','--cpus-per-task=2','--mem=8G',
            '--time=00:30:00','--no-requeue','--dependency=afterok:'+control_id,'--job-name=tiny-D-momentum-final',
            '--chdir='+str(c),'--output='+str(c/'console/analysis_final_%j.log'),str(c/'job_analysis_final.sh')]
        write(c/'FINAL_ANALYSIS_SUBMISSION_INTENT.json',dict(command=args,control_array=control_id))
        result=subprocess.run(args,capture_output=True,text=True,check=True)
        write(c/'submission_final_analysis.json',dict(command=args,job_id=result.stdout.strip(),submitted_unix=time.time()))
        receipt=dict(control_array=control_id,final_analysis_job=result.stdout.strip())
    else:
        subprocess.run(command+['final-report',str(c)],check=True)
        receipt=dict(final_result_sha256=sha(c/'report_final/results.json'))
    write(c/f'AUTOMATION_{phase}_COMPLETE.json',dict(completed_unix=time.time(),**receipt))
except BaseException as error:
    write(c/f'AUTOMATION_{phase}_FAILURE.json',dict(error=repr(error),failed_unix=time.time()))
    raise

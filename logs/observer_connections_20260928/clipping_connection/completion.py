"""Bounded append-only completion audit of eight already executed scalar logs.

Run from project root. No model calls or rewriting the14:17 audit artifacts.
"""
from pathlib import Path
import csv
import datetime
import difflib
import hashlib
import json
import statistics

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
RUNS={
 'M9':('soaudit_batch16m_20260927','M_b16M_lr0.02_mom0.9_s260925_l40s'),
 'M8':('soaudit_mom16m_20260928','M_b16M_lr0.02_mom0.8_s260925_l40s'),
 'MC9':('soaudit_clip16m_20260928','Mclip0.1_b16M_lr0.02_mom0.9_s260925_l40s'),
 'MC8':('soaudit_clip16m_20260928','Mclip0.1_b16M_lr0.02_mom0.8_s260925_l40s'),
 'S9':('soaudit_batch16m_20260927','SPD_a0.5_b16M_lr0.028_mom0.9_s260925_ada'),
 'S8':('soaudit_prefilter16m_20260928','SPD_a0.5_b16M_lr0.028_mom0.8_s260925_ada'),
 'SL9':('soaudit_b16mlong_20260928','SPD_a0.5_b16M_T2x_lr0.028_mom0.9_s260925_ada'),
 'SL8':('soaudit_horizon16m_20260928','SPD_a0.5_b16M_T2x_lr0.028_mom0.8_s260925_ada'),
}
PAIRS=[('M9','M8'),('MC9','MC8'),('S9','S8'),('SL9','SL8'),('S9','SL9'),('S8','SL8')]
WINDOWS=[(1,9),(10,46),(47,83),(84,92),(93,120),(121,150),(151,166),(167,184)]
inputs={}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 inputs[str(p.relative_to(ROOT))]=sha(p)
 return json.loads(p.read_text())
def fingerprint(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()
def diffs(a,b):return {k:[a.get(k,'<missing>'),b.get(k,'<missing>')] for k in sorted(set(a)|set(b)) if a.get(k,'<missing>')!=b.get(k,'<missing>')}
def avg(rows,k):return statistics.mean(r[k] for r in rows)
def slope(rows):
 mt=avg(rows,'step');my=avg(rows,'train_nll')
 return sum((r['step']-mt)*(r['train_nll']-my) for r in rows)/sum((r['step']-mt)**2 for r in rows)

preserved={name:sha(OUT/name) for name in ['audit.json','audit.log','step_weights.csv','REPORT.md','verification.json','audit_initial_141646.json','audit_initial_141646.log']}
runs={}
for label,(cohort,arm) in RUNS.items():
 p=ROOT/'logs/muon_spectra'/cohort/arm/'scientific'
 m=read(p/'metadata.json');c=m['config'];status=read(p/'status.json');pipe=read(p.parent/'pipeline.json')
 rows=[read(f) for f in sorted((p/'steps').glob('step*.json'))]; positive=[r for r in rows if r['step']]
 assert status['status']=='complete' and positive[-1]['step']==status['step'] and positive[-1]['tokens']==c['total_tokens']==status['budget_tokens']
 assert [r['step'] for r in positive]==list(range(1,len(positive)+1))
 assert pipe['phase']=='complete' and all(w['state']=='succeeded' and w['exit_code']==0 for w in pipe['workers'])
 assert c.get('muon_prefilter','none')=='none' and not c['muon_nesterov'] and not c['mean_whitening'] and not c['pmuon']
 source_checks={}
 for fname,expected in m['source_sha256'].items():
  source=p/'source'/fname
  actual=sha(source)
  assert actual==expected
  source_checks[fname]=actual
 beta=c['muon_momentum'];wr=[];wn=[];wu=[];steps=[];tokens=[];outrows=[]
 for row in positive:
  norm=row['gradient_norm_before_clip'];ct=min(1.,c['grad_clip']/(norm+1e-6));t=row['step']
  assert row['gradient_clipped']==(norm>c['grad_clip'])
  steps.append(t);tokens.append(row['tokens'])
  wr=[beta*x for x in wr]+[ct];wn=[beta*x for x in wn]+[ct*norm];wu=[beta*x for x in wu]+[1.]
  outrow={k:row[k] for k in ['step','tokens','lr','aux_lr','train_nll','gradient_norm_before_clip','gradient_clipped']}
  outrow.update(label=label,clip_coefficient=ct)
  for name,w in [('raw',wr),('unit',wn),('nominal',wu)]:
   mass=sum(w)
   outrow[name+'_age_steps']=sum((t-s)*v for s,v in zip(steps,w))/mass
   outrow[name+'_age_tokens']=sum((tokens[-1]-s)*v for s,v in zip(tokens,w))/mass
  outrows.append(outrow)
 windows={}
 for low,high in WINDOWS:
  chosen=[r for r in outrows if low<=r['step']<=high]
  if chosen:windows[f'{low}-{high}']={'n':len(chosen),'clipped':sum(r['gradient_clipped'] for r in chosen),**{k:avg(chosen,k) for k in ['raw_age_steps','unit_age_steps','nominal_age_steps','raw_age_tokens','clip_coefficient','train_nll']}}
 runs[label]={'cohort':cohort,'arm':arm,'config':c,'device':m['device_name'],'initial_hash':m['initial_model_sha256'],
              'data_fingerprints':{k:fingerprint(m[k]) for k in ['train_manifest','validation_manifest']},
              'status':status,'pipeline_phase':pipe['phase'],'scientific_worker':next(w for w in pipe['workers'] if w['name']=='scientific'),
              'source_checks':source_checks,'validation':{str(r['step']):r['validation_nll'] for r in rows if 'validation_nll'in r},
              'total_clipped':sum(r['gradient_clipped'] for r in outrows),'windows':windows,'rows':outrows}
pairs={}
for a,b in PAIRS:
 ra,rb=runs[a],runs[b];rowsa,rowsb=ra['rows'],rb['rows'];shared=min(len(rowsa),len(rowsb))
 pdiff=[{'step':x['step'],'train_nll_delta':y['train_nll']-x['train_nll'],'raw_age_delta':y['raw_age_steps']-x['raw_age_steps'],
         'unit_age_delta':y['unit_age_steps']-x['unit_age_steps']} for x,y in zip(rowsa[:shared],rowsb[:shared])]
 validation={k:rb['validation'][k]-v for k,v in ra['validation'].items() if k in rb['validation']}
 srcdiff=diffs(ra['source_checks'],rb['source_checks']);allpatch=[]
 for filename in srcdiff:
  pa=ROOT/'logs/muon_spectra'/ra['cohort']/ra['arm']/'scientific/source'/filename
  pb=ROOT/'logs/muon_spectra'/rb['cohort']/rb['arm']/'scientific/source'/filename
  allpatch.extend(difflib.unified_diff(pa.read_text().splitlines(keepends=True),pb.read_text().splitlines(keepends=True),fromfile=str(pa.relative_to(ROOT)),tofile=str(pb.relative_to(ROOT))))
 (OUT/f'completion_sources_{a}_{b}.diff').write_text(''.join(allpatch))
 pairs[a+'->'+b]={'same_init':ra['initial_hash']==rb['initial_hash'],'same_data':ra['data_fingerprints']==rb['data_fingerprints'],'same_device':ra['device']==rb['device'],
                  'same_lr_shared_steps':all(x['lr']==y['lr'] and x['aux_lr']==y['aux_lr'] for x,y in zip(rowsa[:shared],rowsb[:shared])),
                  'config_diff':diffs(ra['config'],rb['config']),'source_changes':srcdiff,'validation_delta':validation,
                  'windows':{f'{lo}-{hi}':{'n':len(rr),'train_nll_delta':avg(rr,'train_nll_delta'),'raw_age_delta':avg(rr,'raw_age_delta'),'unit_age_delta':avg(rr,'unit_age_delta')} for lo,hi in WINDOWS if (rr:=[r for r in pdiff if lo<=r['step']<=hi])},
                  'step_differences':pdiff}
 if a=='SL9' and b=='SL8':
  # Exploratory local linear conversion; it is not an observed held-out crossing.
  for lo,hi in WINDOWS:
   key=f'{lo}-{hi}'
   if key not in pairs[a+'->'+b]['windows']:continue
   xr=[r for r in rowsa if lo<=r['step']<=hi];yr=[r for r in rowsb if lo<=r['step']<=hi]
   summary=pairs[a+'->'+b]['windows'][key]
   sr=slope(xr);st=slope(yr)
   summary.update(reference_local_train_slope=sr,treatment_local_train_slope=st,
                  linearized_reference_step_lead=summary['train_nll_delta']/sr if sr<0 else None)
assert preserved=={name:sha(OUT/name) for name in preserved}
summary={'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'preserved_prior_artifact_sha256':preserved,'runs':runs,'pairs':pairs,'inputs_sha256':inputs,'source_code_sha256':sha(Path(__file__))}
(OUT/'completion_audit.json').write_text(json.dumps(summary,indent=2)+'\n')
with (OUT/'completion_steps.csv').open('w',newline='') as f:
 allrows=[r for run in runs.values() for r in run['rows']];w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
for name,p in pairs.items():
 print(name,'init/data/device/lr',p['same_init'],p['same_data'],p['same_device'],p['same_lr_shared_steps'],'config',p['config_diff'],'validation',p['validation_delta'])
 print('windows',p['windows'])
for name in ['MC9','MC8','SL9','SL8']:
 r=runs[name];print(name,'clips',r['total_clipped'],'windows',r['windows'])
print('Verified',len(runs),'complete arms',len(allrows),'steps',sum(len(r['source_checks']) for r in runs.values()),'source hashes; prior snapshot hashes unchanged.')

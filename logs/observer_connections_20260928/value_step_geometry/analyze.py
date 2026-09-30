"""Read-only CPU value-step contractions; no model construction or forwards."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import json,hashlib,time,csv
from pathlib import Path
from statistics import median
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[2]
SOURCE=ROOT/'logs/muon_spectra/second_order_audit_20260926/marginals'
STEPS={10,50,100,200,500,900,1300}
def thash(t): return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def fhash(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def energy(w,c,mu):
 wm=w@mu; mean=float(wm@wm); total=float((w@c*w).sum()); centered=total-mean
 assert centered>=0
 return {'mean':mean,'centered':centered,'total':total,'mean_fraction':mean/total}
rows=[];provenance=[];start=time.time()
for p in sorted(SOURCE.glob('*.pt')):
 d=torch.load(p,map_location='cpu',weights_only=False,mmap=True)
 step=d['step']
 if step not in STEPS: continue
 method=p.name.split('_')[0];a=d['arrays'];arm=ROOT/d['arm'];kp=arm/'scientific/kept'/f'step{step:06d}.pt';np=kp.with_name(f'step{step+1:06d}_weights.pt')
 assert np.exists(),np
 stats={str(x.relative_to(ROOT)):(x.stat().st_size,x.stat().st_mtime_ns) for x in [p,kp,np]}
 s=torch.load(kp,map_location='cpu',weights_only=False,mmap=True);n=torch.load(np,map_location='cpu',weights_only=False,mmap=True)
 assert s['step']==step and n['step']==step+1
 cfg=s['config'];assert cfg['data_norm_decay']=='decoupled'
 lr=cfg['learning_rate']*min(1.,(step+1)/max(1,cfg['warmup_steps']))*max(0.,min(1.,(cfg['total_tokens']-s['tokens'])/(cfg['total_tokens']*cfg['cooldown_fraction'])))
 decay=lr*cfg['weight_decay'];wh={};nh={};mh={}
 for layer in range(1,9):
  name=f'blocks.{layer-1}.attn.v.weight';base=f'block{layer:02d}.q';raw=s['model'][name];wn=n['model'][name].double();w=raw.double();D=wn-w
  wh[name]=thash(raw);nh[name]=thash(n['model'][name]);c=a[base]['C_full'].double();mu=a[base]['x_mean'].double()
  mh[base]={k:thash(a[base][k]) for k in ['C_full','x_mean']}
  # Explicit ideal decay; remainder contains adaptive parameter write plus FP32 rounding.
  dec=-decay*w;adapt=D-dec
  e={k:energy(t,c,mu) for k,t in [('W',w),('D',D),('decay',dec),('adaptive_plus_rounding',adapt)]}
  row={'method':method,'step':step,'layer':layer,'lr_next':lr,'wd':cfg['weight_decay'],'input_mean_energy':float(mu@mu),'input_centered_energy':float(c.trace()-mu@mu)}
  for key,v in e.items(): row.update({key+'_'+k:x for k,x in v.items()})
  wm=w@mu;dm=D@mu;am=adapt@mu
  row.update({'D_mean_gain_over_centered_gain':(e['D']['mean']/row['input_mean_energy'])/(e['D']['centered']/row['input_centered_energy']),
   'mean_step_relative':(e['D']['mean']/e['W']['mean'])**.5,'centered_step_relative':(e['D']['centered']/e['W']['centered'])**.5,
   'mean_radial':float(wm@dm),'mean_radial_relative':float(wm@dm/(wm@wm)),
   'adaptive_mean_radial':float(wm@am),'adaptive_mean_radial_relative':float(wm@am/(wm@wm)),
   'mean_cosine_W_D':float(wm@dm/(wm.norm()*dm.norm())),
   'centered_radial':float((w@c*D).sum()-wm@dm),
   'mean_energy_change_frozen_mu':float((wn@mu).square().sum()-wm.square().sum()),
   'D_parameter_energy':float(D.square().sum()),
   'roundoff_bound_vs_D':float((raw*(1-decay)).double().sub(w+dec).norm()/D.norm())})
  # Polarization identities check signed terms independently.
  assert abs(row['mean_energy_change_frozen_mu']-(2*row['mean_radial']+row['D_mean']))<1e-9
  rows.append(row)
 for name,h in wh.items(): assert thash(s['model'][name])==h and thash(n['model'][name])==nh[name]
 for x in [p,kp,np]: assert (x.stat().st_size,x.stat().st_mtime_ns)==stats[str(x.relative_to(ROOT))]
 provenance.append({'marginal':str(p.relative_to(ROOT)),'checkpoint':str(kp.relative_to(ROOT)),'nextweights':str(np.relative_to(ROOT)),'stats':stats,'checkpoint_V_hashes':wh,'next_V_hashes':nh,'marginal_accessed_hashes':mh,'config':cfg,'step':step,'next_step':n['step'],'tokens':s['tokens'],'next_tokens':n['tokens']})
summary=[]
for method in ['M','PD','S','SPD']:
 for step in sorted(STEPS):
  rr=[r for r in rows if r['method']==method and r['step']==step]
  out={'method':method,'step':step,'layers':len(rr),'radial_positive_layers':sum(r['mean_radial']>0 for r in rr),'adaptive_radial_positive_layers':sum(r['adaptive_mean_radial']>0 for r in rr)}
  for k in rows[0]:
   if k not in ['method','step','layer']:out['median_'+k]=median(r[k] for r in rr)
  for obj in ['W','D','decay','adaptive_plus_rounding']:
   for k in ['mean','centered','total']:out['sum_'+obj+'_'+k]=sum(r[obj+'_'+k] for r in rr)
   out['pooled_'+obj+'_mean_fraction']=out['sum_'+obj+'_mean']/out['sum_'+obj+'_total']
  summary.append(out)
for filename,data in [('layers.csv',rows),('summary.csv',summary)]:
 with (HERE/filename).open('w') as f:
  writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
(HERE/'result.json').write_text(json.dumps({'seconds':time.time()-start,'records':rows,'summary':summary,'provenance':provenance,'source_sha256':fhash(Path(__file__))},indent=2)+'\n')
for r in summary:
 print(r['method'],r['step'],'Wmean',round(r['median_W_mean_fraction'],4),'Dmean',round(r['median_D_mean_fraction'],4),'Dmean_energy',round(r['sum_D_mean'],5),'relmean',round(r['median_mean_step_relative'],4),'relcent',round(r['median_centered_step_relative'],4),'radial',r['radial_positive_layers'],'adaptradial',r['adaptive_radial_positive_layers'],flush=True)
print('seconds',time.time()-start,flush=True)

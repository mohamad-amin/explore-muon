"""CPU-only gauge-invariant V/O constant-route contractions."""
import os
os.environ.update(OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
import json,time,hashlib,csv
from pathlib import Path
from statistics import median
import torch
torch.set_num_threads(2);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];SOURCE=ROOT/'logs/muon_spectra/second_order_audit_20260926/marginals'
def h(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
def sq(t):return float(t@t)
def cos(a,b):return float(a@b/a.norm()/b.norm())
rows=[];steps=[];prov=[];started=time.time()
for p in sorted(SOURCE.glob('*.pt')):
 a=torch.load(p,weights_only=False,map_location='cpu',mmap=True);step=a['step'];method=p.name.split('_')[0];arm=ROOT/a['arm'];cp=arm/'scientific'/('checkpoint.pt' if step==1469 else f'kept/step{step:06d}.pt')
 s=torch.load(cp,weights_only=False,map_location='cpu',mmap=True);assert s['step']==step
 nextp=arm/'scientific/kept/step000501_weights.pt' if step==500 else None
 n=torch.load(nextp,weights_only=False,map_location='cpu',mmap=True) if nextp else None
 if n:assert n['step']==501
 sourcepaths=[p,cp]+([nextp] if nextp else []);stats={str(x.relative_to(ROOT)):[x.stat().st_size,x.stat().st_mtime_ns] for x in sourcepaths};hashes={};nh={};mh={}
 for layer in range(1,9):
  vn=f'blocks.{layer-1}.attn.v.weight';on=f'blocks.{layer-1}.attn.proj.weight'
  if on not in s['model']:on=f'blocks.{layer-1}.attn.o.weight'
  v=s['model'][vn].double();o=s['model'][on].double();mu=a['arrays'][f'block{layer:02d}.q']['x_mean'].double();muz=a['arrays'][f'block{layer:02d}.o']['x_mean'].double()
  hashes.update({k:h(s['model'][k]) for k in [vn,on]});mh[str(layer)]={'mu':h(a['arrays'][f'block{layer:02d}.q']['x_mean']),'muz':h(a['arrays'][f'block{layer:02d}.o']['x_mean'])}
  vv=v@mu;c=o@vv;m=o@muz;r=m-c
  row={'method':method,'step':step,'layer':layer,'input_mean_energy':sq(mu),'preO_constant_energy':sq(vv),'preO_mean_energy':sq(muz),'preO_cosine':cos(vv,muz),'constant_energy':sq(c),'mean_energy':sq(m),'selection_shift_energy':sq(r),'cosine_constant_mean':cos(c,m),'relative_mismatch':float(r.norm()/m.norm()),'constant_norm_over_mean_norm':float(c.norm()/m.norm()),'constant_projection_on_mean':float(c@m/sq(m)),'constant_selection_cross':float(c@r),'postO_over_preO_constant_gain':sq(c)/sq(vv),'postO_over_preO_mean_gain':sq(m)/sq(muz)}
  assert abs(sq(m)-(sq(c)+sq(r)+2*float(c@r)))<1e-9
  rows.append(row)
  if n:
   nv=n['model'][vn].double();no=n['model'][on].double();dv=nv-v;do=no-o
   x=o@(dv@mu);y=do@(v@mu);z=do@(dv@mu);total=no@(nv@mu)-c
   assert float((total-x-y-z).norm())<1e-12
   rr={'method':method,'step':step,'layer':layer,'c_energy':sq(c),'Vonly_energy':sq(x),'Oonly_energy':sq(y),'VOcross_energy':sq(z),'joint_energy':sq(total),'V_O_cosine':cos(x,y),'joint_relative_norm':float(total.norm()/c.norm()),'Vonly_relative_norm':float(x.norm()/c.norm()),'Oonly_relative_norm':float(y.norm()/c.norm()),'joint_radial':float(c@total),'Vonly_radial':float(c@x),'Oonly_radial':float(c@y),'cross_V_O':float(x@y),'cross_V_VO':float(x@z),'cross_O_VO':float(y@z)}
   assert abs(sq(total)-(sq(x)+sq(y)+sq(z)+2*(float(x@y)+float(x@z)+float(y@z))))<1e-10
   steps.append(rr);nh.update({k:h(n['model'][k]) for k in [vn,on]})
 for k,hh in hashes.items():assert h(s['model'][k])==hh
 if n:
  for k,hh in nh.items():assert h(n['model'][k])==hh
 for x in sourcepaths:assert [x.stat().st_size,x.stat().st_mtime_ns]==stats[str(x.relative_to(ROOT))]
 prov.append({'method':method,'step':step,'checkpoint':str(cp.relative_to(ROOT)),'marginal':str(p.relative_to(ROOT)),'nextweights':str(nextp.relative_to(ROOT)) if nextp else None,'stats':stats,'checkpoint_VO_hashes':hashes,'next_VO_hashes':nh,'mean_hashes':mh})
summary=[]
for method in ['M','PD','S','SPD']:
 for step in [10,50,100,200,500,900,1300,1469]:
  rr=[r for r in rows if r['method']==method and r['step']==step];out={'method':method,'step':step,'layers':len(rr)}
  for k in rows[0]:
   if k not in ['method','step','layer']:out['median_'+k]=median(r[k] for r in rr)
  for k in ['constant_energy','mean_energy','selection_shift_energy']:out['sum_'+k]=sum(r[k] for r in rr)
  summary.append(out)
ss=[]
for method in ['M','PD','S','SPD']:
 rr=[r for r in steps if r['method']==method];out={'method':method,'step':500,'layers':len(rr),'joint_radial_positive_layers':sum(r['joint_radial']>0 for r in rr)}
 for k in steps[0]:
  if k not in ['method','step','layer']:out['median_'+k]=median(r[k] for r in rr)
 for k in ['c_energy','Vonly_energy','Oonly_energy','VOcross_energy','joint_energy','cross_V_O','cross_V_VO','cross_O_VO']:out['sum_'+k]=sum(r[k] for r in rr)
 ss.append(out)
for fn,rr in [('layers.csv',rows),('summary.csv',summary),('step500_layers.csv',steps),('step500_summary.csv',ss)]:
 with (HERE/fn).open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
(HERE/'result.json').write_text(json.dumps({'seconds':time.time()-started,'records':rows,'summary':summary,'step_records':steps,'step_summary':ss,'provenance':prov,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},indent=2)+'\n')
for r in summary:
 print(r['method'],r['step'],'constE',round(r['sum_constant_energy'],5),'meanE',round(r['sum_mean_energy'],5),'cos',round(r['median_cosine_constant_mean'],4),'mismatch',round(r['median_relative_mismatch'],4),'c/m',round(r['median_constant_norm_over_mean_norm'],4),flush=True)
for r in ss:print('STEP',r,flush=True)
print('seconds',time.time()-started,flush=True)

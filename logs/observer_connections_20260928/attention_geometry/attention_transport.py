"""CPU contractions of saved marginal statistics and sparse mmap'd V weights.
No model construction or forward. Read-only access to checkpoint tensor storage.
"""
import os
os.environ['OMP_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import torch,json,hashlib,math,time
from pathlib import Path
from collections import defaultdict
from statistics import median

torch.set_num_threads(2);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SOURCE=ROOT/'logs/muon_spectra/second_order_audit_20260926/marginals'

def tensor_hash(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()
records=[];provenance=[];start=time.time()
for p in sorted(SOURCE.glob('*.pt')):
    d=torch.load(p,map_location='cpu',weights_only=False);a=d['arrays'];method=p.name.split('_')[0]
    step=d['step'];checkpoint=ROOT/d['arm']/'scientific'/('checkpoint.pt' if step==1469 else f'kept/step{step:06d}.pt')
    stat=checkpoint.stat();s=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert s['step']==step
    weights={}
    for layer in range(1,9):
        base=f'block{layer:02d}'; q=a[base+'.q']; k=a[base+'.k'];v=a[base+'.v'];o=a[base+'.o']
        name=f'blocks.{layer-1}.attn.v.weight';w=s['model'][name].double();weights[name]=tensor_hash(s['model'][name])
        c=q['C_full'].double(); mu=q['x_mean'].double(); before_mu=w@mu;after_mu=o['x_mean'].double()
        before_second=float((w@c*w).sum());after_second=o['trC'];before_mean=float(before_mu@before_mu);after_mean=float(after_mu@after_mu)
        # Because A1=1, Ax = 1*mu + A(x-mu) exactly; transport shift is data-dependent mean of A(x-mu).
        shift=after_mu-before_mu
        r={'method':method,'step':step,'layer':layer,
           'before_second':before_second,'after_second':after_second,
           'input_mean_share':float(mu@mu/c.trace()),
           'mean_gain_over_centered_gain':float((before_mean/(mu@mu))/((before_second-before_mean)/(c.trace()-mu@mu))),
           'total_energy_transport':after_second/before_second,
           'centered_energy_transport':(after_second-after_mean)/(before_second-before_mean),
           'pre_value_mean_share':before_mean/before_second,
           'post_attention_mean_share':after_mean/after_second,
           'mean_cosine':float(before_mu@after_mu/(before_mu.norm()*after_mu.norm())),
           'selection_mean_shift_relative':float(shift.norm()/after_mu.norm()),
           'key_B_top8_share':float(k['lam_B'][:8].sum()/k['lam_B'].sum()),
           'query_B_top8_share':float(q['lam_B'][:8].sum()/q['lam_B'].sum()),
           'value_B_top8_share':float(v['lam_B'][:8].sum()/v['lam_B'].sum())}
        records.append(r)
    stat2=checkpoint.stat();assert (stat.st_size,stat.st_mtime_ns)==(stat2.st_size,stat2.st_mtime_ns)
    # Hash loaded model tensors again; no full checkpoint hash: irrelevant optimizer and embedding tensors remain mmap'd.
    assert all(tensor_hash(s['model'][name])==h for name,h in weights.items())
    provenance.append({'checkpoint':str(checkpoint.relative_to(ROOT)),'size':stat.st_size,'mtime_ns':stat.st_mtime_ns,'accessed_tensor_hashes':weights,'config_model':s['config']['model']})
(HERE/'attention_transport.json').write_text(json.dumps({'seconds':time.time()-start,'records':records,'provenance':provenance},indent=2)+'\n')
print('seconds',time.time()-start)
for method in ['M','PD','S','SPD']:
    for step in [10,50,100,200,500,900,1300,1469]:
        rr=[r for r in records if r['method']==method and r['step']==step]
        print(method,step,' '.join(f'{median(r[k] for r in rr):.3f}' for k in ['pre_value_mean_share','post_attention_mean_share','centered_energy_transport','total_energy_transport','mean_cosine','selection_mean_shift_relative','key_B_top8_share','query_B_top8_share']))

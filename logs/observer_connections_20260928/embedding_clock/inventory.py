"""Read actual batch support and compare inactive-row Adam arithmetic."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','2');os.environ.setdefault('MKL_NUM_THREADS','2')
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np
import torch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE.parent/'aux_partition/source'))
from adamw_spectra.data import TokenStream
torch.set_num_threads(2);torch.set_num_interop_threads(1)
start=time.time();arm=ROOT/'logs/muon_spectra/soaudit_mom4m_20260927/M_b4M_lr0.014_mom0.9_s260925_l40s'
paths=[arm/'scientific/kept/step000183.pt',arm/'scientific/kept/step000184_weights.pt']
load=lambda p:torch.load(p,map_location='cpu',weights_only=False,mmap=True)
a,b=[load(p) for p in paths]
assert a['step']==183 and b['step']==184
old=json.loads((HERE.parent/'body_aux/result.json').read_text())
aux=old['partition']['aux'];ids=a['optimizer']['param_groups'][1]['params']
assert aux[0]=='embed.weight' and len(ids)==len(aux)
key='embed.weight';state=a['optimizer']['state'][ids[0]];group=a['optimizer']['param_groups'][1]
assert int(state['step'])==183
step=json.loads((arm/'scientific/steps/step000184.json').read_text())
lr=step['aux_lr'];assert lr==group['lr']==.002
stream=TokenStream(str(ROOT/a['config']['train_pattern']))
count=a['config']['batch_tokens'];x=stream.device_tokens(a['tokens'],count+1,torch.device('cpu'))[:-1]
counts=torch.bincount(x,minlength=a['model'][key].shape[0]);assert int(counts.sum())==count
inactive=counts==0
w=a['model'][key].double();nxt=b['model'][key].double();delta=nxt-w
m=state['exp_avg'].double();v=state['exp_avg_sq'].double();beta1,beta2=group['betas'];t=184
pred=-lr*group['weight_decay']*w-lr*(beta1*m/(1-beta1**t))/((beta2*v/(1-beta2**t)).sqrt()+group['eps'])
error=delta[inactive]-pred[inactive]
bound=torch.maximum(torch.full_like(error,2e-7),8*torch.finfo(torch.float32).eps*w[inactive].abs())
assert bool((error.abs()<=bound).all())
den=float(delta.square().sum());rows={}
for name,mask in [('active',~inactive),('inactive',inactive),('inactive_nonzero_second_moment',inactive&(v.sum(1)>0)),('inactive_zero_second_moment',inactive&(v.sum(1)==0))]:
    rows[name]={'row_count':int(mask.sum()),'row_fraction':float(mask.double().mean()),
                'actual_delta_norm':float(delta[mask].norm()),'delta_energy_fraction':float(delta[mask].square().sum())/den}
eval_tokens=np.load(HERE.parent/'aux_partition/run1/tokens.npz');support={}
for name in eval_tokens.files:
    if not name.endswith('_x'):continue
    ids_eval=torch.from_numpy(eval_tokens[name].copy())
    mask=inactive[ids_eval]
    support[name]={'input_tokens':ids_eval.numel(),'tokens_on_inactive_rows':int(mask.sum()),
                   'unique_inactive_rows':int(torch.unique(ids_eval[mask]).numel())}
out={'status':'complete','step_pair':[183,184],'training_input_offset':a['tokens'],'training_input_count':count,
     'groups':rows,'probe_input_support':support,
     'inactive_prediction':{'max_absolute_error':float(error.abs().max()),'rms_error':float(error.square().mean().sqrt()),
                            'relative_frobenius_error':float(error.norm()/delta[inactive].norm()),'max_fraction_of_fp32_bound':float((error.abs()/bound).max())},
     'optimizer':{k:group[k] for k in ['lr','betas','eps','weight_decay','fused']},
     'tensor_sha256':{n:hashlib.sha256(z.detach().contiguous().numpy().tobytes()).hexdigest() for n,z in [('old_weight',a['model'][key]),('next_weight',b['model'][key]),('exp_avg',state['exp_avg']),('exp_avg_sq',state['exp_avg_sq'])]},
     'input_counts_sha256':hashlib.sha256(counts.numpy().tobytes()).hexdigest(),
     'seconds':time.time()-start,
     'scope':'Actual displacements and known zero-gradient row recurrence; movement without current exposure need not be useless or harmful. No model or optimizer call.'}
np.savez(HERE/'counts.npz',counts=counts.numpy())
(HERE/'result.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

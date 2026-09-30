"""Saved-tensor architectural metric decomposition, with no network calls."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import csv
import hashlib
import json
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
torch.set_num_threads(1);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];RUN=HERE/'run1';OUT=RUN/'transmission_analysis'
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig_transmission')


def gelu_prime(z):
    c=.044715;k=math.sqrt(2/math.pi);u=k*(z+c*z**3);t=torch.tanh(u)
    return .5*(1+t)+.5*z*(1-t*t)*k*(1+3*c*z*z)


def sha_tensor(t):return hashlib.sha256(t.contiguous().numpy().tobytes()).hexdigest()


def main():
    started=time.monotonic();assert json.loads((RUN/'status.json').read_text())['status']=='complete'
    assert json.loads((RUN/'analysis/result.json').read_text())['status']=='complete'
    expected={(m,s,b) for m in ['Muon','PD'] for s in [10,500,1300] for b in [1,4,8]}
    observed=[]
    for file in RUN.glob('*/block*_up/summary.json'):
        item=json.loads(file.read_text());observed.append((item['method'],item['step'],item['block']))
    assert len(observed)==18 and set(observed)==expected
    OUT.mkdir(exist_ok=False)
    z=torch.linspace(-12,12,1001,dtype=torch.float64,requires_grad=True)
    reference,=torch.autograd.grad(F.gelu(z,approximate='tanh').sum(),z)
    error=float((reference-gelu_prime(z)).abs().max());assert error<1e-13
    rows=[];contrasts=[];sources=[];raw={};kick_error=0.;identity_error=0.
    for state in sorted(p for p in RUN.iterdir() if p.is_dir() and p.name.startswith(('Muon_','PD_'))):
        meta=json.loads((state/'metadata.json').read_text());record=meta['provenance']['files'][0]
        checkpoint=ROOT/record['path'];info=(checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)
        assert info==(record['bytes'],record['mtime_ns'])
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
        assert saved['step']==meta['step']
        for panel in sorted(state.glob('block*_up')):
            summary=json.loads((panel/'summary.json').read_text());b=summary['block']-1
            assert summary['step']==saved['step'] and summary['method']==meta['method']
            t=torch.load(panel/'tensors.pt',map_location='cpu',weights_only=True,mmap=True)
            a=np.load(panel/'per_token.npz');name=f'blocks.{b}.mlp.down.weight'
            labels=['actual','left0','left1','right_raw','right_white']
            assert a['labels'].tolist()==labels and list(t['directions'])==labels
            assert t['score_inputs'].shape==(4,512,512) and t['score_preactivations'].shape==(4,512,2048)
            down=saved['model'][name];digest=sha_tensor(down)
            assert digest==record['model_tensor_sha256'][name]
            assert down.shape==(512,2048)
            sources.append(dict(checkpoint=record['path'],tensor=name,sha256=digest))
            w=down.double();prime=gelu_prime(t['score_preactivations']);x=t['score_inputs']
            common={k:summary[k] for k in ['method','step','block']}
            cells={}
            for di,label in enumerate(a['labels'].tolist()):
                assert t['directions'][label].shape==(2048,512)
                assert sha_tensor(t['directions'][label])==summary['direction_tensor_hashes'][label]
                dz=F.linear(x,t['directions'][label]);dr=F.linear(dz*prime,w)
                pre=dz.square().sum(-1).detach().numpy();post=dr.square().sum(-1).detach().numpy()
                err=float(np.max(abs(pre-a['activation_radius'][:,di]**2)))
                kick_error=max(kick_error,err);assert err<1e-9*max(1.,float(pre.max()))
                assert np.isfinite(pre).all() and np.isfinite(post).all()
                key=f"{common['method']}_{common['step']:06d}_block{common['block']:02d}_{label}"
                raw[key+'_pre']=pre;raw[key+'_post']=post
                for aggregate,idx in [('bank0',[0,1]),('bank1',[2,3]),('all4',[0,1,2,3])]:
                    ep=float(pre[idx].mean());er=float(post[idx].mean());gn=float(a['gn'][idx,di].mean())
                    h=float(a['hessian'][idx,di].mean());slope=float(a['slope'][idx,di].mean())
                    assert ep>0 and er>=0 and gn>=0
                    trans=er/ep;remaining=gn/er if er>0 else None;lhs=gn/ep
                    resid=abs(lhs-trans*remaining) if remaining is not None else None
                    if resid is not None:
                        identity_error=max(identity_error,resid);assert resid<1e-12*max(1.,abs(lhs))
                    row=dict(**common,direction=label,aggregate=aggregate,preactivation_energy=ep,
                        residual_tangent_energy=er,GN=gn,true_Hessian=h,slope=slope,
                        transmission_gain=trans,GN_per_preactivation_energy=lhs,
                        GN_per_residual_energy=remaining,pooled_identity_error=resid,
                        residual_ratio_status='defined' if er>0 else 'zero_residual_tangent_energy')
                    rows.append(row);cells[(aggregate,label)]=row
            for aggregate in ['bank0','bank1','all4']:
                actual=cells[(aggregate,'actual')]
                for label in ['left0','left1','right_raw','right_white']:
                    other=cells[(aggregate,label)]
                    def ratio(key):
                        den=other[key];num=actual[key]
                        return num/den if den is not None and den>0 and num is not None else None
                    contrasts.append(dict(**common,aggregate=aggregate,control=label,
                        actual_over_control_GN_per_pre=ratio('GN_per_preactivation_energy'),
                        actual_over_control_transmission=ratio('transmission_gain'),
                        actual_over_control_GN_per_residual=ratio('GN_per_residual_energy'),
                        ratio_status=('defined' if all(ratio(k) is not None for k in
                            ['GN_per_preactivation_energy','transmission_gain','GN_per_residual_energy'])
                            else 'zero_or_undefined_control_denominator')))
            del t,a,w,x,prime,dz,dr
        assert (checkpoint.stat().st_size,checkpoint.stat().st_mtime_ns)==info
        del saved
    assert len(rows)==18*5*3 and len(contrasts)==18*4*3
    for name,values in [('metrics.csv',rows),('contrasts.csv',contrasts)]:
        with (OUT/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    np.savez(OUT/'token_energies.npz',**raw)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    for mi,method in enumerate(['Muon','PD']):
        for bi,block in enumerate([1,4,8]):
            ax=axs[mi,bi]
            for label,style in [('left0','-'),('left1','--')]:
                subset=sorted([r for r in contrasts if r['method']==method and r['block']==block and r['aggregate']=='all4' and r['control']==label],key=lambda r:r['step'])
                for key,color,caption in [('actual_over_control_GN_per_pre','#0072B2','Before transmission normalization'),
                                          ('actual_over_control_transmission','#009E73','MLP transmission factor'),
                                          ('actual_over_control_GN_per_residual','#D55E00','Remaining downstream factor')]:
                    ax.plot([r['step'] for r in subset],[r[key] if r[key] is not None else np.nan for r in subset],style,color=color,marker='o',label=caption if label=='left0' else None)
            ax.axhline(1,color='grey',lw=.7);ax.set(title=f'{method}, block{block}',xlabel='Checkpoint update',ylabel='Actual / rotated contrast',yscale='log');ax.grid(alpha=.2)
    axs[0,0].legend(frameon=False,fontsize=8);fig.suptitle('Left-rotation GN contrast decomposed through the fixed MLP Jacobian\nFour-context means; solid/dashed are two rotations; no fitted metric or causal attribution')
    fig.tight_layout();fig.savefig(OUT/'transmission_contrasts.png',dpi=170);fig.savefig(OUT/'transmission_contrasts.pdf');plt.close(fig)
    result=dict(status='complete',seconds=time.monotonic()-started,panels=18,rows=len(rows),contrasts=len(contrasts),
        gelu_derivative_error=error,max_archived_kick_energy_error=kick_error,max_pooled_identity_error=identity_error,
        source_tensor_records=sources,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        undefined_contrasts=sum(r['ratio_status']!='defined' for r in contrasts),
        scope='Retrospective individual-up tangent decomposition only; pooled sequence energies, not tokenwise early-layer causal attributions or finite joint displacements.')
    (OUT/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_tensor_records'}),flush=True)


if __name__=='__main__':main()

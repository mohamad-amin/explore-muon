"""Connect archived displacement moments to the spherical model; no forwards."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
HERE=Path(__file__).resolve().parent
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig_geometry')
RUN=HERE/'run1';OUT=RUN/'geometry_analysis'


def write_csv(name,rows):
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    start=time.monotonic()
    assert json.loads((RUN/'status.json').read_text())['status']=='complete'
    panels=sorted(RUN.glob('*/block*/summary.json'));assert len(panels)==18
    OUT.mkdir(exist_ok=False)
    rows=[];inputs=[];spectra=[];states=[];curvature_panels=[]
    for file in panels:
        summary=json.loads(file.read_text());p=file.parent
        a=np.load(p/'per_token.npz');t=torch.load(p/'tensors.pt',map_location='cpu',weights_only=True,mmap=True)
        x=t['score_inputs'];white=x@t['S_inverse'];n=x.shape[-1]
        common={k:summary[k] for k in ['method','step','block']}
        for frame,coords in [('raw',x),('white',white)]:
            norms=coords.norm(dim=-1)
            assert torch.isfinite(norms).all() and float(norms.min())>0
            unit=coords/norms[...,None]
            angular=unit.flatten(0,1);mu=angular.mean(0)
            cov=angular.T@angular/angular.shape[0]
            inputs.append(dict(**common,frame=frame,mean_input_radius=float(norms.mean()),
                radius_cv=float(norms.std(unbiased=False)/norms.mean()),
                normalized_direction_mean_norm=float(mu.norm()),
                angular_second_moment_isotropy_error=float((n*cov-torch.eye(n)).norm()/n**.5)))
            for di,name in enumerate(a['labels'].tolist()):
                d=t['directions'][name];q=d if frame=='raw' else d@t['S']
                gram=q.T@q;tr=float(gram.trace());tr2=float(gram.square().sum())
                assert tr>0 and tr2>0
                eff=tr*tr/tr2;sphere_mean=tr/n
                sphere_cv2=max(0.,2/(n+2)*(n/eff-1))
                # The raw kick is Q*coords in either chart; normalization isolates angular input geometry.
                squared=a['activation_radius'][:,di]**2/npy(norms.square())
                for bank,idx in [('bank0',[0,1]),('bank1',[2,3]),('all',[0,1,2,3])]:
                    z=squared[idx];mean=float(z.mean());var=float(z.var())
                    rows.append(dict(**common,frame=frame,direction=name,bank=bank,
                        effective_rank=eff,spherical_mean_squared_radius=sphere_mean,
                        spherical_relative_variance=sphere_cv2,observed_mean_squared_radius=mean,
                        observed_relative_variance=var/(mean*mean),
                        observed_to_spherical_mean=mean/sphere_mean,
                        observed_to_spherical_relative_variance=(var/(mean*mean)/sphere_cv2 if sphere_cv2>1e-12 else None)))
        for kind,v in t['singular_values'].items():
            v=npy(v);total=float(np.sum(v*v))
            for rank,value in enumerate(v,1):spectra.append(dict(**common,quantity=kind,rank=rank,
                singular_value=float(value),normalized_singular_value=float(value/total**.5)))
        h=np.asarray(summary['projected_hessian_mean']);gn=np.asarray(summary['projected_gn_mean'])
        curvature_panels.append(dict(**common,H=h,GN=gn,labels=a['labels'].tolist()))
        weight_gram=npy(t['parameter_gram'])
        evals,vec=np.linalg.eigh(weight_gram)
        assert evals[-1]>0 and evals.min()>-1e-10*evals[-1]
        cutoff=evals[-1]*1e-12;keep=evals>cutoff
        assert keep.any()
        basis=vec[:,keep]/np.sqrt(evals[keep])[None,:]
        he=np.linalg.eigvalsh(basis.T@h@basis);ge=np.linalg.eigvalsh(basis.T@gn@basis)
        assert ge.min()>-1e-8*max(1.,ge.max())
        states.append(dict(**common,span_dimension=int(keep.sum()),
            parameter_gram_condition=float(evals[-1]/evals[keep][0]),
            parameter_gram_eigenvalues=evals.tolist(),gram_rank_cutoff=float(cutoff),
            true_hessian_restricted_eigenvalues=he.tolist(),GN_restricted_eigenvalues=ge.tolist(),
            scope='Operator restricted to the five-direction span after Euclidean orthonormalization; not full parameter eigenvalues.'))
        del t,a,x,white
    write_csv('radius_moments.csv',rows);write_csv('input_geometry.csv',inputs);write_csv('spectra.csv',spectra)
    (OUT/'restricted_curvature.json').write_text(json.dumps(states,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    colors={'gradient':'#0072B2','momentum':'#D55E00','actual_write':'#009E73'}
    with PdfPages(OUT/'coefficient_curvature_matrices.pdf') as pdf:
        for panel in curvature_panels:
            fig,axs=plt.subplots(1,3,figsize=(12,4))
            lim=max(float(np.abs(panel['H']).max()),float(np.abs(panel['GN']).max()),
                    float(np.abs(panel['H']-panel['GN']).max()),1e-15)
            for ax,key,title in zip(axs,['H','GN','difference'],['True Hessian','Predictive GN','True Hessian − GN']):
                matrix=panel['H']-panel['GN'] if key=='difference' else panel[key]
                im=ax.imshow(matrix,cmap='RdBu_r',vmin=-lim,vmax=lim)
                ax.set_xticks(range(5),panel['labels'],rotation=40,ha='right');ax.set_yticks(range(5),panel['labels'])
                ax.set_title(title)
            fig.colorbar(im,ax=list(axs),shrink=.75,label='Curvature in actual-scale direction coefficients')
            fig.suptitle(f"{panel['method']} state{panel['step']} block{panel['block']}: finite nonorthonormal direction span")
            fig.subplots_adjust(top=.82,bottom=.25,wspace=.45,right=.84);pdf.savefig(fig);plt.close(fig)
    with PdfPages(OUT/'gradient_momentum_write_spectra.pdf') as pdf:
        for method in ['Muon','PD']:
            fig,axs=plt.subplots(3,3,figsize=(12,10),sharex=True)
            for row,step in enumerate([10,500,1300]):
                for col,block in enumerate([1,4,8]):
                    ax=axs[row,col]
                    for kind,color in colors.items():
                        r=[v for v in spectra if v['method']==method and v['step']==step and v['block']==block and v['quantity']==kind]
                        ax.loglog([v['rank'] for v in r],[v['normalized_singular_value'] for v in r],color=color,label=kind)
                    ax.set_title(f'{method}, state {step}, block {block}')
                    ax.set_xlabel('Singular rank');ax.set_ylabel('Singular value / Frobenius norm');ax.grid(alpha=.2)
            axs[0,0].legend(frameon=False,fontsize=8);fig.suptitle('Measured gradient, stored lagged momentum, and actual next write are separate objects')
            fig.tight_layout();pdf.savefig(fig);fig.savefig(OUT/f'{method}_spectra.png',dpi=150);plt.close(fig)
    fig,axs=plt.subplots(2,3,figsize=(12,7),sharex=True,sharey=True)
    for row,method in enumerate(['Muon','PD']):
        for col,block in enumerate([1,4,8]):
            ax=axs[row,col]
            for frame,color in [('raw','#0072B2'),('white','#D55E00')]:
                r=[v for v in rows if v['method']==method and v['block']==block and v['frame']==frame and v['bank']=='all' and v['direction']=='actual']
                r.sort(key=lambda v:v['step'])
                ax.plot([v['step'] for v in r],[v['observed_relative_variance'] for v in r],'o-',color=color,label=f'{frame}: observed')
                ax.plot([v['step'] for v in r],[v['spherical_relative_variance'] for v in r],'s--',color=color,label=f'{frame}: spherical')
            ax.set_title(f'{method}, block {block}');ax.set(xlabel='Checkpoint update',ylabel='Relative variance of squared angular kick',yscale='log');ax.grid(alpha=.2)
    axs[0,0].legend(frameon=False,fontsize=8);fig.suptitle('Actual-write norm dispersion: measured input directions versus spherical prediction\nWhitening uses regularized C; score-input lengths are normalized separately in each chart')
    fig.tight_layout();fig.savefig(OUT/'norm_dispersion.png',dpi=170);fig.savefig(OUT/'norm_dispersion.pdf');plt.close(fig)
    result=dict(status='complete',seconds=time.monotonic()-start,panels=18,radius_rows=len(rows),
        input_rows=len(inputs),spectral_rows=len(spectra),
        limitations='Descriptive finite-context moments; four contexts are not2048independent samples. Restricted curvature spectra concern only recorded finite spans.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


def npy(t):return t.detach().numpy()


if __name__=='__main__':main()

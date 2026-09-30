import os
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'
import json, hashlib, csv, math, time
from pathlib import Path
import torch

torch.set_num_threads(2)
torch.set_num_interop_threads(1)
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SOURCE=ROOT/'logs/muon_spectra/second_order_audit_20260926/marginals'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def fit(E, xs):
    scales=torch.tensor([x.norm() for x in xs],dtype=torch.float64)
    xn=[x/s for x,s in zip(xs,scales)]
    g=torch.tensor([[float((x*y).sum()) for y in xn] for x in xn],dtype=torch.float64)
    r=torch.tensor([float((E*x).sum()) for x in xn],dtype=torch.float64)
    coef=torch.linalg.solve(g,r)/scales
    residual=float((E-sum(c*x for c,x in zip(coef,xs))).norm()/E.norm())
    return {'coefficients':coef.tolist(),'residual':residual,'scaled_gram_condition':float(torch.linalg.cond(g))}

started=time.time(); hashes={}; rows=[]; reports=[]
for path in sorted(SOURCE.glob('*.pt')):
    hashes[str(path.relative_to(ROOT))]=digest(path)
    jpath=path.with_suffix('.json');hashes[str(jpath.relative_to(ROOT))]=digest(jpath)
    meta=json.loads(jpath.read_text()); saved=torch.load(path,map_location='cpu',weights_only=False)
    method=path.name.split('_')[0];step=meta['step']; arrays=saved['arrays']
    for name,m in meta['matrices'].items():
        layer,kind=name.split('.')
        row={'method':method,'step':step,'layer':int(layer[-2:]),'kind':kind,
             'a_within':m['fit_exact']['a_within'],'b_between':m['fit_exact']['b_between'],
             'fit_residual':m['fit_exact']['residual_fit'],'kfac_residual':m['fit_exact']['residual_kfac'],
             'trace_exact_kfac':m['in']['trace_ratio_exact_kfac'],
             'trace_token_kfac':m['in']['trace_ratio_token_kfac'],
             'mean_direction_exact_token':m['in']['mean_direction']['exact_over_token'],
             'mean_direction_C_rayleigh_share':m['in']['mean_direction']['share_of_C'],
             'input_exact_unit_fro_diff':m['in']['unit_trace_rel_frobenius_diff']['exact'],
             'output_exact_unit_fro_diff':m['out']['unit_trace_rel_frobenius_diff']['exact'],
             'input_top8_overlap':m['in']['top8_overlap_with_factor']['exact'],
             'output_top8_overlap':m['out']['top8_overlap_with_factor']['exact']}
        rows.append(row)
        if kind not in ('q','k','v'):continue
        arr=arrays[name]; common=arrays[layer+'.q']
        C=common['C_full'].double();Bt=common['C_between_full'].double();mu=common['x_mean'].double()
        E=arr['exact_in_full'].double()/arr['trB'];Z=mu[:,None]*mu[None,:];X=C-Bt;Y=Bt-Z
        two=fit(E,[X,Bt]);three=fit(E,[X,Y,Z]);rankone=fit(E,[C,Z])
        Enorm=float(E.norm()); u=mu/mu.norm(); eu=E@u;cu=C@u
        # Frobenius norm of PAP without allocating P and dense multiplications.
        def centered_fro(A,Au):
            return math.sqrt(max(0,float((A*A).sum()-2*(Au*Au).sum()+(u@Au)**2)))
        eperp=centered_fro(E,eu); delta=E-C;du=eu-cu
        delta_perp=centered_fro(delta,du)
        rec={'method':method,'step':step,'layer':int(layer[-2:]),'kind':kind,
             'global_mean_share':float(Z.trace()/C.trace()),'between_cov_share':float(Y.trace()/C.trace()),
             'within_share':float(X.trace()/C.trace()),
             'global_fraction_of_between_trace':float(Z.trace()/Bt.trace()),
             'mean_rayleigh_share':float(u@cu/C.trace()),
             'mean_exact_trace_share':float(u@eu/E.trace()),
             'mean_exact_kfac':float(u@eu/(u@cu)),
             'kfac_residual':float(delta.norm()/E.norm()),
             'kfac_perp_residual':delta_perp/eperp,
             'kfac_error_energy_outside_mean':delta_perp**2/float((delta*delta).sum()),
             'two_fit':two,'three_fit':three,'rankone_fit':rankone,
             'two_archive_absdiff':abs(two['residual']-m['fit_exact']['residual_fit'])}
        reports.append(rec)
    print(method,step,'elapsed',round(time.time()-started,2),flush=True)
(HERE/'results.json').write_text(json.dumps({'protocol':'PROTOCOL.md','seconds':time.time()-started,'records':reports},indent=2)+'\n')
with (HERE/'all_kind_summary.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
unchanged=all(digest(ROOT/p)==h for p,h in hashes.items())
assert unchanged
(HERE/'INPUTS.json').write_text(json.dumps({'inputs':hashes,'unchanged_on_completion':unchanged},indent=2)+'\n')
print('complete',len(reports),len(rows),round(time.time()-started,2),flush=True)

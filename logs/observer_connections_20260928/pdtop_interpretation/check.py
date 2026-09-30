"""Small deterministic algebra and four existing scalar preflight records."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[key]='2'
from pathlib import Path
import csv
import hashlib
import json
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def polar(x):
    u,s,vt=np.linalg.svd(x,full_matrices=False)
    assert min(s)>1e-10
    return u@vt


def matched(m,r):
    f=polar(m@r)@r
    return f*np.sqrt(min(m.shape))/np.linalg.norm(f)


def main():
    assert not (HERE/'result.json').exists()
    started=time.monotonic();manifest={}
    def read(p):
        b=p.read_bytes();manifest[str(p.relative_to(ROOT))]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()};return b
    for p in [Path(__file__), HERE/'PROTOCOL.md', HERE.parent/'progress_lead/PD_TOP_INTERPRETATION.md',
              ROOT/'research/adamw_spectra/muon.py',ROOT/'logs/muon_spectra/second_order_audit_20260926/step_profile_probe.py']:
        read(p)
    rng=np.random.default_rng(260928)
    checks=[]
    for shape in [(6,6),(12,6),(6,12)]:
        m=rng.normal(size=shape)
        q,_=np.linalg.qr(rng.normal(size=(shape[1],shape[1])))
        eig=np.geomspace(.05,4,shape[1]);r=(q*eig)@q.T
        rt=(q*np.minimum(eig,1))@q.T
        d=matched(m,r);k2=min(shape)
        errors=[float(np.linalg.norm(matched(m,c*r)-d)/np.linalg.norm(d)) for c in [.03,100]]
        p=polar(m@r);pi=p.T@p
        gram=k2*(r@pi@r)/np.trace(r@pi@r)
        full_error=float(np.linalg.norm(d.T@d-gram))
        simple_error=float(np.linalg.norm(d.T@d-k2*(r@r)/np.trace(r@r)))
        top_raw=polar(m@rt)@rt
        assert max(errors)<1e-12 and full_error<1e-12
        assert np.linalg.norm(top_raw)<=np.sqrt(k2)+1e-12
        if shape[0]>=shape[1]: assert simple_error<1e-12
        else: assert simple_error>.01
        checks.append(dict(shape=shape,scale_errors=errors,projector_gram_error=full_error,
                           simple_gram_error=simple_error,contractive_root_raw_norm=float(np.linalg.norm(top_raw))))
    r=np.diag(np.array([1.9,.1])**-.5);rt=np.minimum(r,np.eye(2))
    # Roots are diagonal here; off-diagonal zero is preserved by minimum.
    full=np.diag(matched(np.eye(2),r).T@matched(np.eye(2),r))
    top=np.diag(matched(np.eye(2),rt).T@matched(np.eye(2),rt))
    assert np.max(abs(full-np.array([.1,1.9])))<1e-14
    assert np.max(abs(top-np.array([20/29,38/29])))<1e-14
    rows=[];ratios=[]
    files=sorted((ROOT/'logs/muon_spectra/second_order_audit_20260926/step_profile_pdtop').glob('*.json'))
    assert len(files)==4
    for path in files:
        data=json.loads(read(path));dirs=data['directions'];state=data['item']
        summaries={}
        for method in ['muon','pd','pdtop']:
            d=dirs[method];n2=d['norm']**2
            row=dict(state=state,step=data['step'],method=method,norm=d['norm'],slope=d['slope'],
                     energy_fraction_top16=d['energy_top16'],absolute_energy_top16=n2*d['energy_top16'],
                     curvature=d['curvature'],curvature_per_norm2=d['curvature']/n2,
                     curvature_curv_set=d['curvature_curv_set'],c_star=d['c_star'])
            rows.append(row);summaries[method]=row
        a,b,c=(summaries[k] for k in ['muon','pd','pdtop'])
        ratios.append(dict(state=state,step=data['step'],
            muon_over_top_fraction=a['energy_fraction_top16']/c['energy_fraction_top16'],
            muon_over_top_absolute=a['absolute_energy_top16']/c['absolute_energy_top16'],
            muon_over_pd_fraction=a['energy_fraction_top16']/b['energy_fraction_top16'],
            muon_over_pd_absolute=a['absolute_energy_top16']/b['absolute_energy_top16'],
            top_over_pd_fraction=c['energy_fraction_top16']/b['energy_fraction_top16'],
            top_over_pd_absolute=c['absolute_energy_top16']/b['absolute_energy_top16'],
            top_over_muon_norm=c['norm']/a['norm'],pd_over_muon_norm=b['norm']/a['norm'],
            top_over_muon_curvature_per_norm2=c['curvature_per_norm2']/a['curvature_per_norm2'],
            slopes={k:v['slope'] for k,v in summaries.items()}))
    for name,values in [('preflight.csv',rows),('ratios.csv',ratios)]:
        with (HERE/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(values[0]));w.writeheader();w.writerows(values)
    for name,expected in manifest.items(): assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected['sha256']
    output=dict(input_manifest=manifest,algebra=checks,example=dict(full_energy=full.tolist(),top_energy=top.tolist()),
                preflight=rows,ratios=ratios,seconds=time.monotonic()-started,
                scope='Raw exact-polar maps of lagged momentum; no online per-matrix matching, no training rate or feedback claim.')
    (HERE/'result.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in output.items() if k not in ['input_manifest','preflight']},indent=2))


if __name__=='__main__': main()

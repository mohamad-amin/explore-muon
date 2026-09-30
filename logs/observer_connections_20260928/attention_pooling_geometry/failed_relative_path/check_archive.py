"""All retained V fit coefficients versus a necessary idealized pooling relation."""
import os
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
os.environ['MPLCONFIGDIR']=str(HERE/'.mplconfig')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import csv
import hashlib
import json
import math
import statistics
import time

STEPS=[10,50,100,200,500,900,1300,1469]
METHODS=['M','PD','S','SPD']


def describe(xs):
    return dict(median=statistics.median(xs),min=min(xs),max=max(xs))


def main():
    start=time.monotonic();rows=[];manifest={};seen=set();metas={}
    def read(p):
        b=p.read_bytes();manifest[str(p.relative_to(ROOT))]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest());return json.loads(b)
    source=ROOT/'logs/muon_spectra/second_order_audit_20260926/marginals'
    paths=sorted(source.glob('*.json'));assert len(paths)==32
    for p in paths:
        d=read(p);method=p.name.split('_')[0];step=d['step'];assert method in METHODS and step in STEPS
        assert (method,step) not in seen;seen.add((method,step));assert d['sequences']==2048
        arm=Path(d['arm']);mp=arm/'scientific/metadata.json'
        if str(mp) not in metas:metas[str(mp)]=read(mp)
        T=metas[str(mp)]['config']['model']['seq_len'];assert T==512
        assert len(d['matrices'])==48
        upper=T/sum(1/i for i in range(1,T+1));lower_a=(T-upper)/(T-1)
        for layer in range(1,9):
            name=f'block{layer:02d}.v';f=d['matrices'][name]['fit_exact']
            a=f['a_within'];b=f['b_between'];r=((T-1)*a+b)/T-1
            assert all(math.isfinite(v) for v in f.values())
            rows.append(dict(method=method,step=step,layer=layer,matrix=name,T=T,a=a,b=b,
                b_over_a=b/a if a!=0 else None,coefficient_relation_residual=r,
                absolute_relation_residual=abs(r),fit_residual=f['residual_fit'],
                kfac_residual=f['residual_kfac'],between_share_of_C=f['between_share_of_C'],
                independent_equal_query_b_in_causal_range=1<=b<=upper,
                independent_equal_query_a_in_causal_range=lower_a<=a<=1,
                causal_uniform_b_bound=upper,
                source_json=str(p.relative_to(ROOT))))
    assert seen=={(m,s) for m in METHODS for s in STEPS} and len(rows)==256
    summaries=[]
    for method in METHODS:
        for step in STEPS:
            rr=[r for r in rows if r['method']==method and r['step']==step]
            summaries.append(dict(method=method,step=step,layers=len(rr),
                **{key:describe([r[key] for r in rr]) for key in
                   ('a','b','coefficient_relation_residual','absolute_relation_residual','fit_residual','kfac_residual')},
                relation_band_counts={str(t):sum(r['absolute_relation_residual']<=t for r in rr) for t in (.05,.10,.25)},
                fit_band_counts={str(t):sum(r['fit_residual']<=t for r in rr) for t in (.05,.10,.25)}))
    counts={str(t):dict(relation_within=sum(r['absolute_relation_residual']<=t for r in rows),
                         fit_within=sum(r['fit_residual']<=t for r in rows),
                         both_within=sum(r['absolute_relation_residual']<=t and r['fit_residual']<=t for r in rows))
            for t in (.05,.10,.25)}
    for p in paths:
        assert hashlib.sha256(p.read_bytes()).hexdigest()==manifest[str(p.relative_to(ROOT))]['sha256']
    with (HERE/'archive_cells.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    result=dict(scope='Descriptive consistency of all retained fitted V marginals; no population test, independent sample split, measured attention q, or optimizer outcome.',
        cells=rows,summaries=summaries,reference_band_counts=counts,
        total_cells=len(rows),no_models_or_tensor_archives=True,input_manifest=manifest,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        check_contract_sha256=hashlib.sha256((HERE/'ARCHIVE_CHECK.md').read_bytes()).hexdigest())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10.8,7),sharex=True,sharey=True,constrained_layout=True)
    for ax,method in zip(axes.flat,METHODS):
        rs=[r for r in summaries if r['method']==method]
        x=[r['step'] for r in rs];y=[r['coefficient_relation_residual']['median'] for r in rs]
        lo=[r['coefficient_relation_residual']['min'] for r in rs];hi=[r['coefficient_relation_residual']['max'] for r in rs]
        ax.plot(x,y,'o-',color='#0072B2',label='Median over 8 layers')
        ax.fill_between(x,lo,hi,color='#0072B2',alpha=.13,label='Full layer range')
        ax.axhline(0,color='black',linewidth=.8,label='Idealized model')
        ax.axhspan(-.1,.1,color='gray',alpha=.1,label='±.10 reference band')
        ax.set_xscale('log');ax.set_title(method);ax.grid(alpha=.12)
    axes[0,0].legend(frameon=False,fontsize=8)
    fig.supylabel('[(T−1)a + b]/T − 1; archived fit coefficients')
    fig.supxlabel('Training update; shared 2048-sequence marginal panel, one seed')
    fig.suptitle('Necessary iid-pooling relation versus retained value-curvature fits',fontsize=12)
    fig.savefig(HERE/'archive_relation.png',dpi=170);fig.savefig(HERE/'archive_relation.pdf');plt.close(fig)
    result['seconds']=time.monotonic()-start
    (HERE/'archive_check.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'cells':256,'reference_band_counts':counts,
        'summaries':[dict(method=r['method'],step=r['step'],a=r['a']['median'],b=r['b']['median'],
                          relation=r['coefficient_relation_residual']['median'],fit=r['fit_residual']['median']) for r in summaries],
        'seconds':result['seconds']},indent=2))


if __name__=='__main__':main()

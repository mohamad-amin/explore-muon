"""One analytic gauge qualification; no model/training/checkpoint code."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
V=np.array([[.8,.3],[-.2,1.1]])
O=np.array([[1.2,-.4],[.1,.9]])
C=np.array([[2.,.3],[.3,.5]])
K=np.array([[.2,-.4],[.6,.1]])
S=np.array([[2.,.25],[0.,.5]])


def polar(x):
    u,_,v=np.linalg.svd(x,full_matrices=False);return u@v


def root(c,alpha=.5,normalize=False,damping=0.):
    vals,vecs=np.linalg.eigh(c)
    assert vals.min()>0
    if normalize:vals=vals/vals.mean()
    return (vecs*(vals+damping)**(-alpha))@vecs.T


def pd(m,c,normalize=False,damping=0.,graft=False):
    r=root(c,normalize=normalize,damping=damping);d=polar(m@r)@r
    return d*np.sqrt(min(m.shape))/np.linalg.norm(d) if graft else d


def ts(m,c,b,graft=False):
    l,r=root(b),root(c);d=l@polar(l@m@r)@r
    return d*np.sqrt(min(m.shape))/np.linalg.norm(d) if graft else d


def err(a,b):return float(np.linalg.norm(a-b)/np.linalg.norm(b))


def main():
    si=np.linalg.inv(S);vp=S@V;op=O@si
    mv=O.T@K;mo=K@V.T;mpv=np.linalg.inv(S).T@mv;mpo=mo@S.T
    co=V@C@V.T;cop=S@co@S.T;bv=O.T@O;bvp=si.T@bv@si
    errors={'product':err(op@vp,O@V),'covector_V':err(op.T@K,mpv),'covector_O':err(K@vp.T,mpo)}
    variants={}
    for name,kw in [('raw',{}),('mean_normalized',{'normalize':True}),
                    ('mean_damped',{'normalize':True,'damping':.001}),
                    ('raw_grafted',{'graft':True}),
                    ('online_geometry_ideal_polar',{'normalize':True,'damping':.001,'graft':True})]:
        dv,do=pd(mv,C,**kw),pd(mo,co,**kw)
        dpv,dpo=pd(mpv,C,**kw),pd(mpo,cop,**kw)
        expected_v,expected_o=S@dv,do@si
        # Matrix pullback makes all errors refer to the original parameter chart.
        pv,po=si@dpv,dpo@S
        scale=float(np.sum(po*do)/np.sum(do*do))
        variants[name]={'V_covariance_error':err(pv,dv),'O_covariance_error':err(po,do),
            'O_best_scalar':scale,'O_ray_error_after_scalar':err(po,scale*do),
            'pair_tangent_error':err(op@dpv+dpo@vp,O@dv+do@V),
            'original_norms':[float(np.linalg.norm(dv)),float(np.linalg.norm(do))],
            'transformed_norms':[float(np.linalg.norm(dpv)),float(np.linalg.norm(dpo))]}
    for graft in [False,True]:
        dv,do=ts(mv,C,bv,graft),ts(mo,co,np.eye(2),graft)
        dpv,dpo=ts(mpv,C,bvp,graft),ts(mpo,cop,np.eye(2),graft)
        variants['full_two_sided'+('_grafted' if graft else '_raw')]={
            'V_covariance_error':err(si@dpv,dv),'O_covariance_error':err(dpo@S,do),
            'pair_tangent_error':err(op@dpv+dpo@vp,O@dv+do@V)}
    rho=.97
    decay={'scalar_product_equivariance_error':err((rho*op)@(rho*vp),(rho*O)@(rho*V))}
    def shaped(w,c):
        r=root(c,alpha=.25,normalize=True,damping=.001);r2=r@r
        return w@r2/np.trace(r2)*len(r2)
    dv,do=shaped(V,C),shaped(O,co);dpv,dpo=shaped(vp,C),shaped(op,cop)
    decay.update(V_covariance_error=err(si@dpv,dv),O_covariance_error=err(dpo@S,do),
                 pair_tangent_error=err(op@dpv+dpo@vp,O@dv+do@V))
    assert max(errors.values())<1e-10
    assert variants['raw']['O_covariance_error']<1e-10
    assert max(variants['full_two_sided_raw'].values())<1e-10
    assert decay['scalar_product_equivariance_error']<1e-10
    out={'scope':'One fixed full-rank2×2analytic qualification. Errors measure equivariance, not loss, stability, empirical gauge drift or optimizer quality.',
         'matrices':{k:v.tolist() for k,v in [('V',V),('O',O),('C',C),('K',K),('S',S)]},
         'positive_controls':errors,'maps':variants,'decay':decay,
         'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE/'result.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:v for k,v in out.items() if k!='matrices'},indent=2))


if __name__=='__main__':main()

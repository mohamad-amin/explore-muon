"""Numerically qualify one analytic example; no network/optimizer experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
EPS=.02
R=np.diag([1.,.2,.2])
M=np.diag([np.sqrt(1-2*EPS**2),EPS,EPS])
E=np.zeros((3,3)); E[1,2]=1/np.sqrt(2); E[2,1]=-1/np.sqrt(2)
BETA=.8; ETA=.01; LAMBDA=3.


def polar(m):
    u,_,v=np.linalg.svd(m,full_matrices=False)
    return u@v


def phi(m,pd):
    if not pd:return polar(m)
    d=polar(m@R)@R
    return np.sqrt(3)*d/np.linalg.norm(d)


def derivative(d,pd):
    # At this positive diagonal operating point, U=V=I.
    inner=d@R if pd else d
    sigma=np.diag(M@R if pd else M)
    dq=(inner-inner.T)/(sigma[:,None]+sigma[None,:])
    if not pd:return dq
    raw=dq@R
    return np.sqrt(3)/np.linalg.norm(R)*(raw-R*np.sum(R*raw)/np.sum(R*R))


def main():
    assert np.isclose(np.linalg.norm(M),1) and np.isclose(np.linalg.norm(E),1)
    records={}
    eye=np.eye(9)
    for name,pd in [('polar',False),('matched_pd',True)]:
        gain=(np.sqrt(3)*.2/np.linalg.norm(R) if pd else 1)/EPS
        expected=gain*E
        assert np.allclose(derivative(E,pd),expected,rtol=1e-13,atol=1e-13)
        errors=[]
        for h in [1e-4,1e-5,1e-6]:
            fd=(phi(M+h*E,pd)-phi(M-h*E,pd))/(2*h)
            errors.append({'h':h,'relative_error':float(np.linalg.norm(fd-expected)/np.linalg.norm(expected))})
        assert errors[-1]['relative_error']<1e-5
        A=np.column_stack([derivative(e.reshape(3,3),pd).ravel() for e in eye])
        H=LAMBDA*np.outer(E.ravel(),E.ravel())
        block=np.block([[eye-ETA*A@H,-ETA*BETA*A],[H,BETA*eye]])
        def joint(x):
            w=x[:9]; mom=x[9:].reshape(3,3)
            nxt=BETA*mom+(H@w).reshape(3,3)
            return np.concatenate([w-ETA*phi(nxt,pd).ravel(),nxt.ravel()])
        # Operating point along a possible trajectory, not a fixed point.
        x=np.concatenate([np.zeros(9),(M/BETA).ravel()]); h=1e-6
        fdj=np.column_stack([(joint(x+h*e)-joint(x-h*e))/(2*h) for e in np.eye(18)])
        jerr=float(np.linalg.norm(fdj-block)/np.linalg.norm(block))
        assert jerr<1e-5
        output=phi(M,pd)
        records[name]={'momentum_norm':float(np.linalg.norm(M)),
                       'output_norm':float(np.linalg.norm(output)),
                       'output_stiff_projection':float(np.sum(output*E)),
                       'differential_gain_along_E':float(gain),
                       'analytic_radial_derivative_norm':float(np.linalg.norm(A@M.ravel())),
                       'central_difference_errors':errors,
                       'joint_jacobian_relative_error':jerr,
                       'joint_jacobian_max_abs_error':float(np.max(np.abs(fdj-block)))}
    result={'status':'qualified','epsilon':EPS,'root_diagonal':np.diag(R).tolist(),
            'beta':BETA,'eta':ETA,'hessian_lambda':LAMBDA,'results':records,
            'scope':'One analytic counterexample checked numerically. No training simulation, empirical network evidence, rate or stability claim.',
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (HERE/'differential_check.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

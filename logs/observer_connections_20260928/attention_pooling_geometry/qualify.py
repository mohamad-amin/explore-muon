"""Deterministic attention/value partial-trace algebra; no project tensors."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import hashlib
import json
from pathlib import Path
import time
import numpy as np

HERE=Path(__file__).resolve().parent


def relative(a,b):
    return float(np.linalg.norm(a-b)/max(1.,np.linalg.norm(a),np.linalg.norm(b)))


def exact_partial_trace(X,A,H,out_dim):
    """H is unscaled post-attention error covariance; objective has 1/T."""
    T,d=X.shape;Z=A@X
    J=np.zeros((T*out_dim,out_dim*d))
    for t in range(T):
        for a in range(out_dim):J[t*out_dim+a,a*d:(a+1)*d]=Z[t]
    G=J.T@H@J/T
    partial=sum(G[a*d:(a+1)*d,a*d:(a+1)*d] for a in range(out_dim))
    R=np.array([[np.trace(H[t*out_dim:(t+1)*out_dim,u*out_dim:(u+1)*out_dim])
                 for u in range(T)] for t in range(T)])
    predicted=X.T@A.T@R@A@X/T
    routing=np.kron(A,np.eye(out_dim))
    tr_pre=float(np.trace(routing.T@H@routing)/T)
    trace_predicted=float(np.trace(A.T@R@A)/T)
    return partial,predicted,tr_pre,trace_predicted,R


def main():
    start=time.monotonic();rng=np.random.default_rng(8031)
    X=rng.normal(size=(5,3));A=np.tril(rng.uniform(.1,1.,size=(5,5)));A/=A.sum(axis=1,keepdims=True)
    L=rng.normal(size=(10,10));H=L@L.T+.1*np.eye(10)
    F,pred,trpre,trpred,R=exact_partial_trace(X,A,H,2)
    errors={'full_Jacobian_partial_trace':relative(F,pred),'pre_value_trace':abs(trpre-trpred)/max(1.,abs(trpre))}
    B=np.array([[2.,.3],[.3,1.]])
    Fi,predi,tri,_,_=exact_partial_trace(X,A,np.kron(np.eye(5),B),2)
    q=float((A*A).sum()/5)
    errors['independent_position_covariance']=relative(Fi,np.trace(B)*X.T@A.T@A@X/5)
    errors['pre_value_trace_iid']=abs(tri-q*np.trace(B))/max(1.,abs(tri))

    moment_cases=[]
    Sigma=np.array([[2.,.4],[.4,1.]])
    M=np.array([[1.,.1],[.1,.7]])
    for T in (4,512):
        uniform=np.tril(np.ones((T,T)));uniform/=np.arange(1,T+1)[:,None]
        candidates={'identity':np.eye(T),'uniform_causal':uniform,
                    'uniform_full':np.ones((T,T))/T,
                    'causal_mixture':.3*np.eye(T)+.7*uniform}
        Cb=M+Sigma/T;Cw=(1-1/T)*Sigma
        for name,routing in candidates.items():
            q=float((routing*routing).sum()/T)
            a=(T*q-1)/(q*(T-1));b=1/q
            direct=Sigma+M/q;reconstructed=a*Cw+b*Cb
            error=relative(direct,reconstructed);assert error<1e-12
            relation=abs((T-1)*a+b-T)/T;assert relation<1e-12
            moment_cases.append(dict(T=T,attention=name,q=q,a=a,b=b,
                                      b_over_a=b/a if abs(a)>1e-14 else None,
                                      moment_identity_error=error,coefficient_relation_error=relation))

    # Same forward concentration and identical observed X do not fix C_AX.
    Xc=np.array([[1.,0.],[0.,2.],[-1.,0.]])
    A1=np.array([[1.,0.,0.],[1.,0.,0.],[0.,0.,1.]])
    A2=np.array([[1.,0.,0.],[0.,1.,0.],[0.,1.,0.]])
    q1=float(np.sum(A1*A1)/3);q2=float(np.sum(A2*A2)/3);assert q1==q2==1.
    C1=Xc.T@A1.T@A1@Xc/3;C2=Xc.T@A2.T@A2@Xc/3
    assert not np.allclose(C1,C2)
    fixed_input_counterexample=dict(q=q1,X=Xc.tolist(),A1=A1.tolist(),A2=A2.tolist(),
        C_AX_1=C1.tolist(),C_AX_2=C2.tolist(),frobenius_difference=float(np.linalg.norm(C1-C2)))

    # Same attention, changed downstream positional covariance changes Gamma.
    T=4;A4=np.tril(np.ones((T,T)));A4/=np.arange(1,T+1)[:,None]
    Rcases={'independent_equal':np.eye(T),'independent_unequal':np.diag([1.,1.,1.,40.]),
            'coherent':np.ones((T,T)),'centered':np.eye(T)-np.ones((T,T))/T}
    downstream=[]
    Cb=M+Sigma/T;Cw=(1-1/T)*Sigma
    for name,R in Rcases.items():
        S=A4.T@R@A4;den=float(np.trace(S));num=float(np.ones(T)@R@np.ones(T));assert den>0
        gamma=num/den;a=(T-gamma)/(T-1);b=gamma
        direct=Sigma+gamma*M;error=relative(direct,a*Cw+b*Cb);assert error<1e-12
        downstream.append(dict(name=name,q=float(np.sum(A4*A4)/T),Gamma=gamma,a=a,b=b,
                               moment_identity_error=error))
    assert len({round(c['Gamma'],10) for c in downstream})==4
    assert max(errors.values())<1e-12
    elapsed=time.monotonic()-start;assert elapsed<60.
    result=dict(status='passed',scope='Exact synthetic algebra only; no trained model, checkpoint or optimizer outcome.',
        seconds=elapsed,errors=errors,moment_cases=moment_cases,
        fixed_input_same_q_counterexample=fixed_input_counterexample,
        downstream_same_q_counterexamples=downstream,
        historical_tensors_read=False,model_calls=0,gpu_calls=0,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        protocol_sha256=hashlib.sha256((HERE/'PROTOCOL.md').read_bytes()).hexdigest())
    (HERE/'qualification.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':main()

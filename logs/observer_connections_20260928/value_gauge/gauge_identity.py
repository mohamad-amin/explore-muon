"""Exact small V/O gauge counterexample; CPU algebra, no model calls."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent
mu=np.array([np.sqrt(99.),0.]);C=np.diag([100.,1.]);Sigma=C-np.outer(mu,mu)
V=np.eye(2);O=np.eye(2);S=np.diag([.1,1.])
Vp=S@V;Op=O@np.linalg.inv(S)

def metrics(v,o):
    m=v@mu
    gain=(m@m)/(mu@mu)
    centered=np.trace(v@Sigma@v.T)/np.trace(Sigma)
    return {'value_mean_fraction':float((m@m)/np.trace(v@C@v.T)),
            'value_mean_vs_centered_gain':float(gain/centered),
            'residual_constant_vector':(o@m).tolist(),
            'residual_constant_energy':float(np.square(o@m).sum())}

rng=np.random.default_rng(260928)
x=rng.normal(size=(7,2))+mu
logits=x@x.T/np.sqrt(2)
logits[np.triu_indices(7,1)]=-np.inf
prob=np.exp(logits-np.max(logits,axis=1,keepdims=True));prob/=prob.sum(1,keepdims=True)
before=prob@(x@V.T)@O.T;after=prob@(x@Vp.T)@Op.T
err=float(np.max(np.abs(after-before)))
assert err<1e-12
out={'original':metrics(V,O),'equivalent_gauge':metrics(Vp,Op),
     'max_attention_output_error':err,'gauge':S.tolist(),
     'assumptions':'One head shown; any invertible block-diagonal within-head S generalizes. No value normalization/dropout. Q/K and therefore attention weights are unchanged.',
     'scope':'Exact counterexample: pre-O mean ratios need not identify an invariant network function. It does not show real trajectory differences are only gauge.'}
(HERE/'gauge_identity.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

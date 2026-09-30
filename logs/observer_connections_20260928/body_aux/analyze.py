"""Analyze saved four-corner scalar Grams and losses; no model calls."""
from pathlib import Path
import json
import math
import numpy as np

HERE=Path(__file__).resolve().parent
d=json.loads((HERE/'result.json').read_text())
assert d['status']=='complete'
names=['base','body','aux','full']
C=np.array([[d['cross_bank_gradient_gram'][a][b] for b in names] for a in names])
S=(C+C.T)/2
contrasts={'body':[-1,1,0,0],'aux':[-1,0,1,0],'interaction':[1,-1,-1,1],'total':[-1,0,0,1]}
V=np.array(list(contrasts.values()),dtype=float)
change=V@S@V.T
out={'cross_bank_corner_gram_symmetric':S.tolist(),
     'contrast_names':list(contrasts),'contrast_coefficients':contrasts,
     'cross_bank_change_gram':change.tolist(),
     'banks':[]}
all_effects=[]
for bank in d['banks']:
    L=np.array([bank['losses'][c] for c in names]).T
    effects=L@V.T
    rows={}
    for j,k in enumerate(contrasts):
        x=effects[:,j]
        rows[k]={'mean':float(x.mean()),'sample_sd':float(x.std(ddof=1)),
                 'naive_se':float(x.std(ddof=1)/np.sqrt(len(x))),
                 'positive_sequences':int((x>0).sum()),'values':x.tolist()}
    rows['full_minus_body']={'mean':float((L[:,3]-L[:,1]).mean()),'values':(L[:,3]-L[:,1]).tolist()}
    out['banks'].append(rows)
    all_effects.append(effects)
pooled=np.concatenate(all_effects)
out['pooled_loss_effects']={k:{'mean':float(pooled[:,j].mean()),'naive_se':float(pooled[:,j].std(ddof=1)/math.sqrt(len(pooled))),
                             'positive_sequences':int((pooled[:,j]>0).sum())} for j,k in enumerate(contrasts)}
# Symmetrized cross-bank signal estimate vs average same-bank Gram.
# This finite two-bank decomposition is algebraic; it is not a qualified
# extrapolation to the original 128/8192-sequence measurements.
K=sum(np.array([[b['gradient_gram'][a][c] for c in names] for a in names]) for b in d['banks'])/2
out['average_same_bank_corner_gram']=K.tolist()
out['same_minus_cross_corner_gram']=(K-S).tolist()
out['same_minus_cross_psd_eigenvalues']=np.linalg.eigvalsh(K-S).tolist()
out['scope']='Two disjoint eight-sequence banks, common data across corners. Cross-bank ratios noisy; no population CI. Naive loss SE assumes independent sequences and is descriptive. Gradient interaction retained; loss interaction is finite-step, not an isolated Hessian entry.'
(HERE/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

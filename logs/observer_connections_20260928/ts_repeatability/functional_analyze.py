"""Compare conditional parameter and predictive-GN repeatability."""
from pathlib import Path
import json
import numpy as np

HERE=Path(__file__).resolve().parent;RUN=HERE/'run2';OUT=RUN/'functional'
d=json.loads((OUT/'result.json').read_text());assert d['status']=='complete'
index={n:i for i,n in enumerate(d['labels'])};pd=index['PD']
pairs=[('A0','B0'),('A0','A1'),('AB','CD'),('AB','ABCD'),('CD','ABCD'),('ABCD','ABCD_bf16')]

def analyze(q):
    def delta(i,j):return float(q[i,i]+q[j,j]-2*q[i,j])
    result={}
    for a,b in pairs:
        i,j=index[a],index[b];noise=delta(i,j);geometry=delta(i,pd)+delta(j,pd)
        result[a+'__'+b]={'q_a':float(q[i,i]),'q_b':float(q[j,j]),'q_difference':noise,
            'gn_cosine':float(q[i,j]/np.sqrt(q[i,i]*q[j,j])),
            'relative_functional_difference':float(np.sqrt(max(0,noise)/q[i,i])),
            'noise_to_added_TS_geometry':noise/max(geometry,1e-30),'added_geometry_denominator':geometry}
    return result

bank_q=[np.mean([np.array(r['gram']) for r in b],axis=0) for b in d['banks']]
q=np.mean(bank_q,axis=0)
out={'banks':[analyze(g) for g in bank_q],'pooled':analyze(q),'gram_pooled':q.tolist(),'labels':d['labels'],
     'scope':d['scope'],'checks':d['checks'],'seconds':d['seconds']}
(OUT/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
for b,r in enumerate(out['banks']):print('bank',b,json.dumps(r))
print('pooled',json.dumps(out['pooled'],indent=2))

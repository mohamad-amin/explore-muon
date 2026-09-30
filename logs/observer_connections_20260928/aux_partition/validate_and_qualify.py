"""Independent scalar reduction and post-hoc interpretation; no model calls."""
from pathlib import Path
import hashlib
import json
import numpy as np

HERE=Path(__file__).resolve().parent;RUN=HERE/'run1'
raw=json.loads((RUN/'result.json').read_text());a=json.loads((RUN/'analysis.json').read_text())
assert raw['status']=='complete'
groups=[('B',1),('H',2),('E',4),('N',8)]
label=lambda m:''.join(k for k,b in groups if m&b) or 'base'
max_contrast_error=0.;max_plane_error=0.;max_total_error=0.
for bank in raw['banks']:
    for row in bank['rows']:
        for mask in range(16):
            contrast=sum((-1)**(mask.bit_count()-sub.bit_count())*row['loss64'][label(sub)] for sub in range(16) if sub&mask==sub)
            max_contrast_error=max(max_contrast_error,abs(contrast-row['mobius'][label(mask)]))
        for p in row['planes'].values():max_plane_error=max(max_plane_error,abs(p['interaction']-p['overlap']-p['mixed_logit_loss']))
        x=sum(row['mobius'][label(m)] for m in range(16) if m&1 and m!=1)
        max_total_error=max(max_total_error,abs(x-row['planes']['B:HEN|base']['interaction']))
assert max(max_contrast_error,max_plane_error,max_total_error)<1e-10
interpretation={}
for role in ['discovery','fresh']:
    r=a[role];q=np.array(r['finite_logit_gram']);I=r['planes']['B:HEN|base']['mean']['interaction']
    linear={k:r['loss64'][k]-r['loss64']['base']-r['predictive_kl'][k] for k in ['B','H','E','N']}
    projections={}
    for j,k in enumerate(['B','H','E','N']):
        if not j:continue
        c=q[0,j]/q[0,0]
        remain=linear[k]-c*linear['B']
        projections[k]={'coefficient_on_body_finite_response':float(c),
                        'original_label_linear_response':linear[k],
                        'remaining_label_linear_response':float(remain),
                        'remaining_descent_fraction':float(remain/linear[k]),
                        'remaining_metric_energy_fraction':float((q[j,j]-q[0,j]**2/q[0,0])/q[j,j])}
    interpretation[role]={'pair_shares_of_total':{k:r['mobius'][k]/I for k in ['BH','BE','BN']},
         'higher_order_share':sum(r['mobius'][k] for k in ['BHE','BHN','BEN','BHEN'])/I,
         'overlap_share':r['planes']['B:HEN|base']['mean']['overlap']/I,
         'finite_logit_projections':projections}
out={'validation':{'direct_subset_vs_fast_transform_max_abs_error':max_contrast_error,'plane_split_max_abs_error':max_plane_error,'total_body_aux_identity_max_abs_error':max_total_error},
     'posthoc_interpretation':interpretation,
     'scope':'Projection is algebra in the finite-logit response span after seeing results, not an implemented parameter update, new endpoint or training recommendation. Nonlinear loss of a projected response was not measured.',
     'input_sha256':{name:hashlib.sha256((RUN/name).read_bytes()).hexdigest() for name in ['result.json','analysis.json']}}
(RUN/'validation.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))

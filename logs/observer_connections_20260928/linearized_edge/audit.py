"""Preserve old EOS measurements and check one finite-NS differential identity."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SRC=ROOT/'logs/muon_spectra/second_order_audit_20260926'
COEFFICIENTS=[(4.0848,-6.8946,2.927),(3.9505,-6.3029,2.6377),(3.7418,-5.5913,2.3037),
              (2.8769,-3.1427,1.2046),(2.8366,-3.0525,1.2012)]


def scalar_map(x):
    v=x/np.linalg.norm(x)
    for a,b,c in COEFFICIENTS:v=a*v+b*v**3+c*v**5
    return v


def main():
    manifest={};records=[]
    for name in ['eos_linearized.py','eos_linearized_a.json','eos_linearized_b.json','eos_linearized_smoke.json']:
        f=SRC/name;b=f.read_bytes();out=HERE/'inputs'/name;out.parent.mkdir(exist_ok=True)
        if out.exists():assert out.read_bytes()==b
        else:out.write_bytes(b)
        manifest[str(f.relative_to(ROOT))]={'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        if name.endswith('.json'):
            for key,d in json.loads(b).items():
                assert np.isclose(d['lr']*d['mu_top'][0][0],d['invariant'],rtol=1e-12)
                assert np.isclose(d['invariant']/d['threshold'],d['ratio'],rtol=1e-12)
                records.append({'source':name,'state':key,'role':'superseded smoke' if 'smoke' in name else 'completed',**d})
    assert sum(x['role']=='completed' for x in records)==10
    x=np.array([1.,.5]);s=x/np.linalg.norm(x);v=s.copy();deriv=np.ones(2)
    for a,b,c in COEFFICIENTS:
        deriv*=a+3*b*v**2+5*c*v**4
        v=a*v+b*v**3+c*v**5
    J=np.diag(deriv)@(np.eye(2)-np.outer(s,s))/np.linalg.norm(x)
    h=1e-6
    fd=np.column_stack([(scalar_map(x+h*e)-scalar_map(x-h*e))/(2*h) for e in np.eye(2)])
    err=float(np.linalg.norm(J-fd)/np.linalg.norm(J));assert err<1e-6
    test={'input_diagonal':x.tolist(),'normalized_input':s.tolist(),'output_diagonal':v.tolist(),
          'analytic_diagonal_subspace_jacobian':J.tolist(),'finite_difference_jacobian':fd.tolist(),
          'relative_error':err,'antisymmetric_norm':float(np.linalg.norm(J-J.T)),
          'eigenvalues':np.linalg.eigvals(J).tolist(),'symmetric_part_eigenvalues':np.linalg.eigvalsh((J+J.T)/2).tolist(),
          'scope':'One exact source-polynomial example disproves general symmetry/PSD. No trained-checkpoint negative-mode claim.'}
    result={'records':records,'input_manifest':manifest,'coefficients':COEFFICIENTS,'qualification':test}
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'completed_states':10,'smoke_states':2,'qualification':test},indent=2))


if __name__=='__main__':main()

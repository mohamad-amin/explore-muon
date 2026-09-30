"""Finite-history projected replay decomposition, no model execution."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
SRC=ROOT/'logs/muon_spectra/second_order_audit_20260926/transport'


def norm(x):return float(np.linalg.norm(x))


def cosine(a,b):return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))


def main():
    results=[];manifest={}
    for f in sorted(SRC.glob('*.json')):
        content=f.read_bytes();d=json.loads(content)
        manifest[str(f.relative_to(ROOT))]={'sha256':hashlib.sha256(content).hexdigest(),'bytes':len(content)}
        (HERE/'inputs').mkdir(exist_ok=True);target=HERE/'inputs'/f.name
        if target.exists():assert target.read_bytes()==content
        else:target.write_bytes(content)
        p=np.array(d['replay_check']['probe_projections'],dtype=np.float64)
        factors=np.array(d['replay_check']['clip_factors'],dtype=np.float64)
        beta=d['beta'];K=d['replay'];assert p.shape==(100,16) and K==100 and beta==.95
        weights=beta**np.arange(K)*factors;mass=float(weights.sum());mean=p.mean(0)
        replay=weights@p;constant=mass*mean;residual=weights@(p-mean)
        saved=d['ritz']['projections'];ms=np.array(saved['Mstar']);m=np.array(saved['M'])
        mean_saved=np.array(saved['gbar']);delta=m-replay
        errors={'replay_projection_relative':norm(replay-ms)/norm(ms),
                'mean_projection_relative':norm(mean-mean_saved)/norm(mean_saved),
                'decomposition_relative':norm(replay-constant-residual)/norm(replay)}
        assert max(errors.values())<=1e-5,errors
        centered=p-mean;variance=float(np.sum(centered**2)/(K-1))
        lags=[]
        for lag in range(1,11):
            a,b=centered[:-lag],centered[lag:]
            lags.append({'lag_batches':lag,'mean_centered_dot':float(np.sum(a*b)/(K-lag)),
                         'centered_dot_over_trace_variance':float(np.sum(a*b)/(K-lag)/variance),
                         'pooled_centered_cosine':float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b)))})
        # Bound only the old tail in actual M; full-model norm upper-bounds body norm.
        arm=Path(d['arm']);arm=ROOT/'logs/muon_spectra'/str(arm).split('/logs/muon_spectra/',1)[1]
        meta=json.loads((arm/'scientific/metadata.json').read_text());clip=meta['config']['grad_clip']
        bound=0.;complete=True
        for s in range(1,d['step']-K+1):
            sf=arm/'scientific/steps'/f'step{s:06d}.json'
            if not sf.exists():complete=False;break
            raw=sf.read_bytes();rec=json.loads(raw)
            manifest[str(sf.relative_to(ROOT))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            gnorm=rec['gradient_norm_before_clip'];clipped=gnorm*min(1.,clip/(gnorm+1e-6))
            bound=beta*bound+clipped
        tail=(beta**K)*bound if complete else (beta**K)*clip/(1-beta)
        per_mode=[]
        for j in range(16):per_mode.append({'mode':j+1,'ritz_value':d['ritz']['values'][j],
                'mean':float(mean[j]),'sample_variance':float(np.var(p[:,j],ddof=1)),
                'constant_part':float(constant[j]),'weighted_residual':float(residual[j]),
                'replay':float(replay[j]),'actual_momentum':float(m[j])})
        out={'source':str(f.relative_to(ROOT)),'beta':beta,'step':d['step'],'replayed_batches':K,'coordinate_count':16,
             'weight_mass':mass,'historical_clip_min':float(factors.min()),'historical_clip_max':float(factors.max()),
             'historical_clipped_batches':int(np.sum(factors<1)),
             'qualification':errors,
             'projected_norms':{'actual_momentum':norm(m),'replay':norm(replay),'constant_part':norm(constant),'weighted_residual':norm(residual),'actual_minus_replay':norm(delta)},
             'ratios':{'residual_over_constant':norm(residual)/norm(constant),'residual_over_actual_momentum':norm(residual)/norm(m),
                       'residual_over_actual_replay_difference':norm(residual)/norm(delta),'replay_over_actual_momentum':norm(replay)/norm(m)},
             'cosines':{'actual_constant':cosine(m,constant),'actual_replay':cosine(m,replay),'constant_residual':cosine(constant,residual),'difference_negative_constant':cosine(delta,-constant)},
             'signed_cross_terms':{'constant_dot_residual':float(constant@residual),'actual_dot_replay':float(m@replay)},
             'actual_tail_bound':{'beta_power':beta**K,'bound':tail,'from_complete_global_norm_logs':complete,
                                  'relative_to_projected_actual_replay_difference':tail/norm(delta),
                                  'scope':'Full body-norm upper bound for beta^100 M400, not for an unmeasured full stale-free replay tail.'},
             'centered_lag_descriptors':lags,'per_mode':per_mode}
        results.append(out)
    assert len(results)==2
    result={'scope':'Conditional100-batch history at selected trained1Mstates in16retainedRitz coordinates; historical clips held fixed. No independence, population-noise, all-space or training-rate claim.',
            'results':results,'input_manifest':manifest}
    (HERE/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in d.items() if k not in ['per_mode','centered_lag_descriptors']} for d in results],indent=2))


if __name__=='__main__':main()

"""One metadata-defined partial-batch regressor, no data/model execution."""
import hashlib
import json
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]


def main():
    original=json.loads((HERE/'result.json').read_text());out=[]
    for record in original['results']:
        data=json.loads((HERE/'inputs'/Path(record['source']).name).read_text())
        arm=ROOT/'logs/muon_spectra'/data['arm'].split('/logs/muon_spectra/',1)[1]
        meta_path=arm/'scientific/metadata.json';raw=meta_path.read_bytes();meta=json.loads(raw)
        B=meta['config']['batch_tokens'];K=data['replay'];step=data['step']
        starts=(step-np.arange(K)-1)*B;ends=starts+B
        boundaries=np.cumsum([x['tokens'] for x in meta['train_manifest']])
        boundaries=boundaries[(boundaries>starts.min())&(boundaries<ends.max())]
        assert boundaries.tolist()==[500_000_000]
        boundary=int(boundaries[0]);q=np.clip((ends-boundary)/B,0,1);x=q-q.mean()
        assert q[23]==.162841796875 and np.all(q[:23]==1) and np.all(q[24:]==0)
        p=np.asarray(data['replay_check']['probe_projections'],dtype=np.float64);z=p-p.mean(0)
        coeff=(x@z)/(x@x);fitted=x[:,None]*coeff;res=z-fitted
        assert np.linalg.norm(x@res)<1e-10 and np.linalg.norm(res.mean(0))<1e-10
        w=data['beta']**np.arange(K)*np.array(data['replay_check']['clip_factors'])
        delta=w@z;delta_fit=w@fitted;delta_rem=w@res
        err=float(np.linalg.norm(delta-delta_fit-delta_rem)/np.linalg.norm(delta));assert err<1e-10
        var0=float(np.sum(z*z)/(K-1));varr=float(np.sum(res*res)/(K-1))
        lags=[]
        for lag in range(1,11):
            a,b=res[:-lag],res[lag:];dot=float(np.sum(a*b)/(K-lag))
            lags.append({'lag_batches':lag,'pooled_centered_cosine':float(np.sum(a*b)/np.sqrt(np.sum(a*a)*np.sum(b*b))),
                         'dot_over_residual_trace_variance':dot/varr,'dot_over_original_trace_variance':dot/var0})
        denom=float(delta@delta)
        wc=w-w.mean()
        expected_delta_norm2=float((wc@wc)*np.sum(z*z)/(K-1))
        geometry_cos2=float((wc@x)**2/((wc@wc)*(x@x)))
        out.append({'source':record['source'],'metadata_sha256':hashlib.sha256(raw).hexdigest(),
                    'boundary_token':boundary,'q':q.tolist(),'centered_variance_explained':float(np.sum(fitted*fitted)/np.sum(z*z)),
                    'per_coordinate_boundary_coefficient':coeff.tolist(),
                    'weighted_residual_decomposition':{'total_norm':float(np.linalg.norm(delta)),
                        'boundary_norm':float(np.linalg.norm(delta_fit)),'remaining_norm':float(np.linalg.norm(delta_rem)),
                        'boundary_parallel_share':float(delta_fit@delta)/denom,'remaining_parallel_share':float(delta_rem@delta)/denom,
                        'boundary_dot_remaining':float(delta_fit@delta_rem),'relative_identity_error':err},
                    'finite_row_permutation_calibration':{
                        'weight_regressor_cosine_squared':geometry_cos2,
                        'expected_fitted_variance_fraction':1/(K-1),
                        'expected_weighted_residual_norm2':expected_delta_norm2,
                        'observed_over_expected_weighted_residual_norm2':denom/expected_delta_norm2,
                        'ratio_of_expected_parallel_numerator_to_expected_total_norm2':geometry_cos2,
                        'scope':'Exact uniform-permutation expectations for the observed centered rows, not exchangeability or significance. Ratio of expectations differs from expectation of the signed share.'},
                    'residual_lag_descriptors':lags})
    result={'scope':'Post-hoc single manifest-boundary regression; age/content/conditioning confounded. No searched split, iid inference or model.', 'results':out}
    (HERE/'shard_result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([{k:v for k,v in x.items() if k not in ['q','per_coordinate_boundary_coefficient','residual_lag_descriptors']}|
        {'lag_cosines':[round(a['pooled_centered_cosine'],4) for a in x['residual_lag_descriptors']]} for x in out],indent=2))


if __name__=='__main__':main()

"""Engineering qualification of the frozen diagnostic instrument."""
import importlib.util
import json
import time
from pathlib import Path
import torch
import instrument as I


def main():
    started=time.monotonic();out=I.HERE/'qualification';out.mkdir(exist_ok=False)
    manifest={};a,b,meta,dirs,mom,provenance=I.load_pair('Muon',500,manifest)
    cx,cy,x,y=I.data(meta);x=x[:1];y=y[:1]
    # Load the unchanged model source under an independent module name.
    spec=importlib.util.spec_from_file_location('original_diagnostic_reference',I.HERE/'source/original/model.py')
    module=importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    ref=module.GPT(module.ModelConfig(**meta['config']['model']));ref.load_state_dict(a['model']);ref.eval()
    with torch.no_grad():ref_logits=ref(x);ref_losses=torch.logsumexp(ref_logits.double(),-1)-ref_logits.double().gather(-1,y[...,None]).squeeze(-1)
    del ref,ref_logits
    model=I.make_model(meta['config']['model'],a['model'])
    with torch.no_grad():
        caches,z=I.capture(model,x);full=model(x)
        capture_error=float((z-full).abs().max())
        baseline=I.token_loss(z,y)
        baseline_error=float((baseline-ref_losses).abs().max())
        mean_error=float((baseline-ref_losses).mean().abs())
    assert capture_error<1e-11 and baseline_error<5e-5,(capture_error,baseline_error)
    block=3;fn,kick=I.ray_fn(model,block,caches[block],dirs[block])
    with torch.no_grad():suffix_error=float((fn(torch.tensor(0.,dtype=torch.float64))-z).abs().max())
    assert suffix_error<1e-11
    tick=time.monotonic();terms=I.directional(fn,y);derivative_seconds=time.monotonic()-tick
    fd={}
    with torch.no_grad():
        for h in [2**-6,2**-7]:
            lp=I.token_loss(fn(torch.tensor(h,dtype=torch.float64)),y)
            lm=I.token_loss(fn(torch.tensor(-h,dtype=torch.float64)),y)
            fd[h]=((lp-lm)/(2*h),(lp-2*baseline+lm)/(h*h))
    slope_rich=(4*fd[2**-7][0]-fd[2**-6][0])/3
    hess_rich=(4*fd[2**-7][1]-fd[2**-6][1])/3
    slope_error=float((slope_rich-terms['slope']).abs().max())
    hess_error=float((hess_rich-terms['hessian']).abs().max())
    assert slope_error<1e-7 and hess_error<1e-7,(slope_error,hess_error)
    joint=I.joint_fn(model,caches[0],dirs);coeff=torch.tensor([.5,-.25,1.],dtype=torch.float64)
    with torch.no_grad():
        expected=joint(coeff)
        saved={i:model.blocks[i].mlp.up.weight.clone() for i in I.BLOCKS}
        for i,v in zip(I.BLOCKS,coeff):model.blocks[i].mlp.up.weight.copy_(saved[i]+v*dirs[i])
        direct=model(x)
        for i in I.BLOCKS:model.blocks[i].mlp.up.weight.copy_(saved[i])
        joint_error=float((expected-direct).abs().max())
        restoration=float((model(x)-z).abs().max())
    assert joint_error<1e-10 and restoration==0.,(joint_error,restoration)
    # A nontrivial synthetic covariance checks all rotation identities.
    u=caches[block]['input'].flatten(0,1);c=u.T@u/u.shape[0]
    rotated,factors,operators,invariants=I.rotations(dirs[block],c,block)
    left_errors=[]
    for name in ['left0','left1']:
        other=torch.nn.functional.linear(caches[block]['input'],rotated[name])
        left_errors.append(float((other.norm(dim=-1)-kick.norm(dim=-1)).abs().max()))
    assert max(left_errors)<1e-10
    def small(v):return fn(v[0])
    tick=time.monotonic();h,sy=I.projected_hessian(small,y,1);h_seconds=time.monotonic()-tick
    h_compare=abs(float(h[0,0])-float(terms['hessian'].mean()));assert h_compare<1e-9
    result=dict(status='passed',seconds=time.monotonic()-started,capture_error=capture_error,
        fp32_fp64_max_token_nll_difference=baseline_error,fp32_fp64_mean_nll_difference=mean_error,
        suffix_error=suffix_error,fd_slope_error=slope_error,fd_hessian_error=hess_error,
        joint_current_input_error=joint_error,restoration_error=restoration,
        rotation_checks=invariants,left_token_norm_errors=left_errors,AD_hessian_crosscheck=h_compare,
        nested_JVP_seconds=derivative_seconds,reverse_Hessian1_seconds=h_seconds,
        mean_loss=float(baseline.mean()),mean_slope=float(terms['slope'].mean()),
        mean_true_Hessian=float(terms['hessian'].mean()),mean_GN=float(terms['gn'].mean()),
        provenance=provenance,input_manifest=manifest)
    I.write_json(out/'result.json',result)
    torch.save({k:v for k,v in terms.items() if k not in ['tangent','probability']},out/'qualified_derivatives.pt')
    print(json.dumps({k:v for k,v in result.items() if k not in ['provenance','input_manifest']},indent=2),flush=True)


if __name__=='__main__':main()

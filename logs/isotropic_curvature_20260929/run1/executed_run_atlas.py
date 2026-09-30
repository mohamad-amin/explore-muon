"""Produce the fixed 18-panel atlas, preserving raw observations and failures."""
import csv
import gc
import json
import resource
import shutil
import time
import traceback
import numpy as np
import torch
import instrument as I

OUT=I.HERE/'run1'
CAP=10800.


def npy(t):return t.detach().cpu().numpy()


def main():
    OUT.mkdir(exist_ok=False);started=time.monotonic()
    assert json.loads((I.HERE/'qualification/result.json').read_text())['status']=='passed'
    for name in ['PROTOCOL.md','instrument.py','run_atlas.py','source_manifest.json']:
        shutil.copy2(I.HERE/name,OUT/('executed_'+name))
    diagnostic_hashes={}
    for entry in json.loads((I.HERE/'source_manifest.json').read_text()):
        name=entry['original'].split('/')[-1];source=I.HERE/'source/adamw_spectra'/name
        assert I.sha(source.read_bytes())==entry['diagnostic_sha256']
        diagnostic_hashes[name]=entry['diagnostic_sha256']
    shutil.copytree(I.HERE/'source',OUT/'executed_source')
    I.write_json(OUT/'prepared.json',dict(diagnostic_hashes=diagnostic_hashes,
        methods=I.ARMS,steps=I.STEPS,blocks=I.BLOCKS,scales=I.SCALES,labels=I.LABELS,
        calibration_offset=I.CAL_OFFSET,score_offsets=I.SCORE_OFFSETS,cap_seconds=CAP))
    manifest={};all_summaries=[];state_times=[];completed=0;state_index=0
    def progress(**extra):
        elapsed=time.monotonic()-started
        row=dict(status='running',seconds=elapsed,completed_panels=completed,
                 total_panels=18,peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,**extra)
        I.write_json(OUT/'status.json',row)
        print(json.dumps(row),flush=True)
        assert elapsed<CAP,('Numerical time boundary',elapsed)
    reference_meta=None
    for step in I.STEPS:
        for method in I.ARMS:
            state_started=time.monotonic();state_index+=1
            state_dir=OUT/f'{method}_{step:06d}';state_dir.mkdir()
            before,after,meta,actual,momentum,provenance=I.load_pair(method,step,manifest)
            if reference_meta is None:reference_meta=meta
            for field in ['initial_model_sha256','train_manifest','validation_manifest','source_sha256','parameter_count']:
                assert meta[field]==reference_meta[field],(method,step,field)
            cx,cy,sx,sy=I.data(meta)
            if state_index==1:np.savez(OUT/'tokens.npz',calibration_x=npy(cx),calibration_y=npy(cy),score_x=npy(sx),score_y=npy(sy))
            model=I.make_model(meta['config']['model'],before['model'])
            cal={b:[] for b in I.BLOCKS};cal_z_norm={b:[] for b in I.BLOCKS}
            for j in range(8):
                cache,z=I.capture(model,cx[j:j+1])
                for b in I.BLOCKS:
                    cal[b].append(cache[b]['input'].squeeze(0))
                    cal_z_norm[b].append(cache[b]['preactivation'].norm(dim=-1).squeeze(0))
                del cache,z
            cov={}
            for b in I.BLOCKS:
                x=torch.stack(cal[b]);flat=x.flatten(0,1);cov[b]=flat.T@flat/flat.shape[0]
            score=[];base_losses=[]
            for j in range(4):
                cache,z=I.capture(model,sx[j:j+1]);score.append(cache)
                base_losses.append(I.token_loss(z,sy[j:j+1]).squeeze(0));del z
            base_losses=torch.stack(base_losses)
            state_meta=dict(method=method,step=step,metadata=meta,provenance=provenance,
                            baseline_nll_by_sequence=npy(base_losses.mean(-1)).tolist())
            I.write_json(state_dir/'metadata.json',state_meta)
            actual_tangents={};base_probability={};individual={}
            progress(method=method,step=step,stage='calibration_complete')
            for block in I.BLOCKS:
                tick=time.monotonic();panel=state_dir/f'block{block+1:02d}_up';panel.mkdir()
                directions,factors,operators,invariants=I.rotations(actual[block],cov[block],block)
                assert list(directions)==list(I.LABELS)
                loss=np.full((4,5,len(I.SCALES),512),np.nan)
                slope=np.full((4,5,512),np.nan);hess=np.full_like(slope,np.nan);gn=np.full_like(slope,np.nan)
                radius=np.full_like(slope,np.nan);hfull=np.full((4,5,5),np.nan)
                gnfull=np.full((4,512,5,5),np.nan);grads=[];errors=[]
                def partial():
                    np.savez(panel/'partial.npz',loss=loss,slope=slope,hessian=hess,gn=gn,
                             activation_radius=radius,projected_hessian=hfull,projected_gn=gnfull,
                             scales=np.array(I.SCALES),labels=np.array(I.LABELS))
                for j in range(4):
                    cache=score[j][block];target=sy[j:j+1]
                    grad,e=I.gradient_at_site(model,block,cache,target);grads.append(grad);errors.append(e.squeeze(0))
                    tangents=[];p=None
                    for di,(label,d) in enumerate(directions.items()):
                        fn,kick=I.ray_fn(model,block,cache,d)
                        term=I.directional(fn,target)
                        assert float((term['loss'].squeeze(0)-base_losses[j]).abs().max())<1e-10
                        loss[j,di,I.SCALES.index(0.)]=npy(term['loss'].squeeze(0))
                        slope[j,di]=npy(term['slope'].squeeze(0));hess[j,di]=npy(term['hessian'].squeeze(0));gn[j,di]=npy(term['gn'].squeeze(0))
                        radius[j,di]=npy(kick.norm(dim=-1).squeeze(0))
                        tangents.append(term['tangent']);p=term['probability']
                        if label=='actual':
                            actual_tangents[(block,j)]=term['tangent'];base_probability[j]=p
                        if label.startswith('left'):
                            assert np.max(abs(radius[j,di]-radius[j,0]))<1e-10
                        with torch.no_grad():
                            for si,s in enumerate(I.SCALES):
                                if s:loss[j,di,si]=npy(I.token_loss(fn(torch.tensor(s,dtype=torch.float64)),target).squeeze(0))
                        assert np.isfinite(loss[j,di]).all()
                        partial();progress(method=method,step=step,block=block+1,sequence=j,direction=label,
                                           stage='direction_complete')
                        del term,kick,fn
                    gnfull[j]=npy(I.predictive_gram(tangents,p).squeeze(0))
                    vector_fn=I.panel_fn(model,block,cache,list(directions.values()))
                    h,sy_error=I.projected_hessian(vector_fn,target,5);hfull[j]=npy(h)
                    diag_error=np.max(abs(np.diag(hfull[j])-hess[j].mean(-1)))
                    assert diag_error<1e-8*max(1.,np.max(abs(hfull[j]))),(method,step,block,j,diag_error)
                    assert np.max(abs(np.diagonal(gnfull[j],axis1=-2,axis2=-1).T-gn[j]))<1e-9
                    del tangents,p,vector_fn,h
                grad_mean=torch.stack(grads).mean(0)
                for di,(label,d) in enumerate(directions.items()):
                    agreement=abs(float((grad_mean*d).sum())-float(slope[:,di].mean()))
                    assert agreement<1e-8*max(1.,abs(float(slope[:,di].mean())))
                tensor_archive=dict(directions=directions,**factors,calibration_inputs=torch.stack(cal[block]),
                    calibration_preactivation_norm=torch.stack(cal_z_norm[block]),
                    score_inputs=torch.cat([c[block]['input'] for c in score]),
                    score_preactivations=torch.cat([c[block]['preactivation'] for c in score]),
                    score_residuals=torch.cat([c[block]['residual'] for c in score]),
                    output_gradient_sum_loss=torch.stack(errors),gradient_by_sequence=torch.stack(grads),
                    gradient_mean=grad_mean,stored_momentum=momentum[block],
                    weight_before=before['model'][f'blocks.{block}.mlp.up.weight'],
                    weight_after=after['model'][f'blocks.{block}.mlp.up.weight'])
                tensor_archive['singular_values']={name:torch.linalg.svdvals(t.double()) for name,t in
                    [('gradient',grad_mean),('momentum',momentum[block]),('actual_write',actual[block])]}
                tensor_archive['parameter_gram']=torch.tensor([[float((u*v).sum()) for v in directions.values()] for u in directions.values()],dtype=torch.float64)
                metric_directions=[d@factors['S'] for d in directions.values()]
                tensor_archive['input_metric_gram']=torch.tensor([[float((u*v).sum()) for v in metric_directions] for u in metric_directions],dtype=torch.float64)
                torch.save(tensor_archive,panel/'tensors.pt')
                np.savez(panel/'per_token.npz',loss=loss,slope=slope,hessian=hess,gn=gn,activation_radius=radius,
                         projected_hessian=hfull,projected_gn=gnfull,scales=np.array(I.SCALES),labels=np.array(I.LABELS))
                summary=dict(method=method,step=step,block=block+1,seconds=time.monotonic()-tick,
                    rotation_operators=operators,invariants=invariants,
                    direction_norms={k:float(v.norm()) for k,v in directions.items()},
                    direction_tensor_hashes={k:I.tensor_hash(v) for k,v in directions.items()},
                    curvature_mean={label:dict(slope=float(slope[:,i].mean()),hessian=float(hess[:,i].mean()),
                        gn=float(gn[:,i].mean()),activation_rms=float(np.sqrt(np.mean(radius[:,i]**2)))) for i,label in enumerate(I.LABELS)},
                    projected_hessian_mean=hfull.mean(0).tolist(),projected_gn_mean=gnfull.mean((0,1)).tolist())
                I.write_json(panel/'summary.json',summary);all_summaries.append(summary)
                individual[block]=dict(loss=loss[:,0].copy(),slope=slope[:,0].copy(),hessian=hess[:,0].copy(),gn=gn[:,0].copy())
                completed+=1;progress(method=method,step=step,block=block+1,stage='panel_complete')
                del tensor_archive,grads,errors,grad_mean,directions,factors
                gc.collect()
            # Joint actual writes, and full 3x3 coefficient-space Hessian/GN.
            joint_loss=np.full((4,len(I.SCALES),512),np.nan);joint_slope=np.full((4,512),np.nan)
            joint_hess=np.full((4,512),np.nan);joint_gn=np.full((4,512),np.nan)
            joint_hfull=np.full((4,3,3),np.nan);joint_gnfull=np.full((4,512,3,3),np.nan)
            for j in range(4):
                vf=I.joint_fn(model,score[j][0],actual);ones=torch.ones(3,dtype=torch.float64)
                fn=lambda a:vf(a*ones)
                term=I.directional(fn,sy[j:j+1]);joint_slope[j]=npy(term['slope'].squeeze(0))
                joint_hess[j]=npy(term['hessian'].squeeze(0));joint_gn[j]=npy(term['gn'].squeeze(0))
                assert np.max(abs(joint_slope[j]-sum(individual[b]['slope'][j] for b in I.BLOCKS)))<1e-8
                with torch.no_grad():
                    for si,s in enumerate(I.SCALES):joint_loss[j,si]=npy(I.token_loss(fn(torch.tensor(s,dtype=torch.float64)),sy[j:j+1]).squeeze(0))
                h,_=I.projected_hessian(vf,sy[j:j+1],3);joint_hfull[j]=npy(h)
                assert abs(float(h.sum())-float(joint_hess[j].mean()))<1e-8*max(1.,abs(float(h.sum())))
                gram=I.predictive_gram([actual_tangents[(b,j)] for b in I.BLOCKS],base_probability[j])
                joint_gnfull[j]=npy(gram.squeeze(0))
                assert np.max(abs(joint_gnfull[j].sum((-1,-2))-joint_gn[j]))<1e-8
                progress(method=method,step=step,sequence=j,stage='joint_sequence_complete')
            np.savez(state_dir/'joint.npz',loss=joint_loss,slope=joint_slope,hessian=joint_hess,gn=joint_gn,
                projected_hessian=joint_hfull,projected_gn=joint_gnfull,
                individual_loss=np.stack([individual[b]['loss'] for b in I.BLOCKS]),
                individual_slope=np.stack([individual[b]['slope'] for b in I.BLOCKS]),
                individual_hessian=np.stack([individual[b]['hessian'] for b in I.BLOCKS]),
                individual_gn=np.stack([individual[b]['gn'] for b in I.BLOCKS]),
                scales=np.array(I.SCALES),blocks=np.array(I.BLOCKS))
            for rec in provenance['files']:
                path=I.ROOT/rec['path'];assert path.stat().st_size==rec['bytes'] and path.stat().st_mtime_ns==rec['mtime_ns']
            state_seconds=time.monotonic()-state_started;state_times.append(state_seconds)
            I.write_json(state_dir/'status.json',dict(status='complete',seconds=state_seconds,panels=3))
            if state_index==1:
                forecast=(time.monotonic()-started)+5*state_seconds*1.15
                I.write_json(OUT/'forecast.json',dict(first_state_seconds=state_seconds,total_seconds=forecast,cap_seconds=CAP))
                assert forecast<CAP,('Full-panel forecast boundary',forecast)
            progress(method=method,step=step,stage='state_complete')
            del before,after,model,score,cal,actual_tangents,base_probability,individual
            gc.collect()
    I.write_json(OUT/'input_manifest.json',manifest)
    for rel,entry in manifest.items():assert I.sha((I.ROOT/rel).read_bytes())==entry['sha256']
    for name,expected in diagnostic_hashes.items():assert I.sha((I.HERE/'source/adamw_spectra'/name).read_bytes())==expected
    result=dict(status='complete',seconds=time.monotonic()-started,panels=18,states=6,
                directions_per_panel=5,score_contexts=4,calibration_contexts=8,
                scales=I.SCALES,labels=I.LABELS,state_seconds=state_times,summaries=all_summaries,
                executed_instrument_sha256=I.sha((OUT/'executed_instrument.py').read_bytes()),
                executed_protocol_sha256=I.sha((OUT/'executed_PROTOCOL.md').read_bytes()))
    I.write_json(OUT/'result.json',result);I.write_json(OUT/'status.json',{k:v for k,v in result.items() if k!='summaries'})
    print(json.dumps({k:v for k,v in result.items() if k!='summaries'}),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as e:
        if OUT.exists():I.write_json(OUT/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc()))
        raise

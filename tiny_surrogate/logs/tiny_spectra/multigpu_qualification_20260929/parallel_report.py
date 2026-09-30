"""Read the complete parallel execution qualification using its frozen gates."""
import argparse
import importlib.util
import hashlib
import json
import math
from pathlib import Path
import statistics
import torch
from torch.torch_version import TorchVersion


def read(p):return json.loads(Path(p).read_text())
def load(p):
    with torch.serialization.safe_globals([TorchVersion]):return torch.load(p,map_location='cpu',weights_only=True)


def tensors(x,prefix=''):
    if torch.is_tensor(x):yield prefix,x
    elif isinstance(x,dict):
        for k,v in x.items():yield from tensors(v,prefix+'/'+str(k))
    elif isinstance(x,(tuple,list)):
        for i,v in enumerate(x):yield from tensors(v,prefix+'/'+str(i))


def metric(a,b,plan):
    if a.shape!=b.shape or a.dtype!=b.dtype:raise ValueError('Tensor shape/dtype mismatch')
    a,b=a.double().reshape(-1),b.double().reshape(-1)
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():raise ValueError('Nonfinite comparison')
    error=b-a;norm=float(a.norm());other=float(b.norm());err=float(error.norm());n=max(1,a.numel())
    rms=norm/math.sqrt(n);rmse=err/math.sqrt(n)
    relative=err/norm if norm else 0. if err==0 else None
    cosine=float(torch.dot(a,b)/(norm*other)) if norm and other else 1. if norm==other==0 else None
    return dict(rmse=rmse,reference_rms=rms,maximum_absolute=float(error.abs().max()) if a.numel() else 0.,
        relative_l2=relative,cosine=cosine,linear_pass=rmse<=plan['same_state_rmse_absolute']+plan['same_state_rmse_relative']*rms,
        write_pass=relative is not None and cosine is not None and relative<=plan['write_relative_l2'] and cosine>=plan['write_cosine'])


def tree_exact(a,b):
    aa=dict(tensors(a));bb=dict(tensors(b))
    if set(aa)!=set(bb):return False
    return all(aa[k].dtype==bb[k].dtype and torch.equal(aa[k],bb[k]) for k in aa)


def first_update(reference,candidate,plan):
    groups={};passed=True
    pairs={'raw_gradients':(reference['raw_gradients'],candidate['raw_gradients']),
           'actual_deltas':(reference['actual_deltas'],candidate['actual_deltas']),
           'standard_state':(reference['optimizer']['state'],candidate['optimizer']['state']),
           'input_statistics':(reference['optimizer']['model_statistics'],candidate['optimizer']['model_statistics'])}
    for group,(a,b) in pairs.items():
        aa=dict(tensors(a));bb=dict(tensors(b));values={}
        if set(aa)!=set(bb):raise ValueError('Missing gradient/state tensor')
        for name,x in aa.items():
            y=bb[name]
            if not x.is_floating_point() or name.endswith('/step'):
                ok=x.dtype==y.dtype and torch.equal(x,y);values[name]=dict(exact_counter=ok);passed &= ok
            else:
                v=metric(x,y,plan);values[name]=v;passed &= v['write_pass' if group=='actual_deltas' else 'linear_pass']
        groups[group]=values
    return dict(passed=bool(passed),groups=groups)


def report(c):
    torch.set_num_threads(2)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=read(c/'READOUT_PLAN.json')['reader_sha256']:raise ValueError('Changed frozen reader')
    spec=importlib.util.spec_from_file_location('frozen_qualification',c/'qualification_driver.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);plan=module.verify(c)
    if not (c/'EXECUTION_COMPLETE.json').exists():raise ValueError('Qualification is not complete; do not read a partial family')
    out=c/'report'
    if out.exists():raise FileExistsError('Preserve the original qualification result')
    results=[];overall=True
    for method in plan['methods']:
        native_root=c/'runs'/('native_'+method);native=load(native_root/'final.pt');one_root=c/'runs'/('w1_'+method)
        one=load(one_root/'final.pt');one_first=load(one_root/'audit/step000001.pt')
        exact=tree_exact(native['model'],one['model']) and tree_exact(native['optimizer'],one['optimizer'])
        identity_keys=['initial_parameter_sha256','train_window_sha256','validation_bank_sha256','validation_bank_lengths_sha256','data_manifest_sha256']
        native_meta=read(native_root/'metadata.json');one_meta=read(one_root/'metadata.json')
        if any(native_meta[k]!=one_meta[k] for k in identity_keys):raise ValueError('Native/world1 identity mismatch')
        one_summary=read(one_root/'summary.json');comparisons=[]
        for world in [1,2,4]:
            root=c/'runs'/f'w{world}_{method}';meta=read(root/'metadata.json');summary=read(root/'summary.json');snapshot=load(root/'final.pt')
            if any(meta[k]!=one_meta[k] for k in identity_keys) or meta['world_size']!=world or summary['steps']!=22 or summary['tokens']!=21*262144+69*512:
                raise ValueError('Changed data/order/horizon/world size')
            if len(summary['rank_hardware'])!=world or any('A4000' not in h['name'] or not 0<h['total_memory_bytes']<45*1024**3 for h in summary['rank_hardware']):
                raise ValueError('Incorrect hardware inventory')
            if len(set(summary['all_rank_parameter_hashes']))!=1:raise ValueError('Parameter replicas differ')
            if not all(torch.isfinite(t).all() for _,t in tensors(snapshot)):raise ValueError('Nonfinite snapshot')
            pop=load(root/'final_validation.pt');refpop=load(native_root/'final_validation.pt')
            for key in ['starts','target_counts','document_indices']:
                if not torch.equal(pop[key],refpop[key]):raise ValueError('Changed full evaluation population')
            reconstructed=float((pop['sequence_nll']*pop['target_counts']).sum()/pop['target_counts'].sum())
            if abs(reconstructed-summary['full_validation_nll'])>1e-12:raise ValueError('Saved loss reconstruction failed')
            stats=snapshot['optimizer']['model_statistics']
            if method in ['pd','ts','spd','sts']:
                roots=snapshot['optimizer']['external']['data_norm']['roots']
                if len(stats)!=48 or len(roots)!=48 or any(v[1]!=21 for v in roots.values()):raise ValueError('Incomplete geometry state or root clock')
                if any(int(s['_total_forwards'])!=2706 or int(s['_step_forwards'])!=18 or int(s['_step_count'])!=69*511 for s in stats.values()):
                    raise ValueError('Global statistic counters changed')
            if method in ['soap','spd','sts'] and len(snapshot['optimizer']['external']['soap']['states'])!=48:
                raise ValueError('Incomplete SOAP state')
            first=load(root/'audit/step000001.pt');fidelity=first_update(one_first,first,plan)
            one_gn=one_summary['output_gn_sampling'];gn=summary['output_gn_sampling']
            gn_first_exact=(not gn and not one_gn) or bool(gn and one_gn and gn[0]==one_gn[0])
            if not gn_first_exact:raise ValueError('Initial GN draws differ at identical state')
            model_metrics={name:metric(one['model'][name],value,plan) for name,value in snapshot['model'].items()}
            one_records=[json.loads(x) for x in (one_root/'metrics.jsonl').read_text().splitlines()]
            records=[json.loads(x) for x in (root/'metrics.jsonl').read_text().splitlines()]
            curve_drift={key:max(abs(a[key]-b[key]) for a,b in zip(one_records,records) if key in a and key in b) for key in ['train_nll','validation_nll','train_probe_nll']}
            execution=read(c/f'w{world}_{method}_execution.json')
            comparisons.append(dict(world_size=world,first_update=fidelity,free_final_parameter_drift=model_metrics,
                free_curve_maximum_absolute_drift=curve_drift,final_full_loss_difference=summary['full_validation_nll']-one_summary['full_validation_nll'],
                output_gn_hashes_match=gn==one_gn,output_gn_sampling=gn,training_seconds=summary['training_seconds'],
                end_to_end_seconds=execution['wall_seconds'],active_worker_gpu_hours=execution['active_worker_gpu_seconds']/3600,
                training_speedup=one_summary['training_seconds']/summary['training_seconds'],
                end_to_end_speedup=read(c/f'w1_{method}_execution.json')['wall_seconds']/execution['wall_seconds'],
                execution_and_snapshot_checks_passed=True))
            overall &= fidelity['passed']
        overall &= exact
        results.append(dict(method=method,native_world1_tensors_exact=exact,comparisons=comparisons))
    empty=[]
    empty_reference=load(c/'runs/w1_empty/audit/step000001.pt')
    for world in [1,2,4]:
        root=c/'runs'/f'w{world}_empty';x=load(root/'final.pt');summary=read(root/'summary.json')
        states=x['optimizer']['model_statistics']
        if len(states)!=48 or any(int(s['_total_forwards'])!=1 or int(s['_step_forwards'])!=1 or int(s['_step_count'])!=511 for s in states.values()):
            raise ValueError('Zero-work-rank clock failed')
        if len(set(summary['all_rank_parameter_hashes']))!=1:raise ValueError('Empty-rank replicas differ')
        fidelity=first_update(empty_reference,load(root/'audit/step000001.pt'),plan)
        overall &= fidelity['passed']
        empty.append(dict(world_size=world,passed=fidelity['passed'],same_state=fidelity))
    fault=read(c/'worker_failure_execution.json')
    failure_pass=fault['returncode']!=0 and fault['wall_seconds']<plan['worker_failure_deadline_seconds'] and (c/'runs/worker_failure/failure_rank1.json').exists() and 'Intentional qualification worker failure' in read(c/'runs/worker_failure/failure_rank1.json')['error']
    overall &= failure_pass
    result=dict(qualification_passed=bool(overall),methods=results,empty_rank_checks=empty,worker_failure_passed=failure_pass,
        original_cuda_graph_failure_unchanged=True,scientific_ranking_claim=False,sealed_test_scored=False,
        limits='Same-state checks and short trajectories only. FP32 association changes; use the same GPU count within scientific comparisons. Root optimizer/evaluation remain serial.')
    out.mkdir();module.write(out/'results.json',result)
    lines=['# Single-run multi-GPU execution qualification','',f'Qualification passed: **{bool(overall)}**. All checks use16GB A4000s on one node.','',
        '| Method | Native/world1 exact | 2GPU training speedup | 4GPU training speedup | 2GPU wall speedup | 4GPU wall speedup |','|---|---|---:|---:|---:|---:|']
    for row in results:
        a,b=row['comparisons'][1:]
        lines.append(f"| {row['method']} | {row['native_world1_tensors_exact']} | {a['training_speedup']:.2f}x | {b['training_speedup']:.2f}x | {a['end_to_end_speedup']:.2f}x | {b['end_to_end_speedup']:.2f}x |")
    lines += ['',result['limits'],'','Complete per-tensor same-state metrics, accumulated parameter/loss drift, GN hashes, full-state inventories, partial-batch checks and active-worker GPU-hours are retained in results.json. Allocation cost for the benchmark includes all four reserved GPUs even during serial reference runs. No independent test panel was scored.']
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    return dict(qualification_passed=bool(overall),methods=[dict(method=r['method'],native_exact=r['native_world1_tensors_exact'],
        checks=[dict(world_size=x['world_size'],same_state_pass=x['first_update']['passed'],training_speedup=x['training_speedup'],loss_drift=x['final_full_loss_difference']) for x in r['comparisons']]) for r in results])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('cohort',type=Path);a=p.parse_args();print(json.dumps(report(a.cohort.resolve()),indent=2))

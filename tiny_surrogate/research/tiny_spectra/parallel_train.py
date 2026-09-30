"""One standalone experiment across 1, 2 or 4 GPUs; complete state on rank zero."""
import argparse
from dataclasses import asdict
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback
import torch
import torch.distributed as dist
from .model import GPT,ModelConfig
from .optim import make_optimizer,output_second_moments,snapshot_optimizer
from .train import amp,atomic_json,evaluate,model_hash,tensor_hash,schedule_factor,save_model_snapshot
from .parallel_ops import StatisticsSpan,microbatch_span,reduce_gradients,broadcast_parameters


def configure_execution(cfg):
    profile=cfg.get('execution_profile','deterministic')
    if profile not in ('native','deterministic'):raise ValueError('Unknown execution profile')
    if profile=='deterministic' and cfg.get('device','cuda')=='cuda' and os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Deterministic profile requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA; use the supplied launcher')
    torch.use_deterministic_algorithms(profile=='deterministic',warn_only=False)
    torch.backends.cudnn.deterministic=profile=='deterministic'
    torch.backends.cudnn.benchmark=False
    return profile


def initialize(cfg,expected_world):
    configure_execution(cfg)
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    if world not in (1,2,4) or world!=expected_world or int(os.environ.get('LOCAL_WORLD_SIZE',str(world)))!=world:
        raise ValueError('Expected one node with exactly the requested1,2,or4 workers')
    torch.set_num_threads(cfg.get('cpu_threads',2))
    if cfg.get('device','cuda')=='cuda':
        if torch.cuda.device_count()<world:raise ValueError('Not enough allocated visible GPUs')
        torch.cuda.set_device(local);device=torch.device('cuda',local);p=torch.cuda.get_device_properties(device)
        if p.total_memory>=45*1024**3:raise ValueError('Only GPUs strictly below nominal48GB are permitted')
        hardware=dict(name=p.name,total_memory_bytes=p.total_memory,capability=list(torch.cuda.get_device_capability(device)))
        torch.cuda.reset_peak_memory_stats(device)
    else:device=torch.device('cpu');hardware=dict(name='CPU test',total_memory_bytes=None)
    if world>1:dist.init_process_group('nccl' if device.type=='cuda' else 'gloo',device_id=device if device.type=='cuda' else None,timeout=timedelta(minutes=10))
    hardware.update(rank=rank,host=socket.gethostname(),local_rank=local)
    ranks=[None]*world
    if world>1:dist.all_gather_object(ranks,hardware)
    else:ranks=[hardware]
    if len({r['host'] for r in ranks})!=1 or len({r['name'] for r in ranks})!=1:
        raise ValueError('Single-node homogeneous GPU allocation required')
    return rank,world,device,hardware,ranks


def corpus_from_config(cfg):
    kind=cfg.get('corpus_kind','character')
    if kind=='fineweb_byte_bpe_v1':
        from .fineweb import load_training_corpus
        return load_training_corpus(cfg['data_path'],expected_manifest_sha256=cfg['data_sha256'])
    if kind=='tiny_stories_byte_bpe_v1':
        from .stories import load_training_corpus
        return load_training_corpus(cfg['data_path'],expected_manifest_sha256=cfg['data_sha256'])
    if kind=='character':
        from .data import load_corpus
        return load_corpus(cfg['data_path'],expected_sha256=cfg.get('data_sha256'))
    raise ValueError('Training/development corpus required; sealed-test data are unsupported')


def run(cfg,out,expected_world,failure_rank=None):
    if cfg.get('execution_backend','eager')!='eager':raise ValueError('Parallel runner supports eager execution only')
    rank,world,device,hardware,ranks=initialize(cfg,expected_world)
    out=Path(out);started=time.time()
    if rank==0:
        out.mkdir(parents=True,exist_ok=False);atomic_json(out/'config.json',cfg)
        atomic_json(out/'status.json',dict(status='initializing',world_size=world,started_unix=started))
    if world>1:dist.barrier()
    if failure_rank is not None and rank==failure_rank:raise RuntimeError('Intentional qualification worker failure')
    torch.manual_seed(cfg['seed'])
    if device.type=='cuda':torch.cuda.manual_seed_all(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    corpus=corpus_from_config(cfg);documents=hasattr(corpus,'evaluation_lengths')
    geometry=cfg['method'] in ('pd','ts','spd','sts')
    mc=ModelConfig(vocab_size=corpus.vocab_size,n_layer=cfg['n_layer'],n_embd=cfg['n_embd'],n_head=cfg['n_head'],seq_len=cfg['seq_len'],
        track_input_stats=geometry,track_input_cov=geometry,cov_stride=cfg['cov_stride'],cov_decay=cfg['cov_ema'],
        stats_decay=cfg.get('stats_ema',.99),stats_clock=cfg.get('stats_clock','step'))
    model=GPT(mc).to(device)
    # Disable the reused optimizer's automatic owner sharding explicitly. Only
    # rank0 constructs/steps it; every SOAP/root/Adam state is fully checkpointed.
    optimizer=make_optimizer(model,cfg,device,distributed_optimizer=False) if rank==0 else None
    parameters=dict(model.named_parameters());body=list(model.body_parameters());body_set=set(body);aux=[n for n in parameters if n not in body_set]
    if optimizer is not None and hasattr(optimizer,'capture_parameters'):optimizer.capture_parameters={parameters[body[0]],parameters[body[-1]]}
    initial_hash=model_hash(model);initials=[None]*world
    if world>1:dist.all_gather_object(initials,initial_hash)
    else:initials=[initial_hash]
    if len(set(initials))!=1:raise ValueError('Worker initialization differs')
    seqs,remainder=divmod(cfg['total_tokens'],cfg['seq_len']);batch_seqs,batch_remainder=divmod(cfg['batch_tokens'],cfg['seq_len'])
    if remainder or batch_remainder or min(seqs,batch_seqs,cfg['microbatch_sequences'])<1:raise ValueError('Invalid global token/microbatch budget')
    steps=math.ceil(seqs/batch_seqs)
    starts=corpus.window_positions('train',seqs,cfg['seq_len'],generator=torch.Generator().manual_seed(cfg['seed']+1729))
    val_starts=corpus.evaluation_starts('val',cfg['seq_len']) if hasattr(corpus,'evaluation_starts') else torch.arange(0,len(corpus.tokens('val'))-cfg['seq_len'],cfg['seq_len'])
    count=min(len(val_starts),cfg['validation_tokens']//cfg['seq_len'])
    if count<1:raise ValueError('Empty development evaluation bank')
    bank_indices=torch.linspace(0,len(val_starts)-1,count).round().long();bank=val_starts[bank_indices]
    probe=starts[:min(128,count)].clone() if documents else corpus.window_positions('train',min(128,count),cfg['seq_len'],generator=torch.Generator().manual_seed(91761))
    stream_hash=tensor_hash(starts);stream_hashes=[None]*world
    if world>1:dist.all_gather_object(stream_hashes,stream_hash)
    else:stream_hashes=[stream_hash]
    if len(set(stream_hashes))!=1:raise ValueError('Worker training streams differ')
    meta=dict(model_config=asdict(mc),n_parameters=model.num_parameters(),body_parameters=sum(parameters[n].numel() for n in body),
        auxiliary_parameters=sum(parameters[n].numel() for n in aux),initial_parameter_sha256=initial_hash,train_window_sha256=stream_hash,
        validation_bank_sha256=tensor_hash(bank),corpus=corpus.manifest,hardware=hardware,rank_hardware=ranks,world_size=world,
        host=socket.gethostname(),slurm_job_id=os.environ.get('SLURM_JOB_ID'),torch_version=str(torch.__version__),source_file=str(Path(__file__).resolve()),
        total_steps=steps,global_batch_tokens=cfg['batch_tokens'],repeated_corpus_exposure=cfg['total_tokens']/len(corpus.tokens('train')),
        covariance_clock='ordered global microforward EMA' if mc.stats_clock=='microforward' else 'global pooled step EMA',
        covariance_gram_precision='FP32; autocast disabled',statistic_composition='FP32 affine reassociation' if world>1 and mc.stats_clock=='microforward' else 'native serial collector',
        optimizer_state_owner=0,execution_profile=cfg.get('execution_profile','deterministic'),
        deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cublas_workspace_config=os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        gradient_reduction='SUM of globally weighted microbatch gradients before clipping',started_unix=started)
    if documents:
        lengths=corpus.evaluation_lengths('val',cfg['seq_len'])
        meta.update(data_manifest_sha256=cfg['data_sha256'],validation_bank_lengths_sha256=tensor_hash(lengths[bank_indices]),
            validation_bank_target_count=int(lengths[bank_indices].sum()),full_development_target_count=int(lengths.sum()),
            evaluation_policy='all within-document targets; masked right padding; token-weighted NLL')
    keep=cfg.get('keep_model_every',0);audit_steps=cfg.get('parallel_audit_steps',[])
    if type(keep) is not int or keep<0 or any(type(i) is not int or not 1<=i<=steps for i in audit_steps):raise ValueError('Invalid snapshot/audit schedule')
    if rank==0:
        atomic_json(out/'metadata.json',meta);torch.save(dict(train=starts,validation=bank,train_probe=probe),out/'windows.pt')
        metrics=(out/'metrics.jsonl').open('x',buffering=1)
        initial_val,_=evaluate(model,corpus,'val',bank,cfg,device)
        record=dict(step=0,tokens=0,validation_nll=initial_val,elapsed_seconds=time.time()-started)
        metrics.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
        if keep:save_model_snapshot(model,mc,out/'models/step000000.pt',step=0,tokens=0)
    if world>1:dist.barrier()
    training_seconds=0.;science_start=time.perf_counter();gn_hashes=[]
    for step in range(1,steps+1):
        # Keep evaluation/checkpoint time outside the training-step timer.
        if world>1:dist.barrier()
        if device.type=='cuda':torch.cuda.synchronize(device)
        tick=time.perf_counter();factor=schedule_factor(step,steps,cfg['warmup_fraction'],cfg['cooldown_fraction'])
        if rank==0:
            for group in optimizer.param_groups:group['lr']=cfg['lr']*factor*group.get('lr_scale',1.)
        batch=starts[(step-1)*batch_seqs:step*batch_seqs];chunks=list(batch.split(cfg['microbatch_sequences']))
        span=microbatch_span(len(chunks),world,rank);model.zero_grad(set_to_none=True)
        if rank==0 and cfg['method'] in ('ts','sts') and (step-1)%cfg['root_refresh']==0:
            selected=batch[:cfg['out_sequences']];x,y=corpus.batch('train',len(selected),cfg['seq_len'],positions=selected,device=device)
            generator=torch.Generator(device=device).manual_seed(cfg['seed']*1000003+step*97)
            gn_audit={}
            stats=output_second_moments(model,x,y,source='gn',precision=cfg['precision'],generator=generator,audit=gn_audit)
            optimizer.update_output_statistics(stats,cfg['out_ema'])
            gn_hashes.append(dict(step=step,selected_window_sha256=tensor_hash(selected),generator_seed=cfg['seed']*1000003+step*97,**gn_audit))
        collector=StatisticsSpan(model,world,rank,len(chunks),span,cfg);collector.begin()
        accumulated=torch.zeros((),device=device)
        for chunk in chunks[span[0]:span[1]]:
            x,y=corpus.batch('train',len(chunk),cfg['seq_len'],positions=chunk,device=device)
            with amp(device,cfg['precision']):loss=model(x,y)
            weight=len(chunk)/len(batch);(loss*weight).backward();accumulated+=loss.detach().float()*weight
        collector.finish();reduce_gradients(model,world,rank)
        if world>1:dist.reduce(accumulated,dst=0,op=dist.ReduceOp.SUM)
        report=step==1 or step%cfg['eval_every']==0 or step==steps
        if rank==0:
            raw={n:p.grad.detach().cpu().clone() for n,p in parameters.items()} if step in audit_steps else None
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['grad_clip'],error_if_nonfinite=True)
            before={n:p.detach().clone() for n,p in parameters.items()} if report or step in audit_steps else None
            optimizer.step()
        broadcast_parameters(model,world)
        if device.type=='cuda':torch.cuda.synchronize(device)
        if rank==0:
            elapsed=time.perf_counter()-tick;training_seconds+=elapsed
            record=dict(step=step,tokens=min(step*batch_seqs,seqs)*cfg['seq_len'],train_nll=float(accumulated),lr=cfg['lr']*factor,
                gradient_norm_before_clip=float(norm),step_seconds=elapsed,elapsed_seconds=time.perf_counter()-science_start,
                global_microbatches=len(chunks),rank_microbatch_spans=[list(microbatch_span(len(chunks),world,r)) for r in range(world)])
            if not math.isfinite(record['train_nll']):raise FloatingPointError('Nonfinite training loss')
            if report:
                for group_name,names in [('body',body),('auxiliary',aux)]:
                    record[group_name+'_parameter_norm']=float(sum(parameters[n].detach().square().sum() for n in names).sqrt())
                    record[group_name+'_actual_delta_norm']=float(sum((parameters[n].detach()-before[n]).square().sum() for n in names).sqrt())
                record['validation_nll'],_=evaluate(model,corpus,'val',bank,cfg,device)
                record['train_probe_nll'],_=evaluate(model,corpus,'train',probe,cfg,device)
                if hasattr(optimizer,'last_updates'):record['captured_direction_norms']={n:float(optimizer.last_updates[p].norm()) for n,p in parameters.items() if p in optimizer.last_updates}
                print(json.dumps(record),flush=True)
            if step in audit_steps:
                (out/'audit').mkdir(exist_ok=True)
                torch.save(dict(step=step,raw_gradients=raw,actual_deltas={n:(p.detach()-before[n]).cpu() for n,p in parameters.items()},
                    optimizer=snapshot_optimizer(model,optimizer)),out/'audit'/f'step{step:06d}.pt')
            del before
            metrics.write(json.dumps(record)+'\n')
            if keep and (step==1 or step%keep==0 or step==steps):save_model_snapshot(model,mc,out/'models'/f'step{step:06d}.pt',step=step,tokens=record['tokens'])
            atomic_json(out/'status.json',dict(status='running',step=step,total_steps=steps,tokens=record['tokens'],world_size=world,updated_unix=time.time()))
    hashes=[None]*world;final_hash=model_hash(model)
    if world>1:dist.all_gather_object(hashes,final_hash)
    else:hashes=[final_hash]
    if len(set(hashes))!=1:raise ValueError('Parameter replicas differ after broadcast')
    if rank==0:
        full,losses=evaluate(model,corpus,'val',val_starts,cfg,device)
        saved=dict(starts=val_starts,sequence_nll=losses);groups=[]
        if documents:
            counts=corpus.evaluation_lengths('val',cfg['seq_len']);ids=corpus.evaluation_document_indices('val',cfg['seq_len']);docs=corpus.development_documents
            saved.update(target_counts=counts,document_indices=ids)
            sums=torch.zeros(len(docs),dtype=torch.float64).scatter_add_(0,ids,losses.double()*counts)
            totals=torch.zeros(len(docs),dtype=torch.long).scatter_add_(0,ids,counts)
            docrows=[dict(identity=d['identity'],targets=int(totals[i]),nll=float(sums[i]/totals[i]) if totals[i] else None) for i,d in enumerate(docs)]
            atomic_json(out/'development_documents.json',dict(documents=docrows,weighting='token count',total_targets=int(totals.sum())))
            for group in range(4):
                members=[d for d in docrows if int(d['identity'][:2],16)%4==group and d['targets']];n=sum(d['targets'] for d in members)
                groups.append(dict(group_id=group,documents=len(members),targets=n,nll=sum(d['nll']*d['targets'] for d in members)/n if n else None))
        torch.save(saved,out/'final_validation.pt');snapshot=snapshot_optimizer(model,optimizer)
        torch.save(dict(model={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=snapshot,config=cfg,metadata=meta),out/'final.pt')
        summary=dict(status='complete',steps=steps,tokens=cfg['total_tokens'],world_size=world,final_validation_nll=record['validation_nll'],
            full_validation_nll=full,train_probe_nll=record['train_probe_nll'],training_seconds=training_seconds,total_seconds=time.time()-started,
            active_worker_gpu_seconds=world*(time.time()-started),peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else 0,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else 0,tensor_memory=snapshot['tensor_memory'],
            hardware=hardware,rank_hardware=ranks,n_parameters=model.num_parameters(),initial_parameter_sha256=initial_hash,train_window_sha256=stream_hash,
            final_parameter_sha256=final_hash,all_rank_parameter_hashes=hashes,optimizer_state_owner=0,output_gn_sampling=gn_hashes)
        if documents:summary.update(selection_metric=cfg.get('selection_metric','bank'),full_development_target_count=int(counts.sum()),development_groups=groups)
        atomic_json(out/'summary.json',summary);atomic_json(out/'status.json',dict(summary,finished_unix=time.time()));metrics.close()
        print(json.dumps(dict(event='complete',**summary)),flush=True)
    if world>1:dist.barrier();dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--out',required=True)
    p.add_argument('--expected-world-size',type=int,choices=[1,2,4],required=True)
    p.add_argument('--inject-failure-rank',type=int,default=None,help=argparse.SUPPRESS)
    a=p.parse_args();out=Path(a.out)
    if out.exists():raise FileExistsError('Preserve existing output: '+str(out))
    try:run(json.loads(Path(a.config).read_text()),out,a.expected_world_size,a.inject_failure_rank)
    except BaseException as error:
        if out.exists():
            rank=int(os.environ.get('RANK','0'));record=dict(error=repr(error),rank=rank,traceback=traceback.format_exc(),failed_unix=time.time())
            atomic_json(out/f'failure_rank{rank}.json',record)
            if rank==0:atomic_json(out/'failure.json',record);atomic_json(out/'status.json',dict(status='failed',**record))
        raise


if __name__=='__main__':main()

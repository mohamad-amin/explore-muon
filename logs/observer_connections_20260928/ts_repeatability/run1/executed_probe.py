"""CPU fixed-state output-factor repeatability; no optimizer or training step."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys
import time
import traceback
import numpy as np
import torch
import torch.nn.functional as F

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
sys.path.insert(0,str(HERE/'source'))
from adamw_spectra import gn_probe as P
from adamw_spectra import muon
from adamw_spectra.data import TokenStream

torch.set_num_threads(2);torch.set_num_interop_threads(1)
ARM=ROOT/'logs/muon_spectra/soaudit_batch16m_20260927/TS_a0.5_b0.5gn_b16M_lr0.028_mom0.9_s260925_l40s'
OFFSETS=[2_700_000_000,2_710_000_000,2_720_000_000,2_730_000_000]
START=time.time();OUT=None


def write(name,obj):
    tmp=OUT/(name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n');tmp.replace(OUT/name)


def status(stage,**kw):
    obj={'status':stage,'pid':os.getpid(),'seconds':time.time()-START,**kw}
    write('status.json',obj);print(json.dumps(obj),flush=True)
    if obj['seconds']>900:raise TimeoutError('Declared 900-second observer wall-clock bound')


def thash(x):return hashlib.sha256(x.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def inverse_root(c):
    e,u=torch.linalg.eigh(.5*(c.double()+c.double().T))
    unit=e.clamp_min(0)/e.clamp_min(0).mean().clamp_min(1e-30)
    root=((u*(unit+.001).pow(-.5))@u.T).float()
    assert torch.isfinite(root).all()
    return root,{'trace':float(c.double().trace()),'damping_fraction':float((unit<=.001).double().mean()),
                 'negative_eigenvalue_fraction':float((e<0).double().mean()),'max_over_mean':float(unit.max())}


def mapped(m,r,left=None,bf16=False):
    x=m.float()@r
    if left is not None:x=left@x
    if bf16:
        x=x/x.norm().clamp_min(1e-7)
        d=muon._iterate(x.bfloat16(),muon.NS_COEFFICIENTS)
    else:d=muon.newton_schulz(x,muon.NS_COEFFICIENTS)
    d=d@r
    if left is not None:d=left@d
    target=math.sqrt(min(m.shape))*math.sqrt(max(1.,m.shape[0]/m.shape[1]))
    before=float(d.norm());d=d*(target/d.norm().clamp_min(1e-30))
    assert torch.isfinite(d).all()
    assert abs(float(d.norm())/target-1)<1e-5
    return d,{'norm_before_match':before,'norm_after_match':float(d.norm()),'target_norm':target}


def output_factors(model,layers,x,y,seeds,check_gradient=False):
    outputs={};inputs={};handles=[]
    for name,layer in layers.items():
        handles.append(layer.register_forward_pre_hook(lambda mod,args,n=name:inputs.__setitem__(n,args[0].detach())))
        handles.append(layer.register_forward_hook(lambda mod,args,out,n=name:outputs.__setitem__(n,out)))
    try:
        logits=model(x)
        probs=logits.detach().float().softmax(-1).flatten(0,1)
        factors=[];checks={}
        for j,seed in enumerate(seeds):
            generator=torch.Generator().manual_seed(seed)
            target=torch.multinomial(probs,1,generator=generator).view(y.shape)
            loss=F.cross_entropy(logits.float().flatten(0,1),target.flatten(),reduction='sum')
            errors=torch.autograd.grad(loss,list(outputs.values()),retain_graph=check_gradient or j<len(seeds)-1)
            factors.append({name:(e.float().reshape(-1,e.shape[-1]).T@e.float().reshape(-1,e.shape[-1]))/x.numel() for name,e in zip(outputs,errors)})
            if check_gradient:
                weights=[layer.weight for layer in layers.values()]
                gg=torch.autograd.grad(loss,weights,retain_graph=j<len(seeds)-1)
                for (name,inp),e,g in zip(inputs.items(),errors,gg):
                    recon=e.float().reshape(-1,e.shape[-1]).T@inp.float().reshape(-1,inp.shape[-1])
                    err=float((recon.double()-g.double()).norm()/g.double().norm().clamp_min(1e-30))
                    checks[name]=err;assert err<=1e-4,(name,err)
            del errors,loss
    finally:
        for h in handles:h.remove()
    return factors,checks


def pair_metrics(a,b,pd):
    scopes={};by_matrix={}
    for name,x in a.items():
        y=b[name];p=pd[name]
        xx=float(x.double().square().sum());yy=float(y.double().square().sum());xy=float((x.double()*y.double()).sum())
        dist=float((x.double()-y.double()).square().sum())
        added=float((x.double()-p.double()).square().sum()+(y.double()-p.double()).square().sum())
        row={'norm2_a':xx,'norm2_b':yy,'dot':xy,'distance2':dist,'added_geometry_denominator':added}
        by_matrix[name]={**row,'cosine':xy/math.sqrt(xx*yy),'noise_to_added_geometry':dist/max(added,1e-30)}
        for scope in ['all',name.split('.')[-1]]:
            if scope not in scopes:scopes[scope]={k:0. for k in row}
            for k,v in row.items():scopes[scope][k]+=v
    for row in scopes.values():
        row['cosine']=row['dot']/math.sqrt(row['norm2_a']*row['norm2_b'])
        row['relative_distance']=math.sqrt(row['distance2']/row['norm2_a'])
        row['noise_to_added_geometry']=row['distance2']/max(row['added_geometry_denominator'],1e-30)
    return {'scopes':scopes,'matrices':by_matrix}


def factor_distance(a,b):
    result={}
    for name,x in a.items():
        y=b[name];xx=x.double()/x.double().trace().clamp_min(1e-30);yy=y.double()/y.double().trace().clamp_min(1e-30)
        result[name]={'trace_ratio':float(x.double().trace()/y.double().trace().clamp_min(1e-30)),
                      'relative_distance':float((xx-yy).norm()/xx.norm()),'cosine':float((xx*yy).sum()/(xx.norm()*yy.norm()))}
    return result


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='run1');args=parser.parse_args()
    OUT=HERE/args.run;OUT.mkdir(exist_ok=False)
    shutil.copy2(__file__,OUT/'executed_probe.py')
    checkpoint=ARM/'scientific/kept/step000046.pt';cp=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    statfile=checkpoint.with_name('step000046_input_stats_rank0.pt');stats=torch.load(statfile,map_location='cpu',weights_only=False,mmap=True)
    config=cp['config'];assert cp['step']==46
    model=P.build_model(config['model'],cp['model'],torch.device('cpu'));layers=P.hidden_linears(model)
    named=dict(model.named_parameters());keys=P.parameter_keys(model,list(layers))
    assert keys==[k for k,p in named.items() if k.startswith('blocks.') and p.ndim==2]
    for p in model.parameters():p.requires_grad_(False)
    for layer in layers.values():layer.weight.requires_grad_(True)
    ids=cp['optimizer']['param_groups'][0]['params'];assert len(ids)==48
    momentum={n:cp['optimizer']['state'][i]['momentum_buffer'].float().clone() for n,i in zip(layers,ids)}
    cov={n:stats[key[:-7]]['input_cov'].float()/stats[key[:-7]]['input_cov_weight'].clamp_min(1e-12) for n,key in zip(layers,keys)}
    assert all(momentum[n].shape==layers[n].weight.shape for n in layers)
    manifest=json.loads((HERE/'source/manifest.json').read_text())
    for name,row in manifest.items():assert hashlib.sha256((HERE/'source/adamw_spectra'/name).read_bytes()).hexdigest()==row['sha256']
    report={'status':'running','config':config,'source_manifest':manifest,'precision':'CPU FP32 forward/statistics/primary NS; explicit CPU BF16 pooled control is not CUDA replay',
            'fixed_input_source':'rank0 after46 covariance, proxy for otherowners/cached45 roots','momentum_source':'stored M46 at W46, not a replayed next step',
            'checkpoint':str(checkpoint),'input_stats':str(statfile),'input_metadata':{str(p):{'bytes':p.stat().st_size,'mtime_ns':p.stat().st_mtime_ns} for p in [checkpoint,statfile]},
            'accessed_momentum_sha256':{n:thash(m) for n,m in momentum.items()},'accessed_cov_sha256':{n:thash(c) for n,c in cov.items()}}
    stream=TokenStream(str(ROOT/config['train_pattern']));device=torch.device('cpu')
    x,y=stream.batch(2_650_000_000,1,512,device)
    t0=time.time();qual,gradcheck=output_factors(model,layers,x,y,[987654],True);factor_seconds=time.time()-t0
    ref=muon.output_second_moments(model,x,y,'gn',config,device,torch.Generator().manual_seed(987654))
    factor_errors={n:float((qual[0][n].double()-ref[layer.weight].double()).norm()/ref[layer.weight].double().norm().clamp_min(1e-30)) for n,layer in layers.items()}
    assert max(factor_errors.values())<=1e-5
    t0=time.time();r_large,_=inverse_root(cov['block01.down']);root_large=time.time()-t0
    t0=time.time();l_large,_=inverse_root(qual[0]['block01.up']);left_large=time.time()-t0
    r_up,_=inverse_root(cov['block01.up'])
    t0=time.time();_,mapcheck=mapped(momentum['block01.up'],r_up,l_large);map_seconds=time.time()-t0
    t0=time.time();bf,_=mapped(momentum['block01.up'],r_up,l_large,True);bf_seconds=time.time()-t0
    # Deliberately conservative: price every dimension as the largest and every
    # scored forward as qualification (which also reconstructs weight gradients).
    forecast=48*factor_seconds+48*root_large+17*48*(left_large+map_seconds)+48*bf_seconds
    report['qualification']={'factor_reference_relative_errors':factor_errors,'gradient_reconstruction_relative_errors':gradcheck,
        'one_sequence_factor_and_gradient_seconds':factor_seconds,'largest_input_root_seconds':root_large,'largest_output_root_seconds':left_large,
        'largest_map_seconds':map_seconds,'largest_bf16_map_seconds':bf_seconds,'norm_check':mapcheck,'conservative_projected_seconds':forecast}
    write('result.json',report);status('qualified',forecast_seconds=forecast)
    if forecast>900:report['status']='cost_gate_stop';write('result.json',report);status('cost_gate_stop');return
    roots={};root_info={}
    for i,(n,c) in enumerate(cov.items()):
        roots[n],root_info[n]=inverse_root(c)
        if i%8==0:status('input_roots',matrix=i)
    torch.save({'roots':roots,'source':report['fixed_input_source']},OUT/'fixed_R.pt');report['input_root_info']=root_info
    del ref,qual,cov,r_large,l_large,r_up,bf
    banks={};token_arrays={}
    for bank,offset in zip('ABCD',OFFSETS):
        draws=2 if bank in 'AB' else 1
        sums=[{n:torch.zeros(layer.weight.shape[0],layer.weight.shape[0]) for n,layer in layers.items()} for _ in range(draws)]
        bx,by=stream.batch(offset,8,512,device);token_arrays[bank+'_x']=bx.numpy();token_arrays[bank+'_y']=by.numpy()
        for seq in range(8):
            seeds=[260928000+10000*ord(bank)+100*draw+seq for draw in range(draws)]
            obs,_=output_factors(model,layers,bx[seq:seq+1],by[seq:seq+1],seeds)
            for target,one in zip(sums,obs):
                for n,b in one.items():target[n].add_(b,alpha=1/8)
            status('bank_factors',bank=bank,sequence=seq+1)
        for draw,one in enumerate(sums):
            name=bank+str(draw);banks[name]=one;torch.save(one,OUT/f'B_{name}.pt')
    np.savez(OUT/'tokens.npz',**token_arrays);report['tokens_sha256']={n:hashlib.sha256(a.tobytes()).hexdigest() for n,a in token_arrays.items()}
    del model,layers,stats,cp
    def pooled(labels):return {n:sum(banks[k][n] for k in labels)/len(labels) for n in momentum}
    matrices={**banks,'AB':pooled(['A0','B0']),'CD':pooled(['C0','D0']),'ABCD':pooled(['A0','B0','C0','D0'])}
    for bank in 'ABCD':
        anchor=pooled([b+'0' for b in 'ABCD' if b!=bank]);matrices['anchor_'+bank]=anchor
        matrices['refresh_'+bank]={n:.8*anchor[n]+.2*banks[bank+'0'][n] for n in momentum}
    directions={'PD':{n:mapped(m,roots[n])[0] for n,m in momentum.items()}};all_roots={};map_info={}
    torch.save(directions['PD'],OUT/'D_PD.pt')
    for label,factors in matrices.items():
        ds={};ls={};info={}
        for n,m in momentum.items():
            left,meta=inverse_root(factors[n]);d,mi=mapped(m,roots[n],left)
            ds[n]=d;ls[n]=left;info[n]={**meta,**mi}
        directions[label]=ds;all_roots[label]=ls;map_info[label]=info
        if label in ['A0','B0','C0','D0','AB','CD','ABCD']:torch.save(ds,OUT/f'D_{label}.pt')
        status('mapped',label=label)
    directions['ABCD_bf16']={n:mapped(m,roots[n],all_roots['ABCD'][n],True)[0] for n,m in momentum.items()}
    torch.save(directions['ABCD_bf16'],OUT/'D_ABCD_bf16.pt')
    comparisons=[]
    labels=['A0','B0','C0','D0']
    comparisons += [(labels[i],labels[j],'independent8') for i in range(4) for j in range(i+1,4)]
    comparisons += [('A0','A1','same_sequences_new_labels'),('B0','B1','same_sequences_new_labels'),('AB','CD','independent16'),
                    ('AB','ABCD','nested_pool'),('CD','ABCD','nested_pool'),('ABCD','ABCD_bf16','numerical_control')]
    comparisons += [('anchor_'+b,'refresh_'+b,'ema_refresh_proxy') for b in 'ABCD']
    pairs={}
    for a,b,kind in comparisons:
        row=pair_metrics(directions[a],directions[b],directions['PD']);row['comparison_type']=kind
        if b!='ABCD_bf16':
            row['trace_normalized_B']=factor_distance(matrices[a],matrices[b]);row['trace_normalized_L']=factor_distance(all_roots[a],all_roots[b])
        pairs[a+'__'+b]=row
    pd_pairs={label:pair_metrics(ds,directions['PD'],directions['PD']) for label,ds in directions.items() if label!='PD'}
    report['direction_pairs']=pairs;report['versus_PD']=pd_pairs;report['map_info']=map_info
    e8=[v['scopes']['all']['cosine'] for v in pairs.values() if v['comparison_type']=='independent8'];e16=pairs['AB__CD']['scopes']['all']['cosine']
    report['descriptive_flag']='stable' if min(e8)>=.99 and e16>=.995 else ('material_sample_sensitivity' if min(e8)<=.90 and e16>min(e8) else 'mixed')
    report['status']='complete';report['seconds']=time.time()-START
    for path,rec in report['input_metadata'].items():
        s=Path(path).stat();assert s.st_size==rec['bytes'] and s.st_mtime_ns==rec['mtime_ns']
    write('result.json',report);status('complete',flag=report['descriptive_flag'])


if __name__=='__main__':
    try:main()
    except Exception as exc:
        if OUT is not None:
            (OUT/'failure.txt').write_text(traceback.format_exc())
            write('status.json',{'status':'failed','type':type(exc).__name__,'message':str(exc),'seconds':time.time()-START,'pid':os.getpid()})
        raise

import argparse,json,os,sys,time
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('label');p.add_argument('engine');p.add_argument('profile');a=p.parse_args()
c=a.root.resolve();cfg=json.loads((c/'config.json').read_text());out=c/'runs'/a.label
if out.exists():raise FileExistsError(out)
if a.profile=='deterministic':
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Workspace must be set before CUDA initialization')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
start=time.time()
if a.engine=='native':
    import research.tiny_spectra.train as native
    original_make=native.make_optimizer;original_clip=torch.nn.utils.clip_grad_norm_;state={}
    def make(model,*args,**kwargs):
        state['model']=model;opt=original_make(model,*args,**kwargs)
        def after(optimizer,args,kwargs):
            params=dict(model.named_parameters())
            (out/'audit').mkdir(exist_ok=True)
            torch.save(dict(step=1,raw_gradients=state['raw'],actual_deltas={n:(p.detach()-state['before'][n]).cpu() for n,p in params.items()},
                optimizer=native.snapshot_optimizer(model,optimizer)),out/'audit/step000001.pt')
        opt.register_step_post_hook(after);return opt
    def clip(parameters,*args,**kwargs):
        params=list(parameters)
        state['raw']={n:p.grad.detach().cpu().clone() for n,p in state['model'].named_parameters()}
        state['before']={n:p.detach().clone() for n,p in state['model'].named_parameters()}
        return original_clip(params,*args,**kwargs)
    native.make_optimizer=make;torch.nn.utils.clip_grad_norm_=clip
    native.run(cfg,out)
else:
    from research.tiny_spectra.parallel_train import run
    run(cfg,out,1)
(c/(a.label+'_receipt.json')).write_text(json.dumps(dict(engine=a.engine,profile=a.profile,elapsed_seconds=time.time()-start,
    deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),cublas_workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG')),indent=2)+'\n')

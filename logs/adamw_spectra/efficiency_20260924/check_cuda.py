import copy, json, time
from pathlib import Path
import torch
from research.adamw_spectra.model import GPT, ModelConfig
from research.adamw_spectra.measure import adamw_direction, measure_updates
from research.adamw_spectra.train import atomic_json, native_compilers

compilers=native_compilers()
torch.set_num_threads(2)
torch.manual_seed(29)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
root=Path('logs/adamw_spectra/efficiency_20260924')
base=GPT(ModelConfig(vocab_size=128,n_layer=2,n_embd=32,n_head=2,seq_len=16)).cuda()
fast=copy.deepcopy(base)
compiled=torch.compile(fast,dynamic=False)
a=torch.optim.AdamW(base.parameters(),lr=.0006,betas=(.9,.95),weight_decay=.01,eps=1e-8,foreach=False,fused=False)
b=torch.optim.AdamW(fast.parameters(),lr=.0006,betas=(.9,.95),weight_decay=.01,eps=1e-8,foreach=False,fused=True)
x=torch.randint(0,128,(2,16),device='cuda');y=torch.randint(0,128,(2,16),device='cuda')
records=[]
start=time.perf_counter()
for step in range(3):
 a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
 l1=base(x,y);l2=compiled(x,y)
 torch.testing.assert_close(l1,l2,atol=2e-5,rtol=2e-5)
 l1.backward();l2.backward()
 error=max(float((p.grad-q.grad).norm()/p.grad.norm().clamp_min(1e-10)) for p,q in zip(base.parameters(),fast.parameters()))
 diagnostics=[{'name':name,'reference_norm':float(p.grad.norm()),'difference_norm':float((p.grad-q.grad).norm())} for (name,p),q in zip(base.named_parameters(),fast.parameters())]
 global_error=float(torch.sqrt(sum((p.grad-q.grad).square().sum() for p,q in zip(base.parameters(),fast.parameters())))/torch.sqrt(sum(p.grad.square().sum() for p in base.parameters())))
 print(json.dumps({'step':step+1,'global_gradient_relative_l2_difference':global_error,'near_zero_gradients':[v for v in diagnostics if v['reference_norm']<1e-7]}),flush=True)
 for p,q in zip(base.parameters(),fast.parameters()):
  torch.testing.assert_close(p.grad,q.grad,atol=2e-7,rtol=.002)
 assert global_error < .002,global_error
 before=fast.blocks[0].attn.q.weight.detach().clone()
 a.step();b.step()
 update,_=adamw_direction(b,fast.blocks[0].attn.q.weight)
 expected=before*(1-.0006*.01)-.0006*update
 torch.testing.assert_close(fast.blocks[0].attn.q.weight,expected,atol=2e-7,rtol=2e-5)
 records.append({'step':step+1,'loss_abs_difference':abs(l1.item()-l2.item()),'max_parameterwise_gradient_relative_l2_difference':error,'global_gradient_relative_l2_difference':global_error})
# Test real BF16 autocast forward/backward through the compiled path, independently
# of accumulated trajectory differences from the optimizer implementation.
base.load_state_dict(fast.state_dict());a.zero_grad(set_to_none=True);b.zero_grad(set_to_none=True)
with torch.autocast('cuda',dtype=torch.bfloat16):
 eager_loss=base(x,y);compiled_loss=compiled(x,y)
eager_loss.backward();compiled_loss.backward()
torch.testing.assert_close(eager_loss,compiled_loss,atol=.003,rtol=.0005)
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in fast.parameters())
relative_error=float(torch.sqrt(sum((p.grad-q.grad).square().sum() for p,q in zip(base.parameters(),fast.parameters())))/torch.sqrt(sum(p.grad.square().sum() for p in base.parameters())))
assert relative_error < .05,relative_error
b.step()
rows,values=measure_updates(b,fast.measured_parameters(),device='cpu')
assert len(values)==12
assert all(abs(row['normalized_energy_sum']-1)<1e-4 for row in rows.values())
torch.cuda.synchronize()
result={'scope':'Tiny CUDA correctness check; concurrent GPU jobs make this unsuitable as a throughput benchmark','device':torch.cuda.get_device_name(),'compilers':compilers,'fp32_comparison':records,'bf16_loss_abs_difference':abs(eager_loss.item()-compiled_loss.item()),'bf16_gradient_relative_l2_difference':relative_error,'measured_matrices':len(values),'peak_allocated_mb':torch.cuda.max_memory_allocated()/2**20,'elapsed_including_compilation_seconds':time.perf_counter()-start}
atomic_json(root/'cuda_check.json',result)
print(json.dumps(result,indent=2),flush=True)

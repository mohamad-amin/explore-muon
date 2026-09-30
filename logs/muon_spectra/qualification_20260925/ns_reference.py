from pathlib import Path
import json,torch
from research.adamw_spectra.muon import newton_schulz
from research.adamw_spectra.train import native_compilers
native_compilers()
torch.set_num_threads(2)
torch.backends.cuda.matmul.allow_tf32=False
root=Path('logs/muon_spectra/qualification_20260925')
saved=torch.load(root/'cuda_diagnostic/replica_failure_rank0.pt',map_location='cpu',weights_only=False)
banks={}
for state in saved['optimizer']['state'].values():
    if 'momentum_buffer' in state:
        m=state['momentum_buffer']
        banks.setdefault(tuple(m.shape),[]).append(m)
compiled=torch.compile(newton_schulz,fullgraph=True,dynamic=False)
report={}
for shape,matrices in banks.items():
    x=torch.stack(matrices).cuda()
    expected=newton_schulz(x)
    observed=compiled(x)
    relative=float((observed-expected).norm()/expected.norm())
    assert torch.isfinite(observed).all() and relative<.02,(shape,relative)
    report[str(shape)]={'relative_frobenius_difference_compiled_vs_eager_bf16':relative,'max_absolute_difference':float((observed-expected).abs().max())}
(root/'NS_GPU_REFERENCE.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

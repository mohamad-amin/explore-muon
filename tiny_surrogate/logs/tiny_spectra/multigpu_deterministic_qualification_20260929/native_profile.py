import argparse,json,os
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--out',required=True);a=p.parse_args()
cfg=json.loads(Path(a.config).read_text())
if cfg.get('execution_profile')=='deterministic':
    if os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':raise ValueError('Missing deterministic workspace')
    torch.use_deterministic_algorithms(True,warn_only=False)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
from research.tiny_spectra.train import run
run(cfg,a.out)
Path(a.out,'execution_profile.json').write_text(json.dumps(dict(profile=cfg.get('execution_profile','native'),deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),workspace=os.environ.get('CUBLAS_WORKSPACE_CONFIG'))))

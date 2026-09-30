"""Run only the frozen single-attempt graph benchmark from the study root."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
FROZEN=ROOT/'frozen'
for relative,expected in json.loads((ROOT/'source_manifest.json').read_text()).items():
 p=Path(relative)
 if p.is_absolute() or '..' in p.parts or hashlib.sha256((FROZEN/p).read_bytes()).hexdigest()!=expected:
  raise ValueError('Frozen benchmark source changed: '+relative)
sys.path.insert(0,str(FROZEN))
from research.tiny_spectra import execution_benchmark,execution_graph,train,model,optim,fineweb,stories,data
for module in (execution_benchmark,execution_graph,train,model,optim,fineweb,stories,data):
 if not Path(module.__file__).resolve().is_relative_to(FROZEN):
  raise ValueError('Unfrozen execution dependency: '+module.__file__)
execution_benchmark.run(ROOT)

"""Execute only the byte-verified frozen data preparation module."""
import hashlib
import json
from pathlib import Path
import sys
import time

RUN_ROOT = Path(__file__).resolve().parent
FROZEN = RUN_ROOT / "frozen"
CONFIG = Path('/share/data/dl-theory/amin/projects/explore_muon/tiny_surrogate/data/fineweb_d_20260928/preparation_config.json')
CONFIG_SHA = '8558357d06374df89604f535a758d1f474918669b25d945643bf3dd9666a287d'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_new(name, value):
    with (RUN_ROOT / name).open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")

manifest = json.loads((RUN_ROOT / "source_manifest.json").read_text())
for relative, expected in manifest.items():
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or sha(FROZEN / path) != expected:
        raise ValueError("Frozen preparation source changed: " + relative)
if sha(CONFIG) != CONFIG_SHA:
    raise ValueError("Reviewed preparation configuration changed")
sys.path.insert(0, str(FROZEN))
from research.tiny_spectra import fineweb, stories, data
for module in (fineweb, stories, data):
    if not Path(module.__file__).resolve().is_relative_to(FROZEN):
        raise ValueError("Preparation imported a live instead of frozen dependency")
write_new("EXECUTION_STARTED.json", dict(started_unix=time.time(), config_sha256=CONFIG_SHA,
    source_manifest_sha256=sha(RUN_ROOT / "source_manifest.json"),
    source_file=fineweb.__file__, model_inference=False, gpu_requested=False))
try:
    result = fineweb.prepare(CONFIG)
except BaseException as error:
    write_new("EXECUTION_FAILURE.json", dict(error=repr(error), failed_unix=time.time()))
    raise
write_new("EXECUTION_COMPLETE.json", dict(result=result, completed_unix=time.time(),
    model_inference=False, gpu_requested=False))

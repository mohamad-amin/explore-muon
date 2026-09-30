"""Arm configs for the H100 head-to-head (PD vs Newton-Muon vs PMuon vs tuned Muon). Every key explicit."""
import json
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])                       # directory holding the adamw_spectra package
from adamw_spectra.train import load_config

REPO = Path("/share/data/dl-theory/amin/projects/explore_muon")
BASE = json.loads((REPO / "logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925/config_muon.json").read_text())
STATS = {**BASE["model"], "track_input_stats": True, "track_input_cov": True}
METHODS = {
    "M": lambda lr: {"learning_rate": lr},
    "PD": lambda lr: {"learning_rate": lr, "data_norm_alpha": 0.25, "model": STATS},
    "NM": lambda lr: {"learning_rate": lr, "newton_muon": True, "model": STATS},
    "PM": lambda lr: {"learning_rate": lr, "pmuon": True},
}


def arm(method, lr, seed):
    name = f"{method}{'_a0.25' if method == 'PD' else ''}_lr{lr:g}_s{seed}_h100"
    return name, load_config(overrides={**BASE, "seed": seed, **METHODS[method](lr)})


if __name__ == "__main__":
    out = Path(sys.argv[2])
    for spec in sys.argv[3:]:                          # e.g. NM:0.007:260924
        method, lr, seed = spec.split(":")
        name, config = arm(method, float(lr), int(seed))
        (out / name).mkdir(parents=True, exist_ok=False)
        (out / name / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        print(name)

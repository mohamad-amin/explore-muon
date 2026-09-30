"""Track 3 recipe in our setup: wd 0.2 + cooldown 0.7 + batch 524,288 together (MUON_CASE.md, 23:19 CDT decision).

Seed 260925 on A6000 (GPU SVD); 2938 steps (same tokens as the 1M-batch runs). Reading: PD - Muon >= -0.008 means
our setup reproduces Track 3's shrinkage (batch-driven); <= -0.015 points at the model or Track 3's LR regime.
Usage from the project root: .venv/bin/python logs/muon_spectra/t3recipe_20260925/make_arms.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "frozen"))
from adamw_spectra.train import load_config

REPO = HERE.parents[2]
BASE = json.loads((REPO / "logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925/config_muon.json").read_text())
STATS = {"track_input_stats": True, "track_input_cov": True}
RECIPE = {"weight_decay": 0.2, "cooldown_fraction": 0.7, "batch_tokens": 524288}
ARMS = {
    "M": ({"learning_rate": 0.007}, False),
    "PD": ({"learning_rate": 0.01, "data_norm_alpha": 0.25}, True),
    "PDgeo2": ({"learning_rate": 0.01, "data_norm_alpha": 0.25, "data_norm_decay": "geometry", "data_norm_decay_power": 2}, True),
}

if __name__ == "__main__":
    for arm, (change, stats) in ARMS.items():
        overrides = {**BASE, "seed": 260925, "svd_device": "cuda", **RECIPE, **change,
                     "model": {**BASE["model"], **(STATS if stats else {})}}
        name = f"{arm}_t3recipe_s260925_a6000"
        (HERE / name).mkdir(exist_ok=False)
        (HERE / name / "config.json").write_text(json.dumps(load_config(overrides=overrides), indent=2) + "\n")
        print(name)

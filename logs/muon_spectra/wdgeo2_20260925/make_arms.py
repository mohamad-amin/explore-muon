"""The missing 2x2 cell and the cooldown interaction (MUON_CASE.md, 22:30 CDT decision).

Our setup, seed 260925 on A6000 (GPU SVD), wd 0.2. "MG" = Muon's update direction (data-norm pre- and
post-whitening off, so the update is polar(M) at Muon's Frobenius norm) with PD-geometry weight decay
(only the decay uses R). Pairs: wdgeo_20260925 and transfer_20260925 (cooldown 0.1); here also cooldown 0.7.
Usage from the project root: .venv/bin/python logs/muon_spectra/wdgeo2_20260925/make_arms.py
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
GEO = {"data_norm_decay": "geometry", "data_norm_decay_power": 2}
MG = {"learning_rate": 0.007, "data_norm_alpha": 0.25, "data_norm_pre": False, "data_norm_post": False, **GEO}
ARMS = {  # name: (overrides, needs input statistics)
    "MG_wd0.2": (MG, True),
    "M_wd0.2_cd0.7": ({"learning_rate": 0.007, "cooldown_fraction": 0.7}, False),
    "PD_wd0.2_cd0.7": ({"learning_rate": 0.01, "data_norm_alpha": 0.25, "cooldown_fraction": 0.7}, True),
    "PDgeo2_wd0.2_cd0.7": ({"learning_rate": 0.01, "data_norm_alpha": 0.25, "cooldown_fraction": 0.7, **GEO}, True),
    "MG_wd0.2_cd0.7": ({**MG, "cooldown_fraction": 0.7}, True),
}

if __name__ == "__main__":
    for arm, (change, stats) in ARMS.items():
        overrides = {**BASE, "seed": 260925, "svd_device": "cuda", "weight_decay": 0.2, **change,
                     "model": {**BASE["model"], **(STATS if stats else {})}}
        name = f"{arm}_s260925_a6000"
        (HERE / name).mkdir(exist_ok=False)
        (HERE / name / "config.json").write_text(json.dumps(load_config(overrides=overrides), indent=2) + "\n")
        print(name)

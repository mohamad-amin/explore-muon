"""Decay-geometry test: does preconditioning PD's weight decay undo the wd-0.2 flip? (MUON_CASE.md decision, ~19:50 CDT)

Our setup, seed 260925 on A6000 (GPU SVD). Compare with transfer_20260925/{M,PD}_wd0.2_s260925_a6000.
Usage from the project root: .venv/bin/python logs/muon_spectra/wdgeo_20260925/make_arms.py
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
PD = {"data_norm_alpha": 0.25, "weight_decay": 0.2}
ARMS = {
    "PD_wd0.2_geo2": {"learning_rate": 0.01, "data_norm_decay": "geometry", "data_norm_decay_power": 2},
    "PD_wd0.2_geo1": {"learning_rate": 0.01, "data_norm_decay": "geometry", "data_norm_decay_power": 1},
    "PD_lr0.007_wd0.2": {"learning_rate": 0.007},   # Muon's per-step shrink (0.007 x 0.2), decoupled
}

if __name__ == "__main__":
    for arm, change in ARMS.items():
        overrides = {**BASE, "seed": 260925, "svd_device": "cuda", **PD, **change,
                     "model": {**BASE["model"], **STATS}}
        name = f"{arm}_s260925_a6000"
        (HERE / name).mkdir(exist_ok=False)
        (HERE / name / "config.json").write_text(json.dumps(load_config(overrides=overrides), indent=2) + "\n")
        print(name)

"""Weight decay x cooldown in the Track 3 recipe setting (MUON_CASE.md, ~15:00 CDT 2026-09-26 decision).

Our setup with batch 524,288 (2938 steps), seed 260925 on A6000 (GPU SVD). 2x2 of WD {0.1, 0.3} x cooldown {0.5, 0.9}
around the existing center (WD 0.2, cooldown 0.7): PD + geometry decay (alpha 1/8, p = 2, LR 0.01; center 3.63015,
t3screen_20260926/PDgeo2_a0.125) and Muon (LR 0.01; center 3.64032, t3screen_20260926/M_lr0.01).
Usage from the project root: .venv/bin/python logs/muon_spectra/t3wdcd_20260926/make_arms.py
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
METHODS = {
    "PDgeo2_a0.125": ({"learning_rate": 0.01, "data_norm_alpha": 0.125, "data_norm_decay": "geometry",
                       "data_norm_decay_power": 2}, True),
    "M_lr0.01": ({"learning_rate": 0.01}, False),
}
CELLS = [(0.1, 0.5), (0.1, 0.9), (0.3, 0.5), (0.3, 0.9)]

if __name__ == "__main__":
    for method, (change, stats) in METHODS.items():
        for wd, cd in CELLS:
            overrides = {**BASE, "seed": 260925, "svd_device": "cuda", "batch_tokens": 524288,
                         "weight_decay": wd, "cooldown_fraction": cd, **change,
                         "model": {**BASE["model"], **(STATS if stats else {})}}
            name = f"{method}_wd{wd}_cd{cd}_t3recipe_s260925_a6000"
            (HERE / name).mkdir(exist_ok=False)
            (HERE / name / "config.json").write_text(json.dumps(load_config(overrides=overrides), indent=2) + "\n")
            print(name)

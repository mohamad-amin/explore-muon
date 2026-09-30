"""Local screen, round 2 (MUON_CASE.md, 01:47 CDT): Muon LR bracket top, and the alpha 1/8 variants.

Our setup with wd 0.2 + cooldown 0.7 + batch 524,288 (2938 steps), seed 260925 on A6000 (GPU SVD). Reference arms in
t3recipe_20260925: Muon@0.007 3.64436, PD@0.01 decoupled 3.64812, PD@0.01 + geometry decay p=2 3.63521.
Usage from the project root: .venv/bin/python logs/muon_spectra/t3screen2_20260926/make_arms.py
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
PDGEO = {"learning_rate": 0.01, "data_norm_alpha": 0.25, "data_norm_decay": "geometry", "data_norm_decay_power": 2}
ARMS = {
    "M_lr0.014": ({"learning_rate": 0.014}, False),
    "PDgeo2_a0.125_lr0.014": ({**PDGEO, "data_norm_alpha": 0.125, "learning_rate": 0.014}, True),
    "PDgeo1_a0.125": ({**PDGEO, "data_norm_alpha": 0.125, "data_norm_decay_power": 1}, True),
    "PD_a0.125": ({"learning_rate": 0.01, "data_norm_alpha": 0.125}, True),   # decoupled decay
}

if __name__ == "__main__":
    for arm, (change, stats) in ARMS.items():
        overrides = {**BASE, "seed": 260925, "svd_device": "cuda", **RECIPE, **change,
                     "model": {**BASE["model"], **(STATS if stats else {})}}
        name = f"{arm}_t3recipe_s260925_a6000"
        (HERE / name).mkdir(exist_ok=False)
        (HERE / name / "config.json").write_text(json.dumps(load_config(overrides=overrides), indent=2) + "\n")
        print(name)

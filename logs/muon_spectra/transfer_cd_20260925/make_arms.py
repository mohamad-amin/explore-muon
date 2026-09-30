"""Track 3 transfer tests: tuned Muon@0.007 vs PD@0.01 in our setup with one Track 3 feature at a time.

Seed 260925 on A6000 (GPU SVD), so each pair compares with the existing A6000 pair on that seed
(Muon@0.007 3.70630, PD@0.01 3.68909, delta -0.0172). Usage from the project root:
    .venv/bin/python logs/muon_spectra/transfer_20260925/make_arms.py FEATURE [FEATURE ...]
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
METHODS = {"M": {"learning_rate": 0.007}, "PD": {"learning_rate": 0.01, "data_norm_alpha": 0.25}}
FEATURES = {  # Track 3 values, or their per-step equivalent at our learning rates
    "wd0.2": {"weight_decay": 0.2},            # ~Track 3's per-step shrink (0.025 x 0.05) at LR 0.007
    "bt524288": {"batch_tokens": 524288},      # Track 3 batch; same total tokens (2938 steps)
    "cd0.7": {"cooldown_fraction": 0.7},       # Track 3 schedule: linear decay over the last 70%
    "bias": {"model": {"bias": True}},         # Track 3 Linear biases
}
SHORT = {"bias": "bias"}


def arm(method, feature):
    change = FEATURES[feature]
    model = {**BASE["model"], **change.get("model", {}), **(STATS if method == "PD" else {})}
    overrides = {**BASE, "seed": 260925, "svd_device": "cuda", **METHODS[method],
                 **{k: v for k, v in change.items() if k != "model"}, "model": model}
    return f"{method}_{feature}_s260925_a6000", load_config(overrides=overrides)


if __name__ == "__main__":
    for feature in sys.argv[1:]:
        for method in ("M", "PD"):
            name, config = arm(method, feature)
            (HERE / name).mkdir(exist_ok=False)
            (HERE / name / "config.json").write_text(json.dumps(config, indent=2) + "\n")
            print(name)

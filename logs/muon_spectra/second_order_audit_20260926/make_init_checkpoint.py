"""A step-0 kept checkpoint in the trainer's format, so newton_train.py can run a whole schedule from initialization
(2026-09-29 23:3x CDT: coordination from step 0 instead of a branch at @9).

The model is built exactly as distributed.py builds it (random/numpy/torch seeded with config["seed"], then GPT on the
CPU), so the weights equal the run's initialization. The optimizer state holds zero momentum buffers for the 48 body
matrices (MuonAdamW's group 0) and no AdamW state for the auxiliary parameters (group 1), as at step 0.

usage: make_init_checkpoint.py SOURCE_ARM_DIR:STEP OUT_ARM_DIR
  (the config is copied from the source kept checkpoint; OUT_ARM_DIR/scientific/kept/step000000.pt is written)
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from research.adamw_spectra.model import GPT, ModelConfig  # noqa: E402


def main():
    item, out = sys.argv[1], Path(sys.argv[2])
    arm, step = item.rsplit(":", 1)
    saved = torch.load(Path(arm) / "scientific" / "kept" / f"step{int(step):06d}.pt", map_location="cpu", weights_only=False)
    config = saved["config"]
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    model = GPT(ModelConfig(**config["model"]))
    named = list(model.named_parameters())
    body = [i for i, (name, p) in enumerate(named) if name.startswith("blocks.") and p.ndim == 2]
    aux = [i for i, (name, p) in enumerate(named) if not (name.startswith("blocks.") and p.ndim == 2)]
    order = body + aux          # MuonAdamW numbers its state body first, then auxiliary
    index = {orig: new for new, orig in enumerate(order)}
    state = {index[i]: {"momentum_buffer": torch.zeros_like(named[i][1]), "step": torch.tensor(0.0)} for i in body}
    groups = [{**{k: v for k, v in saved["optimizer"]["param_groups"][0].items() if k != "params"},
               "params": [index[i] for i in body]},
              {**{k: v for k, v in saved["optimizer"]["param_groups"][1].items() if k != "params"},
               "params": [index[i] for i in aux]}]
    kept = out / "scientific" / "kept"
    kept.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": {"state": state, "param_groups": groups}, "config": config,
                "step": 0, "init_of": item}, kept / "step000000.pt")
    print("wrote", kept / "step000000.pt", "body", len(body), "aux", len(aux))


if __name__ == "__main__":
    main()

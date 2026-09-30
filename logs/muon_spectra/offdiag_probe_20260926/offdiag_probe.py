"""How much of the data-norm preconditioner comes from the off-diagonal part of C? (read-only)

For each body layer at a final checkpoint: C = E[x x^T] (uncentered, position 0 excluded,
128 validation sequences), scaled to unit mean eigenvalue as in PD. Reports
- offdiag_share: fraction of ||C||_F^2 off the diagonal;
- root_diff: ||R_full - R_diag||_F / ||R_full||_F for R = (C + 1e-3 I)^(-1/4) vs its diagonal-only analogue;
- mean_pr / top_pr: participation ratio (effective number of coordinates) of the token-mean input
  and of C's top eigenvector (512 or 2048 = fully spread, 1 = one coordinate).

.venv/bin/python logs/muon_spectra/offdiag_probe_20260926/offdiag_probe.py LABEL=RUN_DIR
"""
import json
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.adamw_spectra.data import TokenStream, token_views
from research.adamw_spectra.model import GPT, ModelConfig
from research.adamw_spectra.spike_diagnostics import body_layers
from research.adamw_spectra.train import load_config

device = torch.device("cuda")
SEQ, SEQUENCES, MICRO, ALPHA, DAMPING = 512, 128, 16, 0.25, 1e-3


def pr(v):
    p = v.pow(2) / v.pow(2).sum()
    return float(1.0 / p.pow(2).sum())


def main():
    label, run = sys.argv[1].split("=", 1)
    saved = torch.load(REPO / run / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = load_config(overrides=saved["config"])
    model = GPT(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    val = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    x, y = token_views(val.device_tokens(0, SEQUENCES * SEQ + 1, device), 0, SEQUENCES * SEQ, SEQ)
    layers = body_layers(model)
    first = {n: 0 for n in layers}
    second = {n: 0 for n in layers}

    def accumulate(name):
        def hook(module, args):
            s = args[0][:, 1:].reshape(-1, args[0].shape[-1]).double()
            first[name] = first[name] + s.sum(0)
            second[name] = second[name] + s.T @ s
        return hook
    hooks = [layer.register_forward_pre_hook(accumulate(n)) for n, layer in layers.items()]
    with torch.no_grad():
        for i in range(0, SEQUENCES, MICRO):
            model(x[i:i + MICRO], y[i:i + MICRO])
    for h in hooks:
        h.remove()
    count = SEQUENCES * (SEQ - 1)
    out = {}
    for n in layers:
        c = second[n] / count
        mean = first[n] / count
        eig, vec = torch.linalg.eigh(c)
        eig = eig.clamp_min(0)
        scale = eig.mean()
        cn, eig_n = c / scale, eig / scale
        r_full = (vec * (eig_n + DAMPING).pow(-ALPHA)) @ vec.T
        r_diag = torch.diag((torch.diagonal(cn) + DAMPING).pow(-ALPHA))
        out[n] = {"offdiag_share": float(1 - torch.diagonal(cn).pow(2).sum() / cn.pow(2).sum()),
                  "root_diff": float((r_full - r_diag).norm() / r_full.norm()),
                  "mean_pr": pr(mean), "top_pr": pr(vec[:, -1]), "dim": int(c.shape[0])}
    path = Path(__file__).with_name("offdiag.json")
    path.write_text(json.dumps({label: out}, indent=1) + "\n")
    kinds = {}
    for n, r in out.items():
        kinds.setdefault(n.split(".")[1], []).append(r)
    for k, rows in kinds.items():
        avg = {key: sum(r[key] for r in rows) / len(rows) for key in ("offdiag_share", "root_diff", "mean_pr", "top_pr")}
        print(k, f"dim {rows[0]['dim']}", {key: round(v, 3) for key, v in avg.items()}, flush=True)


if __name__ == "__main__":
    main()

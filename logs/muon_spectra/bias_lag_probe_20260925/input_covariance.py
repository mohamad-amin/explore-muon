"""How anisotropic are body-layer inputs beyond the mean? (read-only)

Decides whether full data-norm (input-whitened) Muon, polar(G A^-1/2) A^-1/2 with
A = E[x x^T], would differ from mean-only whitening. For each body matrix at a final
checkpoint: the mean share of E||x||^2, and the eigenvalue profile of the centered
covariance C = E[(x - xbar)(x - xbar)^T] (top / median ratio, top-1 and top-8 shares,
effective rank exp(entropy)). Position 0 excluded. One GPU, FP64 accumulation.

.venv/bin/python logs/muon_spectra/bias_lag_probe_20260925/input_covariance.py LABEL=RUN_DIR [...]
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
SEQ, SEQUENCES, MICRO = 512, 128, 16


def profile(run, x, y):
    saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = load_config(overrides=saved["config"])
    model = GPT(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    layers = body_layers(model)
    first = {n: torch.zeros(l.weight.shape[1], device=device, dtype=torch.float64) for n, l in layers.items()}
    second = {n: torch.zeros(l.weight.shape[1], l.weight.shape[1], device=device, dtype=torch.float64)
              for n, l in layers.items()}

    def accumulate(name):
        def hook(module, args):
            sample = args[0][:, 1:].reshape(-1, args[0].shape[-1]).double()
            first[name].add_(sample.sum(0))
            second[name].add_(sample.T @ sample)
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
        mean = first[n] / count
        raw = second[n] / count
        centered = raw - torch.outer(mean, mean)
        eig = torch.linalg.eigvalsh(centered).clamp_min(0).flip(0)
        p = eig / eig.sum()
        out[n] = {"input_rms": float(raw.trace().sqrt() / mean.numel() ** 0.5),
                  "mean_share": float(mean.pow(2).sum() / raw.trace()),
                  "top_over_median": float(eig[0] / eig[eig.numel() // 2]),
                  "p90_over_median": float(eig[eig.numel() // 10] / eig[eig.numel() // 2]),
                  "median_over_p10": float(eig[eig.numel() // 2] / eig[-(eig.numel() // 10)].clamp_min(1e-30)),
                  "top1_share": float(p[0]), "top8_share": float(p[:8].sum()),
                  "effective_rank": float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum())),
                  "dim": mean.numel()}
    return out


def main():
    val = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    x, y = token_views(val.device_tokens(0, SEQUENCES * SEQ + 1, device), 0, SEQUENCES * SEQ, SEQ)
    path = Path(__file__).with_name("input_covariance.json")
    results = json.loads(path.read_text()) if path.exists() else {}
    for arg in sys.argv[1:]:
        label, run = arg.split("=", 1)
        results[label] = profile(REPO / run, x, y)
        path.write_text(json.dumps(results, indent=1) + "\n")
        for n, r in results[label].items():
            if n.startswith(("block02", "block05", "block08")):
                print(label, n, {k: round(v, 3) for k, v in r.items()}, flush=True)


if __name__ == "__main__":
    main()

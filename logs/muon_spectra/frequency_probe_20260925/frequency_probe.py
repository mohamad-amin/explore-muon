"""Where does the data-norm gain land: frequent or rare target tokens? (read-only)

Hypothesis: partial input whitening enlarges steps along low-variance input directions, so, like
Adam on heavy-tailed data, it should help rarer features most. Evaluate per-token validation NLL
for paired final checkpoints (same seed and hardware) and bin by the target token's unigram
frequency (estimated from 50M fresh training tokens). FP32, 512 validation sequences.

.venv/bin/python logs/muon_spectra/frequency_probe_20260925/frequency_probe.py BASE_LABEL=RUN ARM_LABEL=RUN [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.adamw_spectra.data import TokenStream, token_views
from research.adamw_spectra.model import GPT, ModelConfig
from research.adamw_spectra.train import load_config

torch.backends.cuda.matmul.allow_tf32 = False
device = torch.device("cuda")
SEQ, SEQUENCES, MICRO = 512, 512, 16


def per_token_nll(run, x, y):
    saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = load_config(overrides=saved["config"])
    model = GPT(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, x.shape[0], MICRO):
            xb, yb = x[i:i + MICRO], y[i:i + MICRO]
            logits = model(xb)                       # no targets: the model returns logits
            out.append(F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), yb.reshape(-1),
                                       reduction="none").double().cpu())
    return torch.cat(out).numpy()


def main():
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_train_*.bin"))
    counts = np.bincount(stream.read(2_500_000_000, 50_000_000), minlength=50304).astype(np.float64)
    freq = counts / counts.sum()
    val = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    x, y = token_views(val.device_tokens(0, SEQUENCES * SEQ + 1, device), 0, SEQUENCES * SEQ, SEQ)
    targets = y.reshape(-1).cpu().numpy()
    labels, nll = [], {}
    for arg in sys.argv[1:]:
        label, run = arg.split("=", 1)
        labels.append(label)
        nll[label] = per_token_nll(REPO / run, x, y)
        print(label, "mean NLL", round(float(nll[label].mean()), 5), flush=True)
    base = labels[0]
    edges = [0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1.0]
    bins = np.digitize(freq[targets], edges) - 1
    report = {"labels": labels, "bins": []}
    for b in range(len(edges) - 1):
        mask = bins == b
        if mask.sum() == 0:
            continue
        row = {"freq_range": [edges[b], edges[b + 1]], "tokens": int(mask.sum()), "share": float(mask.mean()),
               "base_nll": float(nll[base][mask].mean())}
        for label in labels[1:]:
            d = nll[label][mask] - nll[base][mask]
            row[f"delta_{label}"] = float(d.mean())
            row[f"contribution_{label}"] = float(d.sum() / len(targets))
        report["bins"].append(row)
        print({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
    Path(__file__).with_name("frequency.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()

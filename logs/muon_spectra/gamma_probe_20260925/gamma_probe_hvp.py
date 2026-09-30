"""Curvature exponent γ from exact Hessian-vector products (read-only; replaces finite differences).

The finite-difference probe (`gamma_probe.py`) quantized its second differences at the float32
spacing of the loss (≈2.4e-7), so low-variance input directions were unresolved and the fitted
γ ≈ 0.5 was an artifact (see README). Here curvature along D = u q_jᵀ is Dᵀ H D computed by double
backpropagation (Pearlmutter-style HVP) in FP32 with no cancellation: q_j are eigenvectors of the
layer's input second moment C at ranks spread over the spectrum, u a random unit output vector
(3 draws, averaged). Same 64 fresh sequences as the finite-difference probe.

.venv/bin/python logs/muon_spectra/gamma_probe_20260925/gamma_probe_hvp.py LABEL=RUN_DIR [...]
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.adamw_spectra.data import TokenStream, token_views
from research.adamw_spectra.model import GPT, ModelConfig
from research.adamw_spectra.spike_diagnostics import body_layers
from research.adamw_spectra.train import load_config

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# Double backward needs the math attention kernel (the fused kernels lack second derivatives).
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
device = torch.device("cuda")
MATRICES = tuple(f"block{b:02d}.{k}" for b in (2, 4, 6, 8) for k in ("q", "k", "v", "o", "up", "down"))
SEQUENCES, MICRO, DRAWS = 64, 8, 3


def probe(run):
    saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = load_config(overrides=saved["config"])
    model = GPT(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    stream = TokenStream(str(REPO / config["train_pattern"]))
    tokens = stream.device_tokens(2_300_000_000, SEQUENCES * 512 + 1, device)
    x, y = token_views(tokens, 0, SEQUENCES * 512, 512)
    layers = body_layers(model)
    generator = torch.Generator(device=device).manual_seed(0)
    out = {"matrices": {}}
    for name in MATRICES:
        layer = layers[name]
        W = layer.weight
        captured = []
        hook = layer.register_forward_pre_hook(lambda m, a: captured.append(a[0][:, 1:].detach().double()) or None)
        with torch.no_grad():
            for i in range(0, SEQUENCES, MICRO):
                model(x[i:i + MICRO], y[i:i + MICRO])
        hook.remove()
        sample = torch.cat([c.reshape(-1, W.shape[1]) for c in captured])
        cov = sample.T @ sample / sample.shape[0]
        eigenvalues, vectors = torch.linalg.eigh(cov)
        eigenvalues, vectors = eigenvalues.flip(0), vectors.flip(1).float()
        d = W.shape[1]
        ranks = sorted({0, 1, 3, 7, 15, 31, 63, 127, d // 4, d // 2, 3 * d // 4, d - 1})
        directions = []
        for rank in ranks:
            for _ in range(DRAWS):
                u = torch.randn(W.shape[0], device=device, generator=generator)
                directions.append(torch.outer(u / u.norm(), vectors[:, rank]))
        curv = torch.zeros(len(directions), dtype=torch.float64, device=device)
        W.requires_grad_(True)
        for i in range(0, SEQUENCES, MICRO):
            loss = model(x[i:i + MICRO], y[i:i + MICRO]) * (MICRO / SEQUENCES)
            (grad,) = torch.autograd.grad(loss, W, create_graph=True)
            for j, D in enumerate(directions):
                (hvp,) = torch.autograd.grad(grad, W, grad_outputs=D, retain_graph=j < len(directions) - 1)
                curv[j] += (hvp * D).sum().double()
            del grad, loss
        W.requires_grad_(False)
        per_rank = curv.reshape(len(ranks), DRAWS).mean(dim=1).cpu().numpy()
        rows = [{"rank": r, "eigenvalue": float(eigenvalues[r]), "curvature": float(c)} for r, c in zip(ranks, per_rank)]
        good = [r for r in rows if r["curvature"] > 0 and r["eigenvalue"] > 0]
        slope = float(np.polyfit(np.log([r["eigenvalue"] for r in good]), np.log([r["curvature"] for r in good]), 1)[0]) \
            if len(good) >= 4 else None
        top = [r for r in good if r["rank"] <= 127]
        slope_top = float(np.polyfit(np.log([r["eigenvalue"] for r in top]), np.log([r["curvature"] for r in top]), 1)[0]) \
            if len(top) >= 4 else None
        out["matrices"][name] = {"rows": rows, "gamma": slope, "gamma_top128": slope_top, "positive": len(good)}
        print(name, "gamma", None if slope is None else round(slope, 3), "gamma(top128)",
              None if slope_top is None else round(slope_top, 3), f"({len(good)}/{len(rows)} positive)", flush=True)
    gammas = [m["gamma"] for m in out["matrices"].values() if m["gamma"] is not None]
    out["median_gamma"] = float(np.median(gammas))
    kinds = {}
    for name, m in out["matrices"].items():
        if m["gamma"] is not None:
            kinds.setdefault(name.split(".")[1], []).append(m["gamma"])
    out["median_gamma_by_kind"] = {k: float(np.median(v)) for k, v in kinds.items()}
    print("median gamma", round(out["median_gamma"], 3), "by kind", {k: round(v, 3) for k, v in out["median_gamma_by_kind"].items()})
    return out


def main():
    path = Path(__file__).with_name("gamma_hvp.json")
    results = json.loads(path.read_text()) if path.exists() else {}
    for arg in sys.argv[1:]:
        label, run = arg.split("=", 1)
        results[label] = probe(REPO / run)
        path.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()

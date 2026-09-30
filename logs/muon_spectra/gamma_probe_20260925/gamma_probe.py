"""How does loss curvature along an input direction scale with that direction's second moment? (read-only)

For a body matrix W (y = W x) with input second moment C = E[x x^T] (uncentered, position 0
excluded), take eigenvectors q_j of C at ranks spread over the spectrum and measure the
finite-difference loss curvature along the rank-1 weight direction u q_j^T (u a fixed random
unit output vector, same for all ranks). Regress log curvature on log eigenvalue: the slope gamma
is the exponent in curvature ~ (q^T C q)^gamma. A curvature-matched data norm uses alpha = gamma/2
(MUON_CASE.md, waves 5-10). FP32 forward on 64 fresh sequences; central differences with step
0.02 ||W||_F / sqrt(d_in); two random output directions per matrix, averaged.

.venv/bin/python logs/muon_spectra/gamma_probe_20260925/gamma_probe.py LABEL=RUN_DIR [...]
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
device = torch.device("cuda")
MATRICES = tuple(f"block{b:02d}.{k}" for b in (2, 4, 6, 8) for k in ("q", "k", "v", "o", "up", "down")) if "--all-kinds" in sys.argv else ("block02.q", "block02.up", "block04.v", "block04.o", "block06.up", "block06.down", "block08.q", "block08.o")


def probe(run):
    saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = load_config(overrides=saved["config"])
    model = GPT(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    stream = TokenStream(str(REPO / config["train_pattern"]))
    tokens = stream.device_tokens(2_300_000_000, 64 * 512 + 1, device)
    x, y = token_views(tokens, 0, 64 * 512, 512)
    layers = body_layers(model)
    inputs = {}
    hooks = [layer.register_forward_pre_hook(lambda m, a, n=name: inputs.__setitem__(n, a[0].detach()) or None)
             for name, layer in layers.items() if name in MATRICES]

    @torch.no_grad()
    def loss():
        return float(model(x, y).double())

    base = loss()
    for h in hooks:
        h.remove()
    generator = torch.Generator(device=device).manual_seed(0)
    out = {"base_loss": base, "matrices": {}}
    for name in MATRICES:
        W = layers[name].weight
        sample = inputs[name][:, 1:].reshape(-1, W.shape[1]).double()
        cov = sample.T @ sample / sample.shape[0]
        eigenvalues, vectors = torch.linalg.eigh(cov)
        eigenvalues, vectors = eigenvalues.flip(0), vectors.flip(1)
        d = W.shape[1]
        ranks = sorted({0, 1, 3, 7, 15, 31, 63, 127, d // 4, d // 2, 3 * d // 4, d - 1})
        eps = 0.02 * float(W.data.norm()) / d ** 0.5
        original = W.data.clone()
        rows = []
        for rank in ranks:
            q = vectors[:, rank].float()
            curvatures = []
            for _ in range(2):
                u = torch.randn(W.shape[0], device=device, generator=generator)
                u /= u.norm()
                direction = torch.outer(u, q)
                values = []
                for sign in (1.0, -1.0):
                    W.data.copy_(original + sign * eps * direction)
                    values.append(loss())
                W.data.copy_(original)
                curvatures.append((values[0] + values[1] - 2 * base) / eps ** 2)
            rows.append({"rank": rank, "eigenvalue": float(eigenvalues[rank]), "curvature": float(np.mean(curvatures))})
        good = [r for r in rows if r["curvature"] > 0 and r["eigenvalue"] > 0]
        slope = float(np.polyfit(np.log([r["eigenvalue"] for r in good]), np.log([r["curvature"] for r in good]), 1)[0]) \
            if len(good) >= 4 else None
        out["matrices"][name] = {"rows": rows, "gamma": slope, "positive": len(good)}
        print(name, "gamma", None if slope is None else round(slope, 3), f"({len(good)}/{len(rows)} positive)", flush=True)
    gammas = [m["gamma"] for m in out["matrices"].values() if m["gamma"] is not None]
    out["median_gamma"] = float(np.median(gammas)) if gammas else None
    print("median gamma", out["median_gamma"], "-> alpha = gamma / 2 =", None if not gammas else round(out["median_gamma"] / 2, 3))
    kinds = {}
    for name, m in out["matrices"].items():
        if m["gamma"] is not None:
            kinds.setdefault(name.split(".")[1], []).append(m["gamma"])
    out["median_gamma_by_kind"] = {k: float(np.median(v)) for k, v in kinds.items()}
    print("median gamma by kind", {k: round(v, 3) for k, v in out["median_gamma_by_kind"].items()})
    return out


def main():
    path = Path(__file__).with_name("gamma.json")
    results = json.loads(path.read_text()) if path.exists() else {}
    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            continue
        label, run = arg.split("=", 1)
        results[label] = probe(REPO / run)
        path.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()

"""Curvature exponent and input anisotropy of a trained Track 3 model (read-only; one GPU).

For hidden matrix W (y = W x + b) with input second moment C = E[x x^T] (position 0 excluded), take
eigenvectors q_j at ranks across the spectrum and measure the loss curvature D^T H D along D = u q_j^T
(u a random unit output vector, three per rank) with exact Hessian-vector products (double backprop,
FP32, math attention kernel). The slope gamma of log curvature against log eigenvalue is the exponent in
curvature ~ (q^T C q)^gamma. In our model the exact probe gives gamma ~1 at widths 512 and 768
(logs/muon_spectra/gamma_probe_20260925/gamma_probe_hvp.py), and PD's alpha = 1/4 is the square root of
the curvature-matched 1/2. Also reports input anisotropy: top/median eigenvalue, mean-direction share of
E||x||^2, and effective rank.

This probe first used central finite differences. Before it produced any result (one dry run only),
they were replaced by HVPs, because FP32 second differences of the loss are quantized at ~2.4e-7 per
token (the flaw found in our own finite-difference probe).

Usage: python track3/gamma_probe_t3.py LABEL=track3/logs/<uuid>_final.pt [...]  (writes track3/gamma_t3.json)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

HERE = Path(__file__).resolve().parent
source = (HERE / "baseline/train_gpt_simple.py").read_text()
start, end = source.index("class RMSNorm"), source.index("########################################\n#              Optimizer")
exec(source[start:end])  # the exact #36 architecture: RMSNorm, Linear, Rotary, CausalSelfAttention, MLP, Block, GPT

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# Double backward needs the math attention kernel (the fused kernels lack second derivatives).
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
device = torch.device("cuda")
BLOCKS = (1, 4, 7, 10)
KINDS = {"q": "attn.q", "v": "attn.v", "o": "attn.proj", "up": "mlp.fc", "down": "mlp.proj"}
MICRO, DRAWS = 4, 3


def tokens():
    """32 fresh sequences of 1024 (32K tokens, as in our probe) from the validation shard, past the 10.5M tokens Track 3 validates on."""
    path = HERE / "data/fineweb10B/fineweb_val_000000.bin"
    offset = 50_000_000
    raw = np.fromfile(path, dtype=np.uint16, count=32 * 1024 + 1, offset=1024 + 2 * offset)
    t = torch.from_numpy(raw.astype(np.int64)).to(device)
    return t[:-1].view(32, 1024), t[1:].view(32, 1024)


def probe(checkpoint):
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = GPT(vocab_size=50304, num_layers=12, model_dim=768).to(device)
    model.embed = model.embed.float()
    model.load_state_dict({k: v.float() for k, v in saved["model"].items()})
    model.eval()
    x, y = tokens()
    n_tokens = y.numel()
    for p in model.parameters():
        p.requires_grad_(False)

    modules = {f"block{b:02d}.{k}": model.blocks[b].get_submodule(path) for b in BLOCKS for k, path in KINDS.items()}
    inputs = {name: [] for name in modules}
    hooks = [m.register_forward_pre_hook(lambda mod, args, name=name: inputs[name].append(args[0].detach()))
             for name, m in modules.items()]
    with torch.no_grad():
        base = sum(float(model(x[i:i + MICRO], y[i:i + MICRO]).double()) for i in range(0, len(x), MICRO)) / n_tokens
    for h in hooks:
        h.remove()
    generator = torch.Generator(device=device).manual_seed(0)
    out = {"base_loss": base, "matrices": {}}
    for name, module in modules.items():
        W = module.weight
        sample = torch.cat([c[:, 1:].reshape(-1, W.shape[1]) for c in inputs[name]]).double()
        inputs[name] = None
        cov = sample.T @ sample / sample.shape[0]
        eigenvalues, vectors = torch.linalg.eigh(cov)
        eigenvalues, vectors = eigenvalues.flip(0), vectors.flip(1)
        lam = eigenvalues.clamp_min(0)
        mean = sample.mean(0)
        stats = {"top_over_median": float(lam[0] / lam[len(lam) // 2]),
                 "mean_share": float(mean.square().sum() / lam.sum()),
                 "effective_rank": float(lam.sum() ** 2 / lam.square().sum())}
        d = W.shape[1]
        ranks = sorted({0, 1, 3, 7, 15, 31, 63, 127, d // 4, d // 2, 3 * d // 4, d - 1})
        directions = []
        for rank in ranks:
            for _ in range(DRAWS):
                u = torch.randn(W.shape[0], device=device, generator=generator)
                directions.append(torch.outer(u / u.norm(), vectors[:, rank].float()))
        curv = torch.zeros(len(directions), dtype=torch.float64, device=device)
        W.requires_grad_(True)
        for i in range(0, len(x), MICRO):   # loss = mean over all tokens, accumulated per micro-batch
            loss = model(x[i:i + MICRO], y[i:i + MICRO]) / n_tokens
            (grad,) = torch.autograd.grad(loss, W, create_graph=True)
            for j, D in enumerate(directions):
                (hvp,) = torch.autograd.grad(grad, W, grad_outputs=D, retain_graph=j < len(directions) - 1)
                curv[j] += (hvp * D).sum().double()
            del grad, loss
        W.requires_grad_(False)
        per_rank = curv.reshape(len(ranks), DRAWS).mean(dim=1).cpu().numpy()
        rows = [{"rank": r, "eigenvalue": float(eigenvalues[r]), "curvature": float(c)} for r, c in zip(ranks, per_rank)]
        good = [r for r in rows if r["curvature"] > 0 and r["eigenvalue"] > 0]
        gamma = float(np.polyfit(np.log([r["eigenvalue"] for r in good]), np.log([r["curvature"] for r in good]), 1)[0]) \
            if len(good) >= 4 else None
        out["matrices"][name] = {"rows": rows, "gamma": gamma, "positive": len(good), **stats}
        print(f"{name:14s} gamma {gamma if gamma is None else round(gamma, 3)} ({len(good)}/{len(rows)} positive)  "
              f"top/median {stats['top_over_median']:.0f}  mean share {stats['mean_share']:.2f}  "
              f"eff. rank {stats['effective_rank']:.0f}/{d}", flush=True)
    kinds = {}
    for name, m in out["matrices"].items():
        if m["gamma"] is not None:
            kinds.setdefault(name.split(".")[1], []).append(m["gamma"])
    gammas = [g for v in kinds.values() for g in v]
    out["median_gamma"] = float(np.median(gammas)) if gammas else None
    out["median_gamma_by_kind"] = {k: float(np.median(v)) for k, v in kinds.items()}
    print("median gamma", out["median_gamma"], "by kind", {k: round(v, 3) for k, v in out["median_gamma_by_kind"].items()})
    return out


def main():
    path = HERE / "gamma_t3.json"
    results = json.loads(path.read_text()) if path.exists() else {}
    for arg in sys.argv[1:]:
        label, checkpoint = arg.split("=", 1)
        results[label] = probe(Path(checkpoint))
        path.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()

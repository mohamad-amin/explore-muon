"""Finite-difference loss curvature along rank-1 weight directions (read-only).

Tests the premise of mean-whitened Muon: for y = W x, the curvature along u z^T scales
with E[(z.x)^2], so the mean-input direction z = xbar/||xbar|| should be far stiffer
than a typical input direction. Also compares the momentum's top pair with bulk pairs.
FP32 forward on 64 fresh sequences at the frontier-norm Muon run's final weights.
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
from research.adamw_spectra.train import load_config, make_optimizer

torch.backends.cuda.matmul.allow_tf32 = False
device = torch.device("cuda")
run = REPO / "logs/muon_spectra/depth8_w512_nobias_rms_qk_20260925/muon/scientific"
saved = torch.load(run / "checkpoint.pt", map_location="cpu", weights_only=False)
config = load_config(overrides=saved["config"])
model = GPT(ModelConfig(**config["model"])).to(device)
model.load_state_dict(saved["model"])
optimizer, _ = make_optimizer(model, config, device)
optimizer.load_state_dict(saved["optimizer"])
stream = TokenStream(str(REPO / config["train_pattern"]))
tokens = stream.device_tokens(2_200_000_000, 64 * 512 + 1, device)
x, y = token_views(tokens, 0, 64 * 512, 512)
layers = body_layers(model)
inputs = {}
hooks = [layer.register_forward_pre_hook(lambda m, a, n=name: inputs.__setitem__(n, a[0].detach()))
         for name, layer in layers.items()]


@torch.no_grad()
def loss():
    return float(model(x, y).double())


base = loss()
for h in hooks:
    h.remove()
generator = torch.Generator(device=device).manual_seed(0)
out = {"base_loss": base, "matrices": {}}
for name in ["block04.q", "block04.v", "block04.up", "block04.o", "block08.q", "block08.o", "block08.down", "block04.down"]:
    layer = layers[name]
    W = layer.weight
    sample = inputs[name][:, 1:].reshape(-1, W.shape[1]).float()
    xbar = sample.mean(0)
    zhat = xbar / xbar.norm()
    second = lambda z: float(((sample @ z) ** 2).mean())
    u_out = torch.randn(W.shape[0], device=device, generator=generator)
    u_out /= u_out.norm()

    def curvature(direction, eps):
        original = W.data.clone()
        values = []
        for sign in (1.0, -1.0):
            W.data.copy_(original + sign * eps * direction)
            values.append(loss())
        W.data.copy_(original)
        return (values[0] + values[1] - 2 * base) / eps ** 2

    eps = 0.02 * float(W.data.norm()) / W.shape[1] ** 0.5
    rows = {"eps": eps, "mean_share_of_second_moment": float(xbar.pow(2).sum() / sample.pow(2).sum(-1).mean())}
    rows["mean_input_curvature"] = curvature(torch.outer(u_out, zhat), eps)
    rows["mean_input_second_moment"] = second(zhat)
    randoms = []
    for _ in range(4):
        z = torch.randn(W.shape[1], device=device, generator=generator)
        z -= (z @ zhat) * zhat
        z /= z.norm()
        randoms.append((curvature(torch.outer(u_out, z), eps), second(z)))
    rows["random_input_curvature"] = [c for c, _ in randoms]
    rows["random_input_second_moment"] = [s for _, s in randoms]
    rows["curvature_ratio_mean_vs_random"] = rows["mean_input_curvature"] / (sum(rows["random_input_curvature"]) / 4)
    rows["second_moment_ratio"] = rows["mean_input_second_moment"] / (sum(rows["random_input_second_moment"]) / 4)
    momentum = optimizer.state[W]["momentum_buffer"].float()
    u, s, vh = torch.linalg.svd(momentum, full_matrices=False)
    pairs = {"top": 0, "rank8": 7, "median": s.numel() // 2}
    rows["momentum_pair_curvature"] = {k: curvature(torch.outer(u[:, i], vh[i]), eps) for k, i in pairs.items()}
    rows["top_pair_v_cos_mean_input"] = abs(float(vh[0] @ zhat))
    out["matrices"][name] = rows
    print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows.items()
                 if k in ("mean_share_of_second_moment", "curvature_ratio_mean_vs_random", "second_moment_ratio",
                          "top_pair_v_cos_mean_input")},
          {k: round(v, 3) for k, v in rows["momentum_pair_curvature"].items()}, flush=True)
(Path(__file__).with_name("curvature.json")).write_text(json.dumps(out, indent=1) + "\n")

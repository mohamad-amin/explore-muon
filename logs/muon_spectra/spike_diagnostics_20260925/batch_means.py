"""Batch-to-batch stability of each layer's mean input x̄ and mean output-gradient ē (read-only).

Follow-up question: batches are never reused, so why is the spike's input vector the same?
At fixed final weights, records x̄ and ē separately for fresh 1,048,576-token batches (the
training batch size) and how much of an average token each mean accounts for.
Run from the project root: .venv/bin/python <this file> RUN_DIR OUT_JSON
"""

import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from research.adamw_spectra.data import TokenStream  # noqa: E402
from research.adamw_spectra.model import GPT, ModelConfig  # noqa: E402
from research.adamw_spectra.spike_diagnostics import batch_gradient, body_layers  # noqa: E402
from research.adamw_spectra.train import load_config, make_optimizer  # noqa: E402

OFFSET, BATCHES, BATCH_TOKENS, MICRO_SEQUENCES = 2_100_000_000, 16, 1_048_576, 32


def cos(a, b):
    return abs(float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def main(run_dir, out):
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    saved = torch.load(Path(run_dir) / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = load_config(overrides=saved["config"])
    model = GPT(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(saved["model"])
    optimizer, _ = make_optimizer(model, config, device)
    optimizer.load_state_dict(saved["optimizer"])
    key = "momentum_buffer" if config["optimizer"] == "muon" else "exp_avg"
    layers = body_layers(model)
    pairs = {}
    for name, layer in layers.items():
        u, _, vh = torch.linalg.svd(optimizer.state[layer.weight][key].double(), full_matrices=False)
        pairs[name] = (u[:, 0].cpu().numpy(), vh[0].cpu().numpy())
    inputs, acc = {}, {}

    def reset():
        for name, layer in layers.items():
            z = lambda n: torch.zeros(n, dtype=torch.float64, device=device)
            acc[name] = {"sx": z(layer.weight.shape[1]), "sd": z(layer.weight.shape[0]),
                         "xx": z(()), "dd": z(()), "n": 0}

    def pre(name, module, args):
        if torch.is_grad_enabled():
            inputs[name] = args[0].detach()

    def post(name, module, args, output):
        if output.requires_grad:
            output.register_hook(partial(grad, name))

    def grad(name, g):
        x = inputs.pop(name).to(torch.bfloat16).float()
        x, d = x.reshape(-1, x.shape[-1]), g.float().reshape(-1, g.shape[-1])
        a = acc[name]
        a["sx"] += x.sum(0).double()
        a["sd"] += d.sum(0).double()
        a["xx"] += (x * x).sum().double()
        a["dd"] += (d * d).sum().double()
        a["n"] += x.shape[0]

    handles = []
    for name, layer in layers.items():
        handles.append(layer.register_forward_pre_hook(partial(pre, name)))
        handles.append(layer.register_forward_hook(partial(post, name)))
    stream = TokenStream(config["train_pattern"])
    if OFFSET < saved["tokens"] + 1:
        raise ValueError("Fresh tokens must start after every trained token")
    per_batch = {name: [] for name in layers}
    for j in range(BATCHES):
        reset()
        tokens = stream.device_tokens(OFFSET + j * BATCH_TOKENS, BATCH_TOKENS + 1, device)
        batch_gradient(model, tokens, model.config.seq_len, MICRO_SEQUENCES, config, device)
        for name, a in acc.items():
            n = a["n"]
            xbar, ebar = (a["sx"] / n).cpu().numpy(), (a["sd"] / n).cpu().numpy()
            per_batch[name].append({
                "xbar": xbar, "ebar": ebar,
                "x_shared": float(xbar @ xbar) / float(a["xx"] / n),   # |x̄|² / mean |x_t|²
                "e_shared": float(ebar @ ebar) / float(a["dd"] / n)})  # |ē|² / mean |e_t|²
        print(json.dumps({"batch": j + 1}), flush=True)
    for handle in handles:
        handle.remove()
    result = {"run": str(run_dir), "offset": OFFSET, "batches": BATCHES, "batch_tokens": BATCH_TOKENS,
              "matrices": {}}
    for name, rows in per_batch.items():
        xs = [r["xbar"] for r in rows]
        es = [r["ebar"] for r in rows]
        xm, em = np.mean(xs, 0), np.mean(es, 0)
        u1, v1 = pairs[name]
        result["matrices"][name] = {
            "xbar_consecutive_cos": float(np.median([cos(a, b) for a, b in zip(xs, xs[1:])])),
            "ebar_consecutive_cos": float(np.median([cos(a, b) for a, b in zip(es, es[1:])])),
            "xbar_vs_16batch_mean_cos": float(np.median([cos(a, xm) for a in xs])),
            "ebar_vs_16batch_mean_cos": float(np.median([cos(a, em) for a in es])),
            "x_shared_fraction": float(np.median([r["x_shared"] for r in rows])),
            "e_shared_fraction": float(np.median([r["e_shared"] for r in rows])),
            "xbar_vs_state_v1_cos": float(np.median([cos(a, v1) for a in xs])),
            "ebar_vs_state_u1_cos": float(np.median([cos(a, u1) for a in es]))}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

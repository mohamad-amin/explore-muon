"""Spike-origin and cross-fitted gradient diagnostics at a saved final checkpoint.

Loads final weights and optimizer state, computes fresh gradients on
never-trained tokens and attributes the state's leading singular pair.
Parameters are never updated. See MUON_CASE.md (spike-origin diagnostic).
Run from the project root: python -m research.adamw_spectra.spike_diagnostics --help
"""

import argparse
import json
import math
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch

from .data import TokenStream, token_views
from .model import GPT, ModelConfig
from .muon import NS_COEFFICIENTS
from .train import amp_context, atomic_json, load_config, make_optimizer, source_hashes


KINDS = ("q", "k", "v", "o", "up", "down")
SCALING_BATCHES = (1, 2, 4, 8, 16, 32, 64)
GATES = {"validation_nll": 0.01, "reconstruction": 5e-3, "hook_vs_gradient": 1e-2}


def body_layers(model):
    layers = {}
    for index, block in enumerate(model.blocks, start=1):
        for kind, layer in zip(KINDS, (block.attn.q, block.attn.k, block.attn.v,
                                       block.attn.o, block.mlp.up, block.mlp.down)):
            layers[f"block{index:02d}.{kind}"] = layer
    return layers


def ns_scalar(x):
    """The paper's five-polynomial NS map on Frobenius-normalized singular values."""
    x = np.asarray(x, dtype=np.float64)
    for a, b, c in NS_COEFFICIENTS:
        x = a * x + b * x**3 + c * x**5
    return x


def state_convention(config):
    """Reference state, its gradient-unit factor and its noise factor (EMA of iid noise)."""
    if config["optimizer"] == "muon":
        mu = config["muon_momentum"]
        return "momentum_buffer", 1 - mu, math.sqrt((1 - mu) / (1 + mu))
    beta = config["betas"][0]
    return "exp_avg", 1.0, math.sqrt((1 - beta) / (1 + beta))


def bands(rank):
    """Mode bands (one-based, inclusive) with the paper's quantile convention."""
    cut = [1, 5, math.ceil(0.1 * rank), math.ceil(0.5 * rank), math.ceil(0.9 * rank), rank]
    cut = [min(rank, max(cut[:i + 1])) for i in range(len(cut))]
    names = ["mode1", "modes2-5", "to_q0.1", "q0.1-q0.5", "q0.5-q0.9", "below_q0.9"]
    edges = [(1, 1)] + [(cut[i] + 1, cut[i + 1]) for i in range(len(cut) - 1)]
    return [(name, lo, hi) for name, (lo, hi) in zip(names, edges) if lo <= hi]


class Recorder:
    """Per-token attribution of each matrix's reference pair (u1, v1).

    For W with inputs x_b and output gradients d_b, the weight gradient is
    sum_b d_b x_b^T, so u1^T G v1 = sum_b (u1.d_b)(x_b.v1) splits over tokens.
    """

    def __init__(self, layers, pairs, seq_len, bf16_inputs):
        self.layers, self.pairs, self.seq_len = layers, pairs, seq_len
        self.bf16_inputs = bf16_inputs
        self.inputs, self.handles = {}, []
        self.reconstruct = False
        self.stats = {}
        for name, layer in layers.items():
            out_dim, in_dim = layer.weight.shape
            device = layer.weight.device
            z = partial(torch.zeros, dtype=torch.float64, device=device)
            self.stats[name] = {
                "tokens": 0, "contribution": z(()), "absolute": z(()), "top1": z(()),
                "position": z(seq_len), "sum_grad": z(out_dim), "sum_input": z(in_dim),
                "sum_grad_pos0": z(out_dim), "sum_input_pos0": z(in_dim),
                "pos0_term": z(out_dim, in_dim), "input_abs": z(in_dim),
                "batch_grad": z(out_dim), "batch_input": z(in_dim), "batch_tokens": 0,
                "batchwise_mean_product": z(out_dim, in_dim), "reconstruction": None}

    def attach(self):
        for name, layer in self.layers.items():
            self.handles.append(layer.register_forward_pre_hook(partial(self._input, name)))
            self.handles.append(layer.register_forward_hook(partial(self._output, name)))

    def detach(self):
        for handle in self.handles:
            handle.remove()
        self.handles, self.inputs = [], {}

    def _input(self, name, module, args):
        if torch.is_grad_enabled():
            self.inputs[name] = args[0].detach()

    def _output(self, name, module, args, output):
        if output.requires_grad:
            output.register_hook(partial(self._gradient, name))

    def _gradient(self, name, grad):
        x = self.inputs.pop(name)
        if self.bf16_inputs:  # autocast multiplies the BF16-rounded input
            x = x.to(torch.bfloat16)
        sequences = x.shape[0]
        x = x.float().reshape(-1, x.shape[-1])
        d = grad.float().reshape(-1, grad.shape[-1])
        u1, v1 = self.pairs[name]
        s = (d @ u1) * (x @ v1)
        st = self.stats[name]
        st["tokens"] += s.numel()
        st["contribution"] += s.sum().double()
        st["absolute"] += s.abs().sum().double()
        st["top1"] += s.abs().topk(max(1, math.ceil(0.01 * s.numel()))).values.sum().double()
        st["position"] += s.view(sequences, -1).sum(0).double()
        sum_d, sum_x = d.sum(0).double(), x.sum(0).double()
        st["sum_grad"] += sum_d
        st["sum_input"] += sum_x
        st["batch_grad"] += sum_d
        st["batch_input"] += sum_x
        st["batch_tokens"] += s.numel()
        d0 = d.view(sequences, -1, d.shape[-1])[:, 0]
        x0 = x.view(sequences, -1, x.shape[-1])[:, 0]
        st["sum_grad_pos0"] += d0.sum(0).double()
        st["sum_input_pos0"] += x0.sum(0).double()
        st["pos0_term"] += (d0.T @ x0).double()
        st["input_abs"] += x.abs().sum(0).double()
        if self.reconstruct:
            term = d.T @ x
            st["reconstruction"] = term if st["reconstruction"] is None else st["reconstruction"] + term
        return None

    def end_batch(self):
        for st in self.stats.values():
            st["batchwise_mean_product"] += torch.outer(st["batch_grad"], st["batch_input"]) / st["batch_tokens"]
            st["batch_grad"].zero_()
            st["batch_input"].zero_()
            st["batch_tokens"] = 0


def batch_gradient(model, tokens, seq_len, micro_sequences, config, device):
    """Mean-NLL gradient over one contiguous batch, weighted as in training."""
    for parameter in model.parameters():
        parameter.grad = None
    count = tokens.numel() - 1
    micro = micro_sequences * seq_len
    total = 0.0
    for offset in range(0, count, micro):
        n = min(micro, count - offset)
        x, y = token_views(tokens, offset, n, seq_len)
        with amp_context(device, config):
            loss = model(x, y)
        (loss * (n / count)).backward()
        total += float(loss.detach()) * (n / count)
    return total


@torch.no_grad()
def validation_nll(model, tokens, config, device, micro_sequences=16):
    count = tokens.numel() - 1
    micro = micro_sequences * model.config.seq_len
    total = 0.0
    for offset in range(0, count, micro):
        n = min(micro, count - offset)
        x, y = token_views(tokens, offset, n, model.config.seq_len)
        with amp_context(device, config):
            total += float(model(x, y).double()) * n
    return total / count


@torch.no_grad()
def forward_structure(model, tokens, config, device, sequences=64):
    """Attention mass on key 0 per head and residual-stream channel statistics."""
    seq_len = model.config.seq_len
    x_ids, _ = token_views(tokens, 0, sequences * seq_len, seq_len)
    with amp_context(device, config):
        h = model.embed(x_ids) + model.position(model._positions[:seq_len])
        residual = [h.float()]
        sink = []
        for block in model.blocks:
            q, k, _ = block.attn.heads(block.ln1(h))
            q, k = q.float(), k.float()
            t = q.shape[2]
            scores = (q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
            mask = torch.ones(t, t, dtype=torch.bool, device=device).tril()
            probs = scores.masked_fill(~mask, float("-inf")).softmax(-1)
            sink.append([probs[:, :, 1:, 0].mean((0, 2)).tolist(),
                         probs[:, :, t // 2:, 0].mean((0, 2)).tolist()])
            h = block(h)
            residual.append(h.float())
    stats = []
    for h in residual:
        norms = h.norm(dim=-1)
        channel_abs = h.abs().mean((0, 1))
        channel_max = h.abs().amax((0, 1))
        top = channel_max.topk(5)
        stats.append({"pos0_norm_over_rest": float(norms[:, 0].mean() / norms[:, 1:].mean()),
                      "channel_max_over_mean_abs": float(channel_max.max() / channel_abs.mean()),
                      "top_channels": top.indices.tolist(), "top_channel_max": top.values.tolist(),
                      "mean_abs": float(channel_abs.mean())})
    return {"attention_to_position0": sink, "residual_stream": stats,
            "note": "residual[0] is the embedding sum; residual[i] is block i output"}


def mode_projections(u, vh, grad):
    """u_i^T G v_i for every singular pair of the reference state."""
    return ((u.T @ grad) * vh).sum(1)


def concentration(vector):
    w = vector / vector.norm()
    top = w.abs().topk(3)
    return {"participation": float(1 / (w**4).sum()), "max_abs": float(top.values[0]),
            "top_coordinates": top.indices.tolist()}


def cosine(a, b):
    return float((a @ b) / (a.norm() * b.norm()).clamp_min(1e-300))


def run(run_dir, out, *, offset, batches, batch_tokens, micro_sequences, device, gate_only=False):
    started = time.perf_counter()
    run_dir, out, device = Path(run_dir), Path(out), torch.device(device)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {out}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "source").mkdir(exist_ok=True)
    for source in Path(__file__).parent.glob("*.py"):
        (out / "source" / source.name).write_bytes(source.read_bytes())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    saved = torch.load(run_dir / "checkpoint.pt", map_location="cpu", weights_only=False)
    config = load_config(overrides=saved["config"])  # fills keys absent from pre-Muon configs
    model = GPT(ModelConfig(**config["model"])).to(device)
    model.load_state_dict(saved["model"])
    optimizer, _ = make_optimizer(model, config, device)
    optimizer.load_state_dict(saved["optimizer"])
    key, units, noise_factor = state_convention(config)
    layers = body_layers(model)
    seq_len = model.config.seq_len
    budget = saved["tokens"]
    if offset < budget + 1:
        raise ValueError("Fresh tokens must start after every trained token")
    metadata = {"run": str(run_dir), "optimizer": config["optimizer"], "state": key,
                "gradient_units": units, "noise_factor": noise_factor, "step": saved["step"],
                "trained_tokens": budget, "fresh_offset": offset, "batches": batches,
                "batch_tokens": batch_tokens, "micro_sequences": micro_sequences,
                "precision": config["precision"], "device": str(device),
                "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
                "torch_version": torch.__version__, "source_sha256": source_hashes(),
                "gates": GATES}
    atomic_json(out / "metadata.json", metadata)
    atomic_json(out / "status.json", {"status": "running"})
    try:
        references, svd = {}, {}
        for name, layer in layers.items():
            state = optimizer.state[layer.weight][key].detach().double() * units
            u, s, vh = torch.linalg.svd(state, full_matrices=False)
            references[name] = state
            svd[name] = (u, s, vh)
        pairs = {name: (u[:, 0].float(), vh[0].float()) for name, (u, s, vh) in svd.items()}
        # ---- gates: loaded weights and hook reconstruction
        validation = TokenStream(config["validation_pattern"])
        val_tokens = validation.device_tokens(0, config["validation_tokens"] + 1, device)
        recorded = json.loads((run_dir / "steps" / f"step{saved['step']:06d}.json").read_text())["validation_nll"]
        measured = validation_nll(model, val_tokens, config, device)
        gates = {"validation_nll": {"recorded": recorded, "measured": measured,
                                    "abs_diff": abs(measured - recorded)}}
        stream = TokenStream(config["train_pattern"])
        recorder = Recorder(layers, pairs, seq_len,
                            bf16_inputs=device.type == "cuda" and config["precision"] == "bf16")
        recorder.attach()
        recorder.reconstruct = True
        probe = stream.device_tokens(offset, micro_sequences * seq_len + 1, device)
        batch_gradient(model, probe, seq_len, micro_sequences, config, device)
        errors = {}
        for name, layer in layers.items():
            grad = layer.weight.grad.float()
            recon = recorder.stats[name]["reconstruction"]
            errors[name] = float((recon - grad).norm() / grad.norm())
        gates["reconstruction"] = {"max_relative_error": max(errors.values()), "per_matrix": errors}
        recorder.detach()
        passed = (gates["validation_nll"]["abs_diff"] <= GATES["validation_nll"]
                  and gates["reconstruction"]["max_relative_error"] <= GATES["reconstruction"])
        atomic_json(out / "gates.json", {"passed": passed, **gates})
        if not passed:
            raise RuntimeError(f"Instrument gate failed: {gates['validation_nll']}, "
                               f"max reconstruction {gates['reconstruction']['max_relative_error']}")
        if gate_only:
            atomic_json(out / "status.json", {"status": "gates_passed",
                        "seconds": time.perf_counter() - started})
            return
        # ---- fresh-gradient pass (fresh recorder: the probe tokens are reused as batch 0)
        recorder = Recorder(layers, pairs, seq_len,
                            bf16_inputs=device.type == "cuda" and config["precision"] == "bf16")
        recorder.attach()
        rank = {name: s.numel() for name, (u, s, vh) in svd.items()}
        projections = {name: [] for name in layers}
        noise = {name: [] for name in layers}
        pair_means = {name: [] for name in layers}
        scaling = {name: [] for name in layers}
        totals = {name: torch.zeros_like(references[name]) for name in layers}
        previous, losses = None, []
        timing = time.perf_counter()
        for j in range(batches):
            tokens = stream.device_tokens(offset + j * batch_tokens, batch_tokens + 1, device)
            losses.append(batch_gradient(model, tokens, seq_len, micro_sequences, config, device))
            recorder.end_batch()
            grads = {name: layer.weight.grad.detach().float().clone() for name, layer in layers.items()}
            for name, grad in grads.items():
                u, s, vh = svd[name]
                g = grad.double()
                projections[name].append(mode_projections(u, vh, g).cpu())
                totals[name] += g
                if j + 1 in SCALING_BATCHES:
                    scaling[name].append(torch.linalg.svdvals(totals[name].float() / (j + 1)).cpu())
            if j % 2 == 1:
                for name in layers:
                    noise[name].append(torch.linalg.svdvals((previous[name] - grads[name]) / 2).cpu())
                    pair_means[name].append(torch.linalg.svdvals((previous[name] + grads[name]) / 2).cpu())
            previous = grads
            if j == 0 or (j + 1) % 8 == 0:
                print(json.dumps({"batch": j + 1, "loss": losses[-1],
                                  "seconds": round(time.perf_counter() - timing, 1)}), flush=True)
        recorder.detach()
        structure = forward_structure(model, stream.device_tokens(offset, 64 * seq_len + 1, device),
                                      config, device)
        # ---- summaries
        summary, arrays = {}, {}
        count = batches
        for name, layer in layers.items():
            st = recorder.stats[name]
            u, s, vh = svd[name]
            u1, v1 = u[:, 0], vh[0]
            mean_grad = totals[name] / count
            gu, gs, gvh = torch.linalg.svd(mean_grad, full_matrices=False)
            proj = torch.stack(projections[name]).double()           # batches x rank
            anchor = float(proj[:, 0].mean())                          # u1^T Gbar v1
            hook_anchor = float(st["contribution"]) / count
            absolute_scale = float(st["absolute"]) / count
            mean_product = torch.outer(st["sum_grad"], st["sum_input"]) / st["tokens"] / count
            batchwise = st["batchwise_mean_product"] / count
            pos0 = st["pos0_term"] / count
            frob2 = float((mean_grad**2).sum())

            def share(term):
                return {"of_u1v1": float(u1 @ term @ v1) / anchor,
                        "frobenius_ratio": float(term.norm()) / math.sqrt(frob2),
                        "inner_share": float((term * mean_grad).sum()) / frob2}

            band_rows = []
            normalized = (s / s.norm()).cpu().numpy()
            for band, lo, hi in bands(rank[name]):
                block = proj[:, lo - 1:hi].sum(1)
                cross, se = float(block.mean()), float(block.std(unbiased=True) / math.sqrt(count))
                insample = float(s[lo - 1:hi].sum())
                band_rows.append({"band": band, "modes": [lo, hi], "in_sample": insample,
                                  "cross_fitted": cross, "se": se,
                                  "ratio": cross / insample if insample else None,
                                  "z": cross / se if se else None})
            per_mode = proj.mean(0).numpy()
            first = band_rows[0]
            reproduced = first["cross_fitted"] >= 0.5 * first["in_sample"] and (first["z"] or 0) >= 3
            weights = ns_scalar(normalized)
            noise_sv = torch.stack(noise[name]).double()
            edge_batch, edge2_batch = float(noise_sv[:, 0].mean()), float(noise_sv[:, 1].mean())
            edge_state, edge2_state = edge_batch * noise_factor, edge2_batch * noise_factor
            above = int((s > edge_state).sum())
            pair_rho = (torch.stack(pair_means[name])[:, 0]**2 / (torch.stack(pair_means[name])**2).sum(1))
            summary[name] = {
                "reference": {"sigma1": float(s[0]), "rho1": float(s[0]**2 / (s**2).sum()),
                              "median_normalized": float(normalized[math.ceil(0.5 * rank[name]) - 1]),
                              "u1": concentration(u1), "v1": concentration(v1)},
                "fresh_mean_gradient": {"rho1": float(gs[0]**2 / (gs**2).sum()),
                                        "u1_alignment": abs(float(gu[:, 0] @ u1)),
                                        "v1_alignment": abs(float(gvh[0] @ v1)),
                                        "u1v1_projection": anchor,
                                        "u1v1_over_in_sample_sigma1": anchor / float(s[0]),
                                        "spike_reproduced": bool(reproduced),
                                        "hook_vs_gradient_rel_diff": abs(hook_anchor - anchor) / absolute_scale},
                "mean_product": {**share(mean_product),
                                 "cos_mean_input_v1": cosine(st["sum_input"], v1),
                                 "cos_mean_grad_u1": cosine(st["sum_grad"], u1),
                                 "mean_input": concentration(st["sum_input"])},
                "batchwise_mean_product": share(batchwise),
                "position0": {**share(pos0),
                              "cos_pos0_input_v1": cosine(st["sum_input_pos0"], v1),
                              "cos_pos0_grad_u1": cosine(st["sum_grad_pos0"], u1),
                              "first4_of_u1v1": float(st["position"][:4].sum() / st["contribution"])},
                "tokens": {"sign_consistency": float(st["contribution"] / st["absolute"]),
                           "top1pct_share_of_abs": float(st["top1"] / st["absolute"]),
                           "count": st["tokens"]},
                "bands": band_rows,
                "ns_weighted_descent": {
                    "in_sample": float((weights * s.cpu().numpy()).sum() / s.sum()),
                    "cross_fitted": float((weights * per_mode).sum() / per_mode.sum())},
                "noise": {"edge_batch_1M": edge_batch, "edge_state": edge_state,
                          "edge_state_over_state_frobenius": edge_state / float(s.norm()),
                          "modes_above_edge": above,
                          "edge2_batch_1M": edge2_batch, "edge2_state": edge2_state,
                          "edge2_state_over_state_frobenius": edge2_state / float(s.norm()),
                          "modes_above_edge2": int((s > edge2_state).sum()),
                          "energy_above_edge": float((s[s > edge_state]**2).sum() / (s**2).sum()),
                          "pair_mean_rho1": float(pair_rho.mean())},
                "scaling": [{"tokens": b * batch_tokens,
                             "rho1": float(v[0]**2 / (v**2).sum()),
                             "median_normalized": float(v[math.ceil(0.5 * rank[name]) - 1] / v.norm()),
                             "sigma1_over_median": float(v[0] / v[math.ceil(0.5 * rank[name]) - 1])}
                            for b, v in zip(SCALING_BATCHES, scaling[name])]}
            arrays.update({f"{name}/state_sigma": s.cpu().numpy(), f"{name}/state_u1": u1.cpu().numpy(),
                           f"{name}/state_v1": v1.cpu().numpy(), f"{name}/fresh_sigma": gs.cpu().numpy(),
                           f"{name}/fresh_u1": gu[:, 0].cpu().numpy(), f"{name}/fresh_v1": gvh[0].cpu().numpy(),
                           f"{name}/projections": proj.numpy(), f"{name}/position": st["position"].cpu().numpy(),
                           f"{name}/mean_grad_direction": st["sum_grad"].cpu().numpy(),
                           f"{name}/mean_input": (st["sum_input"] / st["tokens"]).cpu().numpy(),
                           f"{name}/input_abs_mean": (st["input_abs"] / st["tokens"]).cpu().numpy(),
                           f"{name}/noise_sigma": torch.stack(noise[name]).numpy(),
                           f"{name}/pair_mean_sigma": torch.stack(pair_means[name]).numpy(),
                           f"{name}/scaling_sigma": torch.stack(scaling[name]).numpy()})
        worst = max(v["fresh_mean_gradient"]["hook_vs_gradient_rel_diff"] for v in summary.values())
        gates["hook_vs_gradient"] = worst
        np.savez_compressed(out / "arrays.npz", **arrays)
        atomic_json(out / "summary.json", {"metadata": metadata, "gates": gates,
                                           "fresh_losses": losses, "structure": structure,
                                           "matrices": summary})
        atomic_json(out / "status.json", {"status": "complete" if worst <= GATES["hook_vs_gradient"]
                                          else "hook_vs_gradient_check_failed",
                                          "seconds": time.perf_counter() - started})
    except Exception as exc:
        atomic_json(out / "status.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="Completed run directory with checkpoint.pt")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offset", type=int, default=2_000_000_000)
    parser.add_argument("--batches", type=int, default=64)
    parser.add_argument("--batch-tokens", type=int, default=524_288)
    parser.add_argument("--micro-sequences", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--gate-only", action="store_true")
    args = parser.parse_args()
    if args.batches % 2 or args.batch_tokens % 512:
        raise ValueError("Use an even batch count and whole sequences")
    run(args.run, args.out, offset=args.offset, batches=args.batches, batch_tokens=args.batch_tokens,
        micro_sequences=args.micro_sequences, device=args.device, gate_only=args.gate_only)


if __name__ == "__main__":
    main()

"""Second-order audit, step 1: full-size checks of research/adamw_spectra/gn_probe.py on saved checkpoints,
and the first measurement (exact one-sided Gauss-Newton marginals vs K-FAC, per-token and empirical-Fisher).

usage: check_tools.py OUT_JSON CHECKPOINT [CHECKPOINT ...] [--sequences N] [--micro B]

Measurement only: FP32 eager model, TF32 off, explicit attention for forward-mode AD. Probe sequences come
from the validation stream after its first 2,097,152 tokens (training runs validate on the first 1,048,576).
Checks, each against the unit given in gn_probe.py:
  a) per-sequence gradients from hooks sum to the autograd batch gradient (all 48 matrices);
  b) T * E<g_sampled, D>^2 matches the exact forward-mode q(D) on the same sequences (3 directions);
  c) exact symmetries have ~0 curvature relative to matched random directions (V/O gauge, q/k head
     scale, norm gain vs reader columns, residual-stream scale); single-writer scalings do not;
  d) trace consistency of the marginals (Tr of the input and output marginals agree).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402

OFFSET = 2 * 1048576


def batches(stream, start_sequence, count, micro, seq_len, device):
    for first in range(start_sequence, start_sequence + count, micro):
        b = min(micro, start_sequence + count - first)
        yield stream.batch(OFFSET + first * seq_len, b, seq_len, device)


def check_checkpoint(path, args, device, stream):
    started = time.time()
    model, saved = P.load_checkpoint(path, device)
    T = model.config.seq_len
    cuda_gen = torch.Generator(device=device).manual_seed(0)
    cpu_gen = torch.Generator().manual_seed(0)
    out = {"checkpoint": str(path), "step": saved.get("step"), "tokens": saved.get("tokens")}
    layers = P.hidden_linears(model)

    # a) per-sequence gradients sum to the batch gradient
    recorder = P.Recorder(model)
    x, y = next(batches(stream, 0, args.micro, args.micro, T, device))
    grads, _ = P.gradient_passes(model, recorder, x, y, cuda_gen, draws=1)
    errors = {n: float((P.per_sequence_gradients(recorder.inputs[n], recorder.errors[n][0]).sum(0) - grads[n]).norm()
                       / grads[n].norm()) for n in layers}
    out["a_per_sequence_sum"] = {"max_rel_err": max(errors.values()), "worst": max(errors, key=errors.get)}

    # b) sampled-label second moment vs exact quadratic form, same sequences
    weights = {n: layer.weight.detach() for n, layer in layers.items()}
    directions = {"all_body": P.random_like(weights, cpu_gen),
                  "all_keys": P.random_like({n: w for n, w in weights.items() if n.endswith(".k")}, cpu_gen),
                  "block04.v": P.random_like({"block04.v": weights["block04.v"]}, cpu_gen)}
    exact = {k: [] for k in directions}
    samples = {k: [] for k in directions}
    for x, y in batches(stream, args.micro, args.mc_sequences, 4, T, device):   # 4 sequences: draws keep errors
        for key, d in directions.items():
            exact[key].append(P.gn_quadratic(model, x, d)[1])
        P.gradient_passes(model, recorder, x, y, cuda_gen, draws=args.mc_draws)
        for key, d in directions.items():
            total = torch.zeros(x.shape[0], args.mc_draws, device=device)
            for name, direction in d.items():
                for k, e in enumerate(recorder.errors[name][1:]):
                    total[:, k] += (P.per_sequence_gradients(recorder.inputs[name], e) * direction).sum((1, 2))
            samples[key].append((T * total ** 2).flatten())
    out["b_sampled_vs_exact"] = {}
    for key in directions:
        q = torch.cat(exact[key]).double().mean()
        s = torch.cat(samples[key]).double()
        out["b_sampled_vs_exact"][key] = {"exact": float(q), "sampled": float(s.mean()),
                                          "ratio": float(s.mean() / q),
                                          "mc_rel_se": float(s.std() / s.numel() ** 0.5 / s.mean())}

    # c) symmetry directions (and non-symmetries) vs matched random directions, one microbatch
    x, _ = next(batches(stream, 0, args.micro, args.micro, T, device))
    L = len(model.blocks)
    cases = {}
    for block in (1, L // 2, L):
        for head in (0, model.config.n_head - 1):
            cases[f"vo_gauge_b{block}_h{head}"] = P.vo_gauge_direction(model, block, head, cpu_gen)
    cases["q_head_scale_b2_h3"] = P.radial_head_direction(model, 2, "q", 3)
    cases[f"k_head_scale_b{L - 2}_h1"] = P.radial_head_direction(model, L - 2, "k", 1)
    cases["ln1_gain_vs_qkv_b3"] = P.norm_gain_direction(model, 3, "ln1", cpu_gen)
    cases[f"ln2_gain_vs_up_b{L - 1}"] = P.norm_gain_direction(model, L - 1, "ln2", cpu_gen)
    cases["residual_stream_scale"] = P.residual_scale_direction(model)
    for kind in ("v", "o", "up", "down"):
        name = f"block{L // 2:02d}.{kind}"
        cases[f"not_symmetry_scale_{name}"] = {name: weights[name].clone()}
    out["c_symmetries"] = {}
    for key, d in cases.items():
        null = float(P.gn_quadratic(model, x, d)[0])
        reference = float(P.gn_quadratic(model, x, P.random_like(d, cpu_gen))[0])
        out["c_symmetries"][key] = {"q": null, "random_q": reference, "ratio": null / reference}

    # First measurement: one-sided marginals for all hidden matrices
    marginals = {n: P.Marginals(layer.weight.shape[0], layer.weight.shape[1], T, device)
                 for n, layer in layers.items()}
    start = args.micro + args.mc_sequences
    for x, y in batches(stream, start, args.sequences, args.micro, T, device):
        P.gradient_passes(model, recorder, x, y, cuda_gen, draws=1)
        for n in layers:
            marginals[n].add(recorder.inputs[n], recorder.errors[n][0], recorder.errors[n][1:])
    recorder.remove()
    out["marginals"] = {}
    worst_trace = 0.0
    for n, m in marginals.items():
        s = m.summary()
        worst_trace = max(worst_trace, abs(float(s["exact_in"].trace() / s["exact_out"].trace()) - 1))
        out["marginals"][n] = P.marginal_report(s)
    out["d_trace_consistency_max_rel_err"] = worst_trace
    out["sequences"] = args.sequences
    out["seconds"] = time.time() - started
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    parser.add_argument("--sequences", type=int, default=2048)
    parser.add_argument("--micro", type=int, default=8)
    parser.add_argument("--mc-sequences", type=int, default=64)
    parser.add_argument("--mc-draws", type=int, default=16)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    results = {"args": {k: str(v) for k, v in vars(args).items()},
               "gpu": torch.cuda.get_device_name(0), "runs": []}
    for path in args.checkpoints:
        results["runs"].append(check_checkpoint(path, args, device, stream))
        args.out.write_text(json.dumps(results, indent=1) + "\n")
        print(json.dumps({k: v for k, v in results["runs"][-1].items() if k != "marginals"}, indent=1), flush=True)


if __name__ == "__main__":
    main()

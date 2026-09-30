"""Second-order audit, step 3, the gap map: per-direction curvature, signal and noise of every hidden matrix in
its Kronecker frame, with each optimizer's actual update and momentum projected into the same frame.

usage: measure_frame.py OUT_DIR ARM_DIR:STEP [ARM_DIR:STEP ...] [--sequences N] [--basis-sequences M]

For each hidden matrix W at kept step t:
- frame: U = eigenvectors of B (per-token output factor, model-sampled labels), V = eigenvectors of C (input
  second moment), both from M held-out basis sequences, so the statistics below are not fitted to their own basis;
- per pair (i, j), from N other sequences (one sampled-label draw each; units of gn_probe.py):
    exact  = T E[(u_i^T g_sampled v_j)^2]        exact GN diagonal in the frame
    ekfac  = token mean of (u_i^T e~)^2 (v_j^T x)^2   per-token GN;  K-FAC = lam_B[i] lam_C[j]
    mean   = E[u_i^T g v_j] (data labels),  noise = per-sequence variance;  signal2 = mean^2 - noise/N
    block_var_ratio: variance of 8-sequence contiguous block means x 8 / noise (1 if sequences independent)
- update   = U^T (W[t+1] - W[t]) V            the applied change of step t+1 (lr, decay, everything)
  decay    = U^T (-lr[t+1] wd W[t]) V          its decoupled-decay part (the rest is the optimizer's direction)
  momentum = U^T M[t] V                        the momentum buffer after step t (sum form: M = 0.95 M + g).
Probe sequences: validation stream after its first 2,097,152 tokens; the same sequences for every checkpoint.
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
from research.adamw_spectra.train import learning_rate, token_budget  # noqa: E402

OFFSET = 2 * 1048576


def run_batches(stream, first_sequence, count, micro, T, device):
    for first in range(first_sequence, first_sequence + count, micro):
        yield stream.batch(OFFSET + first * T, min(micro, first_sequence + count - first), T, device)


def measure(arm, step, args, device, stream):
    started = time.time()
    kept = Path(arm) / "scientific" / "kept"
    model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
    after = torch.load(kept / f"step{step + 1:06d}_weights.pt", map_location="cpu", weights_only=False)
    config = saved["config"]
    T = model.config.seq_len
    layers = P.hidden_linears(model)
    names = list(layers)
    cuda_gen = torch.Generator(device=device).manual_seed(0)
    recorder = P.Recorder(model)

    # basis from held-out sequences
    sums = {n: [0, 0, 0] for n in names}
    for x, y in run_batches(stream, 0, args.basis_sequences, args.micro, T, device):
        P.gradient_passes(model, recorder, x, y, cuda_gen, draws=1)
        for n in names:
            c, b, count = P.second_moments(recorder.inputs[n], T * recorder.errors[n][1])
            sums[n][0] = sums[n][0] + c
            sums[n][1] = sums[n][1] + b
            sums[n][2] += count
    frames = {}
    for n in names:
        lam_c, v = P.eigenbasis(sums[n][0], sums[n][2])
        lam_b, u = P.eigenbasis(sums[n][1], sums[n][2])
        frames[n] = P.Frame(u, v, lam_b, lam_c, T)
    del sums

    # statistics on other sequences
    for x, y in run_batches(stream, args.basis_sequences, args.sequences, args.micro, T, device):
        P.gradient_passes(model, recorder, x, y, cuda_gen, draws=1)
        for n in names:
            frames[n].add(recorder.inputs[n], recorder.errors[n][0], recorder.errors[n][1:])
    recorder.remove()

    # applied update of step t+1 and momentum after step t, in each frame
    count = sum(p.numel() for p in model.parameters())
    budget = token_budget(config, count, T)
    lr = learning_rate(config, step + 1, step * config["batch_tokens"], budget)
    keys = P.parameter_keys(model, names)
    state = saved["optimizer"]["state"]
    out = {}
    for index, (n, key) in enumerate(zip(names, keys)):
        f = frames[n]
        s = f.summary()
        w0 = saved["model"][key].to(device=device, dtype=torch.float32)
        w1 = after["model"][key].to(device=device, dtype=torch.float32)
        m = state[index]["momentum_buffer"].to(device=device, dtype=torch.float32)
        if m.shape != w0.shape:
            raise RuntimeError(f"Momentum/parameter order mismatch at {n}")
        project = lambda a: (f.U.T @ a @ f.V).cpu()
        out[n] = {"lam_B": f.lam_B.float().cpu(), "lam_C": f.lam_C.float().cpu(),
                  "exact": s["exact"].float().cpu(), "ekfac": s["ekfac"].float().cpu(),
                  "mean": s["signal"].float().cpu(), "noise": s["noise"].float().cpu(),
                  "block_var_ratio_median": float(s["block_var_ratio"].median()) if "block_var_ratio" in s else None,
                  "update": project(w1 - w0), "decay": project(-lr * config["weight_decay"] * w0),
                  "momentum": project(m)}
    meta = {"arm": str(arm), "step": step, "lr_next": lr, "weight_decay": config["weight_decay"],
            "learning_rate": config["learning_rate"], "sequences": args.sequences,
            "basis_sequences": args.basis_sequences, "T": T, "seconds": time.time() - started,
            "gpu": torch.cuda.get_device_name(0)}
    name = f"{Path(arm).name}_step{step:06d}"
    torch.save({"meta": meta, "matrices": out}, args.out / f"{name}.pt")
    print(json.dumps({**meta, "name": name}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("items", nargs="+")
    parser.add_argument("--sequences", type=int, default=8192)
    parser.add_argument("--basis-sequences", type=int, default=512)
    parser.add_argument("--micro", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    for item in args.items:
        arm, step = item.rsplit(":", 1)
        measure(arm, int(step), args, device, stream)


if __name__ == "__main__":
    main()

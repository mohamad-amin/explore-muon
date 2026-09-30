"""How long does the true gradient along each direction stay correlated with itself? Per frame pair of every
hidden matrix (frame = eigenvectors of B and C at step t, from held-out basis sequences), the data-label mean
gradient is measured at W[t] (sequence set A) and at later weights W[t'] (disjoint set B), so the two estimates
have independent noise. Per curvature bin (exact curvature is not measured here: bins use K-FAC's lam_B lam_C),
corr = sum g_A(t) g_B(t') / sqrt(sum s(t) sum s(t')), with s the debiased squared signal.

usage: measure_persistence.py OUT_DIR ARM_DIR:T:T2[,T2...] [...]
    T2 = "next" for W[t+1], a kept step N (stepN.pt), or "wN" for kept next-step weights (stepN_weights.pt).
    ARM_DIR holds kept/ directly or under scientific/.
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


def batches(stream, first_sequence, count, micro, T, device):
    for first in range(first_sequence, first_sequence + count, micro):
        yield stream.batch(OFFSET + first * T, min(micro, first_sequence + count - first), T, device)


def gradient_frames(model, frames, stream, first, count, micro, T, device):
    """Accumulate data-label per-sequence gradient moments in the given frames (copies of their bases)."""
    acc = {n: P.Frame(f.U, f.V, f.lam_B, f.lam_C, T) for n, f in frames.items()}
    recorder = P.Recorder(model)
    for x, y in batches(stream, first, count, micro, T, device):
        P.gradient_passes(model, recorder, x, y, draws=0)
        for n in acc:
            acc[n].add(recorder.inputs[n], recorder.errors[n][0], [])
    recorder.remove()
    return {n: f.summary() for n, f in acc.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("items", nargs="+")
    parser.add_argument("--sequences", type=int, default=4096)
    parser.add_argument("--basis-sequences", type=int, default=512)
    parser.add_argument("--micro", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    for item in args.items:
        arm, t, later = item.split(":")
        t = int(t)
        started = time.time()
        kept = Path(arm) / "scientific" / "kept"
        kept = kept if kept.exists() else Path(arm) / "kept"
        model, saved = P.load_checkpoint(kept / f"step{t:06d}.pt", device)
        T = model.config.seq_len
        cuda_gen = torch.Generator(device=device).manual_seed(0)
        names = list(P.hidden_linears(model))
        recorder = P.Recorder(model)
        sums = {n: [0, 0, 0] for n in names}
        for x, y in batches(stream, 0, args.basis_sequences, args.micro, T, device):
            P.gradient_passes(model, recorder, x, y, cuda_gen, draws=1)
            for n in names:
                c, b, count = P.second_moments(recorder.inputs[n], T * recorder.errors[n][1])
                sums[n][0] = sums[n][0] + c; sums[n][1] = sums[n][1] + b; sums[n][2] += count
        recorder.remove()
        frames = {}
        for n in names:
            lam_c, v = P.eigenbasis(sums[n][0], sums[n][2])
            lam_b, u = P.eigenbasis(sums[n][1], sums[n][2])
            frames[n] = P.Frame(u, v, lam_b, lam_c, T)
        del sums
        A = args.basis_sequences
        B = A + args.sequences
        stats = {"t": gradient_frames(model, frames, stream, A, args.sequences, args.micro, T, device)}
        config = saved["config"]["model"]
        for target in later.split(","):
            if target == "next":
                weights = torch.load(kept / f"step{t + 1:06d}_weights.pt", map_location="cpu", weights_only=False)
                other = P.build_model(config, weights["model"], device)
                label = "t+1"
            elif target.startswith("w"):
                weights = torch.load(kept / f"step{int(target[1:]):06d}_weights.pt", map_location="cpu", weights_only=False)
                other = P.build_model(config, weights["model"], device)
                label = f"t={int(target[1:])}"
            else:
                other, _ = P.load_checkpoint(kept / f"step{int(target):06d}.pt", device)
                label = f"t={int(target)}"
            stats[label] = gradient_frames(other, frames, stream, B, args.sequences, args.micro, T, device)
            del other
        out = {n: {"lam_B": frames[n].lam_B.float().cpu(), "lam_C": frames[n].lam_C.float().cpu(),
                   **{f"{label}:{key}": s[n][key].float().cpu() for label, s in stats.items()
                      for key in ("signal", "noise")}} for n in names}
        meta = {"arm": arm, "t": t, "targets": later, "sequences": args.sequences, "seconds": time.time() - started}
        name = f"{Path(arm).name}_t{t:06d}"
        torch.save({"meta": meta, "matrices": out}, args.out / f"{name}.pt")
        print(json.dumps({**meta, "name": name}), flush=True)


if __name__ == "__main__":
    main()

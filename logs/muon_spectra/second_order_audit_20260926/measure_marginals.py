"""Second-order audit, step 3, measurement M-A: one-sided GN marginals of every hidden matrix at kept
checkpoints of the trajectory runs, with the within/between-sequence split of the input second moment.

usage: measure_marginals.py OUT_DIR ARM_DIR:STEP [ARM_DIR:STEP ...] [--sequences N]
STEP is a kept step (…/scientific/kept/stepNNNNNN.pt) or "final" (…/scientific/checkpoint.pt).
Writes OUT_DIR/<arm>_step<NNNNNN>.json (per-matrix report, fit, sanity) and a .pt with compact arrays:
C and B spectra, and the exact / per-token / empirical-Fisher / K-FAC curvatures along their eigenvectors;
for attention inputs also the full C, C_between and the q/k/v exact input marginals (FP32).

Probe sequences: validation stream after its first 2,097,152 tokens, the same N sequences for every
checkpoint (paired across training steps and optimizers). One model-sampled label draw per sequence.
Sanity per checkpoint: one V/O gauge direction and the residual-stream scale must be flat (< 1e-6 of random).
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


def checkpoint_path(arm, step):
    base = Path(arm) / "scientific"
    return base / "checkpoint.pt" if step == "final" else base / "kept" / f"step{int(step):06d}.pt"


def along(a, vectors):
    return torch.einsum("ij,ik,jk->k", a, vectors, vectors)


def measure(arm, step, args, device, stream):
    started = time.time()
    model, saved = P.load_checkpoint(checkpoint_path(arm, step), device)
    T = model.config.seq_len
    layers = P.hidden_linears(model)
    cuda_gen = torch.Generator(device=device).manual_seed(0)
    cpu_gen = torch.Generator().manual_seed(0)
    x, _ = stream.batch(OFFSET, args.micro, T, device)
    L = len(model.blocks)
    sanity = {}
    for key, d in (("vo_gauge", P.vo_gauge_direction(model, L // 2, 0, cpu_gen)),
                   ("residual_scale", P.residual_scale_direction(model))):
        sanity[key] = float(P.gn_quadratic(model, x, d)[0] / P.gn_quadratic(model, x, P.random_like(d, cpu_gen))[0])
    if max(sanity.values()) > 1e-6:
        raise RuntimeError(f"Symmetry sanity failed: {sanity}")
    marginals = {n: P.Marginals(layer.weight.shape[0], layer.weight.shape[1], T, device) for n, layer in layers.items()}
    recorder = P.Recorder(model)
    for first in range(0, args.sequences, args.micro):
        x, y = stream.batch(OFFSET + first * T, min(args.micro, args.sequences - first), T, device)
        P.gradient_passes(model, recorder, x, y, cuda_gen, draws=1)
        for n in layers:
            marginals[n].add(recorder.inputs[n], recorder.errors[n][0], recorder.errors[n][1:])
    recorder.remove()
    report, arrays = {}, {}
    for n, m in marginals.items():
        s = m.summary()
        report[n] = P.marginal_report(s)
        report[n]["fit_exact"] = P.within_between_fit(s, "exact_in")
        report[n]["fit_token"] = P.within_between_fit(s, "token_in")
        lam_c, v = torch.linalg.eigh(s["C"])
        lam_b, u = torch.linalg.eigh(s["B"])
        arrays[n] = {"lam_C": lam_c.flip(0).float().cpu(), "lam_B": lam_b.flip(0).float().cpu(),
                     "trB": float(s["B"].trace()), "trC": float(s["C"].trace()),
                     **{f"in_{key}": along(s[key], v).flip(0).float().cpu() for key in ("exact_in", "token_in", "ef_in")},
                     **{f"out_{key}": along(s[key], u).flip(0).float().cpu() for key in ("exact_out", "token_out", "ef_out")},
                     "x_mean": s["x_mean"].float().cpu()}
        kind = n.split(".")[1]
        if kind in ("q", "k", "v"):
            arrays[n]["exact_in_full"] = s["exact_in"].float().cpu()
            if kind == "q":
                arrays[n]["C_full"] = s["C"].float().cpu()
                arrays[n]["C_between_full"] = s["C_between"].float().cpu()
    name = f"{Path(arm).name}_step{(1469 if step == 'final' else int(step)):06d}"
    meta = {"arm": str(arm), "step": saved.get("step"), "tokens": saved.get("tokens"),
            "sequences": args.sequences, "sanity": sanity, "seconds": time.time() - started,
            "gpu": torch.cuda.get_device_name(0)}
    (args.out / f"{name}.json").write_text(json.dumps({**meta, "matrices": report}, indent=1) + "\n")
    torch.save({**meta, "arrays": arrays}, args.out / f"{name}.pt")
    print(json.dumps({**meta, "name": name}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("items", nargs="+")
    parser.add_argument("--sequences", type=int, default=2048)
    parser.add_argument("--micro", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    for item in args.items:
        arm, step = item.rsplit(":", 1)
        measure(arm, step, args, device, stream)


if __name__ == "__main__":
    main()

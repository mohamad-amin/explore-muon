"""Which part of an actual update changes the gradient along flat directions?

At kept step t: frame per hidden matrix (U, V from held-out basis sequences, as in measure_frame.py) and the applied
update dW = W[t+1] - W[t]. The hidden-matrix update is split in each frame by K-FAC curvature rank of its pairs:
stiff (top 1% of pairs by lam_B lam_C), middle (next 9%), flat (bottom 90%); "aux" is the change of every other
parameter (embeddings, head, norm gains). The first-order gradient change caused by each part is H dW_part (full
Hessian, double backward) and G dW_part (Gauss-Newton only), measured on two independent sequence sets A and B and
projected into the frames; bin sums of the A x B products are unbiased for the squared change. They are compared
per curvature bin (K-FAC curvature of the direction whose gradient changes) with the squared signal s there.

usage: measure_step_coupling.py OUT_DIR ARM_DIR:STEP [...] [--sequences 256]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.func import functional_call, jvp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402

OFFSET = 2 * 1048576 + 40000 * 512
PARTS = ("stiff", "middle", "flat", "aux")


def batches(stream, first, count, micro, T, device):
    for start in range(first, first + count, micro):
        yield stream.batch(OFFSET + start * T, min(micro, first + count - start), T, device)


def products(model, x, y, direction):
    """(H d, G d) for the mean token loss, as dicts over all parameters (d given for any subset)."""
    names = [n for n, _ in model.named_parameters()]
    params = dict(model.named_parameters())
    base = {**params, **dict(model.named_buffers())}
    keys = list(direction)
    with P.explicit_attention():
        # full Hessian-vector product by double backward
        loss = P.token_losses(model(x), y).mean()
        grads = torch.autograd.grad(loss, [params[n] for n in names], create_graph=True)
        dot = sum((g * direction[n]).sum() for g, n in zip(grads, names) if n in direction)
        hv = torch.autograd.grad(dot, [params[n] for n in names], allow_unused=True)
        hvp = {n: (h.detach() if h is not None else torch.zeros_like(params[n])) for n, h in zip(names, hv)}
        # Gauss-Newton-vector product: J^T H_z (J d)
        def f(*weights):
            return functional_call(model, {**base, **dict(zip(keys, weights))}, (x,))
        with torch.no_grad():
            _, dz = jvp(f, tuple(base[k] for k in keys), tuple(direction[k] for k in keys))
        logits = model(x)
        p = torch.softmax(logits.float(), -1)
        dz = dz.float()
        hz = p * (dz - (p * dz).sum(-1, keepdim=True))
        gv = torch.autograd.grad(logits, [params[n] for n in names], grad_outputs=hz.to(logits.dtype) / y.numel(),
                                 allow_unused=True)
        gnp = {n: (g.detach() if g is not None else torch.zeros_like(params[n])) for n, g in zip(names, gv)}
    return hvp, gnp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("items", nargs="+")
    parser.add_argument("--sequences", type=int, default=256)
    parser.add_argument("--basis-sequences", type=int, default=256)
    parser.add_argument("--micro", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    for item in args.items:
        arm, step = item.rsplit(":", 1)
        step = int(step)
        started = time.time()
        kept = Path(arm) / "scientific" / "kept"
        model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
        after = torch.load(kept / f"step{step + 1:06d}_weights.pt", map_location="cpu", weights_only=False)["model"]
        T = model.config.seq_len
        layers = P.hidden_linears(model)
        names = list(layers)
        keys = dict(zip(names, P.parameter_keys(model, names)))
        cuda_gen = torch.Generator(device=device).manual_seed(0)
        # frame
        recorder = P.Recorder(model)
        sums = {n: [0, 0, 0] for n in names}
        for x, y in batches(stream, 0, args.basis_sequences, 8, T, device):
            P.gradient_passes(model, recorder, x, y, cuda_gen, draws=1)
            for n in names:
                c, b, count = P.second_moments(recorder.inputs[n], T * recorder.errors[n][1])
                sums[n][0] = sums[n][0] + c; sums[n][1] = sums[n][1] + b; sums[n][2] += count
        recorder.remove()
        frames = {}
        for n in names:
            lam_c, v = P.eigenbasis(sums[n][0], sums[n][2])
            lam_b, u = P.eigenbasis(sums[n][1], sums[n][2])
            frames[n] = (u.float(), v.float(), torch.outer(lam_b, lam_c))
        del sums
        # split the applied update
        parts = {p: {} for p in PARTS}
        for pname, value in model.named_parameters():
            delta = after[pname].to(device=device, dtype=torch.float32) - value.detach()
            hidden = next((n for n in names if keys[n] == pname), None)
            if hidden is None:
                parts["aux"][pname] = delta
                continue
            u, v, kfac = frames[hidden]
            coeff = u.T @ delta @ v
            flat_rank = torch.argsort(torch.argsort(kfac.flatten(), descending=True)).view_as(kfac)
            total = kfac.numel()
            masks = {"stiff": flat_rank < 0.01 * total,
                     "middle": (flat_rank >= 0.01 * total) & (flat_rank < 0.10 * total),
                     "flat": flat_rank >= 0.10 * total}
            for part, mask in masks.items():
                parts[part][pname] = u @ (coeff * mask.to(coeff.dtype)) @ v.T
        # gradient changes on two independent sequence sets, projected into the frames
        results = {}
        for part in PARTS:
            proj = {kind: {"A": {}, "B": {}} for kind in ("H", "G")}
            for label, first in (("A", args.basis_sequences), ("B", args.basis_sequences + args.sequences)):
                acc_h = {n: 0 for n in names}; acc_g = {n: 0 for n in names}; count = 0
                for x, y in batches(stream, first, args.sequences, args.micro, T, device):
                    hvp, gnp = products(model, x, y, parts[part])
                    for n in names:
                        acc_h[n] = acc_h[n] + hvp[keys[n]] * x.shape[0]
                        acc_g[n] = acc_g[n] + gnp[keys[n]] * x.shape[0]
                    count += x.shape[0]
                for n in names:
                    u, v, _ = frames[n]
                    proj["H"][label][n] = (u.T @ (acc_h[n] / count) @ v).cpu()
                    proj["G"][label][n] = (u.T @ (acc_g[n] / count) @ v).cpu()
            results[part] = proj
        out = {"meta": {"arm": arm, "step": step, "sequences": args.sequences, "seconds": time.time() - started},
               "kfac": {n: frames[n][2].float().cpu() for n in names},
               "update_energy": {part: float(sum((d * d).sum() for d in parts[part].values())) for part in PARTS},
               "products": results}
        name = f"{Path(arm).name}_step{step:06d}"
        torch.save(out, args.out / f"{name}.pt")
        print(json.dumps({**out["meta"], "name": name, "update_energy": out["update_energy"]}), flush=True)


if __name__ == "__main__":
    main()

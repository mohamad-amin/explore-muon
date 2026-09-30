"""Where the top of the Gauss-Newton spectrum lives when all parameters are included (2026-09-30 16:0x CDT; after
arXiv 2607.21716, whose top ~vocab-size eigenvalues belong to the unembedding). Our GN products and sharpness readings so
far were body-only (the 48 hidden matrices); the auxiliary parameters (embedding, head, norms, positions) are two thirds
of the model and are trained by AdamW.

torchrun with 4 ranks. At a kept state (ARM:STEP), on --sequences held-out training sequences per rank:
  - Lanczos (--krylov steps, full reorthogonalization, fixed random start) on the exact GN of the mean token loss over
    (a) all parameters and (b) the body matrices only;
  - the top --top Ritz values of each, and for (a) the share of each top Ritz vector's energy in each parameter group
    (embed, position, head, norms, and the body by kind), plus the share weighted over the top Ritz vectors;
  - the Rayleigh quotient of the head-only and body-only parts of the top vector.

usage: .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 full_spectrum_probe.py OUT_JSON ARM:STEP \
           [ARM:STEP ...]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream, token_views  # noqa: E402
from newton_train import HELD_OUT_TRAIN, GNProduct  # noqa: E402


def group_of(key):
    if key.startswith("blocks.") and key.endswith(".weight") and len(key.split(".")) == 4:
        kind = key.split(".")[2]
        if kind in ("q", "k", "v", "o", "up", "down"):
            return "body." + kind
    if key.startswith("head."):
        return "head"
    if key.startswith("embed."):
        return "embed"
    if key.startswith("position."):
        return "position"
    return "norms"


def lanczos(op, n, k, device, seed=0):
    start = torch.randn(n, generator=torch.Generator().manual_seed(seed)).to(device)
    V = torch.zeros(k, n, device=device)
    V[0] = start / start.norm()
    alphas, betas = [], []
    for j in range(k):
        w = op(V[j])
        alphas.append(float(w @ V[j]))
        for _ in range(2):
            w -= V[:j + 1].T @ (V[:j + 1] @ w)
        if j + 1 == k:
            break
        betas.append(float(w.norm()))
        V[j + 1] = w / betas[-1]
    Tk = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    off = torch.tensor(betas, dtype=torch.float64)
    values, Y = torch.linalg.eigh(Tk + torch.diag(off, 1) + torch.diag(off, -1))
    return values, Y, V


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("items", nargs="+")
    parser.add_argument("--sequences", type=int, default=64)
    parser.add_argument("--krylov", type=int, default=48)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--micro", type=int, default=8)
    args = parser.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    results = {}
    for item in args.items:
        started = time.time()
        arm, step = item.rsplit(":", 1)
        model, saved = P.load_checkpoint(Path(arm) / "scientific" / "kept" / f"step{int(step):06d}.pt", device)
        config = dict(saved["config"])
        del saved
        T = model.config.seq_len
        train = TokenStream(str(REPO / config["train_pattern"]))
        xs, _ = token_views(train.device_tokens(HELD_OUT_TRAIN + 200_000_000 + rank * args.sequences * T,
                                                args.sequences * T + 1, device), 0, args.sequences * T, T)
        named = dict(model.named_parameters())
        entry = {}
        for scope in ("all", "body"):
            keys = [k for k in named if scope == "all" or (k.startswith("blocks.") and named[k].ndim == 2)]
            sizes = [named[k].numel() for k in keys]
            gn = GNProduct(model, keys, xs, world * args.sequences * T, args.micro)

            def op(v):
                return torch.cat([o.reshape(-1) for o in gn([part.view_as(named[k]) for part, k in zip(v.split(sizes), keys)])])
            values, Y, V = lanczos(op, sum(sizes), args.krylov, device)
            top_vals = values[-args.top:].flip(0).tolist()
            scope_entry = {"top_ritz": top_vals}
            if scope == "all":
                groups = sorted({group_of(k) for k in keys})
                offsets = [0]
                for s_ in sizes:
                    offsets.append(offsets[-1] + s_)
                shares = []
                for j in range(1, args.top + 1):
                    z = (Y[:, -j].to(device, torch.float32) @ V)
                    energy = {g: 0.0 for g in groups}
                    for i, k in enumerate(keys):
                        energy[group_of(k)] += float(z[offsets[i]:offsets[i + 1]].square().sum())
                    total = sum(energy.values())
                    shares.append({g: e / total for g, e in energy.items()})
                scope_entry["group_shares_top"] = shares
                weights = [float(v) for v in top_vals]
                scope_entry["group_share_weighted"] = {g: sum(w * sh[g] for w, sh in zip(weights, shares)) / sum(weights)
                                                       for g in groups}
                scope_entry["parameter_share"] = {g: sum(sz for k, sz in zip(keys, sizes) if group_of(k) == g) / sum(sizes)
                                                  for g in groups}
            del V
            entry[scope] = scope_entry
        entry["seconds"] = time.time() - started
        results[item] = entry
        if rank == 0:
            print(item, "all top", [round(v, 1) for v in entry["all"]["top_ritz"][:6]], "body top",
                  [round(v, 1) for v in entry["body"]["top_ritz"][:4]], flush=True)
            print("   weighted group shares of the top vectors",
                  {g: round(v, 3) for g, v in entry["all"]["group_share_weighted"].items()}, flush=True)
        del model
        torch.cuda.empty_cache()
    if rank == 0:
        args.out.write_text(json.dumps(results, indent=1) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

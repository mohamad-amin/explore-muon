"""GN-PD with the exact GN restricted to blocks: within-matrix structure vs cross-matrix coupling (20:3x CDT).

The stored-token-factor block operator of blockgnpd_probe.py is noisy (one sampled label per token; it improves with more
tokens and always picks the largest damping), so it cannot say whether full GN-PD's 2x floor gain over PD lives within
matrices or across them. Here each block's operator is the exact GN (forward-mode + reverse-mode products with the full
softmax Hessian, one_step_gn.GaussNewton on the block's weights only), on the same 256 curvature sequences as
gnpd_probe.py:
  matrix   48 blocks, one per hidden matrix
  layer    8 blocks, the 6 matrices of one layer
GN-PD(p) per block: w = (G_b + mu_b)^-p b_b, polar per matrix, z = (G_b + mu_b)^-p P_b (block Lanczos, --krylov steps,
one run from b shared by every damping, one run from P per damping); mu_b = d rho_b; each matrix rescaled to Muon's
norm; scored cross-fitted like gnpd_probe.py on the same inputs and held-out halves.

usage: exactblock_gnpd_probe.py OUT_JSON ARM_DIR:STEP [--blocks matrix|layer] [--inputs momentum_g1M] [--krylov 16]
"""
import argparse
import json
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402
from one_step_gn import CURV, EVAL, GRAD, Flat, GaussNewton, lanczos, muon_norm, polar  # noqa: E402
from one_step_blockgn import cross_fit, halves_score  # noqa: E402
from valley_rescore import G16, fresh_gradient  # noqa: E402
from gnpd_probe import matrix_power  # noqa: E402

POWER = 0.25
DAMPINGS = (1e-3, 1e-4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--blocks", default="matrix", choices=("matrix", "layer"))
    parser.add_argument("--inputs", default="momentum_g1M")
    parser.add_argument("--krylov", type=int, default=16)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    parser.add_argument("--eval-sequences", type=int, default=512)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    started = time.time()
    arm, step = args.item.rsplit(":", 1)
    step = int(step)
    kept = Path(arm) / "scientific" / "kept"
    model, _ = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
    T = model.config.seq_len
    layers = P.hidden_linears(model)
    names = list(layers)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for n in names:
        layers[n].weight.requires_grad_(True)
    if args.blocks == "matrix":
        blocks = [[n] for n in names]
    else:
        prefixes = OrderedDict()
        for n in names:
            prefixes.setdefault(n.split(".")[0], []).append(n)
        blocks = list(prefixes.values())
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    curv = [stream.batch(CURV + first * T, 8, T, device) for first in range(0, args.curvature_sequences, 8)]
    evals = [stream.batch(EVAL + first * T, 8, T, device) for first in range(0, args.eval_sequences, 8)]
    halves = [evals[:len(evals) // 2], evals[len(evals) // 2:]]
    saved = torch.load(kept / f"step{step:06d}.pt", map_location="cpu", weights_only=False)
    beta = saved["config"]["muon_momentum"]
    buffers = OrderedDict((n, saved["optimizer"]["state"][i]["momentum_buffer"].to(device).float()) for i, n in enumerate(names))
    del saved
    result = {"arm": arm, "step": step, "blocks": args.blocks, "krylov": args.krylov, "power": POWER, "inputs": {}}
    sizes = {"g4M": (GRAD, 8192), "g16M": (G16, 32768), "g1M": (GRAD, 2048)}
    for label in args.inputs.split(","):
        t0 = time.time()
        if label.startswith("momentum"):
            offset, sequences = sizes[label.split("_")[1]]
            g = fresh_gradient(model, names, stream, offset, sequences, T, device)
            b = OrderedDict((n, beta * buffers[n] + g[n]) for n in names)
        else:
            offset, sequences = sizes[label]
            b = fresh_gradient(model, names, stream, offset, sequences, T, device)
        directions = {d: OrderedDict() for d in DAMPINGS}
        for block in blocks:
            flat = Flat(OrderedDict((n, layers[n].weight) for n in block))
            op = GaussNewton(model, block, curv, flat)
            b_flat = flat.flat(OrderedDict((n, b[n]) for n in block))
            V, Tk = lanczos(op, b_flat, args.krylov)
            rho = float(Tk[0, 0])
            for d in DAMPINGS:
                mu = d * rho
                w = flat.dict(matrix_power(V, Tk, float(b_flat.norm()), POWER, mu))
                p_flat = flat.flat(OrderedDict((n, polar(w[n])) for n in block))
                V2, T2 = lanczos(op, p_flat, args.krylov)
                z = flat.dict(matrix_power(V2, T2, float(p_flat.norm()), POWER, mu))
                del V2
                for n in block:
                    directions[d][n] = -z[n] * (muon_norm(z[n].shape) / z[n].norm().clamp_min(1e-30))
            del V
            print(json.dumps({"block": block[0] if len(block) == 1 else block[0].split(".")[0], "rho": round(rho, 4),
                              "seconds": round(time.time() - t0)}), flush=True)
        candidates = {f"{args.blocks}gnpd_p{POWER:g}_d{d:g}": directions[d] for d in DAMPINGS}
        scored = {k: halves_score(model, halves, dd) for k, dd in candidates.items()}
        entry = {"cross_fit": cross_fit(scored), "raw": scored}
        result["inputs"][label] = entry
        args.out.write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({"item": args.item, "input": label, "blocks": args.blocks,
                          "cross_fitted_x1e3": round(entry["cross_fit"]["cross_fitted"] * 1e3, 3),
                          "pick": entry["cross_fit"]["picks"]["fit0"]["label"], "seconds": round(time.time() - t0)}), flush=True)
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main()

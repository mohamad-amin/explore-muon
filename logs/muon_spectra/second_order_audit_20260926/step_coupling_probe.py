"""The structure of the coupling: the exact GN curvature of an optimizer's actual step, decomposed into pairs of
matrices (2026-09-29 01:44 CDT; MUON_CASE "Question: the structure of the coupling").

At a kept state N with next weights W_{N+1}, the step of hidden matrix l is D_l = W_{N+1,l} - W_{N,l}. For softmax
cross-entropy the GN quadratic of a logit change dz is Var_p(dz) per token, so for the pieces of the step
  Q_lm = E_tokens[ Cov_p(J_l D_l, J_m D_m) ]   (J_l D_l: the logit change from matrix l's step alone, one JVP each)
  a_l  = E_tokens[ (p - onehot(y)) . J_l D_l ]  (matrix l's first-order loss change)
so the GN model of the whole step is L + sum_l a_l + (1/2) sum_lm Q_lm, the diagonal is each matrix's own curvature
cost, and the off-diagonals are the interactions (coherence = sum Q / trace Q). Held-out validation sequences; per
sequence, the p-weighted centred logit changes of all 48 matrices are held at once and their Gram accumulated.

usage: step_coupling_probe.py OUT_JSON ARM_DIR:STEP [--sequences 24]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.func import functional_call, jvp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402
from one_step_gn import CURV  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--sequences", type=int, default=24)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    started = time.time()
    arm, step = args.item.rsplit(":", 1)
    step = int(step)
    kept = Path(arm) / "scientific" / "kept"
    model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
    config = saved["config"]
    del saved
    nxt, _ = P.load_checkpoint(kept / f"step{step + 1:06d}_weights.pt", device, config["model"])
    layers = P.hidden_linears(model)
    next_layers = P.hidden_linears(nxt)
    names = list(layers)
    keys = P.parameter_keys(model, names)
    steps = {n: (next_layers[n].weight - layers[n].weight).detach() for n in names}
    del nxt, next_layers
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    T = model.config.seq_len
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    L = len(names)
    gram = torch.zeros(L, L, dtype=torch.float64, device=device)
    slope = torch.zeros(L, dtype=torch.float64, device=device)
    count = 0
    for s in range(args.sequences):
        x, y = stream.batch(CURV + s * T, 1, T, device)
        with torch.no_grad(), P.explicit_attention():
            logits = model(x).float()
        p = torch.softmax(logits, dim=-1)
        sqrt_p = p.sqrt()
        onehot_minus = p.clone()
        onehot_minus.scatter_add_(-1, y.unsqueeze(-1), -torch.ones_like(y, dtype=p.dtype).unsqueeze(-1))
        W = torch.empty(L, logits.numel(), device=device, dtype=torch.float32)   # p-weighted centred logit changes
        for l, (name, key) in enumerate(zip(names, keys)):
            def logits_of(weight, key=key):
                return functional_call(model, {**base, key: weight}, (x,))
            with P.explicit_attention():
                _, dz = jvp(logits_of, (base[key],), (steps[name].to(base[key].dtype),))
            dz = dz.float()
            slope[l] += (onehot_minus * dz).sum().double()
            centred = dz - (p * dz).sum(-1, keepdim=True)
            W[l] = (sqrt_p * centred).reshape(-1)
            del dz, centred
        gram += (W @ W.T).double()
        count += x.numel()
        del W
    Q = (gram / count).cpu()
    a = (slope / count).cpu()
    diag = Q.diag()
    total = float(Q.sum())
    corr = Q / (diag.sqrt()[:, None] * diag.sqrt()[None, :]).clamp_min(1e-30)
    blocks = sorted({n.split(".")[0] for n in names})
    kinds = []
    for n in names:
        if n.split(".")[1] not in kinds:
            kinds.append(n.split(".")[1])
    block_of = torch.tensor([blocks.index(n.split(".")[0]) for n in names])
    kind_of = torch.tensor([kinds.index(n.split(".")[1]) for n in names])
    same_block = block_of[:, None] == block_of[None, :]
    same_kind = kind_of[:, None] == kind_of[None, :]
    off = ~torch.eye(L, dtype=torch.bool)
    parts = {"diagonal": float(diag.sum()),
             "same_layer_offdiag": float(Q[same_block & off].sum()),
             "same_kind_other_layers": float(Q[same_kind & ~same_block].sum()),
             "other_kind_other_layers": float(Q[~same_kind & ~same_block].sum())}
    layer_Q = torch.zeros(len(blocks), len(blocks), dtype=torch.float64)
    kind_Q = torch.zeros(len(kinds), len(kinds), dtype=torch.float64)
    for i in range(L):
        for j in range(L):
            layer_Q[block_of[i], block_of[j]] += Q[i, j]
            kind_Q[kind_of[i], kind_of[j]] += Q[i, j]
    result = {"arm": arm, "step": step, "sequences": args.sequences, "names": names, "blocks": blocks, "kinds": kinds,
              "Q": Q.tolist(), "a": a.tolist(), "corr": corr.tolist(), "layer_Q": layer_Q.tolist(), "kind_Q": kind_Q.tolist(),
              "total_q": total, "total_a": float(a.sum()), "coherence": total / float(diag.sum()), "parts": parts,
              "c_star_gn": float(-a.sum() / total) if total > 0 else None, "seconds": time.time() - started}
    args.out.write_text(json.dumps(result) + "\n")
    print(json.dumps({"item": args.item, "coherence": round(result["coherence"], 2),
                      "parts_share": {k: round(v / total, 3) for k, v in parts.items()},
                      "c_star_gn": round(result["c_star_gn"], 3) if result["c_star_gn"] else None,
                      "median_offdiag_corr": round(float(corr[off].median()), 3), "seconds": round(result["seconds"])}), flush=True)


if __name__ == "__main__":
    main()

"""The coupling structure of saved one-step directions (exact GN, K-FAC, Muon, PD from the same gradient), in the same
decomposition as the optimizers' actual steps (2026-09-29 02:16 CDT; follows step_coupling_probe.py).

one_step_gn.py saved, per state and input (fresh 1M / 4M gradient or the momentum), the directions of Muon, PD α ¼,
K-FAC and the exact damped GN (`*_directions.pt`, bf16). For each, the GN curvature of the direction's pieces over
the 48 hidden matrices, Q_lm = E_tokens[Cov_p(J_l D_l, J_m D_m)], and first-order terms a_l, on held-out sequences.
What complementary pieces look like: coherence sum Q / trace Q, the split own / same layer / same kind other layers /
other kinds other layers, the collective mode's share and participation ratio, the per-layer curvature and gain shares.

usage: direction_coupling_saved.py OUT_JSON ARM_DIR:STEP --directions FILE.pt [--input g4M] [--labels a,b] [--sequences 24]
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


def analyze(model, base, names, keys, steps, batches, device):
    L = len(names)
    gram = torch.zeros(L, L, dtype=torch.float64, device=device)
    slope = torch.zeros(L, dtype=torch.float64, device=device)
    count = 0
    for x, y in batches:
        with torch.no_grad(), P.explicit_attention():
            logits = model(x).float()
        p = torch.softmax(logits, dim=-1)
        sqrt_p = p.sqrt()
        residual = p.clone()
        residual.scatter_add_(-1, y.unsqueeze(-1), -torch.ones_like(y, dtype=p.dtype).unsqueeze(-1))
        W = torch.empty(L, logits.numel(), device=device, dtype=torch.float32)
        for l, (name, key) in enumerate(zip(names, keys)):
            def logits_of(weight, key=key):
                return functional_call(model, {**base, key: weight}, (x,))
            with P.explicit_attention():
                _, dz = jvp(logits_of, (base[key],), (steps[name].to(base[key].dtype),))
            dz = dz.float()
            slope[l] += (residual * dz).sum().double()
            W[l] = (sqrt_p * (dz - (p * dz).sum(-1, keepdim=True))).reshape(-1)
            del dz
        gram += (W @ W.T).double()
        count += x.numel()
        del W
    Q = (gram / count).cpu()
    a = (slope / count).cpu()
    if float(a.sum()) > 0:          # orient every direction as a descent direction (a < 0); Q is sign-invariant
        a = -a
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
    parts = {"diagonal": float(diag.sum()), "same_layer_offdiag": float(Q[same_block & off].sum()),
             "same_kind_other_layers": float(Q[same_kind & ~same_block].sum()),
             "other_kind_other_layers": float(Q[~same_kind & ~same_block].sum())}
    ev = torch.linalg.eigvalsh(Q.double()).flip(0)
    rows = Q.sum(1)
    layer_curv = [float(rows[block_of == b].sum() / total) for b in range(len(blocks))]
    layer_gain = [float(a[block_of == b].sum() / a.sum()) for b in range(len(blocks))]
    return {"names": names, "Q": Q.tolist(), "a": a.tolist(), "corr": corr.tolist(), "total_q": total, "total_a": float(a.sum()),
            "coherence": total / float(diag.sum()), "parts": parts, "top_mode": float(ev[0] / ev.sum()),
            "pr": float(ev.sum() ** 2 / (ev ** 2).sum()), "median_offdiag_corr": float(corr[off].median()),
            "min_offdiag_corr": float(corr[off].min()), "layer_curv_share": layer_curv, "layer_gain_share": layer_gain,
            "quality": float(a.sum() ** 2 / (2 * total)) if total > 0 else None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--directions", type=Path, required=True)
    parser.add_argument("--input", default="g4M")
    parser.add_argument("--labels", default="")
    parser.add_argument("--sequences", type=int, default=24)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    started = time.time()
    arm, step = args.item.rsplit(":", 1)
    step = int(step)
    kept = Path(arm) / "scientific" / "kept"
    model, _ = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
    layers = P.hidden_linears(model)
    names = list(layers)
    keys = P.parameter_keys(model, names)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    base = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    T = model.config.seq_len
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    batches = [stream.batch(CURV + s * T, 1, T, device) for s in range(args.sequences)]
    saved = torch.load(args.directions, map_location="cpu", weights_only=False)["directions"][args.input]
    labels = args.labels.split(",") if args.labels else list(saved)
    results = {}
    for label in labels:
        steps = {n: saved[label][n].to(device).float() for n in names}
        r = analyze(model, base, names, keys, steps, batches, device)
        results[label] = r
        print(json.dumps({"label": label, "coherence": round(r["coherence"], 2),
                          "parts_share": {k: round(v / r["total_q"], 3) for k, v in r["parts"].items()},
                          "top_mode": round(r["top_mode"], 3), "pr": round(r["pr"], 2),
                          "median_corr": round(r["median_offdiag_corr"], 3), "min_corr": round(r["min_offdiag_corr"], 3),
                          "layer1_curv/gain": [round(r["layer_curv_share"][0], 3), round(r["layer_gain_share"][0], 3)],
                          "quality": r["quality"]}), flush=True)
    args.out.write_text(json.dumps({"arm": arm, "step": step, "input": args.input, "directions": str(args.directions),
                                    "results": results, "seconds": time.time() - started}) + "\n")


if __name__ == "__main__":
    main()

"""Exact GN Gram between the 48 per-matrix pieces of saved update directions (one_step_gn.py *_directions.pt).

For each chosen (input, label): a_i = <g, D_i> and Q_ij = <D_i, G D_j> on held-out sequences, with D_i the direction
restricted to matrix i. Negative off-diagonal Q (anti-correlated output changes) is cancellation between matrices;
positive is coherence. Writes a JSON next to the directions file and prints the coherence summaries.

usage: gram_saved.py DIRECTIONS_PT CHECKPOINT_PT [--inputs g4M,momentum] [--sequences 128]
"""
import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402
from one_step_gn import EVAL, gram  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directions", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--inputs", default="g4M,momentum")
    parser.add_argument("--sequences", type=int, default=128)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    model, _ = P.load_checkpoint(args.checkpoint, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    T = model.config.seq_len
    names = list(P.hidden_linears(model))
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    batches = [stream.batch(EVAL + first * T, 1, T, device) for first in range(args.sequences)]
    saved = torch.load(args.directions, map_location="cpu", weights_only=False)["directions"]
    out = {"names": names}
    for source in args.inputs.split(","):
        if source not in saved:
            continue
        for label, direction in saved[source].items():
            pieces = [OrderedDict([(n, direction[n].to(device=device, dtype=torch.float32))]) for n in names]
            a, Q = gram(model, batches, pieces)
            a, Q = np.array(a), np.array(Q)
            ones = np.ones(len(a))
            d = np.sqrt(np.diag(Q))
            corr = Q / np.outer(d, d)
            kinds = np.array([n.split(".")[1] for n in names])
            layer = np.array([int(n[5:7]) for n in names])
            same_kind = (kinds[:, None] == kinds[None, :]) & ~np.eye(len(a), dtype=bool)
            same_layer = (layer[:, None] == layer[None, :]) & ~np.eye(len(a), dtype=bool)
            summary = {"coherence_matrix": float(ones @ Q @ ones / np.trace(Q)),
                       "mean_corr_same_kind_across_layers": float(corr[same_kind].mean()),
                       "mean_corr_same_layer_across_kinds": float(corr[same_layer].mean()),
                       "mean_corr_other": float(corr[~same_kind & ~same_layer & ~np.eye(len(a), dtype=bool)].mean()),
                       "per_kind_coherence": {k: float(Q[np.ix_(kinds == k, kinds == k)].sum() / np.trace(Q[np.ix_(kinds == k, kinds == k)]))
                                              for k in P.KINDS},
                       "best_decrease": float((ones @ a) ** 2 / (2 * ones @ Q @ ones)) if ones @ a < 0 else 0.0}
            out[f"{source}:{label}"] = {"a": a.tolist(), "Q": Q.tolist(), "summary": summary}
            print(json.dumps({"input": source, "label": label, **summary}), flush=True)
    target = args.directions.with_name(args.directions.stem.replace("_directions", "") + "_gram.json")
    target.write_text(json.dumps(out) + "\n")


if __name__ == "__main__":
    main()

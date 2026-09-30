"""Frame-diagonal vs exact GN curvature of the actual update's parts (one_step_split.json + the gap map's frame files).

For each (run, step) in one_step_split.json: the exact first-order terms a and GN quadratic form Q of the stiff (top 1%
of K-FAC pairs), middle (next 9%) and flat parts of W[t+1] - W[t], against their frame-diagonal counterparts from
frame/ (sum over pairs of the exact per-pair diagonal h times D^2). Writes split_vs_diag.json.

usage: split_vs_diag.py
"""
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def main():
    split = json.loads((HERE / "one_step_split.json").read_text())
    out = {}
    for key, e in split.items():
        arm, step = key.rsplit(":", 1)
        d = torch.load(HERE / "frame" / f"{arm}_step{int(step):06d}.pt", weights_only=False)
        qd, ad = np.zeros(3), np.zeros(3)
        for m in d["matrices"].values():
            kfac = torch.outer(m["lam_B"].double(), m["lam_C"].double())
            rank = torch.argsort(torch.argsort(kfac.flatten(), descending=True)).view_as(kfac)
            total = kfac.numel()
            masks = [rank < 0.01 * total, (rank >= 0.01 * total) & (rank < 0.10 * total), rank >= 0.10 * total]
            D, g, h = m["update"].double(), m["mean"].double(), m["exact"].double()
            for i, mask in enumerate(masks):
                ad[i] += float((g * D)[mask].sum())
                qd[i] += float((h * D * D)[mask].sum())
        a, Q = np.array(e["a"]), np.array(e["Q"])
        dq = np.sqrt(np.diag(Q))
        out[key] = {"a_exact": a.tolist(), "a_diag": ad.tolist(), "Q_exact_diag": np.diag(Q).tolist(), "Q_diag": qd.tolist(),
                    "ratio_parts": (np.diag(Q) / qd).tolist(), "ratio_whole": float(Q.sum() / qd.sum()),
                    "corr": (Q / np.outer(dq, dq)).tolist(), "taken": e["taken"], "best_single": e["best_single"],
                    "best_single_scale": e["best_single_scale"], "best_separate": e.get("best_separate"),
                    "best_separate_scales": e.get("best_separate_scales"),
                    "diag_best_scale": float(-ad.sum() / qd.sum())}
        print(key, round(out[key]["ratio_whole"], 1), flush=True)
    (HERE / "split_vs_diag.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()

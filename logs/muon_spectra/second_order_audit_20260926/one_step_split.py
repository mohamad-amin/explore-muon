"""What does one shared step size cost? The actual update of step t+1 (W[t+1] - W[t], hidden matrices; everything
else held at W[t]) is split in each matrix's Kronecker frame by K-FAC curvature rank: stiff (top 1% of pairs),
middle (next 9%), flat (bottom 90%). On held-out sequences, forward-mode passes give the first-order terms a_i =
<g, D_i> and the exact GN quadratic form Q_ij = <D_i, G D_j> (from q(D_i + D_j)). The one-step quadratic model
    L(c) = sum_i c_i a_i + (1/2) sum_ij c_i c_j Q_ij
is evaluated at c = 1 (the step actually taken), at the best single scale c_i = c, and at the best separate
scales per part (c = -Q^-1 a, when Q is positive definite). A one-step picture of the stiff/flat compromise,
not a training result.

usage: one_step_split.py OUT_JSON ARM_DIR:STEP [...] [--sequences 256]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402

BASIS = 2 * 1048576 + 70000 * 512
EVAL = 2 * 1048576 + 72000 * 512
PARTS = ("stiff", "middle", "flat")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("items", nargs="+")
    parser.add_argument("--sequences", type=int, default=256)
    parser.add_argument("--basis-sequences", type=int, default=256)
    parser.add_argument("--micro", type=int, default=8)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    results = {}
    for item in args.items:
        arm, step = item.rsplit(":", 1)
        step = int(step)
        kept = Path(arm) / "scientific" / "kept"
        model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
        after = torch.load(kept / f"step{step + 1:06d}_weights.pt", map_location="cpu", weights_only=False)["model"]
        T = model.config.seq_len
        names = list(P.hidden_linears(model))
        keys = dict(zip(names, P.parameter_keys(model, names)))
        gen = torch.Generator(device=device).manual_seed(0)
        recorder = P.Recorder(model)
        sums = {n: [0, 0, 0] for n in names}
        for first in range(0, args.basis_sequences, args.micro):
            x, y = stream.batch(BASIS + first * T, args.micro, T, device)
            P.gradient_passes(model, recorder, x, y, gen, draws=1)
            for n in names:
                c, b, count = P.second_moments(recorder.inputs[n], T * recorder.errors[n][1])
                sums[n][0] = sums[n][0] + c; sums[n][1] = sums[n][1] + b; sums[n][2] += count
        recorder.remove()
        parts = {p: {} for p in PARTS}
        for n in names:
            lam_c, v = P.eigenbasis(sums[n][0], sums[n][2])
            lam_b, u = P.eigenbasis(sums[n][1], sums[n][2])
            u, v = u.float(), v.float()
            kfac = torch.outer(lam_b, lam_c)
            delta = after[keys[n]].to(device=device, dtype=torch.float32) - saved["model"][keys[n]].to(device=device, dtype=torch.float32)
            coeff = u.T @ delta @ v
            rank = torch.argsort(torch.argsort(kfac.flatten(), descending=True)).view_as(kfac)
            total = kfac.numel()
            masks = {"stiff": rank < 0.01 * total, "middle": (rank >= 0.01 * total) & (rank < 0.10 * total),
                     "flat": rank >= 0.10 * total}
            for p, mask in masks.items():
                parts[p][n] = u @ (coeff * mask.to(coeff.dtype)) @ v.T
        del sums
        batches = [stream.batch(EVAL + first * T, args.micro, T, device) for first in range(0, args.sequences, args.micro)]

        def terms(direction):
            first = q = 0.0
            for x, y in batches:
                t = P.directional_terms(model, x, y, direction)
                first += float(t["first"].sum()); q += float(t["q"].sum())
            return first / args.sequences, q / args.sequences

        a, Q = np.zeros(3), np.zeros((3, 3))
        for i, p in enumerate(PARTS):
            a[i], Q[i, i] = terms(parts[p])
        for i in range(3):
            for j in range(i + 1, 3):
                combined = {n: parts[PARTS[i]][n] + parts[PARTS[j]][n] for n in names}
                _, q = terms(combined)
                Q[i, j] = Q[j, i] = 0.5 * (q - Q[i, i] - Q[j, j])
        model_loss = lambda c: float(c @ a + 0.5 * c @ Q @ c)
        ones = np.ones(3)
        single = -(ones @ a) / (ones @ Q @ ones)
        entry = {"a": a.tolist(), "Q": Q.tolist(), "taken": model_loss(ones), "best_single_scale": single,
                 "best_single": model_loss(single * ones)}
        try:
            eig = np.linalg.eigvalsh(Q)
            if eig.min() > 0:
                c = -np.linalg.solve(Q, a)
                entry.update(best_separate_scales=c.tolist(), best_separate=model_loss(c))
            entry["Q_eigenvalues"] = eig.tolist()
        except np.linalg.LinAlgError:
            pass
        # stiff relaxed (Newton on the stiff part given the others at their actual size), others as taken
        c = np.array([-(a[0] + Q[0, 1] + Q[0, 2]) / Q[0, 0], 1.0, 1.0])
        entry.update(stiff_relaxed_scale=c[0], stiff_relaxed=model_loss(c))
        results[f"{Path(arm).name}:{step}"] = entry
        print(json.dumps({"item": item.split('/')[-1], **{k: v for k, v in entry.items() if k not in ('Q',)}}), flush=True)
        args.out.write_text(json.dumps(results, indent=1) + "\n")


if __name__ == "__main__":
    main()

"""Per-entry signal-to-noise of the gradient in SOAP's frame vs the GN output factor's frame, across batch sizes.

Why SOAP's normalization (its output side above all) carries a gain that grows with batch size (2026-09-28 04:1x CDT):
per-entry normalization is informative where an entry's signal exceeds its noise, and that fraction grows with the batch.

At a kept state, for each hidden matrix, K micro-batch gradients G_i (each over --micro sequences, BF16 as in training)
are whitened on the input side as PD alpha 1/2 does, H_i = G_i R, R = (C / mean eig + 1e-3)^-1/2 from the curvature
sequences. In a frame (U, V) the coefficients A_i = U^T H_i V have per entry a mean mu and a micro-batch variance s^2;
at a batch of B tokens the noise variance is s^2 m / B (m tokens per micro-batch), so SNR_B = mu^2 B / (s^2 m).
Frames:
  soap     U, V = eigenbases of sum_i H_i H_i^T and sum_i H_i^T H_i (SOAP's statistics in PD-whitened coordinates)
  gn_out   U = eigenbasis of the sampled-label output factor B (the GN output curvature, TS's L), V = SOAP's V
  raw      U = I, V = I
Reported per frame and batch (1M, 4M, 16M tokens): the fraction of the signal energy (sum mu^2) in entries with
SNR_B > 1, and the entry-count fraction; plus the overlap of SOAP's and GN's top-k output eigenvectors.
The micro-batch mean is itself noisy (K micro-batches): mu^2 is bias-corrected by subtracting s^2 / K.

usage: frame_snr_probe.py OUT_JSON ARM_DIR:STEP [--micro 32] [--micro-batches 64]
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
from one_step_gn import CURV, GRAD, inverse_roots  # noqa: E402

BATCHES = {"1M": 2 ** 20, "4M": 2 ** 22, "16M": 2 ** 24}
TOPK = (8, 32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--micro", type=int, default=32, help="sequences per micro-batch")
    parser.add_argument("--micro-batches", type=int, default=64)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
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
    weights = [layers[n].weight for n in names]
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    curv = [stream.batch(CURV + first * T, 8, T, device) for first in range(0, args.curvature_sequences, 8)]
    # input factor C (positions >= 1, PD's statistic) and the sampled-label output factor B on the curvature sequences
    gen = torch.Generator(device=device).manual_seed(1)
    recorder = P.Recorder(model)
    sums = {n: {"C": 0, "B": 0} for n in names}
    count = 0
    for x, y in curv:
        P.gradient_passes(model, recorder, x, y, gen, draws=1)
        for n in names:
            xi = recorder.inputs[n][:, 1:].double().reshape(-1, recorder.inputs[n].shape[-1])
            es = (T * recorder.errors[n][1]).reshape(-1, recorder.errors[n][1].shape[-1]).double()
            sums[n]["C"] = sums[n]["C"] + xi.T @ xi
            sums[n]["B"] = sums[n]["B"] + es.T @ es
        count += x.numel()
    recorder.remove()
    R = {n: inverse_roots(sums[n]["C"] / count, (0.5,))[0.5] for n in names}
    UB = {n: torch.linalg.eigh(sums[n]["B"])[1].flip(-1).float() for n in names}      # descending
    # K micro-batch gradients, whitened on the input side
    H = {n: [] for n in names}
    for i in range(args.micro_batches):
        x, y = stream.batch(GRAD + i * args.micro * T, args.micro, T, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        grads = torch.autograd.grad(loss, weights)
        for n, g in zip(names, grads):
            H[n].append((g.float() @ R[n]))
    m_tokens = args.micro * T
    result = {"arm": arm, "step": step, "micro_tokens": m_tokens, "micro_batches": args.micro_batches, "matrices": {}}
    agg = {f: {b: [0.0, 0.0, 0, 0] for b in BATCHES} for f in ("soap", "gn_out", "raw")}   # signal>1, signal total, count>1, count
    for n in names:
        stack = torch.stack(H[n])                       # K x m x n
        K = stack.shape[0]
        gram_l = torch.einsum("kij,klj->il", stack, stack)
        gram_r = torch.einsum("kji,kjl->il", stack, stack)
        Us = torch.linalg.eigh(gram_l)[1].flip(-1)
        Vs = torch.linalg.eigh(gram_r)[1].flip(-1)
        entry = {}
        for frame, (U, V) in (("soap", (Us, Vs)), ("gn_out", (UB[n], Vs)), ("raw", (None, None))):
            A = stack if U is None else torch.einsum("ai,kab,bj->kij", U, stack, V)
            mu = A.mean(0)
            s2 = A.var(0, unbiased=True).clamp_min(1e-30)
            mu2 = (mu.pow(2) - s2 / K).clamp_min(0)       # bias-corrected squared signal
            row = {}
            for b, tokens in BATCHES.items():
                snr = mu2 * tokens / (s2 * m_tokens)
                high = snr > 1
                sig_high, sig_all = float(mu2[high].sum()), float(mu2.sum())
                row[b] = {"signal_fraction_snr_gt_1": sig_high / max(sig_all, 1e-30), "entry_fraction_snr_gt_1": float(high.float().mean())}
                agg[frame][b][0] += sig_high
                agg[frame][b][1] += sig_all
                agg[frame][b][2] += int(high.sum())
                agg[frame][b][3] += high.numel()
            entry[frame] = row
        entry["overlap_soap_gn_out"] = {k: float((Us[:, :k].T @ UB[n][:, :k]).pow(2).sum() / k) for k in TOPK}
        result["matrices"][n] = entry
        del stack, H[n]
    result["aggregate"] = {f: {b: {"signal_fraction_snr_gt_1": v[0] / max(v[1], 1e-30), "entry_fraction_snr_gt_1": v[2] / max(v[3], 1)}
                               for b, v in rows.items()} for f, rows in agg.items()}
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"item": args.item, "aggregate": {f: {b: [round(x["signal_fraction_snr_gt_1"], 3), round(x["entry_fraction_snr_gt_1"], 4)]
                                                           for b, x in r.items()} for f, r in result["aggregate"].items()},
                      "overlap_median_top8": sorted(e["overlap_soap_gn_out"][8] for e in result["matrices"].values())[len(names) // 2],
                      "seconds": round(result["seconds"])}), flush=True)


if __name__ == "__main__":
    main()

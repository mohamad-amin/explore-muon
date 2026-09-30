"""Gradient signal and noise per input eigendirection, against the input eigenvalue, across batch sizes
(2026-09-28 21:01 CDT; decision argument "does PD's tail amplification pay only at large batch?", 20:54 CDT).

PD scales the gradient's component along input eigendirection i (eigenvalue lambda_i of the input second moment C,
u_i = lambda_i / mean lambda) by (u_i + delta)^-alpha before and after the polar map, so it amplifies the weak
directions by up to delta^-alpha. That can only help where the component carries signal. At a kept state, for each
hidden matrix:
  - C from --curvature-sequences held-out validation sequences (positions >= 1, PD's statistic): eigenbasis V, u
  - K micro-batch gradients G_k (--micro sequences each, BF16 autocast as in training) on fresh validation sequences,
    projected P_k = G_k V. Per column i: per-micro-batch noise energy N_i = sum over rows of var_k P_k[:, i]; signal
    S_i = |mean_k P_k[:, i]|^2 - N_i / K (bias-corrected); SNR_i(B) = S_i B / (N_i m), m tokens per micro-batch
  - the checkpoint's momentum buffer M, projected the same way. Per column, its cosine with the mean gradient,
    cos_i = <M_i, mu_i> / (|M_i| sqrt(S_i)), with the numerator's square bias-corrected for the mean's noise
  - v2 (21:08 CDT): the noise in the momentum buffer. If the gradient noise is white across steps and as large as now,
    M's noise energy per column is N_i m / B_train / (1 - beta^2) (sum convention M <- beta M + g, B_train the run's
    batch); its ratio to |M_i|^2 is M's noise fraction, and the input maps' noise share of what enters the polar map is
    sum_i f_i^2 noise_M_i / sum_i f_i^2 |M_i|^2 (gradient clipping, if active, would make this an overestimate)
Reported in half-decade bins of u, pooled over matrices and per matrix kind: count, summed signal and noise, median
SNR at 1M / 4M / 16M tokens and the fraction of columns with SNR > 1, median cos. And per matrix, at each batch B, the
noise share of the input-scaled gradient for input maps f(u): 1 (Muon), (u + delta)^-1/4, (u + delta)^-1/2 (PD) and
min(1, (u + delta)^-1/2) (PD-top), applied once (what enters the polar map) and squared (the net input-side scaling):
  noise share = sum_i f_i^2 N_i m / B / sum_i f_i^2 (S_i + N_i m / B)
The polar map, which equalizes singular values, is not modeled. The per-step SNR ignores momentum: a signal that
persisted over the momentum window would gain up to (1 + beta) / (1 - beta) in SNR.

usage: direction_snr_probe.py OUT_JSON ARM_DIR:STEP [--micro 32] [--micro-batches 64]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402
from one_step_gn import CURV, GRAD  # noqa: E402

BATCHES = {"1M": 2 ** 20, "4M": 2 ** 22, "16M": 2 ** 24}
EDGES = [10 ** (k / 2) for k in range(-10, 5)]      # 1e-5 ... 1e2, half-decade bins
DAMPING = 1e-3
MAPS = {"muon": lambda u: torch.ones_like(u),
        "pd_quarter": lambda u: (u + DAMPING).pow(-0.25),
        "pd_half": lambda u: (u + DAMPING).pow(-0.5),
        "pdtop_half": lambda u: (u + DAMPING).pow(-0.5).clamp_max(1.0)}


def median(x):
    return float(x.median()) if x.numel() else None


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
    model, saved = P.load_checkpoint(Path(arm) / "scientific" / "kept" / f"step{step:06d}.pt", device)
    T = model.config.seq_len
    layers = P.hidden_linears(model)
    names = list(layers)
    keys = P.parameter_keys(model, names)
    named = dict(model.named_parameters())
    body_keys = [k for k in named if k.startswith("blocks.") and named[k].ndim == 2]
    body_indices = saved["optimizer"]["param_groups"][0]["params"]
    by_key = {k: saved["optimizer"]["state"][i]["momentum_buffer"].float() for k, i in zip(body_keys, body_indices)}
    momentum = {n: by_key[k].to(device) for n, k in zip(names, keys)}
    beta = saved["config"]["muon_momentum"]
    batch_train = saved["config"]["batch_tokens"]
    del saved, by_key
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for n in names:
        layers[n].weight.requires_grad_(True)
    weights = [layers[n].weight for n in names]
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))

    # input second moment C (positions >= 1), its eigenbasis (descending) and normalized eigenvalues u
    recorder = P.Recorder(model)
    sums = {n: 0 for n in names}
    with torch.no_grad():
        for first in range(0, args.curvature_sequences, 8):
            x, _ = stream.batch(CURV + first * T, 8, T, device)
            model(x)
            for n in names:
                xi = recorder.inputs[n][:, 1:].double()
                xi = xi.reshape(-1, xi.shape[-1])
                sums[n] = sums[n] + xi.T @ xi
    recorder.remove()
    V, U = {}, {}
    for n in names:
        values, vectors = torch.linalg.eigh(sums[n])
        values = values.flip(0).clamp_min(0)
        V[n] = vectors.flip(-1).float()
        U[n] = (values / values.mean().clamp_min(1e-30)).float()
    del sums

    # K micro-batch gradients projected on the input eigenbasis: running first and second moments per entry
    first_moment = {n: torch.zeros_like(layers[n].weight) for n in names}
    second_moment = {n: torch.zeros_like(layers[n].weight) for n in names}
    K = args.micro_batches
    for k in range(K):
        x, y = stream.batch(GRAD + k * args.micro * T, args.micro, T, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        for n, g in zip(names, torch.autograd.grad(loss, weights)):
            p = g.float() @ V[n]
            first_moment[n] += p
            second_moment[n] += p * p
    m_tokens = args.micro * T

    per_column = {}
    for n in names:
        mean = first_moment[n] / K
        var = ((second_moment[n] - K * mean * mean) / (K - 1)).clamp_min(0)
        noise = var.sum(0)                                              # per micro-batch of m tokens
        signal = (mean.pow(2).sum(0) - noise / K).clamp_min(0)
        mp = momentum[n] @ V[n]
        dot = (mp * mean).sum(0)
        dot2 = (dot.pow(2) - (mp.pow(2) * var).sum(0) / K).clamp_min(0)
        cos = torch.sign(dot) * (dot2 / (mp.pow(2).sum(0) * signal).clamp_min(1e-30)).sqrt().clamp_max(1.0)
        cos = torch.where(signal > 0, cos, torch.zeros_like(cos))
        per_column[n] = {"u": U[n], "signal": signal, "noise": noise.clamp_min(1e-30), "cos": cos,
                         "momentum": mp.pow(2).sum(0).clamp_min(1e-30),
                         "momentum_noise": noise * m_tokens / batch_train / (1 - beta ** 2)}
        del first_moment[n], second_moment[n]

    def bins(selected):
        u = torch.cat([per_column[n]["u"] for n in selected])
        s = torch.cat([per_column[n]["signal"] for n in selected])
        z = torch.cat([per_column[n]["noise"] for n in selected])
        c = torch.cat([per_column[n]["cos"] for n in selected])
        me = torch.cat([per_column[n]["momentum"] for n in selected])
        mz = torch.cat([per_column[n]["momentum_noise"] for n in selected])
        rows = []
        for lo, hi in zip(EDGES[:-1], EDGES[1:]):
            mask = (u >= lo) & (u < hi)
            if not mask.any():
                continue
            row = {"u_lo": lo, "u_hi": hi, "count": int(mask.sum()), "signal": float(s[mask].sum()),
                   "noise_per_micro": float(z[mask].sum()), "median_cos_momentum": median(c[mask]),
                   "momentum_energy": float(me[mask].sum()), "momentum_noise_fraction": float(mz[mask].sum() / me[mask].sum()),
                   "median_momentum_noise_fraction": median(mz[mask] / me[mask])}
            for b, tokens in BATCHES.items():
                snr = s[mask] * tokens / (z[mask] * m_tokens)
                row[f"median_snr_{b}"] = median(snr)
                row[f"fraction_snr_gt_1_{b}"] = float((snr > 1).float().mean())
            rows.append(row)
        return rows

    kinds = sorted({n.split(".")[1] for n in names})
    result = {"arm": arm, "step": step, "beta": beta, "batch_train": batch_train, "micro_tokens": m_tokens, "micro_batches": K,
              "curvature_sequences": args.curvature_sequences, "damping": DAMPING,
              "bins_all": bins(names), "bins_by_kind": {kind: bins([n for n in names if n.endswith("." + kind)]) for kind in kinds},
              "noise_share": {}, "matrices": {}}
    for n in names:
        col = per_column[n]
        entry = {"u_min": float(col["u"].min()), "u_max": float(col["u"].max()), "noise_share": {}}
        for b, tokens in BATCHES.items():
            z = col["noise"] * m_tokens / tokens
            for label, f in MAPS.items():
                w = f(col["u"])
                for power in (1, 2):
                    fw = w.pow(2 * power)
                    entry["noise_share"][f"{label}_f{power}_{b}"] = float((fw * z).sum() / (fw * (col["signal"] + z)).sum())
        for label, f in MAPS.items():
            for power in (1, 2):
                fw = f(col["u"]).pow(2 * power)
                entry["noise_share"][f"{label}_f{power}_momentum"] = float((fw * col["momentum_noise"]).sum() / (fw * col["momentum"]).sum())
        entry["momentum_noise_fraction"] = float(col["momentum_noise"].sum() / col["momentum"].sum())
        result["matrices"][n] = entry
    for key in result["matrices"][names[0]]["noise_share"]:
        values = torch.tensor([result["matrices"][n]["noise_share"][key] for n in names])
        result["noise_share"][key] = {"median": float(values.median()), "mean": float(values.mean())}
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    short = {b: {label: round(result["noise_share"][f"{label}_f1_{b}"]["median"], 3) for label in MAPS} for b in (*BATCHES, "momentum")}
    print(json.dumps({"item": args.item, "noise_share_f1_median": short, "seconds": round(result["seconds"])}), flush=True)
    for row in result["bins_all"]:
        print(f"  u [{row['u_lo']:.0e}, {row['u_hi']:.0e}) n {row['count']:5d}  SNR med 1M {row['median_snr_1M']:.3g} "
              f"16M {row['median_snr_16M']:.3g}  frac>1 1M {row['fraction_snr_gt_1_1M']:.2f} 16M {row['fraction_snr_gt_1_16M']:.2f}"
              f"  cos(M, mean) {row['median_cos_momentum']:.2f}  M noise frac {row['momentum_noise_fraction']:.2f}", flush=True)


if __name__ == "__main__":
    main()

"""Drift vs noise of the mean gradient per input eigendirection: what window should the momentum have?
(2026-09-28 21:07 CDT; follows direction_snr_probe.py)

An exponential average of noisy gradients estimates the current mean gradient mu_t. Its window trades noise (per-step
variance r at batch B) against staleness (the mean moves by Delta_t per step). For a local-level model with drift
variance q per step, the steady-state Kalman gain is k = P / (P + r) with P = (q + sqrt(q^2 + 4 q r)) / 2, i.e. an
EMA with beta = 1 - k. Along stiff directions the mean flips sign every step (edge of stability), where momentum damps
rather than estimates; the one-step autocorrelation separates the two cases (near -1: oscillation; near 1: slow drift).

At a kept state N whose next weights W_{N+1} were kept, for each hidden matrix, in the input eigenbasis V of W_N
(C from --curvature-sequences held-out validation sequences, positions >= 1, u = eigenvalue / mean eigenvalue):
  - K micro-batch gradients (--micro sequences each, BF16 autocast as in training) at W_N and at W_{N+1} on the SAME
    fresh validation sequences, projected on V. Per column i, bias-corrected with the micro-batch (co)variances:
    signal S_N, S_{N+1}; <mu_N, mu_{N+1}>; drift q_i = |mu_{N+1} - mu_N|^2; per-micro-batch noise at W_N
  - r_i = noise_i m / B_train: the per-step noise at the run's own batch
  - the checkpoint's momentum buffer M (it includes g_{N-1}, not g_N): <M_i, mu_N>, |M_i|^2, and M's noise energy if
    the gradient noise were white and as large as now: r_i / (1 - beta^2)
Reported in half-decade bins of u (pooled over matrices): pooled one-step autocorrelation
sum <mu_N, mu_{N+1}> / sqrt(sum S_N sum S_{N+1}); pooled q / r; the Kalman beta from the pooled q and r; pooled
cos(M, mu_N); M's noise fraction; and the same pooled over all columns.

usage: direction_drift_probe.py OUT_JSON ARM_DIR:STEP [--micro 32] [--micro-batches 64]
"""
import argparse
import json
import math
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

EDGES = [10 ** (k / 2) for k in range(-10, 5)]      # 1e-5 ... 1e2, half-decade bins


def kalman_beta(q, r):
    if q <= 0:
        return 1.0
    p = (q + math.sqrt(q * q + 4 * q * r)) / 2
    return 1 - p / (p + r)


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
    model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
    config = saved["config"]
    T = model.config.seq_len
    layers = P.hidden_linears(model)
    names = list(layers)
    keys = P.parameter_keys(model, names)
    named = dict(model.named_parameters())
    body_keys = [k for k in named if k.startswith("blocks.") and named[k].ndim == 2]
    body_indices = saved["optimizer"]["param_groups"][0]["params"]
    by_key = {k: saved["optimizer"]["state"][i]["momentum_buffer"].float() for k, i in zip(body_keys, body_indices)}
    momentum = {n: by_key[k].to(device) for n, k in zip(names, keys)}
    beta = config["muon_momentum"]
    batch_train = config["batch_tokens"]
    del saved, by_key
    nxt, _ = P.load_checkpoint(kept / f"step{step + 1:06d}_weights.pt", device, config["model"])
    next_layers = P.hidden_linears(nxt)
    for m in (model, nxt):
        for parameter in m.parameters():
            parameter.requires_grad_(False)
    for n in names:
        layers[n].weight.requires_grad_(True)
        next_layers[n].weight.requires_grad_(True)
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))

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

    zeros = lambda: {n: torch.zeros_like(layers[n].weight) for n in names}  # noqa: E731
    s1a, s2a, s1b, s2b, sab = zeros(), zeros(), zeros(), zeros(), zeros()
    K = args.micro_batches
    for k in range(K):
        x, y = stream.batch(GRAD + k * args.micro * T, args.micro, T, device)
        projected = []
        for m, lay in ((model, layers), (nxt, next_layers)):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = m(x, y)
            grads = torch.autograd.grad(loss, [lay[n].weight for n in names])
            projected.append({n: g.float() @ V[n] for n, g in zip(names, grads)})
        for n in names:
            a, b = projected[0][n], projected[1][n]
            s1a[n] += a
            s2a[n] += a * a
            s1b[n] += b
            s2b[n] += b * b
            sab[n] += a * b
    m_tokens = args.micro * T

    cols = {}
    for n in names:
        ma, mb = s1a[n] / K, s1b[n] / K
        va = ((s2a[n] - K * ma * ma) / (K - 1)).clamp_min(0)
        vb = ((s2b[n] - K * mb * mb) / (K - 1)).clamp_min(0)
        cab = (sab[n] - K * ma * mb) / (K - 1)
        sa = ma.pow(2).sum(0) - va.sum(0) / K
        sb = mb.pow(2).sum(0) - vb.sum(0) / K
        dab = (ma * mb).sum(0) - cab.sum(0) / K
        mp = momentum[n] @ V[n]
        noise = va.sum(0)
        cols[n] = {"u": U[n], "sa": sa, "sb": sb, "dab": dab, "q": sa + sb - 2 * dab,
                   "r": noise * m_tokens / batch_train, "dMa": (mp * ma).sum(0), "M2": mp.pow(2).sum(0),
                   "Mnoise": noise * m_tokens / batch_train / (1 - beta ** 2)}
        del s1a[n], s2a[n], s1b[n], s2b[n], sab[n]

    def pooled(mask_by_name):
        tot = {k: 0.0 for k in ("sa", "sb", "dab", "q", "r", "dMa", "M2", "Mnoise")}
        count = 0
        for n, mask in mask_by_name.items():
            for k in tot:
                tot[k] += float(cols[n][k][mask].sum())
            count += int(mask.sum())
        if count == 0:
            return None
        return {"count": count,
                "autocorr_one_step": tot["dab"] / math.sqrt(max(tot["sa"], 1e-30) * max(tot["sb"], 1e-30)),
                "signal_per_step_snr": tot["sa"] / max(tot["r"], 1e-30),
                "drift_over_noise": tot["q"] / max(tot["r"], 1e-30),
                "drift_over_signal": tot["q"] / max(tot["sa"], 1e-30),
                "kalman_beta": kalman_beta(tot["q"], tot["r"]),
                "cos_momentum_mean": tot["dMa"] / math.sqrt(max(tot["M2"], 1e-30) * max(tot["sa"], 1e-30)),
                "momentum_noise_fraction": tot["Mnoise"] / max(tot["M2"], 1e-30)}

    kinds = sorted({n.split(".")[1] for n in names})
    result = {"arm": arm, "step": step, "beta": beta, "batch_train": batch_train, "micro_tokens": m_tokens, "micro_batches": K,
              "all": pooled({n: torch.ones_like(cols[n]["u"], dtype=torch.bool) for n in names}),
              "bins": [], "by_kind": {}, "matrices": {}}
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        row = pooled({n: (cols[n]["u"] >= lo) & (cols[n]["u"] < hi) for n in names})
        if row:
            result["bins"].append({"u_lo": lo, "u_hi": hi, **row})
    for kind in kinds:
        sel = [n for n in names if n.endswith("." + kind)]
        result["by_kind"][kind] = {"all": pooled({n: torch.ones_like(cols[n]["u"], dtype=torch.bool) for n in sel}),
                                   "top_u_ge_1": pooled({n: cols[n]["u"] >= 1 for n in sel}),
                                   "tail_u_lt_1": pooled({n: cols[n]["u"] < 1 for n in sel})}
    for n in names:
        result["matrices"][n] = {"all": pooled({n: torch.ones_like(cols[n]["u"], dtype=torch.bool)}),
                                 "top_u_ge_1": pooled({n: cols[n]["u"] >= 1}), "tail_u_lt_1": pooled({n: cols[n]["u"] < 1})}
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    a = result["all"]
    print(json.dumps({"item": args.item, "beta": beta, "batch": batch_train, "all": {k: round(v, 3) for k, v in a.items()},
                      "seconds": round(result["seconds"])}), flush=True)
    for row in result["bins"]:
        print(f"  u [{row['u_lo']:.0e}, {row['u_hi']:.0e}) n {row['count']:5d}  autocorr {row['autocorr_one_step']:+.2f}  "
              f"SNR/step {row['signal_per_step_snr']:.3g}  q/r {row['drift_over_noise']:.3g}  q/S {row['drift_over_signal']:.3g}  "
              f"Kalman beta {row['kalman_beta']:.3f}  cos(M, mu) {row['cos_momentum_mean']:+.2f}  M noise {row['momentum_noise_fraction']:.2f}",
              flush=True)


if __name__ == "__main__":
    main()

"""Per-direction temporal statistics of the gradient in PD's input eigenbasis (tiny model, Candidate D).

For a run with per-step snapshots (the regime check's PD runs), in windows of consecutive steps at several
phases, and for each body matrix's input eigendirections v_i (the input second moment at the window start):
  signal     |g v_i|^2 of the mean gradient (two disjoint 256-sequence fresh probes, bias-corrected product)
  oscillation  lag-1 cross-probe correlation of g_t v_i with g_{t+1} v_i (negative: the direction flips)
  persistence  cross-probe correlation at even lags 2, 4, 8 (period-2 flips cancel at even lags)
  noise      batch-to-batch variance of g_B v_i at the run's batch size (independent fresh batches)
Directions are binned by input variance u = lambda / mean(lambda), energy-weighted over matrices. From the
lag correlations and the noise the script predicts, per bin, the heavy-ball beta that minimizes the
normalized EMA's error in tracking the next step's mean gradient (a direct, model-free fit on the measured
sequence), alongside the drift-to-noise ratio. Fresh blocks are training-stream blocks beyond the horizon.
Run from tiny_surrogate/ through ./run.
"""
import argparse
import importlib.util
import json
import math
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "regime_probe", HERE.parent / "regime_check_20260929" / "regime_probe.py")
rp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rp)

BINS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, float("inf")]
LAGS = (1, 2, 4, 8)
BETAS = [0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def projections(run, grad, bases):
    """Per matrix: G V (d_out x d_in coefficients in the input eigenbasis)."""
    tensors = run.unflat(grad)
    return [tensors[n] @ bases[n][0] for n in run.body_names]


def bin_index(unit):
    out = torch.zeros_like(unit, dtype=torch.long)
    for b in range(len(BINS) - 1):
        out[(unit >= BINS[b]) & (unit < BINS[b + 1])] = b
    return out


def window_stats(run, start, length, fresh_batches, batch_size):
    run.load(start)
    _, covariances = rp.input_statistics(run, run.band_probe)
    bases = {}
    for n in run.body_names:
        values, vectors = torch.linalg.eigh(covariances[n].double())
        unit = values.clamp_min(0) / values.clamp_min(0).mean()
        bases[n] = (vectors.float(), unit.float(), bin_index(unit))
    series_A, series_B = [], []
    for t in range(start, start + length + max(LAGS) + 1):
        run.load(t)
        gA, _, _ = rp.body_gradient(run, run.probe["A"], precision="fp32")
        gB, _, _ = rp.body_gradient(run, run.probe["B"], precision="fp32")
        series_A.append(projections(run, gA, bases))
        series_B.append(projections(run, gB, bases))
    # Noise at the run's batch size: independent fresh batches at the window's middle state.
    run.load(start + length // 2)
    size = 4 if run.smoke else batch_size
    fresh = [projections(run, rp.body_gradient(run, run.fresh_pool[i * size:(i + 1) * size])[0], bases)
             for i in range(fresh_batches)]
    nb = len(BINS) - 1
    signal = torch.zeros(nb, dtype=torch.float64)
    lag = {k: torch.zeros(nb, dtype=torch.float64) for k in LAGS}
    noise = torch.zeros(nb, dtype=torch.float64)
    count_dirs = torch.zeros(nb, dtype=torch.float64)
    # Per-bin mean-gradient sequence (cross-probe products avoid the probes' own noise).
    T = length + max(LAGS) + 1
    for j, n in enumerate(run.body_names):
        b = bases[n][2].to(series_A[0][j].device)
        for bi in range(nb):
            mask = b == bi
            if mask.sum() == 0:
                continue
            count_dirs[bi] += float(mask.sum())
            for t in range(length):
                a0 = series_A[t][j][:, mask]
                b0 = series_B[t][j][:, mask]
                signal[bi] += float((a0 * b0).sum())
                for k in LAGS:
                    lag[k][bi] += 0.5 * float((a0 * series_B[t + k][j][:, mask]).sum()
                                              + (b0 * series_A[t + k][j][:, mask]).sum())
            mean_fresh = sum(f[j][:, mask] for f in fresh) / len(fresh)
            noise[bi] += length * float(sum(((f[j][:, mask] - mean_fresh) ** 2).sum() for f in fresh)
                                        / (len(fresh) - 1))
    out = []
    for bi in range(nb):
        if signal[bi] <= 0 and count_dirs[bi] == 0:
            continue
        s = float(signal[bi])
        rho = {k: float(lag[k][bi]) / s if s > 0 else float("nan") for k in LAGS}
        r = float(noise[bi]) / s if s > 0 else float("nan")
        out.append(dict(bin=[BINS[bi], BINS[bi + 1]], directions=float(count_dirs[bi]) / len(run.body_names),
                        signal=s / length, noise_over_signal=r, lag_correlation=rho,
                        best_beta=best_beta(rho, r)))
    return out


def best_beta(rho, r):
    """Heavy-ball beta minimizing the error of the normalized EMA m = (1 - b) sum b^k g_{t-k} in tracking the
    current mean gradient, for a signal with lag correlations rho (linearly interpolated over lags, zero
    beyond the largest measured lag) plus white noise of relative variance r per batch. Meaningful only where
    the direction does not oscillate (lag-1 correlation > 0): in oscillating directions the damping, not the
    tracking error, is what the dynamics use (the split keeps the long beta there by design)."""
    corr = {0: 1.0}
    ks = sorted(rho)
    for k in range(1, 40):
        if k in rho:
            corr[k] = rho[k]
        elif k < ks[-1]:
            lo = max(x for x in ks if x < k)
            hi = min(x for x in ks if x > k)
            corr[k] = rho[lo] + (rho[hi] - rho[lo]) * (k - lo) / (hi - lo)
        else:
            corr[k] = 0.0
    scores = {}
    for b in BETAS:
        w = [(1 - b) * b ** k for k in range(40)]
        total = sum(w)
        w = [x / total for x in w]          # truncated EMA, weights summing to one
        # E|sum w_k s_{t-k} - s_t|^2 / E|s|^2 with signal correlations corr, plus noise sum w_k^2 r
        err = 1.0 - 2 * sum(w[k] * corr[k] for k in range(40))
        err += sum(w[i] * w[j] * corr[abs(i - j)] for i in range(40) for j in range(40))
        err += r * sum(x * x for x in w)
        scores[b] = err
    best = min(scores, key=scores.get)
    return dict(beta=best, error=scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id")
    parser.add_argument("--length", type=int, default=12)
    parser.add_argument("--fresh-batches", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run = rp.Run(HERE.parent / "regime_check_20260929" / "runs" / args.run_id, args.device, smoke=args.smoke)
    out = HERE / "analysis" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    T = run.total_steps
    big = run.batch_sequences >= 2048
    starts = [8, 24, 44, 66, 76] if big else [30, 100, 180, 260, 320, 340]
    if args.smoke:
        starts, args.length, args.fresh_batches = [1], 2, 2
        out = HERE / "smoke" / args.run_id
        out.mkdir(parents=True, exist_ok=True)
    results = []
    for start in starts:
        if start + args.length + max(LAGS) > T:
            continue
        t0 = time.time()
        stats = window_stats(run, start, args.length, args.fresh_batches, run.batch_sequences)
        results.append(dict(start=start, length=args.length, lr=run.metrics[start]["lr"], bins=stats,
                            seconds=time.time() - t0))
        (out / "temporal.json").write_text(json.dumps(results, indent=1) + "\n")
        print(json.dumps(dict(start=start, seconds=round(time.time() - t0),
                              bins=[(b["bin"], round(b["lag_correlation"][1], 3), round(b["lag_correlation"][2], 3),
                                     round(b["noise_over_signal"], 3), b["best_beta"]["beta"]) for b in stats])),
              flush=True)


if __name__ == "__main__":
    main()

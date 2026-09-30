"""Is the momentum's staleness the Gauss-Newton transport term, and what does a fresh, averaged momentum buy?

At a kept checkpoint t of a run with muon_track_displacement (Q_t = sum_{k>=1} beta^k (W_t - W_{t-k}), body matrices):
  replay     the exact stale-free momentum M*_t = sum_{k<K} beta^k f_{t-k} g_{t-k}(W_t): the run's own training batches
             t-K+1..t (same data order and precision, clip factors f from the log) re-evaluated at the current weights.
             The staleness S = M_t - M*_t is then known exactly (the tail beta^K M_{t-K}, ~0.6% at K = 100, is dropped).
             Data order is checked against the logged training loss of step t + 1 (evaluated at W_t).
  transport  in the GN model the next momentum M' = beta M_t + g satisfies M' - M*' = beta S = -H Q_t (M*' = beta M*_t + g).
             Predictions X for H Q_t: the exact GN product G Q, the true Hessian (central differences of the gradient),
             the per-matrix GN blocks, and K-FAC B Q C, on held-out transport sequences. Per matrix and in total:
             cos(beta S, -X), the best multiplier s* = <beta S, -X> / |X|^2, and |beta S + X| / |beta S|.
             (Only the body matrices' displacement is tracked; embeddings, gains and head moved too.)
  one-step   held-out, cross-fitted decrease (scale, and damping for GN, picked on one half of the held-out sequences and
             measured on the other) of GD / Muon / PD / damped-GN maps applied to the inputs g (fresh 1M gradient), M',
             M*' (exact fresh momentum), M' + X for each transport X (and 1/2 G Q), and the replay mean (the current
             gradient over the K replayed batches).
Also the gradient noise at W_t from the K replayed batches (per matrix trace, and its energy against the GN curvature).
One-step and local, like one_step_gn.py.

usage: transport_test.py OUT_DIR ARM_DIR:STEP [--replay 100] [--transport-sequences 512] [--krylov 48]
"""
import argparse
import json
import math
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
from one_step_gn import (BASE, CURV, EVAL, GRAD, Flat, GaussNewton, fresh_gradient, inverse_roots,  # noqa: E402
                         krylov_direction, lanczos, muon_norm, polar, statistics)
from one_step_blockgn import cross_fit, halves_score  # noqa: E402

TRANS = BASE + 120000 * 512     # transport products (held out from the curvature, gradient and scoring sequences)
DAMPINGS = (1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4)


def replay(model, names, config, arm, step, K, device, micro, frames=None, probes=None, flat=None):
    """M*_t over the last K training batches at the current weights; also their plain mean and noise trace, and (with
    frames = {name: (U, V, bin index, bins)}) the per-bin sum over batches of the squared frame coefficients, and (with
    probes, k x n unit vectors over the flat hidden matrices) each batch gradient's k projections."""
    T = model.config.seq_len
    B = config["batch_tokens"]
    beta = config["muon_momentum"]
    train = TokenStream(str(REPO / config["train_pattern"]))
    layers = P.hidden_linears(model)
    weights = [layers[n].weight for n in names]
    logs = Path(arm) / "scientific" / "steps"

    def batch(s):
        tokens = torch.from_numpy(train.read((s - 1) * B, B + 1)).to(device)
        return tokens[:-1].view(-1, T), tokens[1:].view(-1, T)

    # data order: W_t on batch t + 1 must reproduce step t + 1's logged training loss
    x_all, y_all = batch(step + 1)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        check = sum(float(model(x_all[i:i + micro], y_all[i:i + micro])) * x_all[i:i + micro].shape[0]
                    for i in range(0, x_all.shape[0], micro)) / x_all.shape[0]
    logged = json.loads((logs / f"step{step + 1:06d}.json").read_text())["train_nll"]
    star = OrderedDict((n, torch.zeros_like(layers[n].weight)) for n in names)
    mean = OrderedDict((n, torch.zeros_like(layers[n].weight)) for n in names)
    square = OrderedDict((n, 0.0) for n in names)
    binned = OrderedDict((n, torch.zeros(frames[n][3], dtype=torch.float64, device=device)) for n in names) if frames else None
    projections = []
    factors = []
    for k in range(K):
        s = step - k
        log = json.loads((logs / f"step{s:06d}.json").read_text())
        norm = log["gradient_norm_before_clip"]
        f = min(1.0, config["grad_clip"] / (norm + 1e-6))
        factors.append(f)
        x_all, y_all = batch(s)
        g = OrderedDict((n, torch.zeros_like(layers[n].weight)) for n in names)
        for i in range(0, x_all.shape[0], micro):
            x, y = x_all[i:i + micro], y_all[i:i + micro]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            grads = torch.autograd.grad(loss * (x.shape[0] / x_all.shape[0]), weights)
            for n, gr in zip(names, grads):
                g[n] += gr.float()
        for n in names:
            star[n] += (beta ** k) * f * g[n]
            mean[n] += g[n] / K
            square[n] += float(g[n].pow(2).sum()) / K
            if frames:
                U, V, index, bins = frames[n]
                binned[n] += torch.bincount(index, weights=(U.T @ g[n] @ V).flatten().double().pow(2), minlength=bins)
        if probes is not None:
            projections.append((probes @ flat.flat(g)).tolist())
    noise = {n: (square[n] - float(mean[n].pow(2).sum())) * K / (K - 1) for n in names}
    return star, mean, noise, {"check_loss": check, "logged_loss": logged, "clip_factors": factors,
                               "probe_projections": projections}, binned


def hessian_product(model, names, batches, v, flat, eps):
    """H v by central differences of the fp32 gradient of the mean token loss (hidden matrices moved by +-eps v)."""
    layers = P.hidden_linears(model)
    weights = [layers[n].weight for n in names]
    originals = [w.detach().clone() for w in weights]
    parts = flat.dict(v)
    count = sum(x.numel() for x, _ in batches)
    out = []
    for sign in (1.0, -1.0):
        with torch.no_grad():
            for w, w0, n in zip(weights, originals, names):
                w.copy_(w0 + sign * eps * parts[n])
        total = torch.zeros_like(v)
        for x, y in batches:
            loss = P.token_losses(model(x), y).sum() / count
            grads = torch.autograd.grad(loss, weights)
            total += torch.cat([g.reshape(-1) for g in grads])
        out.append(total)
    with torch.no_grad():
        for w, w0 in zip(weights, originals):
            w.copy_(w0)
    return (out[0] - out[1]) / (2 * eps)


def frame_bins(stats, names, count=16, device="cuda"):
    """Global log-spaced bins of the exact per-pair GN diagonal h (each matrix's Kronecker frame), between its
    0.1% and 99.99% quantiles (subsampled); returns edges and {name: (U, V, bin index, bins)}."""
    pooled = torch.cat([stats[n]["h"].flatten() for n in names])
    positive = pooled[pooled > 0]
    sample = positive[torch.randperm(positive.numel(), device=positive.device)[:2_000_000]]
    lo, hi = (float(torch.quantile(sample.float(), q)) for q in (1e-3, 0.9999))
    edges = torch.logspace(math.log10(lo), math.log10(hi), count - 1, device=device)
    bins = count
    frames = {n: (stats[n]["U"], stats[n]["V"], torch.bucketize(stats[n]["h"].flatten().clamp_min(0), edges), bins) for n in names}
    return edges, frames


def frame_products(frames, names, vectors, pairs):
    """Per matrix and curvature bin: sums of c_a * c_b over the frame pairs (c_X = U^T X V), the pair count and sum h."""
    out = {}
    for n in names:
        U, V, index, bins = frames[n]
        coeff = {k: (U.T @ v[n] @ V).flatten().double() for k, v in vectors.items()}
        entry = {"count": torch.bincount(index, minlength=bins).tolist()}
        for a, b in pairs:
            entry[f"{a}*{b}"] = torch.bincount(index, weights=coeff[a] * coeff[b], minlength=bins).tolist()
        out[n] = entry
    return out


def compare(target, prediction):
    """cos, best multiplier and residual fractions of a prediction for target (flat vectors)."""
    tn, pn = float(target.norm()), float(prediction.norm())
    dot = float(target @ prediction)
    cos = dot / max(tn * pn, 1e-30)
    return {"cos": cos, "s_star": dot / max(pn * pn, 1e-30), "residual_s1": float((target - prediction).norm()) / max(tn, 1e-30),
            "residual_best": math.sqrt(max(0.0, 1 - cos * cos)), "target_norm": tn, "prediction_norm": pn}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--replay", type=int, default=100)
    parser.add_argument("--replay-micro", type=int, default=16)
    parser.add_argument("--transport-sequences", type=int, default=512)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    parser.add_argument("--eval-sequences", type=int, default=512)
    parser.add_argument("--krylov", type=int, default=48)
    parser.add_argument("--micro", type=int, default=8)
    parser.add_argument("--ritz-steps", type=int, default=32, help="random-start Lanczos steps for the top GN eigenvectors")
    parser.add_argument("--ritz", type=int, default=16, help="top GN eigenvectors kept as exact stiff probes")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.add_argument("--smoke", action="store_true", help="code-path check on an untracked checkpoint (Q := M / 1000)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    started = time.time()
    arm, step = args.item.rsplit(":", 1)
    step = int(step)
    kept = Path(arm) / "scientific" / "kept"
    model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
    config = saved["config"]
    beta = config["muon_momentum"]
    T = model.config.seq_len
    layers = P.hidden_linears(model)
    names = list(layers)
    body = [p for name, p in model.named_parameters() if name.startswith("blocks.") and p.ndim == 2]
    if len(body) != len(names) or any(p is not layers[n].weight for p, n in zip(body, names)):
        raise RuntimeError("optimizer body order differs from the hidden-matrix order")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for n in names:
        layers[n].weight.requires_grad_(True)
    state = saved["optimizer"]["state"]
    M = OrderedDict((n, state[i]["momentum_buffer"].to(device).float()) for i, n in enumerate(names))
    Q = OrderedDict((n, (state[i]["displacement_ema"].to(device).float() if not args.smoke else 1e-3 * M[n]))
                    for i, n in enumerate(names))
    del saved
    flat = Flat(OrderedDict((n, layers[n].weight) for n in names))
    name = f"{Path(arm).name}_step{step:06d}"
    result = {"arm": arm, "step": step, "beta": beta, "replay": args.replay, "names": names}

    def dump():
        (args.out / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")

    def log(stage, **extra):
        print(json.dumps({"stage": stage, "seconds": round(time.time() - started), **extra}), flush=True)

    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    curv = [stream.batch(CURV + first * T, args.micro, T, device) for first in range(0, args.curvature_sequences, args.micro)]
    stats = statistics(model, names, curv, T, device)
    edges, frames = frame_bins(stats, names)
    result["bins"] = {"edges": edges.tolist(), "h_sum": {n: torch.bincount(frames[n][2], weights=stats[n]["h"].flatten().double(),
                                                                           minlength=frames[n][3]).tolist() for n in names}}
    log("statistics")
    # the stiffest exact GN directions (all hidden matrices; random-start Lanczos on the curvature sequences)
    op_curv = GaussNewton(model, names, curv, flat)
    gen = torch.Generator(device=device).manual_seed(0)
    V, Tk = lanczos(op_curv, torch.randn(sum(flat.sizes), device=device, generator=gen), args.ritz_steps)
    theta, Y = torch.linalg.eigh(Tk)
    order = torch.argsort(theta, descending=True)[:args.ritz]
    probes = (V.T @ Y[:, order].to(V.device, torch.float32)).T.contiguous()
    del V
    result["ritz"] = {"values": theta[order].tolist(), "all_values": torch.linalg.eigvalsh(Tk).flip(0).tolist()}
    log("ritz", top=[round(float(x), 3) for x in theta[order][:4]])
    star, mean, noise, check, binned = replay(model, names, config, arm, step, args.replay, device, args.replay_micro, frames,
                                              probes, flat)
    result["replay_check"] = check
    log("replay", check_loss=round(check["check_loss"], 5), logged_loss=round(check["logged_loss"], 5),
        clipped=sum(f < 1 for f in check["clip_factors"]))
    trans = [stream.batch(TRANS + first * T, args.micro, T, device) for first in range(0, args.transport_sequences, args.micro)]
    evals = [stream.batch(EVAL + first * T, args.micro, T, device) for first in range(0, args.eval_sequences, args.micro)]
    halves = [evals[:len(evals) // 2], evals[len(evals) // 2:]]

    # transport predictions of beta S = M' - M*'
    q_flat = flat.flat(Q)
    op_trans = GaussNewton(model, names, trans, flat)
    predictions = OrderedDict()
    predictions["G"] = op_trans(q_flat)
    w_flat = torch.cat([layers[n].weight.detach().reshape(-1) for n in names])
    eps = 1e-2 * float(w_flat.norm()) / max(float(q_flat.norm()), 1e-30)
    predictions["H"] = hessian_product(model, names, trans, q_flat, flat, eps)
    blocks = []
    for n in names:
        sub = Flat(OrderedDict([(n, Q[n])]))
        blocks.append(GaussNewton(model, [n], trans, sub)(sub.flat({n: Q[n]})))
    predictions["G_block"] = torch.cat(blocks)
    kfac = []
    for n in names:     # B Q C with B = E[e~ e~^T] (sampled labels) and C = E[x x^T], per token (gn_probe's units)
        s = stats[n]
        left = (s["U"] * s["lam_B"].clamp_min(0)) @ s["U"].T
        right = (s["V"] * s["lam_C"].clamp_min(0)) @ s["V"].T
        kfac.append((left @ Q[n] @ right).reshape(-1))
    predictions["kfac"] = torch.cat(kfac)
    log("transport", eps=eps)
    m_flat, star_flat = flat.flat(M), flat.flat(star)
    target = beta * (m_flat - star_flat)
    kinds = {kind: [n for n in names if n.endswith("." + kind)] for kind in P.KINDS}

    def per_group(vec_a, vec_b):
        a, b = flat.dict(vec_a), flat.dict(vec_b)
        out = {"total": compare(vec_a, vec_b)}
        for kind, members in kinds.items():
            out[kind] = compare(torch.cat([a[n].reshape(-1) for n in members]), torch.cat([b[n].reshape(-1) for n in members]))
        out["matrix"] = {n: compare(a[n].reshape(-1), b[n].reshape(-1)) for n in names}
        return out
    result["staleness"] = {
        "norm_M": float(m_flat.norm()), "norm_Mstar": float(star_flat.norm()), "norm_S": float((m_flat - star_flat).norm()),
        "cos_M_Mstar": float(m_flat @ star_flat) / float(m_flat.norm() * star_flat.norm()),
        "norm_Q": float(q_flat.norm()), "eps_hessian": eps,
        "predictions": {label: per_group(target, -x) for label, x in predictions.items()},
        "G_vs_H": per_group(predictions["H"], predictions["G"]),
    }
    mean_flat = flat.flat(mean)
    result["noise"] = {"trace": noise, "mean_norm2": {n: float(mean[n].pow(2).sum()) for n in names},
                       "batch_tokens": config["batch_tokens"],
                       # per bin: sum over the K batches of squared frame coefficients (noise = (this - K gbar^2) / (K - 1))
                       "sum_sq_batches": {n: binned[n].tolist() for n in names}}
    vectors = {"M": M, "Mstar": star, "gbar": mean, "bS": flat.dict(target), "GQ": flat.dict(predictions["G"]),
               "HQ": flat.dict(predictions["H"]), "Q": Q}
    pairs = [("M", "M"), ("Mstar", "Mstar"), ("gbar", "gbar"), ("M", "gbar"), ("Mstar", "gbar"), ("M", "Mstar"),
             ("bS", "bS"), ("GQ", "GQ"), ("bS", "GQ"), ("HQ", "HQ"), ("bS", "HQ"), ("Q", "Q"), ("Q", "gbar"), ("M", "Q")]
    result["frame_bins"] = frame_products(frames, names, vectors, pairs)
    result["ritz"]["projections"] = {k: (probes @ flat.flat(v)).tolist() for k, v in vectors.items()}
    dump()
    summary = {label: {k: round(v, 3) for k, v in d["total"].items() if k in ("cos", "s_star", "residual_s1")}
               for label, d in result["staleness"]["predictions"].items()}
    log("staleness", norm_ratio_S_M=round(result["staleness"]["norm_S"] / result["staleness"]["norm_M"], 3),
        cos_M_Mstar=round(result["staleness"]["cos_M_Mstar"], 3), **summary)
    if args.save:     # M and Q are in the kept checkpoint; G_block and kfac are cheap to recompute
        torch.save({"names": names, "beta": beta, **{k: OrderedDict((n, t.to(torch.bfloat16).cpu()) for n, t in flat.dict(v).items())
                    for k, v in (("Mstar", star_flat), ("replay_mean", mean_flat), ("X_G", predictions["G"]),
                                 ("X_H", predictions["H"]))}},
                   args.out / f"{name}_tensors.pt")

    # one-step: maps on each input, cross-fitted on the two halves of the held-out sequences
    g1 = fresh_gradient(model, names, stream, GRAD, 2048, T, device)
    g_flat = flat.flat(g1)
    m_next = beta * m_flat + g_flat
    inputs = OrderedDict([("g1M", g_flat), ("M'", m_next), ("M*'", beta * star_flat + g_flat),
                          ("M'+GQ", m_next + predictions["G"]), ("M'+GQ/2", m_next + 0.5 * predictions["G"]),
                          ("M'+HQ", m_next + predictions["H"]), ("M'+G_blockQ", m_next + predictions["G_block"]),
                          ("M'+kfacQ", m_next + predictions["kfac"]), ("replay_mean", mean_flat)])
    roots = {n: {a: stats[n][f"R{a}"] for a in (0.25, 0.5)} for n in names}
    del probes
    result["one_step"] = OrderedDict()
    for label, b_flat in inputs.items():
        t0 = time.time()
        b = flat.dict(b_flat)
        directions = OrderedDict()
        directions["gd"] = OrderedDict((n, -b[n]) for n in names)
        directions["muon"] = OrderedDict((n, -polar(b[n]) * math.sqrt(max(1.0, b[n].shape[0] / b[n].shape[1]))) for n in names)
        for a in (0.25, 0.5):
            pd = OrderedDict()
            for n in names:
                d = polar(b[n] @ roots[n][a]) @ roots[n][a]
                pd[n] = -d * (muon_norm(b[n].shape) / d.norm().clamp_min(1e-30))
            directions[f"pd_a{a:g}"] = pd
        families = {key: {key: d} for key, d in directions.items()}
        V, Tk = lanczos(op_curv, b_flat, args.krylov)
        rho = float(Tk[0, 0])
        families["gn"] = {f"gn_d{d:g}": flat.dict(krylov_direction(V, Tk, float(b_flat.norm()), args.krylov, d * rho))
                          for d in DAMPINGS}
        del V
        entry = {"rho": rho, "cos_replay_mean": float(b_flat @ mean_flat) / float(b_flat.norm() * mean_flat.norm()),
                 "cos_g1M": float(b_flat @ g_flat) / float(b_flat.norm() * g_flat.norm()), "families": {}}
        for family, candidates in families.items():
            scored = {k: halves_score(model, halves, d) for k, d in candidates.items()}
            entry["families"][family] = {"cross_fit": cross_fit(scored), "raw": scored}
        result["one_step"][label] = entry
        dump()
        log("one_step", input=label, rho=round(rho, 3), cos_replay_mean=round(entry["cos_replay_mean"], 3),
            **{f: round(v["cross_fit"]["cross_fitted"] * 1e3, 3) for f, v in entry["families"].items()},
            input_seconds=round(time.time() - t0))
    result["seconds"] = time.time() - started
    dump()
    log("done")


if __name__ == "__main__":
    main()

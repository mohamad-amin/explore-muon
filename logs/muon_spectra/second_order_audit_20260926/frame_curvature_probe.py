"""Does SOAP's per-entry normalization act as curvature? Per-entry signal against the exact GN diagonal in SOAP's frame,
and the one-step quality of per-entry normalizations inside PD (2026-09-28 decision after the GN-rate pre-flight).

At a kept state, on 1 GPU:
  - R = (C / mean eig + 1e-3)^-1/2 from 256 curvature sequences (positions >= 1): PD alpha 1/2's input root
  - SOAP's frame in PD-whitened coordinates from K micro-batch gradients H_i = G_i R (--micro sequences each, data
    labels): U, V = eigenbases of sum H_i H_i^T and sum H_i^T H_i (as frame_snr_probe.py; SOAP's own state is not in
    the checkpoints)
  - per entry of A = U^T (M R) V, M the checkpoint's momentum:
      v   the batch-level second moment of the gradient's frame coefficient at the run's batch B:
          mu^2 + s^2 m / B (mu, s^2 over micro-batches of m tokens): SOAP's "gradient" mode
      h   the exact GN diagonal in the frame, T mean_s (U^T g_s R V)^2 over per-sequence gradients g_s of the mean
          token loss with model-sampled labels (256 sequences; E_y[g g^T] = GN / T per sequence)
  - per matrix: energy-weighted slope of log h on log |A| (h ~ |A|^gamma: gamma = 1 makes sign(A) the diagonal
    Newton step A / h up to scale), its correlation, and the spread of log(|A| / h) (the per-entry Newton distance)
  - directions, each matrix rescaled to the run's step lr sqrt(min(m, n)) sqrt(max(1, m/n)):
      muon         polar(M)
      pd           polar(M R) R
      kron         L polar(L M R) R, L the sampled-label output root at 1/2 (TS 1/2/1/2)
      soap         polar(U [A / sqrt(v)] V^T) R           S∘PD with SOAP's gradient-mode normalization
      pow0.25 ... pow1   polar(U [A / |A|^g] V^T) R       per-entry power normalization (g = 1: sign)
      fisher       polar(U [A / sqrt(h)] V^T) R           Adam's small-batch limit with the exact GN diagonal
      fnewton      polar(U [A / h] V^T) R                 per-entry curvature normalization, then polar
      diag_newton  U [A / h] V^T R                        the diagonal Newton step in the frame, no polar
  - one step on held-out training sequences (tokens beyond every 1x budget): slope a and exact GN curvature q along
    each step (set 1), quality a^2 / 2q, c* = -a / q; the true loss change at 0.5, 1, 2 x the step (set 2, paired)

usage: frame_curvature_probe.py OUT_JSON ARM_DIR:STEP [--micro 32] [--micro-batches 64]
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
from research.adamw_spectra.data import TokenStream, token_views  # noqa: E402
from research.adamw_spectra.train import learning_rate, token_budget  # noqa: E402
from one_step_gn import CURV, GRAD  # noqa: E402

HELD_OUT_TRAIN = 2_500_000_000
KINDS = ("q", "k", "v", "o", "up", "down")
POWERS = (0.25, 0.5, 0.75, 1.0)


def polar(x):
    u, _, vh = torch.linalg.svd(x.float(), full_matrices=False)
    return u @ vh


def root(moment, power, damping=1e-3):
    values, vectors = torch.linalg.eigh(0.5 * (moment + moment.T).double())
    unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
    return ((vectors * (unit + damping).pow(-power)) @ vectors.T).float()


def weighted_slope(x, y, w):
    w = w / w.sum()
    mx, my = (w * x).sum(), (w * y).sum()
    cov = (w * (x - mx) * (y - my)).sum()
    var_x, var_y = (w * (x - mx) ** 2).sum(), (w * (y - my) ** 2).sum()
    return float(cov / var_x.clamp_min(1e-30)), float(cov / (var_x * var_y).sqrt().clamp_min(1e-30))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--micro", type=int, default=32)
    parser.add_argument("--micro-batches", type=int, default=64)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    parser.add_argument("--held-sequences", type=int, default=128)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    started = time.time()
    arm, step = args.item.rsplit(":", 1)
    step = int(step)
    model, saved = P.load_checkpoint(Path(arm) / "scientific" / "kept" / f"step{step:06d}.pt", device)
    config = saved["config"]
    T = model.config.seq_len
    layers = P.hidden_linears(model)
    names = list(layers)
    keys = P.parameter_keys(model, names)
    named = dict(model.named_parameters())
    body_keys = [k for k in named if k.startswith("blocks.") and named[k].ndim == 2]
    body_indices = saved["optimizer"]["param_groups"][0]["params"]
    momentum = {k: saved["optimizer"]["state"][i]["momentum_buffer"].to(device).float()
                for k, i in zip(body_keys, body_indices)}
    del saved
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for n in names:
        layers[n].weight.requires_grad_(True)
    weights = [layers[n].weight for n in names]
    budget = token_budget(config, sum(p.numel() for p in model.parameters()), T)
    lr = learning_rate(config, step + 1, step * config["batch_tokens"], budget)
    val = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    train = TokenStream(str(REPO / config["train_pattern"]))
    curv = [val.batch(CURV + first * T, 8, T, device) for first in range(0, args.curvature_sequences, 8)]

    # input root R (PD alpha 1/2) and the sampled-label output root L (TS), then the per-sequence GN diagonal later
    gen = torch.Generator(device=device).manual_seed(1)
    recorder = P.Recorder(model)
    sums = {n: [0, 0] for n in names}
    for x, y in curv:
        P.gradient_passes(model, recorder, x, y, gen, draws=1)
        for n in names:
            xi = recorder.inputs[n][:, 1:].double().reshape(-1, recorder.inputs[n].shape[-1])
            es = (T * recorder.errors[n][1]).double().reshape(-1, recorder.errors[n][1].shape[-1])
            sums[n][0] = sums[n][0] + es.T @ es
            sums[n][1] = sums[n][1] + xi.T @ xi
    R = {n: root(sums[n][1], 0.5) for n in names}
    L = {n: root(sums[n][0], 0.5) for n in names}
    del sums

    # micro-batch gradients whitened on the input side, and SOAP's frame
    H = {n: [] for n in names}
    for i in range(args.micro_batches):
        x, y = val.batch(GRAD + i * args.micro * T, args.micro, T, device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        for n, g in zip(names, torch.autograd.grad(loss, weights)):
            H[n].append(g.float() @ R[n])
    m_tokens = args.micro * T
    frames, stats = {}, {}
    for n in names:
        stack = torch.stack(H[n])
        U = torch.linalg.eigh(torch.einsum("kij,klj->il", stack, stack).double())[1].flip(-1).float()
        V = torch.linalg.eigh(torch.einsum("kji,kjl->il", stack, stack).double())[1].flip(-1).float()
        coef = torch.einsum("ai,kab,bj->kij", U, stack, V)
        mu, s2 = coef.mean(0), coef.var(0, unbiased=True)
        frames[n] = (U, V)
        stats[n] = {"v": mu.pow(2) + s2 * m_tokens / config["batch_tokens"]}
        del stack, coef, H[n]
    print(json.dumps({"stage": "frames", "seconds": round(time.time() - started)}), flush=True)

    # exact GN diagonal in the frame from per-sequence gradients with model-sampled labels
    gen = torch.Generator(device=device).manual_seed(2)
    h = {n: 0 for n in names}
    count = 0
    for x, y in curv:
        P.gradient_passes(model, recorder, x, y, gen, draws=1)
        for n in names:
            U, V = frames[n]
            per_sequence = P.per_sequence_gradients(recorder.inputs[n].float(), recorder.errors[n][1].float())  # (B, o, i)
            h[n] = h[n] + (U.T @ (per_sequence @ R[n]) @ V).pow(2).sum(0)
        count += x.shape[0]
    recorder.remove()
    for n in names:
        stats[n]["h"] = h[n] * T / count
    print(json.dumps({"stage": "gn_diagonal", "seconds": round(time.time() - started)}), flush=True)

    # per-entry relation between signal and curvature, per matrix
    relation = {}
    for n, k in zip(names, keys):
        U, V = frames[n]
        A = U.T @ (momentum[k] @ R[n]) @ V
        hh, vv = stats[n]["h"], stats[n]["v"]
        good = (hh > 0) & (A.abs() > 0)
        x, yh = A.abs()[good].double().log(), hh[good].double().log()
        w = A[good].double().pow(2)
        slope_w, corr_w = weighted_slope(x, yh, w)
        slope_u, corr_u = weighted_slope(x, yh, torch.ones_like(w))
        distance = (x - yh)
        dm = (w * distance).sum() / w.sum()
        relation[n] = {"slope_energy": slope_w, "corr_energy": corr_w, "slope_uniform": slope_u, "corr_uniform": corr_u,
                       "log_distance_energy_sd": float(((w * (distance - dm) ** 2).sum() / w.sum()).sqrt()),
                       "slope_sqrtv_vs_A": weighted_slope(x, vv[good].double().sqrt().clamp_min(1e-30).log(), w)[0]}
        stats[n]["A"] = A

    # directions
    shapes = {k: momentum[k].shape for k in keys}
    targets = {k: lr * math.sqrt(min(shapes[k])) * math.sqrt(max(1.0, shapes[k][0] / shapes[k][1])) for k in keys}

    def family(make):
        out = {}
        for n, k in zip(names, keys):
            d = make(n, k)
            out[k] = -d * (targets[k] / float(d.norm().clamp_min(1e-30)))
        return out

    def in_frame(n, entries):
        U, V = frames[n]
        return polar(U @ entries @ V.T) @ R[n]
    directions = {
        "muon": family(lambda n, k: polar(momentum[k])),
        "pd": family(lambda n, k: polar(momentum[k] @ R[n]) @ R[n]),
        "kron": family(lambda n, k: L[n] @ polar(L[n] @ momentum[k] @ R[n]) @ R[n]),
        "soap": family(lambda n, k: in_frame(n, stats[n]["A"] / stats[n]["v"].sqrt().clamp_min(1e-30))),
        "fisher": family(lambda n, k: in_frame(n, stats[n]["A"] / stats[n]["h"].sqrt().clamp_min(1e-30))),
        "fnewton": family(lambda n, k: in_frame(n, stats[n]["A"] / stats[n]["h"].clamp_min(1e-30))),
        "diag_newton": family(lambda n, k: frames[n][0] @ (stats[n]["A"] / stats[n]["h"].clamp_min(1e-30))
                              @ frames[n][1].T @ R[n]),
    }
    for g in POWERS:
        directions[f"pow{g:g}"] = family(lambda n, k, g=g: in_frame(n, stats[n]["A"] / stats[n]["A"].abs().clamp_min(1e-30).pow(g)))

    # one step on held-out training sequences
    held = []
    for j in range(2):
        first = HELD_OUT_TRAIN + j * args.held_sequences * T
        hx, hy = token_views(train.device_tokens(first, args.held_sequences * T + 1, device), 0, args.held_sequences * T, T)
        held.append([(hx[i:i + 8], hy[i:i + 8]) for i in range(0, args.held_sequences, 8)])

    @torch.no_grad()
    def held_loss(batches):
        return sum(float(P.token_losses(model(x), y).mean(1).sum()) for x, y in batches) / args.held_sequences
    base = held_loss(held[1])
    originals = {k: named[k].detach().clone() for k in keys}
    scores = {}
    for label, d in directions.items():
        a = q = 0.0
        for x, y in held[0]:
            t = P.directional_terms(model, x, y, d)
            a += float(t["first"].sum())
            q += float(t["q"].sum())
        a, q = a / args.held_sequences, q / args.held_sequences
        changes = {}
        for c in (0.5, 1.0, 2.0):
            with torch.no_grad():
                for k in keys:
                    named[k].copy_(originals[k] + c * d[k])
            changes[str(c)] = held_loss(held[1]) - base
        with torch.no_grad():
            for k in keys:
                named[k].copy_(originals[k])
        scores[label] = {"slope": a, "curvature": q, "quality": a * a / (2 * q) if q > 0 else float("nan"),
                         "c_star": -a / q if q > 0 else float("nan"), "true_change": changes}
        print(json.dumps({"direction": label, **{kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in scores[label].items()}}), flush=True)
    cosines = {}
    labels = list(directions)
    for i, x in enumerate(labels):
        for z in labels[i + 1:]:
            per = [float((directions[x][k] * directions[z][k]).sum() / (directions[x][k].norm() * directions[z][k].norm()).clamp_min(1e-30)) for k in keys]
            cosines[f"{x}|{z}"] = sum(per) / len(per)
    by_kind = {kind: {f: sum(relation[n][f] for n in names if n.endswith("." + kind)) / sum(n.endswith("." + kind) for n in names)
                      for f in ("slope_energy", "corr_energy", "slope_uniform", "corr_uniform", "log_distance_energy_sd", "slope_sqrtv_vs_A")}
               for kind in KINDS}
    result = {"item": args.item, "step": step, "lr": lr, "batch_tokens": config["batch_tokens"], "relation": relation,
              "relation_by_kind": by_kind, "scores": scores, "cosines": cosines, "seconds": time.time() - started}
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"relation_by_kind": {k: {f: round(v, 3) for f, v in r.items()} for k, r in by_kind.items()},
                      "seconds": round(result["seconds"])}), flush=True)


if __name__ == "__main__":
    main()

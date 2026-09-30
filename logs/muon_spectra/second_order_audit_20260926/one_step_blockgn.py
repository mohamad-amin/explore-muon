"""Is the gap to Gauss-Newton within matrices or across them? Exact block-diagonal GN vs full GN, one step.

At a kept checkpoint, for a fresh gradient b (1M or 4M tokens), damped GN directions -(G_P + d rho I)^-1 b where G_P keeps
only the exact GN blocks of a partition P of the 48 hidden matrices:
  full      one block (the full GN; Lanczos 96 steps)
  kind      6 blocks (q, k, v, o, up, down across all layers)
  layer     8 blocks (the six matrices of each transformer block)
  matrix    48 blocks (the per-matrix GN, what a "layerwise" second-order method approximates)
  gs        per-matrix blocks solved in order with block Gauss-Seidel: each block sees the gradient after the earlier
            blocks' solutions, through exact cross-block GN products (one sweep)
Each block is solved by Lanczos on its own exact GN block (restricted forward-mode and reverse-mode products on the
curvature sequences), all with the same absolute dampings d rho (rho: the full GN's curvature along b). References:
Muon and PD alpha 1/2 directions. Scoring is cross-fitted on two halves of the held-out sequences: the damping and the
global scale are chosen on one half and the loss change is measured on the other (and the reverse), which removes the
selection bias of scoring at the in-sample optimum. One-step and local, like one_step_gn.py.

usage: one_step_blockgn.py OUT_JSON ARM_DIR:STEP [--input g4M] [--partitions full,kind,layer,matrix,gs]
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
from one_step_gn import (CURV, EVAL, GRAD, Flat, GaussNewton, fresh_gradient, inverse_roots,  # noqa: E402
                         krylov_direction, lanczos, loss_along, muon_norm, polar)

DAMPINGS = (1e-2, 3e-3, 1e-3, 3e-4, 1e-4)


def halves_score(model, halves, direction):
    """(first, q) of a direction on each half of the held-out sequences."""
    out = []
    for batches in halves:
        first = q = 0.0
        count = 0
        for x, y in batches:
            t = P.directional_terms(model, x, y, direction)
            first += float(t["first"].sum())
            q += float(t["q"].sum())
            count += x.shape[0]
        out.append((first / count, q / count))
    return out


def cross_fit(candidates):
    """candidates: {label: [(first_A, q_A), (first_B, q_B)]}. Pick label and scale on one half, score on the other;
    returns the mean decrease over both directions of the fit, the in-sample best, and the picks."""
    result = {}
    for fit, test in ((0, 1), (1, 0)):
        best = max(candidates, key=lambda k: candidates[k][fit][0] ** 2 / (2 * candidates[k][fit][1])
                   if candidates[k][fit][0] < 0 else -1)
        a, q = candidates[best][fit]
        c = -a / q
        at, qt = candidates[best][test]
        result[f"fit{fit}"] = {"label": best, "scale": c, "decrease": -(c * at + 0.5 * c * c * qt)}
    insample = max(((a0 + a1) / 2) ** 2 / (2 * (q0 + q1) / 2) for (a0, q0), (a1, q1) in candidates.values())
    return {"cross_fitted": 0.5 * (result["fit0"]["decrease"] + result["fit1"]["decrease"]),
            "in_sample": insample, "picks": result}


def block_solve(model, names, curv, b, blocks, steps, dampings, rho):
    """Block-Jacobi: each block's damped GN solve from its own gradient. Returns {damping: direction dict}."""
    out = {d: OrderedDict() for d in dampings}
    for block in blocks:
        sub = OrderedDict((n, b[n]) for n in block)
        flat = Flat(sub)
        op = GaussNewton(model, list(block), curv, flat)
        vec = flat.flat(sub)
        V, T = lanczos(op, vec, min(steps, vec.numel()))
        for d in dampings:
            x = flat.dict(krylov_direction(V, T, float(vec.norm()), V.shape[0], d * rho))
            out[d].update(x)
        del V
    return {d: OrderedDict((n, out[d][n]) for n in names) for d in dampings}


def gauss_seidel(model, names, curv, b, steps, damping, rho, full_flat, order="forward", scope="all"):
    """One block Gauss-Seidel sweep over the matrices: each block solves with the gradient updated by the exact GN
    response to the blocks already solved. order: model order ("forward") or its reverse; scope "layer" restarts from
    the original gradient at every transformer block (Gauss-Seidel within layers, Jacobi across them)."""
    residual = full_flat.flat(b).clone()
    full_op = GaussNewton(model, names, curv, full_flat)
    solution = OrderedDict((n, torch.zeros_like(b[n])) for n in names)
    sweep = list(names) if order == "forward" else list(reversed(names))
    current_layer = None
    for n in sweep:
        if scope == "layer" and n[:7] != current_layer:
            current_layer = n[:7]
            residual = full_flat.flat(b).clone()
        r_n = full_flat.dict(residual)[n]
        flat = Flat(OrderedDict([(n, r_n)]))
        op = GaussNewton(model, [n], curv, flat)
        vec = flat.flat({n: r_n})
        V, T = lanczos(op, vec, min(steps, vec.numel()))
        x = flat.dict(krylov_direction(V, T, float(vec.norm()), V.shape[0], damping * rho))[n]
        del V
        solution[n] = x.clone()
        tangent = OrderedDict((m, x if m == n else torch.zeros_like(b[m])) for m in names)
        residual += full_op(full_flat.flat(tangent))   # the gradient after moving block n (GN model)
    return solution


def gauss_seidel_map(model, names, curv, b, maps, scale, full_flat):
    """Gauss-Seidel with an optimizer's own per-matrix map in place of the exact block solve: matrix n takes the
    direction maps[n](r_n) of its corrected gradient r_n, at a common step size `scale` (the Jacobi direction's
    GN-optimal scale on the curvature sequences), and the later matrices see the gradient after that step
    (exact GN products). The staged-update version of Muon or PD, in the GN model."""
    residual = full_flat.flat(b).clone()
    full_op = GaussNewton(model, names, curv, full_flat)
    solution = OrderedDict()
    for n in names:
        r_n = full_flat.dict(residual)[n].clone()
        x = scale * maps[n](r_n)
        solution[n] = x
        tangent = OrderedDict((m, x if m == n else torch.zeros_like(b[m])) for m in names)
        residual += full_op(full_flat.flat(tangent))
    return solution


def coordinated_map(model, names, curv, b, maps, scale, full_flat, scheme, gamma=1.0):
    """Cheaper coordination of an optimizer's own per-matrix maps (2026-09-29): "layer" stages the maps block by block
    (the six matrices of a transformer block share one residual, then the residual takes the GN product of the whole
    block's step: 8 products instead of 48); "lookahead" is one damped Jacobi correction, every map applied to
    b + gamma * G(D0) with D0 the plain maps' step at `scale` (one GN product)."""
    full_op = GaussNewton(model, names, curv, full_flat)
    if scheme == "lookahead":
        first = OrderedDict((n, scale * maps[n](b[n])) for n in names)
        corrected = full_flat.dict(full_flat.flat(b) + gamma * full_op(full_flat.flat(first)))
        return OrderedDict((n, scale * maps[n](corrected[n].clone())) for n in names)
    residual = full_flat.flat(b).clone()
    solution = OrderedDict()
    blocks = []
    for n in names:
        key = n.split(".")[0]
        if not blocks or blocks[-1][0] != key:
            blocks.append((key, []))
        blocks[-1][1].append(n)
    for _, members in blocks:
        current = full_flat.dict(residual)
        step = OrderedDict((m, torch.zeros_like(b[m])) for m in names)
        for n in members:
            x = scale * maps[n](current[n].clone())
            solution[n] = x
            step[n] = x
        residual += full_op(full_flat.flat(step))
    return OrderedDict((n, solution[n]) for n in names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--input", default="g4M", choices=("g1M", "g4M", "g16M", "momentum1M", "momentum4M", "momentum16M"),
                        help="fresh gradient (1M or 4M tokens), or the next momentum M' = c M + g with g of that size")
    parser.add_argument("--momentum-c", type=float, default=0.95,
                        help="weight c on the saved momentum for the momentum inputs (0.95 = the optimizer's next momentum)")
    parser.add_argument("--partitions", default="full,kind,layer,matrix,gs")
    parser.add_argument("--full-steps", type=int, default=96)
    parser.add_argument("--block-steps", type=int, default=24)
    parser.add_argument("--curvature-sequences", type=int, default=256)
    parser.add_argument("--eval-sequences", type=int, default=512)
    parser.add_argument("--micro", type=int, default=8)
    parser.add_argument("--gs-order", default="forward", choices=("forward", "reverse"))
    parser.add_argument("--gs-scope", default="all", choices=("all", "layer"))
    parser.add_argument("--clip-weight", default="none", choices=("none", "run"),
                        help="momentum inputs: 'run' weights the fresh gradient by the clip factor the training applied at "
                             "step N+1 (min(1, grad_clip / logged pre-clip norm)), as in the saved momentum (review 03:46)")
    parser.add_argument("--true-loss", action="store_true",
                        help="also measure the true held-out loss change of each fit's pick at its scale on the other half")
    parser.add_argument("--gs-damping", type=float, default=None,
                        help="damping (relative to rho) for Gauss-Seidel; default: the per-matrix Jacobi pick of this run")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    arm, step = args.item.rsplit(":", 1)
    step = int(step)
    kept = Path(arm) / "scientific" / "kept"
    kept = kept if kept.exists() else Path(arm) / "kept"
    started = time.time()
    model, _ = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
    T = model.config.seq_len
    names = list(P.hidden_linears(model))
    layers = P.hidden_linears(model)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for n in names:
        layers[n].weight.requires_grad_(True)
    curv = [stream.batch(CURV + first * T, args.micro, T, device) for first in range(0, args.curvature_sequences, args.micro)]
    evals = [stream.batch(EVAL + first * T, args.micro, T, device) for first in range(0, args.eval_sequences, args.micro)]
    halves = [evals[:len(evals) // 2], evals[len(evals) // 2:]]
    # 16M inputs added 2026-09-29 (large-batch staging question); 1M/4M unchanged
    size = args.input[-3:].lstrip("gmomentu")
    # the 16M gradient (32768 sequences) gets its own held-out region past GRAD/CURV/EVAL, which hold <= 8192 sequences
    offset = GRAD if size != "16M" else 2 * 1048576 + 100000 * 512
    b = fresh_gradient(model, names, stream, offset, {"1M": 2048, "4M": 8192, "16M": 32768}[size], T, device)
    if args.input.startswith("momentum"):
        saved = torch.load(kept / f"step{step:06d}.pt", map_location="cpu", weights_only=False)
        state = saved["optimizer"]["state"]
        weight = 1.0
        if args.clip_weight == "run":
            record_next = json.loads((Path(arm) / "scientific" / "steps" / f"step{step + 1:06d}.json").read_text())
            clip = saved["config"]["grad_clip"]
            weight = min(1.0, clip / record_next["gradient_norm_before_clip"]) if clip > 0 else 1.0
        b = OrderedDict((n, args.momentum_c * state[i]["momentum_buffer"].to(device).float() + weight * b[n]) for i, n in enumerate(names))
        clip_weight_used = weight
        del saved
    full_flat = Flat(OrderedDict((n, layers[n].weight) for n in names))
    full_op = GaussNewton(model, names, curv, full_flat)
    b_flat = full_flat.flat(b)
    rho = float(b_flat @ full_op(b_flat / b_flat.norm()) / b_flat.norm())
    result = {"arm": arm, "step": step, "input": args.input, "momentum_c": args.momentum_c, "rho": rho, "families": {},
              "clip_weight": args.clip_weight, "clip_weight_used": locals().get("clip_weight_used")}

    def record(family, candidates):
        scored = {label: halves_score(model, halves, d) for label, d in candidates.items()}
        result["families"][family] = {"cross_fit": cross_fit(scored), "raw": scored}
        cf = result["families"][family]["cross_fit"]
        if args.true_loss:
            true = []
            for fit, test in (("fit0", 1), ("fit1", 0)):
                pick = cf["picks"][fit]
                base_loss, moved = loss_along(model, halves[test], candidates[pick["label"]], [0.0, pick["scale"]])
                true.append(base_loss - moved)
            cf["true_decrease"] = sum(true) / 2
        print(json.dumps({"family": family, "cross_fitted_x1e3": round(cf["cross_fitted"] * 1e3, 4),
                          "true_x1e3": round(cf["true_decrease"] * 1e3, 4) if "true_decrease" in cf else None,
                          "in_sample_x1e3": round(cf["in_sample"] * 1e3, 4), "picks": [cf["picks"]["fit0"]["label"], cf["picks"]["fit1"]["label"]],
                          "seconds": round(time.time() - started)}), flush=True)
        args.out.write_text(json.dumps(result, indent=1) + "\n")

    # references: Muon and PD alpha 1/2 (C from the curvature sequences, positions >= 1)
    recorder = P.Recorder(model)
    sums = {n: 0 for n in names}
    count = 0
    with torch.no_grad():
        for x, _ in curv:
            model(x)
            for n in names:
                inp = recorder.inputs[n][:, 1:].reshape(-1, recorder.inputs[n].shape[-1]).double()
                sums[n] = sums[n] + inp.T @ inp
            count += x.shape[0] * (T - 1)
    recorder.remove()
    muon, pd = OrderedDict(), OrderedDict()
    for n in names:
        g = b[n].float()
        muon[n] = -polar(g) * math.sqrt(max(1.0, g.shape[0] / g.shape[1]))
        r = inverse_roots(sums[n] / count, (0.5,))[0.5]
        d = polar(g @ r) @ r
        pd[n] = -d * (muon_norm(g.shape) / d.norm().clamp_min(1e-30))
    del sums
    record("muon", {"muon": muon})
    record("pd_a0.5", {"pd_a0.5": pd})
    partitions = args.partitions.split(",")
    blocks = {"full": [names], "kind": [[n for n in names if n.endswith("." + k)] for k in P.KINDS],
              "layer": [[n for n in names if n.startswith(f"block{i:02d}.")] for i in range(1, model.config.n_layer + 1)],
              "matrix": [[n] for n in names]}
    for family in partitions:
        if family in ("gs", "gsmap", "gsmap_only_extra", "gslayer", "lookahead"):
            continue
        steps = args.full_steps if family == "full" else args.block_steps
        directions = block_solve(model, names, curv, b, blocks[family], steps, DAMPINGS, rho)
        record(family, {f"{family}_d{d:g}": dirs for d, dirs in directions.items()})
    if "gsmap" in partitions:
        # staged (Gauss-Seidel) versions of Muon and PD alpha 1/2: same per-matrix maps, sequential gradients
        roots = {}
        recorder = P.Recorder(model)
        sums, count = {n: 0 for n in names}, 0
        with torch.no_grad():
            for x, _ in curv:
                model(x)
                for n in names:
                    inp = recorder.inputs[n][:, 1:].reshape(-1, recorder.inputs[n].shape[-1]).double()
                    sums[n] = sums[n] + inp.T @ inp
                count += x.shape[0] * (T - 1)
        recorder.remove()
        roots = {n: inverse_roots(sums[n] / count, (0.5,))[0.5] for n in names}
        del sums

        def muon_map(n):
            return lambda g: -polar(g.float()) * math.sqrt(max(1.0, g.shape[0] / g.shape[1]))

        def pd_map(n):
            def f(g):
                d = polar(g.float() @ roots[n]) @ roots[n]
                return -d * (muon_norm(g.shape) / d.norm().clamp_min(1e-30))
            return f
        extra = [p for p in partitions if p in ("gslayer", "lookahead")]
        for label, base, maker in (("gsmuon", muon, muon_map), ("gspd", pd, pd_map)):
            if "gsmap_only_extra" in partitions:
                break
            d_flat = full_flat.flat(base)
            scale = float(-(b_flat @ d_flat) / (d_flat @ full_op(d_flat)))
            candidates = {}
            for mult in (0.5, 1.0, 2.0):     # the staged update depends on the step size the earlier matrices take
                candidates[f"{label}_x{mult:g}"] = gauss_seidel_map(model, names, curv, b, {n: maker(n) for n in names},
                                                                    mult * scale, full_flat)
            record(label, candidates)
        for label, base, maker in (("muon", muon, muon_map), ("pd", pd, pd_map)):
            d_flat = full_flat.flat(base)
            scale = float(-(b_flat @ d_flat) / (d_flat @ full_op(d_flat)))
            if "gslayer" in extra:
                candidates = {f"layer{label}_x{mult:g}": coordinated_map(model, names, curv, b, {n: maker(n) for n in names},
                                                                          mult * scale, full_flat, "layer") for mult in (0.5, 1.0, 2.0)}
                record(f"layer{label}", candidates)
            if "lookahead" in extra:
                candidates = {f"look{label}_g{g:g}_x{mult:g}": coordinated_map(model, names, curv, b, {n: maker(n) for n in names},
                                                                               mult * scale, full_flat, "lookahead", g)
                              for g in (0.25, 0.5, 1.0) for mult in (0.5, 1.0)}
                record(f"look{label}", candidates)
    if "gs" in partitions:
        jacobi = result["families"].get("matrix", {}).get("cross_fit", {}).get("picks", {})
        damping = args.gs_damping if args.gs_damping is not None else (
            float(jacobi["fit0"]["label"].split("_d")[1]) if jacobi else 1e-3)
        label = "gs" if (args.gs_order, args.gs_scope) == ("forward", "all") else f"gs_{args.gs_order}_{args.gs_scope}"
        record(label, {f"{label}_d{damping:g}": gauss_seidel(model, names, curv, b, args.block_steps, damping, rho, full_flat,
                                                              args.gs_order, args.gs_scope)})
    result["seconds"] = time.time() - started
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({"done": True, "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()

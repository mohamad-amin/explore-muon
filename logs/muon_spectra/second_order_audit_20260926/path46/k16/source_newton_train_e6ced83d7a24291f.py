"""One damped Gauss-Newton step per batch (Hessian-free style), as the reference for what a second-order direction buys
at a fixed batch size (after the 12:33 CDT finding that the GN paper's inner-loop recipe mostly buys "more, smaller
steps per batch").

Outer step s (the run's own batch and data order, sequences split over ranks as in distributed.py):
  - g: the batch gradient at theta_s (BF16 autocast, mean token loss), all-reduced;
  - x(d) = -(G + d rho I)^-1 g_body for the 48 body matrices from one k-step Lanczos run on the exact GN matrix
    (forward-mode + reverse-mode products, explicit attention) on the first `--curvature-sequences` of each rank's
    batch, all-reduced, for every d in DAMPINGS (rho = g.G g / g.g);
  - joint line search: theta_body += a x(d) over d in DAMPINGS and a in LINE times the run's schedule factor (warmup,
    constant, linear cooldown to 0, as the baselines' LR), picked by the true loss on held-out training sequences
    (tokens beyond every run's budget). No damping heuristic: the held-out loss chooses damping and step together,
    as the cross-fitted one-step analyses do;
  - with --normalize d instead (GN's shape in the normalized regime, 15:21 CDT decision): no line search; x(d)'s part for
    each body matrix is rescaled to the baseline map's step norm times the run's LR, lr_ref(s) |NS(M)| sqrt(max(1, m/n))
    for Muon runs and lr_ref(s) sqrt(min(m, n)) sqrt(max(1, m/n)) for data-norm runs, so only the direction differs;
  - decoupled weight decay on the body matched to the baseline's per-step shrink (lr_ref(s) * wd);
  - embeddings, gains and head: AdamW with the checkpoint's own state and the run's aux schedule (as in training).
Starts from a kept checkpoint. Validation as in distributed.py, every --validation-every steps.
Added 2026-09-28 (GN-PD as a rate at 16M, 09:40 CDT decision), all off by default: --kron-control (GN-PD's Kronecker
limit, TS p/p with the step's own B and C), --log-kron (per-matrix cosine of the GN-PD step with that map), --clip
(the trainer's global gradient clipping), --sharpness-every (top GN eigenvalue from a fixed random start), --keep
(kept checkpoints) and --transport (the momentum carried to the current weights). Each run stores a copy of this
source and its sha256.

usage: .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node=4 newton_train.py OUT_DIR ARM_DIR:STEP [--cg 16 (Lanczos steps)] [--curvature-sequences 32]
       [--stop-after N]
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
from torch.func import functional_call, jvp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream, token_views  # noqa: E402
from research.adamw_spectra.distributed import partition_sequences  # noqa: E402
from research.adamw_spectra.muon import _into_basis, _out_of_basis, new_soap_state, newton_schulz, ns_schedule, \
    soap_precondition, soap_update_statistics  # noqa: E402
from research.adamw_spectra.train import learning_rate, token_budget  # noqa: E402

LINE = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
DAMPINGS = (1e-1, 3e-2, 1e-2, 1e-3)     # relative to rho = g.G g / g.g, the curvature along g
HELD_OUT_TRAIN = 2_500_000_000


def bf16():
    return torch.autocast("cuda", dtype=torch.bfloat16)


class GNProduct:
    """v -> G v for the body matrices on this rank's curvature sequences, summed over ranks (mean token loss)."""

    def __init__(self, model, body_keys, xs, total_tokens, micro, fp32=False):
        self.model, self.keys, self.xs, self.total, self.micro = model, body_keys, xs, total_tokens, micro
        self.precision = contextlib.nullcontext if fp32 else bf16     # FP32 products for amplifying maps (pre-flight)

    def __call__(self, v_parts):
        base = {**dict(self.model.named_parameters()), **dict(self.model.named_buffers())}
        weights = [base[k] for k in self.keys]
        out = [torch.zeros_like(w) for w in weights]
        for i in range(0, self.xs.shape[0], self.micro):
            x = self.xs[i:i + self.micro]

            def logits_of(*ws):
                return functional_call(self.model, {**base, **dict(zip(self.keys, ws))}, (x,))
            with torch.no_grad(), P.explicit_attention(), self.precision():
                logits, dz = jvp(logits_of, tuple(w.detach() for w in weights), tuple(v.to(w.dtype) for v, w in zip(v_parts, weights)))
            p = torch.softmax(logits.float(), dim=-1)
            dz = dz.float()
            hdz = (p * (dz - (p * dz).sum(-1, keepdim=True))) / self.total
            del logits, dz, p
            with self.precision():
                z = self.model(x)
            grads = torch.autograd.grad((z.float() * hdz).sum(), weights)
            for acc, g in zip(out, grads):
                acc += g.float()
        flat = torch.cat([o.reshape(-1) for o in out])
        dist.all_reduce(flat)
        return [part.view_as(o) for part, o in zip(flat.split([o.numel() for o in out]), out)]


def dot(a, b):
    return sum(float((x * y).sum()) for x, y in zip(a, b))


@torch.no_grad()
def summed_loss(model, x_all, y_all, micro):
    total = torch.zeros((), device=x_all.device, dtype=torch.float64)
    for i in range(0, x_all.shape[0], micro):
        with bf16():
            total += model(x_all[i:i + micro], y_all[i:i + micro]).double() * x_all[i:i + micro].numel()
    dist.all_reduce(total)
    return float(total)


def kron_roots(model, keys, xs, ys, micro, power, damping, generator):
    """Per body matrix (L, R) = ((B / mean eig + damping)^-power, (C / mean eig + damping)^-power) from this step's
    curvature sequences, summed over ranks: B the sampled-label output factor, C the input second moment over
    positions >= 1 (the trainer's TS and PD statistics, instantaneous rather than EMA)."""
    names = list(P.hidden_linears(model))
    by_key = dict(zip(P.parameter_keys(model, names), names))
    recorder = P.Recorder(model, [by_key[k] for k in keys])
    T = xs.shape[1]
    sums = {k: [0, 0] for k in keys}
    for i in range(0, xs.shape[0], micro):
        P.gradient_passes(model, recorder, xs[i:i + micro], ys[i:i + micro], generator, draws=1)
        for k in keys:
            x = recorder.inputs[by_key[k]][:, 1:].float()
            e = T * recorder.errors[by_key[k]][1].float()
            x, e = x.reshape(-1, x.shape[-1]), e.reshape(-1, e.shape[-1])
            sums[k][0] = sums[k][0] + e.T @ e
            sums[k][1] = sums[k][1] + x.T @ x
        recorder.clear()
    recorder.remove()
    roots = {}
    for k in keys:
        pair = []
        for moment in sums[k]:
            dist.all_reduce(moment)
            values, vectors = torch.linalg.eigh(0.5 * (moment + moment.T).double())
            unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
            pair.append(((vectors * (unit + damping).pow(-power)) @ vectors.T).float())
        roots[k] = tuple(pair)
    return roots


@torch.no_grad()
def soap_peek(update, soap, beta2, denom_power, eps=1e-8):
    """soap_precondition without touching the statistics: the same output as the trainer's call on this update
    ('update' second moment, entrywise), for directions that must not count as a step (--stage-jacobi, 16:27 CDT)."""
    if not soap["ready"]:
        return update
    projected = _into_basis(update, soap["q_row"], soap["q_col"])
    second = beta2 * soap["exp_avg_sq"] + (1 - beta2) * projected.square()
    preconditioned = _out_of_basis(projected / second.clamp_min(eps * eps).pow(denom_power), soap["q_row"], soap["q_col"])
    return preconditioned * (update.norm() / preconditioned.norm().clamp_min(1e-30))


def pd_roots(model, keys, xs, micro, alpha, damping, band=None, ema=None, ema_beta=None):
    """PD's input root R = (C / mean eig + damping)^-alpha per body matrix from this step's curvature sequences (C over
    positions >= 1, summed over ranks): the trainer's PD statistic, instantaneous rather than EMA (2026-09-29).
    band="top" / "bulk": also return, per matrix, the projector onto C's above-mean eigenvectors (u > 1, PD-top's split)
    or onto the rest (added 07:40 CDT for --stage-band); band="topK" (e.g. "top4"): onto C's K largest eigenvectors;
    band="mean": onto the input mean's direction, positions >= 1 (added 10:30 CDT).
    ema (a dict kept by the caller) with ema_beta: C (per token) is an EMA over steps, C <- b C + (1 - b) C_step, and R and
    the band come from it (09-30 02:38 CDT: time-averaged statistics for a cheap curvature batch)."""
    names = list(P.hidden_linears(model))
    by_key = dict(zip(P.parameter_keys(model, names), names))
    recorder = P.Recorder(model, [by_key[k] for k in keys])
    sums = {k: 0 for k in keys}
    means = {k: 0 for k in keys}
    for i in range(0, xs.shape[0], micro):
        with bf16():
            model(xs[i:i + micro])
        for k in keys:
            x = recorder.inputs[by_key[k]][:, 1:].float()
            x = x.reshape(-1, x.shape[-1])
            sums[k] = sums[k] + x.T @ x
            if band == "mean":
                means[k] = means[k] + x.sum(0)
        recorder.clear()
    recorder.remove()
    roots, projectors = {}, {}
    count = torch.tensor(float(xs.shape[0] * max(xs.shape[1] - 1, 1)), device=xs.device)
    dist.all_reduce(count)
    for k in keys:
        moment = sums[k]
        dist.all_reduce(moment)
        if ema is not None:     # per-token second moment, averaged over steps
            moment = moment / count
            ema[k] = moment.clone() if k not in ema else ema[k].mul_(ema_beta).add_(moment, alpha=1 - ema_beta)
            moment = ema[k]
        values, vectors = torch.linalg.eigh(0.5 * (moment + moment.T).double())
        unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
        roots[k] = ((vectors * (unit + damping).pow(-alpha)) @ vectors.T).float()
        if band == "mean":
            m = means[k]
            dist.all_reduce(m)
            m = (m / m.norm().clamp_min(1e-30)).double()
            projectors[k] = torch.outer(m, m).float()
        elif band is not None and band.startswith("top") and band != "top":
            chosen = vectors[:, -int(band[3:]):]      # eigh sorts ascending: the K largest
            projectors[k] = (chosen @ chosen.T).float()
        elif band is not None:
            chosen = vectors[:, unit > 1] if band == "top" else vectors[:, unit <= 1]
            projectors[k] = (chosen @ chosen.T).float()
    if band is not None:
        return roots, projectors
    return roots


def kron_direction(roots, keys, parts):
    """-L polar(L M R) R per matrix (polar by SVD): GN-PD's limit when G = B (x) C."""
    out = []
    for k, m in zip(keys, parts):
        left, right = roots[k]
        u, _, vh = torch.linalg.svd(left @ m.float() @ right, full_matrices=False)
        out.append(-(left @ (u @ vh) @ right).reshape(-1))
    return torch.cat(out)


def top_eigenvalue(op, n, k, device, seed=0):
    """Largest GN eigenvalue (top Ritz value of a k-step Lanczos run, full reorthogonalization) from a fixed random
    start that is identical on every rank, so successive steps and arms are measured the same way."""
    start = torch.randn(n, generator=torch.Generator().manual_seed(seed)).to(device)
    V = torch.zeros(k, n, device=device)
    V[0] = start / start.norm()
    alphas, betas = [], []
    for j in range(k):
        w = op(V[j])
        alphas.append(float(w @ V[j]))
        for _ in range(2):
            w -= V[:j + 1].T @ (V[:j + 1] @ w)
        if j + 1 == k:
            break
        betas.append(float(w.norm()))
        V[j + 1] = w / betas[-1]
    Tk = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    off = torch.tensor(betas, dtype=torch.float64)
    return float(torch.linalg.eigvalsh(Tk + torch.diag(off, 1) + torch.diag(off, -1))[-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("item")
    parser.add_argument("--cg", type=int, default=16)
    parser.add_argument("--curvature-sequences", type=int, default=32, help="per rank")
    parser.add_argument("--damping", type=float, default=1e-3, help="initial lambda relative to the curvature along g")
    parser.add_argument("--warm", type=float, default=0.95)
    parser.add_argument("--micro", type=int, default=16)
    parser.add_argument("--line-sequences", type=int, default=32, help="held-out training sequences per rank")
    parser.add_argument("--stop-after", type=int, default=None)
    parser.add_argument("--validation-every", type=int, default=50)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--overshoot", type=float, default=1.0,
                        help="take kappa times the line-search step (the successful optimizers step ~2x their own optimum)")
    parser.add_argument("--normalize", type=float, default=None, metavar="DAMPING",
                        help="GN's shape in the normalized regime: the damped GN direction at this relative damping, each "
                             "body matrix's step rescaled to the baseline map's norm (Muon: |NS(M)| sqrt(max(1, m/n)); "
                             "data-norm runs: sqrt(min(m, n)) sqrt(max(1, m/n))) times the run's LR; no line search")
    parser.add_argument("--polar-control", action="store_true",
                        help="with --normalize: Muon's own direction NS(M) in place of GN's (no Lanczos), the harness "
                             "control that should reproduce the Muon baseline")
    parser.add_argument("--polar-direction", action="store_true",
                        help="the step-rule control (review, 16:40 CDT): the joint held-out line search runs on Muon's own "
                             "step, -lr_peak sqrt(max(1, m/n)) NS(M) per matrix, in place of GN's direction ('greedy Muon')")
    parser.add_argument("--gnpd", type=float, default=None, metavar="POWER",
                        help="with --normalize d: PD in the exact GN geometry (gnpd_probe.py, 19:07 CDT), "
                             "-(G + d rho)^-p polar((G + d rho)^-p M) per matrix, two Lanczos runs per step, then the "
                             "normalized step (the run's LR and per-matrix norms)")
    parser.add_argument("--lr-scale", type=float, default=1.0,
                        help="with --normalize: body step = lr-scale x the run's LR (the new direction's own LR; weight "
                             "decay keeps the run's lr x wd)")
    parser.add_argument("--momentum", type=float, default=0.0,
                        help="EMA of batch gradients as the Newton right-hand side (0: the batch gradient); initialized "
                             "from the checkpoint's momentum buffer as (1 - beta_run) M")
    parser.add_argument("--kron-control", action="store_true",
                        help="with --normalize d --gnpd p: GN-PD's Kronecker limit L polar(L M R) R (TS p/p) with L, R the "
                             "-p roots of the step's own B and C (damping d, as the trainer's data norm); no Lanczos")
    parser.add_argument("--log-kron", action="store_true",
                        help="with --gnpd: also compute the --kron-control map each step and log each matrix's cosine "
                             "with the step taken")
    parser.add_argument("--clip", type=float, default=None,
                        help="clip the batch gradient's global norm (all parameters) as the trainer does, before the "
                             "momentum average and the aux AdamW (default: no clipping, as in the earlier runs)")
    parser.add_argument("--sharpness-every", type=int, default=0,
                        help="every N steps, the top GN eigenvalue on the curvature sequences (Lanczos 20 from a fixed "
                             "random start), measured before the step")
    parser.add_argument("--keep", default="",
                        help="comma-separated steps whose checkpoint is also kept as kept/stepNNNNNN.pt (post-hoc probes)")
    parser.add_argument("--staged-pd", type=float, default=None, metavar="ALPHA",
                        help="with --normalize (any value) and --momentum b: PD's map (NS polar, R from the step's curvature "
                             "sequences, damping 1e-3) on the momentum average, applied layer by layer; each layer sees the "
                             "average plus clip-factor x G (earlier layers' actual steps) (2026-09-29, MUON_CASE 04:21 CDT)")
    parser.add_argument("--stage-band", default="all",
                        help="with --staged-pd: keep each matrix's cross-layer correction only on its above-mean input "
                             "directions (top: C's eigenvectors with eigenvalue > mean, PD-top's split) or only on the rest "
                             "(bulk); topK: C's K largest eigenvectors; mean: the input mean's direction; default all "
                             "(2026-09-29 07:40 / 10:30 CDT, band-restricted coordination)")
    parser.add_argument("--stage-scale", type=float, default=1.0,
                        help="with --staged-pd: multiply the cross-layer correction by this factor (1 = the staged map; "
                             "2026-09-29 08:45 CDT, the correction-size test)")
    parser.add_argument("--stage-dual", action="store_true",
                        help="with --staged-pd: scale each corrected matrix's step by min(1, |r R|_* / |m R|_*), the dual "
                             "(nuclear) norm of its whitened residual over that of its whitened momentum, so a layer whose "
                             "gradient the earlier layers already satisfied shrinks instead of turning against them "
                             "(2026-09-29 10:35 CDT, normalized Gauss-Seidel)")
    parser.add_argument("--soap", action="store_true",
                        help="with --staged-pd: S∘PD's map instead of PD's (the trainer's soap_precondition in PD's whitened "
                             "coordinates, then NS, then R; statistics from the fresh clipped gradient G R after each "
                             "matrix's step; beta2 0.9, power 1/2, basis both, update second moment). The SOAP statistics are "
                             "not in the checkpoints, so they start at the branch point (2026-09-29 14:46 CDT)")
    parser.add_argument("--stage-jacobi", action="store_true",
                        help="with --staged-pd: each block's correction comes from the earlier blocks' plain (uncorrected) "
                             "steps instead of their corrected ones (Jacobi; the corrections could run in parallel in a "
                             "trainer); second review 16:23 CDT")
    parser.add_argument("--stage-reversal-only", action="store_true",
                        help="with --staged-pd and a band: keep only each later matrix's correction component along its own "
                             "momentum's band part A'P (online coefficient <c, A'P>/|A'P|^2), dropping the off-axis part "
                             "(the oracle per-matrix reversal control, second review 16:23 CDT)")
    parser.add_argument("--stage-granularity", choices=("layer", "sublayer", "matrix"), default="layer",
                        help="with --staged-pd: the sweep's stages. layer: a layer's six matrices together (default); "
                             "sublayer: q/k/v, then o, then up, then down within each layer (31 GN products per step; 19:58 CDT)")
    parser.add_argument("--stage-reverse", action="store_true",
                        help="with --staged-pd: run the sweep from the last block to the first (19:06 CDT)")
    parser.add_argument("--stage-c-ema", type=float, default=None, metavar="BETA",
                        help="with --staged-pd: PD's root and the band from an EMA over steps of the curvature batch's "
                             "per-token C (the GN products still use this step's curvature sequences)")
    parser.add_argument("--stage-sweeps", type=int, default=1,
                        help="with --staged-pd: number of Gauss-Seidel sweeps; 2 adds a backward sweep in which every stage "
                             "is recomputed against all other stages' current steps (symmetric Gauss-Seidel), 3 a further "
                             "forward one; SOAP statistics then move once per step, on the final input (09-30 10:36 CDT)")
    parser.add_argument("--stage-off", action="store_true",
                        help="with --staged-pd: the same PD map with no cross-layer correction (the harness control)")
    parser.add_argument("--linearize-at-start", action="store_true",
                        help="every step's gradient is the gradient of its batch's GN model at the start weights theta_0, "
                             "J0^T [grad l(z0) + H0 J0 (theta - theta_0)] (the GN paper's inner-loop model), for the "
                             "step-vs-path probe (MUON_CASE 2026-09-30 12:1x CDT); train_nll then logs the model's value")
    parser.add_argument("--fixed-root", type=int, default=0, metavar="SEQUENCES",
                        help="with --staged-pd: PD's root R (and the band projectors) computed once, at the start weights, "
                             "from SEQUENCES held-out training sequences per rank, and reused at every step")
    parser.add_argument("--freeze-aux", action="store_true",
                        help="no auxiliary-parameter update (the aux AdamW step, including its decay, is skipped)")
    parser.add_argument("--anneal-from", type=int, default=None, metavar="STEP",
                        help="with --anneal-steps K: after STEP, the scheduled LR times 1 - (k - 1) / K at the k-th step "
                             "(anneal_branch.py's 16-step anneal in the harness; floor ledger, 2026-09-29 07:19 CDT); "
                             "validation is also logged at STEP")
    parser.add_argument("--anneal-steps", type=int, default=None)
    parser.add_argument("--transport", action="store_true",
                        help="with --normalize and --momentum b: the direction is computed from the momentum carried to "
                             "the current weights, A + (1 - b) G D with D <- b D + b/(1 - b) dW (floor_steps.py, 02:29 CDT), "
                             "one extra GN product per step; the stored average stays the plain EMA")
    args = parser.parse_args()
    keep = {int(v) for v in args.keep.split(",") if v}
    if args.stage_sweeps < 1 or (args.stage_sweeps > 1 and (args.staged_pd is None or args.stage_off or args.stage_jacobi
                                                              or args.stage_reversal_only or args.stage_dual)):
        parser.error("--stage-sweeps > 1 needs --staged-pd, and no --stage-off / --stage-jacobi / --stage-reversal-only / --stage-dual")
    if (args.stage_jacobi or args.stage_reversal_only) and (args.staged_pd is None or args.stage_off):
        parser.error("--stage-jacobi / --stage-reversal-only need --staged-pd and no --stage-off")
    if args.stage_reversal_only and args.stage_band == "all":
        parser.error("--stage-reversal-only needs a --stage-band (the band part of the momentum defines the axis)")
    if args.fixed_root and (args.staged_pd is None or args.stage_c_ema is not None):
        parser.error("--fixed-root needs --staged-pd and no --stage-c-ema")
    if args.soap and args.staged_pd is None:
        parser.error("--soap needs --staged-pd (the PD maps in the staged harness)")
    if args.stage_dual and (args.staged_pd is None or args.stage_off):
        parser.error("--stage-dual needs --staged-pd and no --stage-off")
    if args.stage_scale != 1.0 and (args.staged_pd is None or args.stage_off):
        parser.error("--stage-scale needs --staged-pd and no --stage-off")
    if not (args.stage_band in ("all", "top", "bulk", "mean") or (args.stage_band.startswith("top") and args.stage_band[3:].isdigit())):
        parser.error("--stage-band: all, top, bulk, mean or topK")
    if args.stage_band != "all" and (args.staged_pd is None or args.stage_off):
        parser.error("--stage-band top/bulk needs --staged-pd and no --stage-off")
    if (args.anneal_from is None) != (args.anneal_steps is None) or (args.anneal_steps is not None and args.anneal_steps < 1):
        parser.error("--anneal-from and --anneal-steps K >= 1 go together")
    if args.transport and (args.normalize is None or args.momentum <= 0):
        parser.error("--transport needs --normalize d and --momentum b > 0")
    if args.kron_control and (args.gnpd is None or args.normalize is None or args.polar_control or args.polar_direction):
        parser.error("--kron-control needs --normalize d and --gnpd p, and no polar modes")
    if args.staged_pd is not None and (args.normalize is None or args.momentum <= 0 or args.gnpd is not None
                                       or args.kron_control or args.polar_control or args.polar_direction or args.transport):
        parser.error("--staged-pd needs --normalize and --momentum b, and no other direction mode")
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    arm, start = args.item.rsplit(":", 1)
    start = int(start)
    kept = Path(arm) / "scientific" / "kept"
    model, saved = P.load_checkpoint(kept / f"step{start:06d}.pt", device)
    config = dict(saved["config"])
    T = model.config.seq_len
    named = dict(model.named_parameters())
    body_keys = [k for k in named if k.startswith("blocks.") and named[k].ndim == 2]
    aux_keys = [k for k in named if k not in body_keys]
    body = [named[k] for k in body_keys]
    aux = [named[k] for k in aux_keys]
    # AdamW for the aux parameters with the checkpoint's own moments (MuonAdamW keeps body first, aux second)
    aux_opt = torch.optim.AdamW(aux, lr=config["aux_learning_rate"], betas=tuple(config["betas"]), eps=config["epsilon"],
                                weight_decay=config["weight_decay"], fused=True)
    saved_state = saved["optimizer"]["state"]
    aux_indices = saved["optimizer"]["param_groups"][1]["params"]
    aux_opt.load_state_dict({"state": {j: saved_state[i] for j, i in enumerate(aux_indices) if i in saved_state},
                             "param_groups": [{**aux_opt.state_dict()["param_groups"][0], "params": list(range(len(aux)))}]})
    average = None
    if args.momentum > 0:
        body_indices = saved["optimizer"]["param_groups"][0]["params"]
        average = torch.cat([(1 - config["muon_momentum"]) * saved_state[i]["momentum_buffer"].float().reshape(-1)
                             for i in body_indices]).to(device)
    displacement = None     # --transport: D = sum_k b^k (theta_s - theta_{s-k}), zero at the start (no history)
    soap_states = {} if args.soap else None     # --soap: per-matrix SOAP statistics, from the branch point
    c_ema = {} if args.stage_c_ema is not None else None     # --stage-c-ema: per-matrix EMA of C, from the branch point
    del saved
    budget = token_budget(config, sum(p.numel() for p in named.values()), T)
    steps_total = math.ceil(budget / config["batch_tokens"])
    stop = min(steps_total, start + args.stop_after) if args.stop_after else steps_total
    train = TokenStream(str(REPO / config["train_pattern"]))
    validation = TokenStream(str(REPO / config["validation_pattern"]))
    val_start, val_sequences = partition_sequences(config["validation_tokens"] // T, world, rank)
    val_x, val_y = token_views(validation.device_tokens(val_start * T, val_sequences * T + 1, device), 0, val_sequences * T, T)
    line_x, line_y = token_views(train.device_tokens(HELD_OUT_TRAIN + rank * args.line_sequences * T,
                                                     args.line_sequences * T + 1, device), 0, args.line_sequences * T, T)
    line_count = world * args.line_sequences * T
    model0 = None
    if args.linearize_at_start:     # the start weights theta_0, frozen: the GN model's linearization point
        model0, _ = P.load_checkpoint(kept / f"step{start:06d}.pt", device)
    fixed_x, fixed_cache = None, None
    if args.fixed_root:     # held-out sequences beyond the line sequences (HELD_OUT_TRAIN + rank * line_sequences * T)
        fixed_x, _ = token_views(train.device_tokens(HELD_OUT_TRAIN + 100_000_000 + rank * args.fixed_root * T,
                                                     args.fixed_root * T + 1, device), 0, args.fixed_root * T, T)
    peak = config["learning_rate"]
    schedule_ns = ns_schedule(config)
    if rank == 0:
        (args.out / "steps").mkdir(parents=True, exist_ok=True)
        source = Path(__file__).resolve().read_bytes()
        source_name = f"source_newton_train_{hashlib.sha256(source).hexdigest()[:16]}.py"
        (args.out / source_name).write_bytes(source)
        (args.out / "run.json").write_text(json.dumps({"item": args.item, "args": {k: str(v) for k, v in vars(args).items()},
                                                       "world": world, "stop": stop, "config": config,
                                                       "source": source_name,
                                                       "source_sha256": hashlib.sha256(source).hexdigest()}, indent=1) + "\n")
    first = start + 1
    resume = args.out / "checkpoint.pt"
    if resume.exists():     # model, aux AdamW state, CG warm start and damping from the last checkpoint
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        aux_opt.load_state_dict(state["aux"])
        first = state["step"] + 1
        if average is not None and state.get("average") is not None:
            average = state["average"].to(device)
        if state.get("displacement") is not None:
            displacement = state["displacement"].to(device)
        del state
    started = time.time()
    if first == start + 1:
        value = summed_loss(model, val_x, val_y, args.micro) / config["validation_tokens"]
        if rank == 0:
            print(json.dumps({"step": start, "validation_nll": value}), flush=True)
    for s in range(first, stop + 1):
        t0 = time.time()
        tokens_before = (s - 1) * config["batch_tokens"]
        lr_ref = learning_rate(config, s, tokens_before, budget)
        if args.anneal_steps is not None and s > args.anneal_from:
            lr_ref *= max(0.0, 1 - (s - args.anneal_from - 1) / args.anneal_steps)
        schedule = lr_ref / peak
        local_start, local_sequences = partition_sequences(config["batch_tokens"] // T, world, rank)
        x_all, y_all = token_views(train.device_tokens(tokens_before + local_start * T, local_sequences * T + 1, device),
                                   0, local_sequences * T, T)
        # batch gradient (all parameters), mean over the global batch's tokens
        for p in named.values():
            p.grad = None
        train_total = torch.zeros((), device=device, dtype=torch.float64)
        if model0 is not None:      # --linearize-at-start: the batch's GN model at theta_0, evaluated at theta
            base0 = {**dict(model0.named_parameters()), **dict(model0.named_buffers())}
            keys0 = list(dict(model0.named_parameters()))
            w0 = [base0[k] for k in keys0]
            tangent = tuple((named[k].detach() - base0[k].detach()).to(base0[k].dtype) for k in keys0)
            for p in model0.parameters():
                p.grad = None
            for i in range(0, local_sequences, args.micro):
                xm, ym = x_all[i:i + args.micro], y_all[i:i + args.micro]

                def logits_of(*ws):
                    return functional_call(model0, {**base0, **dict(zip(keys0, ws))}, (xm,))
                with torch.no_grad(), P.explicit_attention(), bf16():
                    z0, dz = jvp(logits_of, tuple(w.detach() for w in w0), tangent)
                z0, dz = z0.float(), dz.float()
                prob = torch.softmax(z0, dim=-1)
                hdz = prob * (dz - (prob * dz).sum(-1, keepdim=True))
                lse = torch.logsumexp(z0, dim=-1)
                picked = z0.gather(-1, ym.unsqueeze(-1)).squeeze(-1)
                picked_dz = dz.gather(-1, ym.unsqueeze(-1)).squeeze(-1)
                # value l(z0) + grad.dz + dz.H dz / 2, summed over the microbatch's tokens
                train_total += ((lse - picked).sum() + ((prob * dz).sum(-1) - picked_dz).sum() + 0.5 * (dz * hdz).sum()).double()
                grad_z = prob
                grad_z.scatter_add_(-1, ym.unsqueeze(-1), -torch.ones_like(picked).unsqueeze(-1))
                weight = (grad_z + hdz) * (world / config["batch_tokens"])
                del z0, dz, prob, hdz, grad_z
                with bf16():
                    z = model0(xm)
                (z.float() * weight).sum().backward()
                del z, weight
            for p, p0 in zip(named.values(), model0.parameters()):
                p.grad = p0.grad
                p0.grad = None
        else:
            for i in range(0, local_sequences, args.micro):
                with bf16():
                    loss = model(x_all[i:i + args.micro], y_all[i:i + args.micro])
                (loss * (x_all[i:i + args.micro].numel() * world / config["batch_tokens"])).backward()
                train_total += loss.detach().double() * x_all[i:i + args.micro].numel()
        flat = torch.cat([p.grad.reshape(-1) for p in named.values()])
        dist.all_reduce(flat)
        flat /= world
        grad_norm = float(flat.float().norm())
        if args.clip is not None and grad_norm > args.clip:     # the trainer's global clipping, before any state
            flat *= args.clip / grad_norm
        dist.all_reduce(train_total)
        for p, part in zip(named.values(), flat.split([p.numel() for p in named.values()])):
            p.grad = part.view_as(p).clone()
        g = torch.cat([p.grad.float().reshape(-1) for p in body])
        if average is not None:     # Newton on the averaged gradient
            average.mul_(args.momentum).add_(g, alpha=1 - args.momentum)
            g = average.clone()
        sizes = [w.numel() for w in body]
        curv = x_all[:args.curvature_sequences]
        G = GNProduct(model, body_keys, curv, curv.numel() * world, args.micro)

        def op(v):
            return torch.cat([o.reshape(-1) for o in G([part.view_as(w) for part, w in zip(v.split(sizes), body)])])
        extra_pre = {"grad_norm_before_clip": grad_norm}
        if args.transport:          # the momentum carried to the current weights (first order in the GN)
            if displacement is None:
                displacement = torch.zeros_like(g)
            elif float(displacement.norm()) > 0:
                gd = op(displacement)
                extra_pre["transport_ratio"] = float((1 - args.momentum) * gd.norm() / g.norm().clamp_min(1e-30))
                g = g + (1 - args.momentum) * gd
                del gd
        if args.sharpness_every and s % args.sharpness_every == 0:
            extra_pre["gn_top"] = top_eigenvalue(op, sum(sizes), 20, device)
        kron = None
        if args.kron_control or args.log_kron:
            t_kron = time.time()
            roots = kron_roots(model, body_keys, curv, y_all[:args.curvature_sequences], args.micro, args.gnpd,
                               args.normalize, torch.Generator(device=device).manual_seed(1000 * s + rank))
            kron = kron_direction(roots, body_keys, [gi.view_as(w) for gi, w in zip(g.split(sizes), body)])
            del roots
            extra_pre["kron_seconds"] = time.time() - t_kron
        # Lanczos from g (full reorthogonalization): one Krylov basis gives the damped solution for every damping
        solutions, rho, ritz_top = {args.normalize: -g}, float("nan"), float("nan")     # the polar control's placeholder
        stage_log = {}
        if args.staged_pd is not None:      # layer-staged PD (or its control with --stage-off)
            t_stage = time.time()
            projectors = None
            if fixed_cache is not None:     # --fixed-root: the start weights' root and band, reused
                roots, projectors = fixed_cache
            elif args.stage_band != "all":
                roots, projectors = pd_roots(model, body_keys, fixed_x if args.fixed_root else curv, args.micro,
                                             args.staged_pd, 1e-3, band=args.stage_band, ema=c_ema, ema_beta=args.stage_c_ema)
            else:
                roots = pd_roots(model, body_keys, fixed_x if args.fixed_root else curv, args.micro, args.staged_pd, 1e-3,
                                 ema=c_ema, ema_beta=args.stage_c_ema)
            if args.fixed_root and fixed_cache is None:
                fixed_cache = (roots, projectors)
            kappa = min(1.0, args.clip / grad_norm) if args.clip is not None else 1.0
            residual = g.clone()
            offsets = [0]
            for n_el in sizes:
                offsets.append(offsets[-1] + n_el)
            blocks = []
            for idx, key in enumerate(body_keys):
                block = key.split(".")[1]
                if not blocks or blocks[-1][0] != block:
                    blocks.append((block, []))
                blocks[-1][1].append(idx)
            if args.stage_granularity == "matrix":      # every matrix its own stage, in the model's order (22:28 CDT)
                blocks = [(body_keys[idx], [idx]) for _, members in blocks for idx in members]
            if args.stage_granularity == "sublayer":    # q/k/v, then o, then up, then down within each layer
                sub = []
                for name, members in blocks:
                    groups = {}
                    for idx in members:
                        kind = body_keys[idx].split(".")[-2] if body_keys[idx].endswith("weight") else body_keys[idx].split(".")[-1]
                        stage = {"q": 0, "k": 0, "v": 0, "o": 1, "up": 2, "down": 3}.get(kind)
                        if stage is None:
                            raise ValueError(f"unknown matrix kind in {body_keys[idx]}")
                        groups.setdefault(stage, []).append(idx)
                    sub.extend((f"{name}.{stage}", groups[stage]) for stage in sorted(groups))
                blocks = sub
            if args.stage_reverse:      # sweep from the output side: the earlier layers yield to the later ones
                blocks = blocks[::-1]
            direction = torch.zeros_like(g)
            plain_cos, dual_scales = [], []
            matrix_scales = [1.0] * len(body)     # --stage-dual: carried into the applied update (fix 13:18 CDT)
            kept_energy = []
            final_inputs, stage_steps = {}, []
            for b_i, (_, members) in enumerate(blocks):
                block_step = torch.zeros_like(g)
                plain_block_step = torch.zeros_like(g) if args.stage_jacobi else None
                for idx in members:
                    w = body[idx]
                    rows, cols = w.shape
                    r_part = residual[offsets[idx]:offsets[idx + 1]].view_as(w)
                    R = roots[body_keys[idx]]
                    X = r_part.float() @ R
                    if soap_states is not None:     # S∘PD: SOAP normalization in the whitened coordinates, then NS
                        state = soap_states.get(body_keys[idx])
                        if state is None:
                            state = soap_states[body_keys[idx]] = new_soap_state(w, "both")
                        if args.stage_sweeps > 1 and not args.stage_off:    # statistics move once, after the last sweep
                            Z = newton_schulz(soap_peek(X, state, 0.9, 0.5)[None], schedule_ns)[0].float()
                            final_inputs[idx] = X
                        else:
                            Z = newton_schulz(soap_precondition(X, state, 0.9, 0.5)[None], schedule_ns)[0].float()
                            soap_update_statistics(w.grad.float() @ R, state, 0.9)
                    else:
                        Z = newton_schulz(X[None], schedule_ns)[0].float()
                    d = Z @ R
                    target = math.sqrt(min(rows, cols)) * math.sqrt(max(1.0, rows / cols))
                    dual = 1.0
                    if not args.stage_off and b_i > 0:      # how far staging turned this matrix's direction
                        g_part = g[offsets[idx]:offsets[idx + 1]].view_as(w)
                        X0 = g_part.float() @ R
                        Z0 = newton_schulz(X0[None], schedule_ns)[0].float()
                        d0 = Z0 @ R
                        plain_cos.append(float((d * d0).sum() / (d.norm() * d0.norm()).clamp_min(1e-30)))
                        if args.stage_dual:     # steepest descent's step scales with the gradient's dual norm
                            dual = min(1.0, float((Z * X).sum() / (Z0 * X0).sum().clamp_min(1e-30)))
                            dual_scales.append(dual)
                            matrix_scales[idx] = dual
                    step_part = -(args.lr_scale * lr_ref * target * dual / float(d.norm().clamp_min(1e-30))) * d
                    direction[offsets[idx]:offsets[idx + 1]] = -d.reshape(-1)
                    block_step[offsets[idx]:offsets[idx + 1]] = step_part.reshape(-1)
                    if plain_block_step is not None:    # Jacobi: this matrix's plain step, from its momentum alone
                        if b_i == 0:
                            plain_block_step[offsets[idx]:offsets[idx + 1]] = step_part.reshape(-1)
                        else:
                            Xp = g[offsets[idx]:offsets[idx + 1]].view_as(w).float() @ R
                            if soap_states is not None:
                                Xp = soap_peek(Xp, soap_states[body_keys[idx]], 0.9, 0.5)
                            dp = newton_schulz(Xp[None], schedule_ns)[0].float() @ R
                            plain_block_step[offsets[idx]:offsets[idx + 1]] = (
                                -(args.lr_scale * lr_ref * target / float(dp.norm().clamp_min(1e-30))) * dp).reshape(-1)
                if args.stage_sweeps > 1:      # kept only when further sweeps need them (a body-sized vector per stage)
                    stage_steps.append(block_step)
                if not args.stage_off and b_i + 1 < len(blocks):
                    correction = op(plain_block_step if plain_block_step is not None else block_step)
                    if projectors is not None:      # keep each later matrix's correction on its chosen input band
                        for later in [i for _, ms in blocks[b_i + 1:] for i in ms]:
                            w = body[later]
                            part = correction[offsets[later]:offsets[later + 1]].view_as(w)
                            banded = part.float() @ projectors[body_keys[later]]
                            if args.stage_reversal_only:    # only the component along the matrix's own band momentum
                                axis = g[offsets[later]:offsets[later + 1]].view_as(w).float() @ projectors[body_keys[later]]
                                along = (banded * axis).sum() / (axis * axis).sum().clamp_min(1e-30) * axis
                                kept_energy.append(float(along.square().sum() / banded.square().sum().clamp_min(1e-30)))
                                banded = along
                            correction[offsets[later]:offsets[later + 1]] = banded.reshape(-1)
                    residual += (kappa * args.stage_scale) * correction
                    del correction
            sweep_turn = []
            if args.stage_sweeps > 1 and not args.stage_off:     # symmetric Gauss-Seidel: sweep back, and forth
                total = torch.stack(stage_steps).sum(0)
                order = list(range(len(blocks)))
                for sweep in range(1, args.stage_sweeps):
                    for b_i in (order[::-1] if sweep % 2 == 1 else order):
                        members = blocks[b_i][1]
                        other = total - stage_steps[b_i]
                        correction = op(other)      # every other stage's current step, carried by the GN coupling
                        new_block = torch.zeros_like(g)
                        for idx in members:
                            w = body[idx]
                            rows, cols = w.shape
                            part = correction[offsets[idx]:offsets[idx + 1]].view_as(w).float()
                            if projectors is not None:
                                part = part @ projectors[body_keys[idx]]
                            r_part = g[offsets[idx]:offsets[idx + 1]].view_as(w).float() + (kappa * args.stage_scale) * part
                            R = roots[body_keys[idx]]
                            X = r_part @ R
                            if soap_states is not None:
                                Z = newton_schulz(soap_peek(X, soap_states[body_keys[idx]], 0.9, 0.5)[None], schedule_ns)[0].float()
                                final_inputs[idx] = X
                            else:
                                Z = newton_schulz(X[None], schedule_ns)[0].float()
                            d = Z @ R
                            old_d = direction[offsets[idx]:offsets[idx + 1]].view_as(w)
                            sweep_turn.append(float((-d * old_d).sum() / (d.norm() * old_d.norm()).clamp_min(1e-30)))
                            target = math.sqrt(min(rows, cols)) * math.sqrt(max(1.0, rows / cols))
                            direction[offsets[idx]:offsets[idx + 1]] = -d.reshape(-1)
                            new_block[offsets[idx]:offsets[idx + 1]] = (
                                -(args.lr_scale * lr_ref * target / float(d.norm().clamp_min(1e-30))) * d).reshape(-1)
                        total += new_block - stage_steps[b_i]
                        stage_steps[b_i] = new_block
                        del correction
                if soap_states is not None:     # one statistics update per step, on the final input
                    for idx, X in final_inputs.items():
                        state = soap_states[body_keys[idx]]
                        soap_precondition(X, state, 0.9, 0.5)
                        soap_update_statistics(body[idx].grad.float() @ roots[body_keys[idx]], state, 0.9)
            elif args.stage_sweeps > 1 and soap_states is not None:
                raise RuntimeError("unreachable: --stage-sweeps with --stage-off")
            solutions = {args.normalize: direction}
            stage_log = {"stage_seconds": time.time() - t_stage, "kappa": kappa,
                         **({"sweep_turn_mean": sum(sweep_turn) / len(sweep_turn)} if sweep_turn else {}),
                         "staged_cos_plain_mean": sum(plain_cos) / len(plain_cos) if plain_cos else 1.0,
                         "staged_cos_plain_min": min(plain_cos) if plain_cos else 1.0,
                         **({"stage_dual_mean": sum(dual_scales) / len(dual_scales),
                             "stage_dual_min": min(dual_scales)} if dual_scales else {}),
                         **({"reversal_kept_energy": sum(kept_energy) / len(kept_energy)} if kept_energy else {})}
            del roots, residual, projectors
        elif args.kron_control:       # GN-PD's Kronecker limit (TS p/p), no Lanczos
            solutions = {args.normalize: kron}
        elif args.polar_direction:    # Muon's step at the peak LR; the line search scales it by LINE x schedule
            parts = []
            for w, gi in zip(body, g.split(sizes)):
                rows, cols = w.shape
                polar = newton_schulz(gi.view_as(w)[None], schedule_ns)[0].float().reshape(-1)
                parts.append(-peak * math.sqrt(max(1.0, rows / cols)) * polar)
            solutions = {0.0: torch.cat(parts)}
        elif not args.polar_control:
            def lanczos_run(start, k):
                V = torch.zeros(k, start.numel(), device=device)
                V[0] = start / start.norm()
                alphas_l, betas_l = [], []
                for j in range(k):
                    w = op(V[j])
                    alphas_l.append(float(w @ V[j]))
                    for _ in range(2):
                        w -= V[:j + 1].T @ (V[:j + 1] @ w)
                    if j + 1 == k:
                        break
                    beta = float(w.norm())
                    betas_l.append(beta)
                    V[j + 1] = w / beta
                Tk = torch.diag(torch.tensor(alphas_l, dtype=torch.float64))
                off = torch.tensor(betas_l, dtype=torch.float64)
                return V, Tk + torch.diag(off, 1) + torch.diag(off, -1)

            def matrix_power(V, Tk, start_norm, power, mu):     # (G + mu)^-power applied to the Lanczos start vector
                values, vectors = torch.linalg.eigh(Tk)
                y = vectors @ ((values.clamp_min(0) + mu).pow(-power) * vectors[0] * start_norm)
                return V.T @ y.to(device, torch.float32)
            k = args.cg
            V, Tk = lanczos_run(g, k)
            rho = float(Tk[0, 0])
            ritz_top = float(torch.linalg.eigvalsh(Tk)[-1])
            rhs = torch.zeros(k, dtype=torch.float64)
            rhs[0] = float(g.norm())
            solutions = {}
            if args.gnpd is not None:   # PD in the exact GN geometry: whiten, polar per matrix, whiten again
                mu = args.normalize * rho
                white = matrix_power(V, Tk, float(g.norm()), args.gnpd, mu)
                del V
                parts = []
                for w, wi in zip(body, white.split(sizes)):
                    u, _, vh = torch.linalg.svd(wi.view_as(w).float(), full_matrices=False)
                    parts.append((u @ vh).reshape(-1))
                polar_flat = torch.cat(parts)
                V2, T2 = lanczos_run(polar_flat, k)
                solutions[args.normalize] = -matrix_power(V2, T2, float(polar_flat.norm()), args.gnpd, mu)
                del V2
            else:
                for d in (DAMPINGS if args.normalize is None else (args.normalize,)):
                    y = torch.linalg.solve(Tk + d * rho * torch.eye(k, dtype=torch.float64), rhs)
                    solutions[d] = -(V.T @ y.to(device, torch.float32))
                del V
        # aux parameters: AdamW on the batch gradient, the run's aux schedule
        if not args.freeze_aux:
            for group in aux_opt.param_groups:
                group["lr"] = config["aux_learning_rate"] * schedule
            aux_opt.step()
        if args.normalize is not None:
            # GN's shape at the baseline's per-matrix step norms, so only the direction differs from the baseline
            with torch.no_grad():
                base_line = summed_loss(model, line_x, line_y, args.micro)
                before = torch.cat([w.detach().reshape(-1) for w in body]) if args.transport else None
                x = solutions[args.normalize]
                predicted, squares, cosines, kron_cos = 0.0, 0.0, [], []
                kron_parts = kron.split(sizes) if kron is not None else [None] * len(body)
                step_scales = matrix_scales if (args.staged_pd is not None and args.stage_dual) else [1.0] * len(body)
                for w, xi, gi, ki, sc in zip(body, x.split(sizes), g.split(sizes), kron_parts, step_scales):
                    rows, cols = w.shape
                    scale = math.sqrt(max(1.0, rows / cols))
                    polar = newton_schulz(gi.view_as(w)[None], schedule_ns)[0].float().reshape(-1)
                    if args.polar_control:
                        xi = -polar
                    cosines.append(-float(xi @ polar) / max(float(xi.norm() * polar.norm()), 1e-30))     # vs Muon's step
                    if ki is not None:      # vs the Kronecker limit of the same map
                        kron_cos.append(float(xi @ ki) / max(float(xi.norm() * ki.norm()), 1e-30))
                    if config.get("data_norm_alpha", 0) > 0:    # data-norm maps are rescaled to sqrt(min(m, n))
                        target = math.sqrt(min(rows, cols)) * scale
                    else:                                       # Muon: the norm of its own Newton-Schulz output
                        target = float(polar.norm()) * scale
                    factor = sc * args.lr_scale * lr_ref * target / float(xi.norm().clamp_min(1e-30))
                    w.mul_(1 - lr_ref * config["weight_decay"])
                    w.add_(xi.view_as(w), alpha=factor)
                    predicted += factor * float(gi @ xi)
                    squares += (sc * args.lr_scale * lr_ref * target) ** 2
                if before is not None:      # D <- b D + b/(1 - b) dW (weight decay included in dW)
                    b = args.momentum
                    displacement.mul_(b).add_(torch.cat([w.detach().reshape(-1) for w in body]) - before, alpha=b / (1 - b))
                    del before
                best_loss = summed_loss(model, line_x, line_y, args.micro)
            d_best, alpha, step_norm = args.normalize, lr_ref, math.sqrt(squares)
            extra = {"cos_muon_mean": sum(cosines) / len(cosines), "cos_muon_min": min(cosines), **stage_log}
            if kron_cos:
                extra.update({"cos_kron_mean": sum(kron_cos) / len(kron_cos), "cos_kron_min": min(kron_cos),
                              "cos_kron": dict(zip(body_keys, kron_cos))})
        else:
            # joint line search over (damping, step) for the body on held-out training sequences
            with torch.no_grad():
                originals = [w.detach().clone() for w in body]
                base_line = summed_loss(model, line_x, line_y, args.micro)
                trials = {}
                for d, xs in solutions.items():
                    parts = xs.split(sizes)
                    for a in LINE:
                        a_s = a * schedule
                        for w, w0, xi in zip(body, originals, parts):
                            w.copy_(w0 + a_s * xi.view_as(w0))
                        trials[(d, a_s)] = summed_loss(model, line_x, line_y, args.micro)
                (d_best, alpha), best_loss = min(trials.items(), key=lambda kv: kv[1])
                if best_loss >= base_line:      # no trial improves the held-out loss: skip the body step
                    alpha, best_loss = 0.0, base_line
                alpha *= args.overshoot
                parts = solutions[d_best].split(sizes)
                for w, w0, xi in zip(body, originals, parts):
                    w.copy_(w0 + alpha * xi.view_as(w0))
                    w.mul_(1 - lr_ref * config["weight_decay"])
            step_norm = alpha * float(solutions[d_best].norm())
            predicted = alpha * float(g @ solutions[d_best])    # first-order part of the predicted change
            extra = {}
        actual = (best_loss - base_line) / line_count
        row = {"step": s, "lr_ref": lr_ref, "alpha": alpha, "damping": d_best, "rho": rho, "ritz_top": ritz_top,
               "first_order_change": predicted, "held_out_change": actual,
               "line_base": base_line / line_count, "line_best": best_loss / line_count,
               "train_nll": float(train_total) / config["batch_tokens"],
               "step_norm": step_norm, **extra_pre, **extra, "training_seconds": time.time() - t0}
        if s % args.validation_every == 0 or s == stop or s == args.anneal_from:
            row["validation_nll"] = summed_loss(model, val_x, val_y, args.micro) / config["validation_tokens"]
        if rank == 0 and (s % args.checkpoint_every == 0 or s == stop or s in keep):
            tmp = args.out / "checkpoint.pt.tmp"
            torch.save({"model": model.state_dict(), "aux": aux_opt.state_dict(), "step": s, "config": config,
                        "average": average.cpu() if average is not None else None,
                        "displacement": displacement.cpu() if displacement is not None else None}, tmp)
            if s in keep:
                (args.out / "kept").mkdir(exist_ok=True)
                torch.save(torch.load(tmp, map_location="cpu", weights_only=False), args.out / "kept" / f"step{s:06d}.pt")
            tmp.replace(resume)
        if rank == 0:
            (args.out / "steps" / f"step{s:06d}.json").write_text(json.dumps(row) + "\n")
            print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in row.items() if k != "cos_kron"}), flush=True)
    if rank == 0:
        (args.out / "done.json").write_text(json.dumps({"seconds": time.time() - started, "last_step": stop}) + "\n")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()

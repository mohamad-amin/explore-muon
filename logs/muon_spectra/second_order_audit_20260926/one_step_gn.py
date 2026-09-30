"""What would an exact Gauss-Newton step do from this state, and how close do the optimizers' directions get?

At a kept checkpoint and for an input b (a fresh gradient over 2048 or 8192 held-out sequences, i.e. a 1M or 4M
token batch, or the optimizer's next momentum M' = 0.95 M + g_1M), directions over all hidden matrices
(embeddings, head and gains held fixed):
  gd              -b
  muon            -polar(b), with Muon's Frobenius norm sqrt(min(m, n)) and shape factor sqrt(max(1, m/n))
  pd_a            -polar(b R) R, R = (C / mean eig + 1e-3 I)^-a, rescaled like PD (a = 1/4, 1/2)
  muon_p, pd_a0.5_p  U S^p V^T in place of polar's U V^T (p = 1/4, 1/2): is full orthogonalization optimal?
  pd_a            also a = 3/4 and 1
  two_sided       -L polar(L b R) R, L = (B / mean eig + 1e-3 I)^-1/4, a = 1/4 (B: sampled-label output factor)
  kfac_d          -(B + d mean(B))^-1 b (C + d mean(C))^-1            per-token K-FAC, relative damping d
  ekfac_d         -U [(U^T b V) / (h + d mean h)] V^T                 h: exact per-pair GN diagonal in the frame
  gn_k_d          -(G + d rho I)^-1 b in the k-step Krylov space of the exact GN matrix G (Lanczos, = k steps of
                  CG), rho = b^T G b / b^T b the curvature along b; G from the curvature sequences
  gnpow_p_d       -(G + d rho I)^-p b for p = 1/4, 1/2, 3/4 in the plain Krylov space: which power of the exact
                  curvature is best at this gradient noise?
  gni_k_d         CG on (G + d rho' I (x) C) d = -b, input-whitened Lanczos (--krylov-input): converges on the sharp
                  PD-family states where plain CG stalls
  gnp_k_d         the same with EKFAC-preconditioned Lanczos, -P^-1/2 (P^-1/2 G P^-1/2 + d rho' I)^-1 P^-1/2 b: CG on
                  (G + d rho' P) d = -b, which reaches the flat directions in far fewer steps (P: exact per-pair GN
                  diagonal in each matrix's Kronecker frame, relative damping --pre-damping)
  (--mix c,...)   extra inputs g + c M: the run's own fresh gradient plus a fraction c of the saved momentum
                  (c = 0.95 is the next momentum M'), to see how much stale averaging each direction family tolerates
  base-topk       Muon's or PD alpha 1/2's direction with the top-k Ritz vectors of the exact GN (from the plain
                  Lanczos run: global, across all matrices) projected out; base-topk+newton adds the Newton step
                  inside that k-dimensional subspace at the best two-scale combination. Tests whether a few global
                  stiff directions carry the coherence that separates the optimizers from GN.
  base+cgk        (--warm-cg) k CG steps on (G + d rho I) x = -b warm-started from Muon's or PD alpha 1/2's direction at
                  its GN-optimal scale: how many GN-vector products close the gap from an optimizer's direction
  X@muon_norms    direction X with each matrix rescaled to Muon's per-matrix norm (X's shape, Muon's allocation)
  muon@X_norms    Muon's per-matrix direction with X's per-matrix norms (Muon's shape, X's allocation)
Each direction D is scored on held-out sequences at its own best scale: the exact one-step GN decrease
<g,D>^2 / (2 q(D)), and the true loss at 0.5, 1 and 2 times that scale. C, B, the frame and G come from the same
curvature sequences. Per-matrix structure of the main directions is saved for plots; --gram adds the exact GN
Gram between the per-matrix pieces of Muon's and the best GN direction (two halves of the held-out set).
One-step and local: a map of where the gap to Gauss-Newton lives, not a training result.

usage: one_step_gn.py OUT_DIR ARM_DIR:STEP [...] [--inputs g1M,g4M,momentum] [--krylov 96] [--krylov-pre 96] [--gram]
       [--momentum-gradient g4M]   (for 4M-batch runs: M' = 0.95 M + g_4M)
"""
import argparse
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch.func import functional_call, jvp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from research.adamw_spectra import gn_probe as P  # noqa: E402
from research.adamw_spectra.data import TokenStream  # noqa: E402

BASE = 2 * 1048576
GRAD = BASE + 80000 * 512       # fresh gradients (up to 8192 sequences)
CURV = BASE + 90000 * 512       # GN products and the K-FAC / frame statistics
EVAL = BASE + 92000 * 512       # held-out scoring
KRYLOV_TRUNCATIONS = (16, 48, 96)
KRYLOV_DAMPINGS = (3e-1, 1e-1, 3e-2, 1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5)
SPECTRAL_POWERS = (0.25, 0.5)            # D = U S^p V^T instead of polar's U V^T (p = 0)
INPUT_POWERS = (0.25, 0.5, 0.75, 1.0)    # PD's alpha
KFAC_DAMPINGS = (1e-3, 1e-2, 1e-1)
POWERS = (0.25, 0.5, 0.75)
KIND_ALPHAS = (-0.25, 0.0, 0.25, 0.5, 0.75)
DEFLATE_KS = (1, 4, 16, 32)
WARM_CG_STEPS = (1, 2, 4, 8, 16)             # top global GN Ritz vectors removed from Muon's and PD's directions    # --kind-alpha: one kind at a time, others at PD alpha 1/2


class Flat:
    """Hidden-matrix dicts <-> one flat FP32 vector."""

    def __init__(self, tensors):
        self.names = list(tensors)
        self.shapes = [tuple(t.shape) for t in tensors.values()]
        self.sizes = [t.numel() for t in tensors.values()]

    def flat(self, d):
        return torch.cat([d[n].reshape(-1).float() for n in self.names])

    def dict(self, v):
        return OrderedDict((n, part.view(s)) for n, part, s in zip(self.names, v.split(self.sizes), self.shapes))


class Preconditioner:
    """half_inverse applies P^-1/2 to a flat vector. "ekfac": P X = U [(U^T X V) * (h + damping mean h)] V^T, h the exact
    per-pair GN diagonal in each matrix's Kronecker frame. "input": P X = X (C / mean eig + 1e-3 I), the input side
    only (PD's whitening; P^-1/2 is PD's R at alpha = 1/2)."""

    def __init__(self, stats, names, flat, damping, mode="ekfac"):
        self.flat, self.mode = flat, mode
        self.items = []
        for n in names:
            if mode == "input":
                self.items.append(stats[n]["R0.5"])
            else:
                h = stats[n]["h"].clamp_min(0)
                self.items.append((stats[n]["U"], stats[n]["V"], (h + damping * h.mean()).rsqrt()))

    def half_inverse(self, v):
        if self.mode == "input":
            parts = [x @ r for r, x in zip(self.items, self.flat.dict(v).values())]
        else:
            parts = [U @ ((U.T @ x @ V) * w) @ V.T for (U, V, w), x in zip(self.items, self.flat.dict(v).values())]
        return torch.cat([part.reshape(-1) for part in parts])


class GaussNewton:
    """v -> G v, the exact GN matrix of the mean token loss over the given microbatches, restricted to the hidden
    matrices: J^T H J with H the softmax Hessian at every position (the unit of q in gn_probe)."""

    def __init__(self, model, names, batches, flat):
        self.model, self.names, self.batches, self.flat = model, names, batches, flat
        self.keys = P.parameter_keys(model, names)
        self.base = {**dict(model.named_parameters()), **dict(model.named_buffers())}
        self.weights = [self.base[k] for k in self.keys]
        self.count = sum(x.numel() for x, _ in batches)
        self.calls = 0

    def __call__(self, v):
        d = self.flat.dict(v)
        tangents = tuple(d[n].to(w.dtype) for n, w in zip(self.names, self.weights))
        out = torch.zeros_like(v)
        for x, _ in self.batches:
            def logits_of(*weights):
                return functional_call(self.model, {**self.base, **dict(zip(self.keys, weights))}, (x,))
            with torch.no_grad(), P.explicit_attention():
                logits, dz = jvp(logits_of, tuple(self.weights), tangents)
                p = torch.softmax(logits.float(), dim=-1)
                dz = dz.float()
                hdz = p * (dz - (p * dz).sum(-1, keepdim=True))       # softmax Hessian times dz
                del logits, dz, p
            with torch.enable_grad():
                logits = self.model(x)
                grads = torch.autograd.grad((logits.float() * hdz).sum() / self.count, self.weights)
            out += torch.cat([g.reshape(-1) for g in grads])
            del hdz, logits, grads
        self.calls += 1
        return out


def lanczos(op, b, steps):
    """Lanczos from b with full reorthogonalization: orthonormal Krylov basis (steps x n, FP32) and the
    tridiagonal T = V G V^T (FP64)."""
    V = torch.zeros(steps, b.numel(), device=b.device)
    V[0] = b / b.norm()
    alphas, betas = [], []
    for j in range(steps):
        w = op(V[j])
        alphas.append(float(w @ V[j]))
        for _ in range(2):
            w -= V[:j + 1].T @ (V[:j + 1] @ w)
        if j + 1 == steps:
            break
        beta = w.norm()
        betas.append(float(beta))
        V[j + 1] = w / beta
    T = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    off = torch.tensor(betas, dtype=torch.float64)
    return V, T + torch.diag(off, 1) + torch.diag(off, -1)


def krylov_direction(V, T, b_norm, k, damping):
    """-(T_k + damping I)^-1 |b| e_1 mapped back: k steps of CG on (G + damping I) d = -b."""
    rhs = torch.zeros(k, dtype=torch.float64)
    rhs[0] = b_norm
    y = torch.linalg.solve(T[:k, :k] + damping * torch.eye(k, dtype=torch.float64), rhs)
    return -(V[:k].T @ y.to(V.device, torch.float32))


def warm_cg(op, b, x0, damping, record):
    """CG on (G + damping I) x = -b from x0; returns {k: x_k} for k in record (k GN products after the warm start)."""
    x = x0.clone()
    r = -b - (op(x) + damping * x)
    p = r.clone()
    rr = r @ r
    out = {}
    for k in range(1, max(record) + 1):
        Ap = op(p) + damping * p
        step = rr / (p @ Ap)
        x += step * p
        r -= step * Ap
        rr_new = r @ r
        p = r + (rr_new / rr) * p
        rr = rr_new
        if k in record:
            out[k] = x.clone()
    return out


def krylov_power(V, T, b_norm, k, power, damping):
    """-(T_k + damping I)^-power |b| e_1 mapped back: the Krylov approximation of -(G + damping I)^-power b."""
    values, vectors = torch.linalg.eigh(T[:k, :k])
    y = vectors @ ((values.clamp_min(0) + damping).pow(-power) * vectors[0] * b_norm)
    return -(V[:k].T @ y.to(V.device, torch.float32))


def polar(a):
    u, _, vh = torch.linalg.svd(a.float(), full_matrices=False)
    return u @ vh


def inverse_roots(cov, powers, damping=1e-3):
    """(cov / mean eig + damping I)^-p for each power p (one eigendecomposition, FP64)."""
    values, vectors = torch.linalg.eigh(0.5 * (cov + cov.T).double())
    unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
    return {p: ((vectors * (unit + damping).pow(-p)) @ vectors.T).float() for p in powers}


def muon_norm(shape):
    rows, cols = shape
    return math.sqrt(min(rows, cols)) * math.sqrt(max(1.0, rows / cols))


def statistics(model, names, batches, T, device):
    """Per-token C (all positions, and positions >= 1 as PD's statistics), sampled-label B, the Kronecker frame and
    the exact per-pair GN diagonal, from the curvature sequences (two passes, independent label draws)."""
    gen = torch.Generator(device=device).manual_seed(1)
    recorder = P.Recorder(model)
    sums = {n: {"C": 0, "C1": 0, "B": 0} for n in names}
    count = count1 = 0
    for x, y in batches:
        P.gradient_passes(model, recorder, x, y, gen, draws=1)
        for n in names:
            xi = recorder.inputs[n].double()
            es = (T * recorder.errors[n][1]).reshape(-1, recorder.errors[n][1].shape[-1]).double()
            flat = xi.reshape(-1, xi.shape[-1])
            later = xi[:, 1:].reshape(-1, xi.shape[-1])
            sums[n]["C"] = sums[n]["C"] + flat.T @ flat
            sums[n]["C1"] = sums[n]["C1"] + later.T @ later
            sums[n]["B"] = sums[n]["B"] + es.T @ es
        count += x.numel()
        count1 += x.shape[0] * (x.shape[1] - 1)
    stats = {}
    for n in names:
        C, C1, B = sums[n]["C"] / count, sums[n]["C1"] / count1, sums[n]["B"] / count
        lam_c, v = P.eigenbasis(C * count, count)
        lam_b, u = P.eigenbasis(B * count, count)
        right, left = inverse_roots(C1, INPUT_POWERS), inverse_roots(B, (0.25,))
        stats[n] = {"lam_C": lam_c.float(), "lam_B": lam_b.float(), "U": u.float(), "V": v.float(),
                    **{f"R{a}": right[a] for a in INPUT_POWERS}, "L0.25": left[0.25], "C1_full": C1.float()}
    frames = {n: P.Frame(stats[n]["U"], stats[n]["V"], stats[n]["lam_B"].double(), stats[n]["lam_C"].double(), T)
              for n in names}
    for x, y in batches:
        P.gradient_passes(model, recorder, x, y, gen, draws=1)
        for n in names:
            frames[n].add(recorder.inputs[n], recorder.errors[n][0], [recorder.errors[n][1]])
    recorder.remove()
    for n in names:
        stats[n]["h"] = frames[n].summary()["exact"].float()
    return stats


def fresh_gradient(model, names, stream, offset, sequences, T, device, micro=8):
    recorder = P.Recorder(model)
    total = None
    for first in range(0, sequences, micro):
        x, y = stream.batch(offset + first * T, micro, T, device)
        grads, _ = P.gradient_passes(model, recorder, x, y, draws=0)
        total = {n: g.clone() for n, g in grads.items()} if total is None else {n: total[n] + grads[n] for n in grads}
    recorder.remove()
    return OrderedDict((n, total[n] / sequences) for n in names)


def closed_form_directions(b, stats):
    """The optimizer-family and Kronecker directions for input b (dict over hidden matrices)."""
    out = OrderedDict()
    for label in ("gd", "muon", *(f"muon_p{p:g}" for p in SPECTRAL_POWERS), *(f"pd_a{a:g}" for a in INPUT_POWERS),
                  *(f"pd_a0.5_p{p:g}" for p in SPECTRAL_POWERS), "two_sided",
                  *(f"kfac_{d:g}" for d in KFAC_DAMPINGS), *(f"ekfac_{d:g}" for d in KFAC_DAMPINGS)):
        out[label] = OrderedDict()
    for n, g in b.items():
        s = stats[n]
        g = g.float()
        target = muon_norm(g.shape)
        out["gd"][n] = -g
        u, sv, vh = torch.linalg.svd(g, full_matrices=False)
        out["muon"][n] = -(u @ vh) * math.sqrt(max(1.0, g.shape[0] / g.shape[1]))
        for power in SPECTRAL_POWERS:
            d = (u * sv.pow(power)) @ vh
            out[f"muon_p{power:g}"][n] = -d * (target / d.norm().clamp_min(1e-30))
        for alpha in INPUT_POWERS:
            r = s[f"R{alpha}"]
            u, sv, vh = torch.linalg.svd(g @ r, full_matrices=False)
            d = (u @ vh) @ r
            out[f"pd_a{alpha:g}"][n] = -d * (target / d.norm().clamp_min(1e-30))
            if alpha == 0.5:
                for power in SPECTRAL_POWERS:
                    d = ((u * sv.pow(power)) @ vh) @ r
                    out[f"pd_a0.5_p{power:g}"][n] = -d * (target / d.norm().clamp_min(1e-30))
        r, left = s["R0.25"], s["L0.25"]
        d = left @ polar(left @ g @ r) @ r
        out["two_sided"][n] = -d * (target / d.norm().clamp_min(1e-30))
        U, V, lam_b, lam_c, h = s["U"], s["V"], s["lam_B"].clamp_min(0), s["lam_C"].clamp_min(0), s["h"]
        coeff = U.T @ g @ V
        for damping in KFAC_DAMPINGS:
            kfac = torch.outer(lam_b + damping * lam_b.mean(), lam_c + damping * lam_c.mean())
            out[f"kfac_{damping:g}"][n] = -(U @ (coeff / kfac) @ V.T)
            out[f"ekfac_{damping:g}"][n] = -(U @ (coeff / (h.clamp_min(0) + damping * h.mean())) @ V.T)
    return out


def rescaled(direction, norms):
    return OrderedDict((n, d * (norms[n] / d.norm().clamp_min(1e-30))) for n, d in direction.items())


def score(model, batches, direction):
    first = q = 0.0
    count = 0
    for x, y in batches:
        terms = P.directional_terms(model, x, y, direction)
        first += float(terms["first"].sum())
        q += float(terms["q"].sum())
        count += x.shape[0]
    first, q = first / count, q / count
    return {"first": first, "q": q, "best_decrease": first * first / (2 * q) if first < 0 and q > 0 else 0.0,
            "best_scale": -first / q if q > 0 else float("nan")}


@torch.no_grad()
def loss_along(model, batches, direction, scales):
    """True mean held-out loss at W + c D for each c (hidden matrices moved, everything else fixed)."""
    layers = P.hidden_linears(model)
    originals = {n: layers[n].weight.detach().clone() for n in direction}
    out = []
    for c in scales:
        for n, d in direction.items():
            layers[n].weight.copy_(originals[n] + c * d.to(originals[n].dtype))
        total, count = 0.0, 0
        for x, y in batches:
            total += float(P.token_losses(model(x), y).mean(1).sum())
            count += x.shape[0]
        out.append(total / count)
    for n in direction:
        layers[n].weight.copy_(originals[n])
    return out


@torch.no_grad()
def gram(model, batches, pieces):
    """a_i = <g, D_i> and the exact GN Gram Q_ij = <D_i, G D_j> for directions D_i (dicts over hidden matrices),
    one forward-mode pass per direction and microbatch (units of gn_probe's q)."""
    base = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    n = len(pieces)
    a = torch.zeros(n, dtype=torch.float64)
    Q = torch.zeros(n, n, dtype=torch.float64)
    count = 0
    for x, y in batches:
        centered = []
        for i, piece in enumerate(pieces):
            names = list(piece)
            keys = P.parameter_keys(model, names)

            def logits_of(*weights):
                return functional_call(model, {**base, **dict(zip(keys, weights))}, (x,))
            with P.explicit_attention():
                logits, dz = jvp(logits_of, tuple(base[k] for k in keys),
                                 tuple(piece[m].to(base[k].dtype) for m, k in zip(names, keys)))
            p = torch.softmax(logits.float(), dim=-1)
            dz = dz.float()
            mean = (p * dz).sum(-1, keepdim=True)
            first = mean.squeeze(-1) - dz.gather(-1, y.unsqueeze(-1)).squeeze(-1)
            a[i] += float(first.mean(1).sum())
            centered.append((p.sqrt() * (dz - mean)).flatten())
            del logits, dz, p, mean
        stacked = torch.stack(centered)
        Q += (stacked @ stacked.T).double().cpu() / x.shape[1]
        count += x.shape[0]
        del stacked, centered
    return (a / count).tolist(), (Q / count).tolist()


def structure(direction, stats, reference):
    """Per-matrix summaries of a direction: Frobenius norm, singular values, energy by K-FAC rank class (top 1%,
    next 9%, rest), cosine with the reference direction, and energy on the 12 x 12 grid of frame rank bins."""
    out = {}
    for n, d in direction.items():
        s = stats[n]
        d = d.float()
        coeff = s["U"].T @ d @ s["V"]
        kfac = torch.outer(s["lam_B"], s["lam_C"])
        rank = torch.argsort(torch.argsort(kfac.flatten(), descending=True)).view_as(kfac)
        total = kfac.numel()
        energy = coeff.pow(2)
        classes = [float(energy[rank < 0.01 * total].sum()), float(energy[(rank >= 0.01 * total) & (rank < 0.1 * total)].sum()),
                   float(energy[rank >= 0.1 * total].sum())]
        bins_r = torch.clamp(torch.floor(torch.log2(torch.arange(1, coeff.shape[0] + 1, device=d.device, dtype=torch.float32))), max=11).long()
        bins_c = torch.clamp(torch.floor(torch.log2(torch.arange(1, coeff.shape[1] + 1, device=d.device, dtype=torch.float32))), max=11).long()
        cell = (bins_r[:, None] * 12 + bins_c[None, :]).flatten()
        grid = torch.bincount(cell, weights=energy.flatten(), minlength=144).view(12, 12)
        ref = reference[n].float()
        out[n] = {"norm": float(d.norm()), "singular_values": torch.linalg.svdvals(d).cpu().tolist(),
                  "kfac_class_energy": classes, "cos_reference": float((d * ref).sum() / (d.norm() * ref.norm()).clamp_min(1e-30)),
                  "rank_grid_energy": grid.cpu().tolist()}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("items", nargs="+")
    parser.add_argument("--inputs", default="g1M,g4M,momentum")
    parser.add_argument("--krylov", type=int, default=96, help="plain Lanczos steps on G (0: skip)")
    parser.add_argument("--krylov-pre", type=int, default=0,
                        help="Lanczos steps on P^-1/2 G P^-1/2, P the EKFAC exact-diagonal preconditioner (0: skip)")
    parser.add_argument("--pre-damping", type=float, default=1e-2, help="relative damping of P")
    parser.add_argument("--krylov-input", type=int, default=0,
                        help="Lanczos steps on R G R with R = PD's (C / mean + 1e-3)^-1/2 per matrix (0: skip)")
    parser.add_argument("--momentum-gradient", default="g1M", choices=("g1M", "g4M"),
                        help="fresh gradient added to the saved momentum (the run's own batch)")
    parser.add_argument("--no-deflate", dest="deflate", action="store_false",
                        help="skip removing the top global GN Ritz vectors from Muon's and PD's directions")
    parser.add_argument("--warm-cg", action="store_true",
                        help="CG on the exact GN system warm-started from Muon's and PD alpha 1/2's directions")
    parser.add_argument("--warm-cg-damping", type=float, default=1e-3, help="relative to rho, the curvature along b")
    parser.add_argument("--kind-alpha", action="store_true",
                        help="scan PD's input power one matrix kind at a time (others at alpha 1/2), incl. alpha = -1/4")
    parser.add_argument("--mix", default="", help="comma-separated c: extra inputs g + c M (M the saved momentum)")
    parser.add_argument("--curvature-sequences", type=int, default=256)
    parser.add_argument("--eval-sequences", type=int, default=512)
    parser.add_argument("--gram", action="store_true")
    parser.add_argument("--gram-sequences", type=int, default=128)
    parser.add_argument("--micro", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda")
    stream = TokenStream(str(REPO / "data/fineweb10B/fineweb_val_*.bin"))
    inputs = args.inputs.split(",")
    for item in args.items:
        arm, step = item.rsplit(":", 1)
        step = int(step)
        started = time.time()
        kept = Path(arm) / "scientific" / "kept"
        kept = kept if kept.exists() else Path(arm) / "kept"
        model, saved = P.load_checkpoint(kept / f"step{step:06d}.pt", device)
        T = model.config.seq_len
        names = list(P.hidden_linears(model))
        layers = P.hidden_linears(model)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for n in names:
            layers[n].weight.requires_grad_(True)
        flat = Flat(OrderedDict((n, layers[n].weight) for n in names))
        curv = [stream.batch(CURV + first * T, args.micro, T, device)
                for first in range(0, args.curvature_sequences, args.micro)]
        evals = [stream.batch(EVAL + first * T, args.micro, T, device)
                 for first in range(0, args.eval_sequences, args.micro)]
        stats = statistics(model, names, curv, T, device)
        print(json.dumps({"item": item, "stage": "statistics", "seconds": time.time() - started}), flush=True)
        g4 = fresh_gradient(model, names, stream, GRAD, 8192, T, device) \
            if "g4M" in inputs or args.momentum_gradient == "g4M" else None
        g1 = fresh_gradient(model, names, stream, GRAD, 2048, T, device)
        sources = OrderedDict()
        if "g1M" in inputs:
            sources["g1M"] = g1
        if "g4M" in inputs:
            sources["g4M"] = g4
        if "momentum" in inputs and "optimizer" in saved:
            state = saved["optimizer"]["state"]
            fresh = g4 if args.momentum_gradient == "g4M" else g1
            sources["momentum"] = OrderedDict((n, 0.95 * state[i]["momentum_buffer"].to(device).float() + fresh[n])
                                              for i, n in enumerate(names))
        for c in (float(x) for x in args.mix.split(",") if x):
            # fresh gradient plus a fraction c of the saved momentum: how much stale averaging can each direction take?
            state = saved["optimizer"]["state"]
            fresh = g4 if args.momentum_gradient == "g4M" else g1
            sources[f"mix{c:g}"] = OrderedDict((n, c * state[i]["momentum_buffer"].to(device).float() + fresh[n])
                                               for i, n in enumerate(names))
        operator = GaussNewton(model, names, curv, flat)
        kind_roots = {}
        if args.kind_alpha:
            for n in names:
                values, vectors = torch.linalg.eigh(stats[n]["C1_full"].double())
                unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
                kind_roots[n] = {a: (None if a == 0 else ((vectors * (unit + 1e-3).pow(-a)) @ vectors.T).float()) for a in KIND_ALPHAS}
        baseline = loss_along(model, evals, OrderedDict((n, torch.zeros_like(layers[n].weight)) for n in names), [0.0])[0]
        result = {"arm": arm, "step": step, "eval_loss": baseline, "inputs": {}}
        saved_directions = {}
        deflate_newton = {}
        for source, b in sources.items():
            t0 = time.time()
            directions = closed_form_directions(b, stats)
            b_flat = flat.flat(b)
            entry = {}
            if args.krylov:
                V, Tk = lanczos(operator, b_flat, args.krylov)
                rho = float(Tk[0, 0])
                entry.update(rho=rho, ritz_values=torch.linalg.eigvalsh(Tk).flip(0).tolist())
                for k in (k for k in KRYLOV_TRUNCATIONS if k <= args.krylov):
                    for damping in KRYLOV_DAMPINGS:
                        directions[f"gn_k{k}_d{damping:g}"] = flat.dict(krylov_direction(V, Tk, float(b_flat.norm()), k, damping * rho))
                for power in POWERS:     # which power of the exact curvature? (-(G + d)^-p b, same Krylov space)
                    for damping in (1e-2, 1e-3):
                        directions[f"gnpow{power:g}_d{damping:g}"] = flat.dict(
                            krylov_power(V, Tk, float(b_flat.norm()), args.krylov, power, damping * rho))
                if args.deflate:
                    # the top Ritz vectors of G (global, across all matrices): remove them from Muon's and PD's
                    # directions, and add a Newton step inside that small subspace
                    theta, Y = torch.linalg.eigh(Tk)
                    order = torch.argsort(theta, descending=True)[:max(DEFLATE_KS)]
                    theta = theta[order]
                    W = V.T @ Y[:, order].to(V.device, torch.float32)
                    entry["deflation_ritz"] = theta.tolist()
                    for base in ("muon", "pd_a0.5"):
                        d_flat = flat.flat(directions[base])
                        for k in DEFLATE_KS:
                            Wk = W[:, :k]
                            directions[f"{base}-top{k}"] = flat.dict(d_flat - Wk @ (Wk.T @ d_flat))
                            deflate_newton[k] = flat.dict(-(Wk @ ((Wk.T @ b_flat) / (theta[:k].to(V.device, torch.float32) + 1e-3 * rho))))
                    del W
                del V
            for mode, steps, prefix in (("ekfac", args.krylov_pre, "gnp"), ("input", args.krylov_input, "gni")):
                if not steps:
                    continue
                pre = Preconditioner(stats, names, flat, args.pre_damping, mode)
                b_hat = pre.half_inverse(b_flat)
                V, Tk = lanczos(lambda v: pre.half_inverse(operator(pre.half_inverse(v))), b_hat, steps)
                rho_pre = float(Tk[0, 0])
                entry.update({f"rho_{prefix}": rho_pre, f"ritz_values_{prefix}": torch.linalg.eigvalsh(Tk).flip(0).tolist()})
                for k in (k for k in KRYLOV_TRUNCATIONS if k <= steps):
                    for damping in KRYLOV_DAMPINGS:
                        x = krylov_direction(V, Tk, float(b_hat.norm()), k, damping * rho_pre)
                        directions[f"{prefix}_k{k}_d{damping:g}"] = flat.dict(pre.half_inverse(x))
                del V, b_hat
            if args.warm_cg and args.krylov:
                # a few CG steps on the exact GN system, warm-started from Muon's or PD's direction at its GN-optimal
                # scale on the curvature sequences: how many GN products close the gap from the optimizer's direction?
                damping = args.warm_cg_damping * rho
                for base in ("muon", "pd_a0.5"):
                    d0 = flat.flat(directions[base])
                    gd0 = operator(d0) + damping * d0
                    x0 = d0 * (-(b_flat @ d0) / (d0 @ gd0))
                    for k, x in warm_cg(operator, b_flat, x0, damping, WARM_CG_STEPS).items():
                        directions[f"{base}+cg{k}"] = flat.dict(x)
            entry["lanczos_seconds"] = time.time() - t0
            scores = OrderedDict((label, score(model, evals, d)) for label, d in directions.items())
            for k, newton in deflate_newton.items():
                # best two-scale combination of the deflated direction and the top-k Newton step (exact 2 x 2 GN model)
                sn = score(model, evals, newton)
                scores[f"newton-top{k}"] = sn
                for base in ("muon", "pd_a0.5"):
                    sd = scores[f"{base}-top{k}"]
                    both = score(model, evals, OrderedDict((n, directions[f"{base}-top{k}"][n] * (-sd["first"] / sd["q"]) + newton[n])
                                                           for n in names))
                    # model with scales (c1 on the scaled deflated direction, c2 on newton)
                    a = torch.tensor([sd["first"] * (-sd["first"] / sd["q"]), sn["first"]], dtype=torch.float64)
                    q11, q22 = sd["q"] * (sd["first"] / sd["q"]) ** 2, sn["q"]
                    q12 = 0.5 * (both["q"] - q11 - q22)
                    Q = torch.tensor([[q11, q12], [q12, q22]], dtype=torch.float64)
                    try:
                        c = -torch.linalg.solve(Q, a)
                        value = float(-(c @ a + 0.5 * c @ Q @ c))
                    except RuntimeError:
                        value = float("nan")
                    scores[f"{base}-top{k}+newton"] = {"first": both["first"], "q": both["q"], "best_decrease": value,
                                                       "best_scale": float("nan"), "scales": c.tolist() if value == value else None}
            deflate_newton.clear()
            families = {"kfac": "kfac_", "ekfac": "ekfac_", "gn": "gn_", "gnp": "gnp_", "gni": "gni_", "gnpow": "gnpow"}
            best = {f: max((x for x in scores if x.startswith(prefix)), key=lambda x: scores[x]["best_decrease"])
                    for f, prefix in families.items() if any(x.startswith(prefix) for x in scores)}
            best["gnall"] = max((best[f] for f in ("gn", "gnp", "gni") if f in best), key=lambda x: scores[x]["best_decrease"])
            muon_norms = {n: d.norm() for n, d in directions["muon"].items()}
            for label in (best["kfac"], best["gnall"]):
                norms = {n: d.norm() for n, d in directions[label].items()}
                scores[f"{label}@muon_norms"] = score(model, evals, rescaled(directions[label], muon_norms))
                scores[f"muon@{label}_norms"] = score(model, evals, rescaled(directions["muon"], norms))
            scores["gd@muon_norms"] = score(model, evals, rescaled(directions["gd"], muon_norms))
            # where does the within-matrix gap live? PD α½ everywhere except one kind, which takes GN's shape at
            # PD's per-matrix norm (and the complement: GN everywhere except one kind, which takes PD's shape)
            pd_norms = {n: d.norm() for n, d in directions["pd_a0.5"].items()}
            gn_best = directions[best["gnall"]]
            gn_norms = {n: d.norm() for n, d in gn_best.items()}
            for kind in P.KINDS:
                members = {n for n in names if n.endswith("." + kind)}
                swap = OrderedDict((n, gn_best[n] * (pd_norms[n] / gn_best[n].norm().clamp_min(1e-30)) if n in members
                                    else directions["pd_a0.5"][n]) for n in names)
                scores[f"pd_a0.5+gn[{kind}]"] = score(model, evals, swap)
                back = OrderedDict((n, directions["pd_a0.5"][n] * (gn_norms[n] / pd_norms[n].clamp_min(1e-30)) if n in members
                                    else gn_best[n]) for n in names)
                scores[f"gn+pd_a0.5[{kind}]"] = score(model, evals, back)
            if args.kind_alpha:
                # one kind at input power a (PD's rule, Muon's norms), every other kind at PD alpha 1/2
                for kind in P.KINDS:
                    members = {n for n in names if n.endswith("." + kind)}
                    for alpha in KIND_ALPHAS:
                        if alpha == 0.5:
                            continue
                        mixed = OrderedDict()
                        for n in names:
                            if n not in members:
                                mixed[n] = directions["pd_a0.5"][n]
                                continue
                            g = b[n].float()
                            r = kind_roots[n][alpha]
                            d = polar(g @ r) @ r if r is not None else polar(g)
                            mixed[n] = -d * (muon_norm(g.shape) / d.norm().clamp_min(1e-30))
                        scores[f"pd_a0.5|{kind}:a{alpha:g}"] = score(model, evals, mixed)
                # GN's shape in the down projection of one layer at a time (inside PD alpha 1/2)
                for layer in (1, 4, 8):
                    n0 = f"block{layer:02d}.down"
                    swap = OrderedDict((n, gn_best[n] * (pd_norms[n] / gn_best[n].norm().clamp_min(1e-30)) if n == n0
                                        else directions["pd_a0.5"][n]) for n in names)
                    scores[f"pd_a0.5+gn[{n0}]"] = score(model, evals, swap)
            gn_labels = [best[f] for f in ("gn", "gnp", "gni") if f in best]
            main_labels = ["gd", "muon", "pd_a0.25", "pd_a0.5", "two_sided", best["kfac"], best["ekfac"], *gn_labels]
            line = {}
            for label in main_labels:
                c = scores[label]["best_scale"]
                if math.isfinite(c) and c > 0:
                    line[label] = {"scales": [0.5 * c, c, 2 * c],
                                   "loss": loss_along(model, evals, directions[label], [0.5 * c, c, 2 * c])}
            shapes = {label: structure(directions[label], stats, directions["muon"]) for label in main_labels}
            entry.update(best=best, scores=scores, line_search=line, structure=shapes)
            if args.gram and source == "momentum":
                half = args.gram_sequences // 2
                gram_batches = [stream.batch(EVAL + first * T, 1, T, device) for first in range(0, args.gram_sequences)]
                entry["gram"] = {}
                for label in ("muon", best["gnall"]):
                    pieces = [OrderedDict([(n, directions[label][n])]) for n in names]
                    entry["gram"][label] = {"A": gram(model, gram_batches[:half], pieces),
                                            "B": gram(model, gram_batches[half:], pieces)}
            result["inputs"][source] = entry
            saved_directions[source] = {label: OrderedDict((n, d.to(torch.bfloat16).cpu()) for n, d in directions[label].items())
                                        for label in ("gd", "muon", "pd_a0.25", "pd_a0.5", best["kfac"], *gn_labels)}
            summary = {label: round(scores[label]["best_decrease"] * 1e3, 4) for label in
                       main_labels + [f"{best['gnall']}@muon_norms", f"muon@{best['gnall']}_norms", "gd@muon_norms"]}
            print(json.dumps({"item": item, "input": source, "rho": entry.get("rho"), "rho_pre": entry.get("rho_pre"),
                              "best_decrease_x1e3": summary, "seconds": time.time() - t0}), flush=True)
            name = f"{Path(arm).name}_step{step:06d}"
            (args.out / f"{name}.json").write_text(json.dumps(result, indent=1) + "\n")
        torch.save({"meta": {"arm": arm, "step": step}, "directions": saved_directions,
                    "frames": {n: {k: stats[n][k].to(torch.float16).cpu() for k in ("U", "V")} |
                               {k: stats[n][k].cpu() for k in ("lam_B", "lam_C")} for n in names}},
                   args.out / f"{Path(arm).name}_step{step:06d}_directions.pt")
        result["seconds"] = time.time() - started
        (args.out / f"{Path(arm).name}_step{step:06d}.json").write_text(json.dumps(result, indent=1) + "\n")
        print(json.dumps({"item": item, "done": True, "seconds": result["seconds"]}), flush=True)


if __name__ == "__main__":
    main()

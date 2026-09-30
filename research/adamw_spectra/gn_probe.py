"""Second-order audit probes (SECOND_ORDER_AUDIT.md). Measurement only: never trains.

A checkpoint is loaded into an FP32 eager model (TF32 off). A sequence's loss is the mean of its T token
losses; probe losses sum that over the sequences of a batch, so each sequence's gradient is the gradient
of its own mean loss. All curvature estimates share one unit:

- exact Gauss-Newton quadratic form along a weight change D of any hidden matrices (others fixed):
  q(D) = mean over sequences and positions of Var_{p_t}(J_t D), with J_t D by forward-mode AD;
  the GN model of the mean loss is L(W + D) ~ L + <g, D> + q(D) / 2;
- with labels sampled from the model, T * E<g_seq, D>^2 = q(D) in expectation: the GN block is the second
  moment of per-sequence sampled-label gradients, cross-position terms included;
- per-token (K-FAC family) statistics use e~ = T * e, the error of the summed token losses at the matrix
  output, so the per-token estimate of q(D) is the token mean of (e~^T D x)^2.
"""
import contextlib
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch.func import functional_call, jvp

from .model import GPT, ModelConfig

KINDS = ("q", "k", "v", "o", "up", "down")


def hidden_linears(model):
    """Hidden matrices, named as in measured_parameters (block01.q ... blockNN.down), all blocks."""
    out = OrderedDict()
    for index, block in enumerate(model.blocks, start=1):
        for kind, layer in (("q", block.attn.q), ("k", block.attn.k), ("v", block.attn.v),
                            ("o", block.attn.o), ("up", block.mlp.up), ("down", block.mlp.down)):
            out[f"block{index:02d}.{kind}"] = layer
    return out


def build_model(model_config, state_dict, device):
    """FP32 eager model with plain Linear layers (input-statistics buffers are not needed here)."""
    model = GPT(ModelConfig(**{**model_config, "track_input_stats": False, "track_input_cov": False}))
    model.load_state_dict(state_dict)
    return model.to(device=device, dtype=torch.float32).eval()


def load_checkpoint(path, device, model_config=None):
    """A training checkpoint (model, optimizer, config) or a kept next-step weights file (model only;
    pass model_config from its paired checkpoint)."""
    saved = torch.load(path, map_location="cpu", weights_only=False)
    config = saved.get("config") or saved.get("metadata", {}).get("config")
    model_config = model_config or config["model"]
    return build_model(model_config, saved["model"], device), saved


def math_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, enable_gqa=False):
    """Explicit causal attention: differentiable to any order and in forward mode."""
    if attn_mask is not None or dropout_p or enable_gqa:
        raise NotImplementedError("Only the model's causal, dropout-free call is supported")
    scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    scores = (q @ k.transpose(-2, -1)) * scale
    if is_causal:
        t = q.shape[-2]
        causal = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~causal, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


@contextlib.contextmanager
def explicit_attention():
    """Swap SDPA (whose fused kernels lack forward-mode AD) for math_attention."""
    original = F.scaled_dot_product_attention
    F.scaled_dot_product_attention = math_attention
    try:
        yield
    finally:
        F.scaled_dot_product_attention = original


def token_losses(logits, targets):
    return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten(), reduction="none").view(targets.shape)


def probe_loss(logits, targets):
    """Sum over sequences of each sequence's mean token loss."""
    return token_losses(logits, targets).mean(1).sum()


def sample_labels(logits, generator=None):
    probs = torch.softmax(logits.detach().float(), dim=-1)
    return torch.multinomial(probs.flatten(0, 1), 1, generator=generator).view(logits.shape[:-1])


class Recorder:
    """Keeps each hidden matrix's input x (forward pre-hook) and, for every backward pass, the error e at
    its output (tensor hook). errors[name] lists one (B, T, d_out) tensor per backward pass, in order."""

    def __init__(self, model, names=None):
        layers = hidden_linears(model)
        self.layers = layers if names is None else OrderedDict((n, layers[n]) for n in names)
        self.inputs, self.errors = {}, {name: [] for name in self.layers}
        self.handles = []
        for name, layer in self.layers.items():
            self.handles.append(layer.register_forward_pre_hook(self._keep_input(name)))
            self.handles.append(layer.register_forward_hook(self._watch_output(name)))

    def _keep_input(self, name):
        def hook(module, args):
            self.inputs[name] = args[0].detach()
        return hook

    def _watch_output(self, name):
        def hook(module, args, output):
            if output.requires_grad:
                output.register_hook(lambda grad: self.errors[name].append(grad.detach()))
        return hook

    def clear(self):
        self.inputs.clear()
        for errors in self.errors.values():
            errors.clear()

    def remove(self):
        for handle in self.handles:
            handle.remove()


def gradient_passes(model, recorder, tokens, targets, generator=None, draws=1):
    """One forward, one backward with the data labels, then `draws` backwards with model-sampled labels.

    Returns (batch gradients of the probe loss for the recorded matrices, token losses). Afterwards
    recorder.errors[name] = [e_true, e_sampled_1, ...], each the error of the per-sequence mean loss."""
    recorder.clear()
    weights = [layer.weight for layer in recorder.layers.values()]
    logits = model(tokens)
    losses = token_losses(logits, targets)
    grads = torch.autograd.grad(losses.mean(1).sum(), weights, retain_graph=draws > 0)
    for draw in range(draws):
        sampled = sample_labels(logits, generator)
        torch.autograd.grad(probe_loss(logits, sampled), weights, retain_graph=draw < draws - 1)
    return OrderedDict(zip(recorder.layers, grads)), losses.detach()


def per_sequence_gradients(x, e):
    """(B, T, d_in), (B, T, d_out) -> (B, d_out, d_in): each sequence's gradient sum_s e_s x_s^T."""
    return torch.einsum("bto,bti->boi", e, x)


def parameter_keys(model, names):
    """Hidden-matrix names (block01.q) map to their weight; other names are raw parameter names."""
    module_names = {id(module): name for name, module in model.named_modules()}
    layers = hidden_linears(model)
    return [module_names[id(layers[name])] + ".weight" if name in layers else name for name in names]


@torch.no_grad()
def directional_terms(model, tokens, targets, directions):
    """One forward-mode pass along D: the mean loss L, its exact directional derivative <g, D>, and q(D).
    The GN model of the mean loss along D is L + <g, D> + q(D) / 2 (per-sequence values for error bars)."""
    names = list(directions)
    keys = parameter_keys(model, names)
    base = {**dict(model.named_parameters()), **dict(model.named_buffers())}

    def logits_of(*weights):
        return functional_call(model, {**base, **dict(zip(keys, weights))}, (tokens,))

    primals = tuple(base[key] for key in keys)
    tangents = tuple(directions[name].to(device=primal.device, dtype=primal.dtype)
                     for name, primal in zip(names, primals))
    with explicit_attention():
        logits, dz = jvp(logits_of, primals, tangents)
    p = torch.softmax(logits.float(), dim=-1)
    dz = dz.float()
    mean = (p * dz).sum(-1)
    variance = (p * (dz - mean.unsqueeze(-1)) ** 2).sum(-1)
    first = mean - dz.gather(-1, targets.unsqueeze(-1)).squeeze(-1)      # (p - onehot(y)) . dz
    losses = token_losses(logits, targets)
    return {"loss": losses.mean(1), "first": first.mean(1), "q": variance.mean(1)}


@torch.no_grad()
def gn_quadratic(model, tokens, directions):
    """Exact q(D) (see module docstring) for a dict name -> direction over hidden matrices.
    Also returns <J D> statistics: per-sequence q values (B,) for error bars."""
    names = list(directions)
    keys = parameter_keys(model, names)
    base = {**dict(model.named_parameters()), **dict(model.named_buffers())}

    def logits_of(*weights):
        return functional_call(model, {**base, **dict(zip(keys, weights))}, (tokens,))

    primals = tuple(base[key] for key in keys)
    tangents = tuple(directions[name].to(device=primal.device, dtype=primal.dtype)
                     for name, primal in zip(names, primals))
    with explicit_attention():
        logits, dz = jvp(logits_of, primals, tangents)
    p = torch.softmax(logits.float(), dim=-1)
    dz = dz.float()
    mean = (p * dz).sum(-1)
    variance = (p * (dz - mean.unsqueeze(-1)) ** 2).sum(-1)   # Var_p(dz) without cancellation
    per_sequence = variance.mean(1)
    return per_sequence.mean(), per_sequence


def vo_gauge_direction(model, block_index, head, generator=None):
    """First-order V/O gauge change for one head (an exact symmetry without biases):
    W_v[h] <- A W_v[h] and W_o[:, h] <- -W_o[:, h] A, A random. Returns the direction dict."""
    block = model.blocks[block_index - 1]
    width, heads = model.config.n_embd, model.config.n_head
    size = width // heads
    rows = slice(head * size, (head + 1) * size)
    a = torch.randn(size, size, generator=generator).to(block.attn.v.weight)
    dv = torch.zeros_like(block.attn.v.weight)
    do = torch.zeros_like(block.attn.o.weight)
    dv[rows] = a @ block.attn.v.weight[rows]
    do[:, rows] = -block.attn.o.weight[:, rows] @ a
    return {f"block{block_index:02d}.v": dv, f"block{block_index:02d}.o": do}


def radial_head_direction(model, block_index, kind, head):
    """Scale one head's rows of q or k: a symmetry when that head is RMS-normalized (qk_norm)."""
    block = model.blocks[block_index - 1]
    layer = getattr(block.attn, kind)
    size = model.config.n_embd // model.config.n_head
    rows = slice(head * size, (head + 1) * size)
    d = torch.zeros_like(layer.weight)
    d[rows] = layer.weight[rows]
    return {f"block{block_index:02d}.{kind}": d}


def norm_gain_direction(model, block_index, which, generator=None):
    """A norm's gain against the columns of the matrices that read it (ln1: q, k, v; ln2: up):
    gain <- gain * (1 + a), W[:, i] <- W[:, i] / (1 + a_i); first order: d gain = gain * a, dW = -W diag(a)."""
    block = model.blocks[block_index - 1]
    norm = getattr(block, which)
    readers = ("q", "k", "v") if which == "ln1" else ("up",)
    a = torch.randn(norm.weight.shape, generator=generator).to(norm.weight)
    names = {id(module): name for name, module in model.named_modules()}
    out = {names[id(norm)] + ".weight": norm.weight * a}
    for kind in readers:
        layer = getattr(block.attn, kind) if kind != "up" else block.mlp.up
        out[f"block{block_index:02d}.{kind}"] = -layer.weight * a
    return out


def residual_scale_direction(model):
    """Joint scale of the residual stream: token and position embeddings and every writer (o, down).
    Every read is RMS-normalized, so this is a symmetry up to the norms' epsilon."""
    out = {"embed.weight": model.embed.weight.detach().clone(),
           "position.weight": model.position.weight.detach().clone()}
    for index in range(1, len(model.blocks) + 1):
        for kind in ("o", "down"):
            out[f"block{index:02d}.{kind}"] = hidden_linears(model)[f"block{index:02d}.{kind}"].weight.detach().clone()
    return out


def random_like(directions, generator=None):
    """Gaussian directions with the same Frobenius norm, matrix by matrix."""
    out = {}
    for name, d in directions.items():
        r = torch.randn(d.shape, generator=generator).to(d)
        out[name] = r * (d.norm() / r.norm().clamp_min(1e-30))
    return out


def second_moments(x, e_tilde):
    """Per-token second moments C = E[x x^T] and B = E[e~ e~^T] (FP64 sums and token count)."""
    x2 = x.reshape(-1, x.shape[-1]).double()
    e2 = e_tilde.reshape(-1, e_tilde.shape[-1]).double()
    return x2.T @ x2, e2.T @ e2, x2.shape[0]


def eigenbasis(moment_sum, count):
    """Eigenvalues (descending) and eigenvectors of a per-token second moment."""
    values, vectors = torch.linalg.eigh(moment_sum / count)
    return values.flip(0), vectors.flip(1)


class Frame:
    """Per-pair statistics of one hidden matrix in its Kronecker frame (U: eigvecs of B, V: eigvecs of C).

    For every pair (i, j), with e~ = T e and g the per-sequence gradient of the mean loss:
      ekfac[i, j]  = token mean of (u_i^T e~)^2 (v_j^T x)^2         (sampled labels; per-token GN)
      exact[i, j]  = T * sequence mean of (u_i^T g v_j)^2             (sampled labels; exact GN diagonal)
      mean[i, j], second[i, j] = sequence mean of u_i^T g v_j, and of its square (data labels):
                     signal and per-sequence noise.
    K-FAC's prediction is lam_B[i] * lam_C[j]. Sums are FP64."""

    def __init__(self, U, V, lam_B, lam_C, seq_len):
        self.U, self.V = U.float(), V.float()
        self.lam_B, self.lam_C, self.T = lam_B, lam_C, seq_len
        shape = (U.shape[1], V.shape[1])
        device = U.device
        self.ekfac = torch.zeros(shape, dtype=torch.float64, device=device)
        self.exact = torch.zeros(shape, dtype=torch.float64, device=device)
        self.mean = torch.zeros(shape, dtype=torch.float64, device=device)
        self.second = torch.zeros(shape, dtype=torch.float64, device=device)
        # second moment of each add() call's mean (contiguous blocks of sequences): tests the 1/b noise law
        self.block_second = torch.zeros(shape, dtype=torch.float64, device=device)
        self.tokens = self.sequences = self.draws = self.blocks = 0
        self.block_sizes = set()

    def add(self, x, e_true, e_sampled):
        """x: (B, T, d_in); e_true: (B, T, d_out); e_sampled: list of (B, T, d_out), one per label draw."""
        T = self.T
        xs = x.float() @ self.V
        g = torch.einsum("bto,bti->boi", e_true.float() @ self.U, xs)
        self.mean += g.sum(0).double()
        self.second += (g * g).sum(0).double()
        block = g.mean(0)
        self.block_second += (block * block).double()
        self.blocks += 1
        self.block_sizes.add(x.shape[0])
        for e in e_sampled:
            es = (T * e.float()) @ self.U
            self.ekfac += torch.einsum("bto,bti->oi", es * es, xs * xs).double()
            gs = torch.einsum("bto,bti->boi", es, xs) / T
            self.exact += T * (gs * gs).sum(0).double()
            self.tokens += x.shape[0] * x.shape[1]
            self.draws += x.shape[0]
        self.sequences += x.shape[0]

    def summary(self):
        n = self.sequences
        mean = self.mean / n
        noise = (self.second / n - mean ** 2) * n / max(n - 1, 1)   # per-sequence variance
        # mean^2 overestimates the squared signal by noise / n; signal2 is the unbiased estimate
        out = {"kfac": torch.outer(self.lam_B, self.lam_C), "ekfac": self.ekfac / max(self.tokens, 1),
               "exact": self.exact / max(self.draws, 1), "signal": mean, "signal2": mean ** 2 - noise / n,
               "noise": noise, "sequences": n, "draws": self.draws}
        if len(self.block_sizes) == 1 and self.blocks > 1:
            size = next(iter(self.block_sizes))
            block_var = (self.block_second / self.blocks - mean ** 2) * self.blocks / (self.blocks - 1)
            out["block_size"] = size
            out["block_var_ratio"] = block_var * size / noise.clamp_min(1e-300)   # 1 if sequences independent
        return out


class Marginals:
    """One-sided marginals of one hidden matrix's GN block and of its approximations (FP64 sums).

    Input side (d_in x d_in), all in the units of q:
      exact_in  = T * E[g^T g]          per-sequence sampled-label gradients: Tr_out of the exact GN block
      token_in  = E_tok[|e~|^2 x x^T]   per-token (error-weighted C): Tr_out of the per-token GN
      C         = E_tok[x x^T]          K-FAC's input factor (Tr_out of C (x) B is tr(B) C)
      ef_in     = T * E[g_true^T g_true] with data labels (empirical Fisher; includes the signal)
      mean_grad = E[g_true]             for minibatch Grams: E[G_b^T G_b] = gbar^T gbar + Cov/b
    Output side (d_out x d_out): exact_out = T * E[g g^T], token_out = E_tok[|x|^2 e~ e~^T],
    B = E_tok[e~ e~^T], ef_out."""

    def __init__(self, d_out, d_in, seq_len, device):
        z = lambda n: torch.zeros(n, n, dtype=torch.float64, device=device)
        self.T = seq_len
        self.sums = {"exact_in": z(d_in), "token_in": z(d_in), "C": z(d_in), "ef_in": z(d_in),
                     "exact_out": z(d_out), "token_out": z(d_out), "B": z(d_out), "ef_out": z(d_out)}
        self.mean_grad = torch.zeros(d_out, d_in, dtype=torch.float64, device=device)
        self.x_sum = torch.zeros(d_in, dtype=torch.float64, device=device)
        self.between = z(d_in)   # sum over sequences of xbar_seq xbar_seq^T (sequence-mean part of C)
        self.tokens = self.sequences = self.draws = self.sampled_tokens = 0

    def add(self, x, e_true, e_sampled):
        T, s = self.T, self.sums
        x2 = x.reshape(-1, x.shape[-1]).float()
        s["C"] += (x2.T @ x2).double()
        self.x_sum += x2.sum(0).double()
        xbar = x.float().mean(1)
        self.between += (xbar.T @ xbar).double()
        xf = x.reshape(-1, x.shape[-1]).float()
        g = per_sequence_gradients(x.float(), e_true.float())                   # (B, d_out, d_in), FP32
        s["ef_in"] += T * torch.einsum("boi,boj->ij", g, g).double()
        s["ef_out"] += T * torch.einsum("boi,bpi->op", g, g).double()
        self.mean_grad += g.sum(0).double()
        for e in e_sampled:
            ef = (T * e).reshape(-1, e.shape[-1]).float()
            s["B"] += (ef.T @ ef).double()
            s["token_in"] += ((xf * ef.pow(2).sum(1, keepdim=True)).T @ xf).double()
            s["token_out"] += ((ef * xf.pow(2).sum(1, keepdim=True)).T @ ef).double()
            gs = per_sequence_gradients(x.float(), e.float())
            s["exact_in"] += T * torch.einsum("boi,boj->ij", gs, gs).double()
            s["exact_out"] += T * torch.einsum("boi,bpi->op", gs, gs).double()
            self.draws += x.shape[0]
            self.sampled_tokens += xf.shape[0]
        self.tokens += x2.shape[0]
        self.sequences += x.shape[0]

    def summary(self):
        s = self.sums
        out = {"C": s["C"] / self.tokens, "B": s["B"] / self.sampled_tokens,  # noqa: E501
               "token_in": s["token_in"] / self.sampled_tokens, "token_out": s["token_out"] / self.sampled_tokens,
               "exact_in": s["exact_in"] / self.draws, "exact_out": s["exact_out"] / self.draws,
               "ef_in": s["ef_in"] / self.sequences, "ef_out": s["ef_out"] / self.sequences,
               "mean_grad": self.mean_grad / self.sequences, "x_mean": self.x_sum / self.tokens,
               "C_between": self.between / self.sequences,
               "sequences": self.sequences, "draws": self.draws, "T": self.T}
        out["C_within"] = out["C"] - out["C_between"]
        return out


def within_between_fit(m, target="exact_in"):
    """Least-squares fit target ~ a tr(B) C_within + b tr(B) C_between (Frobenius), the attention hypothesis:
    keys see only within-sequence-centered inputs (a ~ 1, b ~ 0 without QK-norm); values add sequence-coherent
    curvature (b > 1); position-local matrices follow K-FAC (a ~ b ~ 1). Returns a, b and relative residuals."""
    scale = m["B"].trace()
    X, Y, E = scale * m["C_within"], scale * m["C_between"], m[target]
    gram = torch.tensor([[(X * X).sum(), (X * Y).sum()], [(X * Y).sum(), (Y * Y).sum()]], dtype=torch.float64)
    rhs = torch.tensor([(E * X).sum(), (E * Y).sum()], dtype=torch.float64)
    a, b = torch.linalg.solve(gram, rhs).tolist()
    norm = float(E.norm())
    return {"a_within": a, "b_between": b,
            "residual_fit": float((E - a * X - b * Y).norm()) / norm,
            "residual_kfac": float((E - scale * m["C"]).norm()) / norm,
            "between_share_of_C": float(m["C_between"].trace() / m["C"].trace())}


def _unit(a):
    return a / a.trace()


def _overlap(a, b, k):
    """Mean squared overlap of the top-k eigenvector subspaces of two symmetric matrices (1 = same)."""
    va = torch.linalg.eigh(a)[1][:, -k:]
    vb = torch.linalg.eigh(b)[1][:, -k:]
    return float((va.T @ vb).pow(2).sum() / k)


def _slope(x, y):
    lx, ly = torch.log(x), torch.log(y)
    lx, ly = lx - lx.mean(), ly - ly.mean()
    return float((lx * ly).sum() / (lx * lx).sum())


def marginal_report(m, top=16, slope_count=64):
    """Compare exact one-sided GN marginals with K-FAC, per-token and empirical-Fisher versions.
    Ratios are curvatures along eigenvectors of C (input side) or B (output side), relative to K-FAC."""
    out = {}
    for side, factor, other, exact, token, ef in (("in", "C", "B", "exact_in", "token_in", "ef_in"),
                                                   ("out", "B", "C", "exact_out", "token_out", "ef_out")):
        f = m[factor]
        scale = m[other].trace()
        lam, vec = torch.linalg.eigh(f)
        lam, vec = lam.flip(0), vec.flip(1)
        def along(a, v):
            return torch.einsum("ij,ik,jk->k", a, v, v)
        kfac = scale * lam
        rows = {name: along(m[key], vec) for name, key in (("exact", exact), ("token", token), ("ef", ef))}
        k = min(top, lam.numel())
        n = min(slope_count, lam.numel())
        positive = lam[:n] > 0
        report = {
            "top_ratio_exact_kfac": (rows["exact"][:k] / kfac[:k]).tolist(),
            "top_ratio_token_kfac": (rows["token"][:k] / kfac[:k]).tolist(),
            "top_ratio_exact_token": (rows["exact"][:k] / rows["token"][:k]).tolist(),
            "slope_exact_vs_factor": _slope(lam[:n][positive], rows["exact"][:n][positive].clamp_min(1e-300)),
            "slope_token_vs_factor": _slope(lam[:n][positive], rows["token"][:n][positive].clamp_min(1e-300)),
            "unit_trace_rel_frobenius_diff": {name: float((_unit(m[key]) - _unit(f)).norm() / _unit(f).norm())
                                              for name, key in (("exact", exact), ("token", token), ("ef", ef))},
            "top8_overlap_with_factor": {name: _overlap(m[key], f, 8) for name, key in
                                         (("exact", exact), ("token", token), ("ef", ef))},
            "trace_ratio_exact_kfac": float(m[exact].trace() / (scale * f.trace())),
            "trace_ratio_token_kfac": float(m[token].trace() / (scale * f.trace())),
            "factor_top_over_median": float(lam[0] / lam.median()),
            "factor_participation_ratio": float(lam.sum() ** 2 / (lam ** 2).sum()),
        }
        if side == "in":
            u = m["x_mean"] / m["x_mean"].norm()
            q = lambda a: float(u @ a @ u)
            report["mean_direction"] = {"share_of_C": float(q(f) / f.trace()),
                                        "exact_over_kfac": q(m[exact]) / (float(scale) * q(f)),
                                        "token_over_kfac": q(m[token]) / (float(scale) * q(f)),
                                        "exact_over_token": q(m[exact]) / q(m[token])}
        out[side] = report
    # Gradient noise scale in sequences (McCandlish B_simple): per-sequence covariance trace over the
    # squared mean gradient; mean^2 is debiased by noise / N.
    g, n = m["mean_grad"], m["sequences"]
    second = float(m["ef_in"].trace()) / m["T"]                 # E|g_seq|^2 (data labels)
    mean2 = float((g * g).sum())
    noise = (second - mean2) * n / max(n - 1, 1)
    signal = mean2 - noise / n
    out["gradient"] = {"signal": signal, "noise": noise,
                       "noise_scale_sequences": noise / signal if signal > 0 else float("inf")}
    return out

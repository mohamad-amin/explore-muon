"""Plain momentum Muon with the paper's five NS polynomials and auxiliary AdamW.

Architecture, data and initialization belong to the existing depth study. This
is a documented reconstruction, not a claim to recover unpublished paper code.
"""

import math
from collections import defaultdict

import torch
import torch.distributed as dist


NS_COEFFICIENTS = (
    (4.0848, -6.8946, 2.9270),
    (3.9505, -6.3029, 2.6377),
    (3.7418, -5.5913, 2.3037),
    (2.8769, -3.1427, 1.2046),
    (2.8366, -3.0525, 1.2012),
)
# Muon's retuned quintic, repeated ns_steps times (the deflation paper's NS mapping).
JORDAN_QUINTIC = (3.4445, -4.7750, 2.0315)


def ns_schedule(config):
    """The paper's five polynomials by default, `ns_steps` classic Muon quintics, or "svd"
    (exact polar factor, a control for incomplete NS convergence)."""
    polynomial = config.get("ns_polynomial", "paper5")
    if polynomial == "paper5":
        return NS_COEFFICIENTS
    if polynomial == "svd":
        return "svd"
    return (JORDAN_QUINTIC,) * config["ns_steps"]


def _iterate(x, schedule):
    if x.is_cuda:
        x = x.bfloat16()
    transpose = x.shape[-2] > x.shape[-1]
    if transpose:
        x = x.mT
    for a, b, c in schedule:
        gram = x @ x.mT
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return (x.mT if transpose else x).float()


def exact_polar(momentum):
    """U V^T from an FP32 SVD (FP64 fallback): every nonzero singular value mapped to one."""
    try:
        u, _, vh = torch.linalg.svd(momentum.float(), full_matrices=False)
    except RuntimeError:
        u, _, vh = torch.linalg.svd(momentum.double(), full_matrices=False)
    return (u @ vh).float()


def newton_schulz(momentum, schedule=NS_COEFFICIENTS):
    """Independent matrices in a batch; FP32 normalization, BF16 CUDA products."""
    if isinstance(schedule, str):
        return exact_polar(momentum)
    x = momentum.float()
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7)
    return _iterate(x, schedule)


# Spectral deflation (Sun, Kang & Yang 2026, arXiv 2609.21102), ported from their
# public code (defmuon/optim/deflation_batched.py at 1eb15ce): batched randomized
# SVD with FP64 CholeskyQR Gram factors, the head gate and remove-then-restore entry.
def _cholesky_qr(x):
    gram = x.double().mT @ x.double()
    factor, _ = torch.linalg.cholesky_ex(gram)
    return torch.linalg.solve_triangular(factor.to(x.dtype).mT, x, upper=True, left=False)


def _randomized_svd(a, omega, subspace_iters):
    q = _cholesky_qr(a @ omega)
    for _ in range(subspace_iters):
        q = _cholesky_qr(a @ _cholesky_qr(a.mT @ q))
    small = q.mT @ a
    u, s, vh = torch.linalg.svd(small, full_matrices=False,
                                driver="gesvda" if small.is_cuda else None)
    return q @ u, s, vh.mT


def deflated_entry(x, generator=None, omega=None, window=0.025, oversample=0.025,
                   subspace_iters=1, threshold=0.1, pad=1.01):
    """FP32 NS entry with the head removed from the Frobenius divisor.

    When the leading window is head-dominated, every singular pair above
    threshold * sigma1 is removed, the remainder is Frobenius-normalized and the head
    is restored at 1/pad. Other matrices take A / (||A||_F + 1e-7). Returns (entry, k).
    """
    tall = x.shape[-2] >= x.shape[-1]
    a = x if tall else x.mT
    batch, _, m = a.shape
    width = min(math.ceil(window * m) + math.ceil(oversample * m), m)
    gate = min(max(1, math.ceil(window * m)), width)
    if omega is None:
        omega = torch.randn(batch, m, width, device=a.device, dtype=a.dtype, generator=generator)
    u, s, v = _randomized_svd(a, omega, subspace_iters)
    above = (s > threshold * s[:, :1]).sum(dim=1)
    fired = (s[:, gate - 1] < threshold * s[:, 0]) & (above >= 1) & (above < s.shape[1])
    k = above.clamp(1, s.shape[1] - 1)
    head = (torch.arange(s.shape[1], device=a.device)[None, :] < k[:, None]) * fired[:, None]
    rest = a - (u * (s * head)[:, None, :]) @ v.mT
    entry = (rest / (rest.norm(dim=(-2, -1), keepdim=True) + 1e-7)
             + (u * ((1.0 / pad) * head)[:, None, :]) @ v.mT)
    return (entry if tall else entry.mT), k * fired


def deflated_newton_schulz(momentum, schedule, generator=None, **deflation):
    entry, k = deflated_entry(momentum.float(), generator=generator, **deflation)
    return _iterate(entry, schedule), k


def tracked_entry(x, v, power_iters=1, head_weight=1.0, pad=1.01):
    """Rank-1 deflation tracked by warm-started power iteration.

    Momentum changes by one gradient per step, so its top singular pair moves slowly;
    `v` (the pair's vector on the shorter side) is refined from the previous step.
    The head is restored at head_weight / pad (0 drops it from the update).
    Returns (entry, v, head energy fraction sigma1^2 / ||A||_F^2, u).
    """
    tall = x.shape[-2] >= x.shape[-1]
    a = x if tall else x.mT
    for _ in range(power_iters):
        v = (a.mT @ (a @ v[..., None]))[..., 0]
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    w = (a @ v[..., None])[..., 0]
    sigma = w.norm(dim=-1, keepdim=True)
    u = w / sigma.clamp_min(1e-30)
    head = u[..., :, None] * v[..., None, :]
    rest = a - sigma[..., None] * head
    entry = rest / (rest.norm(dim=(-2, -1), keepdim=True) + 1e-7) + (head_weight / pad) * head
    energy = sigma[..., 0] ** 2 / (a.norm(dim=(-2, -1)) ** 2).clamp_min(1e-30)
    return (entry if tall else entry.mT), v, energy, u


def _eigenbasis(matrix):
    """Eigenvectors of a symmetric PSD matrix, descending eigenvalues (FP32, FP64 fallback)."""
    try:
        _, q = torch.linalg.eigh(matrix)
    except RuntimeError:
        _, q = torch.linalg.eigh(matrix.double())
        q = q.float()
    return torch.flip(q, [1])


def _into_basis(x, q_row, q_col):
    """Rotate into the eigenbasis; an unused side (None) is the identity."""
    if q_row is not None:
        x = q_row.T @ x
    return x @ q_col if q_col is not None else x


def _out_of_basis(x, q_row, q_col):
    if q_row is not None:
        x = q_row @ x
    return x @ q_col.T if q_col is not None else x


def soap_precondition(update, soap, beta2, denom_power, eps=1e-8, grad=None, column=False):
    """SOAP-style normalization of the momentum before NS (modded-nanogpt Track-3 SOAP-Muon).

    Projects onto the eigenbases of EMA(G G^T) and EMA(G^T G), divides by an EMA of the
    projected momentum's squares to the power `denom_power`, rotates back and restores the
    input's Frobenius norm. Returns the input unchanged until statistics exist. Basis
    ablations keep a side as the identity (soap["sides"] lists the rotated sides). With
    `grad`, the second moment tracks the projected gradient (SOAP's signal-to-noise
    weighting) instead of the projected update (the Track-3 reference, nearly a sign).
    With `column`, each column shares one denominator (the row-mean of its second moment),
    so the normalization is a one-sided Kronecker preconditioner instead of entrywise.
    """
    if not soap["ready"]:
        return update
    q_row, q_col = soap["q_row"], soap["q_col"]
    projected = _into_basis(update, q_row, q_col)
    tracked = projected if grad is None else _into_basis(grad, q_row, q_col)
    soap["exp_avg_sq"].mul_(beta2).add_(tracked.square(), alpha=1 - beta2)
    second = soap["exp_avg_sq"].mean(dim=0, keepdim=True) if column else soap["exp_avg_sq"]
    denom = second.clamp_min(eps * eps).pow(denom_power)
    preconditioned = _out_of_basis(projected / denom, q_row, q_col)
    return preconditioned * (update.norm() / preconditioned.norm().clamp_min(1e-30))


def soap_update_statistics(grad, soap, beta2, col_gram=None):
    """EMA Gram factors from the raw gradient; eigenbasis refreshed every step by one sorted QR step.

    `col_gram` replaces the input-side factor EMA(G^T G) by a given matrix (the layer's
    activation second moment E[x x^T] for the "right_act" basis).
    """
    if "row" in soap["sides"]:
        soap["row_gg"].lerp_(grad @ grad.T, 1 - beta2)
    if "col" in soap["sides"]:
        if col_gram is not None:
            soap["col_gg"].copy_(col_gram)
        else:
            soap["col_gg"].lerp_(grad.T @ grad, 1 - beta2)
    if not soap["ready"]:
        for side in soap["sides"]:
            soap[f"q_{side}"] = _eigenbasis(soap[f"{side}_gg"])
        soap["ready"] = True
        return
    for side, dim in (("row", 0), ("col", 1)):
        if side not in soap["sides"]:
            continue
        q, gram = soap[f"q_{side}"], soap[f"{side}_gg"]
        order = torch.argsort(torch.diag(q.T @ gram @ q), descending=True)
        soap["exp_avg_sq"] = soap["exp_avg_sq"].index_select(dim, order)
        soap[f"q_{side}"], _ = torch.linalg.qr(gram @ q[:, order])


SOAP_SIDES = {"both": ("row", "col"), "left": ("row",), "right": ("col",), "right_act": ("col",), "none": ()}


def new_soap_state(parameter, basis="both"):
    rows, cols = parameter.shape
    sides = SOAP_SIDES[basis]
    state = {"exp_avg_sq": torch.zeros(rows, cols, device=parameter.device), "sides": sides,
             "ready": False, "q_row": None, "q_col": None}
    for side, size in (("row", rows), ("col", cols)):
        if side in sides:
            state[f"{side}_gg"] = torch.zeros(size, size, device=parameter.device)
    return state


def whitening_factors(module, fixed_beta=None, power=0.5):
    """Mean-input direction v and data-norm factor beta for one StatLinear.

    With input second moment Sigma ~ c0 I + xbar xbar^T, steepest descent under
    ||dW Sigma^(1/2)||_op is polar(G P) P with P = I - (1 - beta) v v^T and
    beta = (c0 / (c0 + ||xbar||^2))^power, power 1/2 (power 1/4 is the Shampoo-style
    geometric mean with plain Muon). Before any statistics exist, v = 0 (plain Muon).
    """
    weight = module.input_weight.clamp_min(1e-12)
    mean = module.input_mean / weight
    mean_sq = mean.pow(2).sum()
    c0 = ((module.input_sq / weight - mean_sq) / mean.numel()).clamp_min(1e-12)
    beta = (c0 / (c0 + mean_sq)) ** power if fixed_beta is None else torch.full_like(c0, fixed_beta)
    return mean / mean_sq.sqrt().clamp_min(1e-12), beta


def data_norm_root(module, alpha, damping, center=False, with_cov=False):
    """C^(-alpha) for the layer's input second moment C = E[x x^T] (EMA, bias-corrected).

    C is scaled to unit mean eigenvalue and damped by `damping` (relative), so the root is
    I for isotropic inputs. Steepest descent under ||dW C^alpha||_op is polar(G C^-alpha) C^-alpha;
    alpha = 1/2 is the full data norm (rank-1 C gives mean-whitened Muon), smaller alpha is milder.
    """
    weight = module.input_cov_weight.clamp_min(1e-12)
    cov = (module.input_cov / weight).double()
    if center:  # covariance about the (same-window) EMA mean: the mean direction is not singled out
        mean = (module.input_cov_mean / weight).double()
        cov = cov - torch.outer(mean, mean)
    eigenvalues, vectors = torch.linalg.eigh(0.5 * (cov + cov.T))
    eigenvalues = eigenvalues.clamp_min(0)
    unit = eigenvalues / eigenvalues.mean().clamp_min(1e-30)
    root = ((vectors * (unit + damping).pow(-alpha)) @ vectors.T).float()
    if with_cov:
        return root, ((vectors * unit) @ vectors.T).float()
    return root


# Published comparators, ported from modded-nanogpt records/track_3_optimization (MUON_CASE.md,
# "Head-to-head with Newton-Muon and PMuon"). Only their preconditioners are adopted; momentum,
# NS, shape scaling, decay and the auxiliary AdamW stay this study's Muon.
def diagonal_blocks(matrix, blocks):
    """The `blocks` diagonal (d x d) blocks of a square (blocks * d) matrix, as (blocks, d, d)."""
    d = matrix.size(-1) // blocks
    return torch.stack([matrix[i * d:(i + 1) * d, i * d:(i + 1) * d] for i in range(blocks)])


def newton_inverse(cov, damping):
    """Newton-Muon's (C + damping * mean diag(C) I)^-1 per (d x d) block, by Cholesky; a block whose
    factorization fails becomes I (record #15, `Muon._refresh_preconditioner`)."""
    d = cov.size(-1)
    reg = cov.diagonal(dim1=-2, dim2=-1).sum(-1) / d * damping + 1e-8
    eye = torch.eye(d, device=cov.device, dtype=cov.dtype)
    factor, info = torch.linalg.cholesky_ex(cov + reg[..., None, None] * eye, upper=False)
    inverse = torch.cholesky_inverse(factor, upper=False)
    if info.any():
        inverse[info != 0] = eye
    return inverse


def streaming_cov_power(cov, state, key, gamma, generator, eps=1e-6):
    """PMuon's cov^(-gamma) (record #18, `_streaming_cov_power`): one warm-started QR (subspace)
    iteration per step, Rayleigh-quotient eigenvalues, rescaled by sqrt(n) / ||d||."""
    n = cov.size(0)
    q = state.get(key)
    if q is None:
        q, _ = torch.linalg.qr(torch.randn(n, n, device=cov.device, dtype=cov.dtype, generator=generator))
    q, _ = torch.linalg.qr(cov @ q)
    state[key] = q
    d = (q * (cov @ q)).sum(dim=0).clamp_min(eps).pow(-gamma)
    return (q * d) @ q.T * (n ** 0.5 / (d.norm() + eps))


class MuonAdamW(torch.optim.Optimizer):
    """One serializable state dictionary and scheduled groups for both optimizers."""

    def __init__(self, model, config, device, adam_impl):
        body, auxiliary = [], []
        for name, parameter in model.named_parameters():
            target = body if name.startswith("blocks.") and parameter.ndim == 2 else auxiliary
            target.append(parameter)
        if len(body) != 6 * model.config.n_layer:
            raise ValueError("Muon must cover exactly Q/K/V/O/up/down in each block")
        groups = [dict(params=body, algorithm="muon", lr_scale=1.0),
                  dict(params=auxiliary, algorithm="adamw",
                       lr_scale=config["aux_learning_rate"] / config["learning_rate"])]
        super().__init__(groups, dict(lr=config["learning_rate"], weight_decay=config["weight_decay"]))
        self.momentum = config["muon_momentum"]
        # A standalone data-parallel runner may own all optimizer state on rank 0.
        # Existing callers retain the original owner-sharded behavior by default.
        self.distributed_optimizer = config.get("distributed_optimizer", True)
        self._adam = torch.optim.AdamW(
            [self.param_groups[1]], lr=config["aux_learning_rate"],
            betas=tuple(config["betas"]), eps=config["epsilon"],
            weight_decay=config["weight_decay"], foreach=adam_impl == "foreach",
            fused=adam_impl == "fused")
        self.param_groups[1] = self._adam.param_groups[0]
        self.param_groups[1]["lr"] = config["aux_learning_rate"]
        self._adam.state = self.state
        # Compile the model, but keep the NS polynomial's explicit BF16
        # operation boundaries. Fusing these changed the archived reference
        # direction by 3.4% in Frobenius norm during qualification.
        self._ns = newton_schulz
        self.schedule = ns_schedule(config)
        self.nesterov = config.get("muon_nesterov", False)
        # "two_tap": the momentum averages (g_t + g_{t-1}) / 2, which removes a period-2 (edge-of-stability)
        # oscillation exactly with half a step of lag (second-order audit, 2026-09-28).
        self.prefilter = config.get("muon_prefilter", "none")
        # PD's input whitening for the unembedding (second-order audit, 2026-09-28): Adam in coordinates where the
        # head's input is whitened, dW = -Adam(G R) R with R = (C_h / mean eig + damping)^-alpha (C_h from the head's
        # StatLinear), first moment in weight space, second moment of G R, the update norm-matched to the whitened-
        # coordinate Adam step so only its direction changes; decoupled decay as AdamW. Rank 0 computes the step and
        # broadcasts the head and both moments (its input statistics are rank-local, like the body's owner-local ones).
        self.head_white = None
        if config.get("head_whitening_alpha", 0.0) > 0:
            from .model import StatLinear
            head = getattr(model, "head", None)
            if not isinstance(head, StatLinear) or not head.cov:
                raise ValueError("head_whitening_alpha needs the model's track_head_cov")
            self.head_white = {"alpha": config["head_whitening_alpha"], "module": head, "param": head.weight,
                               "root": None, "damping": config.get("data_norm_damping", 1e-3),
                               "refresh": config.get("data_norm_refresh", 10), "betas": tuple(config["betas"]),
                               "eps": config["epsilon"],
                               # centered: whiten the covariance about the EMA mean, so the input's mean direction
                               # (the output-bias channel) is not singled out and suppressed (2026-09-28 12:25 CDT)
                               "center": config.get("head_whitening_center", False),
                               # "none": pure Adam in the whitened coordinates, dW = -lr Adam(G R) R (the reparametrized
                               # Adam); "match" rescales |dW| to the whitened-coordinate step's norm, which shrinks the
                               # function-space step (the 12:25 CDT arms)
                               "match": config.get("head_whitening_norm", "match") == "match"}
        # Measurement only: Q <- beta Q + beta/(1-beta) dW per body matrix, so that M + G Q transports the momentum
        # to the current weights in the Gauss-Newton model (second-order audit). Does not change the trajectory.
        self.track_displacement = config.get("muon_track_displacement", False)
        # SOAP statistics live outside optimizer state: each owner keeps its own matrices'
        # statistics, so replica audits and rank-0 checkpoints cover parameters and momentum only.
        self.soap = None
        if config.get("soap_precondition", False):
            self.soap = {"beta2": config["soap_beta2"], "denom_power": config["soap_denom_power"],
                         "basis": config.get("soap_basis", "both"),
                         "gradient_moment": config.get("soap_second_moment", "update") == "gradient",
                         "column": config.get("soap_norm", "entry") == "column",
                         "layers": config.get("soap_layers", "all"),
                         "states": {}}
            if self.soap["basis"] == "right_act":
                from .model import StatLinear
                self.soap["modules"] = {m.weight: m for m in model.modules() if isinstance(m, StatLinear) and m.cov}
                if not all(any(p is w for w in self.soap["modules"]) for p in body):
                    raise ValueError("soap_basis right_act needs track_input_cov on every Muon matrix")
        self.whitening = None
        self.whitening_sums = None
        if config.get("mean_whitening", False):
            from .model import StatLinear
            modules = {m.weight: m for m in model.modules() if isinstance(m, StatLinear)}
            if not all(any(p is w for w in modules) for p in body):
                raise ValueError("mean_whitening needs track_input_stats on every Muon matrix")
            beta = config["mean_whitening_beta"]
            self.whitening = {"modules": modules, "beta": None if beta < 0 else float(beta),
                              "power": config["mean_whitening_power"]}
        # Bias-split Muon: with beta = 0 the Muon direction has no component along the mean
        # input v, and the v-column of W, which acts as the implicit bias b = W xbar, takes an
        # Adam step on G v at the auxiliary rate instead. Adam state is owner-local, like SOAP's.
        self.bias_adam = None
        self.bias_sums = None
        if config.get("mean_bias_adam", False):
            self.bias_adam = {"betas": tuple(config["betas"]), "eps": config["epsilon"],
                              "ratio": config["aux_learning_rate"] / config["learning_rate"], "states": {}}
        # Partial data-norm Muon: polar(M R) R with R = C^(-alpha) from the layer's input
        # second moment, refreshed every `refresh` steps (owner-local cache), and the update
        # rescaled to Muon's Frobenius norm so only its shape changes.
        self.data_norm = None
        self.data_norm_sums = None
        if config.get("data_norm_alpha", 0.0) > 0 or config.get("data_norm_out_beta", 0.0) > 0:
            from .model import StatLinear
            modules = {m.weight: m for m in model.modules() if isinstance(m, StatLinear) and m.cov}
            if not all(any(p is w for w in modules) for p in body):
                raise ValueError("data_norm needs track_input_cov on every Muon matrix")
            kinds = {}
            for name, parameter in model.named_parameters():
                if name.startswith("blocks.") and parameter.ndim == 2:
                    kinds[parameter] = name.split(".")[-2]          # q, k, v, o, up or down
            self.data_norm = {"modules": modules, "alpha": config["data_norm_alpha"],
                              "damping": config["data_norm_damping"], "refresh": config["data_norm_refresh"],
                              "post": config.get("data_norm_post", True),
                              "rows": config.get("data_norm_rows", False), "row_sq": {}, "roots": {},
                              "center": config.get("data_norm_center", False),
                              "by_kind": {}, "pre": config.get("data_norm_pre", True),
                              "matched": config.get("data_norm_mode", "sandwich") == "magnitude_matched",
                              # Geometry decay: W <- W - lr*wd * W R^p / mean eig(R^p), i.e. the decay is
                              # preconditioned like the update (decoupled decay under PD penalizes high-variance
                              # input directions more). p = 2 equalizes the random-walk equilibrium norm per input
                              # direction; p = 1 makes the aligned-update equilibrium bound ||W||_op, as for Muon.
                              "geometry_decay": config.get("data_norm_decay", "decoupled") == "geometry",
                              "decay_power": config.get("data_norm_decay_power", 2),
                              # Two-sided data norm: output-side root L from the EMA of E[e e^T] (owner-local use,
                              # not checkpointed), set by update_output_statistics from the training loop.
                              "out_beta": config.get("data_norm_out_beta", 0.0), "out_stats": {}, "out_roots": {},
                              # Placebo control: B's eigenvalues in a fixed random orthogonal basis per matrix
                              # (seeded by the parameter's name), so only the spectrum's alignment is removed.
                              "out_placebo": config.get("data_norm_out_placebo", False), "out_seeds": {}}
            if self.data_norm["out_placebo"]:
                import zlib
                for name, parameter in model.named_parameters():
                    if name.startswith("blocks.") and parameter.ndim == 2:
                        self.data_norm["out_seeds"][parameter] = zlib.crc32(name.encode()) + 7919 * int(config["seed"])
            by_kind = config.get("data_norm_alpha_by_kind", {})
            self.data_norm["by_kind"] = {p: float(by_kind.get(kind, config["data_norm_alpha"])) for p, kind in kinds.items()}
        self.deflation = None
        self.deflation_mode = config.get("deflation_mode", "paper")
        if config.get("deflation", False) and self.deflation_mode == "paper":
            self.deflation = {"window": config["deflation_window"], "oversample": config["deflation_oversample"],
                              "subspace_iters": config["deflation_subspace_iters"],
                              "threshold": config["deflation_threshold"], "pad": config["deflation_pad"]}
        elif config.get("deflation", False):
            self.deflation = {"power_iters": config["deflation_power_iters"],
                              "head_weight": config["deflation_head_weight"], "pad": config["deflation_pad"]}
        # Owner-local [matrices, gate firings, deflated pairs] and summed head energy;
        # the runner reduces and resets them.
        self.deflation_counts = None
        self.deflation_energy = None
        self.deflation_autocos = None
        # Newton-Muon: every body gradient is right-multiplied by a damped inverse of a slow EMA of the
        # layer's activation second moment before it enters the momentum. The statistics are averaged
        # over ranks and the inverses shared from rank 0, so the replicated momentum stays identical.
        # MLP down uses `newton_muon_down_blocks` diagonal blocks, as in the reference.
        self.newton = None
        if config.get("newton_muon", False):
            from .model import StatLinear
            modules = {m.weight: m for m in model.modules() if isinstance(m, StatLinear) and m.cov}
            if not all(any(p is w for w in modules) for p in body):
                raise ValueError("newton_muon needs track_input_cov on every Muon matrix")
            blocks = {parameter: config["newton_muon_down_blocks"] if name.split(".")[-2] == "down" else 1
                      for name, parameter in model.named_parameters()
                      if name.startswith("blocks.") and parameter.ndim == 2}
            self.newton = {"modules": modules, "blocks": blocks, "damping": config["newton_muon_damping"],
                           "refresh": config["newton_muon_refresh"], "ema": config["newton_muon_ema"],
                           "cov": {}, "inverse": {}, "ready": False}
        # PMuon: polar(A^-gamma U C^-gamma) with streaming second moments of the NS input on both sides.
        # Owner-local and not checkpointed, like the SOAP statistics.
        self.pmuon = None
        if config.get("pmuon", False):
            self.pmuon = {"gamma": config["pmuon_gamma"], "beta": config["pmuon_beta"], "states": {}}
        self.capture_parameters = set()
        self.audit_distribution = False
        self.last_updates = {}

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self._adam.param_groups = [self.param_groups[1]]
        self._adam.state = self.state
        self.last_updates = {}

    def _row_normalize(self, parameter, whitened, beta2=0.95):
        """Divide each output row of the input-whitened momentum by its running RMS (owner-local EMA,
        bias-corrected): a diagonal output-side normalization before the polar."""
        state = self.data_norm["row_sq"].get(parameter)
        if state is None:
            state = self.data_norm["row_sq"][parameter] = {"sq": torch.zeros(whitened.shape[0], device=whitened.device), "t": 0}
        state["t"] += 1
        state["sq"].lerp_(whitened.square().mean(dim=1), 1 - beta2)
        rms = (state["sq"] / (1 - beta2 ** state["t"])).sqrt()
        return whitened / rms.clamp_min(1e-30)[:, None]

    def _data_norm_root(self, parameter):
        """Owner-local cached C^(-alpha), recomputed every `refresh` optimizer steps."""
        step = int(self.state[parameter]["step"])
        cached = self.data_norm["roots"].get(parameter)
        if cached is None or step - cached[1] >= self.data_norm["refresh"]:
            module = self.data_norm["modules"][parameter]
            root, cov = data_norm_root(module, self.data_norm["by_kind"].get(parameter, self.data_norm["alpha"]),
                                       self.data_norm["damping"], self.data_norm["center"], with_cov=True)
            cached = (root, step, cov)
            self.data_norm["roots"][parameter] = cached
        return cached[0]

    def update_output_statistics(self, stats, ema):
        """EMA (decay `ema` per call) of each body matrix's output-error second moment E[e e^T]."""
        for parameter, moment in stats.items():
            entry = self.data_norm["out_stats"].get(parameter)
            if entry is None:
                self.data_norm["out_stats"][parameter] = moment.detach().float().clone()
            else:
                entry.mul_(ema).add_(moment.detach().float(), alpha=1 - ema)
            self.data_norm["out_roots"].pop(parameter, None)

    def _data_norm_out_root(self, parameter):
        """(B / mean eig + damping I)^-beta for the output side, or None before the first statistics."""
        moment = self.data_norm["out_stats"].get(parameter)
        if self.data_norm["out_beta"] <= 0 or moment is None:
            return None
        root = self.data_norm["out_roots"].get(parameter)
        if root is None:
            values, vectors = torch.linalg.eigh(0.5 * (moment + moment.T).double())
            if self.data_norm["out_placebo"]:
                generator = torch.Generator().manual_seed(self.data_norm["out_seeds"][parameter])
                vectors, _ = torch.linalg.qr(torch.randn(vectors.shape, generator=generator, dtype=torch.float64))
                vectors = vectors.to(values.device)
            unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
            root = ((vectors * (unit + self.data_norm["damping"]).pow(-self.data_norm["out_beta"])) @ vectors.T).float()
            self.data_norm["out_roots"][parameter] = root
        return root

    def _data_norm_cov(self, parameter):
        """Unit-mean-eigenvalue input second moment cached with the root (for magnitude matching)."""
        return self.data_norm["roots"][parameter][2]

    def _soap_applies(self, parameter):
        wide = parameter.shape[0] < parameter.shape[1]      # MLP down is the only wide body matrix
        return {"all": True, "down": wide, "not_down": not wide}[self.soap["layers"]]

    def _bias_directions(self, chunk, v):
        """Adam on the implicit bias b = W xbar of each matrix in the chunk.

        Returns ratio * adam(G v) v^T / ||xbar||: applied at the Muon rate lr, it moves b by
        aux_lr * adam(G v), as an explicit bias trained by the auxiliary AdamW would move.
        """
        beta1, beta2 = self.bias_adam["betas"]
        out = []
        for (parameter, _), vector in zip(chunk, v.unbind()):
            module = self.whitening["modules"][parameter]
            xbar_norm = (module.input_mean / module.input_weight.clamp_min(1e-12)).norm().clamp_min(1e-12)
            grad = parameter.grad.float()
            g = grad @ vector
            state = self.bias_adam["states"].get(parameter)
            if state is None:
                state = self.bias_adam["states"][parameter] = {
                    "m": torch.zeros_like(g), "s": torch.zeros_like(g), "t": 0}
            state["t"] += 1
            state["m"].lerp_(g, 1 - beta1)
            state["s"].lerp_(g.square(), 1 - beta2)
            direction = (state["m"] / (1 - beta1 ** state["t"])) / (
                (state["s"] / (1 - beta2 ** state["t"])).sqrt() + self.bias_adam["eps"])
            out.append(torch.outer(direction, vector).mul_(self.bias_adam["ratio"] / xbar_norm))
            self.bias_sums += torch.stack([torch.ones((), dtype=torch.float64, device=g.device),
                                           (g.square().sum() / grad.square().sum().clamp_min(1e-30)).double(),
                                           direction.abs().mean().double()])
        return torch.stack(out)

    def _newton_refresh(self, params):
        """Fold the rank-averaged activation second moment into Newton-Muon's EMA (weight `ema`, from
        0.001 I) and recompute the damped inverses; rank 0's inverses are broadcast to all ranks."""
        newton = self.newton
        stats = []
        for parameter in params:
            module = newton["modules"][parameter]
            second = module.input_cov / module.input_cov_weight.clamp_min(1e-12)
            blocks = newton["blocks"][parameter]
            stats.append(diagonal_blocks(second, blocks) if blocks > 1 else second[None])
        flat = torch.cat([s.reshape(-1) for s in stats])
        distributed = self.distributed_optimizer and dist.is_initialized() and dist.get_world_size() > 1
        if distributed:
            dist.all_reduce(flat, op=dist.ReduceOp.SUM)
            flat /= dist.get_world_size()
        offset = 0
        for parameter, stat in zip(params, stats):
            average = flat[offset:offset + stat.numel()].view_as(stat)
            offset += stat.numel()
            cov = newton["cov"].get(parameter)
            if cov is None:
                eye = torch.eye(stat.size(-1), device=stat.device, dtype=stat.dtype)
                cov = newton["cov"][parameter] = (0.001 * eye).expand_as(stat).clone()
            cov.lerp_(average, newton["ema"])
            newton["inverse"][parameter] = newton_inverse(cov, newton["damping"])
        if distributed:
            shared = torch.cat([newton["inverse"][p].reshape(-1) for p in params])
            dist.broadcast(shared, src=0)
            offset = 0
            for parameter in params:
                inverse = newton["inverse"][parameter]
                inverse.copy_(shared[offset:offset + inverse.numel()].view_as(inverse))
                offset += inverse.numel()
        newton["ready"] = True

    def _newton_precondition(self, group):
        """Refresh when (completed updates + 1) is a multiple of `refresh` (the reference's schedule);
        from the first refresh on, replace each body gradient by G K (per block for MLP down)."""
        params = [p for p in group["params"] if p.grad is not None]
        first = self.state[params[0]]
        done = int(first["step"]) if first else 0
        if (done + 1) % self.newton["refresh"] == 0:
            self._newton_refresh(params)
        if not self.newton["ready"]:
            return
        for parameter in params:
            inverse = self.newton["inverse"][parameter]
            grad = parameter.grad.float()
            if inverse.size(0) == 1:
                parameter.grad.copy_(grad @ inverse[0])
            else:
                split = grad.view(grad.size(0), inverse.size(0), inverse.size(-1)).transpose(0, 1)
                parameter.grad.copy_(torch.bmm(split, inverse).transpose(0, 1).reshape_as(grad))

    def _pmuon_direction(self, parameter, update):
        """PMuon's NS input A^-gamma U C^-gamma (record #18, `pmuon_update`). The statistics are taken
        from the NS input itself, as in the reference code, in its EMA-momentum scale (1 - mu) M."""
        state = self.pmuon["states"].get(parameter)
        if state is None:
            index = next(i for i, p in enumerate(self.param_groups[0]["params"]) if p is parameter)
            rows, cols = parameter.shape
            state = self.pmuon["states"][parameter] = {
                "right": torch.zeros(cols, cols, device=parameter.device),
                "left": torch.zeros(rows, rows, device=parameter.device),
                "generator": torch.Generator(device=parameter.device).manual_seed(104729 * (index + 1))}
        beta, gamma, eps = self.pmuon["beta"], self.pmuon["gamma"], 1e-6
        u = update.float() * (1 - self.momentum)
        right = u.T @ u
        state["right"].mul_(beta).add_(right)
        right = right + beta * state["right"]
        right.diagonal().add_(eps)
        left = u @ u.T
        state["left"].mul_(beta).add_(left)
        left = left + beta * state["left"]
        left.diagonal().add_(eps)
        right_power = streaming_cov_power(right, state, "q_right", gamma, state["generator"])
        left_power = streaming_cov_power(left, state, "q_left", gamma, state["generator"])
        return left_power @ u @ right_power

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("This study supplies gradients explicitly")
        self.last_updates = {}
        group = self.param_groups[0]
        world = dist.get_world_size() if self.distributed_optimizer and dist.is_initialized() else 1
        rank = dist.get_rank() if self.distributed_optimizer and dist.is_initialized() else 0
        banks = defaultdict(list)
        offset = 0
        if self.newton is not None:
            self._newton_precondition(group)
        for parameter in group["params"]:
            if parameter.grad is None:
                continue
            state = self.state[parameter]
            if not state:
                state["momentum_buffer"] = torch.zeros_like(parameter)
                state["step"] = torch.zeros((), dtype=torch.int64, device=parameter.device)
            if self.deflation is not None and self.deflation_mode == "track1" and "head_v" not in state:
                # Identical on every rank: seeded by the parameter's position in the group.
                index = next(i for i, p in enumerate(group["params"]) if p is parameter)
                generator = torch.Generator(device=parameter.device).manual_seed(7919 * (index + 1))
                vector = torch.randn(min(parameter.shape), device=parameter.device, generator=generator)
                state["head_v"] = vector / vector.norm()
                state["head_u"] = torch.zeros(max(parameter.shape), device=parameter.device)
            grad = parameter.grad
            if self.prefilter == "two_tap":
                previous = state.get("previous_grad")
                state["previous_grad"] = grad.detach().clone()
                if previous is not None:
                    grad = 0.5 * (grad + previous)
            state["momentum_buffer"].mul_(self.momentum).add_(grad)
            state["step"].add_(1)
            banks[tuple(parameter.shape)].append((parameter, offset))
            offset += parameter.numel()
        # Exactly one rank computes each matrix. Independently repeated BF16 NS
        # can differ across devices/workspace choices despite identical inputs.
        # Sum disjoint owner slices, then apply the same resulting tensor on all
        # ranks. This also avoids repeating the expensive NS matrix products.
        updates = torch.zeros(offset, device=group["params"][0].device, dtype=torch.float32)
        # Geometry decay's extra term W R^2 / mean eig(R^2) - W, computed by owners and shared like the updates.
        extra_decay = (torch.zeros_like(updates) if self.data_norm is not None and self.data_norm["geometry_decay"]
                       else None)
        owner_values = []
        tracked = self.deflation is not None and self.deflation_mode == "track1"
        if self.deflation is not None:
            if self.deflation_counts is None:
                self.deflation_counts = torch.zeros(3, dtype=torch.int64, device=updates.device)
                self.deflation_energy = torch.zeros((), dtype=torch.float64, device=updates.device)
            # Sketch seeds depend only on the step, bank and chunk, so resumes reproduce them.
            step_value = int(self.state[group["params"][0]]["step"])
        if tracked:
            # Owners write refreshed head vectors into disjoint slices, shared by one all-reduce.
            head_slices, head_length, left_length = {}, 0, 0
            for items in banks.values():
                for parameter, _ in items:
                    head_slices[parameter] = (head_length, left_length)
                    head_length += min(parameter.shape)
                    left_length += max(parameter.shape)
            heads = torch.zeros(head_length, device=updates.device, dtype=torch.float32)
            lefts = torch.zeros(left_length, device=updates.device, dtype=torch.float32)
            if self.deflation_autocos is None:
                self.deflation_autocos = torch.zeros(2, dtype=torch.float64, device=updates.device)
        if self.whitening is not None and self.whitening_sums is None:
            self.whitening_sums = torch.zeros(2, dtype=torch.float64, device=updates.device)
        if self.bias_adam is not None and self.bias_sums is None:
            self.bias_sums = torch.zeros(3, dtype=torch.float64, device=updates.device)
        if self.data_norm is not None and self.data_norm_sums is None:
            self.data_norm_sums = torch.zeros(3, dtype=torch.float64, device=updates.device)
        # Small shape batches keep NS efficient and bound temporary GPU memory.
        for bank, (shape, items) in enumerate(banks.items()):
            scale = math.sqrt(max(1.0, shape[0] / shape[1]))
            parameters = [item for index, item in enumerate(items) if index % world == rank]
            for offset in range(0, len(parameters), 8):
                chunk = parameters[offset:offset + 8]
                if self.nesterov:
                    inputs = torch.stack([p.grad + self.momentum * self.state[p]["momentum_buffer"]
                                          for p, _ in chunk])
                else:
                    inputs = torch.stack([self.state[p]["momentum_buffer"] for p, _ in chunk])
                roots = None
                out_roots = None
                if self.data_norm is not None:
                    roots = torch.stack([self._data_norm_root(p) for p, _ in chunk])
                    if self.data_norm["pre"]:
                        inputs = inputs.float() @ roots
                    if self.data_norm["out_beta"] > 0:
                        left_list = [self._data_norm_out_root(p) for p, _ in chunk]
                        if all(left is not None for left in left_list):
                            out_roots = torch.stack(left_list)
                            inputs = out_roots @ inputs.float()
                if self.soap is not None and self._soap_applies(chunk[0][0]):
                    # With data norm, SOAP runs in whitened coordinates: momentum M R, statistics from G R.
                    # With the output factor too (two-sided), both are whitened on both sides: L M R and L G R.
                    preconditioned = []
                    for index, ((parameter, _), matrix) in enumerate(zip(chunk, inputs.unbind())):
                        soap = self.soap["states"].get(parameter)
                        if soap is None:
                            soap = self.soap["states"][parameter] = new_soap_state(parameter, self.soap["basis"])
                        grad = parameter.grad.float()
                        if roots is not None:
                            grad = grad @ roots[index]
                        if out_roots is not None:
                            grad = out_roots[index] @ grad
                        preconditioned.append(soap_precondition(
                            matrix.float(), soap, self.soap["beta2"], self.soap["denom_power"],
                            grad=grad if self.soap["gradient_moment"] else None,
                            column=self.soap["column"]))
                        col_gram = None
                        if self.soap["basis"] == "right_act":
                            module = self.soap["modules"][parameter]
                            col_gram = module.input_cov / module.input_cov_weight.clamp_min(1e-12)
                        soap_update_statistics(grad, soap, self.soap["beta2"], col_gram)
                    inputs = torch.stack(preconditioned)
                if self.data_norm is not None:
                    whitened = inputs.float()
                    if self.data_norm["rows"]:
                        whitened = torch.stack([self._row_normalize(p, x) for (p, _), x in zip(chunk, whitened.unbind())])
                    directions = self._ns(whitened, self.schedule)
                    if self.data_norm["post"]:
                        directions = directions @ roots
                        if out_roots is not None:
                            directions = out_roots @ directions
                    norms = directions.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-30)
                    directions = directions.mul_(math.sqrt(min(shape)) / norms)
                    if self.data_norm["matched"]:
                        # Muon's direction at the data-norm update's per-layer output energy E||dW x||^2.
                        covs = torch.stack([self._data_norm_cov(p) for p, _ in chunk])
                        muon = self._ns(torch.stack([self.state[p]["momentum_buffer"] for p, _ in chunk])
                                        if not self.nesterov else
                                        torch.stack([p.grad + self.momentum * self.state[p]["momentum_buffer"]
                                                     for p, _ in chunk]), self.schedule)
                        energy = lambda d: ((d @ covs) * d).sum(dim=(-2, -1), keepdim=True)
                        directions = muon * (energy(directions) / energy(muon).clamp_min(1e-30)).sqrt()
                    directions = directions.mul_(scale)
                    means = torch.stack([self.data_norm["modules"][p].input_mean for p, _ in chunk])
                    v = means / means.norm(dim=-1, keepdim=True).clamp_min(1e-30)
                    along = ((v[:, None, :] @ roots @ v[..., None])[..., 0, 0] /
                             roots.diagonal(dim1=-2, dim2=-1).mean(-1))
                    self.data_norm_sums += torch.stack([
                        torch.tensor(float(len(chunk)), device=v.device, dtype=torch.float64),
                        along.double().sum(), (norms[..., 0, 0] / math.sqrt(min(shape))).double().sum()])
                    if extra_decay is not None:
                        shrink = roots @ roots if self.data_norm["decay_power"] == 2 else roots
                        shrink = shrink / shrink.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None]
                        weights = torch.stack([p.detach().float() for p, _ in chunk])
                        for (parameter, start), extra in zip(chunk, (weights @ shrink - weights).unbind()):
                            extra_decay[start:start + parameter.numel()].copy_(extra.flatten())
                elif self.whitening is not None:
                    factors = [whitening_factors(self.whitening["modules"][p], self.whitening["beta"],
                                                 self.whitening["power"]) for p, _ in chunk]
                    v = torch.stack([f[0] for f in factors])
                    beta = torch.stack([f[1] for f in factors])
                    shrink = (1 - beta)[:, None, None]
                    projected = inputs - shrink * (inputs @ v[..., None]) * v[:, None, :]
                    directions = self._ns(projected, self.schedule)
                    directions = (directions - shrink * (directions @ v[..., None]) * v[:, None, :]).mul_(scale)
                    if self.bias_adam is not None:
                        directions += self._bias_directions(chunk, v)
                    self.whitening_sums += torch.stack([torch.tensor(float(len(chunk)), device=beta.device,
                                                                     dtype=torch.float64), beta.double().sum()])
                elif self.pmuon is not None:
                    shaped = torch.stack([self._pmuon_direction(p, x) for (p, _), x in zip(chunk, inputs.unbind())])
                    directions = self._ns(shaped, self.schedule).mul_(scale)
                elif self.deflation is None:
                    directions = self._ns(inputs, self.schedule).mul_(scale)
                elif tracked:
                    previous = torch.stack([self.state[p]["head_v"] for p, _ in chunk])
                    previous_u = torch.stack([self.state[p]["head_u"] for p, _ in chunk])
                    entry, vectors, energy, left = tracked_entry(inputs.float(), previous, **self.deflation)
                    directions = _iterate(entry, self.schedule).mul_(scale)
                    for (parameter, _), vector, u in zip(chunk, vectors.unbind(), left.unbind()):
                        start, left_start = head_slices[parameter]
                        heads[start:start + vector.numel()].copy_(vector)
                        lefts[left_start:left_start + u.numel()].copy_(u)
                    self.deflation_counts += len(chunk)
                    self.deflation_energy += energy.double().sum()
                    self.deflation_autocos += torch.stack([(vectors * previous).sum(-1).double().sum(),
                                                           (left * previous_u).sum(-1).double().sum()])
                else:
                    generator = torch.Generator(device=inputs.device)
                    generator.manual_seed((step_value * 1009 + bank * 9176867 + offset * 7919) % 2**62)
                    directions, k = deflated_newton_schulz(inputs, self.schedule, generator, **self.deflation)
                    directions.mul_(scale)
                    self.deflation_counts += torch.stack([torch.tensor(len(chunk), device=k.device),
                                                          (k > 0).sum(), k.sum()])
                for (parameter, start), direction in zip(chunk, directions.unbind()):
                    updates[start:start + parameter.numel()].copy_(direction.flatten())
                    if self.audit_distribution:
                        owner_values.append((start, direction.flatten().clone()))
        if world > 1:
            dist.all_reduce(updates, op=dist.ReduceOp.SUM)
            if extra_decay is not None:
                dist.all_reduce(extra_decay, op=dist.ReduceOp.SUM)
            if tracked:
                dist.all_reduce(heads, op=dist.ReduceOp.SUM)
                dist.all_reduce(lefts, op=dist.ReduceOp.SUM)
        if tracked:
            for parameter, (start, left_start) in head_slices.items():
                state = self.state[parameter]
                state["head_v"].copy_(heads[start:start + state["head_v"].numel()])
                state["head_u"].copy_(lefts[left_start:left_start + state["head_u"].numel()])
        for start, expected in owner_values:
            if not torch.equal(updates[start:start + expected.numel()], expected):
                raise RuntimeError("Distributed Muon direction differs from its single owner")
        for items in banks.values():
            for parameter, start in items:
                direction = updates[start:start + parameter.numel()].view_as(parameter)
                if parameter in self.capture_parameters:
                    self.last_updates[parameter] = direction.clone()
                before = parameter.detach().clone() if self.track_displacement else None
                parameter.mul_(1 - group["lr"] * group["weight_decay"])
                if extra_decay is not None:  # net decay: -lr*wd * W R^p / mean eig(R^p)
                    parameter.add_(extra_decay[start:start + parameter.numel()].view_as(parameter),
                                   alpha=-group["lr"] * group["weight_decay"])
                parameter.add_(direction, alpha=-group["lr"])
                if before is not None:
                    state = self.state[parameter]
                    if "displacement_ema" not in state:
                        state["displacement_ema"] = torch.zeros_like(parameter, dtype=torch.float32)
                    beta = self.momentum
                    state["displacement_ema"].mul_(beta).add_((parameter.detach().float() - before.float()),
                                                              alpha=beta / (1 - beta))
        head_grad = None
        if self.head_white is not None and self.head_white["param"].grad is not None:
            head_grad = self.head_white["param"].grad
            self.head_white["param"].grad = None     # the shared AdamW skips the head
        self._adam.step()
        if head_grad is not None:
            self._whitened_head_step(head_grad)
            self.head_white["param"].grad = head_grad

    @torch.no_grad()
    def _whitened_head_step(self, grad):
        hw, p, group = self.head_white, self.head_white["param"], self.param_groups[1]
        state = self.state[p]
        if "hw_step" not in state:
            state["hw_step"] = torch.zeros((), dtype=torch.int64, device=p.device)
            state["hw_exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            state["hw_exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["hw_step"].add_(1)
        t = int(state["hw_step"])
        world = dist.get_world_size() if self.distributed_optimizer and dist.is_initialized() else 1
        rank = dist.get_rank() if self.distributed_optimizer and dist.is_initialized() else 0
        if rank == 0:
            if hw["root"] is None or (t - 1) % hw["refresh"] == 0:
                module = hw["module"]
                cov = (module.input_cov / module.input_cov_weight.clamp_min(1e-12)).double()
                if hw["center"]:
                    mean = (module.input_cov_mean / module.input_cov_weight.clamp_min(1e-12)).double()
                    cov = cov - torch.outer(mean, mean)
                values, vectors = torch.linalg.eigh(0.5 * (cov + cov.T))
                unit = values.clamp_min(0) / values.clamp_min(0).mean().clamp_min(1e-30)
                hw["root"] = ((vectors * (unit + hw["damping"]).pow(-hw["alpha"])) @ vectors.T).float()
            root = hw["root"]
            beta1, beta2 = hw["betas"]
            g = grad.float()
            state["hw_exp_avg"].mul_(beta1).add_(g, alpha=1 - beta1)
            white = g @ root
            state["hw_exp_avg_sq"].mul_(beta2).addcmul_(white, white, value=1 - beta2)
            step = ((state["hw_exp_avg"] / (1 - beta1 ** t)) @ root) / ((state["hw_exp_avg_sq"] / (1 - beta2 ** t)).sqrt() + hw["eps"])
            update = step @ root
            if hw["match"]:
                update.mul_(step.norm() / update.norm().clamp_min(1e-30))
            p.mul_(1 - group["lr"] * group["weight_decay"])
            p.add_(update.to(p.dtype), alpha=-group["lr"])
        if world > 1:   # one coalesced broadcast keeps weights and moments replicated (replica audits)
            shared = torch.cat([p.data.float().reshape(-1), state["hw_exp_avg"].reshape(-1), state["hw_exp_avg_sq"].reshape(-1)])
            dist.broadcast(shared, src=0)
            weights, first, second = shared.split(p.numel())
            p.data.copy_(weights.view_as(p))
            state["hw_exp_avg"].copy_(first.view_as(p))
            state["hw_exp_avg_sq"].copy_(second.view_as(p))


def output_second_moments(model, x, y, source, config, device, generator=None):
    """Per-token second moments E[e e^T] of the summed token loss's gradient at every body matrix's output.

    "ef": data labels (empirical Fisher); "gn": labels sampled from the model (Gauss-Newton). An eager, eval-mode
    pass (so input statistics do not update) whose gradients are taken with respect to the matrix outputs only,
    leaving parameter gradients untouched. Returns {weight parameter: (d_out, d_out) float32 tensor}."""
    import torch.nn.functional as F
    from .train import amp_context
    body = {module.weight: module for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and name.startswith("blocks.")}
    outputs = {}
    hooks = [module.register_forward_hook(lambda mod, args, out, w=weight: outputs.__setitem__(w, out))
             for weight, module in body.items()]
    was_training = model.training
    model.eval()
    try:
        with amp_context(device, config):
            logits = model(x)
        if source == "gn":
            with torch.no_grad():
                probs = torch.softmax(logits.float(), dim=-1).flatten(0, 1)
                target = torch.multinomial(probs, 1, generator=generator).view(y.shape)
        else:
            target = y
        loss = F.cross_entropy(logits.float().flatten(0, 1), target.flatten(), reduction="sum")
        weights = list(outputs)
        errors = torch.autograd.grad(loss, [outputs[w] for w in weights])
    finally:
        for hook in hooks:
            hook.remove()
        model.train(was_training)
    stats = {}
    for weight, error in zip(weights, errors):
        flat = error.float().reshape(-1, error.shape[-1])
        stats[weight] = flat.T @ flat / flat.shape[0]
    return stats

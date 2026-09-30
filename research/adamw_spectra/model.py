"""Explicit GPT-style reconstruction; see PROTOCOL.md for paper-match limits.

Defaults reproduce the original cohort exactly (Linear biases, LayerNorm). The
optional frontier-norm variant (MUON_CASE.md) drops biases, uses RMSNorm with
learnable gains and adds per-head RMSNorm with gains on queries and keys.
"""

import math
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 50304
    n_layer: int = 8
    n_embd: int = 512
    n_head: int = 8
    seq_len: int = 512
    bias: bool = True
    norm: str = "layernorm"
    qk_norm: bool = False
    track_input_stats: bool = False
    track_input_cov: bool = False
    track_head_cov: bool = False     # the unembedding keeps its input second moment (head whitening, 2026-09-28)

    def __post_init__(self):
        if min(self.vocab_size, self.n_layer, self.n_embd, self.n_head, self.seq_len) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("Width must be divisible by number of heads")
        if self.norm not in ("layernorm", "rmsnorm"):
            raise ValueError("norm must be layernorm or rmsnorm")
        if not all(isinstance(flag, bool) for flag in (self.bias, self.qk_norm, self.track_input_stats,
                                                        self.track_input_cov, self.track_head_cov)):
            raise ValueError("bias, qk_norm, track_input_stats, track_input_cov and track_head_cov must be boolean")
        if self.track_input_cov and not self.track_input_stats:
            raise ValueError("track_input_cov needs track_input_stats")


class RMSNorm(nn.Module):
    """RMS normalization over the last dimension with a learnable gain, no bias."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        y = F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps)
        return (y * self.weight.float()).type_as(x)


def make_norm(config):
    return RMSNorm(config.n_embd) if config.norm == "rmsnorm" else nn.LayerNorm(config.n_embd)


class StatLinear(nn.Linear):
    """nn.Linear that also keeps EMAs of its input's token mean and mean squared norm.

    Position 0 is excluded (attention-sink token). Statistics update only in training
    forwards with gradients enabled, never mix in evaluation data, and are
    non-persistent, so weights, checkpoints and initial-weight hashes are unchanged.
    `input_weight` is the EMA of ones, for bias correction.

    With `cov`, it also keeps an EMA of the uncentered input second-moment matrix
    E[x x^T] from every `cov_stride`-th position (from position 1), with its own decay
    per training forward and bias-correction weight `input_cov_weight`.
    """

    def __init__(self, in_features, out_features, bias=True, decay=0.99, cov=False,
                 cov_decay=0.998, cov_stride=32):
        super().__init__(in_features, out_features, bias=bias)
        self.decay = decay
        self.register_buffer("input_mean", torch.zeros(in_features), persistent=False)
        self.register_buffer("input_sq", torch.zeros(()), persistent=False)
        self.register_buffer("input_weight", torch.zeros(()), persistent=False)
        self.cov = cov
        if cov:
            self.cov_decay, self.cov_stride = cov_decay, cov_stride
            self.register_buffer("input_cov", torch.zeros(in_features, in_features), persistent=False)
            self.register_buffer("input_cov_mean", torch.zeros(in_features), persistent=False)
            self.register_buffer("input_cov_weight", torch.zeros(()), persistent=False)

    def forward(self, x):
        if self.training and torch.is_grad_enabled():
            with torch.no_grad():
                sample = x.detach()[:, 1:].float()
                self.input_mean.mul_(self.decay).add_(sample.mean(dim=(0, 1)), alpha=1 - self.decay)
                self.input_sq.mul_(self.decay).add_(sample.pow(2).sum(-1).mean(), alpha=1 - self.decay)
                self.input_weight.mul_(self.decay).add_(1 - self.decay)
                if self.cov:
                    rows = sample[:, ::self.cov_stride].reshape(-1, sample.shape[-1])
                    self.input_cov.mul_(self.cov_decay).add_(rows.T @ rows / rows.shape[0],
                                                             alpha=1 - self.cov_decay)
                    self.input_cov_mean.mul_(self.cov_decay).add_(rows.mean(0), alpha=1 - self.cov_decay)
                    self.input_cov_weight.mul_(self.cov_decay).add_(1 - self.cov_decay)
        return super().forward(x)


def body_linear(config):
    if not config.track_input_stats:
        return nn.Linear
    return lambda i, o, bias=True: StatLinear(i, o, bias=bias, cov=config.track_input_cov)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        linear = body_linear(config)
        self.q = linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k = linear(config.n_embd, config.n_embd, bias=config.bias)
        self.v = linear(config.n_embd, config.n_embd, bias=config.bias)
        self.o = linear(config.n_embd, config.n_embd, bias=config.bias)
        head_dim = config.n_embd // config.n_head
        self.q_norm = RMSNorm(head_dim) if config.qk_norm else None
        self.k_norm = RMSNorm(head_dim) if config.qk_norm else None

    def heads(self, x):
        """Queries, keys and values as (batch, head, time, head_dim)."""
        b, t, d = x.shape
        def split(layer, norm=None):
            y = layer(x).view(b, t, self.n_head, d // self.n_head)
            return (y if norm is None else norm(y)).transpose(1, 2)
        return split(self.q, self.q_norm), split(self.k, self.k_norm), split(self.v)

    def forward(self, x):
        b, t, d = x.shape
        y = F.scaled_dot_product_attention(*self.heads(x), dropout_p=0.0, is_causal=True)
        return self.o(y.transpose(1, 2).contiguous().view(b, t, d))


class MLP(nn.Module):
    def __init__(self, dim, bias=True, linear=nn.Linear):
        super().__init__()
        self.up = linear(dim, 4 * dim, bias=bias)
        self.down = linear(4 * dim, dim, bias=bias)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = make_norm(config)
        self.ln2 = make_norm(config)
        self.attn = Attention(config)
        self.mlp = MLP(config.n_embd, config.bias, body_linear(config))

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.position = nn.Embedding(config.seq_len, config.n_embd)
        self.register_buffer("_positions", torch.arange(config.seq_len), persistent=False)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = make_norm(config)
        # With track_head_cov the head is a StatLinear (same parameters and initialization, extra non-persistent
        # input statistics), so weights, checkpoints and the initial-weight hash are unchanged.
        self.head = (StatLinear(config.n_embd, config.vocab_size, bias=False, cov=True) if config.track_head_cov
                     else nn.Linear(config.n_embd, config.vocab_size, bias=False))
        self.apply(self._initialize)
        for block in self.blocks:
            for layer in (block.attn.o, block.mlp.down):
                nn.init.normal_(layer.weight, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens, targets=None):
        if tokens.shape[1] > self.config.seq_len:
            raise ValueError("Sequence exceeds configured context")
        x = self.embed(tokens) + self.position(self._positions[:tokens.shape[1]])
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.norm(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())

    def measured_parameters(self):
        """Four relative depths, one-based; deduplicated for tiny smoke models."""
        depths = sorted({max(1, (self.config.n_layer * k) // 4) for k in (1, 2, 3, 4)})
        result = {}
        for depth in depths:
            block = self.blocks[depth - 1]
            for kind, layer in (("q", block.attn.q), ("k", block.attn.k),
                                ("v", block.attn.v), ("o", block.attn.o),
                                ("up", block.mlp.up), ("down", block.mlp.down)):
                result[f"block{depth:02d}.{kind}"] = layer.weight
        return result

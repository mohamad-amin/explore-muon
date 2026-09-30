"""Explicit GPT-style reconstruction; see PROTOCOL.md for paper-match limits."""

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

    def __post_init__(self):
        if min(self.vocab_size, self.n_layer, self.n_embd, self.n_head, self.seq_len) < 1:
            raise ValueError("Model dimensions must be positive")
        if self.n_embd % self.n_head:
            raise ValueError("Width must be divisible by number of heads")


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.q = nn.Linear(config.n_embd, config.n_embd)
        self.k = nn.Linear(config.n_embd, config.n_embd)
        self.v = nn.Linear(config.n_embd, config.n_embd)
        self.o = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x):
        b, t, d = x.shape
        def split(layer):
            return layer(x).view(b, t, self.n_head, d // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(split(self.q), split(self.k), split(self.v),
                                         dropout_p=0.0, is_causal=True)
        return self.o(y.transpose(1, 2).contiguous().view(b, t, d))


class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.Linear(dim, 4 * dim)
        self.down = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return self.down(F.gelu(self.up(x), approximate="tanh"))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = Attention(config)
        self.mlp = MLP(config.n_embd)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class GPT(nn.Module):
    def __init__(self, config=ModelConfig()):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.n_embd)
        self.position = nn.Embedding(config.seq_len, config.n_embd)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
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
        x = self.embed(tokens) + self.position(torch.arange(tokens.shape[1], device=tokens.device))
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

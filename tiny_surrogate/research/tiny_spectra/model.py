"""Tiny GPT architecture, sharing only the qualified pure model implementation.

This imports no production training or distributed pipeline. Reusing the model
keeps body-matrix identities, initialization and StatLinear's optimizer contract
identical while reducing vocabulary, width, depth and context.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from research.adamw_spectra.model import GPT as _GPT
from research.adamw_spectra.model import ModelConfig as _ModelConfig
from research.adamw_spectra.model import StatLinear


class StepStatLinear(StatLinear):
    """Statistics collected only inside an explicit training-step window.

Call ``begin_step()`` before training microbatches and ``finish_step()`` after
their backwards. Covariance estimates weight sampled rows equally, including
unequal microbatches in the default pooled mode. The explicit ``microforward``
mode instead commits the EMA after each forward, including a partial last
microbatch, as in the reference collector. Both modes use FP32 Gram products.
Validation and forwards outside collection never update statistics. Inherited
buffers retain the production optimizer API.
    """

    def __init__(self, in_features, out_features, bias=False, *, decay=0.99,
                 cov=True, cov_decay=0.9, cov_stride=4, stats_clock="step"):
        if stats_clock not in ("step", "microforward"):
            raise ValueError("Statistics clock must be step or microforward")
        super().__init__(in_features, out_features, bias=bias, decay=decay,
                         cov=cov, cov_decay=cov_decay, cov_stride=cov_stride)
        self.stats_clock = stats_clock
        self.collect_stats = False
        self.register_buffer("_step_mean_sum", torch.zeros(in_features), persistent=False)
        self.register_buffer("_step_sq_sum", torch.zeros(()), persistent=False)
        self.register_buffer("_step_count", torch.zeros((), dtype=torch.long), persistent=False)
        if cov:
            self.register_buffer("_step_cov_sum", torch.zeros(in_features, in_features), persistent=False)
            self.register_buffer("_step_cov_mean_sum", torch.zeros(in_features), persistent=False)
            self.register_buffer("_step_cov_count", torch.zeros((), dtype=torch.long), persistent=False)
        if stats_clock == "microforward":
            self.register_buffer("_step_forwards", torch.zeros((), dtype=torch.long), persistent=False)
            self.register_buffer("_total_forwards", torch.zeros((), dtype=torch.long), persistent=False)

    @torch.no_grad()
    def begin_step(self):
        if self.collect_stats:
            raise RuntimeError("finish_step must precede the next begin_step")
        self._step_mean_sum.zero_()
        self._step_sq_sum.zero_()
        self._step_count.zero_()
        if self.stats_clock == "microforward":
            self._step_forwards.zero_()
        if self.cov:
            self._step_cov_sum.zero_()
            self._step_cov_mean_sum.zero_()
            self._step_cov_count.zero_()
        self.collect_stats = True

    @torch.no_grad()
    def finish_step(self, ema=None):
        if not self.collect_stats:
            raise RuntimeError("begin_step must precede finish_step")
        if ema is not None and not 0 <= ema < 1:
            raise ValueError("EMA must lie in [0, 1)")
        if self._step_count.item() == 0:
            raise RuntimeError("No training samples collected for this step")
        if self.stats_clock == "microforward":
            if ema is not None:
                raise ValueError("Per-forward EMAs cannot be overridden at step end")
            self.collect_stats = False
            return
        decay = self.decay if ema is None else ema
        self.input_mean.mul_(decay).add_(self._step_mean_sum / self._step_count, alpha=1-decay)
        self.input_sq.mul_(decay).add_(self._step_sq_sum / self._step_count, alpha=1-decay)
        self.input_weight.mul_(decay).add_(1-decay)
        if self.cov:
            decay = self.cov_decay if ema is None else ema
            self.input_cov.mul_(decay).add_(self._step_cov_sum / self._step_cov_count, alpha=1-decay)
            self.input_cov_mean.mul_(decay).add_(self._step_cov_mean_sum / self._step_cov_count,
                                                alpha=1-decay)
            self.input_cov_weight.mul_(decay).add_(1-decay)
        self.collect_stats = False

    def forward(self, x):
        if self.collect_stats and self.training and torch.is_grad_enabled():
            # Ambient model autocast must not round the covariance GEMM back to
            # BF16 after x.float(); eigensolvers consume an FP32 second moment.
            with torch.no_grad(), torch.autocast(device_type=x.device.type, enabled=False):
                sample = x.detach()[:, 1:].float()
                if sample.numel() == 0:
                    raise ValueError("Statistics exclude position zero and need at least two positions")
                self._step_count.add_(sample.shape[0] * sample.shape[1])
                if self.stats_clock == "step":
                    self._step_mean_sum.add_(sample.sum(dim=(0, 1)))
                    self._step_sq_sum.add_(sample.square().sum())
                else:
                    self.input_mean.mul_(self.decay).add_(sample.mean(dim=(0, 1)), alpha=1-self.decay)
                    self.input_sq.mul_(self.decay).add_(sample.square().sum(-1).mean(), alpha=1-self.decay)
                    self.input_weight.mul_(self.decay).add_(1-self.decay)
                    self._step_forwards.add_(1)
                    self._total_forwards.add_(1)
                if self.cov:
                    rows = sample[:, ::self.cov_stride].reshape(-1, sample.shape[-1])
                    self._step_cov_count.add_(rows.shape[0])
                    if self.stats_clock == "step":
                        self._step_cov_sum.add_(rows.T @ rows)
                        self._step_cov_mean_sum.add_(rows.sum(dim=0))
                    else:
                        self.input_cov.mul_(self.cov_decay).add_(rows.T @ rows / rows.shape[0],
                                                               alpha=1-self.cov_decay)
                        self.input_cov_mean.mul_(self.cov_decay).add_(rows.mean(0), alpha=1-self.cov_decay)
                        self.input_cov_weight.mul_(self.cov_decay).add_(1-self.cov_decay)
        return F.linear(x, self.weight, self.bias)


@dataclass(frozen=True)
class ModelConfig(_ModelConfig):
    vocab_size: int = 65
    n_layer: int = 4
    n_embd: int = 128
    n_head: int = 4
    seq_len: int = 128
    bias: bool = False
    norm: str = "rmsnorm"
    qk_norm: bool = True
    track_input_stats: bool = False
    track_input_cov: bool = False
    stats_decay: float = 0.99
    cov_decay: float = 0.9
    cov_stride: int = 4
    stats_clock: str = "step"

    def __post_init__(self):
        super().__post_init__()
        if not 0 <= self.stats_decay < 1 or not 0 <= self.cov_decay < 1:
            raise ValueError("Statistic EMA decay must lie in [0, 1)")
        if not isinstance(self.cov_stride, int) or self.cov_stride < 1:
            raise ValueError("cov_stride must be a positive integer")
        if self.stats_clock not in ("step", "microforward"):
            raise ValueError("Statistics clock must be step or microforward")
        if (self.track_input_stats or self.track_head_cov) and self.seq_len < 2:
            raise ValueError("Input statistics exclude position zero and need context >= 2")


class GPT(_GPT):
    """Bias-free RMSNorm/QK-normalized character GPT by default.

``model(tokens)`` returns logits; ``model(tokens, targets)`` returns mean
next-token cross entropy. ``return_logits=True`` returns ``(logits, loss)``
when targets are supplied. Embedding and output head are untied.
    """

    def __init__(self, config=ModelConfig()):
        super().__init__(config)
        replacements = []
        for name, module in self.named_modules():
            if isinstance(module, StatLinear):
                # Keep the initialized parameter objects and the caller's RNG
                # exactly unchanged when replacing only the statistics collector.
                with torch.random.fork_rng(devices=[]):
                    replacement = StepStatLinear(module.in_features, module.out_features,
                                                 bias=module.bias is not None,
                                                 decay=config.stats_decay, cov=module.cov,
                                                 cov_decay=config.cov_decay,
                                                 cov_stride=config.cov_stride,
                                                 stats_clock=config.stats_clock)
                replacement.weight, replacement.bias = module.weight, module.bias
                replacements.append((name, replacement))
        for name, replacement in replacements:
            parent_name, _, child_name = name.rpartition(".")
            setattr(self.get_submodule(parent_name), child_name, replacement)

    def forward(self, tokens, targets=None, *, return_logits=False):
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("tokens must have nonempty [batch, time] dimensions")
        if targets is not None and targets.shape != tokens.shape:
            raise ValueError("targets must have the same shape as tokens")
        if (self.training and torch.is_grad_enabled() and tokens.shape[1] < 2
                and (self.config.track_input_stats or self.config.track_head_cov)):
            raise ValueError("Statistics need at least two token positions")
        logits = super().forward(tokens)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().flatten(0, 1), targets.flatten())
        return (logits, loss) if return_logits else loss

    def body_modules(self):
        """Map production-compatible parameter names to Q/K/V/O/up/down modules."""
        return {
            f"blocks.{i}.{section}.{name}.weight": getattr(getattr(block, section), name)
            for i, block in enumerate(self.blocks)
            for section, names in (("attn", ("q", "k", "v", "o")),
                                   ("mlp", ("up", "down")))
            for name in names
        }

    def body_parameters(self):
        return {name: module.weight for name, module in self.body_modules().items()}

    def num_parameters(self):
        return sum(parameter.numel() for parameter in self.parameters())

    def begin_step_stats(self):
        for module in self.modules():
            if isinstance(module, StepStatLinear):
                module.begin_step()

    def finish_step_stats(self, ema=None):
        for module in self.modules():
            if isinstance(module, StepStatLinear):
                module.finish_step(ema)


TinyGPT = GPT
TinyModelConfig = ModelConfig

"""One CUDA graph for the existing four-sequence training microforward/backward.

This module changes execution only. It does not capture an optimizer step,
gradient clearing/clipping, GN statistics, step-statistic begin/end, or evaluation.
No compiler, alternate attention implementation, precision fallback, or disk
cache is introduced. Use the study's ./run wrapper as for eager training.

Construct ``TrainingGraph(model, example_x, example_y, precision)`` once, then
call ``backward(x, y, weight)`` inside the ordinary collection window. It returns
an owned detached *unweighted* loss; the caller retains its normal loss logging.
Inputs of one to three sequences use the same eager microforward/backward.

After construction, clear gradients with ``zero_grad(set_to_none=False)``.
Originally absent gradients remain None immediately after construction, then
the first backward attaches the retained zero buffers used during capture.
Subsequent replacement of a gradient/parameter/buffer allocation is an error.
The same model must not be used concurrently on another thread or CUDA stream.

Capture follows PyTorch 2.11 CUDA graph lifetime and side-stream warmup rules:
https://docs.pytorch.org/docs/2.11/notes/cuda.html#cuda-graphs
Unlike the documented whole-step example, gradients exist *before* capture so
AccumulateGrad captures addition into them, rather than overwriting each replay.
"""
from contextlib import nullcontext
import math
import random
import time

import numpy as np
import torch

from .model import GPT, StepStatLinear


def _tensor_signature(tensor):
    return (tensor.data_ptr(), tensor.shape, tensor.stride(), tensor.dtype, tensor.device, tensor.requires_grad)


def _stat_signature(module):
    return (module.stats_clock, module.decay, module.cov, getattr(module, "cov_decay", None),
            getattr(module, "cov_stride", None), module.in_features, module.out_features)


class TrainingGraph:
    """Fixed-shape CUDA replay, with explicit state restoration and guards."""

    warmup_iterations = 3

    def __init__(self, model, example_x, example_y, precision):
        if type(model) is not GPT:
            raise TypeError("TrainingGraph supports only this study's unwrapped GPT")
        if precision not in ("bf16", "fp32"):
            raise ValueError("Graph precision must be the existing bf16 or fp32 recipe")
        if (not isinstance(example_x, torch.Tensor) or not isinstance(example_y, torch.Tensor)
                or example_x.ndim != 2 or example_x.shape != example_y.shape
                or example_x.shape[0] != 4 or example_x.shape[1] != model.config.seq_len
                or example_x.dtype != torch.long or example_y.dtype != torch.long
                or example_x.device.type != "cuda" or example_y.device != example_x.device):
            raise ValueError("Capture needs matching CUDA long inputs with shape [4, model context]")
        if not torch.is_grad_enabled() or torch.is_inference_mode_enabled() or torch.is_anomaly_enabled():
            raise ValueError("Capture requires ordinary gradient-enabled training without anomaly mode")
        self.model, self.precision, self.device = model, precision, example_x.device
        self._config = model.config
        self._parameters = tuple(model.named_parameters())
        self._buffers = tuple(model.named_buffers())
        self._modules = tuple(model.named_modules())
        self._statistics = tuple(module for _, module in self._modules if isinstance(module, StepStatLinear))
        if (not self._parameters or any(not p.requires_grad or p.dtype != torch.float32
                or p.device != self.device or p.layout != torch.strided for _, p in self._parameters)
                or any(b.requires_grad or b.device != self.device or b.layout != torch.strided for _, b in self._buffers)):
            raise ValueError("Capture requires dense FP32 trainable parameters and local nontrainable buffers")
        self._check_hooks()
        for _, parameter in self._parameters:
            grad = parameter.grad
            if grad is not None and (grad.requires_grad or grad.layout != torch.strided
                    or grad.dtype != parameter.dtype or grad.device != parameter.device
                    or grad.shape != parameter.shape or grad.stride() != parameter.stride()):
                raise ValueError("Existing gradients must be dense detached buffers matching parameter layout")
        self._parameter_signatures = tuple(_tensor_signature(p) for _, p in self._parameters)
        self._buffer_signatures = tuple(_tensor_signature(b) for _, b in self._buffers)
        self._stat_signatures = tuple(_stat_signature(m) for m in self._statistics)
        self._backend_flags = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
        self.replay_count = self.fallback_count = 0
        self._attached = False
        self._valid = False
        started = time.perf_counter()
        with torch.cuda.device(self.device):
            self._stream = torch.cuda.current_stream(self.device)
            torch.cuda.synchronize(self.device)
            original_grads = tuple(p.grad for _, p in self._parameters)
            grad_values = tuple(g.detach().clone() if g is not None else None for g in original_grads)
            buffer_values = tuple(b.detach().clone() for _, b in self._buffers)
            modes = tuple(m.training for _, m in self._modules)
            collection = tuple(m.collect_stats for m in self._statistics)
            cpu_rng, cuda_rng = torch.get_rng_state(), torch.cuda.get_rng_state(self.device)
            python_rng, numpy_rng = random.getstate(), np.random.get_state()
            self._originally_none = tuple(g is None for g in original_grads)
            self._gradients = tuple(g if g is not None else torch.zeros_like(p, memory_format=torch.preserve_format)
                                    for (_, p), g in zip(self._parameters, original_grads))
            self._gradient_signatures = tuple(_tensor_signature(g) for g in self._gradients)
            self._x = example_x.detach().clone(memory_format=torch.contiguous_format)
            self._y = example_y.detach().clone(memory_format=torch.contiguous_format)
            self._weight = torch.ones((), device=self.device, dtype=torch.float32)
            self._graph = torch.cuda.CUDAGraph()
            side = torch.cuda.Stream(device=self.device)
            try:
                for (_, parameter), grad in zip(self._parameters, self._gradients):
                    parameter.grad = grad
                for _, module in self._modules:
                    module.training = True
                for module in self._statistics:
                    module.collect_stats = True
                side.wait_stream(self._stream)
                with torch.cuda.stream(side):
                    for _ in range(self.warmup_iterations):
                        for grad in self._gradients:
                            grad.zero_()
                        self._forward_backward(self._x, self._y, self._weight)
                        self._check_gradient_pointers()
                side.synchronize()
                with torch.cuda.graph(self._graph, stream=side, capture_error_mode="global"):
                    self._loss = self._forward_backward(self._x, self._y, self._weight)
                self._check_gradient_pointers()
            finally:
                # Includes nonpersistent means/covariances, sum buffers, counters
                # and positions. Restore in place; captured addresses stay live.
                torch.cuda.synchronize(self.device)
                with torch.no_grad():
                    for (_, buffer), value in zip(self._buffers, buffer_values):
                        buffer.copy_(value)
                    for (_, parameter), buffer, original, value in zip(
                            self._parameters, self._gradients, original_grads, grad_values):
                        if value is None:
                            buffer.zero_()
                        else:
                            buffer.copy_(value)
                        parameter.grad = original
                for (_, module), mode in zip(self._modules, modes):
                    module.training = mode
                for module, enabled in zip(self._statistics, collection):
                    module.collect_stats = enabled
                torch.set_rng_state(cpu_rng)
                torch.cuda.set_rng_state(cuda_rng, self.device)
                random.setstate(python_rng)
                np.random.set_state(numpy_rng)
                torch.cuda.synchronize(self.device)
        self.setup_seconds = time.perf_counter() - started
        self.capture_seconds = self.setup_seconds
        self._valid = True

    def _amp(self):
        # AMP's cache is Python state and cannot manage parameter casts across
        # replay. Disabling the cache retains the same BF16/FP32 operators.
        return (torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False)
                if self.precision == "bf16" else nullcontext())

    def _forward_backward(self, x, y, weight):
        with self._amp():
            loss = self.model(x, y)
        (loss * weight).backward()
        return loss.detach()

    def _check_hooks(self):
        global_hooks = torch.nn.modules.module
        if any(getattr(global_hooks, name, None) for name in (
                "_global_forward_hooks", "_global_forward_pre_hooks",
                "_global_backward_hooks", "_global_backward_pre_hooks")):
            raise ValueError("Global module hooks are unsupported during graph capture/replay")
        for _, module in self._modules:
            if module._forward_hooks or module._forward_pre_hooks or module._backward_hooks or module._backward_pre_hooks:
                raise ValueError("Module hooks, including GN hooks, must be outside graph capture/replay")
        for _, parameter in self._parameters:
            if getattr(parameter, "_backward_hooks", None) or getattr(parameter, "_post_accumulate_grad_hooks", None):
                raise ValueError("Parameter hooks are unsupported during graph capture/replay")

    def _check_gradient_pointers(self):
        if any(p.grad is None or _tensor_signature(p.grad) != signature
               for (_, p), signature in zip(self._parameters, self._gradient_signatures)):
            raise RuntimeError("Graph gradient buffers changed; use zero_grad(set_to_none=False)")

    def _check_runtime(self, x, y, weight):
        if not self._valid:
            raise RuntimeError("Graph capture did not finish")
        if (not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor) or x.ndim != 2
                or x.shape != y.shape or x.shape[1] != self._x.shape[1] or not 1 <= x.shape[0] <= 4
                or x.dtype != torch.long or y.dtype != torch.long or x.device != self.device or y.device != self.device):
            raise ValueError("Graph inputs must keep context/device/dtype and contain one to four sequences")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight <= 0:
            raise ValueError("Microbatch weight must be a finite positive Python scalar")
        if (not torch.is_grad_enabled() or torch.is_inference_mode_enabled() or torch.is_anomaly_enabled()
                or self.model.config is not self._config
                or any(not module.training for _, module in self._modules)
                or any(not module.collect_stats for module in self._statistics)):
            raise RuntimeError("Replay is restricted to the unchanged model inside its training collection window")
        if torch.cuda.current_stream(self.device).cuda_stream != self._stream.cuda_stream:
            raise RuntimeError("Graph and eager training must use the same caller CUDA stream")
        if (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) != self._backend_flags:
            raise RuntimeError("Backend precision flags changed since capture")
        self._check_hooks()
        current_parameters, current_buffers = tuple(self.model.named_parameters()), tuple(self.model.named_buffers())
        current_modules = tuple(self.model.named_modules())
        for current, original, signatures in ((current_parameters, self._parameters, self._parameter_signatures),
                                               (current_buffers, self._buffers, self._buffer_signatures)):
            if len(current) != len(original) or any(name != expected_name or tensor is not expected
                    or _tensor_signature(tensor) != signature
                    for (name, tensor), (expected_name, expected), signature in zip(current, original, signatures)):
                raise RuntimeError("Captured parameter/buffer identity, shape or storage changed")
        if len(current_modules) != len(self._modules) or any(name != expected_name or module is not expected
                for (name, module), (expected_name, expected) in zip(current_modules, self._modules)):
            raise RuntimeError("Model modules changed since capture")
        if any(_stat_signature(module) != signature for module, signature in zip(self._statistics, self._stat_signatures)):
            raise RuntimeError("Statistic recipe changed since capture")

    def backward(self, x, y, weight):
        """Accumulate a weighted microbatch gradient; return owned detached loss."""
        self._check_runtime(x, y, weight)
        if not self._attached:
            for (_, parameter), grad, originally_none in zip(self._parameters, self._gradients, self._originally_none):
                if parameter.grad is None and originally_none:
                    parameter.grad = grad
            self._attached = True
        self._check_gradient_pointers()
        if x.shape[0] != 4:
            loss = self._forward_backward(x, y, weight)
            self._check_gradient_pointers()
            self.fallback_count += 1
            return loss
        self._x.copy_(x)
        self._y.copy_(y)
        self._weight.fill_(weight)
        self._graph.replay()
        self.replay_count += 1
        # Own the scalar so a caller retaining losses does not observe the next
        # replay's output through an alias of the graph's static result buffer.
        return self._loss.detach().clone()

"""Compatibility helper for qualified kernel-reference tests; no training pipeline."""
import contextlib
import torch

def amp_context(device, config):
    if device.type == "cuda" and config["precision"] == "bf16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()

"""
model_utils.py — Architecture-agnostic model introspection + MPS compatibility.

NOTE: This module requires PyTorch + HuggingFace transformers.
It is NOT required for the OCS simulation — only for live model
introspection during routing capture.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional


def _find_decoder_layers(model: nn.Module) -> list[nn.Module]:
    base = getattr(model, "model", model)
    language_model = getattr(base, "language_model", None)
    if language_model is not None:
        layers = getattr(language_model, "layers", None)
        if layers is not None:
            return layers
    layers = getattr(base, "layers", None)
    if layers is not None:
        return layers
    h = getattr(base, "h", None)
    if h is not None:
        return h
    raise ValueError(
        "Cannot locate decoder layers in model. "
        "Supported patterns: model.model.language_model.layers, "
        "model.model.layers, model.transformer.h"
    )


@dataclass
class ModelLayout:
    model_type: str
    hidden_size: int
    num_layers: int
    num_experts: int
    top_k: int
    moe_layer_indices: list[int]

    @classmethod
    def from_model(cls, model: nn.Module, config) -> "ModelLayout":
        model_type = getattr(config, "model_type", "unknown")
        hidden_size = getattr(config, "hidden_size", -1)
        num_experts = getattr(config, "num_experts", 0)
        top_k = getattr(config, "num_experts_per_tok", 4)

        decoder_layers = _find_decoder_layers(model)
        num_layers = len(decoder_layers)

        moe_indices: list[int] = []
        for idx, layer in enumerate(decoder_layers):
            mlp = getattr(layer, "mlp", None)
            if mlp is None:
                continue
            if not hasattr(mlp, "gate"):
                continue
            if hasattr(mlp, "experts") or hasattr(mlp, "switch_mlp"):
                moe_indices.append(idx)

        if moe_indices:
            first_moe = getattr(decoder_layers[moe_indices[0]], "mlp", None)
            block_top_k = getattr(first_moe, "top_k", None)
            if block_top_k is not None:
                top_k = block_top_k

        return cls(
            model_type=model_type,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_experts=num_experts,
            top_k=top_k,
            moe_layer_indices=moe_indices,
        )

    @property
    def num_moe_layers(self) -> int:
        return len(self.moe_layer_indices)

    def get_decoder_layers(self, model: nn.Module) -> list[nn.Module]:
        return _find_decoder_layers(model)


def enable_router_logits(model: nn.Module) -> bool:
    config = model.config
    patched = False
    for cfg in (config, getattr(config, "text_config", None)):
        if cfg is not None and hasattr(cfg, "output_router_logits"):
            cfg.output_router_logits = True
            patched = True
    return patched


def apply_mps_patches() -> bool:
    if not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        return False
    _orig_histc = torch.histc

    def _mps_safe_histc(input, bins=100, min=0, max=0):
        if input.device.type == "mps" and not torch.is_floating_point(input):
            input = input.to(torch.float32)
        return _orig_histc(input, bins=bins, min=min, max=max)

    torch.histc = _mps_safe_histc
    return True

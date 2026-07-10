"""
interventions.py — Router steering for MoE research.

Provides ``RouterSteering``, a context manager that temporarily patches
gate forward methods to apply forced routing, logit biasing, or expert
ablation — without modifying model weights.

NOTE: This module requires PyTorch + HuggingFace transformers.
It is NOT required for the OCS simulation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.data.model_utils import ModelLayout, _find_decoder_layers


def _make_steered_forward(
    orig_forward,
    gate_module: nn.Module,
    biases: dict[int, float],
    force: dict[int, bool],
    ablate: set[int],
):
    weight = gate_module.weight
    hidden_dim = gate_module.hidden_dim
    top_k = gate_module.top_k
    norm_topk_prob = getattr(gate_module, "norm_topk_prob", False)

    def steered_forward(hidden_states: torch.Tensor):
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        router_logits = F.linear(hidden_states, weight)

        for eid, bias_val in biases.items():
            router_logits[:, eid] = router_logits[:, eid] + bias_val

        for eid in ablate:
            router_logits[:, eid] = float("-inf")

        if force:
            router_logits = router_logits.float()
            for eid, exclusive in force.items():
                if exclusive:
                    router_logits[:, :] = float("-inf")
                router_logits[:, eid] = 100.0
            router_logits = router_logits.to(weight.dtype)

        router_probs = F.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(
            router_probs, top_k, dim=-1
        )
        if norm_topk_prob:
            router_top_value = router_top_value / router_top_value.sum(
                dim=-1, keepdim=True
            )
        router_top_value = router_top_value.to(router_logits.dtype)
        return router_logits, router_top_value, router_indices

    return steered_forward


class RouterSteering:
    def __init__(
        self,
        model: Optional[nn.Module] = None,
        layout: Optional[ModelLayout] = None,
    ):
        self.model = model
        self.layout = layout
        self._interventions: dict[int, dict] = {}
        self._originals: dict[int, object] = {}

    def force_expert(
        self, layer: int, expert_id: int, exclusive: bool = False
    ) -> None:
        cfg = self._interventions.setdefault(layer, {})
        cfg.setdefault("force", {})[expert_id] = exclusive

    def bias_expert(self, layer: int, expert_id: int, bias: float) -> None:
        cfg = self._interventions.setdefault(layer, {})
        cfg.setdefault("biases", {})[expert_id] = (
            cfg.setdefault("biases", {}).get(expert_id, 0.0) + bias
        )

    def ablate_expert(self, layer: int, expert_id: int) -> None:
        cfg = self._interventions.setdefault(layer, {})
        cfg.setdefault("ablate", set()).add(expert_id)

    def __enter__(self) -> "RouterSteering":
        decoder_layers = _find_decoder_layers(self.model)

        for layer_idx, config in self._interventions.items():
            if layer_idx < 0 or layer_idx >= len(decoder_layers):
                print(
                    f"[steering] WARNING: layer {layer_idx} out of range "
                    f"(0-{len(decoder_layers) - 1}), skipping"
                )
                continue

            layer = decoder_layers[layer_idx]
            mlp = getattr(layer, "mlp", None)
            if mlp is None or not hasattr(mlp, "gate"):
                print(
                    f"[steering] WARNING: layer {layer_idx} has no MoE gate, skipping"
                )
                continue

            gate = mlp.gate
            self._originals[layer_idx] = gate.forward

            steered = _make_steered_forward(
                orig_forward=gate.forward,
                gate_module=gate,
                biases=config.get("biases", {}),
                force=config.get("force", {}),
                ablate=config.get("ablate", set()),
            )
            gate.forward = steered

        return self

    def __exit__(self, *args) -> bool:
        decoder_layers = _find_decoder_layers(self.model)
        for layer_idx, orig_forward in self._originals.items():
            if layer_idx < len(decoder_layers):
                decoder_layers[layer_idx].mlp.gate.forward = orig_forward
        self._originals.clear()
        return False

    @property
    def active_layers(self) -> list[int]:
        return sorted(self._interventions.keys())

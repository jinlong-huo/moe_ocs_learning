"""Replay router: replays routing decisions captured from real MoE models.

Instead of computing synthetic routing (fixed, random, top-k), this router
loads a RoutingTrace from a real model and replays expert assignments
token-by-token. This feeds authentic routing patterns into the OCS
communication simulator.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.routing_schema import RoutingTrace


class ReplayRouter(nn.Module):
    """Router that replays captured routing decisions from a RoutingTrace.

    Maps tokens to experts based on pre-recorded routing, cycling through
    the trace if there are more tokens than recorded routes.

    When ``sim_num_experts`` differs from the trace's expert count, expert
    IDs are remapped via modulo (trace_expert % sim_num_experts).

    Parameters
    ----------
    trace : RoutingTrace
        Captured routing from a real MoE model.
    layer_idx : int
        Which MoE layer's routing to replay (0-indexed within moe_layer_indices).
        Default 0 replays the shallowest MoE layer.
    sim_num_experts : int
        Number of experts in the simulation. If different from trace,
        expert IDs are remapped by modulo.
    sim_top_k : int
        Top-K for the simulation. If different from trace,
        weights are sliced/padded.
    """

    def __init__(
        self,
        trace: RoutingTrace,
        layer_idx: int = 0,
        sim_num_experts: int | None = None,
        sim_top_k: int | None = None,
    ):
        super().__init__()
        self.trace = trace
        self.hidden_dim = 1
        self.trace_num_experts = trace.meta.num_experts
        self.trace_top_k = trace.meta.top_k
        self.num_experts = sim_num_experts if sim_num_experts is not None else self.trace_num_experts
        self.top_k = sim_top_k if sim_top_k is not None else self.trace_top_k
        self.strategy = "replay"
        self._need_remap = self.num_experts != self.trace_num_experts
        self._need_resize_k = self.top_k != self.trace_top_k

        lid = str(layer_idx)
        self._cached_experts: list[list[int]] = []
        self._cached_weights: list[list[float]] = []

        for route in trace.routes:
            lr = route.layers.get(lid)
            if lr is not None:
                experts = lr.experts
                weights = lr.weights
                if self._need_remap:
                    experts = [e % self.num_experts for e in experts]
                if self._need_resize_k:
                    if self.top_k < len(experts):
                        experts = experts[: self.top_k]
                        weights = weights[: self.top_k]
                    else:
                        pad = self.top_k - len(experts)
                        experts = experts + [experts[-1] if experts else 0] * pad
                        weights = weights + [0.0] * pad
                self._cached_experts.append(experts)
                self._cached_weights.append(weights)
            else:
                self._cached_experts.append(
                    [i % self.num_experts for i in range(self.top_k)]
                )
                self._cached_weights.append([1.0 / self.top_k] * self.top_k)

        if not self._cached_experts:
            self._cached_experts = [[i % self.num_experts for i in range(self.top_k)]]
            self._cached_weights = [[1.0 / self.top_k] * self.top_k]

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replay routing for a batch of tokens.

        Args:
            tokens: [total_tokens, hidden_dim]

        Returns:
            expert_ids: [total_tokens] or [total_tokens, top_k]
            gate_weights: [total_tokens, top_k]
            logits: [total_tokens, num_experts] dummy logits (zeros)
        """
        total_tokens = tokens.shape[0]
        num_cached = len(self._cached_experts)
        device = tokens.device

        expert_list = []
        weight_list = []
        for i in range(total_tokens):
            expert_list.append(self._cached_experts[i % num_cached])
            weight_list.append(self._cached_weights[i % num_cached])

        expert_ids = torch.tensor(expert_list, device=device, dtype=torch.long)
        gate_weights = torch.tensor(weight_list, device=device, dtype=torch.float32)

        if self.top_k == 1:
            expert_ids = expert_ids.squeeze(-1)

        logits = torch.zeros(total_tokens, self.num_experts, device=device)
        return expert_ids, gate_weights, logits


class LayerCyclingReplayRouter(nn.Module):
    """Replay router that cycles through layers on successive calls.

    Each forward() call advances to the next MoE layer in the trace,
    simulating a multi-layer MoE model's routing patterns over time.
    Useful for simulating a full model pass where each micro-batch
    represents a different transformer layer.

    Supports expert remapping when sim_num_experts != trace_num_experts.
    """

    def __init__(
        self,
        trace: RoutingTrace,
        sim_num_experts: int | None = None,
        sim_top_k: int | None = None,
    ):
        super().__init__()
        self.trace = trace
        self.hidden_dim = 1
        self.trace_num_experts = trace.meta.num_experts
        self.trace_top_k = trace.meta.top_k
        self.num_experts = sim_num_experts if sim_num_experts is not None else self.trace_num_experts
        self.top_k = sim_top_k if sim_top_k is not None else self.trace_top_k
        self.strategy = "replay_cycling"
        self._need_remap = self.num_experts != self.trace_num_experts
        self._need_resize_k = self.top_k != self.trace_top_k

        self._layer_cache: dict[str, tuple[list[list[int]], list[list[float]]]] = {}
        self._layer_list = sorted(trace.routes[0].layers.keys(), key=int) if trace.routes else ["0"]
        self._layer_idx = 0

        for lid in self._layer_list:
            experts = []
            weights = []
            for route in trace.routes:
                lr = route.layers.get(lid)
                if lr is not None:
                    layer_experts = lr.experts
                    layer_weights = lr.weights
                    if self._need_remap:
                        layer_experts = [e % self.num_experts for e in layer_experts]
                    if self._need_resize_k:
                        if self.top_k < len(layer_experts):
                            layer_experts = layer_experts[: self.top_k]
                            layer_weights = layer_weights[: self.top_k]
                        else:
                            pad = self.top_k - len(layer_experts)
                            layer_experts = layer_experts + [layer_experts[-1] if layer_experts else 0] * pad
                            layer_weights = layer_weights + [0.0] * pad
                    experts.append(layer_experts)
                    weights.append(layer_weights)
                else:
                    experts.append([i % self.num_experts for i in range(self.top_k)])
                    weights.append([1.0 / self.top_k] * self.top_k)
            if not experts:
                experts = [[i % self.num_experts for i in range(self.top_k)]]
                weights = [[1.0 / self.top_k] * self.top_k]
            self._layer_cache[lid] = (experts, weights)

    def set_layer(self, layer_idx: int) -> None:
        if 0 <= layer_idx < len(self._layer_list):
            self._layer_idx = layer_idx

    def advance_layer(self) -> None:
        self._layer_idx = (self._layer_idx + 1) % len(self._layer_list)

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_tokens = tokens.shape[0]
        lid = self._layer_list[self._layer_idx]
        cached_experts, cached_weights = self._layer_cache[lid]
        num_cached = len(cached_experts)
        device = tokens.device

        expert_list = []
        weight_list = []
        for i in range(total_tokens):
            expert_list.append(cached_experts[i % num_cached])
            weight_list.append(cached_weights[i % num_cached])

        expert_ids = torch.tensor(expert_list, device=device, dtype=torch.long)
        gate_weights = torch.tensor(weight_list, device=device, dtype=torch.float32)
        if self.top_k == 1:
            expert_ids = expert_ids.squeeze(-1)
        logits = torch.zeros(total_tokens, self.num_experts, device=device)

        self.advance_layer()
        return expert_ids, gate_weights, logits

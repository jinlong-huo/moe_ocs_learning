"""
qwen_experts.py — Real Qwen MoE expert/gate modules in PyTorch.

Replaces the synthetic TinyExpert/FFNExpert with actual Qwen SwitchGLU
expert modules whose weights are exported from a Qwen MoE model (MLX or HF).

Each expert implements the SwitchGLU forward:
    gate = silu(x @ gate_proj)
    up   = x @ up_proj
    out  = (gate * up) @ down_proj

The gate (router) is a linear layer shared across all ranks.

Usage in the OCS simulation:
    from src.model.qwen_experts import QwenExpert, QwenGate, create_qwen_moe_layer
    moe = create_qwen_moe_layer(weight_dir="exported_weights/layer_0", rank=0, ...)
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QwenExpert(nn.Module):
    """A single Qwen SwitchGLU expert with loaded weights.

    The Qwen architecture uses concatenation of gate and up projections:
        gate = silu(x @ gate_proj)   — gate_proj: [hidden_in → gate_out]
        up   = x @ up_proj           — up_proj:   [hidden_in → up_out]
        out  = cat(gate, up) @ down_proj  — down_proj: [gate_out+up_out → hidden_out]

    Note: hidden_in may differ from hidden_out (Qwen uses projections).
    Dimensions are auto-detected from loaded weights.
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.gate_proj(x))
        up = self.up_proj(x)
        combined = torch.cat([gate, up], dim=-1)
        return self.down_proj(combined)

    def load_weights(self, path: str) -> None:
        """Load expert weights from .npz or .pt file, auto-detecting dimensions."""
        if path.endswith(".npz"):
            data = dict(**dict(np.load(path)))
            gp = torch.from_numpy(data["gate_proj"]).float()
            up = torch.from_numpy(data["up_proj"]).float()
            dn = torch.from_numpy(data["down_proj"]).float()
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            gp = checkpoint["gate_proj"]
            up = checkpoint["up_proj"]
            dn = checkpoint["down_proj"]

        gate_out, hidden_in = gp.shape
        up_out, _ = up.shape
        hidden_out, combined_in = dn.shape

        self.hidden_dim = hidden_in
        self.intermediate_dim = gate_out

        self.gate_proj = nn.Linear(hidden_in, gate_out, bias=False)
        self.gate_proj.weight.data = gp

        self.up_proj = nn.Linear(hidden_in, up_out, bias=False)
        self.up_proj.weight.data = up

        self.down_proj = nn.Linear(combined_in, hidden_out, bias=False)
        self.down_proj.weight.data = dn


class QwenGate(nn.Module):
    """Qwen MoE router gate.

    Forward: gate(x) → softmax → top-k → (expert_ids, gate_weights, logits)
    """

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_dim))

    def forward(self, tokens: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        total_tokens = tokens.shape[0]
        logits = F.linear(tokens, self.weight)
        probs = F.softmax(logits, dim=-1)
        gate_weights, expert_ids = torch.topk(probs, k=self.top_k, dim=-1)
        return expert_ids, gate_weights, logits

    def load_state_dict_from_pt(self, path: str, sim_num_experts: int | None = None) -> None:
        """Load gate weights, optionally slicing to fewer experts."""
        if path.endswith(".npz") or path.endswith(".npy"):
            if path.endswith(".npy"):
                w = torch.from_numpy(np.load(path)).float()
            else:
                data = dict(**dict(np.load(path)))
                w = torch.from_numpy(data["weight"]).float()
        else:
            data = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(data, dict) and "weight" in data:
                w = data["weight"]
            else:
                w = data

        if sim_num_experts is not None and sim_num_experts < w.shape[0]:
            w = w[:sim_num_experts]
        self.weight.data = w
        self.num_experts = w.shape[0]


class QwenSharedExpert(nn.Module):
    """Qwen shared expert (runs for every token regardless of routing).

    Uses the same concat architecture as regular experts.
    Forward: sigmoid(gate(x)) * shared_expert(x)
    """

    def __init__(self, hidden_dim: int, intermediate_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.intermediate_dim = intermediate_dim
        self.gate_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.up_proj = nn.Linear(hidden_dim, intermediate_dim, bias=False)
        self.down_proj = nn.Linear(intermediate_dim, hidden_dim, bias=False)
        self.gate = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_out = F.silu(self.gate_proj(x))
        up_out = self.up_proj(x)
        combined = torch.cat([gate_out, up_out], dim=-1)
        expert_out = self.down_proj(combined)
        gate_weight = torch.sigmoid(self.gate(x))
        return gate_weight * expert_out

    def load_weights(self, path: str) -> None:
        """Load shared expert weights from .npz or .pt file."""
        if path.endswith(".npz"):
            data = dict(**dict(np.load(path)))
            gp = torch.from_numpy(data["gate_proj"]).float()
            up = torch.from_numpy(data["up_proj"]).float()
            dn = torch.from_numpy(data["down_proj"]).float()
        else:
            d = torch.load(path, map_location="cpu", weights_only=True)
            gp = d.get("gate_proj", torch.zeros(1))
            up = d.get("up_proj", torch.zeros(1))
            dn = d.get("down_proj", torch.zeros(1))

        gate_out, hidden_in = gp.shape
        hidden_out, combined_in = dn.shape

        self.gate_proj = nn.Linear(hidden_in, gate_out, bias=False)
        self.gate_proj.weight.data = gp
        self.up_proj = nn.Linear(hidden_in, gate_out, bias=False)
        self.up_proj.weight.data = up
        self.down_proj = nn.Linear(combined_in, hidden_out, bias=False)
        self.down_proj.weight.data = dn

        if path.endswith(".npz"):
            sw = torch.from_numpy(data.get("shared_expert_gate", np.zeros((hidden_in, 1)).T)).float()
            self.gate = nn.Linear(hidden_in, 1, bias=False)
            self.gate.weight.data = sw
        elif "shared_expert_gate" in d:
            self.gate.weight.data = d["shared_expert_gate"]
        elif "gate" in d:
            self.gate.weight.data = d["gate"]


def create_qwen_moe_layer(
    weight_dir: str,
    rank: int,
    world_size: int,
    experts_per_rank: int,
    hidden_dim: int = 2048,
    intermediate_dim: int = 512,
    num_experts: int = 256,
    top_k: int = 8,
    layer_idx: int = 0,
) -> "QwenMoELayerWrapper":
    """Build a MoE layer with Qwen experts and gate.

    Each rank loads only its own expert weights from the exported weight files.

    Args:
        weight_dir: Directory containing exported expert_{N}.pt and gate.pt files.
        rank: This process's rank.
        world_size: Total number of ranks.
        experts_per_rank: Number of experts assigned to each rank.
        hidden_dim: Model hidden dimension.
        intermediate_dim: Expert intermediate (FFN) dimension.
        num_experts: Total number of experts.
        top_k: Top-k for routing.
        layer_idx: Which transformer layer this MoE layer corresponds to.
    """
    gate = QwenGate(hidden_dim, num_experts, top_k)

    for ext in (".npy", ".pt"):
        gate_path = os.path.join(weight_dir, f"gate{ext}")
        if os.path.exists(gate_path):
            gate.load_state_dict_from_pt(gate_path, sim_num_experts=num_experts)
            break

    experts = nn.ModuleList()
    for local_idx in range(experts_per_rank):
        expert = QwenExpert(hidden_dim, intermediate_dim)
        global_expert_id = rank * experts_per_rank + local_idx
        for ext in (".npz", ".pt"):
            expert_path = os.path.join(weight_dir, f"expert_{global_expert_id}{ext}")
            if os.path.exists(expert_path):
                expert.load_weights(expert_path)
                break
        experts.append(expert)

    shared_expert = QwenSharedExpert(hidden_dim, intermediate_dim * 4)
    for ext in (".npz", ".pt"):
        shared_path = os.path.join(weight_dir, f"shared_expert{ext}")
        if os.path.exists(shared_path):
            shared_expert.load_weights(shared_path)
            break

    return QwenMoELayerWrapper(
        gate=gate,
        experts=experts,
        shared_expert=shared_expert,
        num_experts=num_experts,
        experts_per_rank=experts_per_rank,
        top_k=top_k,
        rank=rank,
        world_size=world_size,
    )


class QwenMoELayerWrapper(nn.Module):
    """Wraps Qwen gate + experts to match the MoELayer interface used by schedulers.

    Provides .router (gate), .compute_experts(), .expert_id_to_local() to
    be a drop-in replacement for MoELayer in the OCS simulation.
    """

    def __init__(
        self,
        gate: QwenGate,
        experts: nn.ModuleList,
        shared_expert: Optional[QwenSharedExpert],
        num_experts: int,
        experts_per_rank: int,
        top_k: int,
        rank: int,
        world_size: int,
    ):
        super().__init__()
        self.router = gate
        self.experts = experts
        self.shared_expert = shared_expert
        self.num_experts = num_experts
        self.experts_per_rank = experts_per_rank
        self.top_k = top_k
        self.hidden_dim = gate.hidden_dim
        self._rank = rank
        self._world_size = world_size

    @property
    def rank(self) -> int:
        return self._rank

    def set_rank(self, rank: int, world_size: int) -> None:
        self._rank = rank
        self._world_size = world_size

    def expert_id_to_local(self, expert_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        target_rank = expert_id // self.experts_per_rank
        local_expert = expert_id % self.experts_per_rank
        return target_rank, local_expert

    def compute_experts(
        self, routed_tokens: torch.Tensor, local_expert_ids: torch.Tensor
    ) -> torch.Tensor:
        expert_out = self.experts[0](routed_tokens[:1])
        out_dim = expert_out.shape[-1]
        output = torch.zeros(routed_tokens.shape[0], out_dim,
                            device=routed_tokens.device, dtype=routed_tokens.dtype)
        for local_idx in range(self.experts_per_rank):
            mask = (local_expert_ids == local_idx)
            if mask.any():
                output[mask] = self.experts[local_idx](routed_tokens[mask])
        if self.shared_expert is not None:
            output = output + self.shared_expert(routed_tokens)
        return output

    def forward_serial(self, tokens: torch.Tensor, transport) -> torch.Tensor:
        """Serial forward: route → dispatch → compute → gather → combine."""
        from src.comm.all_to_all import scatter_tokens, gather_tokens, combine_expert_outputs

        expert_ids, gate_weights, _logits = self.router(tokens)
        dispatch = scatter_tokens(
            tokens, expert_ids, self.num_experts,
            self.experts_per_rank, transport, async_op=False,
        )
        expert_out = self.compute_experts(dispatch.tokens, dispatch.local_expert_ids)
        gathered = gather_tokens(expert_out, dispatch, transport, async_op=False)
        return combine_expert_outputs(gathered, gate_weights)

    def forward_train(self, tokens: torch.Tensor, transport) -> tuple:
        """Training forward with debug info."""
        from src.comm.all_to_all import scatter_tokens, gather_tokens, combine_expert_outputs

        expert_ids, gate_weights, logits = self.router(tokens)
        dispatch = scatter_tokens(
            tokens, expert_ids, self.num_experts,
            self.experts_per_rank, transport, async_op=False,
        )
        expert_out = self.compute_experts(dispatch.tokens, dispatch.local_expert_ids)
        gathered = gather_tokens(expert_out, dispatch, transport, async_op=False)
        combined = combine_expert_outputs(gathered, gate_weights)
        return combined, logits, dispatch.send_counts, dispatch

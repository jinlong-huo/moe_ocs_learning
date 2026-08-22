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

from src.runtime.placement import Placement


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
        # SwiGLU: element-wise multiply gate and up, NOT concatenation
        # Reference: Qwen3MoeSparseMoeBlock uses gate * up then down_proj
        return self.down_proj(gate * up)

    def load_weights(self, path: str) -> None:
        """Load expert weights from .npz or .pt file, auto-detecting dimensions.

        Handles both properly-dequantized float32 weights and packed uint32
        weights (from an earlier export bug).  Packed uint32 stores 4×int8
        values per element along the last dimension; these are unpacked to
        float32 with a warning about approximate values.
        """
        if path.endswith(".npz"):
            data = dict(**dict(np.load(path)))
            gp_raw = torch.from_numpy(data["gate_proj"])
            up_raw = torch.from_numpy(data["up_proj"])
            dn_raw = torch.from_numpy(data["down_proj"])
        else:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            gp_raw = checkpoint["gate_proj"]
            up_raw = checkpoint["up_proj"]
            dn_raw = checkpoint["down_proj"]

        # Detect packed uint32 (export bug: _try_dequantize fell back to raw packed weights)
        gp, gp_unpacked = self._maybe_unpack(gp_raw, "gate_proj")
        up, up_unpacked = self._maybe_unpack(up_raw, "up_proj")
        dn, dn_unpacked = self._maybe_unpack(dn_raw, "down_proj")
        if gp_unpacked or up_unpacked or dn_unpacked:
            import sys
            print(
                f"[qwen_experts] WARNING: {path} contains packed uint32 weights. "
                f"Unpacked to float32 (approximate — re-export with fixed "
                f"export_qwen_experts.py for accurate values).",
                file=sys.stderr,
            )

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

    @staticmethod
    def _maybe_unpack(weight: torch.Tensor, name: str = "") -> tuple[torch.Tensor, bool]:
        """Detect and unpack uint32-packed 8-bit quantized weights.

        MLX stores 8-bit quantized weights as 4×int8 values packed into each
        uint32 element along the last dimension.  If the tensor is uint32,
        unpack it to float32 with shape [*batch, last_dim * 4].

        Returns (tensor, was_unpacked).
        """
        if weight.dtype == torch.uint32:
            # Reinterpret uint32 as 4×uint8, convert to float32
            w_u8 = weight.view(torch.uint8)  # [*dims, last_dim * 4]
            w_f32 = w_u8.to(torch.float32)
            return w_f32, True
        return weight.float(), False


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
        """Load gate weights, optionally slicing to fewer experts.

        Handles both float32 and packed uint32 gate weights.
        """
        if path.endswith(".npz") or path.endswith(".npy"):
            if path.endswith(".npy"):
                w_raw = torch.from_numpy(np.load(path))
            else:
                data = dict(**dict(np.load(path)))
                w_raw = torch.from_numpy(data["weight"])
        else:
            data = torch.load(path, map_location="cpu", weights_only=True)
            if isinstance(data, dict) and "weight" in data:
                w_raw = data["weight"]
            else:
                w_raw = data

        # Unpack if uint32-packed (8-bit quantized MLX storage)
        w, was_unpacked = QwenExpert._maybe_unpack(w_raw, "gate")
        if was_unpacked:
            import sys
            print(
                f"[qwen_experts] WARNING: {path} gate weights are packed uint32. "
                f"Unpacked to float32 (approximate).",
                file=sys.stderr,
            )

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
        # SwiGLU: element-wise multiply gate and up, NOT concatenation
        expert_out = self.down_proj(gate_out * up_out)
        gate_weight = torch.sigmoid(self.gate(x))
        return gate_weight * expert_out

    def load_weights(self, path: str) -> None:
        """Load shared expert weights from .npz or .pt file.

        Handles both float32 and packed uint32 weights.
        """
        if path.endswith(".npz"):
            data = dict(**dict(np.load(path)))
            gp_raw = torch.from_numpy(data["gate_proj"])
            up_raw = torch.from_numpy(data["up_proj"])
            dn_raw = torch.from_numpy(data["down_proj"])
        else:
            d = torch.load(path, map_location="cpu", weights_only=True)
            gp_raw = d.get("gate_proj", torch.zeros(1))
            up_raw = d.get("up_proj", torch.zeros(1))
            dn_raw = d.get("down_proj", torch.zeros(1))

        # Unpack if uint32-packed
        gp, gp_unpacked = QwenExpert._maybe_unpack(gp_raw, "shared_expert.gate_proj")
        up, up_unpacked = QwenExpert._maybe_unpack(up_raw, "shared_expert.up_proj")
        dn, dn_unpacked = QwenExpert._maybe_unpack(dn_raw, "shared_expert.down_proj")
        if gp_unpacked or up_unpacked or dn_unpacked:
            import sys
            print(
                f"[qwen_experts] WARNING: {path} contains packed uint32 weights. "
                f"Unpacked to float32 (approximate).",
                file=sys.stderr,
            )

        gate_out, hidden_in = gp.shape
        hidden_out, combined_in = dn.shape

        self.gate_proj = nn.Linear(hidden_in, gate_out, bias=False)
        self.gate_proj.weight.data = gp
        self.up_proj = nn.Linear(hidden_in, gate_out, bias=False)
        self.up_proj.weight.data = up
        self.down_proj = nn.Linear(combined_in, hidden_out, bias=False)
        self.down_proj.weight.data = dn

        if path.endswith(".npz"):
            sg_raw = torch.from_numpy(data.get("shared_expert_gate", np.zeros((hidden_in, 1)).T))
            sg, _ = QwenExpert._maybe_unpack(sg_raw, "shared_expert_gate")
            self.gate = nn.Linear(hidden_in, 1, bias=False)
            self.gate.weight.data = sg
        elif "shared_expert_gate" in d:
            sg_raw = d["shared_expert_gate"]
            sg, _ = QwenExpert._maybe_unpack(sg_raw, "shared_expert_gate")
            self.gate.weight.data = sg
        elif "gate" in d:
            sg_raw = d["gate"]
            sg, _ = QwenExpert._maybe_unpack(sg_raw, "shared_expert_gate")
            self.gate.weight.data = sg


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
    placement: Optional["Placement"] = None,
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
    if placement is not None:
        owned_expert_ids = placement.experts_on_rank(rank)
    else:
        owned_expert_ids = [
            rank * experts_per_rank + local_idx for local_idx in range(experts_per_rank)
        ]
    for global_expert_id in owned_expert_ids:
        expert = QwenExpert(hidden_dim, intermediate_dim)
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
        placement=placement,
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
        placement: Optional["Placement"] = None,
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
        self.placement = (
            placement
            if placement is not None
            else Placement.linear(num_experts, experts_per_rank, world_size)
        )

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
            placement=self.placement,
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
            placement=self.placement,
        )
        expert_out = self.compute_experts(dispatch.tokens, dispatch.local_expert_ids)
        gathered = gather_tokens(expert_out, dispatch, transport, async_op=False)
        combined = combine_expert_outputs(gathered, gate_weights)
        return combined, logits, dispatch.send_counts, dispatch

"""MoE Layer: router -> dispatch -> expert compute -> combine.

This is the central module. It ties together routing, communication,
and expert computation. The same module works in both serial and
overlap mode -- the difference is in how the scheduler drives it.

Flow (serial inference):
  1. Route tokens -> expert_ids, gate_weights, logits
  2. All-to-all scatter (blocking) -- routes tokens to the rank that owns each expert
  3. Compute expert outputs locally (dispatch to correct local expert)
  4. All-to-all gather (blocking) -- returns tokens to original order
  5. Combine with gate weights

Flow (training):
  Same as above, but returns (output, logits, send_counts) so the
  scheduler can compute task loss + load-balancing loss and call backward().

Flow (overlap mode):
  Step K-1: compute experts while step K scatters tokens
  Step K:   gather K-1 results while computing experts for K

Expert Parallelism:
  Each rank owns `experts_per_rank` experts. The total number of experts is
  `num_experts = world_size * experts_per_rank`. Expert `e` lives on rank
  `e // experts_per_rank` as local expert index `e % experts_per_rank`.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional

from src.comm.transport import Transport
from src.comm.all_to_all import (
    scatter_tokens, gather_tokens, combine_expert_outputs, DispatchResult,
)
from src.model.router import Router
from src.model.experts import TinyExpert, FFNExpert
from src.runtime.placement import Placement


class MoELayer(nn.Module):
    """A single MoE layer: route -> dispatch -> expert -> combine.

    Each rank builds `experts_per_rank` local experts. Token dispatch
    uses the expert-to-rank mapping: rank = expert_id // experts_per_rank.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        top_k: int = 1,
        expert_type: str = "tiny",
        expert_mult: int = 4,
        routing_strategy: str = "fixed",
        experts_per_rank: int = 1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts_per_rank = experts_per_rank

        # Validate: num_experts must be divisible by experts_per_rank
        if num_experts % experts_per_rank != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by "
                f"experts_per_rank ({experts_per_rank})"
            )

        self.router = Router(hidden_dim, num_experts, top_k, strategy=routing_strategy)

        # Build local experts -- each rank owns experts_per_rank experts
        if expert_type == "tiny":
            expert_cls = TinyExpert
            expert_kwargs = {"hidden_dim": hidden_dim}
        elif expert_type == "ffn":
            expert_cls = FFNExpert
            expert_kwargs = {"hidden_dim": hidden_dim, "expand_mult": expert_mult}
        else:
            raise ValueError(f"Unknown expert type: {expert_type}")

        self.experts = nn.ModuleList([
            expert_cls(**expert_kwargs) for _ in range(experts_per_rank)
        ])

        # Set by worker after construction
        self._rank = None
        self._world_size = None
        self.placement = None

    # -- Rank assignment -------------------------------------------------

    def set_rank(self, rank: int, world_size: int, placement: Optional["Placement"] = None) -> None:
        """Store rank, validate world_size matches expert layout, install placement.

        ``placement`` is the independent expert->rank table (linear by default).
        """
        expected = self.num_experts // self.experts_per_rank
        if world_size != expected:
            raise ValueError(
                f"world_size ({world_size}) must equal "
                f"num_experts / experts_per_rank ({expected}). "
                f"Got num_experts={self.num_experts}, experts_per_rank={self.experts_per_rank}"
            )
        self._rank = rank
        self._world_size = world_size
        self.placement = (
            placement
            if placement is not None
            else Placement.linear(self.num_experts, self.experts_per_rank, world_size)
        )

    @property
    def rank(self) -> int:
        if self._rank is None:
            raise RuntimeError("set_rank() must be called before accessing rank")
        return self._rank

    def expert_id_to_local(self, expert_id: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Map global expert IDs to (target_rank, local_expert_idx)."""
        target_rank = expert_id // self.experts_per_rank
        local_expert = expert_id % self.experts_per_rank
        return target_rank, local_expert

    # -- Forward methods --------------------------------------------------

    def forward_serial(
        self,
        tokens: torch.Tensor,
        transport: Transport,
    ) -> torch.Tensor:
        """Serial (baseline) forward: route -> dispatch -> compute -> gather -> combine."""
        # Step 1: Route
        expert_ids, gate_weights, _logits = self.router(tokens)

        # Step 2: Dispatch (blocking all-to-all) -- handles top-K flattening internally
        dispatch = scatter_tokens(
            tokens, expert_ids, self.num_experts,
            self.experts_per_rank, transport, async_op=False,
            placement=self.placement,
        )

        # Step 3: Expert computation
        expert_out = self.compute_experts(dispatch.tokens, dispatch.local_expert_ids)

        # Step 4: Gather (blocking all-to-all)
        gathered = gather_tokens(expert_out, dispatch, transport, async_op=False)

        # Step 5: Combine multi-expert outputs with gate weights
        combined = combine_expert_outputs(gathered, gate_weights)

        return combined

    def forward_train(
        self,
        tokens: torch.Tensor,
        transport: Transport,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DispatchResult]:
        """Training forward pass: same as forward_serial but returns debug info.

        Returns:
            output: [T, hidden_dim] combined MoE output
            logits: [T, num_experts] raw gate logits (for load-balancing loss)
            send_counts: [world_size] per-rank dispatch counts (for load-balancing loss)
            dispatch: DispatchResult (for gather / gradient routing)
        """
        # Step 1: Route
        expert_ids, gate_weights, logits = self.router(tokens)

        # Step 2: Dispatch (blocking all-to-all)
        dispatch = scatter_tokens(
            tokens, expert_ids, self.num_experts,
            self.experts_per_rank, transport, async_op=False,
            placement=self.placement,
        )

        # Step 3: Expert computation
        expert_out = self.compute_experts(dispatch.tokens, dispatch.local_expert_ids)

        # Step 4: Gather (blocking all-to-all)
        gathered = gather_tokens(expert_out, dispatch, transport, async_op=False)

        # Step 5: Combine multi-expert outputs with gate weights
        combined = combine_expert_outputs(gathered, gate_weights)

        return combined, logits, dispatch.send_counts, dispatch

    def compute_experts(
        self, routed_tokens: torch.Tensor, local_expert_ids: torch.Tensor
    ) -> torch.Tensor:
        """Expert computation -- dispatch tokens to the correct local expert.

        Args:
            routed_tokens: [total_received, hidden_dim] tokens received from all-to-all
            local_expert_ids: [total_received] which local expert (0..experts_per_rank-1)
                              handles each token

        Returns:
            expert_outputs: [total_received, hidden_dim] in same order as input
        """
        output = torch.zeros_like(routed_tokens)
        for local_idx in range(self.experts_per_rank):
            mask = (local_expert_ids == local_idx)
            if mask.any():
                output[mask] = self.experts[local_idx](routed_tokens[mask])
        return output

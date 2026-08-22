"""Token-dispatch all-to-all: the core MoE communication primitive.

This module handles:
  - Grouping tokens by target rank based on router-assigned expert_ids
  - All-to-all scatter via the Transport layer (with proper per-rank packing)
  - All-to-all gather of expert outputs back to original order
  - Top-K multi-expert dispatch and weighted combining

Token metadata (local_expert_id, original_index) is packed as extra float
columns alongside token embeddings so it travels through the all-to-all
with zero extra communication rounds. A single all_to_all_single call
handles both data movement and metadata exchange.

Top-K dispatch (top_k > 1):
  Each token is replicated K times, once per selected expert. The original
  index is replicated accordingly so gather() can re-associate outputs.
  Gate weights stay on the originating rank and are applied after gather
  via combine_expert_outputs().

DispatchResult carries all the metadata needed to reverse the scatter
during the gather phase -- no side-channel communication required.

In overlap mode, scatter and gather are issued as async ops so compute
on micro-batch K-1 can run concurrently with communication for micro-batch K.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist

from src.comm.transport import Transport


@dataclass
class DispatchResult:
    """Metadata from token scatter -- carries everything gather() needs.

    All tensors are on CPU (this is a CPU-only simulation).

    For top_k > 1: original_indices contain duplicates (each token appears
    K times), and total_original_tokens is the pre-flatten count T.
    """
    tokens: torch.Tensor           # [total_received, hidden_dim]
    local_expert_ids: torch.Tensor # [total_received] which local expert handles each token
    original_indices: torch.Tensor # [total_received] original position (for gather reorder)
    source_ranks: torch.Tensor     # [total_received] which rank originally sent each token
    send_counts: torch.Tensor      # [world_size] how many tokens *this rank* sent to each dest
    total_original_tokens: int     # original T (before top-K replication)
    top_k: int = 1                 # number of experts per token (1 or 2)


def scatter_tokens(
    tokens: torch.Tensor,             # [total_tokens, hidden_dim] or [total_tokens*top_k, hidden_dim]
    expert_ids: torch.Tensor,        # [total_tokens] or [total_tokens, top_k]
    num_experts: int,
    experts_per_rank: int,
    transport: Transport,
    async_op: bool = False,
    placement: Optional["Placement"] = None,
):
    """Distribute tokens to expert ranks via all-to-all, using actual expert_ids.

    Supports top-K dispatch: if expert_ids is 2D [T, K], tokens are replicated
    K times (once per expert) before dispatch. The original token count T is
    preserved in DispatchResult.total_original_tokens.

    Args:
        tokens: input token embeddings [T, H]
        expert_ids: expert assignment per token. Shape [T] for top-1,
                   or [T, K] for top-K (each token dispatched to K experts).
        num_experts: total number of experts globally
        experts_per_rank: how many experts each rank owns
        transport: communication layer
        async_op: if True, return the async handle without waiting

    Returns:
        If async_op=False: DispatchResult
        If async_op=True: (DispatchResult, handle)
    """
    original_tokens, hidden_dim = tokens.shape
    world_size = dist.get_world_size()

    # Handle top-K: flatten expert_ids from [T, K] to [T*K]
    if expert_ids.dim() == 2:
        top_k = expert_ids.shape[1]
        expert_ids = expert_ids.reshape(-1)              # [T*K]
        # Replicate tokens K times
        tokens = tokens.repeat_interleave(top_k, dim=0)  # [T*K, H]
        # Original indices: each token i repeats K times
        orig_indices = torch.arange(original_tokens).repeat_interleave(top_k)
    else:
        top_k = 1
        orig_indices = torch.arange(original_tokens)

    total_tokens = tokens.shape[0]  # T * top_k

    # 1. Compute target rank and local expert for each (flattened) token.
    #    Placement-aware: the expert->rank table decides where each token goes.
    #    When no placement is given, fall back to the historical linear mapping
    #    (expert e -> rank e // k), which Placement.linear reproduces exactly.
    if placement is not None:
        target_rank, local_expert = placement.resolve(expert_ids)
    else:
        target_rank = expert_ids // experts_per_rank          # [T*K]
        local_expert = expert_ids % experts_per_rank          # [T*K]

    # 2. Count tokens per destination rank
    send_counts = torch.zeros(world_size, dtype=torch.int64)
    for r in range(world_size):
        send_counts[r] = (target_rank == r).sum().item()

    # 3. Exchange counts so every rank knows what to expect
    all_counts = [torch.zeros(world_size, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_counts, send_counts)
    recv_counts = torch.tensor(
        [all_counts[s][dist.get_rank()].item() for s in range(world_size)]
    )

    # 4. Compute max chunk size across all rank-pairs (for padding)
    max_send = max(int(all_counts[r][c].item()) for r in range(world_size) for c in range(world_size))
    max_send = max(max_send, 1)  # ensure at least 1 row so all_to_all_single works

    # 5. Build packed send buffer: [W * max_send, H + 2]
    #    Extra columns: [local_expert_id (float), original_index (float)]
    pack_dim = hidden_dim + 2
    send_buf = torch.zeros(world_size * max_send, pack_dim)

    for dest_r in range(world_size):
        mask = (target_rank == dest_r)
        count = int(send_counts[dest_r].item())
        if count == 0:
            continue
        idxs = mask.nonzero(as_tuple=False).squeeze(-1)

        start = dest_r * max_send
        send_buf[start:start + count, :hidden_dim] = tokens[idxs]
        send_buf[start:start + count, hidden_dim] = local_expert[idxs].float()
        send_buf[start:start + count, hidden_dim + 1] = orig_indices[idxs].float()

    # 5b. Determine which ranks have non-zero traffic (for per-pair metrics)
    active_ranks = sorted(set(
        r for r in range(world_size)
        if send_counts[r].item() > 0 or recv_counts[r].item() > 0
    ))

    # 6. All-to-all
    recv_buf = torch.zeros_like(send_buf)
    handle = transport.all_to_all(
        recv_buf, send_buf, async_op=async_op, active_ranks=active_ranks,
    )

    result = _build_dispatch_result(
        recv_buf, recv_counts, max_send, hidden_dim, send_counts, original_tokens, top_k
    )

    if async_op:
        return result, handle
    return result


def _build_dispatch_result(
    recv_buf: torch.Tensor,
    recv_counts: torch.Tensor,
    max_send: int,
    hidden_dim: int,
    send_counts: torch.Tensor,
    total_original_tokens: int,
    top_k: int,
) -> DispatchResult:
    """Unpack the all-to-all output buffer into a DispatchResult."""
    world_size = len(recv_counts)

    token_chunks = []
    expert_chunks = []
    index_chunks = []
    src_chunks = []

    for src_r in range(world_size):
        count = int(recv_counts[src_r].item())
        if count == 0:
            continue
        start = src_r * max_send
        chunk = recv_buf[start:start + count]

        token_chunks.append(chunk[:, :hidden_dim])
        expert_chunks.append(chunk[:, hidden_dim].long())
        index_chunks.append(chunk[:, hidden_dim + 1].long())
        src_chunks.append(torch.full((count,), src_r, dtype=torch.int64))

    if token_chunks:
        tokens_out = torch.cat(token_chunks, dim=0)
        local_expert_ids = torch.cat(expert_chunks, dim=0)
        original_indices = torch.cat(index_chunks, dim=0)
        source_ranks = torch.cat(src_chunks, dim=0)
    else:
        tokens_out = torch.zeros(0, hidden_dim)
        local_expert_ids = torch.zeros(0, dtype=torch.int64)
        original_indices = torch.zeros(0, dtype=torch.int64)
        source_ranks = torch.zeros(0, dtype=torch.int64)

    return DispatchResult(
        tokens=tokens_out,
        local_expert_ids=local_expert_ids,
        original_indices=original_indices,
        source_ranks=source_ranks,
        send_counts=send_counts,
        total_original_tokens=total_original_tokens,
        top_k=top_k,
    )


def gather_tokens(
    expert_outputs: torch.Tensor,       # [total_received, hidden_dim]
    dispatch: DispatchResult,
    transport: Transport,
    async_op: bool = False,
):
    """Collect expert outputs back to original token order via all-to-all.

    For top_k=1: returns tensor [T, H] in original order.
    For top_k>1: returns tensor [T*K, H] where each original position
                  appears K times. Call combine_expert_outputs() afterward
                  to apply gate weights and sum.

    Args:
        expert_outputs: expert computation results [T_local, H]
        dispatch: DispatchResult from the corresponding scatter_tokens call
        transport: communication layer
        async_op: if True, return (combined, handle)

    Returns:
        [total_original_tokens * top_k, hidden_dim] tensor in original order
    """
    world_size = dist.get_world_size()
    my_rank = dist.get_rank()
    hidden_dim = expert_outputs.shape[1]
    total_original = dispatch.total_original_tokens
    top_k = dispatch.top_k
    output_size = total_original * top_k

    # 1. Pack outputs with original indices: [T_local, H + 1]
    pack_dim = hidden_dim + 1
    packed = torch.zeros(expert_outputs.shape[0], pack_dim)
    packed[:, :hidden_dim] = expert_outputs
    packed[:, hidden_dim] = dispatch.original_indices.float()

    # 2. Count tokens to send to each destination (return to sender)
    gather_send_counts = torch.zeros(world_size, dtype=torch.int64)
    for s in range(world_size):
        gather_send_counts[s] = (dispatch.source_ranks == s).sum().item()

    # 3. Exchange gather counts
    all_gather_counts = [torch.zeros(world_size, dtype=torch.int64) for _ in range(world_size)]
    dist.all_gather(all_gather_counts, gather_send_counts)
    gather_recv_counts = torch.tensor(
        [all_gather_counts[s][my_rank].item() for s in range(world_size)]
    )

    # 4. Compute max chunk size
    max_gather = max(int(all_gather_counts[r][c].item()) for r in range(world_size) for c in range(world_size))
    max_gather = max(max_gather, 1)

    # 5. Build packed send buffer: [W * max_gather, H + 1]
    gather_send_buf = torch.zeros(world_size * max_gather, pack_dim)
    for dest_r in range(world_size):
        mask = (dispatch.source_ranks == dest_r)
        count = int(mask.sum().item())
        if count == 0:
            continue
        idxs = mask.nonzero(as_tuple=False).squeeze(-1)
        start = dest_r * max_gather
        gather_send_buf[start:start + count] = packed[idxs]

    # 5b. Determine active ranks for gather (non-zero traffic)
    gather_active_ranks = sorted(set(
        r for r in range(world_size)
        if gather_send_counts[r].item() > 0 or gather_recv_counts[r].item() > 0
    ))

    # 6. All-to-all back
    gather_recv_buf = torch.zeros_like(gather_send_buf)
    handle = transport.all_to_all(
        gather_recv_buf, gather_send_buf, async_op=async_op,
        active_ranks=gather_active_ranks,
    )

    if async_op:
        result = _unpack_gather(gather_recv_buf, gather_recv_counts, max_gather, hidden_dim, output_size)
        return result, handle
    return _unpack_gather(gather_recv_buf, gather_recv_counts, max_gather, hidden_dim, output_size)


def _unpack_gather(
    recv_buf: torch.Tensor,
    recv_counts: torch.Tensor,
    max_chunk: int,
    hidden_dim: int,
    output_size: int,
) -> torch.Tensor:
    """Unpack gather all-to-all output into the combined tensor.

    For top_k=1: output_size == total_original_tokens, each index appears once.
    For top_k>1: output_size == total_original_tokens * top_k, indices repeat.
    Uses index_add_ so duplicate indices accumulate (needed for top_k>1).
    """
    out_chunks = []
    idx_chunks = []

    for src_r in range(len(recv_counts)):
        count = int(recv_counts[src_r].item())
        if count == 0:
            continue
        start = src_r * max_chunk
        chunk = recv_buf[start:start + count]
        out_chunks.append(chunk[:, :hidden_dim])
        idx_chunks.append(chunk[:, hidden_dim].long())

    if not out_chunks:
        return torch.zeros(output_size, hidden_dim)

    all_outputs = torch.cat(out_chunks, dim=0)
    all_indices = torch.cat(idx_chunks, dim=0)

    # index_add_ handles duplicate indices by accumulating (needed for top_k>1)
    combined = torch.zeros(output_size, hidden_dim)
    combined.index_add_(0, all_indices, all_outputs)
    return combined


def combine_expert_outputs(
    gathered: torch.Tensor,         # [T * top_k, H]  from gather_tokens
    gate_weights: torch.Tensor,     # [T, top_k]       from router
) -> torch.Tensor:
    """Weight and combine multi-expert outputs into final token representations.

    For top_k=1: gathered already has shape [T, H], gate_weights are all 1s,
                 so this is a no-op passthrough.
    For top_k=2: gathered has shape [T*2, H]. Reshape to [T, 2, H],
                 apply gate_weights [T, 2, 1], sum over expert dim -> [T, H].

    Args:
        gathered: raw output from gather_tokens, shape [T*K, H]
        gate_weights: routing weights from router, shape [T, K]

    Returns:
        Combined token representations [T, H]
    """
    if gate_weights.shape[1] == 1:
        # top_k=1: apply gate weight (router softmax probability) for gradient flow.
        # This provides a differentiable path from task loss → gate_weights → logits → gate.
        return gathered * gate_weights

    T = gate_weights.shape[0]
    K = gate_weights.shape[1]
    H = gathered.shape[1]

    # Reshape to [T, K, H] and apply weights
    reshaped = gathered.reshape(T, K, H)           # [T, K, H]
    weights = gate_weights.unsqueeze(-1)            # [T, K, 1]

    # Softmax-normalize weights across experts (standard MoE practice)
    weights = torch.softmax(weights, dim=1)

    combined = (reshaped * weights).sum(dim=1)      # [T, H]
    return combined

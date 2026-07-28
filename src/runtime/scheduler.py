"""Micro-batch scheduler: serial baseline, async overlap, and OCS-aware modes.

Serial mode:
  For each micro-batch i:
    route -> scatter (blocking) -> compute -> gather (blocking) -> combine

Overlap mode:
  Pipeline across micro-batches using double-buffering:
    Fire scatter for batch K
    While scatter is in-flight, compute expert for batch K-1
    Wait for scatter K, fire gather K
    While gather K is in-flight, fire scatter K+1 and compute K

OCS Pipeline mode (ocs_pipeline):
  Extends overlap with OCS circuit pre-establishment:
    route -> pre_establish_circuits -> scatter (async) -> ...
  Circuit reconfig cost is paid before scatter; hot circuits skip it.

OCS Dual-Batch Overlap (ocs_dbo):
  Pre-establishes circuits for micro-batch K+1 during batch K's compute,
  hiding reconfig cost behind computation. Requires >= 3 microbatches
  for full 3-deep pipeline effect.

OCS Preset (ocs_preset):
  Pre-establishes all OCS circuits from a training-derived plan BEFORE
  the first token is processed. No runtime reconfiguration occurs during
  inference — all circuits are hot from the start. Models the scenario
  where training-time affinity patterns are used to pre-configure OCS
  for inference.

Top-K support:
  For top_k > 1, the router returns expert_ids [T, K] and gate_weights [T, K].
  scatter_tokens flattens to [T*K] internally. combine_expert_outputs()
  applies gate weights and sums after gather.
"""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist

from src.model.moe_layer import MoELayer
from src.comm.transport import Transport
from src.comm.all_to_all import (
    scatter_tokens, gather_tokens, combine_expert_outputs, DispatchResult,
)
from src.train.loss import compute_loss
from src.utils.timer import Timer


def run_serial(
    step: int,
    microbatches: List[torch.Tensor],
    moe: MoELayer,
    transport: Transport,
    timer: Timer,
) -> None:
    """Baseline: comm -> compute -> comm -> combine, no overlap."""
    for mb_idx, tokens in enumerate(microbatches):
        # Route: get expert assignments and gate weights
        timer.start(f"step/{step}/mb_{mb_idx}/route")
        expert_ids, gate_weights, _logits = moe.router(tokens)
        timer.stop(f"step/{step}/mb_{mb_idx}/route")

        # Dispatch (blocking all-to-all) -- top-K flattening handled internally
        timer.start(f"step/{step}/mb_{mb_idx}/scatter")
        dispatch = scatter_tokens(
            tokens, expert_ids, moe.num_experts,
            moe.experts_per_rank, transport, async_op=False,
        )
        timer.stop(f"step/{step}/mb_{mb_idx}/scatter")

        # Compute -- dispatch to correct local expert
        timer.start(f"step/{step}/mb_{mb_idx}/compute")
        expert_out = moe.compute_experts(dispatch.tokens, dispatch.local_expert_ids)
        timer.stop(f"step/{step}/mb_{mb_idx}/compute")

        # Gather (blocking all-to-all) -- returns [T*K, H] for top-K
        timer.start(f"step/{step}/mb_{mb_idx}/gather")
        gathered = gather_tokens(expert_out, dispatch, transport, async_op=False)
        timer.stop(f"step/{step}/mb_{mb_idx}/gather")

        # Combine multi-expert outputs (no-op for top_k=1)
        timer.start(f"step/{step}/mb_{mb_idx}/combine")
        combined = combine_expert_outputs(gathered, gate_weights)
        timer.stop(f"step/{step}/mb_{mb_idx}/combine")


def run_overlap(
    step: int,
    microbatches: List[torch.Tensor],
    moe: MoELayer,
    transport: Transport,
    timer: Timer,
) -> None:
    """Overlap: pipeline scatter/compute/gather across micro-batches.

    For top_k > 1: combine is a cheap local operation done after gather
    returns, outside the critical comm path.

    Timeline (2 micro-batches):
      mb_0: [scatter_0 fire] --------------------+
            (scatter_0 in flight)                |
      mb_1: [scatter_1 fire]                     |
            [scatter_0 wait] <-------------------+
            [compute_0]
            [gather_0 fire] ---------------------+
            (gather_0 in flight)                 |
      tail: [scatter_1 wait]                     |
            [compute_1]                          |
            [gather_1 fire+wait] <---------------+
            [combine_0] [combine_1]              (cheap local ops)
    """
    num_mbs = len(microbatches)

    # Pre-route all micro-batches (record route time per mb)
    routes = []        # List[Tuple[expert_ids, gate_weights]]
    for mb_idx, tokens in enumerate(microbatches):
        timer.start(f"step/{step}/mb_{mb_idx}/route")
        expert_ids, gate_weights, _logits = moe.router(tokens)
        timer.stop(f"step/{step}/mb_{mb_idx}/route")
        routes.append((expert_ids, gate_weights))

    # -- Pipeline ----------------------------------------------------
    prev_dispatch = None   # DispatchResult from the previous micro-batch
    prev_gate_weights = None
    prev_out = None        # expert outputs from the previous micro-batch
    scatter_handle = None
    prev_gather_handle = None   # gather from TWO iterations ago
    gathered_prev = None   # raw gather output from 2-iterations-ago (for tail combine)

    for mb_idx in range(num_mbs):
        tokens = microbatches[mb_idx]
        expert_ids, gate_weights = routes[mb_idx]

        # 1. Wait for previous scatter (so we can read dispatch result safely)
        if scatter_handle is not None:
            timer.start(f"step/{step}/mb_{mb_idx-1}/scatter_wait")
            scatter_handle.wait()
            timer.stop(f"step/{step}/mb_{mb_idx-1}/scatter_wait")
            prev_dispatch = scatter_result

        # 2. Fire scatter for current micro-batch (async)
        timer.start(f"step/{step}/mb_{mb_idx}/scatter")
        scatter_result, scatter_handle = scatter_tokens(
            tokens, expert_ids, moe.num_experts,
            moe.experts_per_rank, transport, async_op=True,
        )
        timer.stop(f"step/{step}/mb_{mb_idx}/scatter")

        # 3. Wait for gather from mb_{idx-2} (it should be done by now)
        if prev_gather_handle is not None:
            timer.start(f"step/{step}/mb_{mb_idx-2}/gather_wait")
            prev_gather_handle.wait()
            timer.stop(f"step/{step}/mb_{mb_idx-2}/gather_wait")

        # 4. While scatter is in-flight, compute mb_{idx-1}
        if prev_dispatch is not None:
            timer.start(f"step/{step}/mb_{mb_idx-1}/compute")
            prev_out = moe.compute_experts(
                prev_dispatch.tokens, prev_dispatch.local_expert_ids,
            )
            timer.stop(f"step/{step}/mb_{mb_idx-1}/compute")

            # Fire gather for mb_{idx-1} (async)
            timer.start(f"step/{step}/mb_{mb_idx-1}/gather")
            gathered_prev, prev_gather_handle = gather_tokens(
                prev_out, prev_dispatch, transport, async_op=True,
            )
            timer.stop(f"step/{step}/mb_{mb_idx-1}/gather")

            # Combine for mb_{idx-1} (cheap local op, do it while comm is in-flight)
            timer.start(f"step/{step}/mb_{mb_idx-1}/combine")
            _combined = combine_expert_outputs(gathered_prev, prev_gate_weights)
            timer.stop(f"step/{step}/mb_{mb_idx-1}/combine")

        # Save gate_weights for next iteration's combine
        prev_gate_weights = gate_weights

    # -- Drain the pipeline tail ------------------------------------
    # Wait for final scatter (mb_{num_mbs-1})
    if scatter_handle is not None:
        last = num_mbs - 1
        timer.start(f"step/{step}/mb_{last}/scatter_wait")
        scatter_handle.wait()
        timer.stop(f"step/{step}/mb_{last}/scatter_wait")
        prev_dispatch = scatter_result

    # Wait for second-to-last gather (mb_{num_mbs-2})
    if prev_gather_handle is not None:
        second_last = num_mbs - 2
        timer.start(f"step/{step}/mb_{second_last}/gather_wait")
        prev_gather_handle.wait()
        timer.stop(f"step/{step}/mb_{second_last}/gather_wait")

    # Compute and gather last micro-batch
    if prev_dispatch is not None:
        last = num_mbs - 1
        timer.start(f"step/{step}/mb_{last}/compute")
        prev_out = moe.compute_experts(
            prev_dispatch.tokens, prev_dispatch.local_expert_ids,
        )
        timer.stop(f"step/{step}/mb_{last}/compute")

        # Gather last micro-batch (blocking -- no more work to overlap with)
        timer.start(f"step/{step}/mb_{last}/gather")
        gathered_last = gather_tokens(prev_out, prev_dispatch, transport, async_op=False)
        timer.stop(f"step/{step}/mb_{last}/gather")

        # Combine last micro-batch
        timer.start(f"step/{step}/mb_{last}/combine")
        combined = combine_expert_outputs(gathered_last, prev_gate_weights)
        timer.stop(f"step/{step}/mb_{last}/combine")


# ── Training schedulers ────────────────────────────────────────────────


def run_train_serial(
    step: int,
    microbatches: List[torch.Tensor],
    targets: List[torch.Tensor],
    moe: MoELayer,
    transport: Transport,
    optimizer: torch.optim.Optimizer,
    timer: Timer,
    alpha: float = 0.01,
) -> dict:
    """Training serial step: forward → loss → backward → optimizer.step().

    Returns a metrics dict with per-microbatch breakdown.
    """
    mb_metrics = []

    for mb_idx, (tokens, tgt) in enumerate(zip(microbatches, targets)):
        # ── Forward ──
        timer.start(f"step/{step}/mb_{mb_idx}/forward")
        output, logits, send_counts, _dispatch = moe.forward_train(tokens, transport)
        timer.stop(f"step/{step}/mb_{mb_idx}/forward")

        # ── Loss ──
        timer.start(f"step/{step}/mb_{mb_idx}/loss")
        loss, loss_task, loss_aux = compute_loss(
            output, tgt, logits, send_counts, moe.experts_per_rank, alpha,
        )
        timer.stop(f"step/{step}/mb_{mb_idx}/loss")

        # ── Backward + step ──
        timer.start(f"step/{step}/mb_{mb_idx}/backward")
        optimizer.zero_grad()
        loss.backward()
        # Sync router gradients across ranks (gate is shared, experts are not)
        sync_router_gradients(moe)
        optimizer.step()
        timer.stop(f"step/{step}/mb_{mb_idx}/backward")

        mb_metrics.append({
            "mb_idx": mb_idx,
            "loss_total": loss.item(),
            "loss_task": loss_task.item(),
            "loss_aux": loss_aux.item(),
        })

    return {
        "step": step,
        "microbatches": mb_metrics,
        "loss_total": sum(m["loss_total"] for m in mb_metrics) / len(mb_metrics),
        "loss_task": sum(m["loss_task"] for m in mb_metrics) / len(mb_metrics),
        "loss_aux": sum(m["loss_aux"] for m in mb_metrics) / len(mb_metrics),
    }


# ── Distributed training utilities ─────────────────────────────────────


def sync_router_gradients(moe: MoELayer) -> None:
    """All-reduce gradients for the router (shared across all ranks).

    Each rank computes different router gradients because it sees different
    expert outputs through all-to-all. Averaging ensures convergence.
    """
    world_size = dist.get_world_size()
    for param in moe.router.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
            param.grad /= world_size


def broadcast_model_params(moe: MoELayer, src: int = 0) -> None:
    """Broadcast all MoE parameters from src rank for identical initialization."""
    for param in moe.parameters():
        dist.broadcast(param.data, src=src)


# ── OCS-aware schedulers ────────────────────────────────────────────────


def _target_ranks_from_experts(expert_ids: torch.Tensor, experts_per_rank: int) -> list:
    """Derive target ranks for communication from expert assignments.

    Each expert_id maps to a target rank via: target_rank = expert_id // experts_per_rank.
    Returns a deduplicated list of rank IDs that this micro-batch will communicate with.
    """
    ids = expert_ids.reshape(-1)
    return list(set((ids // experts_per_rank).tolist()))


def run_ocs_pipeline(
    step: int,
    microbatches: List[torch.Tensor],
    moe: MoELayer,
    transport: Transport,
    timer: Timer,
) -> None:
    """OCS-aware overlap: pre-establish circuits before scatter.

    Extends run_overlap by calling transport.pre_establish_circuits()
    after routing each micro-batch. Circuit reconfig cost (if any) is paid
    before the scatter fires, but subsequent micro-batches benefit from
    already-established circuits.

    Pipeline per micro-batch:
      route -> pre_establish_circuits -> scatter (async) -> ...
      (same compute/gather interleaving as run_overlap)

    Key difference from run_overlap: the first micro-batch pays any cold-circuit
    reconfig cost inline. Subsequent batches reuse hot circuits at OCS fast-path
    latency and bandwidth.
    """
    num_mbs = len(microbatches)

    # Pre-route all micro-batches (same as run_overlap)
    routes: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for mb_idx, tokens in enumerate(microbatches):
        timer.start(f"step/{step}/mb_{mb_idx}/route")
        expert_ids, gate_weights, _logits = moe.router(tokens)
        timer.stop(f"step/{step}/mb_{mb_idx}/route")
        routes.append((expert_ids, gate_weights))

    # -- Pipeline ----------------------------------------------------
    prev_dispatch = None
    prev_gate_weights = None
    prev_out = None
    scatter_handle = None
    scatter_result = None
    prev_gather_handle = None
    gathered_prev = None

    for mb_idx in range(num_mbs):
        tokens = microbatches[mb_idx]
        expert_ids, gate_weights = routes[mb_idx]

        # --- OCS: pre-establish circuits for THIS micro-batch's targets ---
        timer.start(f"step/{step}/mb_{mb_idx}/ocs_pre_establish")
        target_ranks = _target_ranks_from_experts(expert_ids, moe.experts_per_rank)
        transport.pre_establish_circuits(target_ranks)
        timer.stop(f"step/{step}/mb_{mb_idx}/ocs_pre_establish")

        # 1. Wait for previous scatter (so we can read dispatch result safely)
        if scatter_handle is not None:
            timer.start(f"step/{step}/mb_{mb_idx-1}/scatter_wait")
            scatter_handle.wait()
            timer.stop(f"step/{step}/mb_{mb_idx-1}/scatter_wait")
            prev_dispatch = scatter_result

        # 2. Fire scatter for current micro-batch (async) — circuits already established
        timer.start(f"step/{step}/mb_{mb_idx}/scatter")
        scatter_result, scatter_handle = scatter_tokens(
            tokens, expert_ids, moe.num_experts,
            moe.experts_per_rank, transport, async_op=True,
        )
        timer.stop(f"step/{step}/mb_{mb_idx}/scatter")

        # 3. Wait for gather from mb_{idx-2} (it should be done by now)
        if prev_gather_handle is not None:
            timer.start(f"step/{step}/mb_{mb_idx-2}/gather_wait")
            prev_gather_handle.wait()
            timer.stop(f"step/{step}/mb_{mb_idx-2}/gather_wait")

        # 4. While scatter is in-flight, compute mb_{idx-1}
        if prev_dispatch is not None:
            timer.start(f"step/{step}/mb_{mb_idx-1}/compute")
            prev_out = moe.compute_experts(
                prev_dispatch.tokens, prev_dispatch.local_expert_ids,
            )
            timer.stop(f"step/{step}/mb_{mb_idx-1}/compute")

            # Fire gather for mb_{idx-1} (async)
            timer.start(f"step/{step}/mb_{mb_idx-1}/gather")
            gathered_prev, prev_gather_handle = gather_tokens(
                prev_out, prev_dispatch, transport, async_op=True,
            )
            timer.stop(f"step/{step}/mb_{mb_idx-1}/gather")

            # Combine for mb_{idx-1} (cheap local op, do it while comm is in-flight)
            timer.start(f"step/{step}/mb_{mb_idx-1}/combine")
            _combined = combine_expert_outputs(gathered_prev, prev_gate_weights)
            timer.stop(f"step/{step}/mb_{mb_idx-1}/combine")

        # Save gate_weights for next iteration's combine
        prev_gate_weights = gate_weights

    # -- Drain the pipeline tail ------------------------------------
    if scatter_handle is not None:
        last = num_mbs - 1
        timer.start(f"step/{step}/mb_{last}/scatter_wait")
        scatter_handle.wait()
        timer.stop(f"step/{step}/mb_{last}/scatter_wait")
        prev_dispatch = scatter_result

    if prev_gather_handle is not None:
        second_last = num_mbs - 2
        timer.start(f"step/{step}/mb_{second_last}/gather_wait")
        prev_gather_handle.wait()
        timer.stop(f"step/{step}/mb_{second_last}/gather_wait")

    if prev_dispatch is not None:
        last = num_mbs - 1
        timer.start(f"step/{step}/mb_{last}/compute")
        prev_out = moe.compute_experts(
            prev_dispatch.tokens, prev_dispatch.local_expert_ids,
        )
        timer.stop(f"step/{step}/mb_{last}/compute")

        timer.start(f"step/{step}/mb_{last}/gather")
        gathered_last = gather_tokens(prev_out, prev_dispatch, transport, async_op=False)
        timer.stop(f"step/{step}/mb_{last}/gather")

        timer.start(f"step/{step}/mb_{last}/combine")
        combined = combine_expert_outputs(gathered_last, prev_gate_weights)
        timer.stop(f"step/{step}/mb_{last}/combine")


def run_ocs_dbo(
    step: int,
    microbatches: List[torch.Tensor],
    moe: MoELayer,
    transport: Transport,
    timer: Timer,
) -> None:
    """Dual-Batch Overlap on OCS: pre-establish circuits with lookahead.

    Extends run_ocs_pipeline by pre-establishing circuits for micro-batch K+1
    during the pipeline execution for micro-batch K. This hides reconfig cost
    behind the compute of the previous batch.

    Three-stage pipeline (with >= 3 microbatches):
      Batch K-2: combine                        (tail)
      Batch K-1: compute + gather               (active compute)
      Batch K:   scatter                        (in-flight comm)
      Batch K+1: pre_establish_circuits         (reconfig hidden by compute)

    For 2 microbatches, this degenerates to ocs_pipeline behavior.
    A log warning is emitted if num_microbatches < 3.
    """
    num_mbs = len(microbatches)

    # Emit warning for small micro-batch count
    if num_mbs < 3:
        import sys
        print(
            f"[ocs_dbo] WARNING: {num_mbs} microbatches — "
            f"DBO needs >= 3 for full 3-deep pipeline benefit. "
            f"Degenerating to ocs_pipeline-like behavior.",
            file=sys.stderr,
        )

    # Pre-route all micro-batches
    routes: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for mb_idx, tokens in enumerate(microbatches):
        timer.start(f"step/{step}/mb_{mb_idx}/route")
        expert_ids, gate_weights, _logits = moe.router(tokens)
        timer.stop(f"step/{step}/mb_{mb_idx}/route")
        routes.append((expert_ids, gate_weights))

    # -- Pipeline ----------------------------------------------------
    prev_dispatch = None
    prev_gate_weights = None
    prev_out = None
    scatter_handle = None
    scatter_result = None
    prev_gather_handle = None
    gathered_prev = None
    pre_established: set = set()  # track which mb indices have had circuits pre-established

    for mb_idx in range(num_mbs):
        tokens = microbatches[mb_idx]
        expert_ids, gate_weights = routes[mb_idx]

        # --- OCS DBO: pre-establish circuits for batch K+1 during batch K ---
        next_mb = mb_idx + 1
        if next_mb < num_mbs and next_mb not in pre_established:
            timer.start(f"step/{step}/mb_{next_mb}/ocs_pre_establish")
            next_expert_ids, _next_gate_weights = routes[next_mb]
            next_targets = _target_ranks_from_experts(
                next_expert_ids, moe.experts_per_rank,
            )
            transport.pre_establish_circuits(next_targets)
            timer.stop(f"step/{step}/mb_{next_mb}/ocs_pre_establish")
            pre_established.add(next_mb)

        # 1. Wait for previous scatter
        if scatter_handle is not None:
            timer.start(f"step/{step}/mb_{mb_idx-1}/scatter_wait")
            scatter_handle.wait()
            timer.stop(f"step/{step}/mb_{mb_idx-1}/scatter_wait")
            prev_dispatch = scatter_result

        # 2. Fire scatter for current micro-batch (async)
        timer.start(f"step/{step}/mb_{mb_idx}/scatter")
        scatter_result, scatter_handle = scatter_tokens(
            tokens, expert_ids, moe.num_experts,
            moe.experts_per_rank, transport, async_op=True,
        )
        timer.stop(f"step/{step}/mb_{mb_idx}/scatter")

        # 3. Wait for gather from mb_{idx-2}
        if prev_gather_handle is not None:
            timer.start(f"step/{step}/mb_{mb_idx-2}/gather_wait")
            prev_gather_handle.wait()
            timer.stop(f"step/{step}/mb_{mb_idx-2}/gather_wait")

        # 4. Compute mb_{idx-1} while scatter is in-flight
        if prev_dispatch is not None:
            timer.start(f"step/{step}/mb_{mb_idx-1}/compute")
            prev_out = moe.compute_experts(
                prev_dispatch.tokens, prev_dispatch.local_expert_ids,
            )
            timer.stop(f"step/{step}/mb_{mb_idx-1}/compute")

            # Fire gather for mb_{idx-1} (async)
            timer.start(f"step/{step}/mb_{mb_idx-1}/gather")
            gathered_prev, prev_gather_handle = gather_tokens(
                prev_out, prev_dispatch, transport, async_op=True,
            )
            timer.stop(f"step/{step}/mb_{mb_idx-1}/gather")

            # Combine for mb_{idx-1}
            timer.start(f"step/{step}/mb_{mb_idx-1}/combine")
            _combined = combine_expert_outputs(gathered_prev, prev_gate_weights)
            timer.stop(f"step/{step}/mb_{mb_idx-1}/combine")

        # Save gate_weights for next iteration's combine
        prev_gate_weights = gate_weights

    # -- Drain the pipeline tail ------------------------------------
    if scatter_handle is not None:
        last = num_mbs - 1
        timer.start(f"step/{step}/mb_{last}/scatter_wait")
        scatter_handle.wait()
        timer.stop(f"step/{step}/mb_{last}/scatter_wait")
        prev_dispatch = scatter_result

    if prev_gather_handle is not None:
        second_last = num_mbs - 2
        timer.start(f"step/{step}/mb_{second_last}/gather_wait")
        prev_gather_handle.wait()
        timer.stop(f"step/{step}/mb_{second_last}/gather_wait")

    if prev_dispatch is not None:
        last = num_mbs - 1
        timer.start(f"step/{step}/mb_{last}/compute")
        prev_out = moe.compute_experts(
            prev_dispatch.tokens, prev_dispatch.local_expert_ids,
        )
        timer.stop(f"step/{step}/mb_{last}/compute")

        timer.start(f"step/{step}/mb_{last}/gather")
        gathered_last = gather_tokens(prev_out, prev_dispatch, transport, async_op=False)
        timer.stop(f"step/{step}/mb_{last}/gather")

        timer.start(f"step/{step}/mb_{last}/combine")
        combined = combine_expert_outputs(gathered_last, prev_gate_weights)
        timer.stop(f"step/{step}/mb_{last}/combine")
    # --- end run_ocs_dbo ---


def run_ocs_preset(
    step: int,
    microbatches: List[torch.Tensor],
    moe: MoELayer,
    transport: Transport,
    timer: Timer,
) -> None:
    """OCS Preset: pre-established circuits, zero runtime reconfig.

    Circuits are pre-established from a training-derived placement plan
    BEFORE the first micro-batch. During inference, all cross-rank
    communication benefits from hot OCS circuits — no reconfig delay.
    """
    num_mbs = len(microbatches)

    routes = []
    for mb_idx, tokens in enumerate(microbatches):
        timer.start(f"step/{step}/mb_{mb_idx}/route")
        expert_ids, gate_weights, _logits = moe.router(tokens)
        timer.stop(f"step/{step}/mb_{mb_idx}/route")
        routes.append((expert_ids, gate_weights))

    prev_dispatch = None
    prev_gate_weights = None
    prev_out = None
    scatter_handle = None
    scatter_result = None
    prev_gather_handle = None
    gathered_prev = None

    for mb_idx in range(num_mbs):
        tokens = microbatches[mb_idx]
        expert_ids, gate_weights = routes[mb_idx]

        if scatter_handle is not None:
            timer.start(f"step/{step}/mb_{mb_idx-1}/scatter_wait")
            scatter_handle.wait()
            timer.stop(f"step/{step}/mb_{mb_idx-1}/scatter_wait")
            prev_dispatch = scatter_result

        timer.start(f"step/{step}/mb_{mb_idx}/scatter")
        scatter_result, scatter_handle = scatter_tokens(
            tokens, expert_ids, moe.num_experts,
            moe.experts_per_rank, transport, async_op=True,
        )
        timer.stop(f"step/{step}/mb_{mb_idx}/scatter")

        if prev_gather_handle is not None:
            timer.start(f"step/{step}/mb_{mb_idx-2}/gather_wait")
            prev_gather_handle.wait()
            timer.stop(f"step/{step}/mb_{mb_idx-2}/gather_wait")

        if prev_dispatch is not None:
            timer.start(f"step/{step}/mb_{mb_idx-1}/compute")
            prev_out = moe.compute_experts(
                prev_dispatch.tokens, prev_dispatch.local_expert_ids,
            )
            timer.stop(f"step/{step}/mb_{mb_idx-1}/compute")

            timer.start(f"step/{step}/mb_{mb_idx-1}/gather")
            gathered_prev, prev_gather_handle = gather_tokens(
                prev_out, prev_dispatch, transport, async_op=True,
            )
            timer.stop(f"step/{step}/mb_{mb_idx-1}/gather")

            timer.start(f"step/{step}/mb_{mb_idx-1}/combine")
            _combined = combine_expert_outputs(gathered_prev, prev_gate_weights)
            timer.stop(f"step/{step}/mb_{mb_idx-1}/combine")

        prev_gate_weights = gate_weights

    if scatter_handle is not None:
        last = num_mbs - 1
        timer.start(f"step/{step}/mb_{last}/scatter_wait")
        scatter_handle.wait()
        timer.stop(f"step/{step}/mb_{last}/scatter_wait")
        prev_dispatch = scatter_result

    if prev_gather_handle is not None:
        second_last = num_mbs - 2
        timer.start(f"step/{step}/mb_{second_last}/gather_wait")
        prev_gather_handle.wait()
        timer.stop(f"step/{step}/mb_{second_last}/gather_wait")

    if prev_dispatch is not None:
        last = num_mbs - 1
        timer.start(f"step/{step}/mb_{last}/compute")
        prev_out = moe.compute_experts(
            prev_dispatch.tokens, prev_dispatch.local_expert_ids,
        )
        timer.stop(f"step/{step}/mb_{last}/compute")

        timer.start(f"step/{step}/mb_{last}/gather")
        gathered_last = gather_tokens(prev_out, prev_dispatch, transport, async_op=False)
        timer.stop(f"step/{step}/mb_{last}/gather")

        timer.start(f"step/{step}/mb_{last}/combine")
        _combined = combine_expert_outputs(gathered_last, prev_gate_weights)
        timer.stop(f"step/{step}/mb_{last}/combine")
    # --- end run_ocs_preset ---

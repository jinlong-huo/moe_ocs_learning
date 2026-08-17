"""Pre-configuration pipeline: training trace → affinity → circuit plan.

This module implements the end-to-end pipeline for mapping training-time
routing patterns to inference-time OCS circuit pre-configuration.

Pipeline stages:
  1. Load routing trace (from training or replay capture)
  2. Build expert co-activation matrix across all layers
  3. Compute circuit placement plan → ordered rank-pair list
  4. Export plan as JSON for consumption by scheduler

Supports multiple strategies for converting affinity to a circuit plan:
  - coactivation: aggregate pairwise expert co-selection counts
  - volume: weight by token volume per expert pair
  - frequency: weight by how often each expert pair communicates

All strategies produce the same output format: a list of (src, dst, score).
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple

import torch

from src.data.routing_schema import RoutingTrace
from src.ocs.placement import ExpertAffinityTracker


def _build_affinity_from_trace(
    trace: RoutingTrace,
    num_experts: int,
) -> ExpertAffinityTracker:
    """Build an ExpertAffinityTracker from a routing trace.

    Iterates over all routes (tokens) and all MoE layers in the trace
    to accumulate global co-activation counts.

    If ``trace.guide_affinity`` is available, it is used to weight
    co-activations by the dense guide model's semantic centrality —
    tokens that the generalised model considers more central to the
    context contribute more to the affinity graph.

    Args:
        trace: routing trace (can be from training or inference).
        num_experts: total number of experts (for affinity matrix sizing).

    Returns:
        ExpertAffinityTracker with accumulated co-activation counts.
    """
    # ── Compute per-token centrality from guide affinity ──
    token_centrality: dict[int, float] = {}
    guide_affinity = getattr(trace, "guide_affinity", None)
    if guide_affinity is not None and len(guide_affinity) > 0:
        n_tokens = len(guide_affinity)
        for t in range(n_tokens):
            row = guide_affinity[t]
            if row and len(row) == n_tokens:
                # Mean cosine similarity of token t to all others
                token_centrality[t] = sum(row) / n_tokens
        if token_centrality:
            vals = list(token_centrality.values())
            print(f"[affinity] Guide model weights: {len(token_centrality)} tokens, "
                  f"centrality range [{min(vals):.4f}, {max(vals):.4f}]")

    if num_experts <= 0:
        raise ValueError("num_experts must be positive for affinity computation")

    if trace.meta.num_experts not in (0, num_experts):
        raise ValueError(
            f"Trace '{trace.meta.model_id}' reports num_experts="
            f"{trace.meta.num_experts}, but the pipeline is configured for "
            f"num_experts={num_experts} (world_size x experts_per_rank). "
            "Use a matching trace or fix the experiment config — expert ids "
            "outside the configured range would otherwise be dropped."
        )

    tracker = ExpertAffinityTracker(num_experts)

    for route in trace.routes:
        layers = route.layers
        if not layers:
            continue

        for layer_id, layer_data in layers.items():
            expert_ids_list = layer_data.experts
            weights_list = layer_data.weights if layer_data.weights else []

            token_count = len(expert_ids_list)
            if token_count == 0:
                continue

            # ── Token centrality weight from guide affinity ──
            tw = None
            if token_centrality:
                pos = route.token_pos
                c = token_centrality.get(pos)
                if c is not None:
                    tw = torch.tensor([c], dtype=torch.float64)

            # expert_ids_list is [K] or list of [K] per-token, depending on format
            # Handle both [K] (single token) and [[K], ...] (multi-token)
            if expert_ids_list and isinstance(expert_ids_list[0], (int, float)):
                expert_ids = torch.tensor([expert_ids_list], dtype=torch.long)
                if weights_list:
                    gate_weights = torch.tensor([weights_list], dtype=torch.float32)
                else:
                    gate_weights = torch.ones(1, dtype=torch.float32)
            else:
                expert_ids = torch.tensor(expert_ids_list, dtype=torch.long)
                if weights_list:
                    gate_weights = torch.tensor(weights_list, dtype=torch.float32)
                else:
                    gate_weights = torch.ones(token_count, dtype=torch.float32)

            if expert_ids.numel() > 0 and (
                expert_ids.min().item() < 0 or expert_ids.max().item() >= num_experts
            ):
                raise ValueError(
                    f"Trace '{trace.meta.model_id}' layer {layer_id} contains "
                    f"expert ids outside [0, {num_experts}). Trace meta says "
                    f"num_experts={trace.meta.num_experts}; the pipeline is "
                    f"configured for {num_experts}."
                )

            tracker.record_routing(expert_ids, gate_weights, token_weights=tw)

    return tracker


def _check_rank_layout(num_experts: int, world_size: int, experts_per_rank: int) -> None:
    """Fail loudly when the expert→rank mapping is inconsistent.

    The expert_id → rank mapping is ``e // experts_per_rank``; a mismatch
    between the trace's expert count and the experiment's world_size x
    experts_per_rank silently corrupts every rank assignment downstream.
    """
    if experts_per_rank <= 0:
        raise ValueError(f"experts_per_rank must be positive, got {experts_per_rank}")
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if world_size * experts_per_rank != num_experts:
        raise ValueError(
            f"Rank layout mismatch: world_size ({world_size}) x "
            f"experts_per_rank ({experts_per_rank}) = "
            f"{world_size * experts_per_rank}, but the trace has "
            f"num_experts={num_experts}. Update the experiment config so the "
            "product equals the trace's expert count."
        )


def compute_plan_from_trace(
    trace_path: str,
    max_circuits: int = 16,
    experts_per_rank: int = 1,
    world_size: int = 4,
    strategy: str = "coactivation",
) -> List[Tuple[int, int, float]]:
    """Compute OCS circuit placement plan from a routing trace.

    High-level entry point: trace → affinity → circuit plan.

    Args:
        trace_path: path to routing trace JSON (from training or replay).
        max_circuits: maximum circuits to include in the plan.
        experts_per_rank: experts per GPU rank.
        world_size: number of GPU ranks.
        strategy: plan computation strategy (currently "coactivation" only).

    Returns:
        List of (src_rank, dst_rank, score) sorted by score descending.

    Raises:
        ValueError: if the rank layout does not match the trace's num_experts.
    """
    trace = RoutingTrace.load(trace_path)
    num_experts = trace.meta.num_experts
    _check_rank_layout(num_experts, world_size, experts_per_rank)

    tracker = _build_affinity_from_trace(trace, num_experts)

    expert_to_rank = {
        e: e // experts_per_rank for e in range(num_experts)
    }

    plan = tracker.compute_circuit_plan(
        expert_to_rank=expert_to_rank,
        max_circuits=max_circuits,
        experts_per_rank=experts_per_rank,
        world_size=world_size,
    )
    return plan


def compute_plan_from_traces(
    trace_paths: List[str],
    max_circuits: int = 16,
    experts_per_rank: int = 1,
    world_size: int = 4,
    strategy: str = "coactivation",
) -> List[Tuple[int, int, float]]:
    """Compute OCS circuit placement plan from MULTIPLE routing traces.

    Merges all training traces into a single ExpertAffinityTracker, then
    computes a circuit plan from the aggregated co-activation patterns.
    This is the key function for the "network input problem" — the plan
    is built from diverse training inputs and tested on held-out inputs.

    Args:
        trace_paths: list of paths to routing trace JSONs (training set).
        max_circuits: maximum circuits to include in the plan.
        experts_per_rank: experts per GPU rank.
        world_size: number of GPU ranks.
        strategy: plan computation strategy (currently "coactivation" only).

    Returns:
        List of (src_rank, dst_rank, score) sorted by score descending.

    Raises:
        ValueError: if trace_paths is empty, traces have inconsistent num_experts,
            or the rank layout does not match the traces' num_experts.
    """
    if not trace_paths:
        raise ValueError("trace_paths must not be empty")

    # Load first trace to determine num_experts
    first_trace = RoutingTrace.load(trace_paths[0])
    num_experts = first_trace.meta.num_experts
    _check_rank_layout(num_experts, world_size, experts_per_rank)

    # Build one tracker and feed all traces through it
    tracker = _build_affinity_from_trace(first_trace, num_experts)

    for path in trace_paths[1:]:
        trace = RoutingTrace.load(path)
        if trace.meta.num_experts != num_experts:
            raise ValueError(
                f"Inconsistent num_experts: {trace_paths[0]} has "
                f"{num_experts}, but {path} has {trace.meta.num_experts}"
            )
        # Compute token centrality from guide affinity for this trace
        token_cem: dict[int, float] = {}
        guide_aff = getattr(trace, "guide_affinity", None)
        if guide_aff is not None and len(guide_aff) > 0:
            n_tok = len(guide_aff)
            for t in range(n_tok):
                row = guide_aff[t]
                if row and len(row) == n_tok:
                    token_cem[t] = sum(row) / n_tok

        # Manually feed routing events into the existing tracker
        for route in trace.routes:
            tw = None
            if token_cem:
                c = token_cem.get(route.token_pos)
                if c is not None:
                    tw = torch.tensor([c], dtype=torch.float64)
            for layer_data in route.layers.values():
                expert_ids_list = layer_data.experts
                weights_list = layer_data.weights if layer_data.weights else []
                if not expert_ids_list:
                    continue
                if isinstance(expert_ids_list[0], (int, float)):
                    expert_ids = torch.tensor([expert_ids_list], dtype=torch.long)
                    gate_weights = (
                        torch.tensor([weights_list], dtype=torch.float32)
                        if weights_list else torch.ones(1, dtype=torch.float32)
                    )
                else:
                    expert_ids = torch.tensor(expert_ids_list, dtype=torch.long)
                    gate_weights = (
                        torch.tensor(weights_list, dtype=torch.float32)
                        if weights_list
                        else torch.ones(len(expert_ids_list), dtype=torch.float32)
                    )
                tracker.record_routing(expert_ids, gate_weights, token_weights=tw)

    expert_to_rank = {
        e: e // experts_per_rank for e in range(num_experts)
    }

    plan = tracker.compute_circuit_plan(
        expert_to_rank=expert_to_rank,
        max_circuits=max_circuits,
        experts_per_rank=experts_per_rank,
        world_size=world_size,
    )
    return plan


def export_plan(plan: List[Tuple[int, int, float]], path: str) -> None:
    """Export a circuit placement plan to JSON.

    Format:
        {"circuits": [[src, dst, score], ...], "num_circuits": N}
    """
    data = {
        "circuits": [[int(src), int(dst), float(score)] for src, dst, score in plan],
        "num_circuits": len(plan),
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_plan(path: str) -> List[Tuple[int, int, float]]:
    """Load a circuit placement plan from JSON.

    Returns:
        List of (src_rank, dst_rank, score) tuples.
    """
    with open(path) as f:
        data = json.load(f)
    return [
        (int(src), int(dst), float(score))
        for src, dst, score in data.get("circuits", [])
    ]


def plan_summary(plan: List[Tuple[int, int, float]]) -> Dict:
    """Produce a human-readable summary of a circuit plan.

    Returns dict with:
      - num_circuits: total circuits in plan
      - top_pairs: top-5 (src, dst, score) entries
      - score_stats: min, max, mean, median of scores
      - rank_counts: how many circuits per rank (outgoing)
    """
    if not plan:
        return {"num_circuits": 0}

    scores = [s for _, _, s in plan]
    scores_sorted = sorted(scores)
    n = len(scores)
    median = scores_sorted[n // 2] if n > 0 else 0.0

    rank_counts: Dict[int, int] = {}
    for src, dst, _ in plan:
        rank_counts[src] = rank_counts.get(src, 0) + 1

    return {
        "num_circuits": len(plan),
        "top_pairs": [
            {"src": src, "dst": dst, "score": round(score, 4)}
            for src, dst, score in plan[:5]
        ],
        "score_stats": {
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
            "mean": round(sum(scores) / n, 4) if n > 0 else 0.0,
            "median": round(median, 4),
        },
        "rank_counts": rank_counts,
    }

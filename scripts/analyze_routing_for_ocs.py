#!/usr/bin/env python3
"""
analyze_routing_for_ocs.py — Analyze routing traces for OCS circuit placement.

Extracts from captured RoutingTraces:
  - Rank communication heatmap (which rank pairs communicate most)
  - Per-layer target rank sets (for OCS pre-establishment scheduling)
  - Expert co-activation patterns (for affinity-based placement)
  - Circuit reuse potential estimation

Usage:
  python scripts/analyze_routing_for_ocs.py data/routing_traces/routing.json \\
      --experts-per-rank 4 --output ocs_analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.routing_schema import RoutingTrace


def analyze(trace: RoutingTrace, experts_per_rank: int) -> dict:
    """Extract OCS-relevant statistics from a routing trace."""

    def _rank_of(expert_id: int) -> int:
        return expert_id // experts_per_rank

    num_ranks = trace.meta.num_experts // experts_per_rank
    if trace.meta.num_experts % experts_per_rank != 0:
        print(f"WARNING: num_experts={trace.meta.num_experts} not divisible by "
              f"experts_per_rank={experts_per_rank}", file=sys.stderr)

    # 1. Per-layer target ranks per token
    per_layer_targets: dict[str, list[list[int]]] = {}
    for route in trace.routes:
        for lid, lr in route.layers.items():
            if lid not in per_layer_targets:
                per_layer_targets[lid] = []
            per_layer_targets[lid].append(
                sorted(set(_rank_of(e) for e in lr.experts))
            )

    # 2. Rank communication matrix
    comm_matrix: dict[str, int] = {}  # "src->dst" → count
    for route in trace.routes:
        for lr in route.layers.values():
            experts = lr.experts
            for i, src_e in enumerate(experts):
                src_r = _rank_of(src_e)
                for j, dst_e in enumerate(experts):
                    if i == j:
                        continue
                    dst_r = _rank_of(dst_e)
                    key = f"{src_r}->{dst_r}"
                    comm_matrix[key] = comm_matrix.get(key, 0) + 1

    # 3. Hot rank pairs (top by communication frequency)
    hot_pairs = sorted(
        comm_matrix.items(), key=lambda x: -x[1]
    )[:20]

    # 4. Expert co-activation
    coactivation: dict[str, int] = {}
    for route in trace.routes:
        for lr in route.layers.values():
            pairs = lr.experts
            for i in range(len(pairs)):
                for j in range(i + 1, len(pairs)):
                    key = f"e{pairs[i]}_e{pairs[j]}"
                    coactivation[key] = coactivation.get(key, 0) + 1

    top_coactivations = sorted(
        coactivation.items(), key=lambda x: -x[1]
    )[:20]

    # 5. Circuit reuse potential
    #    = unique rank pairs / total communication events
    unique_pairs = len(comm_matrix)
    total_events = sum(comm_matrix.values())
    reuse_potential = 1.0 - (unique_pairs / max(total_events, 1))

    # 6. Per-layer unique target sets
    per_layer_unique_sets: dict[str, int] = {}
    for lid, targets in per_layer_targets.items():
        unique = len(set(tuple(t) for t in targets))
        per_layer_unique_sets[lid] = unique

    return {
        "meta": {
            "model_id": trace.meta.model_id,
            "num_experts": trace.meta.num_experts,
            "top_k": trace.meta.top_k,
            "num_moe_layers": trace.meta.num_moe_layers,
            "total_tokens": trace.meta.total_tokens,
            "experts_per_rank": experts_per_rank,
            "num_ranks": num_ranks,
        },
        "per_layer_target_stats": {
            lid: {
                "total_tokens": len(targets),
                "unique_target_sets": per_layer_unique_sets[lid],
                "avg_target_ranks": sum(len(t) for t in targets) / max(len(targets), 1),
                "repetition_ratio": 1.0 - per_layer_unique_sets[lid] / max(len(targets), 1),
            }
            for lid, targets in per_layer_targets.items()
        },
        "rank_communication": {
            "unique_pairs": unique_pairs,
            "total_events": total_events,
            "circuit_reuse_potential": round(reuse_potential, 4),
            "hot_pairs": [{"pair": k, "count": v} for k, v in hot_pairs],
        },
        "expert_coactivation": {
            "total_pairs": len(coactivation),
            "top_pairs": [{"pair": k, "count": v} for k, v in top_coactivations],
        },
        "ocs_recommendations": _make_recommendations(
            trace, experts_per_rank, comm_matrix, per_layer_targets
        ),
    }


def _make_recommendations(
    trace: RoutingTrace,
    experts_per_rank: int,
    comm_matrix: dict[str, int],
    per_layer_targets: dict[str, list[list[int]]],
) -> dict:
    """Generate OCS configuration recommendations from trace analysis."""

    num_ranks = trace.meta.num_experts // experts_per_rank

    # How many circuits needed to cover 95% of communications
    total = sum(comm_matrix.values())
    cumulative = 0.0
    circuits_for_95pct = 0
    for _, count in sorted(comm_matrix.items(), key=lambda x: -x[1]):
        cumulative += count
        circuits_for_95pct += 1
        if cumulative / total >= 0.95:
            break

    # Per-layer predictability: what fraction of tokens repeat the same target set
    highest_repeat_layer = None
    highest_repeat_ratio = 0.0
    for lid, targets in per_layer_targets.items():
        unique = len(set(tuple(t) for t in targets))
        ratio = 1.0 - unique / max(len(targets), 1)
        if ratio > highest_repeat_ratio:
            highest_repeat_ratio = ratio
            highest_repeat_layer = lid

    return {
        "recommended_circuits_for_95pct_coverage": circuits_for_95pct,
        "max_meaningful_circuits_per_rank": min(circuits_for_95pct, num_ranks),
        "circuit_pool_recommendation": "round_robin"
        if highest_repeat_ratio < 0.5
        else "affinity",
        "most_predictable_layer": {
            "layer": highest_repeat_layer,
            "target_set_repeat_ratio": round(highest_repeat_ratio, 4),
        },
        "dbo_viability": "recommended"
        if highest_repeat_ratio > 0.6
        else "possible"
        if highest_repeat_ratio > 0.3
        else "marginal",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze routing trace for OCS circuit placement"
    )
    parser.add_argument("trace", help="Path to RoutingTrace JSON file")
    parser.add_argument(
        "--experts-per-rank", type=int, default=8,
        help="Experts per rank for rank mapping (default: 8)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output JSON file (default: prints to stdout)"
    )
    args = parser.parse_args()

    trace = RoutingTrace.load(args.trace)
    result = analyze(trace, args.experts_per_rank)

    output_text = json.dumps(result, indent=2, default=str)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            f.write(output_text)
        print(f"Analysis saved → {args.output}")
    else:
        print(output_text)


if __name__ == "__main__":
    main()

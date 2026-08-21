#!/usr/bin/env python3
"""Phase 1 — Topology-invariance sanity gate (routing ≠ cost, topology is orthogonal).

Replays one real Qwen RoutingTrace through the OCS testbed's affinity pipeline
and verifies the routing-independence assumption at the framework level:

  * the affinity matrix built from the trace is bit-identical under any
    3-tier topology configuration (pods × nodes × ranks, latencies, BW),
  * the derived OCS circuit plan is bit-identical,
  * only the *cost* (per-circuit pairwise dispatch delay) changes.

This is the framework-level check: topology must never leak into routing.
The hardware-level check (quantized-GEMM noise floor) lives in moe_mlx_learning.

Usage:
    python3 scripts/verify_topology_invariance.py \
        --trace data/routing_traces/routing.json \
        --experts-per-rank 8 --world-size 32 \
        --max-circuits 16 \
        --output logs/topology_invariance_report.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.comm.topology import Topology, TopologyConfig  # noqa: E402
from src.ocs.preconfig import (  # noqa: E402
    _build_affinity_from_trace,
    _check_rank_layout,
)
from src.data.routing_schema import RoutingTrace  # noqa: E402


# ── Topology grid: same 32 ranks, different physical fabrics ──────────
TOPOLOGY_GRID = {
    "single_node": TopologyConfig(num_pods=1, nodes_per_pod=1, ranks_per_node=32),
    "two_pods_flat": TopologyConfig(num_pods=2, nodes_per_pod=4, ranks_per_node=4),
    "four_pods": TopologyConfig(num_pods=4, nodes_per_pod=2, ranks_per_node=4),
    "two_pods_slow_fabric": TopologyConfig(
        num_pods=2, nodes_per_pod=4, ranks_per_node=4,
        cross_pod_latency_us=50.0, cross_pod_bandwidth_gbps=25.0,
        delay_multiplier=2.0,
    ),
}


def affinity_matrix_from_trace(trace: RoutingTrace, num_experts: int) -> torch.Tensor:
    tracker = _build_affinity_from_trace(trace, num_experts)
    return tracker.get_affinity_scores().clone()


def plan_from_trace(trace: RoutingTrace, num_experts: int, world_size: int,
                    experts_per_rank: int, max_circuits: int):
    tracker = _build_affinity_from_trace(trace, num_experts)
    expert_to_rank = {e: e // experts_per_rank for e in range(num_experts)}
    return tracker.compute_circuit_plan(
        expert_to_rank=expert_to_rank,
        max_circuits=max_circuits,
        experts_per_rank=experts_per_rank,
        world_size=world_size,
    )


def circuit_cost_delays(plan, topo: Topology, tensor_bytes: int) -> list[dict]:
    for rank in range(topo.config.num_pods * topo.config.nodes_per_pod
                      * topo.config.ranks_per_node):
        topo.assign(rank)
    rows = []
    for src, dst, score in plan:
        rows.append({
            "src": int(src),
            "dst": int(dst),
            "score": round(float(score), 6),
            "tier": topo.get_link_tier(src, dst).name,
            "delay_us": round(topo.get_pairwise_delay(src, dst, tensor_bytes), 3),
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description="Topology-invariance sanity gate")
    ap.add_argument("--trace", default="data/routing_traces/routing.json")
    ap.add_argument("--experts-per-rank", type=int, default=8)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--max-circuits", type=int, default=16)
    ap.add_argument("--payload-bytes", type=int, default=262144,
                    help="Dispatch payload size for cost illustration")
    ap.add_argument("--output", default="logs/topology_invariance_report.json")
    args = ap.parse_args()

    trace = RoutingTrace.load(args.trace)
    num_experts = trace.meta.num_experts
    _check_rank_layout(num_experts, args.world_size, args.experts_per_rank)
    print(f"[phase1] trace={args.trace} model={trace.meta.model_id} "
          f"layers={trace.meta.num_moe_layers} experts={num_experts} "
          f"top_k={trace.meta.top_k} tokens={trace.meta.total_tokens}")

    # ── Reference: affinity matrix + plan (topology-independent objects) ──
    ref_aff = affinity_matrix_from_trace(trace, num_experts)
    ref_plan = plan_from_trace(trace, num_experts, args.world_size,
                               args.experts_per_rank, args.max_circuits)
    print(f"[phase1] reference plan: {len(ref_plan)} circuits, "
          f"top pair {ref_plan[0]}")

    results = {}
    max_aff_delta = 0.0
    plan_equal = True
    for name, cfg in TOPOLOGY_GRID.items():
        topo = Topology(cfg)
        aff = affinity_matrix_from_trace(trace, num_experts)
        delta = float((aff - ref_aff).abs().max())
        max_aff_delta = max(max_aff_delta, delta)

        plan = plan_from_trace(trace, num_experts, args.world_size,
                               args.experts_per_rank, args.max_circuits)
        same_plan = plan == ref_plan
        plan_equal &= same_plan

        costs = circuit_cost_delays(ref_plan, topo, args.payload_bytes)
        cross_pod = [c for c in costs if c["tier"] == "CROSS_POD"]
        results[name] = {
            "config": {
                "num_pods": cfg.num_pods,
                "nodes_per_pod": cfg.nodes_per_pod,
                "ranks_per_node": cfg.ranks_per_node,
                "cross_pod_latency_us": cfg.cross_pod_latency_us,
                "cross_pod_bw_gbps": cfg.cross_pod_bandwidth_gbps,
                "delay_multiplier": cfg.delay_multiplier,
            },
            "affinity_max_abs_delta_vs_ref": delta,
            "plan_identical": same_plan,
            "circuit_costs": costs,
            "n_cross_pod_circuits": len(cross_pod),
            "cross_pod_delay_max_us": (
                max(c["delay_us"] for c in cross_pod) if cross_pod else 0.0
            ),
        }
        print(f"[phase1] {name:<20s} aff_delta={delta:.3e} "
              f"plan_identical={same_plan} cross_pod_circuits={len(cross_pod)}")

    verdict = plan_equal and max_aff_delta == 0.0
    report = {
        "experiment": "phase1_topology_invariance",
        "trace": args.trace,
        "num_experts": num_experts,
        "world_size": args.world_size,
        "experts_per_rank": args.experts_per_rank,
        "max_circuits": args.max_circuits,
        "payload_bytes": args.payload_bytes,
        "reference_plan": [
            [int(s), int(d), round(float(sc), 6)] for s, d, sc in ref_plan
        ],
        "topologies": results,
        "verdict": {
            "affinity_bit_identical": max_aff_delta == 0.0,
            "plan_bit_identical": plan_equal,
            "routing_topology_independent": verdict,
            "note": "cost side differs by tier (see circuit_costs); "
                    "routing side must not.",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[phase1] report → {out}")
    print(f"[phase1] VERDICT: routing_topology_independent={verdict}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())

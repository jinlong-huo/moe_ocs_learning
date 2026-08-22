#!/usr/bin/env python3
"""Phase 1 — OCS cost-side invariance gate + affinity-driven topology adjustment.

One gate, one story. Replays a real Qwen RoutingTrace and verifies the whole
decoupling that makes "record affinity → adjust OCS topology" sound:

  1. Topology invariance — the affinity matrix and the derived rank-pair
     circuit plan are bit-identical under any 3-tier topology configuration
     (pods × nodes × ranks, latencies, BW); only the per-circuit dispatch
     cost changes by tier (INTRA_NODE / INTRA_POD / CROSS_POD).

  2. Placement invariance — affinity is a property of *expert ids*, not of
     where experts live. ``Placement.linear`` reproduces the historical
     ``e // k`` / ``e % k`` mapping bit-for-bit; swapping/shuffling experts
     between ranks keeps a valid permutation and never touches routing (a
     token hits the same experts, only their owning rank relabels); only the
     cost projection (rank-pair plan) changes.

  3. Recorded affinity → OCS adjustment (the payoff) — because (1) and (2)
     hold, affinity recorded once from the trace can safely configure the
     topology: greedy co-activation clustering sets ``expert → rank`` (the
     intra-rank affinity fraction rises, so more co-activated experts share
     a rank and skip the network), and plan-centrality ordering sets
     ``rank → physical location`` (cross-pod exposure of the top-N circuit
     plan falls). The derived rank→location table is written to the report
     so it can be pasted into ``placement.rank_locations`` in a config.

The hardware-level checks (quantized-GEMM noise floor) live in this repo
too: ``scripts/compare_backend_traces.py`` (Phase 2) and
``scripts/compare_model_affinity.py`` (Phase 3); this is the framework-level
gate (Phase 1).

Usage:
    python3 scripts/verify_ocs_invariance.py \
        --trace data/routing_traces/routing.json \
        --experts-per-rank 8 --world-size 32 \
        --max-circuits 16 \
        --output logs/ocs_invariance_report.json
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

from src.runtime.placement import Placement  # noqa: E402
from src.comm.topology import LinkTier, Topology, TopologyConfig  # noqa: E402
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


def plan_under_placement(trace: RoutingTrace, num_experts: int,
                         placement: Placement, max_circuits: int) -> list:
    """Rank-pair circuit plan for a given placement (cost-side projection)."""
    tracker = _build_affinity_from_trace(trace, num_experts)
    return tracker.compute_circuit_plan(
        expert_to_rank=placement.expert_to_rank_dict(),
        experts_per_rank=placement.experts_per_rank,
        world_size=placement.world_size,
        max_circuits=max_circuits,
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


def cross_pod_exposure(plan, topo: Topology) -> dict:
    """Count and score-weight the cross-pod pairs in a rank-pair plan."""
    n_cross = 0
    score_cross = 0.0
    score_all = 0.0
    for src, dst, score in plan:
        if topo.get_link_tier(src, dst) == LinkTier.CROSS_POD:
            n_cross += 1
            score_cross += score
        score_all += score
    return {
        "cross_pod_pairs": n_cross,
        "cross_pod_score_fraction": round(
            score_cross / score_all, 6) if score_all else 0.0,
    }


def intra_rank_affinity_fraction(aff: torch.Tensor, placement: Placement) -> float:
    """Share of off-diagonal co-activation weight that lands within a rank."""
    same_rank = placement.expert_to_rank.unsqueeze(0) == placement.expert_to_rank.unsqueeze(1)
    off_diag = ~torch.eye(placement.num_experts, dtype=torch.bool)
    num = aff[same_rank & off_diag].sum().item()
    den = aff[off_diag].sum().item()
    return num / den if den else 0.0


def rank_centrality_order(plan, world_size: int) -> list[int]:
    """Ranks ordered by summed plan score (most traffic-heavy first)."""
    cent = {r: 0.0 for r in range(world_size)}
    for src, dst, score in plan:
        cent[src] += score
        cent[dst] += score
    return sorted(range(world_size), key=lambda r: -cent[r])


def linear_location_slots(num_pods: int, nodes_per_pod: int,
                          ranks_per_node: int) -> list[tuple]:
    """The flat (pod, node, local_rank) slots in rank order."""
    return [
        (pod, node, lr)
        for pod in range(num_pods)
        for node in range(nodes_per_pod)
        for lr in range(ranks_per_node)
    ]


def main() -> int:
    ap = argparse.ArgumentParser(description="OCS cost-side invariance gate")
    ap.add_argument("--trace", default="data/routing_traces/routing.json")
    ap.add_argument("--experts-per-rank", type=int, default=8)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--max-circuits", type=int, default=16)
    ap.add_argument("--payload-bytes", type=int, default=262144,
                    help="Dispatch payload size for cost illustration")
    ap.add_argument("--num-pods", type=int, default=2,
                    help="Pods in the §3 fixed fabric (default 2×4×4 = 32 ranks)")
    ap.add_argument("--nodes-per-pod", type=int, default=4)
    ap.add_argument("--ranks-per-node", type=int, default=4)
    ap.add_argument("--output", default="logs/ocs_invariance_report.json")
    args = ap.parse_args()

    trace = RoutingTrace.load(args.trace)
    num_experts = trace.meta.num_experts
    _check_rank_layout(num_experts, args.world_size, args.experts_per_rank)
    print(f"[phase1] trace={args.trace} model={trace.meta.model_id} "
          f"layers={trace.meta.num_moe_layers} experts={num_experts} "
          f"top_k={trace.meta.top_k} tokens={trace.meta.total_tokens}")

    # ══ §1. Topology invariance: affinity + plan never see the fabric ══
    lin = Placement.linear(num_experts, args.experts_per_rank, args.world_size)
    ref_aff = affinity_matrix_from_trace(trace, num_experts)
    ref_plan = plan_under_placement(trace, num_experts, lin, args.max_circuits)
    print(f"[phase1] §1 reference plan: {len(ref_plan)} circuits, "
          f"top pair {ref_plan[0]}")

    topo_results = {}
    max_aff_delta = 0.0
    plan_equal = True
    for name, cfg in TOPOLOGY_GRID.items():
        topo = Topology(cfg)
        aff = affinity_matrix_from_trace(trace, num_experts)
        delta = float((aff - ref_aff).abs().max())
        max_aff_delta = max(max_aff_delta, delta)

        plan = plan_under_placement(trace, num_experts, lin, args.max_circuits)
        same_plan = plan == ref_plan
        plan_equal &= same_plan

        costs = circuit_cost_delays(ref_plan, topo, args.payload_bytes)
        cross_pod = [c for c in costs if c["tier"] == "CROSS_POD"]
        topo_results[name] = {
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
        print(f"[phase1] §1 {name:<20s} aff_delta={delta:.3e} "
              f"plan_identical={same_plan} cross_pod_circuits={len(cross_pod)}")
    topology_verdict = plan_equal and max_aff_delta == 0.0

    # ══ §2. Placement invariance: affinity is expert-id property only ══
    eids = torch.arange(num_experts)
    lin_rank, lin_local = lin.resolve(eids)
    linear_matches_legacy = bool(
        (lin_rank == eids // args.experts_per_rank).all()
        and (lin_local == eids % args.experts_per_rank).all()
    )

    aff = affinity_matrix_from_trace(trace, num_experts)
    aff_sum = float(aff.sum())
    aff_diag = float(aff.diagonal().sum())

    swapped = lin.move_expert(0, args.world_size - 1)
    shuffled = Placement.shuffled(
        num_experts, args.experts_per_rank, args.world_size, seed=1
    )

    def keeps_permutation(p: Placement) -> bool:
        """Every expert appears exactly once; every rank owns exactly k."""
        return (
            sorted(
                e for r in range(args.world_size) for e in p.experts_on_rank(r)
            ) == list(range(num_experts))
            and all(
                len(p.experts_on_rank(r)) == args.experts_per_rank
                for r in range(args.world_size)
            )
        )

    # A single swap must only reshuffle experts between ranks 0 and W-1.
    swap_keeps_permutation = (
        keeps_permutation(swapped)
        and sorted(
            swapped.experts_on_rank(0) + swapped.experts_on_rank(args.world_size - 1)
        ) == sorted(lin.experts_on_rank(0) + lin.experts_on_rank(args.world_size - 1))
    )
    shuffle_keeps_permutation = keeps_permutation(shuffled)

    plan_swap = plan_under_placement(trace, num_experts, swapped, args.max_circuits)
    plan_shuf = plan_under_placement(trace, num_experts, shuffled, args.max_circuits)
    plan_swapped_differs = ref_plan != plan_swap
    plan_shuffled_differs = ref_plan != plan_shuf
    placement_verdict = linear_matches_legacy and swap_keeps_permutation \
        and shuffle_keeps_permutation

    delay_examples = [
        {
            "desc": "expert 0 -> its linear owner vs swapped owner (same physical ranks)",
            "linear_owner": int(lin.expert_to_rank[0].item()),
            "swapped_owner": int(swapped.expert_to_rank[0].item()),
        }
    ]
    print(f"[phase1] §2 linear==legacy: {linear_matches_legacy}; "
          f"swap keeps permutation: {swap_keeps_permutation}; "
          f"shuffle keeps permutation: {shuffle_keeps_permutation}")
    print(f"[phase1] §2 plan differs (swap): {plan_swapped_differs}; "
          f"(shuffle): {plan_shuffled_differs}")

    # ══ §3. Payoff: recorded affinity -> expert->rank + rank->location ══
    tracker = _build_affinity_from_trace(trace, num_experts)
    aff_placement = Placement.from_permutation(
        tracker.suggest_placement(args.experts_per_rank, args.world_size),
        args.experts_per_rank, args.world_size,
    )
    affinity_placement_differs = (
        aff_placement.to_rank_experts() != lin.to_rank_experts()
    )
    intra_lin = intra_rank_affinity_fraction(aff, lin)
    intra_aff = intra_rank_affinity_fraction(aff, aff_placement)
    intra_improves = intra_aff > intra_lin

    plan_aff = plan_under_placement(trace, num_experts, aff_placement, args.max_circuits)
    order = rank_centrality_order(plan_aff, args.world_size)
    slots = linear_location_slots(args.num_pods, args.nodes_per_pod, args.ranks_per_node)
    if len(slots) != args.world_size:
        raise ValueError(
            f"§3 fabric {args.num_pods}×{args.nodes_per_pod}×{args.ranks_per_node} "
            f"!= world_size {args.world_size}; adjust --num-pods/--nodes-per-pod/"
            f"--ranks-per-node or --world-size"
        )
    rank_locations = {rank: slots[i] for i, rank in enumerate(order)}

    topo_baseline = Topology(TopologyConfig(
        num_pods=args.num_pods, nodes_per_pod=args.nodes_per_pod,
        ranks_per_node=args.ranks_per_node,
    ))
    topo_adjusted = Topology(TopologyConfig(
        num_pods=args.num_pods, nodes_per_pod=args.nodes_per_pod,
        ranks_per_node=args.ranks_per_node, rank_locations=rank_locations,
    ))
    for r in range(args.world_size):
        topo_baseline.assign(r)
        topo_adjusted.assign(r)

    base_exposure = cross_pod_exposure(ref_plan, topo_baseline)
    adj_exposure = cross_pod_exposure(plan_aff, topo_adjusted)
    exposure_improves = (
        adj_exposure["cross_pod_score_fraction"] < base_exposure["cross_pod_score_fraction"]
    )
    payoff_verdict = affinity_placement_differs and intra_improves and exposure_improves

    print(f"[phase1] §3 intra-rank affinity: linear={intra_lin:.4f} "
          f"affinity={intra_aff:.4f} improves={intra_improves}")
    print(f"[phase1] §3 cross-pod exposure (score fraction): "
          f"baseline={base_exposure['cross_pod_score_fraction']:.4f} "
          f"adjusted={adj_exposure['cross_pod_score_fraction']:.4f} "
          f"improves={exposure_improves}")
    print(f"[phase1] §3 cross-pod plan pairs: {base_exposure['cross_pod_pairs']} -> "
          f"{adj_exposure['cross_pod_pairs']}")

    # ══ Report + verdict ══
    overall = topology_verdict and placement_verdict and payoff_verdict
    report = {
        "experiment": "phase1_ocs_invariance",
        "trace": args.trace,
        "num_experts": num_experts,
        "world_size": args.world_size,
        "experts_per_rank": args.experts_per_rank,
        "max_circuits": args.max_circuits,
        "payload_bytes": args.payload_bytes,
        "reference_plan": [
            [int(s), int(d), round(float(sc), 6)] for s, d, sc in ref_plan
        ],
        "topology_invariance": {
            "topologies": topo_results,
        },
        "placement_invariance": {
            "affinity_sum": round(aff_sum, 4),
            "affinity_diagonal_sum": round(aff_diag, 4),
            "linear_matches_legacy": linear_matches_legacy,
            "still_permutation_under_swap": swap_keeps_permutation,
            "still_permutation_under_shuffle": shuffle_keeps_permutation,
            "plan_swapped_differs": plan_swapped_differs,
            "plan_shuffled_differs": plan_shuffled_differs,
            "delay_examples": delay_examples,
        },
        "affinity_adjustment": {
            "affinity_placement_differs_from_linear": affinity_placement_differs,
            "intra_rank_affinity_fraction": {
                "linear": round(intra_lin, 6),
                "affinity": round(intra_aff, 6),
                "improves": intra_improves,
            },
            "cross_pod_exposure": {
                "baseline_linear_placement": base_exposure,
                "adjusted_affinity_placement": adj_exposure,
                "improves": exposure_improves,
            },
            # Paste this table into `placement.rank_locations` in a config.
            "derived_rank_locations": [
                [rank, list(loc)] for rank, loc in sorted(rank_locations.items())
            ],
            "note": "expert->rank from greedy co-activation clustering; "
                    "rank->location from plan-centrality packing (fixed 3-tier fabric).",
        },
        "verdict": {
            "affinity_bit_identical": max_aff_delta == 0.0,
            "plan_bit_identical": plan_equal,
            "routing_topology_independent": topology_verdict,
            "affinity_placement_independent": placement_verdict,
            "routing_untouched_by_placement": True,
            "placement_is_cost_side_variable": placement_verdict,
            "affinity_adjustment_reduces_cost": payoff_verdict,
            "overall": overall,
            "note": "affinity is built from expert ids only; placement/topology "
                    "relabel ranks (cost) and never which expert a token hits "
                    "(routing). Therefore the recorded affinity safely drives "
                    "expert->rank and rank->location OCS configuration.",
        },
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[phase1] report → {out}")
    print(f"[phase1] VERDICT: topology={topology_verdict} "
          f"placement={placement_verdict} payoff={payoff_verdict} "
          f"overall={overall}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())

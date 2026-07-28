#!/usr/bin/env python3
"""Compute an OCS circuit placement plan from a routing trace.

Usage:
    python scripts/compute_preset_plan.py \\
        --trace data/routing_traces/routing.json \\
        --output outputs/preset_plan.json \\
        --max-circuits 16 \\
        --experts-per-rank 4 \\
        --world-size 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ocs.preconfig import compute_plan_from_trace, export_plan, plan_summary


def main():
    parser = argparse.ArgumentParser(
        description="Compute OCS circuit placement plan from routing trace",
    )
    parser.add_argument(
        "--trace", required=True,
        help="Path to routing trace JSON (from training or replay capture)",
    )
    parser.add_argument(
        "--output", default="outputs/preset_plan.json",
        help="Path to output placement plan JSON",
    )
    parser.add_argument(
        "--max-circuits", type=int, default=16,
        help="Maximum circuits in the placement plan",
    )
    parser.add_argument(
        "--experts-per-rank", type=int, default=1,
        help="Experts per GPU rank",
    )
    parser.add_argument(
        "--world-size", type=int, default=4,
        help="Number of GPU ranks",
    )
    parser.add_argument(
        "--strategy", default="coactivation",
        choices=["coactivation"],
        help="Plan computation strategy",
    )
    args = parser.parse_args()

    if not os.path.exists(args.trace):
        print(f"Error: trace file not found: {args.trace}")
        sys.exit(1)

    print(f"Loading trace: {args.trace}")
    plan = compute_plan_from_trace(
        trace_path=args.trace,
        max_circuits=args.max_circuits,
        experts_per_rank=args.experts_per_rank,
        world_size=args.world_size,
        strategy=args.strategy,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    export_plan(plan, args.output)
    print(f"Plan exported -> {args.output}")
    print()

    summary = plan_summary(plan)
    print(f"Plan summary: {summary['num_circuits']} circuits")
    print(f"Score range: {summary['score_stats']['min']} - {summary['score_stats']['max']} "
          f"(mean={summary['score_stats']['mean']:.4f})")
    print("Top-5 pairs:")
    for p in summary["top_pairs"]:
        print(f"  rank {p['src']} -> rank {p['dst']}: {p['score']}")
    print(f"Per-rank outgoing circuits: {summary['rank_counts']}")


if __name__ == "__main__":
    main()

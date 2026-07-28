#!/usr/bin/env python3
"""Validate affinity consistency: train routing vs inference routing.

Runs two simulations — one with training routing, one with inference
routing (replay) — and computes correlation metrics between their
co-activation patterns.

Usage:
    python scripts/validate_affinity.py \\
        --train-trace data/routing_traces/routing_pretrained.json \\
        --infer-trace data/routing_traces/routing_finetuned.json \\
        --num-experts 256 \\
        --top-k 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.data.routing_schema import RoutingTrace
from src.eval.affinity_consistency import (
    layer_consistency_report,
    js_divergence,
    affinity_correlation,
    estimated_hit_rate,
    expert_distribution,
)
from src.ocs.preconfig import _build_affinity_from_trace


def main():
    parser = argparse.ArgumentParser(
        description="Validate training-to-inference affinity consistency",
    )
    parser.add_argument(
        "--train-trace", required=True,
        help="Path to training routing trace JSON",
    )
    parser.add_argument(
        "--infer-trace", required=True,
        help="Path to inference routing trace JSON",
    )
    parser.add_argument(
        "--num-experts", type=int, default=256,
        help="Total number of experts",
    )
    parser.add_argument(
        "--top-k", type=int, default=8,
        help="Top-K routing parameter",
    )
    parser.add_argument(
        "--max-circuits", type=int, default=16,
        help="Max OCS circuits for hit rate estimation",
    )
    parser.add_argument(
        "--output", default="",
        help="Optional path to save JSON report",
    )
    args = parser.parse_args()

    for path in [args.train_trace, args.infer_trace]:
        if not os.path.exists(path):
            print(f"Error: trace not found: {path}")
            sys.exit(1)

    print(f"Loading train trace: {args.train_trace}")
    train_trace = RoutingTrace.load(args.train_trace)
    print(f"Loading infer trace: {args.infer_trace}")
    infer_trace = RoutingTrace.load(args.infer_trace)

    # Build affinity trackers
    train_tracker = _build_affinity_from_trace(train_trace, args.num_experts)
    infer_tracker = _build_affinity_from_trace(infer_trace, args.num_experts)

    train_affinity = train_tracker.get_affinity_scores().numpy()
    infer_affinity = infer_tracker.get_affinity_scores().numpy()

    # Per-layer consistency (if both traces have per-layer data)
    train_layers_raw = []
    infer_layers_raw = []
    for route in train_trace.routes:
        train_layers_raw.append(route.layers)
    for route in infer_trace.routes:
        infer_layers_raw.append(route.layers)

    if train_layers_raw and infer_layers_raw:
        # Flatten: extract expert_ids from each layer
        t_layers = []
        i_layers = []
        for tl, il in zip(train_layers_raw, infer_layers_raw):
            for lid, t_layer in tl.items():
                t_layers.append(t_layer.experts)
            for lid, i_layer in il.items():
                i_layers.append(i_layer.experts)
    else:
        t_layers = []
        i_layers = []

    # Global metrics
    js_global = js_divergence(
        train_tracker.expert_usage.numpy() / max(train_tracker.expert_usage.sum().item(), 1),
        infer_tracker.expert_usage.numpy() / max(infer_tracker.expert_usage.sum().item(), 1),
    )
    aff_corr = affinity_correlation(train_affinity, infer_affinity)
    est_hit = estimated_hit_rate(
        train_affinity,
        max_circuits=args.max_circuits,
        num_ranks=args.num_experts // 1,  # assumes 1 expert per rank
        experts_per_rank=1,
    )

    report = {
        "train_trace": args.train_trace,
        "infer_trace": args.infer_trace,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "train_total_samples": train_tracker.total_samples,
        "infer_total_samples": infer_tracker.total_samples,
        "js_divergence_global": round(js_global, 6),
        "affinity_correlation": round(aff_corr, 6),
        "estimated_hit_rate": round(est_hit, 6),
        "preset_viability": "high" if est_hit > 0.7 else "medium" if est_hit > 0.4 else "low",
    }

    print()
    print("=" * 50)
    print("AFFINITY CONSISTENCY REPORT")
    print("=" * 50)
    print(f"Train samples:  {train_tracker.total_samples}")
    print(f"Infer samples:  {infer_tracker.total_samples}")
    print(f"JS divergence:  {report['js_divergence_global']} (0=identical, 1=different)")
    print(f"Affinity corr:  {report['affinity_correlation']} (1=perfect)")
    print(f"Est. hit rate:  {report['estimated_hit_rate']} (max_circuits={args.max_circuits})")
    print(f"Preset viable:  {report['preset_viability']}")
    print("-" * 50)

    if report["preset_viability"] == "high":
        print("=> Training affinity is a strong predictor. OCS preset should work well.")
    elif report["preset_viability"] == "medium":
        print("=> Moderate correlation. Preset may help but needs runtime fallback.")
    else:
        print("=> Low correlation. Preset unlikely to help; consider runtime OCS instead.")

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved -> {args.output}")


if __name__ == "__main__":
    main()

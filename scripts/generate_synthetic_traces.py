#!/usr/bin/env python3
"""Generate synthetic RoutingTrace JSONs with controlled domain patterns.

Each "domain" favors a specific cluster of experts, simulating how different
prompt types (code, math, prose) activate distinct expert subsets in real MoE.

Design:
  - num_experts experts, grouped into num_domains clusters
  - Each domain's tokens route toK experts from its cluster with noise
  - Train traces: one per domain cluster
  - Test traces: one from each cluster + one from an unseen cluster
  - Output: canonical RoutingTrace JSON files

Usage:
  python scripts/generate_synthetic_traces.py
      → outputs/synthetic_traces/train_domain_00.json ... test_domain_03.json
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.routing_schema import RoutingTrace, TokenRoute, LayerRoute, RunMeta


def _sample_experts(
    cluster_start: int,
    cluster_size: int,
    num_experts: int,
    top_k: int,
    noise_prob: float,
    rng: random.Random,
) -> list[int]:
    """Sample top_k expert IDs biased toward a cluster with noise."""
    experts = []
    cluster_ids = list(range(cluster_start, min(cluster_start + cluster_size, num_experts)))
    all_other = [e for e in range(num_experts) if e not in cluster_ids]

    for _ in range(top_k):
        if rng.random() < (1 - noise_prob) and cluster_ids:
            e = rng.choice(cluster_ids)
        elif all_other:
            e = rng.choice(all_other)
        else:
            e = rng.choice(cluster_ids) if cluster_ids else 0
        experts.append(e)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for e in experts:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    # Pad if we lost some to dedup
    while len(unique) < top_k:
        pool = cluster_ids if cluster_ids else list(range(num_experts))
        e = rng.choice(pool)
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return unique[:top_k]


def generate_trace(
    num_experts: int,
    top_k: int,
    cluster_start: int,
    cluster_size: int,
    noise_prob: float,
    num_tokens: int,
    num_moe_layers: int,
    domain_label: str,
    seed: int = 42,
) -> RoutingTrace:
    """Generate one RoutingTrace with domain-specific expert routing.

    Args:
        num_experts: total experts in the simulated model.
        top_k: experts selected per token.
        cluster_start: first expert ID in this domain's cluster.
        cluster_size: how many experts in the cluster.
        noise_prob: probability a token selects an out-of-cluster expert.
        num_tokens: total tokens (prompt + generated).
        num_moe_layers: how many MoE layers to simulate.
        domain_label: human-readable label for trace metadata.
        seed: random seed for reproducibility.
    """
    rng = random.Random(seed)

    # Generate per-token routing
    routes = []
    for pos in range(num_tokens):
        layer_routes = {}
        for layer_id in range(num_moe_layers):
            experts = _sample_experts(
                cluster_start, cluster_size, num_experts,
                top_k, noise_prob, rng,
            )
            weights = [round(rng.uniform(0.3, 1.0), 4) for _ in range(len(experts))]
            layer_routes[str(layer_id)] = LayerRoute(experts=experts, weights=weights)
        phase = "prefill" if pos < num_tokens // 2 else "decode"
        routes.append(TokenRoute(
            token_pos=pos, token_id=pos % 50000,
            token_str=f"tok_{pos}", phase=phase,
            layers=layer_routes,
        ))

    meta = RunMeta(
        model_id=f"synthetic_{num_experts}e_{domain_label}",
        model_type="synthetic_moe",
        num_layers=num_moe_layers + 4,  # some dense layers too
        num_moe_layers=num_moe_layers,
        num_experts=num_experts,
        top_k=top_k,
        prompt_len=num_tokens // 2,
        generated_len=num_tokens - num_tokens // 2,
        total_tokens=num_tokens,
        backend="synthetic",
    )

    return RoutingTrace(
        meta=meta,
        prompt_tokens=list(range(num_tokens // 2)),
        generated_tokens=list(range(num_tokens // 2, num_tokens)),
        routes=routes,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic routing traces with domain patterns",
    )
    parser.add_argument("--num-experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--num-tokens", type=int, default=128,
                        help="Tokens per trace (half prompt, half generated)")
    parser.add_argument("--num-moe-layers", type=int, default=8)
    parser.add_argument("--num-domains", type=int, default=4,
                        help="Number of distinct expert clusters")
    parser.add_argument("--train-domains", type=int, nargs="+", default=[0, 1, 2],
                        help="Domain indices to use for training (plan building)")
    parser.add_argument("--test-domains", type=int, nargs="+", default=[0, 3],
                        help="Domain indices to use for testing (held-out evaluation)")
    parser.add_argument("--noise-prob", type=float, default=0.15,
                        help="Probability of out-of-cluster expert selection")
    parser.add_argument("--output-dir", default="outputs/synthetic_traces")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    num_experts = args.num_experts
    num_domains = args.num_domains
    experts_per_domain = num_experts // num_domains

    if num_experts % num_domains != 0:
        print(f"WARNING: {num_experts} experts not evenly divisible by {num_domains} domains")
    if args.top_k > experts_per_domain:
        print(f"WARNING: top_k={args.top_k} > experts_per_domain={experts_per_domain} — "
              "will include out-of-cluster experts frequently")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate all domain traces
    traces = {}
    for domain_idx in range(num_domains):
        cluster_start = domain_idx * experts_per_domain
        trace = generate_trace(
            num_experts=num_experts,
            top_k=args.top_k,
            cluster_start=cluster_start,
            cluster_size=experts_per_domain,
            noise_prob=args.noise_prob,
            num_tokens=args.num_tokens,
            num_moe_layers=args.num_moe_layers,
            domain_label=f"domain_{domain_idx:02d}",
            seed=args.seed + domain_idx * 100,
        )
        traces[domain_idx] = trace

    # Save training traces
    train_files = []
    for dom in args.train_domains:
        if dom not in traces:
            print(f"ERROR: domain {dom} not in generated traces (0-{num_domains-1})")
            sys.exit(1)
        path = output_dir / f"train_domain_{dom:02d}.json"
        traces[dom].save(str(path))
        train_files.append(str(path))
        print(f"[train] domain={dom} cluster=experts "
              f"{dom * experts_per_domain}-{dom * experts_per_domain + experts_per_domain - 1} "
              f"→ {path.name} ({args.num_tokens} tokens, top_k={args.top_k})")

    # Save test traces
    test_files = []
    for dom in args.test_domains:
        if dom not in traces:
            print(f"ERROR: domain {dom} not in generated traces (0-{num_domains-1})")
            sys.exit(1)
        path = output_dir / f"test_domain_{dom:02d}.json"
        traces[dom].save(str(path))
        test_files.append(str(path))
        same_as_train = " (in train set)" if dom in args.train_domains else " (HELD OUT)"
        print(f"[test]  domain={dom} cluster=experts "
              f"{dom * experts_per_domain}-{dom * experts_per_domain + experts_per_domain - 1} "
              f"→ {path.name}{same_as_train}")

    # Save manifest
    manifest = {
        "num_experts": num_experts,
        "top_k": args.top_k,
        "num_tokens": args.num_tokens,
        "num_moe_layers": args.num_moe_layers,
        "num_domains": num_domains,
        "experts_per_domain": experts_per_domain,
        "noise_prob": args.noise_prob,
        "train_domains": args.train_domains,
        "test_domains": args.test_domains,
        "train_files": train_files,
        "test_files": test_files,
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[manifest] {manifest_path}")
    print(f"[summary] {len(train_files)} train traces + {len(test_files)} test traces")
    print(f"[experiment] Plan from train domains {args.train_domains} "
          f"→ test on domains {args.test_domains}")
    held_out = [d for d in args.test_domains if d not in args.train_domains]
    if held_out:
        print(f"[hypothesis] Held-out domain {held_out} should show LOWER hit rate "
              "than in-distribution test domains")


if __name__ == "__main__":
    main()

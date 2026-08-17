#!/usr/bin/env python3
"""Close the loop: training trace → circuit plan → test trace → prediction accuracy.

This is the end-to-end closed-loop evaluation pipeline that eliminates the
simulator. It directly measures how well a circuit plan built from captured
inference routing (prompt A) predicts the routing for the next prompt (B).

No simulator. No training. Pure inference-based routing prediction.

Pipeline:
  1. Load training routing traces from real MoE inference captures
  2. Build ExpertAffinityTracker → compute circuit plan (rank-pair predictions)
  3. For each test trace:
     a. Load the test trace directly (no simulator)
     b. For every (token, layer): map selected experts → ranks → pairs
     c. Check: is each inter-rank pair in the plan's predicted set?
     d. Compute: expert_pair_hit_rate, rank_pair_hit_rate, token_hit_rate
  4. Compare against random baseline
  5. Self-consistency check (train == test should achieve near-perfect hit rate)

Core research questions answered:
  - Can captured routing from one prompt predict routing for another?
  - How much does domain similarity matter (same-topic vs. different-topic prompts)?
  - What is the ceiling? (self-consistency → "if the model routes identically,
    all pairs are covered")

Usage:
  # Close the loop with real moe_run.py traces:
  python scripts/close_loop.py \
      --train-traces logs/train/routing.json \
      --test-traces logs/test/routing.json \
      --experts-per-rank 6 \
      --max-circuits 16

  # Multiple training traces, multiple test traces:
  python scripts/close_loop.py \
      --train-traces logs/prompt_a/routing.json logs/prompt_b/routing.json \
      --test-traces logs/prompt_c/routing.json logs/prompt_d/routing.json

  # Self-consistency check (train == test):
  python scripts/close_loop.py \
      --train-traces logs/routing.json \
      --test-traces logs/routing.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.ocs.preconfig import compute_plan_from_traces, export_plan, plan_summary
from src.eval.hit_rate import (
    TraceHitRateReport,
    compute_trace_hit_rate,
    compute_multi_trace_hit_rate,
    format_trace_report,
)


def generate_random_baseline(
    plan: list,
    num_ranks: int,
    seed: int = 42,
) -> list:
    """Generate a random plan with the same size as the real plan.

    Returns a list of (src_rank, dst_rank, 0.0) tuples for comparison.
    Only includes inter-rank pairs (src != dst).
    """
    rng = random.Random(seed)
    all_pairs = [(s, d) for s in range(num_ranks) for d in range(num_ranks) if s != d]
    rng.shuffle(all_pairs)
    random_plan = [(s, d, 0.0) for (s, d) in all_pairs[: len(plan)]]
    return random_plan


def build_plan(
    train_paths: list[str],
    max_circuits: int,
    experts_per_rank: int,
    world_size: int,
    output_path: str | None = None,
) -> list:
    """Build circuit plan from training traces."""
    print(f"\n{'=' * 60}")
    print("PHASE 1: BUILD CIRCUIT PLAN FROM TRAINING TRACES")
    print(f"{'=' * 60}")
    print(f"Training traces: {len(train_paths)}")
    for p in train_paths:
        print(f"  {Path(p).name}")

    plan = compute_plan_from_traces(
        trace_paths=train_paths,
        max_circuits=max_circuits,
        experts_per_rank=experts_per_rank,
        world_size=world_size,
    )

    summary = plan_summary(plan)
    print(f"\nPlan built: {summary['num_circuits']} circuits")
    print(f"Score range: {summary['score_stats']['min']:.4f} – "
          f"{summary['score_stats']['max']:.4f} "
          f"(mean={summary['score_stats']['mean']:.4f})")
    print("Top-10 predicted rank pairs:")
    for i, p in enumerate(summary["top_pairs"][:10]):
        print(f"  {i+1:2d}. rank {p['src']} → rank {p['dst']}: {p['score']:.4f}")

    if output_path:
        export_plan(plan, output_path)
        print(f"\nPlan exported → {output_path}")

    return plan


def evaluate_test_traces(
    plan: list,
    test_paths: list[str],
    experts_per_rank: int,
) -> tuple[TraceHitRateReport, list[TraceHitRateReport]]:
    """Evaluate plan prediction accuracy against test traces."""
    print(f"\n{'=' * 60}")
    print("PHASE 2: EVALUATE PREDICTION ACCURACY")
    print(f"{'=' * 60}")
    print(f"Test traces: {len(test_paths)}")

    agg, per_trace = compute_multi_trace_hit_rate(plan, test_paths, experts_per_rank)

    for i, (path, report) in enumerate(zip(test_paths, per_trace)):
        label = f"Test {i+1}: {Path(path).name}"
        print(f"\n{format_trace_report(report, label)}")

    return agg, per_trace


def run_self_consistency_check(
    train_paths: list[str],
    max_circuits: int,
    experts_per_rank: int,
    world_size: int,
) -> None:
    """Self-consistency: build plan from trace and test against itself.

    If the plan perfectly captures the training trace's routing, you'd expect
    high hit rates (limited only by plan size). This establishes the ceiling.
    """
    print(f"\n{'=' * 60}")
    print("SELF-CONSISTENCY CHECK (ceiling analysis)")
    print(f"{'=' * 60}")
    print("Building plan from training traces and testing on same traces...")

    plan = build_plan(
        train_paths, max_circuits, experts_per_rank, world_size, output_path=None,
    )

    agg, per_trace = evaluate_test_traces(plan, train_paths, experts_per_rank)

    print(f"\n{'─' * 60}")
    print("SELF-CONSISTENCY AGGREGATE:")
    print(format_trace_report(agg, "Train=Test (ceiling)"))

    # Explain the ceiling
    print(f"\nInterpretation:")
    print(f"  Expert pair hit rate: {agg.expert_pair_hit_rate:.1%}")
    print(f"  Rank pair hit rate:   {agg.rank_pair_hit_rate:.1%}")
    print(f"  Token hit rate:       {agg.token_hit_rate:.1%}")
    if agg.rank_pair_hit_rate < 0.95:
        print(f"  → Even on the SAME trace, {1-agg.rank_pair_hit_rate:.1%} of inter-rank")
        print(f"    pairs are not covered. This is the plan capacity ceiling —")
        print(f"    the plan has {len(plan)} slots but there are more unique rank-pair")
        print(f"    communications in the trace. Increase --max-circuits to raise the ceiling.")
    else:
        print(f"  → The plan captures nearly all rank-pair communications from")
        print(f"    the training trace. This is an ideal scenario for OCS preset.")


def print_final_report(
    plan: list,
    agg: TraceHitRateReport,
    random_agg: TraceHitRateReport | None,
    test_paths: list[str],
) -> None:
    """Print the final closed-loop summary."""
    print(f"\n{'=' * 70}")
    print("CLOSED-LOOP REPORT")
    print(f"{'=' * 70}")

    print(f"\nConfiguration:")
    print(f"  Plan size:     {len(plan)} circuits")
    print(f"  Test traces:   {len(test_paths)}")
    for p in test_paths:
        print(f"    {Path(p).name}")

    s = agg.summary()
    print(f"\nPrediction Accuracy (Across All Test Traces):")
    print(f"  Expert pair hit rate:  {s['expert_pair_hit_rate']:.1%}")
    print(f"  Rank pair hit rate:    {s['rank_pair_hit_rate']:.1%}")
    print(f"  Token hit rate:        {s['token_hit_rate']:.1%}")
    print(f"  Plan utilization:      {s['plan_utilization']:.1%}")

    if random_agg is not None:
        rs = random_agg.summary()
        print(f"\nRandom Baseline (Same Plan Size):")
        print(f"  Expert pair hit rate:  {rs['expert_pair_hit_rate']:.1%}")
        print(f"  Rank pair hit rate:    {rs['rank_pair_hit_rate']:.1%}")
        improvement = (
            (s['rank_pair_hit_rate'] - rs['rank_pair_hit_rate'])
            / max(rs['rank_pair_hit_rate'], 0.001)
            * 100
        )
        print(f"  Improvement over random: {improvement:+.1f}%")

    print(f"\nKey Takeaway:")
    if s['rank_pair_hit_rate'] > 0.5:
        print(f"  ✅ The plan from training traces GENERALIZES to test traces.")
        print(f"     {s['rank_pair_hit_rate']:.1%} of inter-rank comms were predicted.")
        print(f"     OCS preset is worthwhile for this workload.")
    elif s['rank_pair_hit_rate'] > 0.2:
        print(f"  ⚠️  Partial generalization: {s['rank_pair_hit_rate']:.1%} hit rate.")
        print(f"     OCS preset helps but needs larger plan or domain-matched traces.")
    else:
        print(f"  ❌ Low generalization: {s['rank_pair_hit_rate']:.1%} hit rate.")
        print(f"     Training traces don't predict test routing well.")
        print(f"     Consider: domain-specific plans, larger plan, or online mode.")

    print(f"{'=' * 70}")


def main():
    parser = argparse.ArgumentParser(
        description="Close the loop: training trace → plan → test trace → prediction accuracy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic closed-loop evaluation:
  python scripts/close_loop.py \\
      --train-traces logs/prompt_a/routing.json \\
      --test-traces logs/prompt_b/routing.json

  # With self-consistency check:
  python scripts/close_loop.py \\
      --train-traces logs/routing.json \\
      --test-traces logs/routing.json \\
      --self-consistency

  # Full evaluation with multiple traces and random baseline:
  python scripts/close_loop.py \\
      --train-traces logs/a/routing.json logs/b/routing.json \\
      --test-traces logs/c/routing.json logs/d/routing.json \\
      --random-baseline --self-consistency
        """,
    )
    parser.add_argument(
        "--train-traces", nargs="+", required=True,
        help="Training routing trace JSONs (from moe_run.py captures)",
    )
    parser.add_argument(
        "--test-traces", nargs="+", required=True,
        help="Test routing trace JSONs (held-out evaluation)",
    )
    parser.add_argument("--experts-per-rank", type=int, default=6,
                        help="Experts per GPU rank (default: 6)")
    parser.add_argument("--max-circuits", type=int, default=16,
                        help="Maximum circuits in the OCS plan (default: 16)")
    parser.add_argument("--world-size", type=int, default=None,
                        help="Number of GPU ranks (default: auto from num_experts / experts_per_rank)")
    parser.add_argument("--output-dir", default="outputs/closed_loop",
                        help="Output directory for plan and results")
    parser.add_argument("--random-baseline", action="store_true",
                        help="Compare against a random plan of same size")
    parser.add_argument("--self-consistency", action="store_true",
                        help="Also test plan on its own training traces (ceiling check)")
    parser.add_argument("--no-self-test", action="store_true",
                        help="Skip testing training traces against themselves")
    args = parser.parse_args()

    # Expand globs
    train_paths = []
    for p in args.train_traces:
        expanded = sorted(glob.glob(p))
        train_paths.extend(expanded if expanded else [p])

    test_paths = []
    for p in args.test_traces:
        expanded = sorted(glob.glob(p))
        test_paths.extend(expanded if expanded else [p])

    # Validate files
    all_paths = train_paths + test_paths
    for p in all_paths:
        if not os.path.exists(p):
            print(f"ERROR: file not found: {p}")
            sys.exit(1)

    # Determine world_size from first trace if not specified
    if args.world_size is None:
        from src.data.routing_schema import RoutingTrace
        first_trace = RoutingTrace.load(train_paths[0])
        num_experts = first_trace.meta.num_experts
        args.world_size = num_experts // args.experts_per_rank
        print(f"Auto world_size: {args.world_size} "
              f"({num_experts} experts / {args.experts_per_rank} per rank)")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train traces:  {len(train_paths)}")
    print(f"Test traces:   {len(test_paths)}")
    print(f"World size:    {args.world_size}")
    print(f"Max circuits:  {args.max_circuits}")
    print(f"Experts/rank:  {args.experts_per_rank}")

    # Phase 1: Build plan
    plan_path = str(output_dir / "circuit_plan.json")
    plan = build_plan(
        train_paths, args.max_circuits, args.experts_per_rank, args.world_size,
        output_path=plan_path,
    )

    # Phase 2: Evaluate on test traces
    agg, per_trace = evaluate_test_traces(plan, test_paths, args.experts_per_rank)

    # Phase 2b: Random baseline
    random_agg = None
    if args.random_baseline:
        print(f"\n{'=' * 60}")
        print("RANDOM BASELINE")
        print(f"{'=' * 60}")
        random_plan = generate_random_baseline(plan, args.world_size)
        random_agg, random_per = compute_multi_trace_hit_rate(
            random_plan, test_paths, args.experts_per_rank,
        )
        print(format_trace_report(random_agg, "Random Plan"))

    # Phase 3 (optional): Self-consistency check
    if args.self_consistency and not args.no_self_test:
        # Skip if training and test sets are identical (already self-tested in Phase 2)
        train_set = set(os.path.abspath(p) for p in train_paths)
        test_set = set(os.path.abspath(p) for p in test_paths)
        if train_set == test_set:
            print(f"\n{'─' * 60}")
            print("SELF-CONSISTENCY: Train == Test — already evaluated in Phase 2.")
            print(f"  Rank pair hit rate (ceiling): {agg.rank_pair_hit_rate:.1%}")
            print(f"  The ceiling is limited by plan_size={len(plan)} vs.")
            print(f"  unique inter-rank pairs used in the trace.")
        else:
            run_self_consistency_check(
                train_paths, args.max_circuits, args.experts_per_rank,
                args.world_size,
            )

    # Phase 4: Final report
    print_final_report(plan, agg, random_agg, test_paths)

    # Save results
    results = {
        "config": {
            "train_traces": train_paths,
            "test_traces": test_paths,
            "experts_per_rank": args.experts_per_rank,
            "max_circuits": args.max_circuits,
            "world_size": args.world_size,
        },
        "plan": plan_summary(plan),
        "aggregate": agg.summary(),
        "per_trace": [r.summary() for r in per_trace],
    }
    if random_agg is not None:
        results["random_baseline"] = random_agg.summary()

    results_path = str(output_dir / "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()

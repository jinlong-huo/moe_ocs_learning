#!/usr/bin/env python3
"""Evaluate OCS preset: plan from training traces → test on held-out traces.

Pipeline:
  1. Load training routing traces (from real MoE inference captures)
  2. Merge into one affinity model → compute circuit plan
  3. For each test trace (held-out):
     a. Pre-configure OCS circuits from the plan
     b. Replay test routing through simulator (routing_replay mode)
     c. Measure operational hit rate: what fraction of communication pairs
        found pre-established circuits?
  4. Compare against EPS baseline (no OCS) for the same test traces

The key research question: does affinity from prompts A, B, C generalize
to unseen prompt D? Measured as operational hit rate on D.

Usage:
  # From synthetic traces (controlled experiment):
  python scripts/evaluate_preset.py \
      --train-traces outputs/synthetic_traces/train_domain_*.json \
      --test-traces outputs/synthetic_traces/test_domain_*.json \
      --world-size 4 --max-circuits 16

  # From real MLX captures:
  python scripts/evaluate_preset.py \
      --train-traces logs/prompt_a/routing.json logs/prompt_b/routing.json \
      --test-traces logs/prompt_c/routing.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml

from src.ocs.preconfig import (
    compute_plan_from_traces, export_plan, load_plan, plan_summary,
)
from src.eval.hit_rate import (
    HitRateReport, compute_hit_rate_from_trace_files, format_report,
)


def build_plan(
    train_paths: list[str],
    max_circuits: int,
    world_size: int,
    experts_per_rank: int,
    output_path: str,
) -> str:
    """Build circuit plan from training traces and save to file."""
    print(f"\n{'='*55}")
    print("PHASE 1: BUILD PLAN")
    print(f"{'='*55}")
    print(f"Training traces: {len(train_paths)}")
    for p in train_paths:
        print(f"  {p}")

    plan = compute_plan_from_traces(
        trace_paths=train_paths,
        max_circuits=max_circuits,
        experts_per_rank=experts_per_rank,
        world_size=world_size,
    )
    export_plan(plan, output_path)

    summary = plan_summary(plan)
    print(f"\nPlan: {summary['num_circuits']} circuits")
    print(f"Scores: {summary['score_stats']['min']:.3f}–{summary['score_stats']['max']:.3f} "
          f"(mean={summary['score_stats']['mean']:.3f})")
    print("Top-5 pairs:")
    for p in summary["top_pairs"]:
        print(f"  rank {p['src']} → rank {p['dst']}: {p['score']:.4f}")

    return output_path


def build_test_config(
    plan_path: str,
    test_trace_path: str,
    world_size: int,
    experts_per_rank: int,
    max_circuits: int,
    trace_dir: str,
) -> dict:
    """Build a simulator config for ocs_preset + mixed_transport + routing_replay."""
    cfg = {
        "world_size": world_size,
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "backend": "gloo",
        "model": {
            "num_experts": world_size * experts_per_rank,
            "experts_per_rank": experts_per_rank,
            "hidden_dim": 256,
            "expert_hidden_mult": 4,
            "top_k": 4,
            "capacity_factor": 1.25,
        },
        "data": {
            "batch_size": 16,
            "seq_len": 64,
            "num_microbatches": 4,
        },
        "routing": {"strategy": "replay", "jitter": 0.000005},
        "routing_replay": {
            "enabled": True,
            "trace_path": test_trace_path,
            "layer_idx": 0,
            "cycle_layers": True,
        },
        "runtime": {"mode": "ocs_preset", "num_steps": 2},
        "delay": {"enabled": False, "comm_delay_us": 0, "comm_delay_jitter_us": 0},
        "topology": {
            "enabled": True,
            "num_pods": 2,
            "nodes_per_pod": 1,
            "ranks_per_node": world_size // 2,
            "intra_node_latency_us": 1.0,
            "intra_pod_latency_us": 5.0,
            "cross_pod_latency_us": 20.0,
            "intra_node_bandwidth_gbps": 900.0,
            "intra_pod_bandwidth_gbps": 200.0,
            "cross_pod_bandwidth_gbps": 100.0,
        },
        "ocs": {
            "enabled": True,
            "max_circuits": max_circuits,
            "reconfig_time_us": 100.0,
            "circuit_latency_us": 1.0,
            "circuit_bandwidth_gbps": 400.0,
            "placement_strategy": "affinity",
            "mixed_transport": {"enabled": True},
            "preset": {
                "source": "plan",
                "plan_path": plan_path,
                "strategy": "coactivation",
            },
        },
        "profiling": {
            "export_trace": True,
            "trace_dir": trace_dir,
        },
    }
    return cfg


def build_eps_config(
    test_trace_path: str,
    world_size: int,
    experts_per_rank: int,
    trace_dir: str,
) -> dict:
    """Build config for EPS baseline (no OCS)."""
    cfg = {
        "world_size": world_size,
        "master_addr": "127.0.0.1",
        "master_port": 29500,
        "backend": "gloo",
        "model": {
            "num_experts": world_size * experts_per_rank,
            "experts_per_rank": experts_per_rank,
            "hidden_dim": 256,
            "expert_hidden_mult": 4,
            "top_k": 4,
            "capacity_factor": 1.25,
        },
        "data": {
            "batch_size": 16,
            "seq_len": 64,
            "num_microbatches": 4,
        },
        "routing": {"strategy": "replay", "jitter": 0.000005},
        "routing_replay": {
            "enabled": True,
            "trace_path": test_trace_path,
            "layer_idx": 0,
            "cycle_layers": True,
        },
        "runtime": {"mode": "overlap", "num_steps": 2},
        "delay": {"enabled": False, "comm_delay_us": 0, "comm_delay_jitter_us": 0},
        "topology": {
            "enabled": True,
            "num_pods": 2,
            "nodes_per_pod": 1,
            "ranks_per_node": world_size // 2,
            "intra_node_latency_us": 1.0,
            "intra_pod_latency_us": 5.0,
            "cross_pod_latency_us": 20.0,
            "intra_node_bandwidth_gbps": 900.0,
            "intra_pod_bandwidth_gbps": 200.0,
            "cross_pod_bandwidth_gbps": 100.0,
        },
        "ocs": {"enabled": False},
        "profiling": {
            "export_trace": True,
            "trace_dir": trace_dir,
        },
    }
    return cfg


def run_simulator(config: dict, trace_dir: str) -> bool:
    """Run the simulator with a given config. Returns True on success."""
    from src.launcher import launch
    print(f"  Running simulator (trace_dir={trace_dir})...", end=" ", flush=True)
    try:
        launch(config, trace_dir=trace_dir)
        print("OK")
        return True
    except SystemExit as e:
        if e.code != 0:
            print(f"FAILED (exit {e.code})")
            return False
        print("OK")
        return True


def collect_trace_files(trace_dir: str) -> list[str]:
    """Find all rank_NN_trace.json files in a directory."""
    pattern = os.path.join(trace_dir, "rank_*_trace.json")
    return sorted(glob.glob(pattern))


def evaluate_test_trace(
    test_path: str,
    plan_path: str,
    world_size: int,
    experts_per_rank: int,
    max_circuits: int,
    output_dir: str,
) -> dict:
    """Run OCS preset and EPS baseline for one test trace, return metrics."""
    test_name = Path(test_path).stem
    print(f"\n{'─'*55}")
    print(f"TEST: {test_name}")
    print(f"  Trace: {test_path}")

    results = {"test": test_name, "trace_path": test_path}

    # -- OCS Preset run --
    ocs_trace_dir = os.path.join(output_dir, f"{test_name}_ocs")
    os.makedirs(ocs_trace_dir, exist_ok=True)
    cfg = build_test_config(plan_path, test_path, world_size, experts_per_rank, max_circuits, ocs_trace_dir)

    cfg_path = os.path.join(output_dir, f"{test_name}_ocs.yaml")
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)

    ok = run_simulator(cfg, ocs_trace_dir)
    if ok:
        trace_files = collect_trace_files(ocs_trace_dir)
        if trace_files:
            report = compute_hit_rate_from_trace_files(trace_files)
            results["ocs"] = report.summary()
            results["ocs"]["trace_dir"] = ocs_trace_dir
            print(format_report(report))
        else:
            print("  WARNING: No trace files found")
            results["ocs"] = {"error": "no traces"}
    else:
        results["ocs"] = {"error": "simulator failed"}

    # -- EPS Baseline run --
    eps_trace_dir = os.path.join(output_dir, f"{test_name}_eps")
    os.makedirs(eps_trace_dir, exist_ok=True)
    eps_cfg = build_eps_config(test_path, world_size, experts_per_rank, eps_trace_dir)

    ok = run_simulator(eps_cfg, eps_trace_dir)
    if ok and collect_trace_files(eps_trace_dir):
        results["eps"] = {"trace_dir": eps_trace_dir}
    else:
        results["eps"] = {"error": "simulator failed"}

    return results


def print_comparison(all_results: list[dict]) -> None:
    """Print comparison table across test traces."""
    print(f"\n{'='*70}")
    print("COMPARISON: OCS Preset vs EPS Baseline")
    print(f"{'='*70}")
    header = f"{'Test':<25} {'OCS Hit Rate':<14} {'OCS Coverage':<14} {'Preset Util':<14}"
    print(header)
    print("-" * 70)
    for r in all_results:
        ocs = r.get("ocs", {})
        hit = ocs.get("operational_hit_rate", 0)
        cov = ocs.get("ocs_coverage", 0)
        util = ocs.get("preset_utilization", 0)
        print(f"{r['test']:<25} {hit:<14.1%} {cov:<14.1%} {util:<14.1%}")
    print("=" * 70)

    # Highlight the hypothesis
    in_dist = [r for r in all_results if "domain_00" in r["test"]]
    held_out = [r for r in all_results if "domain_03" in r["test"]]
    if in_dist and held_out:
        in_hit = in_dist[0].get("ocs", {}).get("operational_hit_rate", 0)
        out_hit = held_out[0].get("ocs", {}).get("operational_hit_rate", 0)
        print(f"\nHypothesis check:")
        print(f"  In-distribution hit rate:  {in_hit:.1%}")
        print(f"  Held-out hit rate:         {out_hit:.1%}")
        if in_hit > out_hit * 1.5:
            print(f"  → Affinity from training domains generalizes poorly to unseen domain.")
            print(f"    OCS preset is domain-sensitive. Circuit plan should be built")
            print(f"    from representative prompts covering the expected input types.")
        elif in_hit > out_hit:
            print(f"  → Some generalization but held-out domain benefits less.")
        else:
            print(f"  → Unexpected: held-out domain performed as well or better.")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate OCS preset: plan from training → test on held-out"
    )
    parser.add_argument("--train-traces", nargs="+", required=True,
                        help="Training routing trace JSONs (for plan building)")
    parser.add_argument("--test-traces", nargs="+", required=True,
                        help="Test routing trace JSONs (held-out evaluation)")
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--experts-per-rank", type=int, default=8)
    parser.add_argument("--max-circuits", type=int, default=16)
    parser.add_argument("--output-dir", default="outputs/eval_results")
    args = parser.parse_args()

    # Expand globs in train/test traces
    train_paths = []
    for p in args.train_traces:
        expanded = sorted(glob.glob(p))
        if expanded:
            train_paths.extend(expanded)
        else:
            train_paths.append(p)

    test_paths = []
    for p in args.test_traces:
        expanded = sorted(glob.glob(p))
        if expanded:
            test_paths.extend(expanded)
        else:
            test_paths.append(p)

    # Validate files exist
    for p in train_paths + test_paths:
        if not os.path.exists(p):
            print(f"ERROR: file not found: {p}")
            sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Train traces: {len(train_paths)}")
    print(f"Test traces:  {len(test_paths)}")
    print(f"World size:   {args.world_size}")
    print(f"Max circuits: {args.max_circuits}")

    # Phase 1: Build plan
    plan_path = str(output_dir / "circuit_plan.json")
    build_plan(train_paths, args.max_circuits, args.world_size,
               args.experts_per_rank, plan_path)

    # Phase 2: Evaluate each test trace
    all_results = []
    for test_path in test_paths:
        result = evaluate_test_trace(
            test_path, plan_path,
            args.world_size, args.experts_per_rank,
            args.max_circuits, str(output_dir),
        )
        all_results.append(result)

    # Phase 3: Comparison report
    print_comparison(all_results)

    # Save results
    results_path = str(output_dir / "results.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()

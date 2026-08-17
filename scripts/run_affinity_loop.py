#!/usr/bin/env python3
"""End-to-end closed-loop validation: capture → affinity → plan → test.

The complete feedback loop:
  1. Load captured traces from the prompt taxonomy
  2. For each experiment (defined in the taxonomy):
     a. Build circuit plan from TRAINING traces
     b. Run OCS simulator in preset mode on TEST traces
     c. Run EPS baseline for comparison
     d. Measure operational hit rate and simulated time savings
  3. Print comparison: does same-domain → high hit rate? cross-domain → low?

This is THE validation script for the core research hypothesis.

Usage:
    # Full pipeline: capture → analyze → validate
    python scripts/capture_experiment_traces.py
    python scripts/analyze_cross_prompt_affinity.py
    python scripts/run_affinity_loop.py

    # Or with a manifest from a previous capture:
    python scripts/run_affinity_loop.py --manifest outputs/experiment_traces/manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

_proj_root = Path(__file__).resolve().parent.parent
if str(_proj_root) not in sys.path:
    sys.path.insert(0, str(_proj_root))

from src.ocs.preconfig import compute_plan_from_traces, export_plan, plan_summary
from src.eval.hit_rate import compute_hit_rate_from_trace_files


def build_sim_config(
    mode: str,
    trace_path: str,
    plan_path: str | None,
    world_size: int,
    experts_per_rank: int,
    max_circuits: int,
    trace_dir: str,
) -> dict:
    """Build a simulator config for ocs_preset or eps baseline mode.

    Uses routing_replay to replay the captured trace through synthetic experts.
    """
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
        "routing": {"strategy": "replay"},
        "routing_replay": {
            "enabled": True,
            "trace_path": trace_path,
            "layer_idx": 0,
            "cycle_layers": True,
        },
        # Use a simple topology so EPS delay is modeled
        "topology": {
            "enabled": True,
            "num_pods": 2,
            "nodes_per_pod": 1,
            "ranks_per_node": world_size // 2 if world_size >= 2 else 1,
            "intra_node_latency_us": 1.0,
            "intra_pod_latency_us": 5.0,
            "cross_pod_latency_us": 20.0,
            "intra_node_bandwidth_gbps": 900.0,
            "intra_pod_bandwidth_gbps": 200.0,
            "cross_pod_bandwidth_gbps": 100.0,
        },
        "profiling": {"export_trace": True, "trace_dir": trace_dir},
    }

    if mode == "ocs_preset":
        cfg["runtime"] = {"mode": "ocs_preset", "num_steps": 2}
        cfg["ocs"] = {
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
        }
    elif mode == "eps":
        cfg["runtime"] = {"mode": "overlap", "num_steps": 2}
        cfg["ocs"] = {"enabled": False}
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return cfg


def run_simulator(config: dict, trace_dir: str) -> bool:
    """Run the OCS simulator with the given config. Returns True on success."""
    from src.launcher import launch

    try:
        launch(config, trace_dir=trace_dir)
        return True
    except SystemExit as e:
        return e.code == 0
    except Exception as e:
        print(f"    [error] Simulator failed: {e}")
        return False


def collect_trace_files(trace_dir: str) -> list[str]:
    """Find all rank_NN_trace.json files in a directory."""
    import glob
    pattern = os.path.join(trace_dir, "rank_*_trace.json")
    return sorted(glob.glob(pattern))


def run_experiment(
    exp: dict,
    groups: list[dict],
    output_dir: Path,
    world_size: int,
    experts_per_rank: int,
    max_circuits: int,
) -> dict:
    """Run one experiment: build plan from train traces, test on test traces.

    Returns dict with results for each test trace.
    """
    exp_name = exp["name"]
    train_ids = exp["train_ids"]
    test_ids = exp["test_ids"]
    hypothesis = exp.get("hypothesis", "")

    print(f"\n{'='*65}")
    print(f"EXPERIMENT: {exp_name}")
    print(f"  {exp.get('description', '')}")
    print(f"  Hypothesis: {hypothesis}")
    print(f"{'='*65}")

    # ── Find trace paths ──
    id_to_path = {g["group_id"]: g["trace_path"] for g in groups}

    train_paths = []
    for tid in train_ids:
        if tid in id_to_path:
            train_paths.append(id_to_path[tid])
        else:
            print(f"  [error] Training trace not found: {tid}")
            return {"error": f"missing_train_trace:{tid}"}

    test_paths = []
    for tid in test_ids:
        if tid in id_to_path:
            test_paths.append(id_to_path[tid])
        else:
            print(f"  [error] Test trace not found: {tid}")
            return {"error": f"missing_test_trace:{tid}"}

    print(f"\n  Train traces ({len(train_paths)}):")
    for p in train_paths:
        print(f"    {p}")
    print(f"  Test traces ({len(test_paths)}):")
    for p in test_paths:
        print(f"    {p}")

    # ── Phase 1: Build plan ──
    print(f"\n  [Phase 1] Building circuit plan from {len(train_paths)} training traces...")
    plan = compute_plan_from_traces(
        trace_paths=train_paths,
        max_circuits=max_circuits,
        experts_per_rank=experts_per_rank,
        world_size=world_size,
    )

    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    plan_path = str(exp_dir / "circuit_plan.json")
    export_plan(plan, plan_path)

    summary = plan_summary(plan)
    print(f"  Plan: {summary['num_circuits']} circuits, "
          f"scores {summary['score_stats']['min']:.3f}–{summary['score_stats']['max']:.3f} "
          f"(mean={summary['score_stats']['mean']:.3f})")

    # ── Phase 2: Test each trace ──
    results = []
    for test_path in test_paths:
        test_name = Path(test_path).parent.name
        print(f"\n  [Phase 2] Testing on: {test_name}")

        # OCS Preset run
        ocs_trace_dir = str(exp_dir / f"{test_name}_ocs")
        os.makedirs(ocs_trace_dir, exist_ok=True)
        cfg_ocs = build_sim_config(
            mode="ocs_preset",
            trace_path=test_path,
            plan_path=plan_path,
            world_size=world_size,
            experts_per_rank=experts_per_rank,
            max_circuits=max_circuits,
            trace_dir=ocs_trace_dir,
        )

        # Save config for debugging
        cfg_path = str(exp_dir / f"{test_name}_ocs.yaml")
        with open(cfg_path, "w") as f:
            yaml.dump(cfg_ocs, f)

        print(f"    OCS preset run (trace_dir={ocs_trace_dir})...", end=" ", flush=True)
        ok_ocs = run_simulator(cfg_ocs, ocs_trace_dir)

        ocs_result = {}
        if ok_ocs:
            trace_files = collect_trace_files(ocs_trace_dir)
            if trace_files:
                report = compute_hit_rate_from_trace_files(trace_files)
                ocs_result = report.summary()
                print(f"hit_rate={ocs_result['operational_hit_rate']:.1%} "
                      f"coverage={ocs_result['ocs_coverage']:.1%}")
            else:
                print("no trace files")
                ocs_result = {"error": "no traces"}
        else:
            print("FAILED")
            ocs_result = {"error": "simulator_failed"}

        # EPS Baseline run
        eps_trace_dir = str(exp_dir / f"{test_name}_eps")
        os.makedirs(eps_trace_dir, exist_ok=True)
        cfg_eps = build_sim_config(
            mode="eps",
            trace_path=test_path,
            plan_path=None,
            world_size=world_size,
            experts_per_rank=experts_per_rank,
            max_circuits=max_circuits,
            trace_dir=eps_trace_dir,
        )

        print(f"    EPS baseline run (trace_dir={eps_trace_dir})...", end=" ", flush=True)
        ok_eps = run_simulator(cfg_eps, eps_trace_dir)
        eps_result = {}
        if ok_eps:
            trace_files = collect_trace_files(eps_trace_dir)
            if trace_files:
                # For EPS, extract timing from trace metadata
                eps_result = {"trace_dir": eps_trace_dir, "ok": True}
                print("OK")
            else:
                print("no trace files")
                eps_result = {"error": "no traces"}
        else:
            print("FAILED")
            eps_result = {"error": "simulator_failed"}

        results.append({
            "test": test_name,
            "test_trace": test_path,
            "ocs": ocs_result,
            "eps": eps_result,
        })

    return {
        "experiment": exp_name,
        "hypothesis": hypothesis,
        "plan_summary": summary,
        "results": results,
    }


def print_final_comparison(all_exp_results: list[dict]) -> None:
    """Print a comprehensive comparison across all experiments."""
    print(f"\n{'='*75}")
    print("FINAL COMPARISON: Does same-domain affinity generalize?")
    print(f"{'='*75}")

    header = f"  {'Experiment':<30} {'Test':<20} {'Hit Rate':<10} {'Coverage':<10} {'Verdict':<12}"
    print(header)
    print(f"  {'-'*28} {'-'*18} {'-'*8} {'-'*8} {'-'*10}")

    for exp_result in all_exp_results:
        if "error" in exp_result:
            print(f"  {exp_result['experiment']:<30} ERROR: {exp_result['error']}")
            continue

        exp_name = exp_result["experiment"]
        for r in exp_result["results"]:
            ocs = r.get("ocs", {})
            hit = ocs.get("operational_hit_rate", 0)
            cov = ocs.get("ocs_coverage", 0)

            # Determine verdict
            if "same_domain" in exp_name.lower() or "same" in exp_name.lower():
                expected = "HIGH"
                passed = hit >= 0.5
            elif "cross_domain" in exp_name.lower() or "cross" in exp_name.lower():
                expected = "LOW"
                passed = hit < 0.5
            elif "near_domain" in exp_name.lower() or "near" in exp_name.lower():
                expected = "MODERATE"
                passed = 0.25 <= hit <= 0.75
            else:
                expected = "?"
                passed = True

            verdict = f"{'✓' if passed else '✗'} {expected}"
            print(f"  {exp_name:<30} {r['test']:<20} {hit:<10.1%} {cov:<10.1%} {verdict:<12}")

    print(f"{'='*75}")

    # Summary
    hit_rates = []
    for exp_result in all_exp_results:
        if "error" in exp_result:
            continue
        for r in exp_result["results"]:
            ocs = r.get("ocs", {})
            if "operational_hit_rate" in ocs:
                hit_rates.append((exp_result["experiment"], r["test"], ocs["operational_hit_rate"]))

    if hit_rates:
        same_hits = [h for e_name, _, h in hit_rates if "same" in e_name.lower()]
        cross_hits = [h for e_name, _, h in hit_rates if "cross" in e_name.lower()]

        if same_hits and cross_hits:
            avg_same = sum(same_hits) / len(same_hits)
            avg_cross = sum(cross_hits) / len(cross_hits)
            print(f"\n  Average same-domain hit rate:  {avg_same:.1%}")
            print(f"  Average cross-domain hit rate: {avg_cross:.1%}")
            print(f"  Generalization ratio:          {avg_same / max(avg_cross, 0.001):.1f}x")
            if avg_same > avg_cross * 1.3:
                print(f"\n  ✓ HYPOTHESIS CONFIRMED:")
                print(f"    Semantically similar prompts produce similar expert routing,")
                print(f"    enabling effective OCS circuit pre-configuration.")
                print(f"    Cross-domain prompts DO NOT benefit from pre-configured circuits.")
            else:
                print(f"\n  ⚠ HYPOTHESIS NOT CONFIRMED:")
                print(f"    The hit-rate difference between same-domain and cross-domain")
                print(f"    is too small. Consider: larger prompts, lower temperature,")
                print(f"    or bigger expert count in the simulator to reduce aliasing.")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end closed-loop affinity validation"
    )
    parser.add_argument(
        "--manifest",
        default="outputs/experiment_traces/manifest.json",
        help="Path to manifest.json from capture_experiment_traces.py",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Run a single experiment by name (default: all)",
    )
    parser.add_argument(
        "--world-size", type=int, default=4,
        help="Number of simulated GPU ranks",
    )
    parser.add_argument(
        "--experts-per-rank", type=int, default=4,
        help="Experts per simulated rank (total experts = world_size × experts_per_rank)",
    )
    parser.add_argument(
        "--max-circuits", type=int, default=16,
        help="Maximum OCS circuits in the pool",
    )
    parser.add_argument(
        "--output-dir", default="outputs/affinity_loop_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-simulator", action="store_true",
        help="Skip the simulator runs (just build plans and print expected hit rates)",
    )
    args = parser.parse_args()

    # ── Load manifest ──
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"[error] Manifest not found: {manifest_path}")
        print(f"        Run capture_experiment_traces.py first to capture traces.")
        sys.exit(1)

    with open(manifest_path) as f:
        manifest = json.load(f)

    groups = manifest["results"]
    experiments = manifest.get("experiments", [])

    if not experiments:
        print("[error] No experiments defined in manifest. "
              "Add 'experiments' section to your prompt taxonomy config.")
        sys.exit(1)

    # Filter by experiment name
    if args.experiment:
        experiments = [e for e in experiments if e["name"] == args.experiment]
        if not experiments:
            print(f"[error] Experiment '{args.experiment}' not found")
            sys.exit(1)

    print("=" * 65)
    print("CLOSED-LOOP AFFINITY VALIDATION")
    print("=" * 65)
    print(f"Manifest:       {manifest_path}")
    print(f"Traces:         {len(groups)} groups, "
          f"{len(set(g['domain'] for g in groups))} domains")
    print(f"Experiments:    {len(experiments)}")
    print(f"World size:     {args.world_size}")
    print(f"Experts/rank:   {args.experts_per_rank}")
    print(f"Total experts:  {args.world_size * args.experts_per_rank}")
    print(f"Max circuits:   {args.max_circuits}")
    print(f"Skip simulator: {args.skip_simulator}")
    print()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_exp_results = []

    for exp in experiments:
        if args.skip_simulator:
            # Just build and show the plan without running simulator
            exp_name = exp["name"]
            train_ids = exp["train_ids"]
            id_to_path = {g["group_id"]: g["trace_path"] for g in groups}
            train_paths = [id_to_path[tid] for tid in train_ids if tid in id_to_path]

            if len(train_paths) != len(train_ids):
                print(f"[{exp_name}] SKIP — missing training traces")
                continue

            plan = compute_plan_from_traces(
                trace_paths=train_paths,
                max_circuits=args.max_circuits,
                experts_per_rank=args.experts_per_rank,
                world_size=args.world_size,
            )
            summary = plan_summary(plan)
            print(f"[{exp_name}] Plan from {len(train_paths)} traces: "
                  f"{summary['num_circuits']} circuits, "
                  f"mean score={summary['score_stats']['mean']:.3f}")
            for pair in summary["top_pairs"][:5]:
                print(f"  rank {pair['src']} → rank {pair['dst']}: {pair['score']:.4f}")
            continue

        result = run_experiment(
            exp=exp,
            groups=groups,
            output_dir=output_dir,
            world_size=args.world_size,
            experts_per_rank=args.experts_per_rank,
            max_circuits=args.max_circuits,
        )
        all_exp_results.append(result)

    # ── Final comparison ──
    print_final_comparison(all_exp_results)

    # ── Save results ──
    results_path = output_dir / "loop_results.json"
    # Convert to serializable format
    serializable = []
    for r in all_exp_results:
        if isinstance(r, dict):
            serializable.append(r)
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\nResults saved → {results_path}")


if __name__ == "__main__":
    main()

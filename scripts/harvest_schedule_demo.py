#!/usr/bin/env python3
"""
harvest_schedule_demo.py — trace → per-layer steps → Harvest DP → report.

Additive demo for the trace-guided OCS scheduling idea (Harvest paper /
arXiv:2602.09188 reproduced in ~/Downloads/Projects/harvest):

    routing cells (real inference)
        → per-layer dispatch demand  (placement/topology-independent, Q1)
        → restricted circuit pool    (eps / fit-static / per-layer plans)
        → DCT per (layer, plan)      (repo's own bottleneck model)
        → Harvest DP + Theorem-1 sweep over rewire count
        → when to repoint the OCS, and whether it pays at all (k* = 0 or not)

No MILP: pool + analytic scoring only.  Nothing in the existing codebase is
modified — this script only reads traces and writes a new report file.

Usage (from the repo root):
    .venv/bin/python scripts/harvest_schedule_demo.py \
        --workload logs/workload/smoke --world-size 20 --topology multi_pod

    # real workload, EPS-only regime (expects k* = 0 — the honest negative)
    .venv/bin/python scripts/harvest_schedule_demo.py \
        --workload logs/workload/qwen36 --world-size 32 --topology realistic
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from src.eval.cost_model import CostConfig, DispatchMode, Placement, hierarchy_for
from src.eval.harvest_sched import schedule_report
from src.eval.ocs_eval import OcsConfig
from src.eval.placement_opt import make_placement
from src.eval.trace_ir import load_workload


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workload", required=True,
                    help="dir containing manifest.json + traces/")
    ap.add_argument("--world-size", type=int, default=20, help="EP ranks (E %% W == 0)")
    ap.add_argument("--topology", default="multi_pod",
                    choices=("single_node", "single_pod", "multi_pod", "realistic"))
    ap.add_argument("--gpus-per-node", type=int, default=None)
    ap.add_argument("--nodes-per-pod", type=int, default=None)
    ap.add_argument("--max-runs", type=int, default=8)
    ap.add_argument("--decode-only", action="store_true")
    ap.add_argument("--mode", default="dedup_rank",
                    choices=("replicated", "dedup_rank", "dedup_node"))
    ap.add_argument("--placement", default="linear")
    ap.add_argument("--n-circuits", type=int, default=8)
    ap.add_argument("--ports-per-rank", type=int, default=1)
    ap.add_argument("--alpha-r-us", type=float, default=None,
                    help="if set, single switch class (us); else sweep all classes")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="report JSON path")
    args = ap.parse_args()

    root = Path(args.workload)
    t = load_workload(root / "manifest.json", max_runs=args.max_runs,
                      decode_only=args.decode_only)
    E = t.num_experts
    if E % args.world_size:
        sys.exit(f"E={E} not divisible by world_size={args.world_size}; "
                 f"choose a divisor of {E}")
    print(f"loaded {t}  ({t.n_cells} cells over {t.n_layers} layers)")

    placement = make_placement(args.placement, t, args.world_size,
                               seed=args.seed)
    topo = hierarchy_for(args.world_size, args.topology)
    if args.gpus_per_node is not None or args.nodes_per_pod is not None:
        from src.eval.cost_model import Topology
        topo = Topology(
            args.world_size,
            gpus_per_node=args.gpus_per_node or topo.gpus_per_node,
            nodes_per_pod=args.nodes_per_pod or topo.nodes_per_pod,
            fabric=topo.fabric, circuits=set(), rank_to_slot=topo.rank_to_slot)
    ocs = OcsConfig(n_circuits=args.n_circuits,
                    ports_per_rank=args.ports_per_rank,
                    reconfig_us=args.alpha_r_us if args.alpha_r_us else 10_000.0)

    mode = DispatchMode[args.mode.upper()]
    rep = schedule_report(t, placement, topo, ocs,
                          cost=CostConfig(), mode=mode,
                          n_dp=args.world_size, seed=args.seed,
                          alpha_r_us=args.alpha_r_us)

    print(f"\ntopology  : {rep['topology']}")
    print(f"pool      : {[m['name'] for m in rep['pool']]}")
    print(f"eps_static: {rep['eps_static_us']} us | "
          f"pool_static: {rep['pool_static_us']} us "
          f"(over {rep['n_steps']} layers)")
    print(f"\n{'switch class':<12} {'alpha_r':>9} {'rewires':>7} {'harvest_us':>12} "
          f"{'static_us':>11} {'saved%':>8}  note")
    for name, r in rep["alpha_r_sweep"].items():
        note = ("k*=0: static plan certified"
                if r["certifies_static_k0"]
                else "rewire pays: schedule per window")
        print(f"{name:<12} {r['alpha_r_us']:>9.0f} "
              f"{r['harvest']['rewires']:>7} "
              f"{r['harvest']['total_us']:>12.2f} "
              f"{rep['eps_static_us']:>11.2f} "
              f"{r['saved_vs_eps_static_pct'] or 0:>7.2f}%  {note}")
        if r["harvest"]["rewires"]:
            layers = rep["steps"]
            print("    harvest schedule: "
                  + " -> ".join(f"{s['member']}@L{layers[s['a']]}-"
                                f"L{layers[s['b']]}"
                                for s in r["harvest"]["segment_list"]))

    out = args.out or (root.parent / "harvest_schedule"
                       / f"{root.name}_ws{args.world_size}_{args.topology}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(rep, indent=2, default=str))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()

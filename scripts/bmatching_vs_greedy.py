#!/usr/bin/env python3
"""bmatching_vs_greedy.py — Phase 6: greedy vs LP vs exact MILP circuit selection.

Tests two things at once:

1. **The repo's V11 claim** that the b-matching polytope is integral, so the LP is
   exact.  The rank graph is complete (not bipartite) and the plan also carries a
   cardinality constraint, neither of which is covered by the integrality
   argument — so this is measured, not assumed.
2. **Whether the greedy 1/2-approximation actually leaves anything on the table**
   in the quantity that matters: the OCS gain on a held-out window.

Usage
─────
    python3.12 scripts/bmatching_vs_greedy.py --workload logs/workload/qwen36 --world-size 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "scripts"))

import src.eval.cost_model as cm  # noqa: E402
from src.eval.bmatching import plan_circuits_lp, plan_circuits_milp  # noqa: E402
from src.eval.cost_model import (  # noqa: E402
    CostConfig, DispatchMode, FabricConfig, Tier, Topology, evaluate,
    hierarchy_for, traffic_matrix,
)
from src.eval.ocs_eval import OcsConfig, plan_circuits, with_circuits  # noqa: E402
from src.eval.placement_opt import make_placement  # noqa: E402
from src.eval.trace_ir import load_workload  # noqa: E402
from src.serving.suite import build_suite, split_by_category  # noqa: E402
from ownership_ocs_envelope import measured_k, patch_token_rank, restore_token_rank  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measured", type=Path,
                    default=Path("outputs/ownership/multi_tenant_ownership.json"))
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cm.token_rank_original = cm.token_rank
    W, seed = args.world_size, args.seed
    k = args.k or measured_k(args.measured) or 4
    models = ["hash", f"measured_packed_{k}"]

    t = load_workload(args.workload / "manifest.json", decode_only=True)
    specs = build_suite(n_repeats=0)
    fu, eu = split_by_category(specs, seed=seed)
    present = {r.uid for r in t.runs}
    fit = t.by_runs([u for u in fu if u in present])
    ev = t.by_runs([u for u in eu if u in present])

    cost = CostConfig(hidden_size=2048)
    base = hierarchy_for(W, "multi_pod")
    fab = FabricConfig(core_oversubscription=4.0, pod_oversubscription=1.0)
    topo = Topology(W, base.gpus_per_node, base.nodes_per_pod, fab, set(),
                    base.rank_to_slot, (Tier.CROSS_POD,))
    cfg = OcsConfig(n_circuits=max(4, W // 2), ports_per_rank=2)

    print(f"== circuit selection: greedy vs LP vs exact MILP  "
          f"({args.workload.name}, W={W}, {cfg.n_circuits} circuits, "
          f"{cfg.ports_per_rank} ports/rank)\n")
    rows: dict = {}
    t0 = time.time()
    try:
        for model in models:
            patch_token_rank(model, k)
            for kind in ("linear", "affinity_coordinated_layer"):
                p = make_placement(kind, fit, W, seed=seed)
                tm = traffic_matrix(fit, p, topo, DispatchMode.DEDUP_RANK,
                                    n_dp=W, seed=seed)
                eps = evaluate(ev, p, topo, cost, DispatchMode.DEDUP_RANK,
                               n_dp=W, seed=seed)["bottleneck_us"]
                plans = {
                    "greedy": plan_circuits(tm.counts, topo, cfg),
                    "lp": plan_circuits_lp(tm.counts, topo, cfg),
                    "milp": plan_circuits_milp(tm.counts, topo, cfg),
                }
                for name, (plan, info) in plans.items():
                    oc = evaluate(ev, p, with_circuits(topo, plan), cost,
                                  DispatchMode.DEDUP_RANK, n_dp=W, seed=seed)["bottleneck_us"]
                    gain = 100.0 * (1 - oc / eps) if eps else None
                    rows[f"{model}|{kind}|{name}"] = {
                        "ocs_us": round(oc, 2), "eps_us": round(eps, 2),
                        "ocs_gain_pct": None if gain is None else round(gain, 4),
                        **{q: info.get(q) for q in (
                            "n_candidate_promotable_pairs", "n_circuits_provisioned",
                            "promotable_traffic_covered_fraction", "port_saturated_ranks",
                            "fractional_edges", "polytope_integral", "lp_objective",
                            "milp_objective", "status")},
                    }
    finally:
        restore_token_rank()

    print(f"{'source':<18}{'placement':<26}{'method':<8}{'cover':>8}{'gain%':>8}"
          f"{'circuits':>9}{'frac_edges':>11}{'obj':>12}")
    for model in models:
        for kind in ("linear", "affinity_coordinated_layer"):
            for name in ("greedy", "lp", "milp"):
                v = rows[f"{model}|{kind}|{name}"]
                obj = v.get("lp_objective") if name == "lp" else v.get("milp_objective")
                g = v["ocs_gain_pct"]
                print(f"{model:<18}{kind:<26}{name:<8}"
                      f"{v['promotable_traffic_covered_fraction']:>8.4f}"
                      f"{(f'{g:.3f}' if g is not None else 'n/a'):>8}"
                      f"{v['n_circuits_provisioned']:>9}"
                      f"{(v.get('fractional_edges') if name == 'lp' else '-'):>11}"
                      f"{(f'{obj:.1f}' if obj is not None else '-'):>12}")
            print()

    # the verdict on the integrality claim
    lps = [v for kk, v in rows.items() if kk.endswith("|lp")]
    frac = sum(int(v.get("fractional_edges") or 0) for v in lps)
    print(f"   integrality: {'NO fractional edges anywhere' if frac == 0 else f'{frac} fractional edges'}"
          f" -> the LP {'IS' if frac == 0 else 'IS NOT'} exact on these instances")
    print(f"   ({time.time()-t0:.1f} s)")

    dest = args.out or Path(f"outputs/affinity/bmatching_vs_greedy.{args.workload.name}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "workload": str(args.workload), "world_size": W, "k": k,
        "config": {"n_circuits": cfg.n_circuits, "ports_per_rank": cfg.ports_per_rank},
        "rows": rows, "total_fractional_edges": frac}, indent=1))
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

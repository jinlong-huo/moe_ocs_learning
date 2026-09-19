#!/usr/bin/env python3
"""plan_window_stability.py — is the static plan still valid on the next window?

```ownership_ocs_envelope.py``` fits the circuit plan on the **fit** window and
scores it on the **eval** window (the out-of-sample discipline the plan doc
requires).  One ownership model breaks the pattern: under ```per_sequence```
the plan covers 10 % of promotable traffic yet moves the bottleneck by 0.03-0.09 %.
Covering traffic only helps if the covered *pairs* are the binding ones, so this
script measures the thing that decides it -- the overlap between the plan the
fit window wants and the plan the eval window wants.

    Jaccard = |plan(fit) INTERSECT plan(eval)| / |plan(fit) UNION plan(eval)|

Low overlap = the static plan is stale and the gain is not a property of the
ownership model but of window-to-window drift.  Additive: imports and
monkey-patches ```token_rank```, modifies nothing.

Usage
─────
    python3.12 scripts/plan_window_stability.py --workload logs/workload/qwen36 --world-size 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "scripts"))

import src.eval.cost_model as cm  # noqa: E402
from src.eval.cost_model import (  # noqa: E402
    CostConfig, DispatchMode, FabricConfig, Tier, Topology, hierarchy_for, traffic_matrix,
)
from src.eval.ocs_eval import OcsConfig, plan_circuits  # noqa: E402
from src.eval.placement_opt import make_placement  # noqa: E402
from src.eval.trace_ir import load_workload  # noqa: E402
from src.serving.suite import build_suite, split_by_category  # noqa: E402
from ownership_ocs_envelope import (  # noqa: E402
    PLACEMENTS, measured_k, patch_token_rank, restore_token_rank,
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measured", type=Path,
                    default=Path("outputs/ownership/multi_tenant_ownership.json"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cm.token_rank_original = cm.token_rank
    W = args.world_size
    k = measured_k(args.measured) or 4
    models = ["hash", "per_sequence", f"measured_packed_{k}", f"measured_spread_{k}"]

    t = load_workload(args.workload / "manifest.json", decode_only=True)
    specs = build_suite(n_repeats=0)
    fu, eu = split_by_category(specs, seed=args.seed)
    present = {r.uid for r in t.runs}
    fit = t.by_runs([u for u in fu if u in present])
    ev = t.by_runs([u for u in eu if u in present])

    cost = CostConfig(hidden_size=2048)
    base = hierarchy_for(W, "multi_pod")
    fab = FabricConfig(core_oversubscription=4.0, pod_oversubscription=1.0)
    topo = Topology(W, base.gpus_per_node, base.nodes_per_pod, fab, set(),
                    base.rank_to_slot, (Tier.CROSS_POD,))
    cfg = OcsConfig(n_circuits=max(4, W // 2), ports_per_rank=2)

    print(f"== plan-window stability: {args.workload.name} world={W} "
          f"(plan fitted on A, wanted on B; {cfg.n_circuits} circuits)")
    print(f"{'source':<18}{'placement':<26}{'jaccard':>9}{'|A|':>5}{'|B|':>5}"
          f"{'shared':>8}")
    rows = {}
    try:
        for model in models:
            patch_token_rank(model, k)
            for kind in PLACEMENTS:
                p = make_placement(kind, fit, W, seed=args.seed)
                pa, ia = plan_circuits(
                    traffic_matrix(fit, p, topo, DispatchMode.DEDUP_RANK,
                                   n_dp=W, seed=args.seed).counts, topo, cfg)
                pb, ib = plan_circuits(
                    traffic_matrix(ev, p, topo, DispatchMode.DEDUP_RANK,
                                   n_dp=W, seed=args.seed).counts, topo, cfg)
                shared = len(pa & pb)
                jac = shared / max(len(pa | pb), 1)
                rows[f"{model}|{kind}"] = {
                    "jaccard_fit_vs_eval": round(jac, 4),
                    "n_fit": len(pa), "n_eval": len(pb), "n_shared": shared,
                    "covered_fraction": ia["promotable_traffic_covered_fraction"],
                }
                print(f"{model:<18}{kind:<26}{jac:>9.4f}{len(pa):>5}{len(pb):>5}{shared:>8}")
            print()
    finally:
        restore_token_rank()

    print("   cost_model.token_rank restored")
    dest = args.out or Path(f"outputs/ownership/plan_window_stability.{args.workload.name}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"workload": str(args.workload), "world_size": W,
                                "k": k, "rows": rows}, indent=1))
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

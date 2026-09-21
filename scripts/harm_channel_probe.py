#!/usr/bin/env python3
"""harm_channel_probe.py -- P0: does the closed-form model contain ANY channel
through which a circuit set can make the makespan worse?

Claim under test (docs/hot_spine_ocs.md section 2.1, read out of cost_model.py):

    drain = sum over pairs of bytes/bandwidth(tier)          (l.458-463)
    bandwidth is non-decreasing under promotion (12.5 -> 50)  (l.99-106)
    alpha  = latency(max tier present)                        (l.469-470)
    OPTICAL (idx 3) has the LOWEST latency (6 us < 12 us)     (l.118-124)
  =>  M(C) <= M(empty) for every circuit set C.

If that holds for 400 adversarial + random sets on real traces, then "hot-only is
harmless" is arithmetic in this model, not a property of the rule, and the rule
needs a harm channel (port displacement / reconfiguration) before it can be
tested.  If it fails, the plan changes before anything is built.

Nothing existing is modified: this script only calls evaluate().

Usage
    python3.12 scripts/harm_channel_probe.py --workload logs/workload/qwen36 --world-size 32
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_repo_root / "scripts"))

import src.eval.cost_model as cm  # noqa: E402
from src.eval.cost_model import (  # noqa: E402
    CostConfig, DispatchMode, FabricConfig, Tier, Topology, evaluate,
    hierarchy_for, traffic_matrix,
)
from src.eval.placement_opt import make_placement  # noqa: E402
from src.eval.trace_ir import load_workload  # noqa: E402
from ownership_ocs_envelope import patch_token_rank, restore_token_rank  # noqa: E402

TOL = 1e-9


def candidate_pairs(mat: np.ndarray, topo: Topology) -> list:
    """Promotable unordered pairs, ranked by symmetric bytes."""
    W = mat.shape[0]
    T = topo.tier_matrix()
    promotable = {int(x) for x in topo.promote_from
                  if topo.fabric.oversubscription(Tier(int(x))) > 1.0}
    out = []
    for a in range(W):
        for b in range(a + 1, W):
            if int(T[a, b]) in promotable:
                w = float(mat[a, b] + mat[b, a])
                if w > 0:
                    out.append((frozenset((a, b)), w))
    out.sort(key=lambda t: -t[1])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--source-model", default="measured_packed_4")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--n-random", type=int, default=200)
    ap.add_argument("--n-adversarial", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--slice", default="layer0,full")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    W = args.world_size
    cm.token_rank_original = cm.token_rank
    cost = CostConfig(hidden_size=2048)
    base = hierarchy_for(W, "multi_pod")
    fab = FabricConfig(core_oversubscription=4.0, pod_oversubscription=1.0)

    def topo_with(circuits):
        return Topology(W, base.gpus_per_node, base.nodes_per_pod, fab, set(circuits),
                        base.rank_to_slot, (Tier.CROSS_POD,))

    t_all = load_workload(args.workload / "manifest.json", decode_only=True)
    layer0 = t_all.by_layer(int(np.sort(np.unique(t_all.layers))[0]))
    slices = {"layer0": layer0, "full": t_all}

    rng = np.random.default_rng(args.seed)
    report = {"workload": str(args.workload), "world_size": W,
              "source_model": args.source_model, "tol": TOL, "slices": {}}

    try:
        patch_token_rank(args.source_model, args.k)
        pl = make_placement("linear", layer0, W, seed=args.seed)
        for sname in [s.strip() for s in args.slice.split(",") if s.strip()]:
            t = slices[sname]
            tm = traffic_matrix(t, pl, topo_with([]), DispatchMode.DEDUP_RANK, n_dp=W, seed=args.seed)
            mat = tm.counts * (cost.hidden_size * cost.dtype_bytes)
            cands = candidate_pairs(mat, topo_with([]))
            m0 = evaluate(t, pl, topo_with([]), cost, DispatchMode.DEDUP_RANK, n_dp=W,
                          seed=args.seed)["bottleneck_us"]

            sets = {}
            sets["empty"] = []
            sets["all"] = [p for p, _ in cands]
            for k in (8, 16, 32):
                sets[f"hot_top{k}"] = [p for p, _ in cands[:k]]
                sets[f"cold_bottom{k}"] = [p for p, _ in cands[-k:]] if len(cands) >= k else []
            # adversarial: pairs that touch the rank holding the most bytes (the
            # place where a circuit has the best chance of changing the max)
            busiest = int(np.argmax(mat.sum(1)))
            sets["busiest_rank"] = [p for p, _ in cands if busiest in p]
            for i in range(args.n_random):
                k = int(rng.integers(0, max(1, len(cands)) + 1))
                idx = rng.choice(len(cands), size=min(k, len(cands)), replace=False)
                sets[f"random_{i:03d}"] = [cands[j][0] for j in idx]
            for i in range(args.n_adversarial):
                # every pair touching the busiest rank, plus noise
                s = [p for p, _ in cands if busiest in p]
                pool = [p for p, _ in cands if p not in s]
                if pool:
                    j = int(rng.integers(0, len(pool)))
                    s.append(pool[j])
                sets[f"adversarial_{i:03d}"] = s

            rows, worst = {}, None
            for name, circ in sets.items():
                m = evaluate(t, pl, topo_with(circ), cost, DispatchMode.DEDUP_RANK,
                             n_dp=W, seed=args.seed)["bottleneck_us"]
                d = m - m0
                rows[name] = {"bottleneck_us": round(float(m), 6), "delta_us": round(float(d), 6),
                              "n_circuits": len(circ)}
                if worst is None or d > worst[1]:
                    worst = (name, d, m)
            viol = [n for n, r in rows.items() if r["delta_us"] > TOL]
            report["slices"][sname] = {
                "n_candidate_promotable_pairs": len(cands),
                "baseline_us": round(float(m0), 6),
                "n_sets": len(sets),
                "violations": viol,
                "worst_set": worst[0], "worst_delta_us": round(float(worst[1]), 9),
                "worst_bottleneck_us": round(float(worst[2]), 6),
                "hot_top16_delta_us": rows["hot_top16"]["delta_us"],
                "cold_bottom16_delta_us": rows["cold_bottom16"]["delta_us"],
                "all_delta_us": rows["all"]["delta_us"],
                "rows": rows,
            }
            print(f"== {sname}: candidates={len(cands)} baseline={m0:.4f} us")
            print(f"   sets={len(sets)}  violations={len(viol)}  "
                  f"worst=({worst[0]}, {worst[1]:+.9f} us)")
            print(f"   hot16 {rows['hot_top16']['delta_us']:+.6f}  "
                  f"cold16 {rows['cold_bottom16']['delta_us']:+.6f}  "
                  f"all {rows['all']['delta_us']:+.6f}")
            if viol:
                print(f"   VIOLATION SETS: {viol[:5]}")
    finally:
        restore_token_rank()

    dest = args.out or Path(f"outputs/spine/harm_channel_probe.{args.workload.name}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=1))
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

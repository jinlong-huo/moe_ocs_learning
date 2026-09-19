#!/usr/bin/env python3
"""bottleneck_attribution.py — which resource is the bottleneck, and is it the
one a circuit can promote?

WHY
───
```ocs_moe_program.md``` prices a circuit with

    Delta_oracle = f * (1 - 1/sigma),   f = share of *bottleneck* bytes on
                                        promotable (oversubscribed) pairs

but the pipeline only reports the *global* promotable share
(```promotable_traffic_covered_fraction```).  Those are different quantities, and
the ownership sweep makes the difference visible: under ```per_sequence```
ownership the global covered fraction is 0.103 (=> 7.7 % predicted) while the
measured OCS gain is 0.088 %.  The formula is not wrong; ```f``` is.

```cost_model.evaluate``` takes the bottleneck as

    max(egress_nic, ingress_nic, egress_nvl, ingress_nvl)

so the binding resource can be **NVLink**, not the oversubscribed cross-pod NIC.
Promoting a cross-pod pair is worth exactly nothing when the wall is a rank's
NVLink port, however much traffic sits on promotable pairs.

This script recomputes those four resources (same arithmetic, read out rather
than edited — no existing module is modified) and reports which one binds, plus
the promotable share *of the binding resource*.  That share, not the global one,
is the ```f``` the OCS claim needs.

Usage
─────
    python3.12 scripts/bottleneck_attribution.py --workload logs/workload/qwen36 \
        --world-size 32 --out outputs/ownership/bottleneck_attribution.qwen36.json
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

import src.eval.cost_model as cm  # noqa: E402
from src.eval.cost_model import (  # noqa: E402
    CostConfig, DispatchMode, FabricConfig, Tier, Topology, hierarchy_for, traffic_matrix,
)
from src.eval.placement_opt import make_placement  # noqa: E402
from src.eval.trace_ir import load_workload  # noqa: E402
from src.serving.suite import build_suite, split_by_category  # noqa: E402

sys.path.insert(0, str(_repo_root / "scripts"))
from ownership_ocs_envelope import (  # noqa: E402
    PLACEMENTS, compute_src, measured_k, patch_token_rank, restore_token_rank,
)


def attribute(t, placement, topo, cost, mode, n_dp: int, seed: int) -> dict:
    """The four bottleneck resources + which binds + its promotable share."""
    tm = traffic_matrix(t, placement, topo, mode, n_dp=n_dp, seed=seed)
    B = cost.hidden_size * cost.dtype_bytes
    bytes_mat = tm.counts * B

    T = topo.tier_matrix()
    W, D = tm.world_size, tm.n_dp
    Tb = T[:D, :W]
    bw = np.array([topo.fabric.bandwidth(Tier(i)) for i in range(4)])

    # self-pairs are "local" (no network).  evaluate() writes loc = ~off, where
    # off = ~eye, i.e. loc is the DIAGONAL.  Inverting this zeroes every
    # cross-rank byte and makes NVLink look like the bottleneck everywhere.
    loc = np.zeros_like(Tb, dtype=bool)
    loc[:min(D, W), :min(D, W)] = np.eye(min(D, W), dtype=bool)
    net = bytes_mat.copy()
    net[loc] = 0.0

    nvl = (Tb == int(Tier.INTRA_NODE)) & ~loc
    egress_nvl = (net * nvl).sum(1) / topo.fabric.intra_node_gbytes_per_s
    eff = np.where(nvl, np.inf, bw[Tb])
    with np.errstate(divide="ignore", invalid="ignore"):
        drain = np.where(np.isfinite(eff), net / eff, 0.0)
    egress_nic = drain.sum(1)
    ingress_nic = drain.sum(0)
    ingress_nvl = (net * nvl).sum(0) / topo.fabric.intra_node_gbytes_per_s

    resources = {
        "egress_nic": (egress_nic, 1),
        "ingress_nic": (ingress_nic, 0),
        "egress_nvl": (egress_nvl, 1),
        "ingress_nvl": (ingress_nvl, 0),
    }
    binding, (vec, axis) = max(resources.items(), key=lambda kv: float(kv[1][0].max()))
    rank = int(np.argmax(vec))

    # what fraction of the binding rank's bytes sit on a promotable pair?
    promotable = {int(x) for x in topo.promote_from
                  if topo.fabric.oversubscription(Tier(int(x))) > 1.0}
    if axis == 1:                      # egress: the rank's row
        row, tier_row = net[rank, :], Tb[rank, :]
    else:                              # ingress: the rank's column
        row, tier_row = net[:, rank], Tb[:, rank]
    tot = float(row.sum())
    promo = float(row[np.isin(tier_row, list(promotable))].sum())

    return {
        "binding_resource": binding,
        "binding_rank": rank,
        "binding_us": round(float(vec.max()) / 1e3, 4),
        "promotable_share_of_binding": round(promo / tot, 6) if tot > 0 else 0.0,
        "cross_pod_bytes": float(bytes_mat[Tb == int(Tier.CROSS_POD)].sum()),
        "intra_node_bytes": float(bytes_mat[nvl].sum()),
        "nic_max_us": round(float(max(egress_nic.max(), ingress_nic.max())) / 1e3, 4),
        "nvl_max_us": round(float(max(egress_nvl.max(), ingress_nvl.max())) / 1e3, 4),
    }


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
    W = args.world_size
    k = args.k or measured_k(args.measured) or 4
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

    print(f"== bottleneck attribution: {args.workload.name} world={W} "
          f"n_dp={W} k={k}")
    print(f"{'source':<18}{'placement':<26}{'binds':>12}{'bind_us':>10}"
          f"{'promo_share':>12}{'nic_us':>9}{'nvl_us':>9}")
    rows = {}
    try:
        for model in models:
            patch_token_rank(model, k)
            for kind in PLACEMENTS:
                p = make_placement(kind, fit, W, seed=args.seed)
                r = attribute(ev, p, topo, cost, DispatchMode.DEDUP_RANK, W, args.seed)
                rows[f"{model}|{kind}"] = r
                print(f"{model:<18}{kind:<26}{r['binding_resource']:>12}"
                      f"{r['binding_us']:>10.2f}{r['promotable_share_of_binding']:>12.4f}"
                      f"{r['nic_max_us']:>9.1f}{r['nvl_max_us']:>9.1f}")
            print()
    finally:
        restore_token_rank()

    print("   cost_model.token_rank restored")
    dest = args.out or Path(f"outputs/ownership/bottleneck_attribution.{args.workload.name}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "workload": str(args.workload), "world_size": W, "k": k,
        "note": ("f in Delta = f*(1-1/sigma) is the promotable share of the "
                 "*binding* resource, not of all traffic; read out of evaluate()'s "
                 "arithmetic, nothing modified"),
        "rows": rows}, indent=1))
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

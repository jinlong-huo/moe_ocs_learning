#!/usr/bin/env python3
"""contention_before_after.py — how much of the OCS gain survives the two fixes?

Phase 3.  Runs the SAME configuration through:

    current                 src/eval/cost_model.evaluate()  (the reference)
    fair_share              + rate = uncontended(tier) / max(1, min(sigma_tier, k_rank))
    fair_share+cap @ W      + rate <= W / RTT(tier), swept over W

The first row is also a **self-check**: with both fixes off the overlay must
reproduce ```evaluate()``` exactly, or it is measuring something other than the
two fixes.  (It caught two real bugs on the first run: intra-node bytes counted in
the NIC drain, and the CORE sigma applied to the POD tier.)

The per-flow cap needs an outstanding-bytes figure, and any single choice would be
arbitrary — so it is **swept**, and the interesting output is the window at which
the cap stops binding rather than one number.

Independent evidence that these idealisations matter: on the ASTRA-sim 16-NPU toy
cycles = busy + 240*latency, so a third of the time is not bandwidth-proportional
(docs/astra_ordering.md).

Usage
─────
    python3.12 scripts/contention_before_after.py --workload logs/workload/qwen36 --world-size 32
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
from src.eval.cost_model import (  # noqa: E402
    CostConfig, DispatchMode, FabricConfig, Tier, Topology, evaluate,
    hierarchy_for, traffic_matrix,
)
from src.eval.contention import ContentionConfig, evaluate_contended  # noqa: E402
from src.eval.ocs_eval import OcsConfig, plan_circuits  # noqa: E402
from src.eval.placement_opt import make_placement  # noqa: E402
from src.eval.trace_ir import load_workload  # noqa: E402
from src.serving.suite import build_suite, split_by_category  # noqa: E402
from ownership_ocs_envelope import measured_k, patch_token_rank, restore_token_rank  # noqa: E402

WINDOWS_KIB = (64, 256, 1024, 4096, 16384)


def variants() -> dict:
    v = {
        "current": ContentionConfig(fair_share=False, per_flow_cap=False),
        "fair_share": ContentionConfig(fair_share=True, per_flow_cap=False),
    }
    for k in WINDOWS_KIB:
        v[f"fs+cap@{k}KiB"] = ContentionConfig(fair_share=True, per_flow_cap=True,
                                               window_bytes=k * 1024.0)
    return v


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
    models = ["hash", "per_sequence", f"measured_packed_{k}"]
    VAR = variants()

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

    print(f"== contention before/after: {args.workload.name} world={W} k={k} "
          f"sigma_core={fab.core_oversubscription:g} sigma_pod={fab.pod_oversubscription:g}\n")

    rows: dict = {}
    selfcheck: dict = {}
    t0 = time.time()
    try:
        for model in models:
            patch_token_rank(model, k)
            for kind in ("linear", "affinity_coordinated_layer"):
                p = make_placement(kind, fit, W, seed=args.seed)
                tm = traffic_matrix(fit, p, topo, DispatchMode.DEDUP_RANK,
                                    n_dp=W, seed=args.seed)
                plan, _ = plan_circuits(tm.counts, topo, cfg)
                ref = evaluate(ev, p, topo, cost, DispatchMode.DEDUP_RANK,
                               n_dp=W, seed=args.seed)["bottleneck_us"]
                for name, cc in VAR.items():
                    eps = evaluate_contended(ev, p, topo, cost, DispatchMode.DEDUP_RANK,
                                             W, args.seed, circuits=None, cc=cc)["bottleneck_us"]
                    oc = evaluate_contended(ev, p, topo, cost, DispatchMode.DEDUP_RANK,
                                            W, args.seed, circuits=plan, cc=cc)["bottleneck_us"]
                    gain = 100.0 * (1 - oc / eps) if eps else None
                    rows[f"{model}|{kind}|{name}"] = {
                        "eps_us": round(eps, 2), "ocs_us": round(oc, 2),
                        "ocs_gain_pct": None if gain is None else round(gain, 4),
                    }
                    if name == "current" and kind == "linear":
                        selfcheck[model] = {
                            "reference_evaluate_us": round(ref, 2),
                            "overlay_us": round(eps, 2),
                            "rel_err": round(abs(eps - ref) / max(ref, 1e-9), 12),
                        }
    finally:
        restore_token_rank()

    print("self-check (both fixes OFF must equal evaluate()):")
    ok = True
    for m, v in selfcheck.items():
        good = v["rel_err"] < 1e-9
        ok = ok and good
        print(f"   {m:<18} evaluate {v['reference_evaluate_us']:>10.2f}  overlay "
              f"{v['overlay_us']:>10.2f}  rel_err {v['rel_err']:.1e}  "
              f"{'OK' if good else 'MISMATCH'}")
    print()

    for kind in ("linear", "affinity_coordinated_layer"):
        print(f"-- placement: {kind}")
        hdr = f"   {'source':<18}"
        for name in VAR:
            hdr += f"{name:>16}"
        print(hdr)
        for model in models:
            line = f"   {model:<18}"
            for name in VAR:
                g = rows[f"{model}|{kind}|{name}"]["ocs_gain_pct"]
                line += f"{(f'{g:.3f}' if g is not None else 'n/a'):>16}"
            print(line)
        print()
    print(f"   ({time.time()-t0:.1f} s)  cost_model.token_rank restored")

    dest = args.out or Path(f"outputs/ownership/contention_before_after.{args.workload.name}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({
        "workload": str(args.workload), "world_size": W, "k": k,
        "sigma_core": fab.core_oversubscription, "sigma_pod": fab.pod_oversubscription,
        "windows_kib": list(WINDOWS_KIB), "selfcheck_ok": ok,
        "selfcheck": selfcheck, "rows": rows}, indent=1))
    print(f"-> {dest}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

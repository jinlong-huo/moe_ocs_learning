#!/usr/bin/env python3
"""ownership_ocs_envelope.py — what does the measured source support do to the
OCS answer?

Written as an ADDITIVE layer: it imports the pipeline, monkey-patches
```cost_model.token_rank``` for the duration of the run, and restores it in a
```finally```.  No existing module is modified.  (Same technique as
```scripts/source_model_sensitivity.py```, which this script deliberately copies
rather than edits, so the earlier result stays reproducible.)

THE SOURCE MODELS
─────────────────
    hash                (run*1_000_003 + pos) % n_dp   — the repo default
    per_sequence        run % n_dp                     — one sequence per replica
    measured_packed_k   k live replicas, contiguous    — k from the capture
    measured_spread_k   k live replicas, spread across pods

```scripts/measure_ownership.py``` measures the number of sequences that are
actually co-resident per serving step (max 4 over the 4-tenant capture, max 3
over the 3-tenant capture) — that is an upper bound on how many **owning ranks**
are live at once.  The hash assumes all ```n_dp``` ranks carry tokens.

What the capture does *not* record is **which** replicas those live sequences
occupy.  That is the one remaining assumption, so it is swept as two extremes:
packed into the first k ranks, or spread across pods.  If the OCS answer moves
between them, the unknown is the placement of live sources, not their count.

Usage
─────
    python3.12 scripts/ownership_ocs_envelope.py \
        --workload logs/workload/qwen36 --world-size 32 \
        --measured outputs/ownership/multi_tenant_ownership.json \
        --out outputs/ownership/ocs_envelope.qwen36.ws32.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import src.eval.cost_model as cm  # noqa: E402
from src.eval.cost_model import (  # noqa: E402
    CostConfig, DispatchMode, FabricConfig, Tier, Topology, evaluate,
    hierarchy_for, traffic_matrix,
)
from src.eval.ocs_eval import OcsConfig, plan_circuits, with_circuits  # noqa: E402
from src.eval.placement_opt import make_placement  # noqa: E402
from src.eval.trace_ir import load_workload  # noqa: E402
from src.serving.suite import build_suite, split_by_category  # noqa: E402

PLACEMENTS = ("linear", "load_balanced_layer", "affinity_coordinated_layer")
BASE_MODELS = ("hash", "per_sequence")


# ── source models ────────────────────────────────────────────────────────

def measured_k(measured_path: Path | None, run_name: str = "run_burst_4t") -> int | None:
    """Max live sequences per step, from the capture (upper bound on live ranks)."""
    if not measured_path or not measured_path.is_file():
        return None
    doc = json.loads(measured_path.read_text())
    for r in doc.get("runs", []):
        if r.get("run") == run_name:
            return int(r["source_support_measured_max"])
    return int(doc["runs"][0]["source_support_measured_max"]) if doc.get("runs") else None


def compute_src(model: str, t, n_dp: int, k: int) -> np.ndarray:
    if model == "hash":
        return cm.token_rank_original(t, n_dp, 0)
    if model == "per_sequence":
        return (t.run % n_dp).astype(np.int32)
    if model.startswith("measured_packed"):
        return (t.run % max(k, 1)).astype(np.int32)
    if model.startswith("measured_spread"):
        step = max(n_dp // max(k, 1), 1)
        return ((t.run % max(k, 1)) * step).astype(np.int32)
    raise ValueError(model)


def patch_token_rank(model: str, k: int) -> None:
    def patched(t, n_dp, seed=0):
        return compute_src(model, t, n_dp, k)
    cm.token_rank = patched


def restore_token_rank() -> None:
    cm.token_rank = cm.token_rank_original


# ── the measurement, driven directly so n_dp can differ from world_size ───

def static_gain(fit, ev, placement, topo, cfg, cost, mode, n_dp: int, seed: int) -> dict:
    base = evaluate(ev, placement, topo, cost, mode, n_dp=n_dp, seed=seed)
    tm = traffic_matrix(fit, placement, topo, mode, n_dp=n_dp, seed=seed)
    plan, info = plan_circuits(tm.counts, topo, cfg)
    oc = evaluate(ev, placement, with_circuits(topo, plan), cost, mode,
                  n_dp=n_dp, seed=seed)
    gain = (100.0 * (1 - oc["bottleneck_us"] / base["bottleneck_us"])
            if base["bottleneck_us"] else None)
    live = int(np.unique(compute_src._current_model_src).size) if False else None
    return {
        "eps_bottleneck_us": round(base["bottleneck_us"], 4),
        "ocs_bottleneck_us": round(oc["bottleneck_us"], 4),
        "static_ocs_gain_pct": None if gain is None else round(gain, 4),
        "n_candidate_promotable_pairs": info["n_candidate_promotable_pairs"],
        "covered_fraction": info["promotable_traffic_covered_fraction"],
        "cross_pod_fraction": round(base["cross_pod_bytes"]
                                    / max(base["network_bytes"], 1e-9), 4),
    }


# ── main ─────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--measured", type=Path,
                    default=Path("outputs/ownership/multi_tenant_ownership.json"))
    ap.add_argument("--k", type=int, default=None, help="override measured k")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    cm.token_rank_original = cm.token_rank

    W = args.world_size
    k = args.k or measured_k(args.measured) or 4
    models = list(BASE_MODELS) + [f"measured_packed_{k}", f"measured_spread_{k}"]

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
    dp_values = sorted({W, max(1, W // 2), max(1, W // 4)}, reverse=True)

    print(f"== ownership -> OCS envelope: {args.workload.name} world={W}")
    print(f"   measured live-sequence bound k = {k} "
          f"(source: {args.measured if args.measured and args.measured.is_file() else 'default'})")
    print(f"   fit {fit.n_cells} cells / eval {ev.n_cells} cells (decode only)")
    print(f"   source models: {', '.join(models)}\n")

    out = {
        "workload": str(args.workload), "world_size": W, "k": k,
        "measured_source": str(args.measured),
        "additive": ("imports + monkey-patches cost_model.token_rank; "
                     "no existing module modified"),
        "topology": topo.describe(), "dp_values": dp_values, "rows": {},
    }
    t0 = time.time()
    try:
        for model in models:
            patch_token_rank(model, k)
            for nd in dp_values:
                for kind in PLACEMENTS:
                    p = make_placement(kind, fit, W, seed=args.seed)
                    key = f"{model}|dp{nd}|{kind}"
                    out["rows"][key] = static_gain(
                        fit, ev, p, topo, cfg, cost, DispatchMode.DEDUP_RANK, nd, args.seed)
    finally:
        restore_token_rank()
    out["wall_s"] = round(time.time() - t0, 2)

    print(f"{'source model':<20} {'dp':>3} {'placement':<26} {'eps_us':>10} "
          f"{'ocs_gain%':>9} {'promo_pairs':>11} {'covered':>8} {'crosspod':>9}")
    for model in models:
        for nd in dp_values:
            for kind in PLACEMENTS:
                v = out["rows"][f"{model}|dp{nd}|{kind}"]
                g = v["static_ocs_gain_pct"]
                print(f"{model:<20} {nd:>3} {kind:<26} {v['eps_bottleneck_us']:>10.1f} "
                      f"{(f'{g:.3f}' if g is not None else 'n/a'):>9} "
                      f"{v['n_candidate_promotable_pairs']:>11} "
                      f"{v['covered_fraction']:>8.4f} {v['cross_pod_fraction']:>9.4f}")
        print()

    print(f"   ({out['wall_s']} s)\n   cost_model.token_rank restored")
    dest = args.out or Path(f"outputs/ownership/ocs_envelope.{args.workload.name}.json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=1))
    print(f"-> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

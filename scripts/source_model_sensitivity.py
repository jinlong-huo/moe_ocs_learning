#!/usr/bin/env python3
"""source_model_sensitivity.py — does the OCS result depend on an assumption
that the founding invariance claim does not cover?

The gap being tested
────────────────────
``docs/assumptions.md`` A1 establishes that **routing** is a pure function of
(inputs, weights): a token hits the same experts regardless of topology, node
distribution or engine.  Every claim in this repo leans on it.

But the quantity an OCS is priced against is not the expert map — it is the
**rank-pair byte matrix**, and that needs a second fact A1 does not supply:
which rank *owns* the token in the first place.  ``cost_model.token_rank``
answers it with

    src = (run * 1_000_003 + pos) % n_dp

a synthetic hash of (sequence, position) that is uniform over DP ranks by
construction.  It is never measured, never cited, and no gate covers it.  Yet it
decides the source side of every traffic matrix this pipeline computes, and the
tier of a pair — hence whether a circuit can promote it and how much it is worth
— depends on the source as much as on the destination.

This script replaces that one function with three alternative ownership models
that correspond to real serving layouts, and re-runs the same comparison:

    hash             the repo default: uniform over ranks (no layout)
    per_sequence     a whole sequence lives on one rank — standard data
                     parallel / continuous-batching replica sharding
    sequence_block   contiguous blocks of sequences per rank (static partition)
    token_roundrobin pos % n_dp — tokens interleaved across ranks

If the OCS gain and the placement ranking move materially between these models,
then the OCS conclusion rests on an assumption outside A1, and any claim must
name the ownership model it assumes.

Implementation note: this script monkey-patches ``cost_model.token_rank`` for the
duration of the run rather than modifying the module, so the audit is additive
and the existing pipeline is untouched.  The patch is scoped, restored in a
``finally``, and printed so the run is self-documenting.

Usage
-----
    python3 scripts/source_model_sensitivity.py --workload logs/workload/qwen36 \
        --world-size 32
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
from src.eval.ocs_eval import (  # noqa: E402
    OcsConfig, ocs_comparison, plan_circuits, with_circuits,
)
from src.eval.placement_opt import make_placement  # noqa: E402
from src.eval.trace_ir import load_workload  # noqa: E402
from src.serving.suite import build_suite, split_by_category  # noqa: E402

SOURCE_MODELS = ("hash", "per_sequence", "sequence_block", "token_roundrobin")
PLACEMENTS = ("linear", "load_balanced_layer", "affinity_coordinated_layer")


def compute_src(model: str, t, n_dp: int) -> np.ndarray:
    """The token->owning-rank map for each ownership model."""
    if model == "hash":                       # the repo default, for reference
        return cm.token_rank_original(t, n_dp, 0)
    if model == "per_sequence":               # one sequence per replica
        return (t.run % n_dp).astype(np.int32)
    if model == "sequence_block":             # contiguous sequence blocks
        n_runs = max(int(t.run.max()) + 1, 1)
        per = max(1, n_runs // max(n_dp, 1))
        return ((t.run // per) % n_dp).astype(np.int32)
    if model == "token_roundrobin":           # interleaved tokens
        return (t.pos % n_dp).astype(np.int32)
    raise ValueError(model)


def patch_token_rank(model: str) -> None:
    def patched(t, n_dp, seed=0):
        return compute_src(model, t, n_dp)
    cm.token_rank = patched


def restore_token_rank() -> None:
    cm.token_rank = cm.token_rank_original


def static_gain(fit, ev, placement, topo, cfg, cost, mode, n_dp: int, seed: int
                ) -> dict:
    """``ocs_comparison``'s static path, but with an explicit source-shard count.

    ``ocs_comparison`` and ``evaluate`` both default ``n_dp`` to ``world_size``,
    so the pipeline as used *cannot express DP < EP*: 24 of 32 ranks owning
    experts but no tokens is a standard MoE deployment and is unrepresentable.
    ``evaluate`` does accept ``n_dp``, so the regime can be measured by driving
    the same two calls directly.
    """
    base = evaluate(ev, placement, topo, cost, mode, n_dp=n_dp, seed=seed)
    tm = traffic_matrix(fit, placement, topo, mode, n_dp=n_dp, seed=seed)
    plan, info = plan_circuits(tm.counts, topo, cfg)
    oc = evaluate(ev, placement, with_circuits(topo, plan), cost, mode,
                  n_dp=n_dp, seed=seed)
    gain = (100.0 * (1 - oc["bottleneck_us"] / base["bottleneck_us"])
            if base["bottleneck_us"] else None)
    return {
        "eps_bottleneck_us": round(base["bottleneck_us"], 4),
        "ocs_bottleneck_us": round(oc["bottleneck_us"], 4),
        "static_ocs_gain_pct": None if gain is None else round(gain, 4),
        "n_candidate_promotable_pairs": info["n_candidate_promotable_pairs"],
        "covered_fraction": info["promotable_traffic_covered_fraction"],
        "cross_pod_fraction": round(base["cross_pod_bytes"]
                                    / max(base["network_bytes"], 1e-9), 4),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-dp", type=int, default=None,
                    help="source shards (default: world_size)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    # keep a handle on the original before any patching
    cm.token_rank_original = cm.token_rank

    t = load_workload(args.workload / "manifest.json", decode_only=True)
    specs = build_suite(n_repeats=0)
    fu, eu = split_by_category(specs, seed=args.seed)
    present = {r.uid for r in t.runs}
    fit = t.by_runs([u for u in fu if u in present])
    ev = t.by_runs([u for u in eu if u in present])
    W = args.world_size
    n_dp = args.n_dp or W
    cost = CostConfig(hidden_size=2048)

    base = hierarchy_for(W, "multi_pod")           # 8 GPU/node, 2 nodes/pod
    fab = FabricConfig(core_oversubscription=4.0, pod_oversubscription=1.0)
    topo = Topology(W, base.gpus_per_node, base.nodes_per_pod, fab, set(),
                    base.rank_to_slot, (Tier.CROSS_POD,))
    cfg = OcsConfig(n_circuits=max(4, W // 2), ports_per_rank=2)

    print(f"== source-model sensitivity: {args.workload.name} world={W} n_dp={n_dp}")
    print(f"   fit {fit.n_cells} cells / eval {ev.n_cells} cells (decode only)")
    print(f"   patching cost_model.token_rank (not modifying the module)\n")

    out: dict = {"workload": str(args.workload), "world_size": W, "n_dp": n_dp,
                 "dp_ep_note": ("evaluate()/ocs_comparison() default n_dp to "
                                "world_size, so DP<EP cannot be expressed through "
                                "them; the dp_ep rows below drive evaluate with an "
                                "explicit n_dp instead"),
                 "topology": topo.describe(),
                 "note": ("token->rank ownership is an assumption outside A1; "
                          "A1 fixes only the expert side of the traffic matrix"),
                 "models": {}}
    try:
        for model in SOURCE_MODELS:
            patch_token_rank(model)
            rows = {}
            for kind in PLACEMENTS:
                p = make_placement(kind, fit, W, seed=args.seed)
                r = ocs_comparison(fit, ev, p, topo, cfg, cost,
                                   DispatchMode.DEDUP_RANK, args.seed)
                eps = evaluate(ev, p, topo, cost, DispatchMode.DEDUP_RANK,
                               seed=args.seed)
                # how much of the traffic is even eligible, and does the matrix
                # still look rank-1 (the assessment's structural argument)?
                tm = traffic_matrix(ev, p, topo, DispatchMode.DEDUP_RANK,
                                    n_dp=n_dp, seed=args.seed)
                rows[kind] = {
                    "eps_bottleneck_us": round(eps["bottleneck_us"], 4),
                    "cross_pod_bytes": eps["cross_pod_bytes"],
                    "cross_pod_fraction": round(
                        eps["cross_pod_bytes"] / max(eps["network_bytes"], 1e-9), 4),
                    "rank1_energy": round(tm.rank1_energy(), 6),
                    "applicable": bool(r.get("applicable")),
                    "static_ocs_gain_pct": r.get("static_ocs", {}).get(
                        "bottleneck_reduction_pct"),
                    "oracle_ocs_gain_pct": r.get("oracle_ocs", {}).get(
                        "bottleneck_reduction_pct"),
                }
            out["models"][model] = rows
            print(f"  -- source model: {model}")
            for kind, v in rows.items():
                print(f"     {kind:<28} eps={v['eps_bottleneck_us']:10.1f} us "
                      f"ocs={v['static_ocs_gain_pct']} "
                      f"crosspod_frac={v['cross_pod_fraction']:.3f} "
                      f"rank1={v['rank1_energy']:.4f}")
            sys.stdout.flush()
    finally:
        restore_token_rank()
        print("\n   cost_model.token_rank restored (first sweep)")

    # ── DP vs EP: the source axis need not equal the expert axis ──────
    print("\n== DP < EP (source shards fewer than ranks; "
          "unrepresentable in ocs_comparison)")
    dp_rows: dict = {}
    try:
        for model in ("hash", "per_sequence"):
            patch_token_rank(model)
            for nd in sorted({W, max(1, W // 2), max(1, W // 4)}, reverse=True):
                for kind in (PLACEMENTS[0], PLACEMENTS[2]):
                    p = make_placement(kind, fit, W, seed=args.seed)
                    v = static_gain(fit, ev, p, topo, cfg, cost,
                                    DispatchMode.DEDUP_RANK, nd, args.seed)
                    dp_rows[f"{model}|dp{nd}|{kind}"] = v
                    print(f"   {model:<17} dp={nd:<3} {kind:<28} "
                          f"eps={v['eps_bottleneck_us']:10.1f} "
                          f"ocs={v['static_ocs_gain_pct']}% "
                          f"cover={v['covered_fraction']:.4f}")
                    sys.stdout.flush()
    finally:
        restore_token_rank()
        print("\n   cost_model.token_rank restored")

    t0 = time.time()
    out_path = args.out or (_repo_root / "outputs" / "four_cell" /
                            f"source_sensitivity.{args.workload.name}.ws{W}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out["dp_ep"] = dp_rows
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"   wrote {out_path}")

    # ── the verdict: does the ranking of placements survive? ──────────
    print("\n== verdict")
    for model, rows in out["models"].items():
        gains = {k: v["static_ocs_gain_pct"] for k, v in rows.items()}
        eps = {k: v["eps_bottleneck_us"] for k, v in rows.items()}
        best_eps = min(eps, key=eps.get)
        best_ocs = max((k for k in gains if gains[k] is not None),
                       key=lambda k: gains[k], default=None)
        print(f"   {model:<17} best-eps placement = {best_eps:<28} "
              f"largest OCS gain = {best_ocs:<28} "
              f"({gains.get(best_ocs) if best_ocs else None}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

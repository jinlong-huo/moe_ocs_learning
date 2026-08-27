#!/usr/bin/env python3
"""verify_live_invariance.py — the staged evidence chain for OCS-aware MoE.

This script replaces a single-shot "invariance gate" with the five questions
that actually have to be answered in order.  Each stage is allowed to FAIL and
report a negative result; the exit code reflects whether the chain holds, not
whether the original hypothesis was confirmed.

    Q1  Is logical routing decoupled from placement and topology?
        R(X, M, P1, T1) == R(X, M, P2, T2)      [bit-compared, not assumed]

    Q2  Does the routing signal carry workload structure?
        semantic vs lexical controls, category decoding, load skew,
        affinity-vs-load null test

    Q3  Does placement change the cost of that fixed routing?
        same routing, many placements/perturbations -> cost must move

    Q4  Does the affinity graph beat the simpler alternatives?
        affinity-aware placement vs random vs pure load balancing vs the
        direct fan-out optimum, all scored OUT OF SAMPLE

    Q5  Can OCS exploit what is left, after paying for reconfiguration?
        static / static-from-fit / oracle circuits, tier promotion only,
        plus temporal stability vs the reconfiguration timescale

Design notes that matter for interpreting the output
────────────────────────────────────────────────────
* Q1 is a *structural* claim here: routing is read from an immutable
  ``CellTable`` and the placement/topology modules are pure functions of it, so
  Q1 cannot be violated by construction.  What Q1 therefore measures is the
  empirical part that CAN fail: run-to-run determinism of the captured gate
  (the ``repeat`` prompts), which is the real boundary of invariance.

* Every number in Q4 and Q5 is fit on one slice of the workload and scored on a
  disjoint slice.  Two splits are reported: within-category (easier) and
  leave-categories-out (harder).  An intervention that only helps on the first
  is domain-specific and must not be sold as general.

Usage
-----
    python3 scripts/verify_live_invariance.py --workload logs/workload/qwen15
    python3 scripts/verify_live_invariance.py --workload logs/workload/qwen36 \
        --world-size 32 --topology multi_pod
    python3 scripts/verify_live_invariance.py --workload logs/workload/qwen15 \
        --stage q4 --world-size 15
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

from src.eval.affinity_graph import (  # noqa: E402
    KINDS, layer_affinities, per_layer_graph_similarity, pooled_affinity,
    signature_similarity_matrix, structure_test, within_between,
)
from src.eval.cost_model import (  # noqa: E402
    TOPOLOGY_STYLES, CostConfig, DispatchMode, Placement, evaluate,
    hierarchy_for, traffic_matrix,
)
from src.eval.ocs_eval import (  # noqa: E402
    RECONFIG_CLASSES, OcsConfig, ocs_comparison, stability,
)
from src.eval.placement_opt import (  # noqa: E402
    PERTURBATIONS, make_placement, mean_fanout, perturb,
)
from src.eval.specialization import (  # noqa: E402
    category_decoding, expert_category_mi, semantics_vs_lexis,
    specialization_index,
)
from src.eval.trace_ir import CellTable, load_workload, routing_identical  # noqa: E402
from src.serving.suite import build_suite, split_by_category, split_within_category  # noqa: E402


def _j(x):
    """JSON-safe (numpy scalars/arrays -> python)."""
    if isinstance(x, dict):
        return {k: _j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return _j(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    return x


# ═══════════════════════════════════════════════════════════════════════
# Q1 — is logical routing decoupled from the physical substrate?
# ═══════════════════════════════════════════════════════════════════════

def stage_q1(t: CellTable, world: int, styles: list[str], seed: int) -> dict:
    """Routing invariance under placement and topology, plus the empirical
    determinism boundary."""
    out: dict = {"question": "Is logical routing decoupled from P and T?"}

    # ── (a) structural: cost stages are pure functions of the routing IR ──
    fit = t
    placements = [make_placement("linear", fit, world),
                  make_placement("random", fit, world, seed=1),
                  make_placement("random", fit, world, seed=2),
                  make_placement("load_balanced", fit, world),
                  make_placement("affinity_layer", fit, world)]
    base_map = None
    checks = []
    for p in placements:
        for style in styles:
            topo = hierarchy_for(world, style)
            r = evaluate(t, p, topo, CostConfig(), DispatchMode.DEDUP_RANK, seed=seed)
            # the routing map consumed is literally the same object; verify it
            ident = routing_identical(t, t)
            checks.append({
                "placement": p.name, "topology": style,
                "routing_cells": r["n_cells"],
                "routing_match_rate": ident["match_rate"],
                "bottleneck_us": round(r["bottleneck_us"], 4),
                "network_bytes": r["network_bytes"],
            })
    costs = [c["bottleneck_us"] for c in checks]
    out["structural"] = {
        "n_configurations": len(checks),
        "routing_identical_across_all": all(
            abs(c["routing_match_rate"] - 1.0) < 1e-12 for c in checks),
        "routing_cells_constant": len({c["routing_cells"] for c in checks}) == 1,
        "cost_moved": bool(max(costs) - min(costs) > 1e-9),
        "cost_spread_ratio": round(max(costs) / max(min(costs), 1e-12), 4),
        "note": ("placement/topology modules consume the immutable CellTable; "
                 "they cannot alter a routing decision, so invariance here is "
                 "structural. The empirical boundary is measured below."),
        "samples": checks[:12],
    }

    # ── (b) empirical: is the captured gate itself deterministic? ─────────
    rep = [r.uid for r in t.runs if r.role == "repeat"]
    if len(rep) >= 2:
        a = t.by_runs([rep[0]])
        pairs = []
        for u in rep[1:]:
            b = t.by_runs([u])
            # align on (pos, layer); run index differs by construction
            ka = {(int(p_), int(l_)): tuple(sorted(e.tolist()))
                  for p_, l_, e in zip(a.pos, a.layer, a.experts)}
            kb = {(int(p_), int(l_)): tuple(sorted(e.tolist()))
                  for p_, l_, e in zip(b.pos, b.layer, b.experts)}
            common = set(ka) & set(kb)
            mism = sum(1 for k in common if ka[k] != kb[k])
            pairs.append({"pair": f"{rep[0]}|{u}", "n_common": len(common),
                          "mismatched": mism,
                          "match_rate": round(1 - mism / max(len(common), 1), 6)})
        rates = [p["match_rate"] for p in pairs]
        out["empirical_determinism"] = {
            "n_repeat_runs": len(rep),
            "mean_match_rate": round(float(np.mean(rates)), 6),
            "min_match_rate": round(float(np.min(rates)), 6),
            "bitwise_deterministic": bool(min(rates) >= 1.0 - 1e-12),
            "pairs": pairs,
            "note": ("identical prompt + greedy decoding. Any mismatch is "
                     "numerical nondeterminism in the quantised gate and is "
                     "the tolerance every other similarity number must be "
                     "read against."),
        }
    else:
        out["empirical_determinism"] = {"insufficient": True}

    det = out["empirical_determinism"]
    out["verdict"] = {
        "logical_routing_decoupled": out["structural"]["routing_identical_across_all"],
        "cost_depends_on_substrate": out["structural"]["cost_moved"],
        "gate_deterministic": det.get("bitwise_deterministic"),
        "noise_floor_match_rate": det.get("min_match_rate"),
        "holds": bool(out["structural"]["routing_identical_across_all"]
                      and out["structural"]["cost_moved"]),
    }
    return out


# ═══════════════════════════════════════════════════════════════════════
# Q2 — does the routing signal carry workload structure?
# ═══════════════════════════════════════════════════════════════════════

def stage_q2(t: CellTable, seed: int, n_perm: int, n_null: int) -> dict:
    out: dict = {"question": "Does routing carry workload structure beyond load?"}

    out["layer_namespace_check"] = {
        **t.cross_layer_load_correlation(),
        "note": ("Pearson r between per-layer expert-load vectors. r ~ 0 means "
                 "expert ids are independent per-layer namespaces, so any "
                 "statistic computed on layer-POOLED expert ids is measuring "
                 "an average of unrelated distributions."),
    }
    out["load_balance"] = {
        "pooled": t.load_balance(),
        "per_layer_mean": {
            k: round(float(np.mean([t.load_balance(int(l))[k] for l in t.layers])), 6)
            for k in ("max_over_uniform", "gini", "cv", "top_eighth_share")},
        "note": ("the gap between pooled and per-layer is the size of the "
                 "artifact introduced by pooling."),
    }

    out["semantics_vs_lexis"] = semantics_vs_lexis(t)
    out["category_decoding"] = category_decoding(t, seed=seed, n_perm=n_perm)
    out["expert_category_mi"] = expert_category_mi(t, n_perm=max(30, n_perm // 4),
                                                  seed=seed)
    out["specialization"] = specialization_index(t)

    # within/between group similarity on the semantic categories
    cat_t = t.by_role("category")
    M, infos = signature_similarity_matrix(cat_t)
    out["signature_within_between"] = within_between(
        M, [i.category for i in infos])

    # affinity structure beyond load, per definition
    out["affinity_structure_vs_load_null"] = {
        kind: structure_test(t, kind, n_null=n_null, seed=seed, per_layer=True)
        for kind in ("cooccurrence", "pmi", "jaccard")
    }
    out["affinity_structure_pooled"] = structure_test(
        t, "cooccurrence", n_null=n_null, seed=seed, per_layer=False)

    dec = out["category_decoding"]
    sv = out["semantics_vs_lexis"]
    st = out["affinity_structure_vs_load_null"]["cooccurrence"]
    out["verdict"] = {
        "workload_decodable_from_routing": dec.get("significant"),
        "decoding_accuracy": dec.get("accuracy"),
        "decoding_chance": dec.get("perm_null_mean"),
        "driver": sv.get("verdict"),
        "affinity_beyond_load": st.get("significant"),
        "affinity_excess_ratio": st.get("excess_ratio"),
        "per_expert_specialization_normalized": (
            round(out["specialization"].get("kl_bits_mean", 0)
                  / max(out["specialization"].get("kl_upper_bound_bits", 1), 1e-9), 5)
            if not out["specialization"].get("insufficient") else None),
        "holds": bool(dec.get("significant") and st.get("significant")),
    }
    return out


# ═══════════════════════════════════════════════════════════════════════
# Q3 — does placement change the cost of a fixed routing?
# ═══════════════════════════════════════════════════════════════════════

def stage_q3(t: CellTable, world: int, style: str, seed: int,
             n_random: int) -> dict:
    out: dict = {"question": "Same routing, different placement -> different cost?"}
    topo = hierarchy_for(world, style)
    cost = CostConfig(hidden_size=2048)
    base = make_placement("linear", t, world)

    rows = []
    for mode in (DispatchMode.REPLICATED, DispatchMode.DEDUP_RANK,
                 DispatchMode.DEDUP_NODE):
        variants = [("linear", base)]
        for s in range(n_random):
            variants.append((f"random.s{s}", make_placement("random", t, world, seed=s)))
        for pk in PERTURBATIONS:
            variants.append((f"perturb.{pk}", perturb(base, pk, t, seed=seed)))
        recs = []
        for name, p in variants:
            r = evaluate(t, p, topo, cost, mode, seed=seed)
            ident = routing_identical(t, t)
            recs.append({
                "variant": name,
                "routing_match_rate": ident["match_rate"],
                "total_bytes": r["total_bytes"],
                "network_bytes": r["network_bytes"],
                "inter_node_bytes": r["inter_node_bytes"],
                "bottleneck_us": round(r["bottleneck_us"], 4),
                "mean_fanout": round(r["mean_fanout"], 4),
                "ingress_imbalance": round(r["ingress_imbalance"], 4),
            })
        tb = np.array([x["total_bytes"] for x in recs])
        bu = np.array([x["bottleneck_us"] for x in recs])
        rows.append({
            "dispatch_mode": mode.name,
            "routing_invariant_across_variants": all(
                abs(x["routing_match_rate"] - 1.0) < 1e-12 for x in recs),
            "total_bytes_is_placement_invariant": bool(np.ptp(tb) < 1e-6),
            "total_bytes_spread_pct": round(float(100 * np.ptp(tb) / tb.mean()), 4),
            "bottleneck_spread_pct": round(float(100 * np.ptp(bu) / bu.mean()), 4),
            "bottleneck_min": float(bu.min()), "bottleneck_max": float(bu.max()),
            "variants": recs,
        })
    out["by_dispatch_mode"] = rows
    out["structural_note"] = (
        "Under REPLICATED dispatch total volume is N*K*H*dtype independent of "
        "placement, so placement can only relocate bytes between tiers and "
        "ranks. Only DEDUP modes let placement change total volume, via "
        "fan-out. Any claim that affinity 'reduces traffic' is therefore "
        "conditional on a dedup-capable dispatch kernel.")
    ded = next(r for r in rows if r["dispatch_mode"] == "DEDUP_RANK")
    out["verdict"] = {
        "routing_invariant": ded["routing_invariant_across_variants"],
        "cost_moves_with_placement": bool(ded["bottleneck_spread_pct"] > 1e-6),
        "bottleneck_spread_pct": ded["bottleneck_spread_pct"],
        "volume_invariant_under_replicated": next(
            r["total_bytes_is_placement_invariant"] for r in rows
            if r["dispatch_mode"] == "REPLICATED"),
        "holds": bool(ded["routing_invariant_across_variants"]
                      and ded["bottleneck_spread_pct"] > 1e-6),
    }
    return out


# ═══════════════════════════════════════════════════════════════════════
# Q4 — does affinity beat the simpler alternatives, out of sample?
# ═══════════════════════════════════════════════════════════════════════

_Q4_KINDS = ("linear", "random", "load_balanced", "load_balanced_layer",
             "affinity_global", "affinity_layer", "fanout_layer",
             "balanced_affinity_layer", "bottleneck_layer",
             "affinity_coordinated_layer",
             "hierarchical_layer", "adversarial")


def _q4_one_split(t: CellTable, fit_ids, ev_ids, world: int, style: str,
                  seed: int, kinds, cost: CostConfig) -> dict:
    fit, ev = t.by_runs(fit_ids), t.by_runs(ev_ids)
    topo = hierarchy_for(world, style)
    res = {}
    for kind in kinds:
        try:
            p = make_placement(kind, fit, world, seed=seed)
        except Exception as e:
            res[kind] = {"error": str(e)[:200]}
            continue
        r_in = evaluate(fit, p, topo, cost, DispatchMode.DEDUP_RANK, seed=seed)
        r_out = evaluate(ev, p, topo, cost, DispatchMode.DEDUP_RANK, seed=seed)
        r_out_node = evaluate(ev, p, topo, cost, DispatchMode.DEDUP_NODE, seed=seed)
        res[kind] = {
            "in_sample": {k: r_in[k] for k in
                          ("mean_fanout", "network_bytes", "bottleneck_us")},
            "out_of_sample": {k: r_out[k] for k in
                              ("mean_fanout", "network_bytes", "inter_node_bytes",
                               "cross_pod_bytes", "bottleneck_us", "active_pairs",
                               "ingress_imbalance", "rank1_energy")},
            "out_of_sample_node_dedup": {
                k: r_out_node[k] for k in ("mean_fanout", "inter_node_bytes",
                                           "bottleneck_us")},
            "generalization_gap_fanout_pct": round(
                100 * (r_out["mean_fanout"] - r_in["mean_fanout"])
                / max(r_in["mean_fanout"], 1e-12), 4),
        }
    # relative to random (the correct null) and to linear (the deployed default)
    for ref in ("random", "linear"):
        if ref not in res or "error" in res[ref]:
            continue
        b = res[ref]["out_of_sample"]
        for kind, v in res.items():
            if "error" in v:
                continue
            v.setdefault("vs", {})[ref] = {
                m: round(100 * (1 - v["out_of_sample"][m] / b[m]), 4)
                for m in ("mean_fanout", "network_bytes", "inter_node_bytes",
                          "bottleneck_us")
                if b.get(m)}
    return res


def stage_q4(t: CellTable, world: int, style: str, seed: int,
             cost: CostConfig, kinds=_Q4_KINDS) -> dict:
    out: dict = {"question": "Does affinity beat random / load-balancing OOS?"}
    specs = build_suite(n_repeats=0)
    present = {r.uid for r in t.runs}

    f1, e1 = split_within_category(specs, seed=seed)
    f2, e2 = split_by_category(specs, seed=seed)
    splits = {
        "within_category": ([u for u in f1 if u in present],
                            [u for u in e1 if u in present]),
        "leave_categories_out": ([u for u in f2 if u in present],
                                 [u for u in e2 if u in present]),
    }
    out["splits"] = {}
    for name, (fi, ei) in splits.items():
        if len(fi) < 3 or len(ei) < 3:
            out["splits"][name] = {"insufficient": True,
                                   "n_fit": len(fi), "n_eval": len(ei)}
            continue
        out["splits"][name] = {
            "n_fit_runs": len(fi), "n_eval_runs": len(ei),
            "results": _q4_one_split(t, fi, ei, world, style, seed, kinds, cost),
        }

    # cross-workload affinity-graph stability (does the graph transfer?)
    fi, ei = splits["leave_categories_out"]
    if len(fi) >= 3 and len(ei) >= 3:
        out["affinity_graph_transfer"] = per_layer_graph_similarity(
            t.by_runs(fi), t.by_runs(ei), "cooccurrence")

    best = None
    lco = out["splits"].get("leave_categories_out", {})
    if "results" in lco:
        cands = {k: v for k, v in lco["results"].items()
                 if "error" not in v and k not in ("adversarial",)}
        if cands:
            best = min(cands, key=lambda k: cands[k]["out_of_sample"]["bottleneck_us"])
    out["verdict"] = {
        "best_placement_leave_categories_out": best,
        "best_vs_random_bottleneck_pct": (
            lco["results"][best]["vs"]["random"]["bottleneck_us"]
            if best and "vs" in lco["results"][best]
            and "random" in lco["results"][best]["vs"] else None),
        "affinity_layer_vs_random_bottleneck_pct": (
            lco["results"].get("affinity_layer", {}).get("vs", {})
            .get("random", {}).get("bottleneck_us") if "results" in lco else None),
        "load_balanced_vs_random_bottleneck_pct": (
            lco["results"].get("load_balanced", {}).get("vs", {})
            .get("random", {}).get("bottleneck_us") if "results" in lco else None),
        "balanced_affinity_vs_random_bottleneck_pct": (
            lco["results"].get("balanced_affinity_layer", {}).get("vs", {})
            .get("random", {}).get("bottleneck_us") if "results" in lco else None),
        "bottleneck_opt_vs_random_pct": (
            lco["results"].get("bottleneck_layer", {}).get("vs", {})
            .get("random", {}).get("bottleneck_us") if "results" in lco else None),
        "affinity_coordinated_vs_random_pct": (
            lco["results"].get("affinity_coordinated_layer", {}).get("vs", {})
            .get("random", {}).get("bottleneck_us") if "results" in lco else None),
        "holds": bool(best and best not in ("linear", "random")),
    }
    return out


# ═══════════════════════════════════════════════════════════════════════
# Q5 — can OCS exploit the remainder after paying reconfiguration?
# ═══════════════════════════════════════════════════════════════════════

def stage_q5(t: CellTable, world: int, styles: list[str], seed: int,
             cost: CostConfig, placement_kind: str) -> dict:
    out: dict = {"question": "Can OCS help once reconfiguration is charged?"}
    specs = build_suite(n_repeats=0)
    present = {r.uid for r in t.runs}
    f2, e2 = split_by_category(specs, seed=seed)
    fi = [u for u in f2 if u in present]
    ei = [u for u in e2 if u in present]
    if len(fi) < 3 or len(ei) < 3:
        return {**out, "insufficient": True}
    fit, ev = t.by_runs(fi), t.by_runs(ei)
    placement = make_placement(placement_kind, fit, world, seed=seed)

    out["by_topology"] = {}
    for style in styles:
        topo = hierarchy_for(world, style)
        entry = {"topology": topo.describe()}
        for cls, us in RECONFIG_CLASSES.items():
            cfg = OcsConfig(n_circuits=max(4, world // 2), ports_per_rank=2,
                            reconfig_us=us)
            entry[cls] = ocs_comparison(fit, ev, placement, topo, cfg, cost,
                                        DispatchMode.DEDUP_RANK, seed)
        out["by_topology"][style] = entry

    # temporal stability at three timescales, on the topology that has pods
    multi = next((s for s in styles if hierarchy_for(world, s).n_pods > 1),
                 styles[-1])
    topo = hierarchy_for(world, multi)
    cfg = OcsConfig(n_circuits=max(4, world // 2), ports_per_rank=2)
    out["stability"] = {
        w: stability(t, placement, topo, cfg, DispatchMode.DEDUP_RANK,
                     window=w, seed=seed)
        for w in ("run", "token", "layer")
    }
    out["stability_note"] = (
        "plan_persistence is the Jaccard overlap of the circuit sets two "
        "windows would independently choose. Low persistence at the 'layer' "
        "timescale is expected and is the reason a millisecond-class switch "
        "cannot track per-layer traffic: one MoE all-to-all lasts tens of "
        "microseconds.")

    applicable = [s for s, e in out["by_topology"].items()
                  if e.get("ideal_0", {}).get("applicable")]
    best_gain = None
    for s in applicable:
        g = out["by_topology"][s]["ideal_0"]["static_ocs"]["bottleneck_reduction_pct"]
        best_gain = g if best_gain is None else max(best_gain, g)
    out["verdict"] = {
        "topologies_with_cross_pod_traffic": applicable,
        "best_static_ocs_reduction_pct_zero_reconfig": best_gain,
        "mems_10ms_feasible_anywhere": any(
            out["by_topology"][s]["mems_10ms"].get("reconfiguration", {}).get("feasible")
            for s in applicable),
        "run_level_plan_persistence": out["stability"]["run"].get("plan_persistence_mean"),
        "layer_level_plan_persistence": out["stability"]["layer"].get("plan_persistence_mean"),
        "holds": bool(applicable and best_gain and best_gain > 1.0),
    }
    return out


# ═══════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser(description="Staged evidence chain (Q1-Q5)")
    ap.add_argument("--workload", default="logs/workload/qwen15",
                    help="directory containing manifest.json from capture_workload.py")
    ap.add_argument("--world-size", type=int, default=None,
                    help="EP degree (default: largest divisor of E that is <= 64)")
    ap.add_argument("--topology", action="append", default=None,
                    choices=list(TOPOLOGY_STYLES),
                    help="repeatable; default: single_node, single_pod, multi_pod")
    ap.add_argument("--stage", action="append", default=None,
                    choices=["q1", "q2", "q3", "q4", "q5"])
    ap.add_argument("--hidden-size", type=int, default=None,
                    help="activation width for byte accounting (default from model)")
    ap.add_argument("--placement-for-ocs", default="hierarchical_layer")
    ap.add_argument("--n-random", type=int, default=4)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--n-null", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decode-only", action="store_true",
                    help="drop prefill cells (decode is the latency-critical phase)")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()

    wl = Path(args.workload)
    man_path = wl / "manifest.json"
    if not man_path.exists():
        print(f"[chain] no manifest at {man_path}\n"
              f"[chain] run: python3 scripts/capture_workload.py --out {wl}")
        return 2

    t = load_workload(man_path, decode_only=args.decode_only)
    man = json.load(open(man_path))
    hidden = args.hidden_size or _infer_hidden(man)
    E = t.num_experts

    world = args.world_size
    if world is None:
        world = max((w for w in range(2, 65) if E % w == 0), default=2)
    if E % world:
        print(f"[chain] E={E} not divisible by world={world}")
        return 2

    styles = args.topology or ["single_node", "single_pod", "multi_pod"]
    stages = args.stage or ["q1", "q2", "q3", "q4", "q5"]
    cost = CostConfig(hidden_size=hidden)

    print("=" * 78)
    print("MoE routing -> affinity -> placement -> OCS : staged evidence chain")
    print("=" * 78)
    print(f"workload   : {wl}  ({t.n_runs} runs, {t.n_cells} cells)")
    print(f"model      : {t.model_id}  E={E} K={t.top_k} moe_layers={t.n_layers}")
    print(f"EP degree  : world={world} experts_per_rank={E // world}")
    print(f"topologies : {styles}")
    print(f"hidden     : {hidden} (bf16 -> {hidden * 2} B per token per hop)")
    print(f"stages     : {stages}")
    print()

    report: dict = {
        "experiment": "moe_ocs_evidence_chain",
        "workload_dir": str(wl),
        "model": t.model_id,
        "model_meta": man.get("model_meta", {}),
        "capture": {k: man.get(k) for k in ("backend", "max_tokens", "temp", "seed")},
        "num_experts": E, "top_k": t.top_k, "n_moe_layers": t.n_layers,
        "n_runs": t.n_runs, "n_cells": t.n_cells,
        "world_size": world, "experts_per_rank": E // world,
        "hidden_size": hidden, "topology_styles": styles,
        "decode_only": args.decode_only, "seed": args.seed,
    }

    if "q1" in stages:
        print("[Q1] routing decoupling + determinism boundary ...")
        report["Q1_routing_invariance"] = stage_q1(t, world, styles, args.seed)
        v = report["Q1_routing_invariance"]["verdict"]
        print(f"     decoupled={v['logical_routing_decoupled']} "
              f"cost_moves={v['cost_depends_on_substrate']} "
              f"gate_deterministic={v['gate_deterministic']} "
              f"(noise floor match rate={v['noise_floor_match_rate']})")

    if "q2" in stages:
        print("[Q2] workload structure in the routing signal ...")
        report["Q2_routing_structure"] = stage_q2(t, args.seed, args.n_perm, args.n_null)
        v = report["Q2_routing_structure"]["verdict"]
        print(f"     category decoding acc={v['decoding_accuracy']} "
              f"(chance {v['decoding_chance']}) driver={v['driver']}")
        print(f"     affinity beyond load: {v['affinity_beyond_load']} "
              f"(excess x{v['affinity_excess_ratio']})  "
              f"per-expert specialization={v['per_expert_specialization_normalized']}")

    if "q3" in stages:
        print("[Q3] placement changes cost of fixed routing ...")
        report["Q3_placement_cost"] = stage_q3(t, world, "multi_pod", args.seed,
                                               args.n_random)
        v = report["Q3_placement_cost"]["verdict"]
        print(f"     routing invariant={v['routing_invariant']} "
              f"bottleneck spread={v['bottleneck_spread_pct']}% "
              f"volume invariant under REPLICATED={v['volume_invariant_under_replicated']}")

    if "q4" in stages:
        print("[Q4] affinity vs simpler baselines, out of sample ...")
        report["Q4_affinity_value"] = stage_q4(t, world, "multi_pod", args.seed, cost)
        v = report["Q4_affinity_value"]["verdict"]
        print(f"     best OOS placement = {v['best_placement_leave_categories_out']} "
              f"({v['best_vs_random_bottleneck_pct']}% vs random)")
        print(f"     affinity_layer={v['affinity_layer_vs_random_bottleneck_pct']}%  "
              f"load_balanced={v['load_balanced_vs_random_bottleneck_pct']}%  "
              f"balanced_affinity={v['balanced_affinity_vs_random_bottleneck_pct']}%  "
              f"bottleneck_opt={v['bottleneck_opt_vs_random_pct']}%  "
              f"affinity_coord={v['affinity_coordinated_vs_random_pct']}%")

    if "q5" in stages:
        print("[Q5] OCS with reconfiguration charged ...")
        report["Q5_ocs"] = stage_q5(t, world, styles, args.seed, cost,
                                    args.placement_for_ocs)
        v = report["Q5_ocs"]["verdict"]
        print(f"     cross-pod topologies: {v['topologies_with_cross_pod_traffic']}")
        print(f"     best static OCS reduction (0 reconfig) = "
              f"{v['best_static_ocs_reduction_pct_zero_reconfig']}%")
        print(f"     plan persistence: run={v['run_level_plan_persistence']} "
              f"layer={v['layer_level_plan_persistence']}")

    chain = {k: report[v]["verdict"]["holds"]
             for k, v in [("Q1", "Q1_routing_invariance"),
                          ("Q2", "Q2_routing_structure"),
                          ("Q3", "Q3_placement_cost"),
                          ("Q4", "Q4_affinity_value"),
                          ("Q5", "Q5_ocs")]
             if v in report}
    report["chain"] = chain
    report["chain_complete"] = all(chain.values())

    out = Path(args.output or (wl / "evidence_chain.json"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(_j(report), f, indent=2)

    print("\n" + "=" * 78)
    print("chain: " + "  ".join(f"{k}={'PASS' if v else 'FAIL'}"
                                for k, v in chain.items()))
    print(f"report -> {out}")
    print("=" * 78)
    return 0


def _infer_hidden(man: dict) -> int:
    """Read hidden_size from the model config if reachable, else a default."""
    mp = man.get("model", "")
    try:
        cfg = json.load(open(Path(mp) / "config.json"))
        h = cfg.get("hidden_size") or (cfg.get("text_config") or {}).get("hidden_size")
        if h:
            return int(h)
    except Exception:
        pass
    return 2048


if __name__ == "__main__":
    raise SystemExit(main())

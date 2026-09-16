#!/usr/bin/env python3
"""ocs_four_cell.py — the Affinity-note 2x2, measured end to end.

The note's design is two axes and four cells:

                       EPS only            with OCS
    no affinity        A (reference)       C
    affinity-guided    B                   D   <- claimed minimum time

and the claim to test is that **D is the fastest of the four**.  This script
measures all four on the captured routing traces, out of sample (the circuit
plan and every placement are fitted on one category split and scored on a
disjoint one), and reports the interaction term that the claim actually rests
on:

    synergy = delta(A->D) - [ delta(A->B) + delta(A->C) ]

  synergy > 0  affinity and OCS are **complements** (D is better than the sum
               of the parts, as the note expects)
  synergy < 0  they are **substitutes** (affinity coalesces away exactly the
               cross-pod rank pairs a circuit would have promoted, so the two
               mechanisms compete for the same traffic)

The second half of the note's point (4) — that port count and reconfiguration
latency decide the outcome — is measured as an explicit envelope over
reconfiguration class, circuit budget and ports per rank, never as a single
point.

Stages (``--stage``): ``fourcell`` | ``regime`` | ``timing`` | ``milp`` | all.

Usage
-----
    python3 scripts/ocs_four_cell.py --workload logs/workload/qwen36
    python3 scripts/ocs_four_cell.py --workload logs/workload/qwen36 --quick
    python3 scripts/ocs_four_cell.py --workload logs/workload/whittle \
        --stage fourcell --world-size 16
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from src.eval.cost_model import (  # noqa: E402
    CostConfig, DispatchMode, FabricConfig, Placement, Tier, Topology,
    evaluate, hierarchy_for, traffic_matrix,
)
from src.eval.completion import (  # noqa: E402
    ServingModel, comm_us, first_decode_step, prefill_pass, speedup,
    timing_report,
)
from src.eval.ocs_eval import (  # noqa: E402
    RECONFIG_CLASSES, OcsConfig, breakeven, ocs_comparison, plan_circuits,
    with_circuits,
)
from src.eval.placement_opt import make_placement  # noqa: E402
from src.eval.trace_ir import CellTable, load_workload  # noqa: E402
from src.serving.suite import build_suite, split_by_category  # noqa: E402

# ── the two axes ─────────────────────────────────────────────────────
# ``load_balanced_layer`` is the important member of the no-affinity family: it
# is per-layer but uses no affinity graph at all, and the bound analysis
# (stage ``milp``) shows it is within ~2% of the optimal min-max load packing.
# Without it, a "y-axis" win cannot be separated from the far simpler
# observation that per-layer placement beats one pooled map.
NO_AFFINITY = ("random", "linear", "load_balanced", "load_balanced_layer")
AFFINITY = ("affinity_layer", "affinity_coordinated_layer")
# used for the headline 2x2: the note's "solving or not" axis
REFERENCE_PLACEMENT = "linear"
AFFINITY_PLACEMENT = "affinity_coordinated_layer"

REGIMES = (
    dict(name="multi_pod_s4", style="multi_pod", core_os=4.0, pod_os=1.0),
    dict(name="multi_pod_s2", style="multi_pod", core_os=2.0, pod_os=1.0),
    dict(name="multi_pod_s8", style="multi_pod", core_os=8.0, pod_os=1.0),
    dict(name="multi_pod_pod_s2", style="multi_pod", core_os=4.0, pod_os=2.0,
         promote_intra_pod=True),
    dict(name="single_pod", style="single_pod", core_os=4.0, pod_os=1.0),
    dict(name="realistic", style="realistic", core_os=4.0, pod_os=1.0),
)
FOCUS_REGIME = "multi_pod_s4"


def build_topology(regime: dict, world_size: int) -> Topology:
    """A named regime -> a Topology (the only place σ and promote_from are set)."""
    base = hierarchy_for(world_size, regime["style"])
    fab = FabricConfig(core_oversubscription=regime["core_os"],
                       pod_oversubscription=regime["pod_os"])
    promote = ((Tier.CROSS_POD, Tier.INTRA_POD) if regime.get("promote_intra_pod")
               else (Tier.CROSS_POD,))
    return Topology(world_size, base.gpus_per_node, base.nodes_per_pod, fab,
                    set(), base.rank_to_slot, promote)


def _j(x):
    """JSON-safe (numpy scalars/arrays -> python)."""
    if isinstance(x, dict):
        return {k: _j(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_j(v) for v in x]
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return float(x)
    if isinstance(x, np.ndarray):
        return _j(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    return x


# ═══════════════════════════════════════════════════════════════════════
# Inputs and placement cache
# ═══════════════════════════════════════════════════════════════════════

def load_splits(workload: Path, seed: int, max_runs: int | None = None
                ) -> tuple[CellTable, CellTable, CellTable]:
    """(all, fit, eval) — the eval slice is disjoint in *category*."""
    t = load_workload(workload / "manifest.json", max_runs=max_runs)
    specs = build_suite(n_repeats=0)
    fu, eu = split_by_category(specs, seed=seed)
    present = {r.uid for r in t.runs}
    fi = [u for u in fu if u in present]
    ei = [u for u in eu if u in present]
    if len(fi) < 3 or len(ei) < 3:
        raise SystemExit(f"split too small: fit={len(fi)} eval={len(ei)}")
    return t, t.by_runs(fi), t.by_runs(ei)


def _cache_key(kind: str, fit: CellTable, world: int, extra: str) -> str:
    h = hashlib.sha1()
    h.update(f"{kind}|{world}|{fit.n_cells}|{fit.n_layers}|{extra}".encode())
    h.update(np.ascontiguousarray(fit.run).tobytes())
    h.update(np.ascontiguousarray(fit.layer).tobytes())
    h.update(np.ascontiguousarray(fit.experts).tobytes())
    return h.hexdigest()[:16]


class PlacementCache:
    """Placement generators are the expensive part; cache them across stages."""

    def __init__(self, root: Path, enabled: bool = True):
        self.root = root
        self.enabled = enabled
        self.hits = 0
        self.misses = 0
        if enabled:
            root.mkdir(parents=True, exist_ok=True)

    def get(self, kind: str, fit: CellTable, world: int, *, extra: str = "",
            build):
        key = _cache_key(kind, fit, world, extra)
        f = self.root / f"{kind}.{world}.{key}.npz"
        if self.enabled and f.exists():
            z = np.load(f, allow_pickle=False)
            self.hits += 1
            meta = json.loads(str(z["meta"]))
            p = Placement(z["map"], world, meta["scope"], layers=fit.layers,
                          name=meta["name"])
            return p, meta.get("aux", {})
        self.misses += 1
        t0 = time.time()
        p, aux = build()
        if self.enabled:
            np.savez_compressed(f, map=p.expert_to_rank,
                                meta=json.dumps({"scope": p.scope, "name": p.name,
                                                 "aux": _j(aux)}))
        return p, {**aux, "build_s": round(time.time() - t0, 2)}


def placements_for(fit: CellTable, world: int, kinds: list[str],
                   cache: PlacementCache, topo: Topology | None = None,
                   ocs_cfg: OcsConfig | None = None,
                   promote_evals: int = 3000, n_sweeps: int = 4,
                   verbose: bool = True) -> dict[str, Placement]:
    """Build or load every requested placement for one fit slice."""
    out: dict[str, Placement] = {}
    for kind in kinds:
        if kind in ("promote_aware_layer", "bottleneck_search_layer"):
            if topo is None or ocs_cfg is None:
                raise SystemExit("promote_aware_layer needs topo and ocs_cfg")
            # The control variant runs the *same* alternating search with an
            # empty circuit set, so its objective is the electrical bottleneck
            # measured by this cost model.  Without it, a co-design result
            # cannot be separated from "this search is simply weaker than the
            # affinity optimiser it was initialised from".
            cfg_used = (ocs_cfg if kind == "promote_aware_layer"
                        else OcsConfig(n_circuits=0, ports_per_rank=ocs_cfg.ports_per_rank,
                                       reconfig_us=ocs_cfg.reconfig_us))
            extra = (f"{topo.n_pods}pod|{topo.fabric.core_oversubscription}|"
                     f"{topo.fabric.pod_oversubscription}|{len(topo.promote_from)}|"
                     f"{cfg_used.n_circuits}|{cfg_used.ports_per_rank}|"
                     f"evals{promote_evals}|sweeps{n_sweeps}")

            def build(_cfg=cfg_used):
                from src.eval.promote_aware import promote_aware_placement
                res = promote_aware_placement(
                    fit, topo, _cfg, world, n_sweeps=min(n_sweeps, 2),
                    max_evals_per_layer=promote_evals, verbose=False)
                return res["placement"], {"history": res["history"],
                                          "plan_size": len(res["circuits"]),
                                          "best_round": res["best_round"],
                                          "best_objective_us": res["best_objective_us"]}
        else:
            extra = f"seed0|sweeps{n_sweeps}"

            def build():
                return make_placement(kind, fit, world, seed=0,
                                      n_sweeps=n_sweeps), {}

        t0 = time.time()
        p, aux = cache.get(kind, fit, world, extra=extra, build=build)
        out[kind] = p
        if verbose:
            note = "" if "build_s" not in aux else f" (built {aux['build_s']}s)"
            print(f"  placement {kind:<28} {time.time() - t0:6.1f}s{note}",
                  flush=True)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Stage 1 — the 2x2
# ═══════════════════════════════════════════════════════════════════════

def cell_row(fit: CellTable, ev: CellTable, placement: Placement,
             topo: Topology, cfg: OcsConfig, cost: CostConfig,
             mode: DispatchMode, seed: int) -> dict:
    """One placement under both substrates, scored on the eval slice."""
    r = ocs_comparison(fit, ev, placement, topo, cfg, cost, mode, seed)
    row = {"placement": placement.name, "applicable": r.get("applicable", False)}
    if not r.get("applicable"):
        row["eps_bottleneck_us"] = r["baseline"]["bottleneck_us"]
        row["reason"] = r.get("reason", "")
        return row
    row.update({
        "eps_bottleneck_us": r["baseline_electrical"]["bottleneck_us"],
        "static_ocs_us": r["static_ocs"]["bottleneck_us"],
        "oracle_ocs_us": r["oracle_ocs"]["bottleneck_us"],
        "static_reduction_pct": r["static_ocs"]["bottleneck_reduction_pct"],
        "oracle_reduction_pct": r["oracle_ocs"]["bottleneck_reduction_pct"],
        "value_of_prediction_pct": r["value_of_prediction_pct"],
        "n_circuits": r["static_ocs"]["n_circuits_provisioned"],
        "n_candidate_promotable_pairs": r["static_ocs"]["n_candidate_promotable_pairs"],
        "covered_fraction": r["static_ocs"]["promotable_traffic_covered_fraction"],
        "port_saturated_ranks": r["static_ocs"]["port_saturated_ranks"],
        "plan_overlap_fit_vs_oracle": r["plan_overlap_fit_vs_oracle"],
        "breakeven_token_passes": r["reconfiguration"]["breakeven_token_passes"],
        "cross_pod_bytes": r["baseline_electrical"]["cross_pod_bytes"],
    })
    return row


def two_by_two(rows: dict) -> dict | None:
    """The A/B/C/D table and the interaction term, from any placement->cell map.

    Factored out so the dispatch-mode stage can report the same reading for each
    ``DispatchMode``: the question there is precisely whether the *verdict*
    (complements vs substitutes) survives a change of dispatch semantics.
    """
    ref = rows.get(REFERENCE_PLACEMENT, {})
    aff = rows.get(AFFINITY_PLACEMENT, {})
    base = ref.get("eps_bottleneck_us")
    if not (base and ref.get("applicable") and aff.get("applicable")):
        return None
    d_aff = 100.0 * (1 - aff["eps_bottleneck_us"] / base)          # A -> B
    d_ocs = ref["static_reduction_pct"]                            # A -> C
    d_both = 100.0 * (1 - aff["static_ocs_us"] / base)             # A -> D
    syn = d_both - (d_aff + d_ocs)
    return {
        "A_eps_no_affinity_us": base,
        "B_eps_affinity_us": aff["eps_bottleneck_us"],
        "C_ocs_no_affinity_us": ref["static_ocs_us"],
        "D_ocs_affinity_us": aff["static_ocs_us"],
        "delta_affinity_pct": round(d_aff, 4),
        "delta_ocs_pct": round(d_ocs, 4),
        "delta_both_pct": round(d_both, 4),
        "synergy_pct": round(syn, 4),
        "verdict": ("complements" if syn > 0.5
                    else "substitutes" if syn < -0.5 else "additive"),
        "D_is_minimum": bool(aff["static_ocs_us"] <= min(
            base, aff["eps_bottleneck_us"], ref["static_ocs_us"])),
        "best_cell": min(
            (("A", base), ("B", aff["eps_bottleneck_us"]),
             ("C", ref["static_ocs_us"]), ("D", aff["static_ocs_us"])),
            key=lambda kv: kv[1])[0],
    }


def stage_four_cell(t: CellTable, fit: CellTable, ev: CellTable, args,
                    pl: dict[str, Placement], regime: dict, cfg: OcsConfig,
                    cost: CostConfig, mode: DispatchMode) -> dict:
    topo = build_topology(regime, args.world_size)
    rows = {}
    for kind, p in pl.items():
        t0 = time.time()
        rows[kind] = cell_row(fit, ev, p, topo, cfg, cost, mode, args.seed)
        rows[kind]["wall_s"] = round(time.time() - t0, 2)
        print(f"  {kind:<28} {json.dumps({k: v for k, v in rows[kind].items() if k in ('eps_bottleneck_us','static_reduction_pct','oracle_reduction_pct','applicable')})}",
              flush=True)

    out = {"regime": regime, "topology": topo.describe(), "cells": rows,
           "ocs": {"n_circuits": cfg.n_circuits, "ports_per_rank": cfg.ports_per_rank,
                   "reconfig_us": cfg.reconfig_us}}

    two = two_by_two(rows)
    if two:
        out["two_by_two"] = two
    aff = rows.get(AFFINITY_PLACEMENT, {})
    # The co-design comparison.  Three readings, because they answer different
    # questions:
    #   promote vs electrical  — did the co-designed search beat the best
    #                            electrical placement once OCS is available?
    #   control vs electrical  — is the search machinery itself competitive
    #                            when there are no circuits at all?  If the
    #                            control loses on the EPS substrate too, any
    #                            co-design loss is the search's fault, not OCS's.
    if aff.get("applicable"):
        cd = {}
        for tag in ("promote_aware_layer", "bottleneck_search_layer"):
            if tag not in rows or not rows[tag].get("applicable"):
                continue
            cd[tag] = {
                "ocs_us": rows[tag]["static_ocs_us"],
                "eps_us": rows[tag]["eps_bottleneck_us"],
                "affinity_coordinated_ocs_us": aff["static_ocs_us"],
                "affinity_coordinated_eps_us": aff["eps_bottleneck_us"],
                "ocs_improvement_vs_electrical_pct": round(100.0 * (
                    1 - rows[tag]["static_ocs_us"] / aff["static_ocs_us"]), 4),
                "eps_improvement_vs_electrical_pct": round(100.0 * (
                    1 - rows[tag]["eps_bottleneck_us"] / aff["eps_bottleneck_us"]), 4),
                "circuit_gain_pct": rows[tag]["static_reduction_pct"],
            }
        if cd:
            out["co_design"] = cd
    return out


# ═══════════════════════════════════════════════════════════════════════
# Stage 2 — regime / port / reconfiguration envelope
# ═══════════════════════════════════════════════════════════════════════

def stage_regime(t: CellTable, fit: CellTable, ev: CellTable, args, pl,
                 cost: CostConfig, mode: DispatchMode) -> dict:
    """Every regime x placement, plus a port/latency sweep on the focus regime."""
    out: dict = {"regimes": {}, "regime_note": (
        "an OCS removes contention; it does not add bandwidth. A regime with no "
        "oversubscribed pair carrying traffic returns applicable=false, which is "
        "a statement about that configuration, not about OCS")}
    for regime in REGIMES:
        topo = build_topology(regime, args.world_size)
        cfg = OcsConfig(n_circuits=max(4, args.world_size // 2), ports_per_rank=2,
                        reconfig_us=RECONFIG_CLASSES["mems_1ms"])
        entry = {"topology": topo.describe(), "cells": {}}
        for kind, p in pl.items():
            if kind == "promote_aware_layer":
                continue          # co-design is regime-specific: skipped here
            entry["cells"][kind] = cell_row(fit, ev, p, topo, cfg, cost, mode,
                                            args.seed)
        out["regimes"][regime["name"]] = entry
        print(f"  regime {regime['name']:<18} "
              f"{ {k: v.get('static_reduction_pct', 'n/a') for k, v in entry['cells'].items()} }",
              flush=True)

    # ── port budget x latency envelope on the focus regime ───────────
    focus = next(r for r in REGIMES if r["name"] == FOCUS_REGIME)
    topo = build_topology(focus, args.world_size)
    sweep = {}
    for kind in (REFERENCE_PLACEMENT, AFFINITY_PLACEMENT):
        p = pl[kind]
        for n_circ in (8, 16, 32):
            for ppr in (1, 2, 4):
                cfg = OcsConfig(n_circuits=n_circ, ports_per_rank=ppr)
                r = ocs_comparison(fit, ev, p, topo, cfg, cost, mode, args.seed)
                key = f"{kind}|c{n_circ}|p{ppr}"
                if not r.get("applicable"):
                    sweep[key] = {"applicable": False}
                    continue
                sweep[key] = {
                    "applicable": True,
                    "covered_fraction": r["static_ocs"]["promotable_traffic_covered_fraction"],
                    "port_saturated_ranks": r["static_ocs"]["port_saturated_ranks"],
                    "static_reduction_pct": r["static_ocs"]["bottleneck_reduction_pct"],
                    "oracle_reduction_pct": r["oracle_ocs"]["bottleneck_reduction_pct"],
                    "value_of_prediction_pct": r["value_of_prediction_pct"],
                }
        print(f"  port sweep {kind} done", flush=True)

    # ── reconfiguration-class envelope (breakeven passes) ────────────
    reconfig = {}
    p = pl[AFFINITY_PLACEMENT]
    base = evaluate(ev, p, topo, cost, mode, seed=args.seed)
    for cls, us in RECONFIG_CLASSES.items():
        cfg = OcsConfig(n_circuits=max(4, args.world_size // 2), ports_per_rank=2,
                        reconfig_us=us)
        r = ocs_comparison(fit, ev, p, topo, cfg, cost, mode, args.seed)
        if not r.get("applicable"):
            reconfig[cls] = {"applicable": False}
            continue
        saved = base["bottleneck_us"] - r["static_ocs"]["bottleneck_us"]
        reconfig[cls] = {
            "reconfig_us": us,
            "static_reduction_pct": r["static_ocs"]["bottleneck_reduction_pct"],
            **breakeven(saved, ev.n_layers, r["static_ocs"]["n_circuits_provisioned"],
                        cfg),
        }
    out["port_sweep"] = sweep
    out["reconfig_envelope"] = reconfig
    out["reconfig_note"] = (
        "breakeven_token_passes is the number of token passes N over which ONE "
        "reconfiguration is amortised; a win that survives only in the ideal_0 "
        "class is not a deployable win")
    return out


# ═══════════════════════════════════════════════════════════════════════
# Stage 2b — dispatch semantics (the axis the verdict may depend on)
# ═══════════════════════════════════════════════════════════════════════

def stage_dispatch(t: CellTable, fit: CellTable, ev: CellTable, args, pl,
                   regime: dict, cost: CostConfig) -> dict:
    """Re-run the four cells under every ``DispatchMode``.

    ``DispatchMode`` is documented in ``cost_model`` as changing the conclusion
    *qualitatively*: under REPLICATED the total dispatch volume is
    placement-invariant (affinity cannot remove bytes, only move them between
    tiers), under DEDUP it can.  A1 says nothing about dispatch semantics, so
    "affinity and OCS are substitutes" is only a claim about the mode it was
    measured in until this stage is run.

    ``total_bytes`` is reported per mode precisely to show the invariance: it
    should be identical across placements under REPLICATED and should move under
    the DEDUP modes.
    """
    topo = build_topology(regime, args.world_size)
    cfg = OcsConfig(n_circuits=max(4, args.world_size // 2), ports_per_rank=2,
                    reconfig_us=RECONFIG_CLASSES["mems_1ms"])
    out: dict = {
        "regime": regime["name"],
        "note": ("dispatch semantics are not covered by A1; the verdict is "
                 "conditional on the mode until all three agree"),
        "by_mode": {},
    }
    for mode in (DispatchMode.REPLICATED, DispatchMode.DEDUP_RANK,
                 DispatchMode.DEDUP_NODE):
        rows = {}
        for kind, p in pl.items():
            if kind == "promote_aware_layer":
                continue          # regime-specific; covered by the fourcell stage
            row = cell_row(fit, ev, p, topo, cfg, cost, mode, args.seed)
            eps = evaluate(ev, p, topo, cost, mode, seed=args.seed)
            row["total_bytes"] = eps["total_bytes"]
            row["network_bytes"] = eps["network_bytes"]
            row["inter_node_bytes"] = eps["inter_node_bytes"]
            row["mean_fanout"] = round(eps["mean_fanout"], 4)
            rows[kind] = row
        entry = {"cells": rows, "two_by_two": two_by_two(rows)}
        # volume invariance: identical total_bytes across placements?
        tot = {k: v.get("total_bytes") for k, v in rows.items()}
        vals = [v for v in tot.values() if v is not None]
        entry["total_bytes_spread_pct"] = (
            round(100.0 * (max(vals) - min(vals)) / max(max(vals), 1e-9), 6)
            if vals else None)
        out["by_mode"][mode.name] = entry
        two = entry["two_by_two"] or {}
        print(f"  {mode.name:<12} volume_spread={entry['total_bytes_spread_pct']}% "
              f"aff={two.get('delta_affinity_pct')} ocs={two.get('delta_ocs_pct')} "
              f"both={two.get('delta_both_pct')} synergy={two.get('synergy_pct')} "
              f"-> {two.get('verdict')}", flush=True)
    return out


# ═══════════════════════════════════════════════════════════════════════
# Stage 3 — completion time (TTFT / ITL / throughput)
# ═══════════════════════════════════════════════════════════════════════

def stage_timing(t: CellTable, fit: CellTable, ev: CellTable, args, pl,
                 regime: dict, cost: CostConfig, mode: DispatchMode) -> dict:
    """The note's axis (4): does the OCS win survive on TTFT / throughput?"""
    topo = build_topology(regime, args.world_size)
    pre = prefill_pass(ev)
    step = first_decode_step(ev)
    out: dict = {
        "prefill_cells": pre.n_cells, "decode_step_cells": step.n_cells,
        "note": ("compute_us_per_layer is a parameter, not a measurement: with "
                 "the default 0 the report is the communication component, and "
                 "the sensitivity rows show how much of the saving survives as "
                 "compute grows. An OCS can only ever remove communication."),
        "models": {},
    }
    for tag, model in (
        ("comm_only", ServingModel(compute_us_per_layer=0.0)),
        ("reconfig_10ms_N100", ServingModel(compute_us_per_layer=0.0,
                                            reconfig_us=10_000.0,
                                            passes_per_reconfig=100)),
        ("compute_200us_layer_N100", ServingModel(compute_us_per_layer=200.0,
                                                  reconfig_us=1_000.0,
                                                  passes_per_reconfig=100)),
    ):
        rows = {}
        for kind, p in pl.items():
            if kind == "promote_aware_layer":
                continue
            cfg = OcsConfig(n_circuits=max(4, args.world_size // 2),
                            ports_per_rank=2)
            r = ocs_comparison(fit, ev, p, topo, cfg, cost, mode, args.seed)
            if not r.get("applicable"):
                rows[kind] = {"applicable": False}
                continue
            # the plan is fitted on fit and applied to BOTH eval slices
            tm = traffic_matrix(fit, p, topo, mode, seed=args.seed)
            circ, info = plan_circuits(tm.counts, topo, cfg)
            tc = with_circuits(topo, circ)
            eps = timing_report(comm_us(pre, p, topo, cost, mode, args.seed),
                                comm_us(step, p, topo, cost, mode, args.seed),
                                ev, model)
            ocs = timing_report(comm_us(pre, p, tc, cost, mode, args.seed),
                                comm_us(step, p, tc, cost, mode, args.seed),
                                ev, model)
            rows[kind] = {"applicable": True, "eps": eps, "static_ocs": ocs,
                          **speedup(eps, ocs),
                          "covered_fraction": info["promotable_traffic_covered_fraction"]}
        out["models"][tag] = rows
        print(f"  timing[{tag}] done", flush=True)

    # the 2x2 read on completion time
    ref, aff = out["models"]["comm_only"].get(REFERENCE_PLACEMENT), \
        out["models"]["comm_only"].get(AFFINITY_PLACEMENT)
    if ref and aff and ref.get("applicable") and aff.get("applicable"):
        out["two_by_two_timing"] = {
            "A_ttft_us": ref["eps"]["ttft_us"], "B_ttft_us": aff["eps"]["ttft_us"],
            "C_ttft_us": ref["static_ocs"]["ttft_us"],
            "D_ttft_us": aff["static_ocs"]["ttft_us"],
            "A_itl_us": ref["eps"]["itl_us"], "B_itl_us": aff["eps"]["itl_us"],
            "C_itl_us": ref["static_ocs"]["itl_us"],
            "D_itl_us": aff["static_ocs"]["itl_us"],
            "D_best_ttft": bool(aff["static_ocs"]["ttft_us"] <= min(
                ref["eps"]["ttft_us"], aff["eps"]["ttft_us"],
                ref["static_ocs"]["ttft_us"])),
            "D_best_itl": bool(aff["static_ocs"]["itl_us"] <= min(
                ref["eps"]["itl_us"], aff["eps"]["itl_us"],
                ref["static_ocs"]["itl_us"])),
            "A_throughput_tok_s": ref["eps"]["throughput_tok_s"],
            "D_throughput_tok_s": aff["static_ocs"]["throughput_tok_s"],
        }
    return out


# ═══════════════════════════════════════════════════════════════════════
# Stage 4 — MILP optimality bounds
# ═══════════════════════════════════════════════════════════════════════

def stage_milp(fit: CellTable, args) -> dict:
    from src.eval.milp_bound import optimality_report
    layers = [int(l) for l in fit.layers][: args.milp_layers]
    out = {"n_layers_bounded": len(layers), "layers": {}}
    for l in layers:
        print(f"  MILP bound on layer {l} ...", flush=True)
        out["layers"][str(l)] = optimality_report(
            fit, args.world_size, layer=l,
            mip_time_limit=args.milp_time,
            lp_max_sets=args.milp_max_sets,
            union_max_sets=args.milp_max_sets,
            progress=lambda s: print(s, flush=True))
        print(f"    linear bound gap "
              f"{out['layers'][str(l)]['gaps'].get('linear_vs_bound_pct')} | "
              f"union bound gap "
              f"{out['layers'][str(l)]['gaps'].get('union_vs_bound_pct')}",
              flush=True)
    return out


# ═══════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", default="all",
                    choices=("fourcell", "regime", "dispatch", "timing",
                             "milp", "all"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--quick", action="store_true",
                    help="smoke path: fewer runs, fewer sweeps, one MILP layer")
    ap.add_argument("--max-runs", type=int, default=None)
    ap.add_argument("--promote-evals", type=int, default=1500)
    ap.add_argument("--milp-layers", type=int, default=1)
    ap.add_argument("--milp-time", type=float, default=30.0)
    ap.add_argument("--milp-max-sets", type=int, default=300)
    ap.add_argument("--decode-only", action="store_true", default=True)
    ap.add_argument("--no-promote-aware", action="store_true",
                    help="skip the co-design placement (it is the expensive one)")
    args = ap.parse_args(argv)

    name = args.workload.name
    out_path = args.out or (_repo_root / "outputs" / "four_cell" /
                            f"{name}.ws{args.world_size}"
                            f"{'.quick' if args.quick else ''}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"== ocs_four_cell: {name} world={args.world_size} "
          f"quick={args.quick} stage={args.stage}", flush=True)
    t0 = time.time()
    t, fit, ev = load_splits(args.workload, args.seed,
                             max_runs=8 if args.quick else args.max_runs)
    # The comm comparison stages follow the evidence chain and score decode
    # cells only (prefill is a different traffic regime).  The completion-time
    # stage cannot: TTFT *is* the prefill pass, so it needs the prefill cells
    # back.  Keeping both views available is what stops one stage's slicing
    # choice from silently emptying another stage's input.
    fit_dec, ev_dec = fit.decode_only(), ev.decode_only()
    cost = CostConfig(hidden_size=2048)
    mode = DispatchMode.DEDUP_RANK
    print(f"   all={t} fit={fit} eval={ev}\n"
          f"   decode-only: fit={fit_dec} eval={ev_dec}  "
          f"({time.time() - t0:.1f}s)", flush=True)

    focus = next(r for r in REGIMES if r["name"] == FOCUS_REGIME)
    topo_focus = build_topology(focus, args.world_size)
    cfg_focus = OcsConfig(n_circuits=max(4, args.world_size // 2), ports_per_rank=2,
                          reconfig_us=RECONFIG_CLASSES["mems_1ms"])

    kinds = list(NO_AFFINITY) + list(AFFINITY)
    if not args.no_promote_aware:
        kinds.append("promote_aware_layer")
        kinds.append("bottleneck_search_layer")
    cache = PlacementCache(out_path.parent / f"cache_{name}_ws{args.world_size}",
                           enabled=True)
    print("  building placements ...", flush=True)
    pl = placements_for(fit_dec, args.world_size, kinds, cache, topo=topo_focus,
                        ocs_cfg=cfg_focus, promote_evals=args.promote_evals,
                        n_sweeps=2 if args.quick else 4)
    print(f"  placements ready ({cache.hits} cached, {cache.misses} built)",
          flush=True)

    result: dict = {
        "workload": str(args.workload), "model": t.model_id,
        "world_size": args.world_size, "seed": args.seed,
        "quick": bool(args.quick),
        "split": "leave-categories-out (fit and eval see disjoint domains)",
        "fit": {"runs": fit_dec.n_runs, "cells": fit_dec.n_cells},
        "eval": {"runs": ev_dec.n_runs, "cells": ev_dec.n_cells},
        "timing_split": {"fit_cells": fit.n_cells, "eval_cells": ev.n_cells,
                         "prefill_eval_cells": int((ev.phase == 0).sum())},
        "num_experts": fit_dec.num_experts, "top_k": fit_dec.top_k,
        "n_moe_layers": fit_dec.n_layers, "decode_only": bool(args.decode_only),
        "placements": {k: p.name for k, p in pl.items()},
        "evidence_chain": str(_repo_root / "logs" / "workload" / name /
                              "evidence_chain.json"),
    }

    def flush_out() -> None:
        """Write after every stage.

        The stages cost minutes each; holding the whole payload until the end
        means one late failure (an argument typo in the last stage, say) throws
        away every earlier stage's completed work.
        """
        result["wall_s"] = round(time.time() - t0, 1)
        payload = _j(result)
        if args.stage != "all" and out_path.exists():
            try:
                with open(out_path) as fh:
                    prev = json.load(fh)
                prev.update(payload)
                payload = prev
            except (OSError, json.JSONDecodeError) as exc:
                print(f"   (could not merge prior results: {exc})", flush=True)
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=1)

    if args.stage in ("fourcell", "all"):
        print("-- stage: four cell", flush=True)
        result["four_cell"] = stage_four_cell(t, fit_dec, ev_dec, args, pl, focus,
                                              cfg_focus, cost, mode)
        flush_out()
    if args.stage in ("regime", "all"):
        print("-- stage: regime + port/latency envelope", flush=True)
        result["regime"] = stage_regime(t, fit_dec, ev_dec, args, pl, cost, mode)
        flush_out()
    if args.stage in ("dispatch", "all"):
        print("-- stage: dispatch semantics", flush=True)
        result["dispatch"] = stage_dispatch(t, fit_dec, ev_dec, args, pl, focus,
                                            cost)
        flush_out()
    if args.stage in ("timing", "all"):
        print("-- stage: completion time (prefill + one decode step)", flush=True)
        result["timing"] = stage_timing(t, fit, ev, args, pl, focus, cost, mode)
        flush_out()
    if args.stage in ("milp", "all"):
        print("-- stage: MILP optimality bounds", flush=True)
        result["milp"] = stage_milp(fit_dec, args)
        flush_out()

    flush_out()

    # ── human summary ────────────────────────────────────────────────
    print("\n== summary", flush=True)
    fc = result.get("four_cell", {}).get("two_by_two")
    if fc:
        print(f"   A eps/no-affinity   {fc['A_eps_no_affinity_us']:12.1f} us")
        print(f"   B eps/affinity      {fc['B_eps_affinity_us']:12.1f} us"
              f"   (affinity alone {fc['delta_affinity_pct']:+.2f}%)")
        print(f"   C ocs/no-affinity   {fc['C_ocs_no_affinity_us']:12.1f} us"
              f"   (ocs alone      {fc['delta_ocs_pct']:+.2f}%)")
        print(f"   D ocs/affinity      {fc['D_ocs_affinity_us']:12.1f} us"
              f"   (both           {fc['delta_both_pct']:+.2f}%)")
        print(f"   -> {fc['verdict']} (synergy {fc['synergy_pct']:+.2f}%); "
              f"D is minimum: {fc['D_is_minimum']}; best cell = {fc['best_cell']}")
    cd = result.get("four_cell", {}).get("co_design")
    if cd:
        for k, v in cd.items():
            print(f"   co-design {k}: vs the electrical optimum "
                  f"| OCS substrate {v['ocs_improvement_vs_electrical_pct']:+.2f}% "
                  f"| EPS substrate {v['eps_improvement_vs_electrical_pct']:+.2f}% "
                  f"| its own circuit gain {v['circuit_gain_pct']:+.2f}%")
    print(f"\n   wrote {out_path}  ({result['wall_s']}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

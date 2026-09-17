#!/usr/bin/env python3
"""exflow_compare.py — ExFlow's objective vs this repo's, measured on the same traces.

The two methods optimise **different objects**, and the point of this script is to
quantify that rather than argue it.

ExFlow (arXiv:2401.08383, formulation in the Zotero note) minimises
**cross-layer token re-routing** — how often a token's layer-l expert and its
layer-(l+1) expert sit on different groups:

    min  Σ_k Σ_{s=1}^{L-1}  cost_{k,s}
    s.t. cost_{k,s} ≥ x[r_{k,s}, c] − x[r_{k,s+1}, c]      (and the mirror)
         Σ_c x[n,c] = 1,   Σ_{n∈N_ℓ} x[n,c] = B

It is coefficient-free and *inter*-layer: a token that does not move between
layers pays nothing, so their paper fuses the combine of layer l with the
dispatch of layer l+1 ("one Alltoall instead of two").

This repo minimises **intra**-layer reach and per-rank accumulation: fanout
(distinct destination ranks per token) and the cross-layer accumulated dedup
ingress, on a tier-weighted byte model.

The two are orthogonal: ExFlow never looks at a token's fan-out *within* a
layer, and this repo never looks at whether a token *stays put* across a layer
boundary. This script measures both objectives for placements produced by both
families, plus the metric that decides OCS value: how much of the promotable
traffic sits on the few rank pairs a switch could actually promote.

    python3 scripts/exflow_compare.py --workload logs/workload/qwen36 --world-size 32
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.eval.cost_model import (  # noqa: E402
    CostConfig, DispatchMode, FabricConfig, Placement, Tier, Topology, evaluate,
    hierarchy_for, traffic_matrix,
)
from src.eval.ocs_eval import OcsConfig, ocs_comparison  # noqa: E402
from src.eval.placement_opt import make_placement  # noqa: E402
from src.eval.trace_ir import CellTable, load_workload  # noqa: E402
from src.serving.suite import build_suite, split_by_category  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════
# ExFlow's objective, built from the trace
# ═══════════════════════════════════════════════════════════════════════

def build_transitions(t: CellTable) -> list[tuple[np.ndarray, np.ndarray]]:
    """The (layer l expert, layer l+1 expert) pairs observed per token.

    ExFlow's ``r_{k,j}`` is the expert a token visits at layer j — one per token
    per layer.  With top-K routing there are K candidates; the repo's ``CellTable``
    stores them sorted by descending gate weight, so ``experts[:, 0]`` is the
    argmax and is what we use, matching the paper's single-expert index.

    Returns one (n_pairs, 2) array per consecutive MoE-layer pair.
    """
    layers = [int(x) for x in np.sort(np.unique(t.layers))]
    n_runs = int(t.run.max()) + 1
    pmax = int(t.pos.max()) + 1
    # [n_runs, pos, layer] -> top-1 expert per (run, pos, layer)
    top1 = np.full((n_runs, pmax, len(layers)), -1, dtype=np.int32)
    lidx = {l: i for i, l in enumerate(layers)}
    for cell in range(t.n_cells):
        top1[t.run[cell], t.pos[cell], lidx[int(t.layer[cell])]] = int(t.experts[cell, 0])

    trans = []
    for i in range(len(layers) - 1):
        a, b = top1[:, :, i], top1[:, :, i + 1]
        m = (a >= 0) & (b >= 0)
        trans.append(np.stack([a[m], b[m]], axis=1))
    return trans


def exflow_reroute_fraction(trans: list[tuple], P: np.ndarray) -> float:
    """ExFlow's objective: share of layer boundaries where the token must move.

    ``P`` is [L, E] expert→rank.  Lower is better; this is the quantity their ILP
    minimises (their cost_{k,s} summed and normalised).
    """
    tot = moved = 0
    for i, pair in enumerate(trans):
        a, b = pair[:, 0], pair[:, 1]
        moved += int((P[i, a] != P[i + 1, b]).sum())
        tot += int(a.shape[0])
    return moved / max(tot, 1)


def as_per_layer(pl: Placement, n_layers: int) -> np.ndarray:
    m = pl.expert_to_rank
    return (np.repeat(m[None, :], n_layers, axis=0) if m.ndim == 1 else m)


def exflow_local_search(trans: list[tuple], init: np.ndarray, E: int,
                        rng: np.random.Generator, n_sweeps: int = 30,
                        n_cand: int = 12) -> tuple[np.ndarray, float]:
    """Local search on ExFlow's objective — a stand-in for their MILP.

    Their solver is an ILP with warm starts and multilevel bipartitioning; this
    is plain swap-based local search on the *identical objective*, so it answers
    "what does the objective want" without claiming to reproduce their
    optimiser.  Swaps happen **within a layer**, which preserves the per-layer
    capacity B = E/W that both their constraints and this repo's generators
    require.

    Deltas are exact and local: swapping two experts in layer l only changes the
    transitions at the (l−1)→l and l→(l+1) boundaries.
    """
    L = init.shape[0]
    P = init.copy()

    # Which *boundary* indices touch each layer.  Boundary i connects layer i and
    # layer i+1, so layer l appears in boundary l (as its "from" side, when it has
    # a successor) and in boundary l-1 (as its "to" side).
    touch: list[list[int]] = [[] for _ in range(L)]
    for l in range(L):
        if l <= L - 2:
            touch[l].append(l)
        if l >= 1:
            touch[l].append(l - 1)

    def boundary_cost(i: int, P: np.ndarray) -> int:
        a, b = trans[i][:, 0], trans[i][:, 1]
        return int((P[i, a] != P[i + 1, b]).sum())

    cur = {i: boundary_cost(i, P) for i in range(L - 1)}
    total = sum(cur.values())
    n_pairs = sum(int(p.shape[0]) for p in trans)

    for _ in range(n_sweeps):
        improved = False
        for l in rng.permutation(L):
            for _ in range(max(1, E // 8)):
                u, v = rng.choice(E, size=2, replace=False)
                if P[l, u] == P[l, v]:
                    continue
                before = sum(cur[i] for i in touch[l])
                P[l, u], P[l, v] = P[l, v], P[l, u]
                after = 0
                for i in touch[l]:
                    after += boundary_cost(i, P)
                if after < before:
                    for i in touch[l]:
                        cur[i] = boundary_cost(i, P)
                    total += after - before
                    improved = True
                else:
                    P[l, u], P[l, v] = P[l, v], P[l, u]
        if not improved:
            break
    return P, total / max(n_pairs, 1)


# ═══════════════════════════════════════════════════════════════════════
# The metrics both families are judged on
# ═══════════════════════════════════════════════════════════════════════

def ocs_alignment(tm_counts: np.ndarray, topo: Topology, budget: int) -> dict:
    """How much of the promotable traffic sits on the pairs a switch could promote.

    An OCS is a *pair* resource with a degree bound, so what decides whether
    optical circuits can pay is not total bytes but the **concentration** of
    cross-pod bytes on the hottest few rank pairs.  ``top_budget_share`` is the
    share of all cross-pod bytes carried by the ``budget`` heaviest pairs — the
    same pairs ``plan_circuits`` would pick.
    """
    T = topo.tier_matrix()
    W = topo.world_size
    sym = np.zeros((W, W))
    d = min(tm_counts.shape[0], W)
    sym[:d, :W] += tm_counts[:d, :W]
    sym[:W, :d] += tm_counts[:d, :W].T
    iu = np.triu_indices(W, 1)
    w = sym[iu]
    tier = T[iu]
    m = tier == int(Tier.CROSS_POD)
    cross = w[m]
    if cross.size == 0 or cross.sum() == 0:
        return {"applicable": False}
    order = np.argsort(-cross)
    k = min(budget, cross.size)
    return {
        "applicable": True,
        "cross_pod_pairs_active": int((cross > 0).sum()),
        "top_budget_share": float(cross[order[:k]].sum() / cross.sum()),
        "top_budget": k,
        # Herfindahl index over cross-pod pair weights: 1/k means perfectly
        # spread, 1.0 means every promotable byte is on one pair.
        "herfindahl": float(((cross / cross.sum()) ** 2).sum()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workload", type=Path, required=True)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-circuits", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    t = load_workload(args.workload / "manifest.json", decode_only=True)
    specs = build_suite(n_repeats=0)
    fu, eu = split_by_category(specs, seed=args.seed)
    present = {r.uid for r in t.runs}
    fit = t.by_runs([u for u in fu if u in present]).decode_only()
    ev = t.by_runs([u for u in eu if u in present]).decode_only()
    W, E, L = args.world_size, t.num_experts, t.n_layers

    base = hierarchy_for(W, "multi_pod")
    topo = Topology(W, base.gpus_per_node, base.nodes_per_pod,
                    FabricConfig(core_oversubscription=4.0), set(),
                    base.rank_to_slot, (Tier.CROSS_POD,))
    cost = CostConfig(hidden_size=2048)
    cfg = OcsConfig(n_circuits=args.n_circuits, ports_per_rank=2)

    print(f"== {t.model_id}  E={E} K={t.top_k} L={L} W={W}")
    trans = build_transitions(ev)
    n_pairs = sum(int(p.shape[0]) for p in trans)
    print(f"   layer boundaries: {len(trans)}  transitions: {n_pairs:,}")

    # ── placements: ours + an ExFlow-objective search from two starts ──
    rows: dict[str, np.ndarray | None] = {}
    for kind in ("random", "linear", "load_balanced_layer", "affinity_layer",
                 "affinity_coordinated_layer"):
        rows[kind] = as_per_layer(make_placement(kind, fit, W, seed=args.seed), L)

    rng = np.random.default_rng(args.seed)
    print("   ExFlow-objective local search (from linear init) ...")
    t0 = time.time()
    ef_from_linear, ef_obj = exflow_local_search(
        trans, as_per_layer(make_placement("linear", fit, W, seed=0), L), E, rng)
    print(f"     -> re-route {ef_obj:.4f} in {time.time() - t0:.0f}s")
    rows["exflow_ls (from linear)"] = ef_from_linear
    print("   ExFlow-objective local search (from affinity-coordinated init) ...")
    t0 = time.time()
    ef_from_aff, ef_obj2 = exflow_local_search(
        trans, as_per_layer(make_placement("affinity_coordinated_layer", fit, W,
                                           seed=args.seed), L), E, rng)
    print(f"     -> re-route {ef_obj2:.4f} in {time.time() - t0:.0f}s")
    rows["exflow_ls (from affinity)"] = ef_from_aff

    # ── score every placement on BOTH families of metric ──────────────
    out: dict = {"workload": str(args.workload), "world_size": W,
                 "n_circuits": args.n_circuits, "cells": int(ev.n_cells),
                 "transitions": int(n_pairs), "placements": {}}
    print(f"\n   {'placement':<28} {'reroute%':>9} {'fanout':>7} {'bn_us':>9} "
          f"{'xpod_GB':>8} {'ocs_gain%':>10} {'cover%':>7} {'top16share%':>12}")
    for name, P in rows.items():
        pl = Placement(P, W, "per_layer", layers=ev.layers, name=name)
        r = evaluate(ev, pl, topo, cost, DispatchMode.DEDUP_RANK, seed=args.seed)
        tm = traffic_matrix(ev, pl, topo, DispatchMode.DEDUP_RANK, seed=args.seed)
        al = ocs_alignment(tm.counts, topo, args.n_circuits)
        oc = ocs_comparison(fit, ev, pl, topo, cfg, cost, DispatchMode.DEDUP_RANK,
                            args.seed)
        gain = oc.get("static_ocs", {}).get("bottleneck_reduction_pct")
        cover = oc.get("static_ocs", {}).get("promotable_traffic_covered_fraction")
        rr = exflow_reroute_fraction(trans, P)
        out["placements"][name] = {
            "exflow_reroute_pct": round(100 * rr, 3),
            "mean_fanout": round(r["mean_fanout"], 4),
            "bottleneck_us": round(r["bottleneck_us"], 1),
            "cross_pod_GB": round(r["cross_pod_bytes"] / 1e9, 4),
            "static_ocs_gain_pct": gain,
            "ocs_covered_fraction": cover,
            "ocs_alignment": al,
        }
        print(f"   {name:<28} {100 * rr:8.2f}% {r['mean_fanout']:7.3f} "
              f"{r['bottleneck_us']:9.1f} {r['cross_pod_bytes'] / 1e9:8.3f} "
              f"{(gain if gain is not None else float('nan')):10.2f} "
              f"{100 * (cover or 0):6.2f}% "
              f"{100 * al.get('top_budget_share', 0):11.2f}%")
        sys.stdout.flush()

    # ── are the two objectives aligned or in conflict? ────────────────
    names = list(out["placements"])
    x = np.array([out["placements"][n]["exflow_reroute_pct"] for n in names])
    y = np.array([out["placements"][n]["bottleneck_us"] for n in names])
    z = np.array([out["placements"][n]["ocs_alignment"].get("top_budget_share", 0)
                  for n in names])
    out["correlations"] = {
        "pearson_exflow_vs_bottleneck": float(np.corrcoef(x, y)[0, 1]),
        "pearson_exflow_vs_ocs_concentration": float(np.corrcoef(x, z)[0, 1]),
        "note": ("positive exflow-vs-bottleneck means the two objectives AGREE on "
                 "which placement is better; negative means they conflict"),
    }
    print(f"\n   corr(ExFlow re-route, bottleneck)      = "
          f"{out['correlations']['pearson_exflow_vs_bottleneck']:+.3f}")
    print(f"   corr(ExFlow re-route, OCS concentration) = "
          f"{out['correlations']['pearson_exflow_vs_ocs_concentration']:+.3f}")

    out_path = args.out or (_REPO / "outputs" / "four_cell" /
                            f"exflow_compare.{args.workload.name}.ws{W}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"   wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
